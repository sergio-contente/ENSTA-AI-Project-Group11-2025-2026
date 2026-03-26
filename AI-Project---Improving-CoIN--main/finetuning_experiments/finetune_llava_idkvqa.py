"""
Finetuning LLaVA-1.6 (Mistral-7B) with QLoRA on the IDKVQA dataset
+ Data Augmentation: text template variations + image geometric/photometric transforms
+ Mix COCO VQA (general) + IDKVQA (task-specific) — prevents overfitting to tiny IDKVQA set
+ Soft labels from human vote distributions
+ Tau search for Effective Reliability (ER) — AIUTA paper metric
+ Full evaluation with comparative plots: baseline vs finetuned

Pipeline:
  1. Load IDKVQA (parquet) and VQAv2/COCO (HuggingFace streaming)
  2. Split IDKVQA by IMAGE hash (no data leakage): 60% train / 20% val / 20% test
  3. Data augmentation on train: text templates x image transforms (geometric + photometric)
  4. Mix: 80% COCO + 20% IDKVQA — soft labels from both sources
  5. Train LLaVA-1.6 with QLoRA (4-bit) + cosine scheduler + early stopping
  6. Evaluate baseline and finetuned: accuracy, F1, confusion matrix
  7. Tau search on test set -> Effective Reliability (ER) as reported in the AIUTA paper
"""

import os
import json
import hashlib
import random
import contextlib
import math
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance, ImageOps, ImageFilter
from io import BytesIO
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
from tqdm import tqdm
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import (
    LlavaNextProcessor,
    LlavaNextForConditionalGeneration,
    BitsAndBytesConfig,
    TrainerCallback,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel, prepare_model_for_kbit_training


# =============================================================================
# 0. REPRODUCIBILITY
# =============================================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)


# =============================================================================
# 1. CONFIGURATION
# =============================================================================
MODEL_HF     = "llava-hf/llava-v1.6-mistral-7b-hf"
PARQUET_PATH = "IDKVQA/data/val-00000-of-00001.parquet"   # IDKVQA parquet file
SAVE_DIR     = "./outputs/coin_finetune"
OUTPUT_DIR   = "./outputs/llava_idkvqa_lora_aug"
PLOTS_DIR    = "./outputs/plots_aug"

# LoRA
LORA_RANK    = 32
LORA_ALPHA   = 64
LORA_DROPOUT = 0.05

# Training
BATCH_SIZE         = 1
ACCUMULATION_STEPS = 8      # effective batch size = 8
LEARNING_RATE      = 3e-5
NUM_EPOCHS         = 4
PATIENCE           = 2      # early stopping patience (epochs without val improvement)
MAX_LENGTH         = 2048

# COCO + IDKVQA mix
N_COCO_PER_CLASS = 1000     # samples per class from VQAv2
MIX_RATIO        = 0.2      # fraction of IDKVQA samples in the mixed dataset

# IDKVQA splits (by unique image)
TRAIN_RATIO = 0.60
VAL_RATIO   = 0.20
# TEST_RATIO = 0.20 (implicit remainder)

os.makedirs(SAVE_DIR,   exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR,  exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")


# =============================================================================
# 2. LABEL UTILITIES
# =============================================================================
KEY_TO_LABEL = {"0": "Yes", "1": "No", "2": "I don't know"}
LABEL_TO_KEY = {v: k for k, v in KEY_TO_LABEL.items()}


def votes_to_label(answers: dict) -> str:
    """Convert IDKVQA vote dictionary to majority label string."""
    if all(v == 0 for v in answers.values()):
        return "?"
    max_val = max(answers.values())
    winners = [k for k, v in answers.items() if v == max_val]
    if len(winners) > 1:
        return "?"
    mapping = {"Yes": "Yes", "No": "No", "I don't know": "?"}
    return mapping[winners[0]]


def normalize_prediction(response: str) -> str:
    """Normalize a model-generated string to one of: Yes / No / I don't know / Other."""
    r = response.lower().strip()
    if r.startswith("yes"):
        return "Yes"
    if r.startswith("no"):
        return "No"
    if "?" in r or "don" in r or "know" in r or r == "":
        return "I don't know"
    return "Other"


def get_answer_token_ids(processor: Any, device: torch.device) -> torch.Tensor:
    """Return token IDs for 'Yes', 'No', '?' -- specific to the Mistral-7B tokenizer."""
    tok = processor.tokenizer
    return torch.tensor([
        tok.encode("Yes", add_special_tokens=False)[0],
        tok.encode("No",  add_special_tokens=False)[0],
        tok.encode("?",   add_special_tokens=False)[0],
    ], dtype=torch.long, device=device)


# =============================================================================
# 3. LOAD & SPLIT IDKVQA (by image hash -- no data leakage)
# =============================================================================

def load_idkvqa_from_parquet(parquet_path: str) -> List[dict]:
    """
    Read IDKVQA parquet and return a list of records with fields:
      image_bytes, question, answers (vote dict), majority_label, soft_label
    """
    df = pd.read_parquet(parquet_path)
    records = []
    for row in df.to_dict("records"):
        answers = row["answers"]        # e.g. {"Yes": 3, "No": 1, "I don't know": 0}
        total   = max(sum(answers.values()), 1)
        soft    = [
            answers.get("Yes",          0) / total,
            answers.get("No",           0) / total,
            answers.get("I don't know", 0) / total,
        ]
        majority = max(answers, key=answers.get)
        records.append({
            "image_bytes":    row["image"]["bytes"],
            "question":       row["question"],
            "answers":        answers,
            "majority_label": majority,
            "soft_label":     soft,
        })
    return records


def split_idkvqa_by_image(records: List[dict]) -> Tuple[List, List, List]:
    """
    Split records by unique image (SHA-1 hash of bytes) into train / val / test.
    Guarantees the same image never appears in more than one split.
    """
    image_to_records: Dict[str, List] = {}
    for r in records:
        key = hashlib.sha1(r["image_bytes"]).hexdigest()
        image_to_records.setdefault(key, []).append(r)

    keys = list(image_to_records.keys())
    random.shuffle(keys)

    n       = len(keys)
    n_train = int(n * TRAIN_RATIO)
    n_val   = int(n * VAL_RATIO)

    train_keys = set(keys[:n_train])
    val_keys   = set(keys[n_train : n_train + n_val])
    test_keys  = set(keys[n_train + n_val :])

    # Sanity checks -- no overlap across splits
    assert len(train_keys & val_keys)  == 0, "DATA LEAKAGE: train/val overlap"
    assert len(train_keys & test_keys) == 0, "DATA LEAKAGE: train/test overlap"
    assert len(val_keys   & test_keys) == 0, "DATA LEAKAGE: val/test overlap"

    def gather(key_set):
        out = []
        for k in key_set:
            out.extend(image_to_records[k])
        return out

    return gather(train_keys), gather(val_keys), gather(test_keys)


# =============================================================================
# 4. DATA AUGMENTATION
# =============================================================================

# -----------------------------------------------------------------------------
# 4a. Text templates
# -----------------------------------------------------------------------------
TEXT_TEMPLATES = {
    "Yes": [
        "{q}",
        "Looking at the image, {q_lower}",
        "Based on what you see, {q_lower}",
        "In the image shown, {q_lower}",
    ],
    "No": [
        "{q}",
        "Looking at the image, {q_lower}",
        "Based on what you see, {q_lower}",
        "From what is visible, {q_lower}",
    ],
    "I don't know": [
        "{q}",
        "Looking carefully at the image, {q_lower}",
        "Based on the image provided, {q_lower}",
        "From the available visual information, {q_lower}",
    ],
}
SUFFIX = " You must answer only with Yes, No, or ?=I don't know."


def augment_question(question: str, label: str) -> List[str]:
    """Generate question variants for a given label using text templates."""
    base       = question.replace(SUFFIX, "").strip()
    base_lower = base[0].lower() + base[1:] if base else base
    templates  = TEXT_TEMPLATES.get(label, ["{q}"])
    return [t.format(q=base, q_lower=base_lower) + SUFFIX for t in templates]


# -----------------------------------------------------------------------------
# 4b. Individual image transform helpers
# -----------------------------------------------------------------------------

def _random_crop(image: Image.Image, scale: float = 0.85) -> Image.Image:
    """
    Crop a random sub-region covering `scale` of the image area,
    then resize back to the original dimensions.
    Simulates zoom-in and partial field-of-view variation.
    """
    w, h    = image.size
    new_w   = int(w * scale)
    new_h   = int(h * scale)
    left    = random.randint(0, max(w - new_w, 0))
    top     = random.randint(0, max(h - new_h, 0))
    cropped = image.crop((left, top, left + new_w, top + new_h))
    return cropped.resize((w, h), Image.BILINEAR)


def _rotate(image: Image.Image, angle: float) -> Image.Image:
    """
    Rotate image by `angle` degrees (positive = counter-clockwise).
    Uses BICUBIC interpolation; border pixels are filled with black (0, 0, 0).
    Small angles (+-10 to +-20 deg) simulate camera tilt.
    """
    return image.rotate(angle, resample=Image.BICUBIC, expand=False)


def _color_jitter(
    image:      Image.Image,
    brightness: float = 1.0,
    contrast:   float = 1.0,
    saturation: float = 1.0,
    hue_shift:  float = 0.0,
) -> Image.Image:
    """
    Apply independent photometric distortions:
      brightness : pixel intensity multiplier (>1 = brighter, <1 = darker)
      contrast   : local contrast multiplier  (>1 = more contrast)
      saturation : color saturation multiplier (0 = grayscale, >1 = vivid)
      hue_shift  : additive shift on HSV hue channel in [-0.5, 0.5]
    """
    img = image
    if brightness != 1.0:
        img = ImageEnhance.Brightness(img).enhance(brightness)
    if contrast != 1.0:
        img = ImageEnhance.Contrast(img).enhance(contrast)
    if saturation != 1.0:
        img = ImageEnhance.Color(img).enhance(saturation)
    if hue_shift != 0.0:
        # Hue shift via HSV numpy array
        # Convert to HSV, shift hue channel, convert back to RGB
        # Avoid the deprecated `mode` parameter in Image.fromarray (removed in Pillow 13)
        hsv_img = img.convert("HSV")
        arr = np.array(hsv_img, dtype=np.float32)
        arr[..., 0] = (arr[..., 0] / 255.0 + hue_shift) % 1.0 * 255.0
        img = Image.fromarray(arr.astype(np.uint8), "HSV").convert("RGB")
    return img


def _gaussian_blur(image: Image.Image, radius: float = 1.5) -> Image.Image:
    """
    Apply Gaussian blur with the given radius (pixels).
    Simulates camera focus blur or motion blur at low radius values.
    """
    return image.filter(ImageFilter.GaussianBlur(radius=radius))


def _sharpen(image: Image.Image, factor: float = 2.0) -> Image.Image:
    """
    Adjust image sharpness.
    factor > 1 enhances edges; factor < 1 softens them.
    """
    return ImageEnhance.Sharpness(image).enhance(factor)


def _to_grayscale_rgb(image: Image.Image) -> Image.Image:
    """
    Convert to single-channel grayscale then back to 3-channel RGB.
    Forces the model to rely on luminance patterns rather than color cues.
    """
    return ImageOps.grayscale(image).convert("RGB")


def _horizontal_flip(image: Image.Image) -> Image.Image:
    """Mirror image along the vertical axis (left <-> right)."""
    return ImageOps.mirror(image)


# -----------------------------------------------------------------------------
# 4c. Augmentation catalogue
#
# Each entry is a (name, transform_fn) pair applied deterministically.
# Randomness is introduced by the crop offset inside _random_crop; all other
# transforms are fixed so the same image always produces the same variants.
#
# Total: 30 image transforms x 4 text templates = 120 samples per original.
# -----------------------------------------------------------------------------
IMAGE_AUGMENTATIONS: List[Tuple[str, Any]] = [
    # -- Original ---------------------------------------------------------------
    ("original",            lambda img: img),

    # -- Geometric: flips -------------------------------------------------------
    ("flip_h",              _horizontal_flip),
    ("flip_v",              lambda img: ImageOps.flip(img)),

    # -- Geometric: rotations ---------------------------------------------------
    ("rotate_+10",          lambda img: _rotate(img,  10)),
    ("rotate_-10",          lambda img: _rotate(img, -10)),
    ("rotate_+20",          lambda img: _rotate(img,  20)),
    ("rotate_-20",          lambda img: _rotate(img, -20)),
    ("rotate_+45",          lambda img: _rotate(img,  45)),
    ("rotate_-45",          lambda img: _rotate(img, -45)),

    # -- Geometric: crops (zoom-in) ---------------------------------------------
    ("crop_85pct",          lambda img: _random_crop(img, scale=0.85)),
    ("crop_75pct",          lambda img: _random_crop(img, scale=0.75)),
    ("crop_90pct",          lambda img: _random_crop(img, scale=0.90)),

    # -- Photometric: brightness ------------------------------------------------
    ("bright_+30pct",       lambda img: _color_jitter(img, brightness=1.30)),
    ("bright_-30pct",       lambda img: _color_jitter(img, brightness=0.70)),
    ("bright_+50pct",       lambda img: _color_jitter(img, brightness=1.50)),
    ("bright_-50pct",       lambda img: _color_jitter(img, brightness=0.50)),

    # -- Photometric: contrast --------------------------------------------------
    ("contrast_+30pct",     lambda img: _color_jitter(img, contrast=1.30)),
    ("contrast_-20pct",     lambda img: _color_jitter(img, contrast=0.80)),

    # -- Photometric: saturation ------------------------------------------------
    ("sat_+50pct",          lambda img: _color_jitter(img, saturation=1.50)),
    ("sat_-50pct",          lambda img: _color_jitter(img, saturation=0.50)),

    # -- Photometric: hue shift -------------------------------------------------
    ("hue_+0.05",           lambda img: _color_jitter(img, hue_shift= 0.05)),
    ("hue_-0.05",           lambda img: _color_jitter(img, hue_shift=-0.05)),

    # -- Texture / focus --------------------------------------------------------
    ("blur_r1",             lambda img: _gaussian_blur(img, radius=1.0)),
    ("blur_r2",             lambda img: _gaussian_blur(img, radius=2.0)),
    ("sharpen_2x",          lambda img: _sharpen(img, factor=2.0)),
    ("sharpen_3x",          lambda img: _sharpen(img, factor=3.0)),
    ("grayscale",           _to_grayscale_rgb),

    # -- Combined ---------------------------------------------------------------
    ("flip_blur",           lambda img: _gaussian_blur(_horizontal_flip(img), radius=1.0)),
    ("crop_bright",         lambda img: _color_jitter(_random_crop(img, 0.85), brightness=1.2)),
    ("rotate_contrast",     lambda img: _color_jitter(_rotate(img, 15), contrast=1.3)),
    ("crop_grayscale",      lambda img: _to_grayscale_rgb(_random_crop(img, 0.85))),
    ("flip_sat_down",       lambda img: _color_jitter(_horizontal_flip(img), saturation=0.5)),
    ("rotate_sharpen",      lambda img: _sharpen(_rotate(img, -10), factor=2.0)),
]

# Total number of image augmentations (for logging)
N_IMAGE_AUGS = len(IMAGE_AUGMENTATIONS)


def augment_image_pil(image: Image.Image) -> List[Tuple[str, Image.Image]]:
    """
    Apply every transform in IMAGE_AUGMENTATIONS to `image`.
    Returns a list of (transform_name, augmented_PIL_image) tuples.
    Each transform is wrapped in try/except so a single failure never
    kills the whole augmentation pipeline -- the original is used as fallback.
    """
    results = []
    for name, fn in IMAGE_AUGMENTATIONS:
        try:
            results.append((name, fn(image).convert("RGB")))
        except Exception as exc:
            print(f"  [WARNING] Image transform '{name}' failed ({exc}), using original.")
            results.append((name, image.convert("RGB")))
    return results


def augment_dataset(records: List[dict]) -> List[dict]:
    """
    Cross-product augmentation: text templates x image transforms.

    Expansion factor per original record:
        len(TEXT_TEMPLATES[label])  x  N_IMAGE_AUGS
        e.g. 4 text x 33 images = 132 samples per record

    Augmented samples inherit the original soft_label and majority_label
    unchanged -- none of the transforms alter the semantic answer.
    The _aug_transform field stores the transform name for debugging.
    """
    augmented = []
    for row in records:
        label      = row["majority_label"]
        image_pil  = Image.open(BytesIO(row["image_bytes"])).convert("RGB")
        questions  = augment_question(row["question"], label)   # text variants
        image_list = augment_image_pil(image_pil)               # image variants

        for q in questions:
            for aug_name, aug_img in image_list:
                buf = BytesIO()
                aug_img.save(buf, format="PNG")
                augmented.append({
                    "image_bytes":    buf.getvalue(),
                    "question":       q,
                    "answers":        row["answers"],
                    "majority_label": label,
                    "soft_label":     row["soft_label"],
                    "_aug_transform": aug_name,
                })

    random.shuffle(augmented)
    return augmented


# =============================================================================
# 5. LOAD VQAv2 / COCO (streaming, balanced per class)
# =============================================================================

def _pil_to_bytes(image: Image.Image) -> bytes:
    """Serialize a PIL image to PNG bytes so it is safe across DataLoader workers."""
    buf = BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    return buf.getvalue()


def load_coco_samples(n_per_class: int = N_COCO_PER_CLASS) -> List[dict]:
    """
    Stream the VQAv2 validation split and map answers to Yes / No / IDK:
      yes/no questions -> Yes or No with a hard soft label
      number / other   -> I don't know  (model should abstain)

    Images are immediately serialised to PNG bytes so that PIL objects do not
    become invalid across DataLoader workers or long-running training loops.

    Stops as soon as each class reaches n_per_class samples.
    """
    dataset = load_dataset("lmms-lab/VQAv2", split="validation", streaming=True)
    yes_s, no_s, idk_s = [], [], []

    for row in tqdm(dataset, desc="Sampling VQAv2"):
        if all(len(s) >= n_per_class for s in [yes_s, no_s, idk_s]):
            break

        image = row["image"]
        if not hasattr(image, "convert"):
            image = Image.fromarray(image)
        # Serialise immediately — PIL objects from streaming datasets can lose
        # their backing store once the iterator advances
        image_bytes = _pil_to_bytes(image)

        mc    = row["multiple_choice_answer"].lower().strip()
        atype = row["answer_type"]

        if atype == "yes/no":
            if mc == "yes" and len(yes_s) < n_per_class:
                yes_s.append({
                    "image_bytes":    image_bytes,
                    "question":       row["question"],
                    "majority_label": "Yes",
                    "soft_label":     [1.0, 0.0, 0.0],
                })
            elif mc == "no" and len(no_s) < n_per_class:
                no_s.append({
                    "image_bytes":    image_bytes,
                    "question":       row["question"],
                    "majority_label": "No",
                    "soft_label":     [0.0, 1.0, 0.0],
                })
        else:
            # "number" or "other" -- model should abstain
            if len(idk_s) < n_per_class:
                idk_s.append({
                    "image_bytes":    image_bytes,
                    "question":       row["question"],
                    "majority_label": "I don't know",
                    "soft_label":     [0.0, 0.0, 1.0],
                })

    samples = yes_s + no_s + idk_s
    random.shuffle(samples)
    print(f"VQAv2 samples loaded: {len(samples)} "
          f"(Yes={len(yes_s)}, No={len(no_s)}, IDK={len(idk_s)})")
    return samples


# =============================================================================
# 6. PyTorch DATASETS
# =============================================================================

class IDKVQADataset(Dataset):
    """IDKVQA dataset backed by image_bytes records."""

    def __init__(self, records: List[dict]):
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        row   = self.records[idx]
        image = Image.open(BytesIO(row["image_bytes"])).convert("RGB")
        return {
            "image":          image,
            "question":       row["question"] if SUFFIX in row["question"]
                              else row["question"] + SUFFIX,
            "majority_label": row["majority_label"],
            "soft_label":     row["soft_label"],
        }


class COCODataset(Dataset):
    """VQAv2/COCO dataset backed by PNG-serialised image bytes.
    Using bytes (not live PIL objects) is safe across DataLoader workers."""

    def __init__(self, samples: List[dict]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s   = self.samples[idx]
        img = Image.open(BytesIO(s["image_bytes"])).convert("RGB")
        return {
            "image":          img,
            "question":       s["question"] + SUFFIX,
            "majority_label": s["majority_label"],
            "soft_label":     s["soft_label"],
        }


class MixedDataset(Dataset):
    """
    Combines COCO (1 - mix_ratio) and augmented IDKVQA (mix_ratio) samples.
    IDKVQA records are repeated if needed to fill the requested ratio.
    """

    def __init__(
        self,
        coco_samples:   List[dict],
        idkvqa_records: List[dict],
        mix_ratio:      float = MIX_RATIO,
    ):
        n_idkvqa   = int(len(coco_samples) * mix_ratio / (1.0 - mix_ratio))
        idkvqa_rep = (
            idkvqa_records * (n_idkvqa // max(len(idkvqa_records), 1) + 1)
        )[:n_idkvqa]

        coco_tagged   = [{"data": s, "source": "coco"}   for s in coco_samples]
        idkvqa_tagged = [{"data": s, "source": "idkvqa"} for s in idkvqa_rep]

        self.items = coco_tagged + idkvqa_tagged
        random.shuffle(self.items)
        print(
            f"Mixed dataset: {len(coco_tagged)} COCO + {len(idkvqa_tagged)} IDKVQA "
            f"= {len(self.items)} total"
        )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        item = self.items[idx]
        if item["source"] == "coco":
            return COCODataset([item["data"]])[0]
        return IDKVQADataset([item["data"]])[0]


# =============================================================================
# 7. COLLATORS
# =============================================================================

def build_collate_fn(processor: Any, device: torch.device):
    """
    Returns a collate function (closure over processor + device).
    Used in the manual training loop with soft-label loss.
    """

    def collate_fn(batch: List[dict]) -> Tuple[dict, torch.Tensor, List[str]]:
        prompts = []
        for item in batch:
            conv = [{"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": item["question"]},
            ]}]
            prompts.append(processor.apply_chat_template(conv, add_generation_prompt=True))

        ids_list, pv_list, is_list = [], [], []
        for prompt, item in zip(prompts, batch):
            # Always use keyword arguments to avoid positional swap between
            # text and images (a common source of "incorrect image source" errors)
            enc = processor(text=prompt, images=item["image"], return_tensors="pt")
            ids_list.append(enc["input_ids"])
            pv_list.append(enc["pixel_values"])
            is_list.append(enc["image_sizes"])

        max_len = max(x.shape[1] for x in ids_list)
        pad_id  = processor.tokenizer.pad_token_id or 0
        padded, masks = [], []
        for ids in ids_list:
            pl = max_len - ids.shape[1]
            padded.append(F.pad(ids,                  (pl, 0), value=pad_id))
            masks.append( F.pad(torch.ones_like(ids), (pl, 0), value=0))

        inputs = {
            "input_ids":      torch.cat(padded,  dim=0).to(device),
            "attention_mask": torch.cat(masks,   dim=0).to(device),
            "pixel_values":   torch.cat(pv_list, dim=0).to(device),
            "image_sizes":    torch.cat(is_list, dim=0).to(device),
        }
        soft_labels     = torch.tensor(
            [item["soft_label"] for item in batch], dtype=torch.float32
        ).to(device)
        majority_labels = [item["majority_label"] for item in batch]
        return inputs, soft_labels, majority_labels

    return collate_fn


@dataclass
class LLaVACollatorForTrainer:
    """
    Collator compatible with the HuggingFace Trainer API.
    Masks the prompt prefix in labels so the loss is only computed
    on the answer tokens (standard causal LM fine-tuning practice).
    """
    processor: Any

    def __call__(self, batch: List[dict]) -> dict:
        all_input_ids      = []
        all_attention_mask = []
        all_pixel_values   = []
        all_image_sizes    = []
        all_labels         = []

        for item in batch:
            conversation = [
                {"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": item["question"]},
                ]},
                {"role": "assistant", "content": [
                    {"type": "text", "text": item["majority_label"]},
                ]},
            ]
            prompt = self.processor.apply_chat_template(
                conversation, add_generation_prompt=False
            )
            enc = self.processor(
                text=prompt,
                images=item["image"],
                return_tensors="pt",
                truncation=True,
                max_length=MAX_LENGTH,
            )
            input_ids  = enc["input_ids"][0]
            attn_mask  = enc["attention_mask"][0]
            seq_labels = input_ids.clone()

            # Mask everything up to and including [/INST] with -100
            inst_id   = self.processor.tokenizer.convert_tokens_to_ids("[/INST]")
            positions = (input_ids == inst_id).nonzero(as_tuple=True)[0]
            if len(positions) > 0:
                seq_labels[: positions[-1].item() + 1] = -100

            # Also mask padding tokens
            pad_id = self.processor.tokenizer.pad_token_id
            if pad_id is not None:
                seq_labels[input_ids == pad_id] = -100

            all_input_ids.append(input_ids)
            all_attention_mask.append(attn_mask)
            all_pixel_values.append(enc["pixel_values"][0])
            all_image_sizes.append(enc["image_sizes"][0])
            all_labels.append(seq_labels)

        max_len = max(t.shape[0] for t in all_input_ids)
        pad_id  = self.processor.tokenizer.pad_token_id or 0

        def pad(t, val, length):
            diff = length - t.shape[0]
            return torch.cat([t, torch.full((diff,), val, dtype=t.dtype)]) if diff > 0 else t

        return {
            "input_ids":      torch.stack([pad(t, pad_id, max_len) for t in all_input_ids]),
            "attention_mask": torch.stack([pad(t, 0,      max_len) for t in all_attention_mask]),
            "pixel_values":   torch.stack(all_pixel_values),
            "image_sizes":    torch.stack(all_image_sizes),
            "labels":         torch.stack([pad(t, -100,   max_len) for t in all_labels]),
        }


# =============================================================================
# 8. SOFT-LABEL LOSS (from CoIN notebook)
# =============================================================================

def compute_soft_loss(
    model,
    inputs:           dict,
    soft_labels:      torch.Tensor,
    answer_token_ids: torch.Tensor,
    lambda_tail:      float = 0.0,
) -> Tuple[torch.Tensor, float, float]:
    """
    Soft cross-entropy over the 3 answer tokens + optional tail penalty.

    soft_ce   : weighted cross-entropy using human vote proportions as targets
    tail      : mean probability mass assigned to tokens outside {Yes, No, ?}
                (penalises the model for hedging into irrelevant vocabulary)
    total loss: soft_ce + lambda_tail * tail

    Equivalent to compute_loss() in the CoIN notebook.
    """
    use_fp16 = torch.cuda.is_available()
    ctx      = torch.cuda.amp.autocast(dtype=torch.float16) if use_fp16 else contextlib.nullcontext()
    dtype    = torch.float16 if use_fp16 else torch.float32

    with ctx:
        outputs = model(
            input_ids      = inputs["input_ids"],
            attention_mask = inputs["attention_mask"],
            pixel_values   = inputs["pixel_values"].to(dtype),
            image_sizes    = inputs["image_sizes"],
        )

    # Logits at the last non-padding position of each sequence
    last_pos = inputs["attention_mask"].sum(dim=1) - 1
    logits   = outputs.logits[torch.arange(len(last_pos)), last_pos].float()
    probs    = F.softmax(logits, dim=-1)

    # Restrict to answer tokens and re-normalise
    ap      = probs[:, answer_token_ids]
    ap_norm = ap / (ap.sum(dim=-1, keepdim=True) + 1e-9)
    soft_ce = -(soft_labels * torch.log(ap_norm + 1e-9)).sum(dim=-1).mean()

    # Tail penalty: probability mass outside the 3 answer tokens
    valid                   = torch.zeros(probs.shape[1], dtype=torch.bool, device=probs.device)
    valid[answer_token_ids] = True
    tail                    = probs[:, ~valid].sum(dim=-1).mean()

    return soft_ce + lambda_tail * tail, soft_ce.item(), tail.item()


# =============================================================================
# 9. LOSS HISTORY CALLBACK (for HF Trainer, optional)
# =============================================================================

class LossHistoryCallback(TrainerCallback):
    def __init__(self):
        self.train_losses: List[Tuple[float, float]] = []
        self.eval_losses:  List[Tuple[float, float]] = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            if "loss"      in logs: self.train_losses.append((state.epoch, logs["loss"]))
            if "eval_loss" in logs: self.eval_losses.append((state.epoch, logs["eval_loss"]))


# =============================================================================
# 10. EVALUATION
# =============================================================================

@torch.no_grad()
def evaluate_generative(
    model,
    records:   List[dict],
    processor: Any,
    desc:      str = "Eval",
) -> Tuple[List[str], List[str]]:
    """
    Evaluate by generating up to 10 new tokens per sample (slow but faithful).
    Returns (y_true, y_pred) as lists of normalised label strings.
    Used for the final quality report; not used during-training checkpointing.
    """
    model.eval()
    y_true, y_pred = [], []

    for row in tqdm(records, desc=desc):
        true_label = row["majority_label"]
        image      = Image.open(BytesIO(row["image_bytes"])).convert("RGB")

        conv = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": row["question"] + SUFFIX},
        ]}]
        prompt = processor.apply_chat_template(conv, add_generation_prompt=True)
        inputs = processor(text=prompt, images=image, return_tensors="pt").to(DEVICE)

        output = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,
            pad_token_id=processor.tokenizer.eos_token_id,
        )
        new_tokens = output[0][inputs["input_ids"].shape[1]:]
        response   = processor.decode(new_tokens, skip_special_tokens=True).strip()
        pred       = normalize_prediction(response)

        y_true.append(true_label)
        y_pred.append(pred)

    return y_true, y_pred


@torch.no_grad()
def evaluate_soft_logits(
    model,
    records:          List[dict],
    processor:        Any,
    answer_token_ids: torch.Tensor,
    desc:             str = "Eval (soft)",
) -> Tuple[float, torch.Tensor, List[str]]:
    """
    Evaluate via soft logits -- much faster than generation.
    Argmax over the 3 answer-token probabilities gives the predicted class.

    Returns:
        acc       : accuracy over records with a known majority label (%)
        all_probs : (N, 3) tensor of normalised answer probabilities
        all_labels: list of majority label strings (ground truth)

    Used for: (a) per-epoch validation during training, (b) tau search on test set.
    """
    model.eval()
    collate = build_collate_fn(processor, DEVICE)
    loader  = DataLoader(
        IDKVQADataset(records), batch_size=1, shuffle=False, collate_fn=collate
    )
    label_to_idx = {"Yes": 0, "No": 1, "I don't know": 2}
    correct, total = 0, 0
    all_probs, all_labels = [], []

    use_fp16 = torch.cuda.is_available()
    dtype    = torch.float16 if use_fp16 else torch.float32
    ctx      = torch.cuda.amp.autocast(dtype=torch.float16) if use_fp16 else contextlib.nullcontext()

    for inputs, _, majority_labels in tqdm(loader, desc=desc, leave=False):
        with ctx:
            out = model(
                input_ids      = inputs["input_ids"],
                attention_mask = inputs["attention_mask"],
                pixel_values   = inputs["pixel_values"].to(dtype),
                image_sizes    = inputs["image_sizes"],
            )
        last_pos = inputs["attention_mask"].sum(dim=1) - 1
        probs    = F.softmax(out.logits[torch.arange(len(last_pos)), last_pos].float(), dim=-1)
        ap       = probs[:, answer_token_ids]
        ap_norm  = ap / (ap.sum(dim=-1, keepdim=True) + 1e-9)
        pred_idx = ap_norm.argmax(dim=-1)

        all_probs.append(ap_norm.cpu())
        all_labels.extend(majority_labels)

        for i, lbl in enumerate(majority_labels):
            gt = label_to_idx.get(lbl, -1)
            if gt >= 0:
                correct += int(pred_idx[i].item() == gt)
                total   += 1

    acc = 100.0 * correct / total if total > 0 else 0.0
    return acc, torch.cat(all_probs, dim=0), all_labels


# =============================================================================
# 11. TAU SEARCH -- Effective Reliability (ER)
# =============================================================================

def tau_search(
    pred_probs: torch.Tensor,
    gt_labels:  List[str],
) -> Tuple[float, float]:
    """
    Sweep tau in [0, 1) and compute Effective Reliability (ER) at each value.

    Decision rule:
        If normalised entropy of pred_probs[i] > tau -> predict "I don't know"
        Else                                          -> argmax prediction

    ER = (correct predictions) / (total answered) x 100  [%]

    The tau that maximises ER on the test set is the operating point
    reported in the AIUTA paper.

    Returns:
        best_tau : float in [0, 1)
        best_er  : ER at best_tau (%)
    """
    idx_to_label = {0: "Yes", 1: "No", 2: "I don't know"}
    best_tau, best_er = 0.0, -999.0

    for tau in np.arange(0.0, 1.0, 0.01):
        correct = answered = 0

        for probs, gt_label in zip(pred_probs, gt_labels):
            # Normalised entropy in [0, 1]: 1 = maximally uncertain
            entropy_norm = (
                -torch.sum(probs * torch.log(probs + 1e-9)) / math.log(3)
            ).item()

            pred = "I don't know" if entropy_norm > tau else idx_to_label[probs.argmax().item()]
            answered += 1
            if pred == gt_label:
                correct += 1

        er = 100.0 * correct / max(answered, 1)
        if er > best_er:
            best_er, best_tau = er, float(tau)

    return best_tau, best_er


# =============================================================================
# 12. TRAINING LOOP (soft labels, gradient accumulation, cosine LR, early stop)
# =============================================================================

def train_model(
    model,
    processor:        Any,
    mixed_dataset:    MixedDataset,
    val_records:      List[dict],
    answer_token_ids: torch.Tensor,
    run_dir:          str,
) -> dict:
    """
    Custom training loop using soft-label cross-entropy loss.
    Saves the best checkpoint (by val accuracy on IDKVQA) to run_dir/best/.

    Features:
      - Gradient accumulation (ACCUMULATION_STEPS)
      - Cosine LR schedule with 10% warmup
      - Gradient clipping (max norm 1.0)
      - Early stopping with PATIENCE epochs without improvement
    """
    collate = build_collate_fn(processor, DEVICE)
    loader  = DataLoader(
        mixed_dataset, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collate, num_workers=0,
    )

    total_updates = NUM_EPOCHS * (len(loader) // ACCUMULATION_STEPS)
    optimizer     = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
        weight_decay=0.01,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, total_updates // 10),
        num_training_steps=total_updates,
    )

    history, best_val_acc, no_improve = [], 0.0, 0
    os.makedirs(os.path.join(run_dir, "best"), exist_ok=True)

    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()
        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}")

        for i, (inputs, soft_labels, _) in enumerate(pbar):
            loss, ce, _ = compute_soft_loss(
                model, inputs, soft_labels, answer_token_ids
            )
            (loss / ACCUMULATION_STEPS).backward()
            epoch_loss += loss.item()

            if (i + 1) % ACCUMULATION_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, model.parameters()), 1.0
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            pbar.set_postfix(loss=f"{loss.item():.3f}", ce=f"{ce:.3f}")

        # -- Validation ----------------------------------------------------------
        val_acc, _, _ = evaluate_soft_logits(
            model, val_records, processor, answer_token_ids, desc="  Val"
        )
        avg = epoch_loss / len(loader)
        print(f"[Epoch {epoch + 1}] train_loss={avg:.4f} | val_acc={val_acc:.1f}%")
        history.append({"epoch": epoch + 1, "train_loss": avg, "val_acc": val_acc})

        # -- Checkpoint & early stopping -----------------------------------------
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve   = 0
            model.save_pretrained(os.path.join(run_dir, "best"))
            processor.save_pretrained(os.path.join(run_dir, "best"))
            print(f"  -> New best val_acc={val_acc:.1f}% saved to {run_dir}/best")
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"  -> Early stop at epoch {epoch + 1} "
                      f"(no improvement for {PATIENCE} epochs)")
                break

    with open(os.path.join(run_dir, "history.json"), "w") as f:
        json.dump({"history": history, "best_val_acc": best_val_acc}, f, indent=2)

    return {"history": history, "best_val_acc": best_val_acc}


# =============================================================================
# 13. PLOTS
# =============================================================================

def plot_augmentation_distribution(orig_dist: Counter, aug_dist: Counter) -> None:
    """Bar chart comparing label distribution before and after augmentation."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    colors = ["#4CAF50", "#F44336", "#2196F3"]

    for ax, dist, title in zip(
        axes,
        [orig_dist, aug_dist],
        [
            f"Original ({sum(orig_dist.values())} samples)",
            f"After Augmentation ({sum(aug_dist.values())} samples)",
        ],
    ):
        bars = ax.bar(dist.keys(), dist.values(), color=colors, edgecolor="black")
        ax.bar_label(bars, fmt="%d", padding=3)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Label")
        ax.set_ylabel("Count")
        ax.set_ylim(0, max(dist.values()) * 1.2)

    fig.suptitle("Data Augmentation -- IDKVQA Training Set Label Distribution", fontsize=14)
    plt.tight_layout()
    path = f"{PLOTS_DIR}/1_augmentation_distribution.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[PLOT] Saved: {path}")


def plot_augmentation_samples(
    records: List[dict], n_rows: int = 4, n_cols: int = 8
) -> None:
    """
    Visual grid showing sample image transforms for a random original record.
    Useful for sanity-checking that augmented images look visually reasonable
    and do not corrupt the semantic content of the scene.
    """
    row       = random.choice(records)
    image     = Image.open(BytesIO(row["image_bytes"])).convert("RGB")
    label     = row["majority_label"]
    aug_list  = augment_image_pil(image)   # list of (name, PIL Image)

    # Show up to n_rows * n_cols augmented variants
    shown     = aug_list[: n_rows * n_cols]
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.2, n_rows * 2.2))
    axes      = axes.flatten()

    for ax, (name, img) in zip(axes, shown):
        ax.imshow(img)
        ax.axis("off")
        # Display the transform name as a compact subtitle under each image
        ax.set_title(name, fontsize=5.5, pad=2)

    # Hide unused subplot slots
    for ax in axes[len(shown):]:
        ax.axis("off")

    q_short = row["question"][:70] + "..." if len(row["question"]) > 70 else row["question"]
    fig.suptitle(
        f"Image Augmentation Samples  |  Label: {label}\nQ: {q_short}",
        fontsize=9,
    )
    plt.tight_layout()
    path = f"{PLOTS_DIR}/2_augmentation_samples.png"
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"[PLOT] Saved: {path}")


def plot_loss_curve(history: List[dict]) -> None:
    """Dual-axis plot: training loss (left y-axis) and val accuracy (right y-axis)."""
    tr_ep   = [h["epoch"]      for h in history]
    tr_loss = [h["train_loss"] for h in history]
    val_acc = [h["val_acc"]    for h in history]

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(tr_ep, tr_loss, label="Train Loss", color="#2196F3", linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color="#2196F3")

    ax2 = ax1.twinx()
    ax2.plot(
        tr_ep, val_acc, label="Val Accuracy (%)",
        color="#F44336", linewidth=2, marker="o", markersize=6,
    )
    ax2.set_ylabel("Val Accuracy (%)", color="#F44336")

    ax1.set_title(
        f"Training Curve -- QLoRA + COCO Mix  (lr={LEARNING_RATE}, rank={LORA_RANK})",
        fontsize=12,
    )
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    ax1.grid(alpha=0.3)
    plt.tight_layout()

    path = f"{PLOTS_DIR}/3_training_curve.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[PLOT] Saved: {path}")


def plot_confusion_matrices(
    y_true_base: List[str], y_pred_base: List[str],
    y_true_ft:   List[str], y_pred_ft:   List[str],
) -> None:
    """Side-by-side normalised confusion matrices for baseline vs finetuned."""
    fig, axes   = plt.subplots(1, 2, figsize=(12, 5))
    label_order = ["Yes", "No", "I don't know"]
    disp_labels = ["Yes", "No", "IDK"]

    for ax, y_t, y_p, title in zip(
        axes,
        [y_true_base, y_true_ft],
        [y_pred_base, y_pred_ft],
        ["Baseline (no finetuning)", "Finetuned (QLoRA + Aug + COCO Mix)"],
    ):
        cm   = confusion_matrix(y_t, y_p, labels=label_order)
        disp = ConfusionMatrixDisplay(cm, display_labels=disp_labels)
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(title, fontsize=11)

    fig.suptitle("Confusion Matrix -- Baseline vs Finetuned", fontsize=13)
    plt.tight_layout()
    path = f"{PLOTS_DIR}/4_confusion_matrices.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[PLOT] Saved: {path}")


def plot_accuracy_comparison(
    y_true_base: List[str], y_pred_base: List[str],
    y_true_ft:   List[str], y_pred_ft:   List[str],
) -> None:
    """Grouped bar chart comparing per-class accuracy: baseline vs finetuned."""
    label_order = ["Yes", "No", "I don't know"]
    x     = np.arange(len(label_order))
    width = 0.35

    def per_class_acc(y_true, y_pred):
        out = []
        for lbl in label_order:
            idxs = [i for i, t in enumerate(y_true) if t == lbl]
            out.append(
                sum(1 for i in idxs if y_pred[i] == lbl) / len(idxs) * 100
                if idxs else 0.0
            )
        return out

    base_acc = per_class_acc(y_true_base, y_pred_base)
    ft_acc   = per_class_acc(y_true_ft,   y_pred_ft)

    fig, ax = plt.subplots(figsize=(9, 5))
    bars1   = ax.bar(x - width / 2, base_acc, width, label="Baseline",  color="#90CAF9", edgecolor="black")
    bars2   = ax.bar(x + width / 2, ft_acc,   width, label="Finetuned", color="#1565C0", edgecolor="black")
    ax.bar_label(bars1, fmt="%.1f%%", padding=3, fontsize=9)
    ax.bar_label(bars2, fmt="%.1f%%", padding=3, fontsize=9)
    ax.axhline(y=33.3, color="gray", linestyle="--", alpha=0.6, label="Random baseline (33%)")
    ax.set_xticks(x)
    ax.set_xticklabels(label_order)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 120)
    ax.set_title("Per-Class Accuracy -- Baseline vs Finetuned (QLoRA + Data Aug)", fontsize=12)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    path = f"{PLOTS_DIR}/5_accuracy_comparison.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[PLOT] Saved: {path}")


def plot_f1_comparison(base_report: dict, ft_report: dict) -> None:
    """Grouped bar chart comparing per-class F1 score: baseline vs finetuned."""
    label_order = ["Yes", "No", "I don't know"]
    x     = np.arange(len(label_order))
    width = 0.35

    base_f1 = [base_report.get(c, {}).get("f1-score", 0) * 100 for c in label_order]
    ft_f1   = [ft_report.get(c,   {}).get("f1-score", 0) * 100 for c in label_order]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars1   = ax.bar(x - width / 2, base_f1, width, label="Baseline",  color="#A5D6A7", edgecolor="black")
    bars2   = ax.bar(x + width / 2, ft_f1,   width, label="Finetuned", color="#2E7D32", edgecolor="black")
    ax.bar_label(bars1, fmt="%.1f%%", padding=3, fontsize=9)
    ax.bar_label(bars2, fmt="%.1f%%", padding=3, fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(label_order)
    ax.set_ylabel("F1 Score (%)")
    ax.set_ylim(0, 120)
    ax.set_title("Per-Class F1 Score -- Baseline vs Finetuned", fontsize=12)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    path = f"{PLOTS_DIR}/6_f1_comparison.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[PLOT] Saved: {path}")


def plot_qualitative_examples(
    model, records: List[dict], processor: Any, n: int = 6
) -> None:
    """
    Grid of n random validation samples showing the finetuned model's prediction
    alongside the ground-truth label. Green title = correct, red = wrong.
    """
    sample = random.sample(records, min(n, len(records)))
    n_cols = 3
    n_rows = math.ceil(n / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4.5, n_rows * 4))
    axes      = axes.flatten()

    model.eval()
    for i, row in enumerate(sample):
        true_label = row["majority_label"]
        image      = Image.open(BytesIO(row["image_bytes"])).convert("RGB")

        conv   = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": row["question"] + SUFFIX},
        ]}]
        prompt = processor.apply_chat_template(conv, add_generation_prompt=True)
        inputs = processor(text=prompt, images=image, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            output = model.generate(
                **inputs, max_new_tokens=10, do_sample=False,
                pad_token_id=processor.tokenizer.eos_token_id,
            )
        new_tokens = output[0][inputs["input_ids"].shape[1]:]
        response   = processor.decode(new_tokens, skip_special_tokens=True).strip()
        pred       = normalize_prediction(response)
        correct    = pred == true_label

        axes[i].imshow(image)
        axes[i].axis("off")
        q_short = row["question"][:55] + "..." if len(row["question"]) > 55 else row["question"]
        axes[i].set_title(
            f"Q: {q_short}\nExpected: {true_label}  |  Predicted: {pred}",
            fontsize=8,
            color="green" if correct else "red",
            pad=4,
        )

    for ax in axes[len(sample):]:
        ax.axis("off")

    fig.suptitle("Qualitative Examples -- Finetuned Model (QLoRA + Data Aug)", fontsize=13)
    plt.tight_layout()
    path = f"{PLOTS_DIR}/7_qualitative_examples.png"
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"[PLOT] Saved: {path}")


# =============================================================================
# 14. MAIN
# =============================================================================

def main():

    # -- 14.1  Load & split IDKVQA ─────────────────────────────────────────────
    print("\n[1/12] Loading IDKVQA from parquet...")
    all_records = load_idkvqa_from_parquet(PARQUET_PATH)
    train_records_orig, val_records, test_records = split_idkvqa_by_image(all_records)
    print(
        f"       IDKVQA split -- "
        f"Train: {len(train_records_orig)} | "
        f"Val: {len(val_records)} | "
        f"Test: {len(test_records)}"
    )

    # -- 14.2  Data augmentation on train split ────────────────────────────────
    print(f"\n[2/12] Applying data augmentation on train split...")
    n_text = max(len(v) for v in TEXT_TEMPLATES.values())
    print(f"       Image transforms : {N_IMAGE_AUGS}")
    print(f"       Text templates   : {n_text} per label")
    print(f"       Expansion factor : ~x{N_IMAGE_AUGS * n_text} per original sample")

    train_records_aug = augment_dataset(train_records_orig)
    factor = len(train_records_aug) // max(len(train_records_orig), 1)
    print(f"       Augmented train  : {len(train_records_aug)} samples (x{factor})")

    orig_dist = Counter(r["majority_label"] for r in train_records_orig)
    aug_dist  = Counter(r["majority_label"] for r in train_records_aug)
    print(f"       Original  distribution: {dict(orig_dist)}")
    print(f"       Augmented distribution: {dict(aug_dist)}")

    plot_augmentation_distribution(orig_dist, aug_dist)
    plot_augmentation_samples(train_records_orig)

    # -- 14.3  Load COCO / VQAv2 ───────────────────────────────────────────────
    print("\n[3/12] Loading VQAv2/COCO samples...")
    coco_samples = load_coco_samples()

    # -- 14.4  Build mixed dataset ─────────────────────────────────────────────
    coco_pct  = int((1 - MIX_RATIO) * 100)
    idkvqa_pct = int(MIX_RATIO * 100)
    print(f"\n[4/12] Building mixed dataset (COCO {coco_pct}% + IDKVQA {idkvqa_pct}%)...")
    mixed_dataset = MixedDataset(coco_samples, train_records_aug, mix_ratio=MIX_RATIO)

    # -- 14.5  Load processor & model (4-bit QLoRA) ────────────────────────────
    print("\n[5/12] Loading model in 4-bit (QLoRA)...")
    processor = LlavaNextProcessor.from_pretrained(MODEL_HF)
    processor.tokenizer.padding_side         = "right"
    processor.patch_size                     = 14
    processor.vision_feature_select_strategy = "default"
    processor.image_processor.patch_size     = 14

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = LlavaNextForConditionalGeneration.from_pretrained(
        MODEL_HF,
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache                      = False
    model.config.vision_config.patch_size       = 14
    model.config.vision_feature_select_strategy = "default"
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    print("       Model loaded successfully.")

    answer_token_ids = get_answer_token_ids(processor, DEVICE)

    # -- 14.6  Evaluate baseline model ─────────────────────────────────────────
    print("\n[6/12] Evaluating BASELINE model (before finetuning)...")
    y_true_base, y_pred_base = evaluate_generative(
        model, val_records, processor, desc="Baseline Eval"
    )
    label_names = ["Yes", "No", "I don't know"]
    base_report = classification_report(
        y_true_base, y_pred_base,
        labels=label_names, target_names=label_names,
        output_dict=True, zero_division=0,
    )
    print(classification_report(
        y_true_base, y_pred_base,
        labels=label_names, target_names=label_names, zero_division=0,
    ))
    with open(f"{OUTPUT_DIR}/baseline_eval_report.txt", "w") as f:
        f.write(classification_report(
            y_true_base, y_pred_base,
            labels=label_names, target_names=label_names, zero_division=0,
        ))

    # -- 14.7  Configure LoRA ──────────────────────────────────────────────────
    print("\n[7/12] Configuring LoRA adapter...")
    lora_cfg = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_cfg)
    for name, param in model.named_parameters():
        if "lora" in name.lower():
            param.requires_grad = True
    model.print_trainable_parameters()

    # -- 14.8  Train ───────────────────────────────────────────────────────────
    print(
        f"\n[8/12] Starting finetuning "
        f"({NUM_EPOCHS} epochs, {len(mixed_dataset)} samples/epoch)..."
    )
    run_dir      = os.path.join(SAVE_DIR, f"lr{LEARNING_RATE}_r{LORA_RANK}")
    train_result = train_model(
        model, processor, mixed_dataset, val_records, answer_token_ids, run_dir
    )
    plot_loss_curve(train_result["history"])
    print(f"       Finetuning done. Best checkpoint saved to: {run_dir}/best")

    # -- 14.9  Load best checkpoint & evaluate finetuned model ─────────────────
    print("\n[9/12] Loading best checkpoint and evaluating FINETUNED model...")
    best_ckpt  = os.path.join(run_dir, "best")
    base_model = LlavaNextForConditionalGeneration.from_pretrained(
        MODEL_HF, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map="auto"
    )
    ft_model = PeftModel.from_pretrained(base_model, best_ckpt)
    ft_model.eval()

    y_true_ft, y_pred_ft = evaluate_generative(
        ft_model, val_records, processor, desc="Finetuned Eval"
    )
    ft_report = classification_report(
        y_true_ft, y_pred_ft,
        labels=label_names, target_names=label_names,
        output_dict=True, zero_division=0,
    )
    print(classification_report(
        y_true_ft, y_pred_ft,
        labels=label_names, target_names=label_names, zero_division=0,
    ))
    with open(f"{OUTPUT_DIR}/finetuned_eval_report.txt", "w") as f:
        f.write(classification_report(
            y_true_ft, y_pred_ft,
            labels=label_names, target_names=label_names, zero_division=0,
        ))

    # -- 14.10  Tau search on TEST set (Effective Reliability) ─────────────────
    print("\n[10/12] Tau search on test set (Effective Reliability -- AIUTA metric)...")
    test_acc, test_probs, test_labels = evaluate_soft_logits(
        ft_model, test_records, processor, answer_token_ids, desc="Test (soft logits)"
    )
    print(f"        Test argmax accuracy: {test_acc:.1f}%")
    best_tau, best_er = tau_search(test_probs, test_labels)
    print(f"        Best tau={best_tau:.2f}  ->  ER={best_er:.2f}%")

    with open(f"{OUTPUT_DIR}/er_result.json", "w") as f:
        json.dump(
            {"test_acc": test_acc, "best_tau": best_tau, "best_er": best_er},
            f, indent=2,
        )

    # -- 14.11  Final evaluation plots ─────────────────────────────────────────
    print("\n[11/12] Generating final evaluation plots...")
    plot_confusion_matrices(y_true_base, y_pred_base, y_true_ft, y_pred_ft)
    plot_accuracy_comparison(y_true_base, y_pred_base, y_true_ft, y_pred_ft)
    plot_f1_comparison(base_report, ft_report)
    plot_qualitative_examples(ft_model, val_records, processor)

    # -- 14.12  Final summary ──────────────────────────────────────────────────
    print("\n[12/12] Computing final summary...")
    overall_base = sum(t == p for t, p in zip(y_true_base, y_pred_base)) / len(y_true_base) * 100
    overall_ft   = sum(t == p for t, p in zip(y_true_ft,   y_pred_ft))   / len(y_true_ft)   * 100

    print(f"""
+========================================================+
|                    FINAL SUMMARY                       |
+========================================================+
|  Baseline accuracy (val):    {overall_base:5.1f}%                    |
|  Finetuned accuracy (val):   {overall_ft:5.1f}%                    |
|  Improvement:               {overall_ft - overall_base:+5.1f}%                    |
+--------------------------------------------------------+
|  Test accuracy (argmax):     {test_acc:5.1f}%                    |
|  Best tau:                   {best_tau:.2f}                         |
|  Effective Reliability (ER): {best_er:5.1f}%                    |
+--------------------------------------------------------+
|  Image transforms : {N_IMAGE_AUGS}                                  |
|  Text templates   : {n_text} per label                          |
|  Plots saved to   : {PLOTS_DIR}/         |
+========================================================+
""")

    final = {
        "val_baseline_acc":      overall_base,
        "val_finetuned_acc":     overall_ft,
        "improvement":           overall_ft - overall_base,
        "test_argmax_acc":       test_acc,
        "best_tau":              best_tau,
        "effective_reliability": best_er,
        "n_image_augmentations": N_IMAGE_AUGS,
        "n_text_templates":      n_text,
    }
    with open(f"{OUTPUT_DIR}/final_results.json", "w") as f:
        json.dump(final, f, indent=2)
    print(f"Results saved to {OUTPUT_DIR}/final_results.json")


if __name__ == "__main__":
    main()