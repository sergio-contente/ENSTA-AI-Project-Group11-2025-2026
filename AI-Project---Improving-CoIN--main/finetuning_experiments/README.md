<h1 align="center">
    Fine-tuning LLaVA-1.6 with QLoRA<br/>on IDKVQA & COCO/VQAv2
</h1>

<p align="center">
    <b>AI Project — ENSTA Paris, 2025-2026</b>
</p>

<p align="center">
<img src="https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white"/>
<img src="https://img.shields.io/badge/LLaVA--1.6-Mistral--7B-green?style=for-the-badge"/>
<img src="https://img.shields.io/badge/QLoRA-4bit-blueviolet?style=for-the-badge"/>
</p>

---

## Overview

Fine-tuning of [LLaVA-1.6](https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf) (Mistral-7B) using **QLoRA** (4-bit quantization) for uncertainty-aware visual question answering (Yes / No / I don't know), as used in the [AIUTA](https://intelligolabs.github.io/CoIN/) navigation pipeline.

The core strategy mixes a **large general-purpose dataset** (COCO/VQAv2) with the **small task-specific dataset** (IDKVQA) to prevent overfitting while optimizing for the Yes/No/IDK classification used in embodied navigation.

---

## Files

### `finetune_llava_idkvqa.py`

Standalone Python script — the **full fine-tuning pipeline** designed to run on a GPU cluster (e.g. via SLURM).

### `coin_finetune_coco_fixed.ipynb`

Jupyter notebook — **interactive version** of the same pipeline, designed for step-by-step exploration and debugging in JupyterLab on a GPU node. Includes inline outputs and visualizations for inspecting training progress and evaluation results.

---

## Pipeline

Both files implement the same 7-step pipeline:

| Step | Description |
|------|-------------|
| 1 | Load IDKVQA dataset (parquet) and stream COCO/VQAv2 from HuggingFace |
| 2 | Split IDKVQA by image SHA-1 hash (60% train / 20% val / 20% test) — no data leakage |
| 3 | Data augmentation on training samples (text templates x image transforms) |
| 4 | Mix COCO (general VQA) + IDKVQA (task-specific) with soft labels from human vote distributions |
| 5 | Fine-tune LLaVA-1.6 with QLoRA + cosine scheduler + early stopping |
| 6 | Evaluate both baseline and fine-tuned models: accuracy, F1, confusion matrix |
| 7 | Tau search on the test set to compute Effective Reliability (ER) as defined in the AIUTA paper |

---

## Data Augmentation

Each training sample is expanded via a cross-product of **text template variations** and **image transforms**, producing ~120 samples per original record.

### Text Templates (4 per label)

```
Original:       "{question}"
Variant 1:      "Looking at the image, {question}"
Variant 2:      "Based on what you see, {question}"
Variant 3:      "In the image shown, {question}"
```

### Image Transforms (30 total)

| Category | Transforms |
|----------|------------|
| **Original** | Identity (no change) |
| **Geometric — flips** | Horizontal flip, vertical flip |
| **Geometric — rotations** | +/-10, +/-20, +/-45 degrees |
| **Geometric — crops** | Random crop at 75%, 85%, 90% scale |
| **Photometric — brightness** | +/-30%, +/-50% |
| **Photometric — contrast** | +30%, -20% |
| **Photometric — saturation** | +/-50% |
| **Photometric — hue shift** | +/-0.05 |
| **Texture / focus** | Gaussian blur (r=1, r=2), sharpen (2x, 3x), grayscale |
| **Combined** | flip+blur, crop+bright, rotate+contrast, crop+grayscale, flip+sat_down, rotate+sharpen |

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| **Model** | `llava-hf/llava-v1.6-mistral-7b-hf` |
| **Quantization** | QLoRA 4-bit (NF4, double quantization) |
| **LoRA rank** | 32 |
| **LoRA alpha** | 64 |
| **LoRA dropout** | 0.05 |
| **Batch size** | 1 (gradient accumulation = 8, effective = 8) |
| **Learning rate** | 3e-5 |
| **Scheduler** | Cosine with warmup |
| **Epochs** | 4 (early stopping, patience = 2) |
| **Max sequence length** | 2048 |
| **COCO samples per class** | 1000 |
| **Mix ratio** | 80% COCO + 20% IDKVQA |
| **IDKVQA splits** | 60% train / 20% val / 20% test (by image hash) |

---

## Evaluation

The pipeline evaluates both the **baseline** (pre-trained LLaVA-1.6) and **fine-tuned** models on the held-out IDKVQA test set:

- **Accuracy** and **F1 score** (per-class and macro)
- **Confusion matrix** with comparative plots (baseline vs fine-tuned)
- **Tau search** — sweep over confidence thresholds to find the optimal tau for **Effective Reliability (ER)**, the primary metric from the AIUTA paper that penalizes confident wrong answers more than abstentions

---

## Dependencies

```
torch >= 2.0
transformers >= 4.43
peft
datasets
pillow
scikit-learn
matplotlib
pandas
tqdm
bitsandbytes
```
