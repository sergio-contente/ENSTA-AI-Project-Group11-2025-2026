# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

from __future__ import annotations

import os
import tempfile

import numpy as np
import torch
from PIL import Image

from aiuta_vlmr1.utils.model_loader import ModelLoader


class VLMr1Oracle:
    """
    Replaces VLMOracle (LLaVA) when running with VLM-R1.
    Answers yes/no/IDK questions about the instance_imagegoal using the same
    VLM as the detector via ModelLoader singleton.
    Interface: set_instance_image(img, target) + answer(question) -> str
    """

    def __init__(self) -> None:
        self._instance_image: np.ndarray | None = None
        self._target_object: str = ""
        self._loader = None  # lazy -- loaded when needed

    def set_instance_image(self, instance_image: np.ndarray, target_object: str) -> None:
        self._instance_image = instance_image.astype(np.uint8)
        self._target_object = target_object
        print(f"[VLMr1Oracle] Instance image set for target: {target_object}")

    @staticmethod
    def _is_yesno_question(question: str) -> bool:
        q = question.strip().lower()
        # "A or B" choice needs an open answer, not yes/no
        if q.startswith(("is ", "are ")) and " or " in q and "yes or no" not in q:
            return False
        yesno_prefixes = ("is ", "does ", "can ", "are ", "has ", "do ", "was ", "were ", "could ")
        return q.startswith(yesno_prefixes)

    def answer(self, question: str) -> str:
        """Answer a question about the instance_imagegoal.
        For yes/no questions: returns 'yes', 'no', or 'I don't know'.
        For open questions (What/Which/Where/How): returns the raw answer.
        """
        if self._instance_image is None:
            return "I don't know"
        try:
            if self._loader is None:
                if ModelLoader._instances:
                    key = next(iter(ModelLoader._instances))
                    self._loader = ModelLoader._instances[key]
                else:
                    return "I don't know"

            is_yesno = self._is_yesno_question(question)

            if is_yesno:
                system_prompt = "You are a helpful assistant. Answer only with yes, no, or I don't know."
            else:
                system_prompt = (
                    "You are a helpful assistant. "
                    "Answer the question briefly in 1-3 words based on what you see in the image."
                )

            pil_image = Image.fromarray(self._instance_image)

            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                tmp_path = tmp.name
            try:
                pil_image.save(tmp_path, format="JPEG", quality=95)
                img_url = f"file://{os.path.abspath(tmp_path)}"

                messages = [
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "image": img_url,
                                "min_pixels": 256 * 28 * 28,
                                "max_pixels": 512 * 28 * 28,
                            },
                            {"type": "text", "text": question},
                        ],
                    },
                ]

                from qwen_vl_utils import process_vision_info

                proc = self._loader.processor
                text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                img_in, vid_in = process_vision_info(messages)
                inputs = proc(
                    text=[text],
                    images=img_in,
                    videos=vid_in,
                    padding=True,
                    return_tensors="pt",
                ).to(self._loader.device)

                with torch.inference_mode():
                    gen = self._loader.model.generate(
                        **inputs,
                        max_new_tokens=32,
                        do_sample=False,
                        use_cache=False,
                    )

                trimmed = [o[len(i) :] for i, o in zip(inputs.input_ids, gen)]
                raw = proc.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                raw_l = raw.strip().lower()

                print(f"[VLMr1Oracle] Q: {question!r} -> A: {raw_l!r} (yesno={is_yesno})")

                if is_yesno:
                    if raw_l.startswith("yes"):
                        return "yes"
                    if raw_l.startswith("no"):
                        return "no"
                    return "I don't know"
                else:
                    # Open question: return raw answer stripped of trailing punctuation
                    answer = raw.strip().rstrip(".")
                    if not answer:
                        return "I don't know"
                    return answer
            finally:
                try:
                    os.unlink(tmp_path)
                except FileNotFoundError:
                    pass
        except Exception as e:
            print(f"[VLMr1Oracle] Error: {e}")
            return "I don't know"

    def answer_with_detection_image(
        self,
        detected_crop: np.ndarray,
    ) -> tuple[bool, float]:
        """
        Visually compares the instance_imagegoal with the detected crop.
        Returns (is_match, entropy) where entropy is the model uncertainty [0,1].
        High entropy = uncertain model = possibly the correct object.
        """
        if self._instance_image is None or detected_crop is None:
            return False, 1.0
        try:
            if self._loader is None:
                if ModelLoader._instances:
                    key = next(iter(ModelLoader._instances))
                    self._loader = ModelLoader._instances[key]
                else:
                    return False, 1.0

            import tempfile

            from qwen_vl_utils import process_vision_info

            pil_target = Image.fromarray(self._instance_image.astype(np.uint8))
            pil_detected = Image.fromarray(detected_crop.astype(np.uint8))

            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as t1:
                path_target = t1.name
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as t2:
                path_detected = t2.name

            try:
                pil_target.save(path_target, format="JPEG", quality=95)
                pil_detected.save(path_detected, format="JPEG", quality=95)

                url_target = f"file://{os.path.abspath(path_target)}"
                url_detected = f"file://{os.path.abspath(path_detected)}"

                messages = [
                    {
                        "role": "system",
                        "content": (
                            "You are a visual comparison assistant. "
                            "You will be shown two images of indoor objects. "
                            "Answer only yes or no."
                        ),
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"This is the target {self._target_object} I am looking for:"},
                            {
                                "type": "image",
                                "image": url_target,
                                "min_pixels": 256 * 28 * 28,
                                "max_pixels": 512 * 28 * 28,
                            },
                            {"type": "text", "text": f"This is the {self._target_object} I just detected:"},
                            {
                                "type": "image",
                                "image": url_detected,
                                "min_pixels": 256 * 28 * 28,
                                "max_pixels": 512 * 28 * 28,
                            },
                            {
                                "type": "text",
                                "text": (
                                    "Are these the same specific object instance? "
                                    "Look carefully at color, material, style, size, and distinctive features. "
                                    "Answer only yes or no."
                                ),
                            },
                        ],
                    },
                ]

                proc = self._loader.processor
                text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                img_in, vid_in = process_vision_info(messages)
                inputs = proc(
                    text=[text],
                    images=img_in,
                    videos=vid_in,
                    padding=True,
                    return_tensors="pt",
                ).to(self._loader.device)

                with torch.inference_mode():
                    gen = self._loader.model.generate(
                        **inputs,
                        max_new_tokens=16,
                        do_sample=False,
                        use_cache=False,
                        output_scores=True,
                        return_dict_in_generate=True,
                    )

                trimmed = [o[len(i) :] for i, o in zip(inputs.input_ids, gen.sequences)]
                raw = proc.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                raw_l = raw.strip().lower()
                print(f"[VLMr1Oracle] Visual comparison -> A: {raw_l!r}")

                entropy = 1.0
                if getattr(gen, "scores", None) and len(gen.scores) > 0:
                    from aiuta_vlmr1.evaluation.vlm_inference_utils import compute_logits_entropy

                    logits = gen.scores[0][0]
                    entropy = compute_logits_entropy(logits)

                is_match = raw_l.startswith("yes")
                return is_match, entropy

            finally:
                for p in [path_target, path_detected]:
                    try:
                        os.unlink(p)
                    except FileNotFoundError:
                        pass
        except Exception as e:
            print(f"[VLMr1Oracle] answer_with_detection_image error: {e}")
            return False, 1.0

    def reset(self) -> None:
        self._instance_image = None
        self._target_object = ""
