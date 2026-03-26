"""vlmr1_detector.py -- VLM-R1 detector adapter.
Refactored from benchmark_ovd.py::run_single_inference()."""
from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .base import AbstractDetector, Detection, DetectionResult
from .output_parser import OutputParser
from .prompt_templates import PromptBuilder
from ..utils.model_loader import ModelLoader


MAX_AREA_RATIO = 0.30
MIN_AREA_PX = 40 * 40
MAX_ASPECT_RATIO = 4.0


class VLMr1Detector(AbstractDetector):
    def __init__(self, config):
        self._config = config
        self._loader = ModelLoader.get_instance(config.model)

    @staticmethod
    def _filter_detections(
        dets: list[Detection], width: int, height: int
    ) -> list[Detection]:
        """Reject oversized, tiny, elongated, and origin-anchored bboxes."""
        frame_area = width * height
        if frame_area <= 0:
            return dets
        filtered: list[Detection] = []
        for d in dets:
            x1, y1, x2, y2 = d.bbox
            det_w = max(0.0, x2 - x1)
            det_h = max(0.0, y2 - y1)
            det_area = det_w * det_h
            ratio = det_area / frame_area
            if ratio > MAX_AREA_RATIO:
                print(
                    f"[DETECT_FILTER] REJECTED {d.label} bbox={d.bbox} "
                    f"area_ratio={ratio:.1%} > {MAX_AREA_RATIO:.0%}"
                )
                continue
            if det_area < MIN_AREA_PX:
                print(
                    f"[DETECT_FILTER] REJECTED {d.label} bbox={d.bbox} "
                    f"too small ({det_area:.0f}px < {MIN_AREA_PX})"
                )
                continue
            if det_w > 0 and det_h > 0:
                aspect = max(det_w / det_h, det_h / det_w)
                if aspect > MAX_ASPECT_RATIO:
                    print(
                        f"[DETECT_FILTER] REJECTED {d.label} bbox={d.bbox} "
                        f"aspect={aspect:.1f} > {MAX_ASPECT_RATIO}"
                    )
                    continue
            if x1 == 0 and y1 == 0 and ratio > 0.20:
                print(
                    f"[DETECT_FILTER] REJECTED {d.label} bbox={d.bbox} "
                    f"origin-anchored + large ({ratio:.1%})"
                )
                continue
            filtered.append(d)
        if not filtered and dets:
            print(f"[DETECT_FILTER] All {len(dets)} detections filtered out")
        return filtered

    @staticmethod
    def _crop_for_bbox(observation: np.ndarray, bbox: list[float]) -> np.ndarray | None:
        h, w = observation.shape[:2]
        x1, y1, x2, y2 = bbox
        print(f"[CROP_DEBUG] obs_size={w}x{h}, raw_bbox=[{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")

        # Qwen2.5-VL may output coords in a 1000x1000 normalized grid
        if max(x1, x2) <= 1000 and max(y1, y2) <= 1000 and (x2 > w or y2 > h):
            print(f"[CROP_DEBUG] Detected 1000-grid coords -- rescaling to {w}x{h}")
            x1, y1 = x1 * w / 1000, y1 * h / 1000
            x2, y2 = x2 * w / 1000, y2 * h / 1000

        if x2 > w * 1.5 or y2 > h * 1.5:
            print(
                f"[CROP_DEBUG] bbox coords ({x2:.0f},{y2:.0f}) >> "
                f"obs dims ({w},{h}) -- likely WRONG SPACE"
            )

        xi1 = max(0, min(w - 1, int(round(x1))))
        yi1 = max(0, min(h - 1, int(round(y1))))
        xi2 = max(0, min(w, int(round(x2))))
        yi2 = max(0, min(h, int(round(y2))))
        crop_w, crop_h = xi2 - xi1, yi2 - yi1
        print(f"[CROP_DEBUG] clipped=[{xi1},{yi1},{xi2},{yi2}], crop_size={crop_w}x{crop_h}")

        if xi2 <= xi1 or yi2 <= yi1:
            print("[CROP_DEBUG] EMPTY CROP -> returning None")
            return None

        area_ratio = (crop_w * crop_h) / (w * h)
        print(f"[CROP_DEBUG] crop OK, area_ratio={area_ratio:.2%}")
        return observation[yi1:yi2, xi1:xi2].copy()

    def _run_forward(
        self,
        messages: list,
    ) -> tuple[str, DetectionResult]:
        from qwen_vl_utils import process_vision_info

        proc = self._loader.processor
        t_pre0 = time.perf_counter()
        text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        img_in, vid_in = process_vision_info(messages)
        inputs = proc(
            text=[text], images=img_in, videos=vid_in, padding=True, return_tensors="pt"
        ).to(self._loader.device)
        t_pre = time.perf_counter() - t_pre0

        t_gen0 = time.perf_counter()
        with torch.inference_mode():
            gen = self._loader.model.generate(
                **inputs,
                max_new_tokens=self._config.model.max_new_tokens,
                do_sample=False,
                use_cache=False,
            )
        t_gen = time.perf_counter() - t_gen0

        t_parse0 = time.perf_counter()
        trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, gen)]
        raw = proc.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        parsed = OutputParser.parse_full(raw)
        dets = []
        if parsed.bboxes:
            for item in parsed.bboxes:
                if isinstance(item, dict) and "bbox_2d" in item and "label" in item:
                    try:
                        dets.append(Detection(
                            bbox=[float(b) for b in item["bbox_2d"]],
                            label=item["label"].strip().lower(),
                            reasoning=parsed.reasoning_text,
                        ))
                    except (ValueError, TypeError):
                        pass
        t_parse = time.perf_counter() - t_parse0
        total = t_pre + t_gen + t_parse

        return raw, DetectionResult(
            detections=dets,
            raw_output=raw,
            reasoning_text=parsed.reasoning_text,
            json_valid=parsed.json_valid,
            latency_sec=total,
            preprocess_latency_sec=t_pre,
            generate_latency_sec=t_gen,
            parse_latency_sec=t_parse,
        )

    def detect(self, image_path: str, target_categories: list[str],
               kg_context: str | None = None) -> DetectionResult:
        builder = PromptBuilder().set_categories(target_categories)
        if kg_context:
            builder.set_kg_context(kg_context).set_scene_type("indoor")
        system, user_prompt = builder.build()
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": [
                {"type": "image", "image": f"file://{Path(image_path).resolve()}"},
                {"type": "text", "text": user_prompt},
            ]},
        ]
        _, result = self._run_forward(messages)
        try:
            from PIL import Image as _PILImg
            _im = _PILImg.open(image_path)
            result.detections = self._filter_detections(
                result.detections, _im.width, _im.height
            )
        except Exception:
            pass
        return result

    def detect_from_observation(self, observation: np.ndarray,
                                 target_categories: list[str],
                                 kg_context: str | None = None) -> DetectionResult:
        pil_image = Image.fromarray(observation.astype(np.uint8))
        builder = PromptBuilder().set_categories(target_categories)
        if kg_context:
            builder.set_kg_context(kg_context).set_scene_type("indoor")
        system, user_prompt = builder.build()
        tmp_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                tmp_path = tmp.name
            pil_image.save(tmp_path, format="JPEG", quality=95)
            img_url = f"file://{Path(tmp_path).resolve()}"

            messages = [
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": img_url,
                            "min_pixels": 256 * 28 * 28,
                            "max_pixels": 512 * 28 * 28,
                        },
                        {"type": "text", "text": user_prompt},
                    ],
                },
            ]
            _, result = self._run_forward(messages)
            h, w = observation.shape[:2]
            result.detections = self._filter_detections(result.detections, w, h)
            crops_ok = 0
            crops_fail = 0
            for d in result.detections:
                crop = self._crop_for_bbox(observation, d.bbox)
                if crop is not None:
                    d.image = crop
                    crops_ok += 1
                else:
                    crops_fail += 1
            print(
                f"[DETECT_DEBUG] {len(result.detections)} detections, "
                f"{crops_ok} crops OK, {crops_fail} crops FAILED"
            )
            return result
        finally:
            if tmp_path is not None:
                try:
                    os.unlink(tmp_path)
                except FileNotFoundError:
                    pass
