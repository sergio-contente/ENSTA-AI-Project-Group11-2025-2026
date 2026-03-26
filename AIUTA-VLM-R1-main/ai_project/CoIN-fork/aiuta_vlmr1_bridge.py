"""
Bridge between CoIN/VLFM codebase and our AIUTA-VLM-R1 modules.

Goal:
  - Keep CoIN/VLFM navigation loop intact
  - Swap expensive API-based modules with local VLM-R1 + KG logic

This module is intentionally self-contained so the CoIN fork changes stay surgical.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np

# Entropy threshold for visual judge soft-maybe.
# Below this: model is confident in rejection -> hard "no".
# At or above: model is uncertain -> treat as "maybe" and delegate to KG alignment.
ENTROPY_MAYBE_THRESHOLD = 0.04


def _import_aiuta() -> Any:
    try:
        import aiuta_vlmr1  # noqa: F401
        from aiuta_vlmr1.config import Config as VLMConfig
        from aiuta_vlmr1.pipeline.aiuta_pipeline import AIUTAPipeline
        from aiuta_vlmr1.pipeline.aiuta_pipeline import PolicySignal
        from aiuta_vlmr1.detector.vlmr1_detector import VLMr1Detector
        return VLMConfig, AIUTAPipeline, PolicySignal, VLMr1Detector
    except Exception as e:
        raise ImportError(
            "Could not import `aiuta_vlmr1`. Install it first, e.g.\n"
            "  pip install -e ~/ai_project/aiuta_vlmr1_project\n"
            "Then re-run."
        ) from e


@dataclass
class BridgeDetection:
    bbox_xyxy: list[float]  # [x1,y1,x2,y2] in pixel coords
    label: str
    reasoning: str | None = None
    image: np.ndarray | None = None


@dataclass
class BridgeDetectionResult:
    detections: list[BridgeDetection]
    reasoning_text: str | None


class VLMr1Bridge:
    """
    Minimal bridge used by VLFM policies.

    - `detect(...)` returns bbox + label for compatibility with VLFM's ObjectDetections.
    - `pipeline_step(...)` runs the AIUTA pipeline and returns STOP/CONTINUE + optional question.
    """

    def __init__(self, config_path: str | None = None, ask_human: Optional[Callable[[str], str]] = None) -> None:
        VLMConfig, AIUTAPipeline, PolicySignal, VLMr1Detector = _import_aiuta()
        self._PolicySignal = PolicySignal

        self.config = VLMConfig.from_yaml(config_path) if config_path else VLMConfig()
        self.detector = VLMr1Detector(self.config)
        self.pipeline = AIUTAPipeline(self.config, ask_human=ask_human)
        self._target_category: str = ""
        self._oracle = None
        self._last_visual_entropy = 1.0
        self._cached_det_result: Any = None
        self._cached_det_frame_id: int | None = None

    def _make_vlm_judge(self) -> Callable[[str, str], bool]:
        """Returns (obj_desc, target_desc) -> bool using VLM-R1 visual comparison."""

        def judge(obj_desc: str, target_desc: str, detected_crop=None) -> bool:
            if not hasattr(self, "_oracle") or self._oracle is None:
                return False

            # Count visual comparisons separately from NQ
            try:
                self.pipeline._num_visual_comparisons = getattr(
                    self.pipeline, "_num_visual_comparisons", 0
                ) + 1
            except Exception:
                pass

            if detected_crop is not None:
                try:
                    is_match, entropy = self._oracle.answer_with_detection_image(detected_crop)
                    try:
                        self._last_visual_entropy = entropy
                    except Exception:
                        pass
                    if is_match:
                        print(f"[VLMr1Bridge] Visual judge -> yes (entropy={entropy:.3f})")
                        return True
                    if entropy >= ENTROPY_MAYBE_THRESHOLD:
                        print(f"[VLMr1Bridge] Visual judge -> soft-maybe (entropy={entropy:.3f} >= {ENTROPY_MAYBE_THRESHOLD})")
                        return True
                    print(f"[VLMr1Bridge] Visual judge -> hard-no (entropy={entropy:.3f} < {ENTROPY_MAYBE_THRESHOLD})")
                    return False
                except Exception as e:
                    print(f"[VLMr1Bridge] Visual judge fallback to text: {e}")

            # Textual fallback -- counts as a question to the user
            try:
                self.pipeline._num_questions_asked += 1
            except Exception:
                pass
            question = (
                f"I am looking for: {target_desc}\n"
                f"I detected: {obj_desc}\n"
                f"Is this the object I am looking for? Answer only yes or no."
            )
            answer = self._oracle.answer(question)
            return answer.strip().lower().startswith("yes")

        return judge

    def set_oracle(self, oracle: Any) -> None:
        """Connect VLMr1Oracle after policy construction; wires VLM judge into KG trigger."""
        self._oracle = oracle
        self.pipeline.set_vlm_judge(self._make_vlm_judge())

    def new_episode(self, target_category: str) -> None:
        self._target_category = str(target_category).split("|")[0].strip().lower()
        self.pipeline.reset_episode(self._target_category)

    def detect(self, rgb: np.ndarray, timestep: int = 0) -> BridgeDetectionResult:
        res = self.detector.detect_from_observation(rgb, [self._target_category], kg_context=None)
        self._cached_det_result = res
        self._cached_det_frame_id = id(rgb)
        dets: list[BridgeDetection] = []
        for d in res.detections:
            dets.append(
                BridgeDetection(
                    bbox_xyxy=list(map(float, d.bbox)),
                    label=str(d.label),
                    reasoning=d.reasoning,
                    image=getattr(d, "image", None),
                )
            )
        return BridgeDetectionResult(detections=dets, reasoning_text=res.reasoning_text)

    def pipeline_step(self, rgb: np.ndarray, timestep: int) -> Any:
        if self._cached_det_result is not None and self._cached_det_frame_id == id(rgb):
            step = self.pipeline.on_detection_with_result(
                self._cached_det_result, rgb, timestep=timestep
            )
            self._cached_det_result = None
        else:
            step = self.pipeline.on_detection(rgb, timestep=timestep)
        return step

