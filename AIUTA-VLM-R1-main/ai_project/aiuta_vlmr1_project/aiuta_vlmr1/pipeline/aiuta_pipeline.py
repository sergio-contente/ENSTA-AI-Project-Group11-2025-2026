"""
aiuta_pipeline.py -- Main AIUTA orchestration loop.
Strategy pattern selects components from config.

Supports:
  - Embodied / detection-driven episodes (CoIN offline static integration tests)
  - Offline QA-style use is provided via ``evaluation.idkvqa_eval`` (primary benchmark)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable

from ..config import Config, DetectorType, QuestionerType, TriggerType
from ..detector.base import AbstractDetector
from ..detector.vlmr1_detector import VLMr1Detector
from ..self_questioner.base import AbstractSelfQuestioner
from ..self_questioner.vlmr1_questioner import VLMr1SelfQuestioner
from ..self_questioner.two_pass_questioner import TwoPassSelfQuestioner
from ..interaction_trigger.base import AbstractInteractionTrigger, ActionType
from ..interaction_trigger.kg_trigger import KGInteractionTrigger
from ..knowledge_graph.scene_graph import SceneKnowledgeGraph


class PolicySignal(Enum):
    STOP = "stop"
    CONTINUE = "continue"


@dataclass
class PipelineStepResult:
    """Result of one ``on_detection`` call (introspection for offline runners)."""
    signal: PolicySignal
    num_raw_detections: int
    num_valid_detections: int
    detector_latency_sec: float | None
    detector_preprocess_sec: float | None
    detector_generate_sec: float | None
    detector_parse_sec: float | None
    asked_questions_in_step: int


class AIUTAPipeline:
    def __init__(
        self,
        config: Config,
        ask_human: Callable[[str], str] | None = None,
        vlm_judge_fn: Callable[[str, str], bool] | None = None,
    ):
        self._config = config
        self._ask_human = ask_human or (lambda q: "I don't know")
        self._vlm_judge_fn = vlm_judge_fn

        self._detector = self._create_detector(config)
        self._questioner = self._create_questioner(config)
        self._trigger = self._create_trigger(config)

        self._kg = SceneKnowledgeGraph()
        self._target_category = ""
        self._timestep = 0
        self._num_questions_asked = 0
        self._num_visual_comparisons = 0
        self._episode_log: list[dict] = []
        self._last_step_result: PipelineStepResult | None = None
        self._current_observation: Any = None

    def _ask_about_detection(self, question: str) -> str | None:
        """VQA on the current FPV / detection frame (same ModelLoader as detector)."""
        obs = getattr(self, "_current_observation", None)
        if obs is None or not (question or "").strip():
            return None
        tmp_path: str | None = None
        try:
            import os
            import tempfile

            import numpy as np
            import torch
            from PIL import Image
            from qwen_vl_utils import process_vision_info

            from ..utils.model_loader import ModelLoader

            if not ModelLoader._instances:
                return None
            loader = ModelLoader._instances[next(iter(ModelLoader._instances))]

            if isinstance(obs, str):
                img_url = f"file://{os.path.abspath(obs)}"
            elif isinstance(obs, np.ndarray):
                pil = Image.fromarray(obs.astype(np.uint8))
                fd, tmp_path = tempfile.mkstemp(suffix=".jpg")
                os.close(fd)
                pil.save(tmp_path, format="JPEG", quality=95)
                img_url = f"file://{os.path.abspath(tmp_path)}"
            else:
                return None

            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant. "
                        "Answer the question briefly in 1-3 words based on what you see in the image."
                    ),
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

            proc = loader.processor
            text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            img_in, vid_in = process_vision_info(messages)
            inputs = proc(
                text=[text],
                images=img_in,
                videos=vid_in,
                padding=True,
                return_tensors="pt",
            ).to(loader.device)

            with torch.inference_mode():
                gen = loader.model.generate(
                    **inputs,
                    max_new_tokens=32,
                    do_sample=False,
                    use_cache=False,
                )

            trimmed = [o[len(i) :] for i, o in zip(inputs.input_ids, gen)]
            raw = proc.batch_decode(
                trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            answer = raw.strip().rstrip(".")
            print(f"[AIUTAPipeline] Detection VQA: Q={question!r} -> A={answer!r}")
            low = answer.lower()
            if not answer or low in ("i don't know", "i dont know", "unknown"):
                return None
            return answer
        except Exception as e:
            print(f"[AIUTAPipeline] Detection VQA error: {e}")
            return None
        finally:
            if tmp_path:
                try:
                    import os

                    os.unlink(tmp_path)
                except OSError:
                    pass

    def _create_detector(self, config: Config) -> AbstractDetector:
        if config.detector_type == DetectorType.VLMR1:
            return VLMr1Detector(config)
        raise NotImplementedError(f"Detector {config.detector_type} not implemented")

    def _create_questioner(self, config: Config) -> AbstractSelfQuestioner:
        if config.questioner_type == QuestionerType.VLMR1:
            return VLMr1SelfQuestioner()
        if config.questioner_type == QuestionerType.TWO_PASS:
            return TwoPassSelfQuestioner(config)
        raise NotImplementedError(f"Questioner {config.questioner_type} not implemented")

    def _create_trigger(self, config: Config) -> AbstractInteractionTrigger:
        if config.trigger_type == TriggerType.KG:
            loader = None
            try:
                from ..utils.model_loader import ModelLoader

                if ModelLoader._instances:
                    key = next(iter(ModelLoader._instances))
                    loader = ModelLoader._instances[key]
            except Exception:
                pass
            return KGInteractionTrigger(
                config.trigger,
                vlm_judge_fn=self._vlm_judge_fn,
                loader=loader,
            )
        raise NotImplementedError(f"Trigger {config.trigger_type} not implemented")

    def set_ask_human(self, fn: Callable[[str], str]) -> None:
        """Replace the human callback (used by offline runners and tests)."""
        self._ask_human = fn

    def set_vlm_judge(self, fn: Callable[[str, str], bool] | None) -> None:
        """Optional judge (obj_desc, target_desc) -> bool for inconclusive alignment scores."""
        self._vlm_judge_fn = fn
        if isinstance(self._trigger, KGInteractionTrigger):
            self._trigger._vlm_judge_fn = fn
            if self._trigger._loader is None:
                try:
                    from ..utils.model_loader import ModelLoader

                    if ModelLoader._instances:
                        key = next(iter(ModelLoader._instances))
                        self._trigger._loader = ModelLoader._instances[key]
                except Exception:
                    pass

    def new_episode(self, target_category: str) -> None:
        self.reset_episode(target_category)

    def reset_episode(self, target_category: str) -> None:
        """Clear per-episode graph state and counters."""
        from ..knowledge_graph.graph_matcher import clear_embedding_cache

        clear_embedding_cache()
        self._kg.reset()
        self._kg.target_facts.category = target_category
        self._target_category = target_category
        self._timestep = 0
        self._num_questions_asked = 0
        self._num_visual_comparisons = 0
        self._episode_log = []
        self._last_step_result = None
        self._current_observation = None

    def on_detection(self, observation, timestep: int) -> PipelineStepResult:
        self._timestep = timestep
        self._current_observation = observation
        kg_context = self._kg.get_kg_context_string(self._target_category)

        if isinstance(observation, str):
            det_result = self._detector.detect(
                observation, [self._target_category], kg_context=kg_context
            )
        else:
            det_result = self._detector.detect_from_observation(
                observation, [self._target_category], kg_context=kg_context
            )

        return self._process_detections(det_result, observation, timestep)

    def on_detection_with_result(
        self, det_result: Any, observation: Any, timestep: int
    ) -> PipelineStepResult:
        """Same as on_detection but reuses a pre-computed DetectionResult."""
        self._timestep = timestep
        self._current_observation = observation
        return self._process_detections(det_result, observation, timestep)

    def _process_detections(
        self, det_result: Any, observation: Any, timestep: int
    ) -> PipelineStepResult:
        asked_here = 0
        final_signal = PolicySignal.CONTINUE
        raw_n = len(det_result.detections)
        valid_n = 0

        max_q = getattr(self._config.trigger, "max_questions_per_episode", 6)
        if self._num_questions_asked >= max_q and raw_n > 0:
            print(
                f"[AIUTAPipeline] Question budget exhausted "
                f"({self._num_questions_asked}/{max_q}) -- forcing STOP"
            )
            final_signal = PolicySignal.STOP
            self._last_step_result = PipelineStepResult(
                signal=final_signal,
                num_raw_detections=raw_n,
                num_valid_detections=0,
                detector_latency_sec=det_result.latency_sec,
                detector_preprocess_sec=det_result.preprocess_latency_sec,
                detector_generate_sec=det_result.generate_latency_sec,
                detector_parse_sec=det_result.parse_latency_sec,
                asked_questions_in_step=0,
            )
            return self._last_step_result

        self._questioner._current_observation = observation  # type: ignore[attr-defined]

        for det in det_result.detections:
            refined = self._questioner.process(
                det, self._kg.target_facts, self._kg, timestep
            )

            if not refined.is_valid:
                continue
            valid_n += 1

            for round_idx in range(self._config.trigger.max_interaction_rounds):
                action = self._trigger.decide(
                    refined, self._kg.target_facts, self._kg
                )

                log_entry: dict[str, Any] = {
                    "timestep": timestep,
                    "detection": det.label,
                    "action": action.type.value,
                    "score": action.alignment_score,
                    "question": action.question,
                    "reason": action.reason,
                    "round": round_idx,
                    "alignment_explanation": action.alignment_explanation,
                }

                if action.type == ActionType.STOP:
                    self._episode_log.append(log_entry)
                    final_signal = PolicySignal.STOP
                    self._last_step_result = PipelineStepResult(
                        signal=final_signal,
                        num_raw_detections=raw_n,
                        num_valid_detections=valid_n,
                        detector_latency_sec=det_result.latency_sec,
                        detector_preprocess_sec=det_result.preprocess_latency_sec,
                        detector_generate_sec=det_result.generate_latency_sec,
                        detector_parse_sec=det_result.parse_latency_sec,
                        asked_questions_in_step=asked_here,
                    )
                    return self._last_step_result

                if action.type == ActionType.ASK:
                    response = self._ask_human(action.question or "")
                    self._kg.update_target_facts(response, timestep, question=action.question)
                    is_idk = response.strip().lower().rstrip(".") in (
                        "i don't know",
                        "i dont know",
                        "unknown",
                        "not sure",
                    )
                    # Always count -- fair comparison with AIUTA (counts all Oracle queries)
                    self._num_questions_asked += 1
                    asked_here += 1
                    if is_idk:
                        print("[AIUTAPipeline] IDK response (still counted toward NQ)")
                    log_entry["user_response"] = response

                    oracle_no = response.strip().lower().startswith("no")

                    attr_name = SceneKnowledgeGraph._infer_attribute_from_question(
                        action.question or ""
                    )
                    is_think_q = attr_name is not None and attr_name.startswith("think_")

                    if not is_think_q:
                        det_answer = self._ask_about_detection(action.question or "")
                        log_entry["detection_answer"] = det_answer
                        on = refined.object_node
                        if det_answer is not None and on is not None:
                            oid = getattr(on, "obj_id", None)
                            if attr_name is not None and oid and self._kg.get_object(oid) is not None:
                                from ..knowledge_graph.schema import (
                                    Attribute,
                                    AttributeSource,
                                    Certainty,
                                )

                                det_low = det_answer.strip().lower().rstrip(".")
                                parsed_yesno = SceneKnowledgeGraph._parse_yesno_question(
                                    action.question or ""
                                )

                                if parsed_yesno is not None:
                                    yn_attr, yn_val = parsed_yesno
                                    if det_low.startswith("yes"):
                                        attr = Attribute(
                                            name=yn_attr,
                                            value=yn_val,
                                            certainty=Certainty.MEDIUM,
                                            source=AttributeSource.VLM_REASONING,
                                            timestep=timestep,
                                        )
                                        self._kg.update_attributes(oid, [attr])
                                        print(
                                            f"[AIUTAPipeline] Obj {oid!r} <- {yn_attr}={yn_val} "
                                            f"(detection VQA confirmed)"
                                        )
                                    elif det_low.startswith("no"):
                                        print(
                                            f"[AIUTAPipeline] Obj {oid!r}: detection VQA denied "
                                            f"{yn_attr}={yn_val} -- not storing"
                                        )
                                    else:
                                        val = SceneKnowledgeGraph._normalize_open_answer_value(
                                            det_answer
                                        )
                                        if val and val not in ("i don't know", "unknown"):
                                            attr = Attribute(
                                                name=yn_attr,
                                                value=val,
                                                certainty=Certainty.LOW,
                                                source=AttributeSource.VLM_REASONING,
                                                timestep=timestep,
                                            )
                                            self._kg.update_attributes(oid, [attr])
                                            print(
                                                f"[AIUTAPipeline] Obj {oid!r} <- {yn_attr}={val} "
                                                f"(detection VQA open answer)"
                                            )
                                else:
                                    val = SceneKnowledgeGraph._normalize_open_answer_value(
                                        det_answer
                                    )
                                    if val and val not in ("yes", "no", "i don't know", "unknown"):
                                        attr = Attribute(
                                            name=attr_name,
                                            value=val,
                                            certainty=Certainty.MEDIUM,
                                            source=AttributeSource.VLM_REASONING,
                                            timestep=timestep,
                                        )
                                        self._kg.update_attributes(oid, [attr])
                                        print(
                                            f"[AIUTAPipeline] Obj {oid!r} <- {attr_name}={val} "
                                            f"(from detection VQA)"
                                        )
                                    elif val in ("yes", "no"):
                                        print(
                                            f"[AIUTAPipeline] Obj {oid!r}: detection VQA "
                                            f"returned '{val}' for open question -- skipping"
                                        )

                        if oracle_no and det_answer and det_answer.strip().lower().startswith("yes"):
                            print(
                                f"[AIUTAPipeline] Oracle='no' vs Detection='yes' "
                                f"for: {action.question!r} -- definitive mismatch"
                            )
                            log_entry["target_facts_snapshot"] = {
                                "known": dict(self._kg.target_facts.known_attributes),
                                "negative": dict(self._kg.target_facts.negative_attributes),
                            }
                            self._episode_log.append(log_entry)
                            break
                    else:
                        log_entry["detection_answer"] = "(think feature -- obj already has attribute)"

                    log_entry["target_facts_snapshot"] = {
                        "known": dict(self._kg.target_facts.known_attributes),
                        "negative": dict(self._kg.target_facts.negative_attributes),
                    }
                    self._episode_log.append(log_entry)
                else:
                    self._episode_log.append(log_entry)
                    break

        self._last_step_result = PipelineStepResult(
            signal=final_signal,
            num_raw_detections=raw_n,
            num_valid_detections=valid_n,
            detector_latency_sec=det_result.latency_sec,
            detector_preprocess_sec=det_result.preprocess_latency_sec,
            detector_generate_sec=det_result.generate_latency_sec,
            detector_parse_sec=det_result.parse_latency_sec,
            asked_questions_in_step=asked_here,
        )
        return self._last_step_result

    def get_episode_summary(self) -> dict[str, Any]:
        """Structured snapshot for logging (no private field access)."""
        tf = self._kg.target_facts
        out: dict[str, Any] = {
            "target_category": self._target_category,
            "timestep": self._timestep,
            "num_questions_asked": self._num_questions_asked,
            "num_visual_comparisons": getattr(self, "_num_visual_comparisons", 0),
            "num_kg_objects": self._kg.num_objects,
            "target_facts": {
                "category": tf.category,
                "num_facts": tf.num_facts,
                "asked_questions_count": len(tf.asked_questions),
                "fact_provenance": dict(tf.fact_provenance),
            },
            "episode_log_len": len(self._episode_log),
            "last_step": None,
        }
        if self._last_step_result is not None:
            ls = self._last_step_result
            out["last_step"] = {
                "signal": ls.signal.value,
                "num_raw_detections": ls.num_raw_detections,
                "num_valid_detections": ls.num_valid_detections,
                "detector_latency_sec": ls.detector_latency_sec,
                "detector_preprocess_sec": ls.detector_preprocess_sec,
                "detector_generate_sec": ls.detector_generate_sec,
                "detector_parse_sec": ls.detector_parse_sec,
                "asked_questions_in_step": ls.asked_questions_in_step,
            }
        return out

    @property
    def kg(self) -> SceneKnowledgeGraph:
        return self._kg

    @property
    def num_questions_asked(self) -> int:
        return self._num_questions_asked

    @property
    def episode_log(self) -> list[dict]:
        return list(self._episode_log)
