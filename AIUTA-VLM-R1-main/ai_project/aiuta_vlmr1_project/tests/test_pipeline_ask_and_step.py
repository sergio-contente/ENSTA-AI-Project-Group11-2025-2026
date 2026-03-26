"""Pipeline ASK logging and PipelineStepResult."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from aiuta_vlmr1.config import Config, TriggerConfig, TriggerType
from aiuta_vlmr1.detector.base import Detection, DetectionResult
from aiuta_vlmr1.interaction_trigger.base import ActionType, TriggerAction
from aiuta_vlmr1.pipeline.aiuta_pipeline import AIUTAPipeline, PolicySignal
from aiuta_vlmr1.self_questioner.base import RefinedDescription


@patch("aiuta_vlmr1.pipeline.aiuta_pipeline.VLMr1Detector")
def test_ask_logs_user_response(mock_det: MagicMock) -> None:
    det = Detection(bbox=[0, 0, 10, 10], label="cabinet", reasoning="A wooden cabinet.")
    mock_det.return_value.detect.return_value = DetectionResult(
        detections=[det],
        raw_output="",
        reasoning_text="",
        json_valid=True,
        latency_sec=0.1,
        preprocess_latency_sec=0.01,
        generate_latency_sec=0.08,
        parse_latency_sec=0.01,
    )

    cfg = Config()
    cfg.trigger_type = TriggerType.KG
    cfg.trigger = TriggerConfig(tau_stop=0.99, tau_skip=0.01, max_interaction_rounds=2)

    calls = []

    def ask(q: str) -> str:
        calls.append(q)
        return "it is brown"

    with patch("aiuta_vlmr1.pipeline.aiuta_pipeline.VLMr1SelfQuestioner") as mock_q:
        node = MagicMock()
        node.obj_id = "cabinet_001"
        node.category = "cabinet"
        node.attributes = {}
        node.to_natural_language.return_value = "cabinet"
        mock_q.return_value.process.return_value = RefinedDescription(
            object_node=node,
            text_description="x",
            is_valid=True,
        )

        with patch("aiuta_vlmr1.pipeline.aiuta_pipeline.KGInteractionTrigger") as mock_tr:
            inst = mock_tr.return_value
            inst.decide.return_value = TriggerAction(
                type=ActionType.ASK,
                question="What color?",
                alignment_score=0.5,
                reason="ask",
                alignment_explanation={"score": 0.5, "matched": []},
            )

            p = AIUTAPipeline(cfg, ask_human=ask)
            p.reset_episode("cabinet")
            step = p.on_detection("/tmp/fake.jpg", timestep=0)

            assert step.signal == PolicySignal.CONTINUE
            assert step.detector_latency_sec == 0.1
            assert step.detector_preprocess_sec == 0.01
            log = p.episode_log
            assert len(log) >= 1
            assert log[0].get("user_response") == "it is brown"
            assert "target_facts_snapshot" in log[0]
