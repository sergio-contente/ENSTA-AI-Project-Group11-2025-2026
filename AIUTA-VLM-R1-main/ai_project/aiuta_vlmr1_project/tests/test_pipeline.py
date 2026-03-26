"""AIUTAPipeline public API tests."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from aiuta_vlmr1.config import Config
from aiuta_vlmr1.pipeline.aiuta_pipeline import AIUTAPipeline


@patch("aiuta_vlmr1.pipeline.aiuta_pipeline.VLMr1Detector")
def test_reset_and_question_count(mock_det: MagicMock) -> None:
    mock_det.return_value = MagicMock()
    cfg = Config()
    p = AIUTAPipeline(cfg, ask_human=lambda q: "I don't know")
    p.reset_episode("cabinet")
    assert p.num_questions_asked == 0
    s = p.get_episode_summary()
    assert s["target_category"] == "cabinet"
    assert s["num_questions_asked"] == 0


@patch("aiuta_vlmr1.pipeline.aiuta_pipeline.VLMr1Detector")
def test_set_ask_human(mock_det: MagicMock) -> None:
    mock_det.return_value = MagicMock()
    cfg = Config()
    p = AIUTAPipeline(cfg, ask_human=lambda q: "a")
    p.set_ask_human(lambda q: "b")
    assert p.get_episode_summary()["num_questions_asked"] == 0
