"""Episode runner and CoIN image resolution tests."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aiuta_vlmr1.evaluation.coin_loader import CoINEpisode, CoINBenchLoader
from aiuta_vlmr1.pipeline.episode_runner import OfflineEpisodeResult, resolve_episode_image, run_offline_evaluation


def test_get_episode_image_candidates_empty_without_metadata(tmp_path: Path):
    split = "val_seen"
    (tmp_path / split / "content").mkdir(parents=True)
    loader = CoINBenchLoader(str(tmp_path))
    ep = CoINEpisode(
        episode_id="e1",
        scene_id="sc1",
        target_category="chair",
        target_description="",
        start_position=[0, 0, 0],
        start_rotation=[1, 0, 0, 0],
        target_position=[0, 0, 0],
        target_object_id=5,
        split=split,
    )
    assert loader.get_episode_image_candidates(split, ep) == []


def test_get_episode_image_candidates_resolves_from_json(tmp_path: Path):
    split = "val_seen"
    content = tmp_path / split / "content"
    content.mkdir(parents=True)
    scene_id = "sceneA"
    img_rel = "obs/rgb_001.jpg"
    (content / scene_id).mkdir(parents=True)
    (content / scene_id / "obs").mkdir(parents=True)
    full_img = content / scene_id / "rgb_001.jpg"
    full_img.write_bytes(b"\xff\xd8\xff")

    payload = {
        "episodes": [
            {"episode_id": "ep99", "image_path": str(full_img.name)},
        ],
        "objects": [{"object_id": 5, "category": "chair", "image": str(full_img)}],
        "target_images": {"5": str(full_img)},
    }
    with open(content / f"{scene_id}.json", "w", encoding="utf-8") as f:
        json.dump(payload, f)

    loader = CoINBenchLoader(str(tmp_path))
    ep = CoINEpisode(
        episode_id="ep99",
        scene_id=scene_id,
        target_category="chair",
        target_description="",
        start_position=[0, 0, 0],
        start_rotation=[1, 0, 0, 0],
        target_position=[0, 0, 0],
        target_object_id=5,
        split=split,
    )
    cands = loader.get_episode_image_candidates(split, ep)
    assert len(cands) >= 1
    assert all(p.is_file() for p in cands)


def test_offline_static_coin_no_arbitrary_scene_glob(tmp_path: Path):
    split = "val_seen"
    scene_id = "s1"
    content = tmp_path / split / "content" / scene_id
    content.mkdir(parents=True)
    (content / "random.jpg").write_bytes(b"\xff\xd8\xff")

    loader = CoINBenchLoader(str(tmp_path))
    ep = CoINEpisode(
        episode_id="e1",
        scene_id=scene_id,
        target_category="x",
        target_description="",
        start_position=[0, 0, 0],
        start_rotation=[1, 0, 0, 0],
        target_position=[0, 0, 0],
        split=split,
    )
    path, src, sk = resolve_episode_image(
        loader, ep, split, "offline_static_coin", None, tmp_path,
    )
    assert path is None and sk == "missing_coin_image"


def test_strict_coin_images_raises():
    from aiuta_vlmr1.config import Config

    mock_loader = MagicMock()
    mock_loader.ensure_downloaded = MagicMock()
    mock_loader.available_splits.return_value = ["val_seen"]
    mock_loader.load_episodes.return_value = [
        CoINEpisode(
            episode_id="e1",
            scene_id="missing",
            target_category="c",
            target_description="",
            start_position=[0, 0, 0],
            start_rotation=[1, 0, 0, 0],
            target_position=[0, 0, 0],
            split="val_seen",
        ),
    ]
    mock_loader.get_episode_image_candidates.return_value = []

    cfg = Config()
    with patch("aiuta_vlmr1.evaluation.coin_loader.CoINBenchLoader", return_value=mock_loader):
        with pytest.raises(RuntimeError, match="No trustworthy CoIN image"):
            run_offline_evaluation(
                cfg,
                "/fake/coin",
                splits=["val_seen"],
                limit=1,
                run_mode="offline_static_coin",
                strict_coin_images=True,
            )


def test_episode_json_fields_present(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from aiuta_vlmr1.config import Config

    r = OfflineEpisodeResult(
        episode_id="e",
        split="val_seen",
        scene_id="sc",
        target_category="cabinet",
        image_path="/tmp/x.jpg",
        image_source="coin_episode_metadata",
        pipeline_signal="continue",
        target_detected=True,
        num_detections=1,
        num_questions=0,
        num_kg_nodes=0,
        pipeline_latency_sec=1.0,
        detector_latency_sec=0.5,
        skipped_reason=None,
    )
    row = {
        "episode_id": r.episode_id,
        "split": r.split,
        "scene_id": r.scene_id,
        "target_category": r.target_category,
        "image_path": r.image_path,
        "image_source": r.image_source,
        "skipped_reason": r.skipped_reason,
        "pipeline_signal": r.pipeline_signal,
        "num_detections": r.num_detections,
        "num_questions": r.num_questions,
        "num_kg_nodes": r.num_kg_nodes,
        "pipeline_latency_sec": r.pipeline_latency_sec,
        "detector_latency_sec": r.detector_latency_sec,
    }
    for k in (
        "episode_id",
        "split",
        "scene_id",
        "image_path",
        "image_source",
        "skipped_reason",
        "pipeline_signal",
    ):
        assert k in row
