"""Tests for CoIN-Bench env + runner (mocked VLM / loader)."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from PIL import Image

from aiuta_vlmr1.evaluation.coin_loader import CoINEpisode
from aiuta_vlmr1.evaluation.coin_metrics import EpisodeResult, compute_all_metrics, compute_sr
from aiuta_vlmr1.pipeline.coin_bench_env import CoINBenchEnv, coin_vqa_question, target_facts_from_coin_episode
from aiuta_vlmr1.pipeline.coin_bench_runner import build_arg_parser, run_coin_bench_evaluation
from aiuta_vlmr1.pipeline.entropy_coin_agent import EntropyEpisodeResult


def _minimal_episode(**kwargs) -> CoINEpisode:
    base = dict(
        episode_id="ep1",
        scene_id="sceneA",
        target_category="chair",
        target_description="wooden chair",
        start_position=[0.0, 0.0, 0.0],
        start_rotation=[1.0, 0.0, 0.0, 0.0],
        target_position=[1.0, 0.0, 1.0],
        split="val_seen",
    )
    base.update(kwargs)
    return CoINEpisode(**base)


def test_coin_bench_env_navigation_and_commit(tmp_path: Path):
    img1 = tmp_path / "a.png"
    img2 = tmp_path / "b.png"
    Image.new("RGB", (8, 8), color=(255, 0, 0)).save(img1)
    Image.new("RGB", (8, 8), color=(0, 255, 0)).save(img2)

    ep = _minimal_episode()
    tf = target_facts_from_coin_episode(ep)
    env = CoINBenchEnv(ep, None, [img1, img2], tf)

    assert env.get_observation().getpixel((0, 0))[:3] == (255, 0, 0)
    env.step("turn_left")
    assert env._idx == 0
    env.step("move_forward")
    assert env._idx == 1
    assert env.get_observation().getpixel((0, 0))[:3] == (0, 255, 0)
    assert env.exploration_steps == 2
    assert env.path_length == 3.0
    assert env.shortest_path_length == 1.0

    assert env.evaluate_commit("<answer>Yes</answer>") is True
    assert env.evaluate_commit("<answer>No</answer>") is False


def test_coin_vqa_question_contains_category():
    q = coin_vqa_question(_minimal_episode(target_category="table"))
    assert "table" in q
    assert "Is this the" in q


def test_cli_build_parser_minimal():
    p = build_arg_parser()
    args = p.parse_args(
        [
            "--coin-bench-path",
            "/data/coin",
            "--mode",
            "entropy",
            "--tau",
            "0.2",
            "--split",
            "val_seen",
            "--output",
            "/tmp/out.json",
            "--limit",
            "5",
        ]
    )
    assert args.coin_bench_path == "/data/coin"
    assert args.mode == "entropy"
    assert abs(args.tau - 0.2) < 1e-6
    assert args.limit == 5


def test_metrics_from_episode_results():
    results = [
        EpisodeResult(
            episode_id="a",
            split="val_seen",
            target_category="chair",
            success=True,
            path_length=2.0,
            shortest_path_length=1.0,
            num_questions=0,
        ),
        EpisodeResult(
            episode_id="b",
            split="val_seen",
            target_category="table",
            success=False,
            path_length=5.0,
            shortest_path_length=1.0,
            num_questions=2,
        ),
    ]
    assert abs(compute_sr(results) - 0.5) < 1e-6
    m = compute_all_metrics(results)
    assert m["num_episodes"] == 2
    assert m["SR"] == 50.0
    assert m["NQ"] == 1.0


def test_run_coin_bench_evaluation_entropy_mocked(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    from aiuta_vlmr1.config import Config
    from aiuta_vlmr1.pipeline import coin_bench_runner as m

    img = tmp_path / "x.png"
    Image.new("RGB", (4, 4), color="blue").save(img)

    ep = _minimal_episode()

    def _fake_load(self, split: str):
        assert split == "val_seen"
        return [ep]

    def _fake_cands(self, split: str, episode: CoINEpisode):
        return [img]

    monkeypatch.setattr(m.CoINBenchLoader, "load_episodes", _fake_load)
    monkeypatch.setattr(m.CoINBenchLoader, "get_episode_image_candidates", _fake_cands)

    def _fake_entropy(*_a, **_k):
        return EntropyEpisodeResult(
            committed=True,
            final_answer="Yes",
            success=True,
            steps=1,
            final_entropy=0.05,
            entropy_trajectory=[0.1],
            step_logs=[],
        )

    monkeypatch.setattr(m, "run_entropy_coin_episode", _fake_entropy)

    fake_ml = SimpleNamespace(model=object(), processor=object())
    monkeypatch.setattr(m.ModelLoader, "get_instance", lambda _cfg: fake_ml)

    out = tmp_path / "eval.json"
    payload = run_coin_bench_evaluation(
        Config(),
        str(tmp_path),
        mode="entropy",
        split="val_seen",
        limit=10,
        tau=0.15,
        max_steps_per_episode=3,
        seed=0,
        output_path=str(out),
    )

    assert payload["skipped_no_images"] == 0
    assert payload["num_episodes_run"] == 1
    assert payload["metrics"]["num_episodes"] == 1
    assert len(payload["per_episode"]) == 1
    row = payload["per_episode"][0]
    assert row["success"] is True
    assert row["num_questions"] == 0
    assert row["path_length"] == 1.0
    assert out.is_file()


def test_run_skips_when_no_candidates(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    from aiuta_vlmr1.config import Config
    from aiuta_vlmr1.pipeline import coin_bench_runner as m

    monkeypatch.setattr(
        m.CoINBenchLoader,
        "load_episodes",
        lambda self, split: [_minimal_episode()],
    )
    monkeypatch.setattr(
        m.CoINBenchLoader,
        "get_episode_image_candidates",
        lambda self, split, episode: [],
    )

    fake_ml = SimpleNamespace(model=object(), processor=object())
    monkeypatch.setattr(m.ModelLoader, "get_instance", lambda _cfg: fake_ml)
    monkeypatch.setattr(m, "run_entropy_coin_episode", MagicMock())

    payload = run_coin_bench_evaluation(
        Config(),
        str(tmp_path),
        mode="entropy",
        output_path=str(tmp_path / "o.json"),
    )
    assert payload["skipped_no_images"] == 1
    assert payload["num_episodes_run"] == 0
    assert payload["metrics"]["num_episodes"] == 0
