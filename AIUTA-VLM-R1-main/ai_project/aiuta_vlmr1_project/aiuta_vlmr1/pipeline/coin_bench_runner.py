"""
coin_bench_runner -- CoIN-Bench offline evaluation with SR / SPL / NQ (static proxy).

Modes:
  - ``entropy``: ``run_entropy_coin_episode`` + ``CoINBenchEnv``
  - ``aiuta_pipeline``: full ``AIUTAPipeline`` with simulated user
  - ``raw``: single VLM VQA call (no entropy gating)

SPL/SR here use ``shortest_path_length=1`` and discrete step counts -- not Habitat geodesics.
"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ..config import Config
from ..evaluation.coin_loader import CoINBenchLoader, CoINEpisode
from ..evaluation.coin_metrics import (
    EpisodeResult,
    compute_all_metrics,
    compute_metrics_by_split,
)
from ..utils.model_loader import ModelLoader
from .aiuta_pipeline import AIUTAPipeline, PolicySignal
from .coin_bench_env import CoINBenchEnv, coin_vqa_question, target_facts_from_coin_episode
from .entropy_coin_agent import run_entropy_coin_episode, vlm_vqa_with_entropy
from .episode_runner import make_simulated_user


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="CoIN-Bench offline evaluation (entropy / AIUTA / raw).")
    p.add_argument("--coin-bench-path", type=str, required=True, help="Local CoIN-Bench root directory")
    p.add_argument(
        "--mode",
        type=str,
        default="entropy",
        choices=("entropy", "aiuta_pipeline", "raw"),
    )
    p.add_argument("--split", type=str, default="val_seen")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--tau", type=float, default=0.15)
    p.add_argument("--max-steps-per-episode", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=str, default="results/coin_bench/eval.json")
    p.add_argument("--config", type=str, default=None, help="Optional YAML config for AIUTA / model IDs")
    return p


def _load_config(config_path: str | None) -> Config:
    if config_path:
        return Config.from_yaml(config_path)
    return Config()


def _episode_result_to_json_dict(r: EpisodeResult) -> dict[str, Any]:
    d: dict[str, Any] = {
        "episode_id": r.episode_id,
        "split": r.split,
        "target_category": r.target_category,
        "success": r.success,
        "path_length": r.path_length,
        "shortest_path_length": r.shortest_path_length,
        "num_questions": r.num_questions,
        "num_detections": r.num_detections,
        "num_kg_nodes": r.num_kg_nodes,
        "final_distance_to_target": r.final_distance_to_target,
        "total_timesteps": r.total_timesteps,
        "total_inference_time": r.total_inference_time,
        "episode_log": list(r.episode_log),
    }
    return d


def _run_entropy_episode(
    env: CoINBenchEnv,
    question: str,
    model: Any,
    processor: Any,
    *,
    tau: float,
    max_steps: int,
    rng: random.Random,
) -> EpisodeResult:
    ep_res = run_entropy_coin_episode(
        env,
        question,
        model,
        processor,
        tau=tau,
        max_steps=max_steps,
        rng=rng,
    )
    success = bool(ep_res.success) if ep_res.success is not None else False
    step_logs = [asdict(x) for x in ep_res.step_logs]
    return EpisodeResult(
        episode_id=env.episode_id,
        split=env.episode.split or "",
        target_category=env.target_category,
        success=success,
        path_length=env.path_length,
        shortest_path_length=env.shortest_path_length,
        num_questions=0,
        num_detections=int(ep_res.steps),
        num_kg_nodes=0,
        total_timesteps=int(ep_res.steps),
        episode_log=step_logs,
    )


def _run_raw_episode(
    env: CoINBenchEnv,
    question: str,
    model: Any,
    processor: Any,
    *,
    max_new_tokens: int,
) -> EpisodeResult:
    obs = env.get_observation()
    answer, _entropy, _logits = vlm_vqa_with_entropy(
        obs,
        question,
        model,
        processor,
        history=None,
        max_new_tokens=max_new_tokens,
    )
    success = env.evaluate_commit(answer)
    return EpisodeResult(
        episode_id=env.episode_id,
        split=env.episode.split or "",
        target_category=env.target_category,
        success=success,
        path_length=env.path_length,
        shortest_path_length=env.shortest_path_length,
        num_questions=0,
        num_detections=1,
        num_kg_nodes=0,
        total_timesteps=1,
        episode_log=[{"mode": "raw", "answer": answer}],
    )


def _run_aiuta_episode(
    env: CoINBenchEnv,
    pipeline: AIUTAPipeline,
    *,
    max_steps: int,
) -> EpisodeResult:
    pipeline.reset_episode(env.target_category)
    tf = target_facts_from_coin_episode(env.episode)
    pipeline.kg.target_facts = tf

    last_signal = PolicySignal.CONTINUE
    total_raw_dets = 0
    log: list[dict[str, Any]] = []

    for t in range(max_steps):
        path_str = str(env.current_image_path)
        step_res = pipeline.on_detection(path_str, timestep=t)
        last_signal = step_res.signal
        total_raw_dets += int(step_res.num_raw_detections)
        log.append(
            {
                "timestep": t,
                "signal": step_res.signal.value,
                "num_raw_detections": step_res.num_raw_detections,
                "num_valid_detections": step_res.num_valid_detections,
            }
        )
        if step_res.signal == PolicySignal.STOP:
            break
        env.step("move_forward")

    success = last_signal == PolicySignal.STOP
    return EpisodeResult(
        episode_id=env.episode_id,
        split=env.episode.split or "",
        target_category=env.target_category,
        success=success,
        path_length=env.path_length,
        shortest_path_length=env.shortest_path_length,
        num_questions=pipeline.num_questions_asked,
        num_detections=total_raw_dets,
        num_kg_nodes=pipeline.kg.num_objects,
        total_timesteps=len(log),
        episode_log=log,
    )


def run_coin_bench_evaluation(
    config: Config,
    coin_bench_path: str,
    mode: str = "entropy",
    split: str = "val_seen",
    limit: int | None = None,
    tau: float = 0.15,
    max_steps_per_episode: int = 20,
    seed: int = 42,
    output_path: str = "results/coin_bench/eval.json",
) -> dict[str, Any]:
    random.seed(seed)
    rng = random.Random(seed)

    loader = CoINBenchLoader(coin_bench_path)
    episodes = loader.load_episodes(split)
    if limit is not None:
        episodes = episodes[: max(0, limit)]

    results: list[EpisodeResult] = []
    skipped_no_images = 0

    ml: ModelLoader | None = None
    model = processor = None
    pipeline: AIUTAPipeline | None = None

    if mode in ("entropy", "raw"):
        ml = ModelLoader.get_instance(config.model)
        model, processor = ml.model, ml.processor
    elif mode == "aiuta_pipeline":
        ask_dummy = lambda _q: "I don't know"
        pipeline = AIUTAPipeline(config, ask_human=ask_dummy)

    for ep in episodes:
        cands = loader.get_episode_image_candidates(split, ep)
        if not cands:
            skipped_no_images += 1
            continue

        tf = target_facts_from_coin_episode(ep)
        env = CoINBenchEnv(ep, loader, cands, tf)
        question = coin_vqa_question(ep)

        if mode == "entropy":
            assert model is not None and processor is not None
            er = _run_entropy_episode(
                env,
                question,
                model,
                processor,
                tau=tau,
                max_steps=max_steps_per_episode,
                rng=rng,
            )
        elif mode == "raw":
            assert model is not None and processor is not None
            er = _run_raw_episode(
                env,
                question,
                model,
                processor,
                max_new_tokens=config.model.max_new_tokens,
            )
        else:
            assert pipeline is not None
            user_fn = make_simulated_user(
                ep.target_description or "",
                ep.target_category or "",
                mode="description_based",
            )
            pipeline.set_ask_human(user_fn)
            er = _run_aiuta_episode(env, pipeline, max_steps=max_steps_per_episode)

        results.append(er)

    metrics = compute_all_metrics(results)
    by_split = compute_metrics_by_split(results)

    payload: dict[str, Any] = {
        "coin_bench_path": str(Path(coin_bench_path).resolve()),
        "mode": mode,
        "split": split,
        "tau": tau,
        "max_steps_per_episode": max_steps_per_episode,
        "seed": seed,
        "limit_requested": limit,
        "skipped_no_images": skipped_no_images,
        "num_episodes_run": len(results),
        "metrics": metrics,
        "metrics_by_split": by_split,
        "offline_static_note": (
            "SR/SPL/NQ are computed on a static offline proxy: shortest_path_length=1, "
            "path_length = exploration actions + 1 commit. Not comparable to online Habitat CoIN."
        ),
        "config": config.to_serializable_dict(),
        "per_episode": [_episode_result_to_json_dict(r) for r in results],
    }

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    payload["output_path"] = str(out_p.resolve())
    return payload


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    config = _load_config(args.config)
    run_coin_bench_evaluation(
        config=config,
        coin_bench_path=args.coin_bench_path,
        mode=args.mode,
        split=args.split,
        limit=args.limit,
        tau=args.tau,
        max_steps_per_episode=args.max_steps_per_episode,
        seed=args.seed,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
