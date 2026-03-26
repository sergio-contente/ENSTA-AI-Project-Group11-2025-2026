"""
episode_runner.py -- Auxiliary **offline** CoIN-derived **integration / smoke tests** only.

This is **not** the primary research benchmark. Use **IDKVQA** (``evaluation.idkvqa_eval``) for
Yes/No/IDK calibration, abstention, and KG ablations.

What this runner does:
  Runs the embodied pipeline (detector -> questioner -> KG trigger) on **static** images
  when a **trustworthy** image path can be resolved for an episode.

What it does **not** do:
  - Official online CoIN / Habitat timestep navigation, SR, SPL, or success distance.
  - Arbitrary "first image in scene folder" substitution for ``offline_static_coin``.

Modes:
  - ``offline_static_coin``: only images returned by ``CoINBenchLoader.get_episode_image_candidates``
    (metadata-linked). Never uses a random scene file.
  - ``offline_proxy``: tries trustworthy CoIN candidates first; may use IDKVQA proxy images
    when CoIN paths are missing (explicit ``image_source``).
  - ``habitat``: not implemented (no fake support).
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from collections import Counter
from typing import Callable

import numpy as np
import torch


# =============================================================================
# Offline result (per episode JSON must match this schema)
# =============================================================================


@dataclass
class OfflineEpisodeResult:
    """One auxiliary offline run. Not comparable to online CoIN."""
    episode_id: str
    split: str
    scene_id: str
    target_category: str
    image_path: str | None
    image_source: str
    pipeline_signal: str
    target_detected: bool
    num_detections: int
    num_questions: int
    num_kg_nodes: int
    pipeline_latency_sec: float
    detector_latency_sec: float | None
    skipped_reason: str | None = None
    log_summary: list[dict] = field(default_factory=list)
    detector_preprocess_sec: float | None = None
    detector_generate_sec: float | None = None
    detector_parse_sec: float | None = None
    not_official_coin_online: bool = True

    # Legacy / extra fields for debugging (still serialized)
    target_description: str = ""
    num_target_detections: int = 0
    non_target_detection_labels: list[str] = field(default_factory=list)
    kg_attributes_extracted: int = 0
    total_latency_sec: float = 0.0


# =============================================================================
# Simulated user
# =============================================================================


def make_simulated_user(
    target_description: str,
    target_category: str,
    mode: str = "description_based",
) -> Callable[[str], str]:
    if mode == "always_idk":
        return lambda q: "I don't know"

    desc_lower = target_description.lower() if target_description else ""

    def user(question: str) -> str:
        q = question.lower()
        for color in [
            "red", "blue", "green", "black", "white", "yellow", "brown", "gray", "grey",
            "orange", "pink", "dark", "light",
        ]:
            if "color" in q and color in desc_lower:
                return f"Yes, it is {color}"
        for mat in ["wood", "metal", "glass", "leather", "fabric", "plastic"]:
            if "material" in q and mat in desc_lower:
                return f"It is made of {mat}"
        for size in ["large", "small", "big", "tiny"]:
            if ("size" in q or size in q) and size in desc_lower:
                return f"Yes, it is {size}"
        if target_category.lower() in q and desc_lower:
            return desc_lower[:100]
        return "I don't know"

    return user


# =============================================================================
# Image resolution (strict vs proxy)
# =============================================================================


def resolve_episode_image(
    loader,
    episode,
    split: str,
    run_mode: str,
    idkvqa_images: dict | None,
    coin_bench_dir: Path,
) -> tuple[str | None, str, str | None]:
    """
    Returns (image_path, image_source, skipped_reason).

    For ``offline_static_coin``, only ``get_episode_image_candidates`` paths are used
    (deterministic first of sorted candidates). No directory globbing.
    """
    if run_mode == "offline_static_coin":
        cands = loader.get_episode_image_candidates(split, episode)
        if not cands:
            return None, "none", "missing_coin_image"
        chosen = sorted(cands, key=lambda p: str(p))[0]
        return str(chosen), "coin_episode_metadata", None

    # offline_proxy: trustworthy CoIN first
    cands = loader.get_episode_image_candidates(split, episode)
    if cands:
        chosen = sorted(cands, key=lambda p: str(p))[0]
        return str(chosen), "coin_episode_metadata", None

    if idkvqa_images:
        cat = episode.target_category.lower()
        for key in idkvqa_images:
            if cat in key or key in cat:
                imgs = idkvqa_images[key]
                if imgs:
                    return imgs[random.randint(0, len(imgs) - 1)], "idkvqa_proxy", None
        all_imgs = [p for paths in idkvqa_images.values() for p in paths]
        if all_imgs:
            return random.choice(all_imgs), "idkvqa_proxy_random", None

    return None, "none", "missing_coin_image"


def preload_idkvqa_images(cache_dir):
    """Load IDKVQA images grouped by category. Caches to disk."""
    index_file = cache_dir / "idkvqa_image_index.json"
    images_dir = cache_dir / "idkvqa_images"

    if index_file.exists():
        with open(index_file) as f:
            return json.load(f)

    print("[Runner] Downloading and caching IDKVQA images (first run)...")
    from datasets import load_dataset
    ds = load_dataset("ftaioli/IDKVQA", split="val")

    images_dir.mkdir(parents=True, exist_ok=True)
    index = {}

    cat_re = re.compile(
        r"(?:is|are|does)\s+the\s+(\w+[\s\w]*?)\s+"
        r"(?:a |an |in |on |made|have|open|closed|wall|position|design|color|"
        r"upholster|lock|height|taller|large|style|comfort|modern|antique|"
        r"locate|near|next|against|corner|turned|stand|simple|sleek|smooth|"
        r"visible|curved)",
        re.IGNORECASE,
    )

    for i, row in enumerate(ds):
        match = cat_re.search(row["question"])
        cat = match.group(1).strip().lower() if match else "unknown"
        img_path = str(images_dir / f"{i:04d}.jpg")
        if not Path(img_path).exists():
            row["image"].save(img_path, "JPEG")
        index.setdefault(cat, []).append(img_path)

    with open(index_file, "w") as f:
        json.dump(index, f)

    print(f"  Cached {sum(len(v) for v in index.values())} images, {len(index)} categories")
    return index


def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# Main offline evaluation
# =============================================================================


def run_offline_evaluation(
    config,
    coin_bench_dir: str,
    splits=None,
    limit=None,
    seed: int = 42,
    user_mode: str = "description_based",
    run_mode: str = "offline_proxy",
    use_idkvqa_proxy: bool | None = None,
    strict_coin_images: bool = False,
):
    """
    Static integration test. See module docstring -- **not** a primary benchmark.

    ``strict_coin_images``: if True and ``offline_static_coin`` cannot resolve a
    metadata-linked image, raises ``RuntimeError`` immediately.
    """
    from .aiuta_pipeline import AIUTAPipeline, PolicySignal
    from ..evaluation.coin_loader import CoINBenchLoader

    if use_idkvqa_proxy is not None:
        run_mode = "offline_proxy" if use_idkvqa_proxy else "offline_static_coin"

    if run_mode == "habitat":
        raise NotImplementedError(
            "habitat / online CoIN is not implemented in episode_runner; use Habitat elsewhere."
        )

    set_seeds(seed)
    coin_path = Path(coin_bench_dir)

    loader = CoINBenchLoader(coin_bench_dir)
    loader.ensure_downloaded()

    if splits is None:
        splits = loader.available_splits()

    idkvqa_images = None
    if run_mode == "offline_proxy":
        try:
            idkvqa_images = preload_idkvqa_images(coin_path / ".cache")
        except Exception as e:
            print(f"[Runner] IDKVQA proxy not available: {e}")

    all_results: list[OfflineEpisodeResult] = []

    for split in splits:
        episodes = loader.load_episodes(split)
        if limit is not None:
            episodes = episodes[:limit]

        print(f"\n[Runner] Split: {split} ({len(episodes)} episodes) mode={run_mode}")

        placeholder_user = lambda q: "I don't know"
        pipeline = None

        for ep_idx, episode in enumerate(episodes):
            image_path, image_source, skipped_reason = resolve_episode_image(
                loader, episode, split, run_mode, idkvqa_images, coin_path,
            )

            if image_path is None:
                if strict_coin_images and run_mode == "offline_static_coin":
                    raise RuntimeError(
                        f"No trustworthy CoIN image for episode_id={episode.episode_id!r} "
                        f"scene_id={episode.scene_id!r} split={split!r}. "
                        f"See get_episode_image_candidates (metadata-linked paths only)."
                    )
                all_results.append(
                    OfflineEpisodeResult(
                        episode_id=episode.episode_id,
                        split=split,
                        scene_id=episode.scene_id or "",
                        target_category=episode.target_category,
                        image_path=None,
                        image_source=image_source,
                        pipeline_signal="skipped",
                        target_detected=False,
                        num_detections=0,
                        num_questions=0,
                        num_kg_nodes=0,
                        pipeline_latency_sec=0.0,
                        detector_latency_sec=None,
                        skipped_reason=skipped_reason or "missing_coin_image",
                        log_summary=[],
                        target_description=episode.target_description or "",
                    )
                )
                continue

            if pipeline is None:
                pipeline = AIUTAPipeline(config, ask_human=placeholder_user)

            pipeline.new_episode(episode.target_category)
            pipeline.set_ask_human(
                make_simulated_user(
                    target_description=episode.target_description,
                    target_category=episode.target_category,
                    mode=user_mode,
                )
            )

            t0 = time.perf_counter()
            step = pipeline.on_detection(observation=image_path, timestep=0)
            latency = time.perf_counter() - t0

            log = pipeline.episode_log
            detected_labels = [e["detection"] for e in log if "detection" in e]
            target_cat_lower = episode.target_category.lower()

            target_detected = any(target_cat_lower in d.lower() for d in detected_labels)
            non_target = [d for d in detected_labels if target_cat_lower not in d.lower()]

            kg_nodes = pipeline.kg.num_objects
            kg_attrs = sum(len(n.attributes) for n in pipeline.kg.all_objects())

            questions = [e["question"] for e in log if e.get("action") == "ask" and e.get("question")]

            all_results.append(
                OfflineEpisodeResult(
                    episode_id=episode.episode_id,
                    split=split,
                    scene_id=episode.scene_id or "",
                    target_category=episode.target_category,
                    image_path=image_path,
                    image_source=image_source,
                    pipeline_signal=step.signal.value,
                    target_detected=target_detected,
                    num_detections=len(detected_labels),
                    num_questions=pipeline.num_questions_asked,
                    num_kg_nodes=kg_nodes,
                    pipeline_latency_sec=latency,
                    detector_latency_sec=step.detector_latency_sec,
                    detector_preprocess_sec=step.detector_preprocess_sec,
                    detector_generate_sec=step.detector_generate_sec,
                    detector_parse_sec=step.detector_parse_sec,
                    skipped_reason=None,
                    log_summary=log[:50],
                    target_description=episode.target_description or "",
                    num_target_detections=sum(1 for d in detected_labels if target_cat_lower in d.lower()),
                    non_target_detection_labels=non_target,
                    kg_attributes_extracted=kg_attrs,
                    total_latency_sec=latency,
                )
            )

            if (ep_idx + 1) % 5 == 0 or (ep_idx + 1) == len(episodes):
                det_r = sum(1 for r in all_results if r.target_detected and r.skipped_reason is None)
                denom = sum(1 for r in all_results if r.skipped_reason is None)
                dr = det_r / denom if denom else 0
                avg_kg = mean(r.num_kg_nodes for r in all_results if r.skipped_reason is None) if denom else 0
                print(
                    f"  [{ep_idx+1}/{len(episodes)}] {episode.target_category:15s} "
                    f"det={dr:.0%} kg={avg_kg:.1f} lat={latency:.1f}s sig={step.signal.value}"
                )

    skipped_n = sum(1 for r in all_results if r.skipped_reason is not None)
    if skipped_n:
        print(f"\n[Runner] Skipped or failed episodes (no image): {skipped_n}")

    return all_results


def compute_offline_metrics(results):
    if not results:
        return {"num_episodes": 0}
    n = len(results)
    ran = [r for r in results if r.skipped_reason is None]
    n_ran = len(ran) or 1

    det_rate = sum(1 for r in ran if r.target_detected) / n_ran
    total_non_target = sum(len(r.non_target_detection_labels) for r in ran)
    total_dets = sum(r.num_detections for r in ran)
    non_target_rate = total_non_target / total_dets if total_dets > 0 else 0
    signal_dist = Counter(r.pipeline_signal for r in results)

    per_cat = {}
    for cat in sorted(set(r.target_category for r in results)):
        cr = [r for r in results if r.target_category == cat]
        nc = len(cr)
        cr_ran = [r for r in cr if r.skipped_reason is None]
        nc_r = len(cr_ran) or 1
        per_cat[cat] = {
            "n": nc,
            "detection_rate": round(sum(1 for r in cr_ran if r.target_detected) / nc_r * 100, 1),
            "avg_kg_nodes": round(mean(r.num_kg_nodes for r in cr_ran), 1) if cr_ran else 0,
            "avg_questions": round(mean(r.num_questions for r in cr_ran), 1) if cr_ran else 0,
        }

    return {
        "num_episodes": n,
        "detection_rate": round(det_rate * 100, 2),
        "avg_detections": round(mean(r.num_detections for r in ran), 2) if ran else 0,
        "non_target_detection_rate": round(non_target_rate * 100, 2),
        "avg_kg_nodes": round(mean(r.num_kg_nodes for r in ran), 2) if ran else 0,
        "avg_kg_attrs": round(mean(r.kg_attributes_extracted for r in ran), 2) if ran else 0,
        "avg_questions": round(mean(r.num_questions for r in ran), 2) if ran else 0,
        "signal_distribution": dict(signal_dist),
        "avg_latency_sec": round(mean(r.pipeline_latency_sec for r in ran), 2) if ran else 0,
        "per_category": per_cat,
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Auxiliary offline CoIN **static integration test** only -- NOT the primary IDKVQA "
            "benchmark and NOT equivalent to official online CoIN / Habitat results."
        ),
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--mode",
        choices=["offline_static_coin", "offline_proxy", "offline", "habitat"],
        default="offline_proxy",
        help="offline aliases to offline_proxy. habitat is not implemented.",
    )
    parser.add_argument("--coin_bench_dir", type=str, default="./data/CoIN-Bench")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--splits", nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--user_mode", choices=["always_idk", "description_based"], default="description_based")
    parser.add_argument("--no_idkvqa_proxy", action="store_true")
    parser.add_argument(
        "--strict_coin_images",
        action="store_true",
        help="Fail immediately if offline_static_coin cannot resolve a metadata-linked image.",
    )
    args = parser.parse_args()

    run_mode = args.mode
    if run_mode == "offline":
        run_mode = "offline_proxy"
    if run_mode == "habitat":
        print(
            "[Runner] habitat / online CoIN is not implemented in this module "
            "(requires Habitat / VLFM). No results written."
        )
        return

    from ..config import Config
    config = Config.from_yaml(args.config)

    if args.no_idkvqa_proxy:
        run_mode = "offline_static_coin"

    if run_mode == "offline_static_coin":
        print(
            "\n*** WARNING: offline_static_coin is a **static auxiliary** integration test only.\n"
            "    It is NOT equivalent to the official **online** CoIN benchmark (Habitat, SR/SPL).\n"
            "    Primary research metrics: use IDKVQA (evaluation.idkvqa_eval).\n"
        )

    results = run_offline_evaluation(
        config=config,
        coin_bench_dir=args.coin_bench_dir,
        splits=args.splits,
        limit=args.limit,
        seed=args.seed,
        user_mode=args.user_mode,
        run_mode=run_mode,
        strict_coin_images=args.strict_coin_images,
    )

    metrics = compute_offline_metrics(results)
    per_split = {}
    for s in sorted(set(r.split for r in results)):
        per_split[s] = compute_offline_metrics([r for r in results if r.split == s])

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    import yaml
    with open(args.config) as f:
        config_content = yaml.safe_load(f)

    def _episode_row(r: OfflineEpisodeResult) -> dict:
        return {
            "episode_id": r.episode_id,
            "split": r.split,
            "scene_id": r.scene_id,
            "target_category": r.target_category,
            "image_path": r.image_path,
            "image_source": r.image_source,
            "skipped_reason": r.skipped_reason,
            "pipeline_signal": r.pipeline_signal,
            "target_detected": r.target_detected,
            "num_detections": r.num_detections,
            "num_questions": r.num_questions,
            "num_kg_nodes": r.num_kg_nodes,
            "pipeline_latency_sec": round(r.pipeline_latency_sec, 6),
            "detector_latency_sec": None if r.detector_latency_sec is None else round(r.detector_latency_sec, 6),
            "detector_preprocess_sec": r.detector_preprocess_sec,
            "detector_generate_sec": r.detector_generate_sec,
            "detector_parse_sec": r.detector_parse_sec,
            "not_official_coin_online": r.not_official_coin_online,
            "non_target_detection_labels": r.non_target_detection_labels,
            "log_summary": r.log_summary,
        }

    output = {
        "config": config_content,
        "config_serializable": config.to_serializable_dict(),
        "seed": args.seed,
        "user_mode": args.user_mode,
        "run_mode": run_mode,
        "benchmark_role": "auxiliary_offline_integration_test",
        "intended_use": "integration_only_not_primary_evidence",
        "not_official_coin_online": True,
        "strict_coin_images": args.strict_coin_images,
        "model": config.model.model_id,
        "metrics": metrics,
        "per_split": per_split,
        "episodes": [_episode_row(r) for r in results],
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    m = metrics
    print(f"\n{'='*65}")
    print("AUXILIARY OFFLINE CoIN STATIC -- NOT official online CoIN")
    print(f"{'='*65}")
    print(f"  Episodes (records): {m['num_episodes']}")
    print(f"  Detection rate:     {m['detection_rate']}%")
    print(f"  Non-target det rate: {m['non_target_detection_rate']}%")
    print(f"  Avg KG nodes:       {m['avg_kg_nodes']}")
    print(f"  Avg questions:      {m['avg_questions']}")
    print(f"  Signals:            {m['signal_distribution']}")
    print(f"  Avg latency:        {m['avg_latency_sec']}s")
    print(f"\n  Per category:")
    for cat, cm in m.get("per_category", {}).items():
        print(
            f"    {cat:20s} det={cm['detection_rate']:5.1f}% "
            f"kg={cm['avg_kg_nodes']:.1f} NQ={cm['avg_questions']:.1f}"
        )
    print(f"{'='*65}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
