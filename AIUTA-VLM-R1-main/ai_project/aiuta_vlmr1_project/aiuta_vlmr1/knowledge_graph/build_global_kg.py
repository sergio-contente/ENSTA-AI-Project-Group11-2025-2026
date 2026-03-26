"""
build_global_kg -- Offline **global** knowledge graph for IDKVQA.

Runs the same detection prompt as ``mode=kg`` once per **unique image** (fingerprint:
func:`aiuta_vlmr1.evaluation.idkvqa_eval.stable_idkvqa_image_id`), extracts triples with
:class:`TripleExtractor`, and accumulates nodes in one :class:`SceneKnowledgeGraph`.

**Oracle / leakage note:** building on the same split you evaluate is an upper-bound-style
analysis; document accordingly in papers.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

from ..config import Config
from ..evaluation.idkvqa_eval import (
    DETECTION_SYSTEM,
    _generate_chat,
    _idkvqa_eval_options,
    stable_idkvqa_image_id,
)
from ..evaluation.vlm_inference_utils import extract_answer_and_reasoning
from ..utils.model_loader import ModelLoader
from .scene_graph import SceneKnowledgeGraph
from .triple_extractor import TripleExtractor

_DET_PROMPT = (
    "Carefully inspect this indoor scene. Describe all visible objects, their colors, "
    "materials, sizes, and spatial relationships."
)


def _load_hf_rows(split: str, seed: int | None) -> list[dict[str, Any]]:
    from datasets import load_dataset

    rows: list[dict[str, Any]] = list(load_dataset("ftaioli/IDKVQA", split=split))
    if seed is not None:
        random.Random(seed).shuffle(rows)
    return rows


def build_global_kg(
    config: Config,
    split: str = "val",
    limit: int | None = None,
    seed: int = 42,
    output_path: str = "results/global_kg.json",
) -> SceneKnowledgeGraph:
    """
    Train phase: detection pass on unique images, merge triples into one KG, save JSON.

    Parameters
    ----------
    limit :
        Maximum number of **unique images** to process (each costs one detection VLM call).
        ``None`` processes every unique image in the (shuffled) split rows.
    """
    opts = _idkvqa_eval_options(config)
    det_tokens = int(opts["detection_max_new_tokens"])

    random.seed(seed)
    rows = _load_hf_rows(split, seed)

    loader = ModelLoader.get_instance(config.model)
    kg = SceneKnowledgeGraph()
    seen_fp: set[str] = set()
    n_det = 0

    for row in rows:
        if limit is not None and n_det >= limit:
            break
        pil_image = row["image"]
        fp = stable_idkvqa_image_id(pil_image)
        if fp in seen_fp:
            continue
        seen_fp.add(fp)

        det_raw, _dt, _, _, _ = _generate_chat(
            loader,
            pil_image,
            DETECTION_SYSTEM,
            _DET_PROMPT,
            max_new_tokens=det_tokens,
            output_scores=False,
        )
        det_ans, det_think = extract_answer_and_reasoning(det_raw)
        reasoning = det_think or det_ans

        extraction = TripleExtractor.extract_all(
            reasoning=reasoning,
            category="object",
            queried_objects=[],
            timestep=0,
        )
        node = kg.add_object_merged("scene", bbox=None, timestep=0, image_id=fp)
        if extraction.attributes:
            kg.update_attributes(node.obj_id, extraction.attributes)
        for rel in extraction.spatial_relations:
            kg.add_spatial_relation(node.obj_id, rel)

        n_det += 1
        if n_det % 50 == 0:
            print(f"[build_global_kg] detection passes: {n_det}, graph nodes: {kg.num_objects}", flush=True)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    kg.save_json(str(out))
    print(
        f"[build_global_kg] Wrote {out} ({kg.num_objects} nodes, {n_det} detection passes, "
        f"{len(seen_fp)} unique image ids)",
        flush=True,
    )
    return kg


def main() -> None:
    p = argparse.ArgumentParser(description="Build global IDKVQA KG (1 detection VLM call per unique image)")
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--split", type=str, default="val")
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max unique images to process (each = one detection call)",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=str, default="results/global_kg.json")
    args = p.parse_args()

    cfg = Config.from_yaml(args.config) if args.config else Config()
    build_global_kg(
        cfg,
        split=args.split,
        limit=args.limit,
        seed=args.seed,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
