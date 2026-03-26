"""
graph_matcher.py -- Alignment scoring between detected objects and target facts.
Replaces AIUTA P_score LLM prompt with deterministic graph matching.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Any

import numpy as np

from .schema import ObjectNode, TargetFacts

# Text embedding cache (valid during process, cleared between episodes if needed)
_EMBEDDING_CACHE: dict[str, np.ndarray] = {}


def clear_embedding_cache() -> None:
    """Clear cache on episode reset to free memory."""
    global _EMBEDDING_CACHE
    _EMBEDDING_CACHE.clear()


def _get_text_embedding(text: str, loader: Any) -> np.ndarray | None:
    global _EMBEDDING_CACHE
    cache_key = f"{id(loader)}:{text}"
    if cache_key in _EMBEDDING_CACHE:
        return _EMBEDDING_CACHE[cache_key]
    try:
        import torch

        proc = loader.processor
        tokenizer = getattr(proc, "tokenizer", None)
        if tokenizer is None:
            return None
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
        inputs = {k: v.to(loader.device) for k, v in inputs.items()}
        with torch.inference_mode():
            outputs = loader.model.model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                output_hidden_states=True,
            )
        hidden = outputs.hidden_states[-1]
        mask = (
            inputs.get("attention_mask", torch.ones_like(inputs["input_ids"]))
            .unsqueeze(-1)
            .float()
        )
        embedding = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
        embedding = embedding.squeeze(0).float().cpu().numpy()
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        _EMBEDDING_CACHE[cache_key] = embedding
        # Limitar tamanho do cache para evitar OOM
        if len(_EMBEDDING_CACHE) > 1000:
            oldest = next(iter(_EMBEDDING_CACHE))
            del _EMBEDDING_CACHE[oldest]
        return embedding
    except Exception:
        return None


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Similaridade cosine entre dois vetores normalizados."""
    return float(np.dot(a, b))


def _embedding_match(
    obj_val: str,
    target_val: str,
    loader: Any,
    threshold_match: float = 0.85,
    threshold_mismatch: float = 0.50,
) -> str:
    """
    Compare obj_val and target_val via embeddings.
    Returns:
      "match"    -- similarity >= threshold_match
      "mismatch" -- similarity < threshold_mismatch
      "uncertain" -- grey zone, needs VLM judge
    """
    emb_obj = _get_text_embedding(obj_val, loader)
    emb_tgt = _get_text_embedding(target_val, loader)
    if emb_obj is None or emb_tgt is None:
        return "uncertain"
    sim = _cosine_similarity(emb_obj, emb_tgt)
    if sim >= threshold_match:
        return "match"
    if sim < threshold_mismatch:
        return "mismatch"
    return "uncertain"


def _short_match(a: str, b: str) -> bool:
    """Substring + synonym check for short attribute values."""
    if a == b:
        return True
    if a in b or b in a:
        return True
    from .think_feature_extractor import qualifier_matches_response
    return qualifier_matches_response(a, b)


def _known_pair_result(obj_val: str, target_val: str, loader: Any | None) -> str:
    o = obj_val.strip().lower()
    t = target_val.strip().lower()
    if len(o.split()) <= 3 or len(t.split()) <= 3:
        return "match" if _short_match(o, t) else "mismatch"
    if loader is not None:
        return _embedding_match(obj_val, target_val, loader)
    return "match" if _short_match(o, t) else "mismatch"


def _negative_pair_result(obj_val: str, neg_val: str, loader: Any | None) -> str:
    o = obj_val.strip().lower()
    n = neg_val.strip().lower()
    if len(o.split()) <= 3 or len(n.split()) <= 3:
        return "match" if _short_match(o, n) else "mismatch"
    if loader is not None:
        return _embedding_match(obj_val, neg_val, loader)
    return "match" if _short_match(o, n) else "mismatch"


MIN_RESOLVED_FOR_STOP = 1


class GraphMatcher:
    @staticmethod
    def compute_alignment(
        obj: ObjectNode,
        target: TargetFacts,
        loader: Any = None,
    ) -> float:
        if target.num_facts == 0:
            return -1.0
        obj_attrs = {k: v.value for k, v in obj.attributes.items()} if obj.attributes else {}
        print(
            f"[GraphMatcher] Comparing target={dict(target.known_attributes)} "
            f"neg={dict(target.negative_attributes)} "
            f"vs obj={obj_attrs}"
        )
        total = 0
        matched = 0
        contradicted = 0

        for attr_name, target_val in target.known_attributes.items():
            total += 1
            obj_val = obj.get_attribute_value(attr_name)
            if obj_val is not None:
                result = _known_pair_result(obj_val, target_val, loader)
                if result == "match":
                    matched += 1
                elif result == "mismatch":
                    contradicted += 1

        for attr_name, neg_val in target.negative_attributes.items():
            total += 1
            obj_val = obj.get_attribute_value(attr_name)
            if obj_val is not None:
                result = _negative_pair_result(obj_val, neg_val, loader)
                if result == "match":
                    contradicted += 1
                elif result == "mismatch":
                    matched += 1

        if contradicted > 0:
            return 0.0
        if total == 0:
            return -1.0
        resolved = matched + contradicted
        if resolved == 0:
            return -1.0
        if resolved < MIN_RESOLVED_FOR_STOP:
            print(
                f"[GraphMatcher] alignment resolved={resolved} < MIN_RESOLVED={MIN_RESOLVED_FOR_STOP}, "
                f"forcing ASK (matched={matched}, total={total})"
            )
            return -1.0

        positive_matched = 0
        for attr_name, target_val in target.known_attributes.items():
            obj_val = obj.get_attribute_value(attr_name)
            if obj_val is not None:
                if _known_pair_result(obj_val, target_val, loader) == "match":
                    positive_matched += 1

        score = matched / resolved

        if positive_matched == 0 and matched > 0:
            capped = min(score, 0.5)
            print(
                f"[GraphMatcher] alignment score={score:.2f} but positive_matched=0 "
                f"(all matches from negatives) -- capping to {capped:.2f}"
            )
            score = capped

        print(
            f"[GraphMatcher] alignment score={score:.2f} "
            f"(matched={matched}, positive_matched={positive_matched}, "
            f"resolved={resolved}, total_target={total})"
        )
        return score

    @staticmethod
    def compute_alignment_with_vlm_fallback(
        obj: ObjectNode,
        target: TargetFacts,
        tau_stop: float = 0.8,
        vlm_judge_fn=None,
        loader=None,
        detected_crop=None,
    ) -> float:
        score = GraphMatcher.compute_alignment(obj, target, loader=loader)
        if score >= tau_stop or score == 0.0:
            return score
        if vlm_judge_fn is None:
            return score
        # Score -1.0 = insufficient information (empty KG, no obj/target overlap, or nothing resolved).
        # Do not let vlm_judge override the decision in these cases -- force ASK.
        if score < tau_stop:
            if score == -1.0:
                print(f"[GraphMatcher] vlm_judge skipped -- alignment inconclusive "
                    f"(score={score}, target_facts={target.num_facts})")
            else:
                print(f"[GraphMatcher] vlm_judge skipped -- KG score {score:.2f} "
                    f"< tau_stop {tau_stop:.2f}, alignment authoritative")
            return score
        obj_desc = obj.to_natural_language()
        target_desc = target.to_natural_language()
        try:
            import inspect

            sig = inspect.signature(vlm_judge_fn)
            if "detected_crop" in sig.parameters:
                is_match = vlm_judge_fn(obj_desc, target_desc, detected_crop=detected_crop)
            else:
                is_match = vlm_judge_fn(obj_desc, target_desc)
            if is_match:
                return tau_stop
        except Exception:
            pass
        return score

    @staticmethod
    def explain_alignment(
        obj: ObjectNode,
        target: TargetFacts,
        loader: Any = None,
    ) -> dict[str, Any]:
        """Structured breakdown for debugging / papers (not a second scoring API)."""
        matched: list[dict[str, str]] = []
        missing: list[str] = []
        contradictions: list[str] = []

        for attr_name, target_val in target.known_attributes.items():
            obj_val = obj.get_attribute_value(attr_name)
            if obj_val is None:
                missing.append(attr_name)
            else:
                result = _known_pair_result(obj_val, target_val, loader)
                if result == "mismatch":
                    contradictions.append(
                        f"{attr_name}: obj={obj_val!r} vs target={target_val!r}"
                    )
                elif result == "match":
                    matched.append({"attribute": attr_name, "value": obj_val})

        for attr_name, neg_val in target.negative_attributes.items():
            obj_val = obj.get_attribute_value(attr_name)
            if obj_val is not None:
                result = _negative_pair_result(obj_val, neg_val, loader)
                if result == "match":
                    contradictions.append(
                        f"{attr_name}: obj={obj_val!r} matches forbidden {neg_val!r}"
                    )

        score = GraphMatcher.compute_alignment(obj, target, loader=loader)
        return {
            "score": score,
            "matched": matched,
            "missing_attributes": missing,
            "contradictions": contradictions,
            "target_known": dict(target.known_attributes),
            "target_negative": dict(target.negative_attributes),
        }

    @staticmethod
    def find_contradictions(
        obj: ObjectNode,
        target: TargetFacts,
        loader: Any = None,
    ) -> list[str]:
        contradictions: list[str] = []
        for attr_name, target_val in target.known_attributes.items():
            obj_val = obj.get_attribute_value(attr_name)
            if obj_val is not None:
                if _known_pair_result(obj_val, target_val, loader) == "mismatch":
                    contradictions.append(
                        f"{attr_name}: obj={obj_val}, target={target_val}"
                    )
        for attr_name, neg_val in target.negative_attributes.items():
            obj_val = obj.get_attribute_value(attr_name)
            if obj_val is not None:
                if _negative_pair_result(obj_val, neg_val, loader) == "match":
                    contradictions.append(
                        f"{attr_name}: obj={obj_val}, target NOT {neg_val}"
                    )
        return contradictions

    @staticmethod
    def compute_discriminative_power(attr_name: str, instances: list[ObjectNode]) -> float:
        if len(instances) <= 1:
            return 0.0
        values = []
        for inst in instances:
            val = inst.get_attribute_value(attr_name)
            if val is not None:
                values.append(val.lower())
        if not values:
            return 0.5
        unique = set(values)
        if len(unique) == 1:
            return 0.0
        counts = Counter(values)
        total = len(values)
        entropy = -sum((c / total) * math.log2(c / total) for c in counts.values())
        max_ent = math.log2(len(unique))
        return entropy / max_ent if max_ent > 0 else 0.0
