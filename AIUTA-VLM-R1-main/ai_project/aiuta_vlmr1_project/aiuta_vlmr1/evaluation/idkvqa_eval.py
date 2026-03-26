"""
Primary offline benchmark: IDKVQA (ftaioli/IDKVQA).

This module is the canonical entry point for uncertainty-aware Yes/No/IDK
evaluation. Official online CoIN / Habitat metrics are out of scope here; see
``coin_metrics`` for SR/SPL definitions (online embodiment).

Usage:
    python -m aiuta_vlmr1.evaluation.idkvqa_eval --output results/idkvqa/run.json --mode raw --limit 20
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from dataclasses import dataclass, replace
from io import BytesIO
from pathlib import Path
from typing import Any

import torch

from ..config import Config, ModelConfig
from ..knowledge_graph.scene_graph import SceneKnowledgeGraph
from ..utils.model_loader import ModelLoader, model_configs_equivalent
from .answer_normalization import LABEL_IDK, normalize_yes_no_idk
from .idkvqa_kg import (
    build_kg_attributes_from_detection,
    classify_question_type,
    compute_kg_hybrid_prediction,
    compute_kg_hybrid_prediction_entropy,
    compute_kg_hybrid_prediction_relaxed,
    kg_answer_from_attributes,
    parse_question_attribute,
)
from .idkvqa_types import (
    QAExampleResult,
    aggregate_idkvqa_metrics,
    benchmark_run_metadata,
)
from .uncertainty_abstention import AbstentionDecision, apply_uncertainty_threshold
from .vlm_inference_utils import (
    compute_answer_token_entropy,
    compute_first_token_entropy,
    compute_first_token_max_prob,
    estimate_reasoning_certainty,
    extract_answer_and_reasoning,
)
from ..self_questioner.two_pass_questioner import TwoPassSelfQuestioner

IDKVQA_MODES = (
    "raw",
    "raw_two_pass",
    "threshold",
    "kg",
    "kg_threshold",
    "two_pass_kg",
    "two_pass_kg_relaxed",
    "two_pass_kg_entropy",
    "global_kg",
    "global_kg_entropy",
)

# Matches the dataset prompt style (CoIN / VLM-R1 VQA protocol).
IDKVQA_SYSTEM = (
    "You are a helpful assistant that answers visual questions about images. "
    "You must reason carefully about what you can see in the image. "
    "The reasoning process must be enclosed within <think></think> tags, "
    "and the final answer must be enclosed within <answer> </answer> tags."
)

DETECTION_SYSTEM = (
    "You are a helpful assistant specialized in visual reasoning for indoor scenes. "
    "Carefully inspect the image and describe what you see in detail, including colors, "
    "materials, sizes, spatial relationships. The reasoning must be in <think></think> tags, "
    "answer in <answer></answer> tags."
)


def load_idkvqa(limit: int | None = None, split: str = "val", seed: int | None = None) -> list[dict[str, Any]]:
    """
    Load IDKVQA from HuggingFace.

    Returns rows with: sample_id, image (PIL), question, answers, ground_truth,
    annotator_agreement, is_uncertain.
    """
    from datasets import load_dataset

    print("[IDKVQA] Loading dataset from ftaioli/IDKVQA...")
    ds = load_dataset("ftaioli/IDKVQA", split=split)
    rows = list(ds)
    if seed is not None:
        rng = random.Random(seed)
        rng.shuffle(rows)

    samples: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        if limit is not None and i >= limit:
            break
        answers = row["answers"]
        gt = max(answers, key=answers.get)
        total_votes = sum(answers.values())
        agreement = answers[gt] / total_votes if total_votes > 0 else 0.0
        samples.append({
            "sample_id": str(row.get("id", i)),
            "idx": i,
            "image": row["image"],
            "question": row["question"],
            "answers": answers,
            "ground_truth": gt,
            "annotator_agreement": agreement,
            "is_uncertain": gt == LABEL_IDK,
        })
    print(f"[IDKVQA] Loaded {len(samples)} samples (split={split})")
    return samples


def stable_idkvqa_image_id(pil_image: Any) -> str:
    """
    Stable fingerprint for an IDKVQA image (PNG bytes SHA-256).

    Used for global KG deduplication and lookup: same pixels => same id, including
    across samples that share one image but have different ``sample_id`` values.
    """
    buf = BytesIO()
    im = pil_image.copy() if hasattr(pil_image, "copy") else pil_image
    im.save(buf, format="PNG")
    return hashlib.sha256(buf.getvalue()).hexdigest()


def _idkvqa_eval_options(config: Config) -> dict[str, Any]:
    d = getattr(config, "_idkvqa_eval", None) or {}
    return {
        "entropy_threshold": float(d.get("entropy_threshold", 0.35)),
        "abstention_rule": str(d.get("abstention_rule", "entropy_above_tau_to_idk")),
        "vqa_max_new_tokens": int(d.get("vqa_max_new_tokens", 256)),
        "detection_max_new_tokens": int(d.get("detection_max_new_tokens", 512)),
        "second_pass_model_id": d.get("second_pass_model_id"),
        "second_pass_processor_id": d.get("second_pass_processor_id"),
    }


@dataclass
class _ChatResult:
    raw_output: str
    gen_time: float
    entropy_first_token: float | None
    entropy_answer_token: float | None
    max_prob: float | None


def _generate_chat(
    loader: ModelLoader,
    pil_image: Any,
    system: str,
    user_text: str,
    max_new_tokens: int,
    output_scores: bool,
) -> tuple[str, float, float | None, float | None, float]:
    """Returns raw_output, latency_sec, entropy, max_prob, gen_time only for VQA split."""
    result = _generate_chat_full(loader, pil_image, system, user_text, max_new_tokens, output_scores)
    return result.raw_output, result.gen_time, result.entropy_first_token, result.max_prob, result.gen_time


def _generate_chat_full(
    loader: ModelLoader,
    pil_image: Any,
    system: str,
    user_text: str,
    max_new_tokens: int,
    output_scores: bool,
) -> _ChatResult:
    """Full chat generation returning both entropy variants."""
    proc = loader.processor
    model = loader.model
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": [
            {"type": "image", "image": pil_image},
            {"type": "text", "text": user_text},
        ]},
    ]
    text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = proc(
        text=[text], images=[pil_image], padding=True, return_tensors="pt",
    ).to(loader.device)

    t0 = time.perf_counter()
    gen_kw: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
    }
    if output_scores:
        gen_kw["output_scores"] = True
        gen_kw["return_dict_in_generate"] = True

    with torch.inference_mode():
        outputs = model.generate(**inputs, **gen_kw)

    gen_time = time.perf_counter() - t0

    if output_scores and hasattr(outputs, "sequences"):
        gen_ids = outputs.sequences
        trimmed = [o[len(inp):] for inp, o in zip(inputs.input_ids, gen_ids)]
        raw_output = proc.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False,
        )[0]
        vocab_size = getattr(proc, "tokenizer", None)
        vs = None
        if vocab_size is not None and hasattr(vocab_size, "vocab_size"):
            vs = int(vocab_size.vocab_size)
        entropy_first = compute_first_token_entropy(outputs, vocab_size=vs)
        max_prob = compute_first_token_max_prob(outputs)

        # Answer-token entropy (token after <answer> tag)
        generated_ids = list(trimmed[0].tolist()) if trimmed else []
        entropy_answer, _ = compute_answer_token_entropy(outputs, proc, generated_ids)
        entropy_answer = entropy_answer if entropy_answer >= 0 else None

        return _ChatResult(
            raw_output=raw_output,
            gen_time=gen_time,
            entropy_first_token=entropy_first,
            entropy_answer_token=entropy_answer,
            max_prob=max_prob,
        )

    gen_ids = outputs
    trimmed = [o[len(inp):] for inp, o in zip(inputs.input_ids, gen_ids)]
    raw_output = proc.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False,
    )[0]
    return _ChatResult(
        raw_output=raw_output,
        gen_time=gen_time,
        entropy_first_token=None,
        entropy_answer_token=None,
        max_prob=None,
    )


def _build_qa_result(
    *,
    sample: dict[str, Any],
    mode: str,
    raw_answer: str,
    raw_normalized: str,
    final_prediction: str,
    confidence_score: float | None,
    entropy_score: float | None,
    used_kg: bool,
    used_threshold: bool,
    used_abstention: bool,
    latency_sec: float,
    metadata: dict[str, Any],
    question_type: str,
    num_model_calls: int,
    num_detector_calls: int,
    num_questioner_calls: int,
    num_trigger_calls: int,
    num_questions_asked: int,
    num_kg_nodes: int | None,
    total_latency_sec: float,
    detector_latency_sec: float | None,
    decision_latency_sec: float | None,
    uncertainty_score: float | None,
    threshold: float | None,
    abstained: bool,
) -> QAExampleResult:
    gt = sample["ground_truth"]
    pred_n = normalize_yes_no_idk(final_prediction)
    correct = pred_n == gt
    return QAExampleResult(
        sample_id=str(sample["sample_id"]),
        question=sample["question"],
        ground_truth=gt,
        raw_prediction=raw_answer,
        final_prediction=final_prediction,
        confidence_score=confidence_score,
        entropy_score=entropy_score,
        used_kg=used_kg,
        used_threshold=used_threshold,
        used_abstention=used_abstention,
        latency_sec=latency_sec,
        metadata={
            **metadata,
            "raw_prediction_normalized": raw_normalized,
            "mode": mode,
        },
        question_type=question_type,
        mode=mode,
        raw_prediction_label=raw_normalized,
        correct=correct,
        num_model_calls=num_model_calls,
        num_detector_calls=num_detector_calls,
        num_questioner_calls=num_questioner_calls,
        num_trigger_calls=num_trigger_calls,
        num_questions_asked=num_questions_asked,
        num_kg_nodes=num_kg_nodes,
        total_latency_sec=total_latency_sec,
        detector_latency_sec=detector_latency_sec,
        decision_latency_sec=decision_latency_sec,
        uncertainty_score=uncertainty_score,
        threshold=threshold,
        abstained=abstained,
        annotator_answers=sample.get("answers"),
    )


def finalize_for_mode(
    mode: str,
    *,
    raw_normalized: str,
    uncertainty_score: float | None,
    threshold: float,
    rule: str,
    kg_hybrid: str | None,
) -> tuple[str, bool, bool, bool, AbstentionDecision | None]:
    """
    Apply ablation post-processing. Returns
    (final_label, used_kg, used_threshold, used_abstention, abstention_decision_or_none).
    """
    used_kg = mode in (
        "kg",
        "kg_threshold",
        "two_pass_kg",
        "two_pass_kg_relaxed",
        "two_pass_kg_entropy",
        "global_kg",
        "global_kg_entropy",
    )
    used_th = mode in ("threshold", "kg_threshold")
    abst_dec: AbstentionDecision | None = None

    if mode == "raw":
        return raw_normalized, False, False, False, None

    if mode == "raw_two_pass":
        # Second-pass label only; no KG / graph fusion (extra compute baseline vs ``kg``).
        return raw_normalized, False, False, False, None

    if mode == "threshold":
        abst_dec = apply_uncertainty_threshold(raw_normalized, uncertainty_score, threshold, rule)
        return (
            abst_dec.final_prediction,
            False,
            True,
            abst_dec.abstained,
            abst_dec,
        )

    if mode == "kg":
        assert kg_hybrid is not None
        return kg_hybrid, True, False, False, None

    if mode == "kg_threshold":
        assert kg_hybrid is not None
        abst_dec = apply_uncertainty_threshold(kg_hybrid, uncertainty_score, threshold, rule)
        return (
            abst_dec.final_prediction,
            True,
            True,
            abst_dec.abstained,
            abst_dec,
        )

    if mode == "two_pass_kg":
        assert kg_hybrid is not None
        return kg_hybrid, True, False, False, None

    if mode == "two_pass_kg_relaxed":
        assert kg_hybrid is not None
        return kg_hybrid, True, False, False, None

    if mode == "two_pass_kg_entropy":
        assert kg_hybrid is not None
        return kg_hybrid, True, False, False, None

    if mode == "global_kg":
        assert kg_hybrid is not None
        return kg_hybrid, True, False, False, None

    if mode == "global_kg_entropy":
        assert kg_hybrid is not None
        return kg_hybrid, True, False, False, None

    raise ValueError(f"Unknown mode {mode!r}; expected one of {IDKVQA_MODES}")


def _raw_two_pass_refine_prompt(question: str, first_answer_text: str) -> str:
    return (
        f"Question: {question}\n"
        f"Your first draft answer was: {first_answer_text}\n"
        "Take a second look at the image and answer with exactly one of: Yes, No, or I don't know. "
        "Do not use external knowledge graphs; rely only on visual evidence. "
        "Put the final label only inside <answer></answer> tags."
    )


def _build_second_pass_model_config(
    config: Config,
    opts: dict[str, Any],
    cli_second_pass_model_id: str | None,
    cli_second_pass_processor_id: str | None,
) -> ModelConfig | None:
    cfg = replace(config.second_pass_model) if config.second_pass_model is not None else None
    yml_model_id = opts.get("second_pass_model_id")
    yml_processor_id = opts.get("second_pass_processor_id")
    final_model_id = cli_second_pass_model_id or yml_model_id
    final_processor_id = cli_second_pass_processor_id or yml_processor_id

    if not final_model_id and not final_processor_id and cfg is None:
        return None
    if cfg is None:
        cfg = replace(config.model)
    if final_model_id:
        cfg.model_id = str(final_model_id)
    if final_processor_id:
        cfg.processor_id = str(final_processor_id)
    return cfg


def _textual_uncertainty_fallback(reasoning_bucket: str, rule: str) -> float:
    # Used only when logits-based uncertainty is unavailable.
    if rule == "maxprob_below_tau_to_idk":
        if reasoning_bucket == "high":
            return 0.9
        if reasoning_bucket == "low":
            return 0.1
        return 0.5
    if reasoning_bucket == "high":
        return 0.1
    if reasoning_bucket == "low":
        return 0.9
    return 0.5


def _resolve_uncertainty_signal(
    *,
    rule: str,
    entropy: float | None,
    max_prob: float | None,
    reasoning_bucket: str,
) -> tuple[float | None, str]:
    # Entropy/maxprob are first-class signals; textual certainty is fallback only.
    if rule == "entropy_above_tau_to_idk":
        if entropy is not None:
            return float(entropy), "entropy"
        if max_prob is not None:
            return float(1.0 - max_prob), "max_prob_inverted"
    elif rule == "maxprob_below_tau_to_idk":
        if max_prob is not None:
            return float(max_prob), "max_prob"
        if entropy is not None:
            return float(1.0 - entropy), "entropy_inverted"
    if entropy is not None:
        return float(entropy), "entropy_fallback"
    if max_prob is not None:
        return float(max_prob), "max_prob_fallback"
    return _textual_uncertainty_fallback(reasoning_bucket, rule), "textual_fallback"


def run_idkvqa_benchmark(
    config: Config,
    mode: str,
    limit: int | None = None,
    seed: int = 42,
    split: str = "val",
    entropy_threshold: float | None = None,
    abstention_rule: str | None = None,
    second_pass_model_id: str | None = None,
    second_pass_processor_id: str | None = None,
    global_kg_path: str | None = None,
) -> list[QAExampleResult]:
    """
    Run the primary offline IDKVQA benchmark with a single ablation ``mode``:

    - ``raw``: VLM answer only (normalized).
    - ``raw_two_pass``: same pipeline depth as ``kg`` (detection + two VQA passes) but **no** KG
      matching, hybrid fusion, or graph triggers -- controls for extra compute vs ``kg``.
    - ``threshold``: abstain via uncertainty rule (default: normalized entropy).
    - ``kg``: detection reasoning -> triples -> KG hybrid answer.
    - ``kg_threshold``: KG hybrid, then uncertainty gate to IDK.
    - ``two_pass_kg`` / ``two_pass_kg_relaxed`` / ``two_pass_kg_entropy``: detection + attribute
      pass + VQA; ``two_pass_kg_relaxed`` trusts VLM when no KG slot and no hedging, while
      ``two_pass_kg_entropy`` applies an entropy gate on that fallback.
    - ``global_kg`` / ``global_kg_entropy``: one VQA call per sample; attributes come from a
      pre-built JSON graph (see ``knowledge_graph.build_global_kg``). Uses the same hybrid
      fusion as ``kg`` / ``two_pass_kg_entropy`` respectively. Lookup key is
      :func:`stable_idkvqa_image_id` (content hash), not raw ``sample_id``.

      **Oracle note:** building the global KG on the same val split is an upper-bound-style
      analysis (full-graph context built from the same images you evaluate on).
    """
    if mode not in IDKVQA_MODES:
        raise ValueError(f"mode must be one of {IDKVQA_MODES}, got {mode!r}")
    if mode in ("global_kg", "global_kg_entropy"):
        if not global_kg_path:
            raise ValueError(f"global_kg_path is required for mode {mode!r}")

    opts = _idkvqa_eval_options(config)
    tau = float(entropy_threshold if entropy_threshold is not None else opts["entropy_threshold"])
    rule = str(abstention_rule if abstention_rule is not None else opts["abstention_rule"])
    vqa_tokens = int(opts["vqa_max_new_tokens"])
    det_tokens = int(opts["detection_max_new_tokens"])

    random.seed(seed)
    torch.manual_seed(seed)

    second_pass_cfg = _build_second_pass_model_config(
        config, opts, second_pass_model_id, second_pass_processor_id,
    )
    # When a *different* second checkpoint is used, only one model stays on GPU at a time
    # (see ``two_pass_kg`` swap: unload primary -> attribute pass -> reload primary for VQA).
    use_sequential_second_pass = (
        second_pass_cfg is not None
        and not model_configs_equivalent(config.model, second_pass_cfg)
    )

    loader = ModelLoader.get_instance(config.model)
    if second_pass_cfg is not None and not use_sequential_second_pass:
        second_pass_loader = ModelLoader.get_instance(second_pass_cfg)
    else:
        second_pass_loader = loader

    global_kg: SceneKnowledgeGraph | None = None
    if mode in ("global_kg", "global_kg_entropy"):
        global_kg = SceneKnowledgeGraph.load_json(global_kg_path)
        print(f"[IDKVQA] Loaded global KG from {global_kg_path} ({global_kg.num_objects} objects)")

    samples = load_idkvqa(limit=limit, split=split, seed=seed)
    results: list[QAExampleResult] = []

    det_prompt = (
        "Carefully inspect this indoor scene. Describe all visible objects, their colors, "
        "materials, sizes, and spatial relationships."
    )

    for sample in samples:
        pil_image = sample["image"]
        question = sample["question"]
        attr_type, attr_value = parse_question_attribute(question)

        t0 = time.perf_counter()
        kg_strict: str | None = None
        kg_hybrid: str | None = None
        detection_reasoning = ""
        vqa_reasoning = ""
        raw_output = ""
        raw_answer = ""
        entropy: float | None = None
        entropy_answer: float | None = None
        max_prob: float | None = None
        det_latency = 0.0
        vqa_latency = 0.0

        # Request token entropy for ``raw`` as well so offline threshold sweeps (``threshold_sweep``) work.
        need_scores = mode in ("raw", "threshold", "kg_threshold", "global_kg", "global_kg_entropy")

        num_model_calls = 0
        num_detector_calls = 0
        num_questioner_calls = 0
        num_trigger_calls = 0
        num_questions_asked = 0
        num_kg_nodes: int | None = None
        kg_attributes: dict[str, str] = {}
        det_latency = 0.0
        vqa_latency = 0.0
        meta_first: dict[str, Any] = {}

        if mode in ("two_pass_kg", "two_pass_kg_relaxed", "two_pass_kg_entropy"):
            # Pass 1: detection reasoning (same as kg mode)
            det_raw, det_latency, _, _, _ = _generate_chat(
                loader, pil_image, DETECTION_SYSTEM, det_prompt,
                max_new_tokens=det_tokens, output_scores=False,
            )
            num_model_calls += 1
            num_detector_calls = 1
            det_ans, det_think = extract_answer_and_reasoning(det_raw)
            detection_reasoning = det_think or det_ans
            kg_attributes, _extraction = build_kg_attributes_from_detection(detection_reasoning)

            # Pass 2: structured attribute extraction
            attr_category = "object"
            # Try to guess a category from the question
            import re as _re
            _cat_match = _re.search(r"(?:the|a|an)\s+(\w+(?:\s+\w+)?)\s*\?", question.lower())
            if _cat_match:
                attr_category = _cat_match.group(1).strip()

            if use_sequential_second_pass:
                assert second_pass_cfg is not None
                meta_first["sequential_second_pass_gpu"] = True
                print(
                    "[IDKVQA] Sequential GPU: unloading primary checkpoint for attribute pass, "
                    "then reloading primary for VQA.",
                    flush=True,
                )
                ModelLoader.reset(config.model)
                spl = ModelLoader.get_instance(second_pass_cfg)
                attr_attrs = TwoPassSelfQuestioner.run_attribute_pass_with_image(
                    spl,
                    pil_image,
                    category=attr_category,
                    timestep=0,
                    question_hint=question,
                    target_attr_type=attr_type,
                    existing_attributes=kg_attributes,
                )
                ModelLoader.reset(second_pass_cfg)
                loader = ModelLoader.get_instance(config.model)
            else:
                attr_attrs = TwoPassSelfQuestioner.run_attribute_pass_with_image(
                    second_pass_loader,
                    pil_image,
                    category=attr_category,
                    timestep=0,
                    question_hint=question,
                    target_attr_type=attr_type,
                    existing_attributes=kg_attributes,
                )
            num_model_calls += 1
            num_questioner_calls = 1
            meta_first["two_pass_attr_raw"] = {a.name: a.value for a in attr_attrs}
            meta_first["two_pass_attr_category"] = attr_category
            meta_first["two_pass_model_id"] = (
                second_pass_cfg.model_id if second_pass_cfg is not None else config.model.model_id
            )
            meta_first["two_pass_processor_id"] = (
                second_pass_cfg.processor_id if second_pass_cfg is not None else config.model.processor_id
            )

            # Merge attribute-pass results into kg_attributes
            for attr in attr_attrs:
                if attr.name not in kg_attributes:
                    kg_attributes[attr.name] = attr.value

            num_kg_nodes = len(kg_attributes)
            kg_strict = kg_answer_from_attributes(kg_attributes, attr_type, attr_value)

            # VQA pass
            vqa_result = _generate_chat_full(
                loader, pil_image, IDKVQA_SYSTEM, question,
                max_new_tokens=vqa_tokens, output_scores=True,
            )
            vqa_latency = vqa_result.gen_time
            entropy = vqa_result.entropy_first_token
            entropy_answer = vqa_result.entropy_answer_token
            max_prob = vqa_result.max_prob
            num_model_calls += 1
            num_questions_asked = 1
            raw_output = vqa_result.raw_output
            raw_answer, vqa_reasoning = extract_answer_and_reasoning(raw_output)
            raw_normalized = normalize_yes_no_idk(raw_answer)

            if mode == "two_pass_kg_relaxed":
                kg_hybrid = compute_kg_hybrid_prediction_relaxed(
                    raw_normalized,
                    vqa_reasoning,
                    kg_attributes,
                    attr_type,
                    attr_value,
                    detection_reasoning,
                )
            elif mode == "two_pass_kg_entropy":
                kg_hybrid = compute_kg_hybrid_prediction_entropy(
                    raw_normalized,
                    vqa_reasoning,
                    kg_attributes,
                    attr_type,
                    attr_value,
                    detection_reasoning,
                    entropy=entropy,
                    entropy_tau=tau,
                )
            else:
                kg_hybrid = compute_kg_hybrid_prediction(
                    raw_normalized,
                    vqa_reasoning,
                    kg_attributes,
                    attr_type,
                    attr_value,
                    detection_reasoning,
                )
        elif mode in ("kg", "kg_threshold"):
            det_raw, det_latency, _, _, _ = _generate_chat(
                loader, pil_image, DETECTION_SYSTEM, det_prompt,
                max_new_tokens=det_tokens, output_scores=False,
            )
            num_model_calls += 1
            num_detector_calls = 1
            det_ans, det_think = extract_answer_and_reasoning(det_raw)
            detection_reasoning = det_think or det_ans
            kg_attributes, _extraction = build_kg_attributes_from_detection(detection_reasoning)
            num_kg_nodes = len(kg_attributes)
            kg_strict = kg_answer_from_attributes(kg_attributes, attr_type, attr_value)

            vqa_result = _generate_chat_full(
                loader, pil_image, IDKVQA_SYSTEM, question,
                max_new_tokens=vqa_tokens, output_scores=True,
            )
            vqa_latency = vqa_result.gen_time
            entropy = vqa_result.entropy_first_token
            entropy_answer = vqa_result.entropy_answer_token
            max_prob = vqa_result.max_prob
            num_model_calls += 1
            num_questions_asked = 1
            raw_output = vqa_result.raw_output
            raw_answer, vqa_reasoning = extract_answer_and_reasoning(raw_output)
            raw_normalized = normalize_yes_no_idk(raw_answer)

            kg_hybrid = compute_kg_hybrid_prediction(
                raw_normalized,
                vqa_reasoning,
                kg_attributes,
                attr_type,
                attr_value,
                detection_reasoning,
            )
        elif mode in ("global_kg", "global_kg_entropy"):
            assert global_kg is not None
            image_key = stable_idkvqa_image_id(pil_image)
            kg_attributes = dict(global_kg.get_attributes_for_image(image_key))
            num_kg_nodes = len(kg_attributes)
            kg_strict = kg_answer_from_attributes(kg_attributes, attr_type, attr_value)

            vqa_result = _generate_chat_full(
                loader, pil_image, IDKVQA_SYSTEM, question,
                max_new_tokens=vqa_tokens, output_scores=True,
            )
            vqa_latency = vqa_result.gen_time
            entropy = vqa_result.entropy_first_token
            entropy_answer = vqa_result.entropy_answer_token
            max_prob = vqa_result.max_prob
            num_model_calls += 1
            num_questions_asked = 1
            raw_output = vqa_result.raw_output
            raw_answer, vqa_reasoning = extract_answer_and_reasoning(raw_output)
            raw_normalized = normalize_yes_no_idk(raw_answer)
            detection_reasoning = ""

            if mode == "global_kg_entropy":
                kg_hybrid = compute_kg_hybrid_prediction_entropy(
                    raw_normalized,
                    vqa_reasoning,
                    kg_attributes,
                    attr_type,
                    attr_value,
                    "",
                    entropy=entropy,
                    entropy_tau=tau,
                )
            else:
                kg_hybrid = compute_kg_hybrid_prediction(
                    raw_normalized,
                    vqa_reasoning,
                    kg_attributes,
                    attr_type,
                    attr_value,
                    "",
                )
            meta_first["global_kg_image_key"] = image_key
            meta_first["global_kg_path"] = global_kg_path
        elif mode == "raw_two_pass":
            det_raw, det_latency, _, _, _ = _generate_chat(
                loader, pil_image, DETECTION_SYSTEM, det_prompt,
                max_new_tokens=det_tokens, output_scores=False,
            )
            num_model_calls += 1
            num_detector_calls = 1
            det_ans, det_think = extract_answer_and_reasoning(det_raw)
            detection_reasoning = det_think or det_ans
            kg_strict = None
            kg_hybrid = None

            vqa_raw1, vqa_lat1, _, _, _ = _generate_chat(
                loader, pil_image, IDKVQA_SYSTEM, question,
                max_new_tokens=vqa_tokens, output_scores=False,
            )
            num_model_calls += 1
            num_questions_asked += 1
            raw_answer1, vqa_reasoning1 = extract_answer_and_reasoning(vqa_raw1)

            refine_user = _raw_two_pass_refine_prompt(question, raw_answer1)
            vqa_result2 = _generate_chat_full(
                loader, pil_image, IDKVQA_SYSTEM, refine_user,
                max_new_tokens=vqa_tokens, output_scores=True,
            )
            entropy = vqa_result2.entropy_first_token
            entropy_answer = vqa_result2.entropy_answer_token
            max_prob = vqa_result2.max_prob
            num_model_calls += 1
            num_questions_asked += 1
            vqa_latency = vqa_lat1 + vqa_result2.gen_time
            raw_output = vqa_result2.raw_output
            raw_answer, vqa_reasoning = extract_answer_and_reasoning(vqa_raw2)
            raw_normalized = normalize_yes_no_idk(raw_answer)
            meta_first = {
                "raw_two_pass_first_answer": raw_answer1,
                "raw_two_pass_first_reasoning": vqa_reasoning1,
                "raw_two_pass_vqa1_latency_sec": vqa_lat1,
                "raw_two_pass_vqa2_latency_sec": vqa_lat2,
            }
        else:  # raw, threshold
            vqa_result = _generate_chat_full(
                loader, pil_image, IDKVQA_SYSTEM, question,
                max_new_tokens=vqa_tokens, output_scores=need_scores,
            )
            vqa_latency = vqa_result.gen_time
            entropy = vqa_result.entropy_first_token
            entropy_answer = vqa_result.entropy_answer_token
            max_prob = vqa_result.max_prob
            num_model_calls = 1
            num_questions_asked = 1
            raw_output = vqa_result.raw_output
            raw_answer, vqa_reasoning = extract_answer_and_reasoning(raw_output)
            raw_normalized = normalize_yes_no_idk(raw_answer)
            kg_hybrid = None
            kg_strict = None

        if entropy is not None and entropy < 0:
            entropy = None

        reasoning_bucket = estimate_reasoning_certainty(vqa_reasoning or extract_answer_and_reasoning(raw_output)[1])
        uncertainty_score_used, uncertainty_source = _resolve_uncertainty_signal(
            rule=rule, entropy=entropy, max_prob=max_prob, reasoning_bucket=reasoning_bucket,
        )

        t_dec = time.perf_counter()
        final_label, used_kg, used_th, used_abs, abst_dec = finalize_for_mode(
            mode,
            raw_normalized=raw_normalized,
            uncertainty_score=uncertainty_score_used,
            threshold=tau,
            rule=rule,
            kg_hybrid=kg_hybrid,
        )
        decision_latency_sec = time.perf_counter() - t_dec

        total_latency = time.perf_counter() - t0
        conf: float | None = None
        if entropy is not None and entropy >= 0:
            conf = max(0.0, min(1.0, 1.0 - float(entropy)))
        elif max_prob is not None:
            conf = float(max_prob)

        uncertainty_score: float | None = float(entropy) if entropy is not None else None

        meta = {
            **meta_first,
            "reasoning_certainty": reasoning_bucket,
            "kg_strict_prediction": kg_strict,
            "kg_hybrid_prediction": kg_hybrid,
            "abstention": abst_dec.__dict__ if abst_dec else None,
            "entropy_threshold": tau,
            "abstention_rule": rule,
            "uncertainty_signal_source": uncertainty_source,
            "uncertainty_score_used": uncertainty_score_used,
            "entropy_answer_token": entropy_answer,
            "detection_latency_sec": det_latency,
            "vqa_latency_sec": vqa_latency,
            "max_token_prob": max_prob,
        }

        qtype = classify_question_type(sample["question"])
        abstained_flag = normalize_yes_no_idk(final_label) == LABEL_IDK

        results.append(
            _build_qa_result(
                sample=sample,
                mode=mode,
                raw_answer=raw_answer,
                raw_normalized=raw_normalized,
                final_prediction=final_label,
                confidence_score=conf,
                entropy_score=entropy,
                used_kg=used_kg,
                used_threshold=used_th,
                used_abstention=used_abs,
                latency_sec=total_latency,
                metadata=meta,
                question_type=qtype,
                num_model_calls=num_model_calls,
                num_detector_calls=num_detector_calls,
                num_questioner_calls=num_questioner_calls,
                num_trigger_calls=num_trigger_calls,
                num_questions_asked=num_questions_asked,
                num_kg_nodes=num_kg_nodes,
                total_latency_sec=total_latency,
                detector_latency_sec=det_latency if det_latency > 0 else None,
                decision_latency_sec=decision_latency_sec,
                uncertainty_score=uncertainty_score,
                threshold=tau if used_th else None,
                abstained=abstained_flag,
            )
        )

    return results


def save_idkvqa_benchmark_json(
    path: str | Path,
    config: Config,
    mode: str,
    seed: int,
    results: list[QAExampleResult],
    metrics: dict[str, Any],
) -> None:
    """Write full reproducibility bundle (config dict, per-sample rows, aggregates)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = benchmark_run_metadata(
        config_dict=config.to_serializable_dict(),
        mode=mode,
        seed=seed,
        model_id=config.model.model_id,
        processor_id=config.model.processor_id,
    )
    payload = {
        **meta,
        "metrics": metrics,
        "per_sample": [r.to_serializable() for r in results],
        "benchmark": "IDKVQA",
        "role": "primary_offline",
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Primary offline IDKVQA benchmark (uncertainty-aware VLM-R1)",
    )
    parser.add_argument("--config", type=str, default=None, help="YAML config (optional)")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--mode", type=str, choices=list(IDKVQA_MODES), required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--model_id", type=str, default=None)
    parser.add_argument("--processor_id", type=str, default=None)
    parser.add_argument("--second_pass_model_id", type=str, default=None)
    parser.add_argument("--second_pass_processor_id", type=str, default=None)
    parser.add_argument(
        "--export-normalization-audit",
        type=str,
        default=None,
        help="Write JSON audit sample of normalization (appendix / manual QA).",
    )
    parser.add_argument("--normalization-audit-size", type=int, default=50)
    parser.add_argument(
        "--normalization-audit-strategy",
        type=str,
        choices=("random", "first_mismatches", "ambiguous"),
        default="random",
    )
    parser.add_argument(
        "--export-paper-artifacts-dir",
        type=str,
        default=None,
        help="Write paper CSV/JSON slices (main/reliability/cost) for this single-mode run.",
    )
    parser.add_argument(
        "--export-threshold-sweep",
        type=str,
        default=None,
        help="If set, JSON path for entropy tau sweep rows (uses per-sample entropy from this run).",
    )
    parser.add_argument(
        "--global-kg-path",
        type=str,
        default=None,
        help="Path to pre-built global KG JSON (required for global_kg / global_kg_entropy).",
    )
    args = parser.parse_args()

    if args.config:
        cfg = Config.from_yaml(args.config)
    else:
        cfg = Config()
    if args.model_id:
        cfg.model.model_id = args.model_id
    if args.processor_id:
        cfg.model.processor_id = args.processor_id

    if args.mode in ("global_kg", "global_kg_entropy") and not args.global_kg_path:
        parser.error("--global-kg-path is required for modes global_kg and global_kg_entropy")

    results = run_idkvqa_benchmark(
        cfg,
        mode=args.mode,
        limit=args.limit,
        seed=args.seed,
        split=args.split,
        second_pass_model_id=args.second_pass_model_id,
        second_pass_processor_id=args.second_pass_processor_id,
        global_kg_path=args.global_kg_path,
    )
    metrics = aggregate_idkvqa_metrics(results)
    save_idkvqa_benchmark_json(args.output, cfg, args.mode, args.seed, results, metrics)

    if args.export_normalization_audit:
        from .normalization_audit import export_normalization_audit

        export_normalization_audit(
            results,
            args.export_normalization_audit,
            n=args.normalization_audit_size,
            strategy=args.normalization_audit_strategy,
            seed=args.seed,
        )
        print(f"Wrote normalization audit {args.export_normalization_audit}")

    if args.export_paper_artifacts_dir:
        from .paper_artifacts import export_mode_tables

        export_mode_tables({args.mode: results}, args.export_paper_artifacts_dir)

    if args.export_threshold_sweep:
        from .threshold_sweep import default_tau_grid, sweep_entropy_threshold

        sweep_rows = sweep_entropy_threshold(results, default_tau_grid(0.05))
        p = Path(args.export_threshold_sweep)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(sweep_rows, f, indent=2, ensure_ascii=False, default=str)
        print(f"Wrote threshold sweep {args.export_threshold_sweep}")

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
