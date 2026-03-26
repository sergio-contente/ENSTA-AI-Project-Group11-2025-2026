"""Shared VLM inference utilities for offline benchmarks (entropy, timing)."""
from __future__ import annotations

import math
import re
import time

import torch


def compute_first_token_entropy(outputs, vocab_size: int | None = None) -> float:
    """
    Normalized Shannon entropy of the first generated token (CoIN-style).
    Returns float in [0, 1], or -1.0 if unavailable.
    """
    if not hasattr(outputs, "scores") or not outputs.scores:
        return -1.0
    first_logits = outputs.scores[0][0]
    probs = torch.softmax(first_logits.float(), dim=-1)
    log_probs = torch.log2(probs + 1e-10)
    entropy = -(probs * log_probs).sum().item()
    vs = probs.shape[0] if vocab_size is None else vocab_size
    max_entropy = math.log2(vs)
    return entropy / max_entropy if max_entropy > 0 else 0.0


def estimate_reasoning_certainty(reasoning: str) -> str:
    """Coarse certainty bucket from reasoning text."""
    reasoning_lower = reasoning.lower()
    hedging = [
        "appears", "seems", "might", "possibly", "likely",
        "probably", "could be", "looks like", "may be",
        "i'm not sure", "it's hard to tell", "difficult to determine",
        "unclear", "cannot determine", "not certain",
    ]
    confident = [
        "clearly", "definitely", "certainly", "obviously",
        "it is", "i can see", "the image shows",
    ]
    n_hedge = sum(1 for h in hedging if h in reasoning_lower)
    n_confident = sum(1 for c in confident if c in reasoning_lower)
    if n_hedge >= 2:
        return "low"
    if n_hedge >= 1 and n_confident == 0:
        return "low"
    if n_confident >= 2:
        return "high"
    return "medium"


def extract_answer_and_reasoning(raw_output: str) -> tuple[str, str]:
    """Parse <answer> and </think> blocks from VLM-R1 style output."""
    ans_m = re.search(r"<answer>(.*?)</answer>", raw_output, flags=re.DOTALL | re.IGNORECASE)
    raw_answer = ans_m.group(1).strip() if ans_m else raw_output.strip()
    think_m = re.search(r"<think>(.*?)</think>", raw_output, flags=re.DOTALL | re.IGNORECASE)
    reasoning = think_m.group(1).strip() if think_m else ""
    return raw_answer, reasoning


def model_first_device(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


def compute_first_token_max_prob(outputs) -> float | None:
    """Max softmax probability of the first generated token, in (0, 1], or None if unavailable."""
    if not hasattr(outputs, "scores") or not outputs.scores:
        return None
    first_logits = outputs.scores[0][0]
    probs = torch.softmax(first_logits.float(), dim=-1)
    return float(probs.max().item())


# ---------------------------------------------------------------------------
# Answer-token entropy (shared with entropy_coin_agent)
# ---------------------------------------------------------------------------

def compute_logits_entropy(logits: torch.Tensor, vocab_size: int | None = None) -> float:
    """Normalized Shannon entropy over a single-token logits vector."""
    probs = torch.softmax(logits.float(), dim=-1)
    log_probs = torch.log2(probs + 1e-10)
    entropy = -(probs * log_probs).sum().item()
    vs = probs.shape[0] if vocab_size is None else int(vocab_size)
    max_entropy = torch.log2(torch.tensor(float(vs))).item() if vs > 0 else 1.0
    return float(entropy / max_entropy) if max_entropy > 0 else 0.0


def _find_subsequence_index(seq: list[int], sub: list[int]) -> int | None:
    """Return start index of ``sub`` in ``seq``, or None."""
    if not sub or len(sub) > len(seq):
        return None
    for i in range(len(seq) - len(sub) + 1):
        if seq[i : i + len(sub)] == sub:
            return i
    return None


def _answer_token_score_index(
    generated_ids: list[int],
    tokenizer: object | None,
) -> int | None:
    """
    Return index of the first answer token *after* ``<answer>``.

    The index matches ``outputs.scores[index]``.
    """
    if tokenizer is None or not hasattr(tokenizer, "encode"):
        return None

    candidates = []
    for marker in ("<answer>", " <answer>"):
        try:
            marker_ids = tokenizer.encode(marker, add_special_tokens=False)
        except TypeError:
            marker_ids = tokenizer.encode(marker)
        if marker_ids:
            candidates.append(marker_ids)

    for marker_ids in candidates:
        start = _find_subsequence_index(generated_ids, list(marker_ids))
        if start is not None:
            idx = start + len(marker_ids)
            if idx < len(generated_ids):
                return idx
    return None


def compute_answer_token_entropy(
    outputs: object,
    processor: object,
    generated_ids: list[int],
) -> tuple[float, torch.Tensor | None]:
    """
    Entropy on the first token of the final answer (after ``<answer>``).

    Falls back to first generated token if ``<answer>`` marker is not found.
    """
    scores = getattr(outputs, "scores", None)
    if not scores:
        return -1.0, None

    tokenizer = getattr(processor, "tokenizer", None)
    idx = _answer_token_score_index(generated_ids, tokenizer)
    if idx is None or idx >= len(scores):
        idx = 0

    raw_logits = scores[idx][0]
    logits = raw_logits.detach().cpu()
    vocab_size = getattr(tokenizer, "vocab_size", None)
    entropy = compute_logits_entropy(logits, vocab_size=vocab_size)
    return float(entropy), logits
