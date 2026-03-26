"""
Minimal entropy-first embodied loop for CoIN-style navigation.

Core invariants:
- one VLM forward pass per step
- entropy is the only decision signal
- no KG gating or multi-pass answer fusion
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch

from ..evaluation.vlm_inference_utils import (
    compute_answer_token_entropy,
    compute_logits_entropy,
    extract_answer_and_reasoning,
)


@dataclass
class EntropyStepLog:
    timestep: int
    entropy: float
    previous_entropy: float | None
    delta_entropy: float | None
    best_entropy_so_far: float
    answer: str
    action_type: str
    action: str | None


@dataclass
class EntropyEpisodeResult:
    committed: bool
    final_answer: str | None
    success: bool | None
    steps: int
    final_entropy: float | None
    entropy_trajectory: list[float]
    step_logs: list[EntropyStepLog]


def _generate_with_scores(
    image: Any,
    question: str,
    model: Any,
    processor: Any,
    max_new_tokens: int = 64,
    system_prompt: str | None = None,
) -> tuple[str, float, Any]:
    """Generate answer text and answer-token entropy."""
    system = system_prompt or (
        "You are a helpful assistant that answers visual questions. "
        "Use <think></think> for reasoning and <answer></answer> for final answer."
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": question}]},
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt")

    device = next(model.parameters()).device
    inputs = inputs.to(device)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True,
        )

    trimmed = [o[len(inp):] for inp, o in zip(inputs.input_ids, outputs.sequences)]
    raw_output = processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False,
    )[0]
    answer, _reasoning = extract_answer_and_reasoning(raw_output)
    generated_ids = list(trimmed[0].tolist()) if trimmed else []
    entropy, logits = compute_answer_token_entropy(outputs, processor, generated_ids)
    return answer, float(entropy), logits


def vlm_vqa_with_entropy(
    image: Any,
    question: str,
    model: Any,
    processor: Any,
    history: list[Any] | None = None,
    max_new_tokens: int = 64,
    system_prompt: str | None = None,
) -> tuple[str, float, Any]:
    """
    Returns:
        answer: str
        entropy: float (normalized answer-token entropy)
        logits: first-token logits tensor or None
    """
    if history:
        # Optional context history; keeps single generation call.
        history_text = f"Context includes {len(history)} previous observations."
        question = f"{history_text}\n{question}"
    return _generate_with_scores(
        image=image,
        question=question,
        model=model,
        processor=processor,
        max_new_tokens=max_new_tokens,
        system_prompt=system_prompt,
    )


def decide_action(entropy: float, tau: float = 0.10) -> str:
    return "explore" if entropy > tau else "commit"


def random_exploration_policy(_obs: Any, rng: Any) -> str:
    """Simple random move/rotation policy for first embodied version."""
    actions = ["move_forward", "turn_left", "turn_right", "look_up", "look_down"]
    return str(rng.choice(actions))


def entropy_guided_exploration_policy(
    obs: Any,
    rng: Any,
    *,
    current_entropy: float,
    previous_entropy: float | None,
    previous_action: str | None,
) -> str:
    """
    Keep direction when entropy improves; otherwise switch action.
    """
    if previous_entropy is None or previous_action is None:
        return random_exploration_policy(obs, rng)
    if current_entropy < previous_entropy:
        return previous_action
    actions = ["move_forward", "turn_left", "turn_right", "look_up", "look_down"]
    alternatives = [a for a in actions if a != previous_action]
    return str(rng.choice(alternatives or actions))


def run_entropy_coin_episode(
    env: Any,
    question: str,
    model: Any,
    processor: Any,
    *,
    tau: float = 0.10,
    max_steps: int = 20,
    max_new_tokens: int = 64,
    exploration_policy: Callable[..., str] | None = None,
    rng: Any | None = None,
    stagnation_patience: int = 3,
    history_size: int = 0,
) -> EntropyEpisodeResult:
    """
    Entropy-first CoIN-style loop:
    - high entropy -> explore
    - low entropy -> commit answer
    """
    import random

    local_rng = rng if rng is not None else random.Random(42)
    explore = exploration_policy or entropy_guided_exploration_policy
    logs: list[EntropyStepLog] = []
    entropy_traj: list[float] = []
    obs_history: list[Any] = []
    previous_entropy: float | None = None
    previous_action: str | None = None
    best_answer: str | None = None
    best_entropy: float = float("inf")
    non_improving_steps = 0

    for step in range(max_steps):
        obs = env.get_observation()
        history = obs_history[-history_size:] if history_size > 0 else None
        answer, entropy, _logits = vlm_vqa_with_entropy(
            obs, question, model, processor, history=history, max_new_tokens=max_new_tokens,
        )
        if entropy < best_entropy:
            best_entropy = entropy
            best_answer = answer
            non_improving_steps = 0
        else:
            non_improving_steps += 1

        delta_entropy = None if previous_entropy is None else (entropy - previous_entropy)
        action_type = decide_action(entropy, tau=tau)
        entropy_traj.append(entropy)

        should_force_commit = step == (max_steps - 1) or non_improving_steps >= stagnation_patience
        commit_answer = best_answer if best_answer is not None else answer
        if action_type == "commit" or (should_force_commit and commit_answer is not None):
            success = env.evaluate_commit(commit_answer) if hasattr(env, "evaluate_commit") else None
            logs.append(
                EntropyStepLog(
                    timestep=step,
                    entropy=entropy,
                    previous_entropy=previous_entropy,
                    delta_entropy=delta_entropy,
                    best_entropy_so_far=best_entropy,
                    answer=commit_answer,
                    action_type="commit",
                    action=None,
                )
            )
            return EntropyEpisodeResult(
                committed=True,
                final_answer=commit_answer,
                success=success,
                steps=step + 1,
                final_entropy=best_entropy if commit_answer == best_answer else entropy,
                entropy_trajectory=entropy_traj,
                step_logs=logs,
            )

        action = explore(
            obs,
            local_rng,
            current_entropy=entropy,
            previous_entropy=previous_entropy,
            previous_action=previous_action,
        )
        env.step(action)
        logs.append(
            EntropyStepLog(
                timestep=step,
                entropy=entropy,
                previous_entropy=previous_entropy,
                delta_entropy=delta_entropy,
                best_entropy_so_far=best_entropy,
                answer=answer,
                action_type="explore",
                action=action,
            )
        )
        previous_entropy = entropy
        previous_action = action
        if history_size > 0:
            obs_history.append(obs)

    return EntropyEpisodeResult(
        committed=False,
        final_answer=best_answer,
        success=False if hasattr(env, "evaluate_commit") else None,
        steps=max_steps,
        final_entropy=best_entropy if best_entropy != float("inf") else None,
        entropy_trajectory=entropy_traj,
        step_logs=logs,
    )


def aggregate_entropy_coin_metrics(episodes: list[EntropyEpisodeResult]) -> dict[str, float | int | None]:
    if not episodes:
        return {
            "num_episodes": 0,
            "success_rate": 0.0,
            "avg_steps_to_answer": 0.0,
            "final_entropy": None,
            "entropy_reduction_over_time": 0.0,
            "avg_entropy_reduction_per_step": 0.0,
            "success_given_low_entropy": 0.0,
            "entropy_at_commit_distribution": [],
        }

    finals = [e.final_entropy for e in episodes if e.final_entropy is not None]
    step_counts = [e.steps for e in episodes]
    known_success = [e.success for e in episodes if e.success is not None]
    success_rate = (sum(1 for s in known_success if s) / len(known_success)) if known_success else 0.0

    reductions = []
    for e in episodes:
        if len(e.entropy_trajectory) >= 2:
            reductions.append(e.entropy_trajectory[0] - e.entropy_trajectory[-1])
        else:
            reductions.append(0.0)

    reduction_per_step = []
    for e in episodes:
        if len(e.entropy_trajectory) >= 2:
            total_red = e.entropy_trajectory[0] - e.entropy_trajectory[-1]
            reduction_per_step.append(total_red / max(1, len(e.entropy_trajectory) - 1))
        else:
            reduction_per_step.append(0.0)

    commit_entropies = [e.final_entropy for e in episodes if e.committed and e.final_entropy is not None]
    low_entropy_eps = [e for e in episodes if e.final_entropy is not None and e.final_entropy <= 0.10]
    low_entropy_success = [e for e in low_entropy_eps if e.success is not None]
    success_given_low_entropy = (
        sum(1 for e in low_entropy_success if e.success) / len(low_entropy_success)
        if low_entropy_success else 0.0
    )

    return {
        "num_episodes": len(episodes),
        "success_rate": float(success_rate),
        "avg_steps_to_answer": float(sum(step_counts) / len(step_counts)),
        "final_entropy": float(sum(finals) / len(finals)) if finals else None,
        "entropy_reduction_over_time": float(sum(reductions) / len(reductions)),
        "avg_entropy_reduction_per_step": float(sum(reduction_per_step) / len(reduction_per_step)),
        "success_given_low_entropy": float(success_given_low_entropy),
        "entropy_at_commit_distribution": [float(v) for v in commit_entropies],
    }
