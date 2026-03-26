from __future__ import annotations

from types import SimpleNamespace

import torch

from aiuta_vlmr1.pipeline.entropy_coin_agent import (
    aggregate_entropy_coin_metrics,
    compute_answer_token_entropy,
    decide_action,
    entropy_guided_exploration_policy,
    run_entropy_coin_episode,
    vlm_vqa_with_entropy,
)


class _DummyInputs(dict):
    def to(self, _device):
        return self

    @property
    def input_ids(self):
        return self["input_ids"]


class _FakeProcessor:
    def __init__(self):
        # token map for tests:
        # <answer> -> [90, 91], "Yes" starts at 99
        self.tokenizer = SimpleNamespace(
            vocab_size=4,
            encode=lambda s, add_special_tokens=False: [90, 91] if "<answer>" in s else [],
        )

    def apply_chat_template(self, _messages, tokenize=False, add_generation_prompt=True):
        assert tokenize is False
        assert add_generation_prompt is True
        return "prompt"

    def __call__(self, **_kwargs):
        return _DummyInputs({"input_ids": [torch.tensor([11, 12])]})

    def batch_decode(self, _trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False):
        assert skip_special_tokens is True
        assert clean_up_tokenization_spaces is False
        return ["<think>ok</think><answer>Yes</answer>"]


class _FakeModel:
    def __init__(self):
        self._p = torch.nn.Parameter(torch.zeros(1))

    def parameters(self):
        yield self._p

    def generate(self, **_kwargs):
        # scores[2] is the first token after "<answer>" in generated ids [90,91,99]
        low_entropy_logits = torch.tensor([[8.0, -8.0, -8.0, -8.0]])
        high_entropy_logits = torch.tensor([[0.0, 0.0, 0.0, 0.0]])
        return SimpleNamespace(
            sequences=[torch.tensor([11, 12, 90, 91, 99])],
            scores=[high_entropy_logits, high_entropy_logits, low_entropy_logits],
        )


class _FakeEnv:
    def __init__(self, observations: list[str], success_answer: str = "Yes"):
        self.observations = observations
        self.idx = 0
        self.actions: list[str] = []
        self.success_answer = success_answer

    def get_observation(self):
        obs = self.observations[min(self.idx, len(self.observations) - 1)]
        self.idx += 1
        return obs

    def step(self, action: str):
        self.actions.append(action)
        return None

    def evaluate_commit(self, answer: str) -> bool:
        return answer == self.success_answer


def test_decide_action_threshold():
    assert decide_action(0.20, tau=0.10) == "explore"
    assert decide_action(0.09, tau=0.10) == "commit"


def test_vlm_vqa_with_entropy_returns_normalized_entropy():
    answer, entropy, logits = vlm_vqa_with_entropy(
        image="img",
        question="Is there a chair?",
        model=_FakeModel(),
        processor=_FakeProcessor(),
    )
    assert answer == "Yes"
    assert entropy < 0.01
    assert logits is not None


def test_compute_answer_token_entropy_uses_answer_marker():
    outputs = SimpleNamespace(
        scores=[
            torch.tensor([[0.0, 0.0, 0.0, 0.0]]),
            torch.tensor([[0.0, 0.0, 0.0, 0.0]]),
            torch.tensor([[8.0, -8.0, -8.0, -8.0]]),
        ]
    )
    processor = _FakeProcessor()
    entropy, _logits = compute_answer_token_entropy(outputs, processor, [90, 91, 99])
    assert entropy < 0.01


def test_run_entropy_episode_commits_when_low_entropy(monkeypatch):
    env = _FakeEnv(["obs0"])

    def _stub_vqa(*_args, **_kwargs):
        return "Yes", 0.05, None

    monkeypatch.setattr(
        "aiuta_vlmr1.pipeline.entropy_coin_agent.vlm_vqa_with_entropy",
        _stub_vqa,
    )

    ep = run_entropy_coin_episode(
        env=env,
        question="Is there a chair?",
        model=object(),
        processor=object(),
        tau=0.10,
        max_steps=3,
    )
    assert ep.committed is True
    assert ep.final_answer == "Yes"
    assert ep.success is True
    assert ep.steps == 1
    assert len(ep.step_logs) == 1
    assert ep.step_logs[0].action_type == "commit"


def test_run_entropy_episode_explores_when_high_entropy(monkeypatch):
    env = _FakeEnv(["obs0", "obs1", "obs2"])

    def _stub_vqa(*_args, **_kwargs):
        return "Maybe", 0.90, None

    monkeypatch.setattr(
        "aiuta_vlmr1.pipeline.entropy_coin_agent.vlm_vqa_with_entropy",
        _stub_vqa,
    )

    ep = run_entropy_coin_episode(
        env=env,
        question="Is there a chair?",
        model=object(),
        processor=object(),
        tau=0.10,
        max_steps=3,
        stagnation_patience=10,
    )
    assert ep.committed is True
    assert ep.steps == 3
    assert len(env.actions) == 2
    assert [log.action_type for log in ep.step_logs] == ["explore", "explore", "commit"]


def test_entropy_guided_policy_keeps_action_on_improvement():
    rng = SimpleNamespace(choice=lambda xs: xs[0])
    a = entropy_guided_exploration_policy(
        "obs",
        rng,
        current_entropy=0.2,
        previous_entropy=0.4,
        previous_action="turn_left",
    )
    assert a == "turn_left"


def test_entropy_metrics_schema():
    episodes = [
        SimpleNamespace(
            committed=True,
            success=True,
            steps=2,
            final_entropy=0.2,
            entropy_trajectory=[0.7, 0.2],
        ),
        SimpleNamespace(
            committed=True,
            success=False,
            steps=3,
            final_entropy=0.4,
            entropy_trajectory=[0.6, 0.5, 0.4],
        ),
    ]
    m = aggregate_entropy_coin_metrics(episodes)  # type: ignore[arg-type]
    assert m["num_episodes"] == 2
    assert "success_rate" in m
    assert "avg_steps_to_answer" in m
    assert "final_entropy" in m
    assert "entropy_reduction_over_time" in m
    assert "avg_entropy_reduction_per_step" in m
    assert "success_given_low_entropy" in m
    assert "entropy_at_commit_distribution" in m
