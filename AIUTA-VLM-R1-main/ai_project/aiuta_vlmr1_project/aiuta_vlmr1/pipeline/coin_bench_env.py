"""
CoINBenchEnv -- adapts one CoIN-Bench episode to the interface expected by
``entropy_coin_agent.run_entropy_coin_episode`` (and related offline runners).

Offline static proxy:
  - Observations are metadata-linked image candidates (no Habitat simulation).
  - ``move_forward`` advances the candidate index; turns are no-ops for imagery.
  - ``shortest_path_length`` is fixed to 1.0 for aggregated SPL (approximation; see ``coin_metrics``).
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image

from ..evaluation.answer_normalization import LABEL_YES, normalize_yes_no_idk
from ..evaluation.vlm_inference_utils import extract_answer_and_reasoning
from ..evaluation.coin_loader import CoINBenchLoader, CoINEpisode
from ..knowledge_graph.schema import TargetFacts


class CoINBenchEnv:
    """Wraps a single CoIN-Bench episode for the entropy agent loop."""

    def __init__(
        self,
        episode: CoINEpisode,
        coin_loader: CoINBenchLoader | None,
        image_candidates: list[Path | str],
        target_facts: TargetFacts,
    ):
        self._episode = episode
        self._coin_loader = coin_loader
        self._paths: list[Path] = [Path(p).resolve() for p in image_candidates]
        self._target_facts = target_facts
        self._idx: int = 0
        self._nav_steps: int = 0
        self._cache: dict[str, Image.Image] = {}

    def _pil_for_path(self, path: Path) -> Image.Image:
        key = str(path)
        if key not in self._cache:
            self._cache[key] = Image.open(path).convert("RGB")
        return self._cache[key]

    def get_observation(self) -> Image.Image:
        """Return the current observation (current candidate image)."""
        if not self._paths:
            raise RuntimeError("CoINBenchEnv has no image candidates")
        p = self._paths[min(self._idx, len(self._paths) - 1)]
        return self._pil_for_path(p)

    @property
    def current_image_path(self) -> Path:
        if not self._paths:
            raise RuntimeError("CoINBenchEnv has no image candidates")
        return self._paths[min(self._idx, len(self._paths) - 1)]

    def step(self, action: str) -> None:
        """Simulate a navigation action; counts toward path length for the offline proxy."""
        self._nav_steps += 1
        a = str(action).strip().lower()
        if a in ("move_forward", "move_fwd", "forward"):
            if len(self._paths) > 1:
                self._idx = min(self._idx + 1, len(self._paths) - 1)

    def evaluate_commit(self, answer: str) -> bool:
        """
        Success if the model commits to **Yes** under IDKVQA-style normalization.

        Target facts anchor the intended target (category + description); for this static
        VQA proxy, ground-truth alignment is **Yes** = correct stop.
        """
        extracted, _reason = extract_answer_and_reasoning(answer)
        label = normalize_yes_no_idk(extracted)
        return label == LABEL_YES

    @property
    def episode_id(self) -> str:
        return str(self._episode.episode_id)

    @property
    def target_category(self) -> str:
        return str(self._episode.target_category or "")

    @property
    def exploration_steps(self) -> int:
        """Navigation actions executed (excludes the implicit commit step)."""
        return self._nav_steps

    @property
    def path_length(self) -> float:
        """Exploration steps plus one commit step (offline SPL proxy)."""
        return float(self._nav_steps + 1)

    @property
    def shortest_path_length(self) -> float:
        return 1.0

    @property
    def target_facts(self) -> TargetFacts:
        return self._target_facts

    @property
    def episode(self) -> CoINEpisode:
        return self._episode

    @property
    def image_paths(self) -> list[Path]:
        return list(self._paths)


def target_facts_from_coin_episode(episode: CoINEpisode) -> TargetFacts:
    """Build ``TargetFacts`` from CoIN episode metadata."""
    tf = TargetFacts(category=(episode.target_category or "").strip())
    desc = (episode.target_description or "").strip()
    if desc:
        tf.add_positive("target_description", desc[:2000], "coin_episode")
    return tf


def coin_vqa_question(episode: CoINEpisode) -> str:
    cat = episode.target_category or "object"
    desc = (episode.target_description or "").strip()
    return (
        f"Is this the {cat}? {desc}\n"
        "Answer with exactly Yes, No, or I don't know inside <answer></answer> tags."
    )
