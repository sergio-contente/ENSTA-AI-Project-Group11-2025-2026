"""kg_trigger.py -- KG-based Interaction Trigger. 0 LLM calls."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..config import TriggerConfig
from ..self_questioner.base import RefinedDescription
from ..knowledge_graph.schema import TargetFacts
from ..knowledge_graph.scene_graph import SceneKnowledgeGraph
from ..knowledge_graph.graph_matcher import GraphMatcher
from ..knowledge_graph.question_generator import QuestionGenerator
from .base import AbstractInteractionTrigger, TriggerAction, ActionType


class KGInteractionTrigger(AbstractInteractionTrigger):
    def __init__(
        self,
        config: TriggerConfig,
        vlm_judge_fn: Callable[[str, str], bool] | None = None,
        loader: Any = None,
    ):
        self._tau_stop = config.tau_stop
        self._tau_skip = config.tau_skip
        self._vlm_judge_fn = vlm_judge_fn
        self._loader = loader

    def decide(self, description: RefinedDescription, target_facts: TargetFacts,
               kg: SceneKnowledgeGraph) -> TriggerAction:
        if not description.is_valid or description.object_node is None:
            return TriggerAction(type=ActionType.CONTINUE, reason="Invalid detection")
        node = description.object_node
        score = GraphMatcher.compute_alignment_with_vlm_fallback(
            node,
            target_facts,
            tau_stop=self._tau_stop,
            vlm_judge_fn=getattr(self, "_vlm_judge_fn", None),
            loader=getattr(self, "_loader", None),
            detected_crop=getattr(node, "detected_crop", None),
        )
        node.alignment_score = score
        explanation = GraphMatcher.explain_alignment(
            node, target_facts, loader=getattr(self, "_loader", None)
        )
        contradictions = explanation.get("contradictions", [])
        if contradictions:
            return TriggerAction(
                type=ActionType.CONTINUE,
                alignment_score=0.0,
                reason=f"Contradictions: {contradictions}",
                alignment_explanation=explanation,
            )
        if score >= self._tau_stop:
            return TriggerAction(
                type=ActionType.STOP,
                alignment_score=score,
                reason=f"Score {score:.2f} >= tau_stop",
                alignment_explanation=explanation,
            )
        if score != -1.0 and score < self._tau_skip:
            return TriggerAction(
                type=ActionType.CONTINUE,
                alignment_score=score,
                reason=f"Score {score:.2f} < tau_skip",
                alignment_explanation=explanation,
            )
        question = QuestionGenerator.generate(node, kg)
        kg.target_facts.record_question(question)
        return TriggerAction(
            type=ActionType.ASK,
            question=question,
            alignment_score=score,
            reason=f"Score {score:.2f}, asking user",
            alignment_explanation=explanation,
        )
