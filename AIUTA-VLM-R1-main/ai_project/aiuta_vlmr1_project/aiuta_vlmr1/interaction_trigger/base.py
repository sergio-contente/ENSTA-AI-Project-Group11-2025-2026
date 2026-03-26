"""base.py -- Abstract Interaction Trigger interface."""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..self_questioner.base import RefinedDescription
from ..knowledge_graph.schema import TargetFacts
from ..knowledge_graph.scene_graph import SceneKnowledgeGraph

class ActionType(Enum):
    STOP = "stop"
    CONTINUE = "continue"
    ASK = "ask"

@dataclass
class TriggerAction:
    type: ActionType
    question: str | None = None
    alignment_score: float = -1.0
    reason: str = ""
    alignment_explanation: dict[str, Any] = field(default_factory=dict)

class AbstractInteractionTrigger(ABC):
    @abstractmethod
    def decide(self, description: RefinedDescription, target_facts: TargetFacts,
               kg: SceneKnowledgeGraph) -> TriggerAction:
        ...
