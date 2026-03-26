"""base.py -- Abstract Self-Questioner interface."""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from ..detector.base import Detection
from ..knowledge_graph.schema import ObjectNode, TargetFacts
from ..knowledge_graph.scene_graph import SceneKnowledgeGraph

@dataclass
class RefinedDescription:
    object_node: ObjectNode | None
    text_description: str
    is_valid: bool

class AbstractSelfQuestioner(ABC):
    @abstractmethod
    def process(self, detection: Detection, target_facts: TargetFacts,
                kg: SceneKnowledgeGraph, timestep: int = 0) -> RefinedDescription:
        ...
