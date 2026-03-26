"""schema.py -- Data model for the Knowledge Graph."""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum

class Certainty(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"
    def __ge__(self, other):
        order = {self.UNKNOWN: 0, self.LOW: 1, self.MEDIUM: 2, self.HIGH: 3}
        return order[self] >= order[other]
    def __gt__(self, other):
        order = {self.UNKNOWN: 0, self.LOW: 1, self.MEDIUM: 2, self.HIGH: 3}
        return order[self] > order[other]

class AttributeSource(Enum):
    VLM_REASONING = "vlm_reasoning"
    USER_RESPONSE = "user_response"
    INFERRED = "inferred"

@dataclass
class Attribute:
    name: str
    value: str
    certainty: Certainty = Certainty.MEDIUM
    source: AttributeSource = AttributeSource.VLM_REASONING
    timestep: int = 0
    def to_dict(self): return {"name": self.name, "value": self.value, "certainty": self.certainty.value}

@dataclass
class SpatialRelation:
    relation: str
    reference: str
    certainty: Certainty = Certainty.MEDIUM
    timestep: int = 0
    def to_dict(self): return {"relation": self.relation, "reference": self.reference}

@dataclass
class ObjectNode:
    obj_id: str
    category: str
    bbox: list[float] | None = None
    image_id: str | None = None
    timestep_first: int = 0
    timestep_last: int = 0
    attributes: dict[str, Attribute] = field(default_factory=dict)
    spatial_relations: list[SpatialRelation] = field(default_factory=list)
    is_target_candidate: bool = False
    alignment_score: float = -1.0
    detected_crop: object = None  # np.ndarray | None -- RGB crop of detected object

    def get_attribute_value(self, name: str) -> str | None:
        attr = self.attributes.get(name)
        return attr.value if attr else None

    def has_attribute(self, name: str) -> bool:
        return name in self.attributes

    def to_natural_language(self) -> str:
        parts = [f"{self.obj_id} ({self.category})"]
        for name, attr in self.attributes.items():
            if attr.certainty >= Certainty.MEDIUM:
                parts.append(f"{name}={attr.value}")
        for rel in self.spatial_relations:
            if rel.certainty >= Certainty.MEDIUM:
                parts.append(f"{rel.relation} {rel.reference}")
        return ", ".join(parts)

    def to_dict(self):
        return {"obj_id": self.obj_id, "category": self.category,
                "attributes": {k: v.to_dict() for k, v in self.attributes.items()}}

def _normalize_fact_value(attr_name: str, value: str) -> str:
    v = value.strip().lower()
    v = " ".join(v.split())
    return v


@dataclass
class TargetFacts:
    category: str = ""
    known_attributes: dict[str, str] = field(default_factory=dict)
    negative_attributes: dict[str, str] = field(default_factory=dict)
    source_history: list[str] = field(default_factory=list)
    fact_provenance: dict[str, str] = field(default_factory=dict)
    asked_questions: list[str] = field(default_factory=list)

    def add_positive(self, attr_name: str, attr_value: str, source: str = "user"):
        nv = _normalize_fact_value(attr_name, attr_value)
        self.known_attributes[attr_name] = nv
        self.fact_provenance[attr_name] = source
        self.source_history.append(f"+{attr_name}={nv}|{source}")

    def add_negative(self, attr_name: str, attr_value: str, source: str = "user"):
        nv = _normalize_fact_value(attr_name, attr_value)
        self.negative_attributes[attr_name] = nv
        self.fact_provenance[f"NOT_{attr_name}"] = source
        self.source_history.append(f"-{attr_name}={nv}|{source}")

    def record_question(self, question: str) -> None:
        q = question.strip()
        if q and q not in self.asked_questions:
            self.asked_questions.append(q)

    def to_natural_language(self) -> str:
        parts = [f"Target: {self.category}"]
        for k, v in self.known_attributes.items(): parts.append(f"  {k}: {v}")
        for k, v in self.negative_attributes.items(): parts.append(f"  NOT {k}: {v}")
        return "\n".join(parts)

    @property
    def num_facts(self) -> int:
        return len(self.known_attributes) + len(self.negative_attributes)
