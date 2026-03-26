"""
triple_extractor.py -- Extract structured triples from VLM-R1 reasoning.

Refactored from analyze_reasoning.py pattern detectors:
  - detect_filtering() -> extract_absence()
  - detect_scene_description() + detect_attribute_verification() -> extract_attributes()
  - spatial_terms list -> extract_spatial()
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from .schema import Attribute, SpatialRelation, Certainty, AttributeSource

NEGATION_PHRASES = [
    "not present", "not visible", "not in", "not found",
    "no indication", "no sign", "cannot see",
    "don't see", "do not see", "is not", "are not",
    "absent", "missing", "none of", "no evidence",
    "doesn't contain", "does not contain",
    "not detected", "not observed", "irrelevant",
]

COLOR_WORDS = [
    "red", "blue", "green", "black", "white", "yellow",
    "brown", "gray", "grey", "orange", "pink", "purple",
    "bright", "dark", "light", "beige", "cream", "tan",
]

MATERIAL_WORDS = [
    "metal", "metallic", "wooden", "wood", "plastic", "glass",
    "leather", "fabric", "ceramic", "stone", "marble",
    "stainless", "steel", "aluminum", "chrome",
]

SIZE_WORDS = [
    "large", "small", "big", "tiny", "medium", "huge",
    "compact", "tall", "short", "wide", "narrow",
]

SPATIAL_TERMS = [
    "left of", "right of", "above", "below", "near",
    "next to", "beside", "behind", "in front of",
    "on top of", "underneath", "between",
    "in the corner", "against the wall",
    "in the kitchen", "in the bedroom", "in the living room",
    "in the bathroom", "in the hallway",
]

HEDGING_WORDS = [
    "appears", "seems", "might", "possibly", "likely",
    "probably", "could be", "looks like", "may be",
]

SCENE_INDICATORS = [
    "the image shows", "the image contains", "the scene",
    "in this image", "looking at", "i can see",
    "the image features", "the image depicts",
]


@dataclass
class ExtractionResult:
    attributes: list[Attribute] = field(default_factory=list)
    spatial_relations: list[SpatialRelation] = field(default_factory=list)
    absent_objects: list[str] = field(default_factory=list)
    has_filtering: bool = False
    has_scene_description: bool = False


class TripleExtractor:
    """Extracts structured triples from VLM-R1 <think> block text."""

    @staticmethod
    def extract_attributes(reasoning: str, timestep: int = 0) -> list[Attribute]:
        reasoning_lower = reasoning.lower()
        attributes = []

        for word in COLOR_WORDS:
            if word in reasoning_lower:
                certainty = TripleExtractor._estimate_certainty(reasoning_lower, word)
                attributes.append(Attribute(
                    name="color", value=word, certainty=certainty,
                    source=AttributeSource.VLM_REASONING, timestep=timestep,
                ))
                break

        for word in MATERIAL_WORDS:
            if word in reasoning_lower:
                certainty = TripleExtractor._estimate_certainty(reasoning_lower, word)
                attributes.append(Attribute(
                    name="material", value=word, certainty=certainty,
                    source=AttributeSource.VLM_REASONING, timestep=timestep,
                ))
                break

        for word in SIZE_WORDS:
            if word in reasoning_lower:
                attributes.append(Attribute(
                    name="size", value=word, certainty=Certainty.MEDIUM,
                    source=AttributeSource.VLM_REASONING, timestep=timestep,
                ))
                break

        bool_patterns = [
            (r"has (?:a )?glass door", "has_glass_door", "true"),
            (r"(?:does not|doesn't) have (?:a )?glass door", "has_glass_door", "false"),
            (r"has (?:a )?handle", "has_handle", "true"),
            (r"has (?:a )?drawers?", "has_drawer", "true"),
            (r"is open", "is_open", "true"),
            (r"is closed", "is_open", "false"),
        ]
        for pattern, attr_name, attr_value in bool_patterns:
            if re.search(pattern, reasoning_lower):
                attributes.append(Attribute(
                    name=attr_name, value=attr_value, certainty=Certainty.MEDIUM,
                    source=AttributeSource.VLM_REASONING, timestep=timestep,
                ))

        return attributes

    @staticmethod
    def extract_spatial(reasoning: str, timestep: int = 0) -> list[SpatialRelation]:
        reasoning_lower = reasoning.lower()
        relations = []

        for term in SPATIAL_TERMS:
            if term in reasoning_lower:
                pattern = re.escape(term) + r"\s+(?:the\s+)?([\w\s]+?)(?:[.,;]|$)"
                match = re.search(pattern, reasoning_lower)
                reference = match.group(1).strip() if match else "unknown"
                relations.append(SpatialRelation(
                    relation=term.replace(" ", "_"), reference=reference,
                    certainty=Certainty.MEDIUM, timestep=timestep,
                ))

        return relations

    @staticmethod
    def extract_absence(reasoning: str, queried_objects: list[str]) -> list[str]:
        reasoning_lower = reasoning.lower()
        absent = []

        if not any(p in reasoning_lower for p in NEGATION_PHRASES):
            return absent

        for obj in queried_objects:
            obj_lower = obj.lower()
            if obj_lower in reasoning_lower:
                for phrase in NEGATION_PHRASES:
                    pattern = f"{phrase}.*?{re.escape(obj_lower)}|{re.escape(obj_lower)}.*?{phrase}"
                    if re.search(pattern, reasoning_lower):
                        absent.append(obj)
                        break
            else:
                words = obj_lower.split()
                if len(words) > 1 and any(w in reasoning_lower for w in words if len(w) > 3):
                    absent.append(obj)

        return absent

    @staticmethod
    def _estimate_certainty(reasoning_lower: str, keyword: str) -> Certainty:
        window = 50
        idx = reasoning_lower.find(keyword)
        if idx >= 0:
            context = reasoning_lower[max(0, idx - window):idx + window]
            if any(h in context for h in HEDGING_WORDS):
                return Certainty.LOW
        return Certainty.MEDIUM

    @classmethod
    def extract_all(cls, reasoning: str, category: str,
                    queried_objects: list[str], timestep: int = 0) -> ExtractionResult:
        reasoning_lower = reasoning.lower()
        has_scene = any(p in reasoning_lower for p in SCENE_INDICATORS)
        has_filtering = any(p in reasoning_lower for p in NEGATION_PHRASES)

        return ExtractionResult(
            attributes=cls.extract_attributes(reasoning, timestep),
            spatial_relations=cls.extract_spatial(reasoning, timestep),
            absent_objects=cls.extract_absence(reasoning, queried_objects),
            has_filtering=has_filtering,
            has_scene_description=has_scene,
        )
