"""
question_generator.py -- Generate discriminative questions for the user.
Replaces AIUTA LLM-based question generation with KG analysis.
"""
from __future__ import annotations

from .graph_matcher import GraphMatcher
from .schema import ObjectNode, TargetFacts
from .scene_graph import SceneKnowledgeGraph
from .think_feature_extractor import feature_to_attribute_name, feature_to_question

YESNO_TEMPLATES = {
    "color": "Is the {category} {value} in color?",
    "material": "Is the {category} made of {value}?",
    "size": "Is the {category} {value}?",
    "texture": "Does the {category} have a {value} texture?",
    "pattern": "Does the {category} have a {value} pattern?",
    "shape": "Is the {category} {value} in shape?",
    "fabric": "Is the {category} made of {value} fabric?",
    "surface": "Does the {category} have a {value} surface?",
    "has_glass_door": "Does the {category} have a glass door?",
    "has_handle": "Does the {category} have a handle?",
    "has_drawer": "Does the {category} have drawers?",
    "is_open": "Is the {category} open?",
    "location": "Is the {category} in the {value}?",
    "near": "Is the {category} near a {value}?",
}

YESNO_CATCHALL = "Does the {category} have {value}?"

COLOR_WORDS = frozenset({
    "red", "blue", "green", "black", "white", "yellow", "brown", "gray", "grey",
    "orange", "pink", "purple", "beige", "dark", "light", "tan", "cream",
    "ivory", "maroon", "navy", "teal", "turquoise", "gold", "silver",
})
MATERIAL_WORDS = frozenset({
    "wood", "wooden", "metal", "metallic", "glass", "plastic", "leather",
    "fabric", "stone", "marble", "ceramic", "steel", "iron",
    "upholstered", "velvet", "cotton", "linen", "wicker", "bamboo",
})
SIZE_WORDS = frozenset({
    "large", "small", "medium", "big", "tiny", "huge", "compact",
    "oversized", "tall", "short", "wide", "narrow",
})


def _infer_semantic_type(attr_name: str, value: str) -> str | None:
    """Map a value to its canonical YESNO_TEMPLATES key."""
    v = value.strip().lower()
    if v in COLOR_WORDS:
        return "color"
    parts = v.split()
    if len(parts) == 2 and parts[0] in ("light", "dark", "bright", "pale", "deep"):
        if parts[1] in COLOR_WORDS or parts[1] in (
            "brown", "blue", "green", "red", "gray", "grey", "pink", "yellow",
        ):
            return "color"
    if v in MATERIAL_WORDS:
        return "material"
    if v in SIZE_WORDS:
        return "size"
    clean = attr_name.lower().replace("think_", "")
    if clean in ("color", "colour"):
        return "color"
    if clean in ("material", "fabric", "texture"):
        return "material"
    if clean == "size":
        return "size"
    if clean in ("near", "next_to"):
        return "near"
    if clean in ("location", "room"):
        return "location"
    return None


QUESTION_TEMPLATES = {
    "color": "What color is the {category}?",
    "material": "What material is the {category} made of?",
    "size": "Is the {category} large or small?",
    "has_glass_door": "Does the {category} have a glass door?",
    "has_handle": "Does the {category} have a handle?",
    "has_drawer": "Does the {category} have drawers?",
    "is_open": "Is the {category} open or closed?",
    "location": "In which room is the {category} located?",
    "near": "What is the {category} near or next to?",
}

GENERIC_QUESTIONS = [
    "Can you describe any distinctive features of the {category}?",
    "What does the {category} look like?",
]


class QuestionGenerator:
    @staticmethod
    def generate(obj: ObjectNode, kg: SceneKnowledgeGraph) -> str:
        target = kg.target_facts
        asked = set(target.asked_questions)
        known = set(target.known_attributes.keys()) | set(target.negative_attributes.keys())

        # Priority 0: Confirm attributes the detected object already has (yes/no)
        for attr_name, attr_obj in obj.attributes.items():
            if attr_name in known:
                continue
            sem = _infer_semantic_type(attr_name, attr_obj.value)
            if sem and sem in known:
                continue
            if sem and sem in YESNO_TEMPLATES:
                template = YESNO_TEMPLATES[sem]
            else:
                real_name = attr_name
                if real_name.startswith("think_"):
                    real_name = real_name[len("think_"):]
                template = YESNO_TEMPLATES.get(real_name, YESNO_CATCHALL)
            candidate = template.format(category=obj.category, value=attr_obj.value)
            if candidate not in asked:
                return candidate

        # Priority 1: Think features from description pass (highly discriminative)
        think_features: list[str] = getattr(obj, "_think_features", None) or []
        for feat in think_features:
            attr_name = feature_to_attribute_name(feat)
            if attr_name not in known:
                candidate = feature_to_question(feat, obj.category)
                if candidate not in asked and f"the {obj.category} of the {obj.category}" not in candidate.lower():
                    return candidate

        # Priority 2: Template questions (common attributes)
        all_instances = kg.get_objects_by_category(obj.category)
        all_attr_names = set()
        for inst in all_instances:
            all_attr_names.update(
                a for a in inst.attributes.keys() if not a.startswith("think_")
            )
        unknown = [a for a in all_attr_names if a not in known]
        for common in ["color", "material", "size", "location", "near"]:
            if common not in known and common not in unknown:
                unknown.append(common)
        if not unknown:
            for g in GENERIC_QUESTIONS:
                candidate = g.format(category=obj.category)
                if candidate not in asked:
                    return candidate
            return GENERIC_QUESTIONS[0].format(category=obj.category)
        ranked = [(a, GraphMatcher.compute_discriminative_power(a, all_instances)) for a in unknown]
        ranked.sort(key=lambda x: -x[1])
        for attr_name, _score in ranked:
            template = QUESTION_TEMPLATES.get(attr_name, f"What is the {attr_name} of the {{category}}?")
            candidate = template.format(category=obj.category)
            if candidate not in asked:
                return candidate
        for g in GENERIC_QUESTIONS:
            candidate = g.format(category=obj.category)
            if candidate not in asked:
                return candidate
        best = ranked[0][0]
        template = QUESTION_TEMPLATES.get(best, f"What is the {best} of the {{category}}?")
        return template.format(category=obj.category)
