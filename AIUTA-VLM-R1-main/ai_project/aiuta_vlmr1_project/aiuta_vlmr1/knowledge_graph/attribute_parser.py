"""attribute_parser.py -- Parse structured JSON attributes from VLM <answer> tags."""
from __future__ import annotations

import json
import re
from typing import Any

from .schema import Attribute, AttributeSource, Certainty
from .triple_extractor import TripleExtractor

# Canonical fields used by KG matching.
CANONICAL_FIELDS = (
    "color", "material", "size", "style", "features", "location",
    "near", "spatial", "exists", "is_open", "pattern",
)
FIELD_ALIASES = {
    "adjacent_to": "near",
    "next_to": "near",
    "room": "location",
    "texture": "pattern",
    "finish": "features",
}
BOOL_FIELDS = {"exists", "is_open"}


def _canonical_field_name(field_name: str) -> str:
    low = field_name.strip().lower()
    return FIELD_ALIASES.get(low, low)


def _normalize_attr_value(field_name: str, value: Any) -> str:
    if isinstance(value, list):
        return ", ".join(str(v).strip() for v in value if str(v).strip())
    text = str(value).strip()
    if field_name in BOOL_FIELDS:
        low = text.lower()
        if low in ("true", "yes", "y", "1"):
            return "yes"
        if low in ("false", "no", "n", "0"):
            return "no"
    return text


def _is_nullish(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and value.strip().lower() in ("null", "none", ""):
        return True
    if isinstance(value, list) and len(value) == 0:
        return True
    return False


def _extract_answer_json(text: str) -> str | None:
    """Extract content between <answer> tags, stripping whitespace."""
    m = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL)
    return m.group(1).strip() if m else None


def _try_parse_json(raw: str) -> dict[str, Any] | None:
    """Best-effort JSON parse with common VLM quirks (trailing commas, etc.)."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    # Strip trailing commas before closing brace
    cleaned = re.sub(r",\s*}", "}", raw)
    cleaned = re.sub(r",\s*]", "]", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    # Try to extract the first {...} block
    m = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return None


def parse_attribute_json(
    text: str,
    category: str = "object",
    timestep: int = 0,
) -> list[Attribute]:
    """
    Parse the attribute JSON from a VLM response.

    Returns a list of ``Attribute`` objects for each non-null field.
    Falls back to ``TripleExtractor`` if JSON parsing fails entirely.
    """
    answer_str = _extract_answer_json(text)
    if answer_str is None:
        # No <answer> tags -- try the whole text as JSON
        answer_str = text

    data = _try_parse_json(answer_str)
    if data is None:
        # Fallback: use TripleExtractor on the raw text
        return TripleExtractor.extract_attributes(text, timestep=timestep)

    attributes: list[Attribute] = []
    seen: set[str] = set()
    freeform_pairs: list[str] = []

    for raw_name, raw_value in data.items():
        field_name = _canonical_field_name(str(raw_name))
        if _is_nullish(raw_value):
            continue
        norm_val = _normalize_attr_value(field_name, raw_value)
        if not norm_val:
            continue
        if field_name in CANONICAL_FIELDS:
            if field_name in seen:
                continue
            seen.add(field_name)
            attributes.append(Attribute(
                name=field_name,
                value=norm_val,
                certainty=Certainty.HIGH,
                source=AttributeSource.VLM_REASONING,
                timestep=timestep,
            ))
            if field_name == "near" and "spatial" not in seen:
                seen.add("spatial")
                attributes.append(Attribute(
                    name="spatial",
                    value=norm_val,
                    certainty=Certainty.HIGH,
                    source=AttributeSource.VLM_REASONING,
                    timestep=timestep,
                ))
        else:
            freeform_pairs.append(f"{raw_name}:{norm_val}")

    if freeform_pairs:
        attributes.append(Attribute(
            name="open_vocab_attributes",
            value="; ".join(freeform_pairs[:8]),
            certainty=Certainty.HIGH,
            source=AttributeSource.VLM_REASONING,
            timestep=timestep,
        ))

    return attributes
