"""
Structured parsing of user responses into target facts (provenance + normalization).

Falls back to generic ``user_stated`` when no pattern matches.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# Light synonym map for attribute names / values
SYNONYM_ATTR = {
    "colour": "color",
    "colors": "color",
    "materials": "material",
    "room": "location",
}

SYNONYM_VALUE: dict[str, str] = {}


def _norm_token(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return SYNONYM_ATTR.get(s, s)


def _plural_singular(w: str) -> str:
    w = w.strip().lower()
    if w.endswith("ies") and len(w) > 3:
        return w[:-3] + "y"
    if w.endswith("es") and len(w) > 2:
        return w[:-2]
    if w.endswith("s") and len(w) > 2:
        return w[:-1]
    return w


@dataclass
class ParsedFact:
    attribute: str
    value: str
    negative: bool
    provenance: str  # pattern id or "fallback"


def parse_user_response_to_facts(response: str) -> list[ParsedFact]:
    """
    Extract structured facts from a free-form user reply.

    Examples:
      "it is brown" -> color=brown
      "it is not wooden" -> material=wood, negative
      "it is in the kitchen" -> location=kitchen
      "it has drawers" -> has_drawer=yes
    """
    text = response.strip()
    if not text:
        return []
    low = text.lower()
    out: list[ParsedFact] = []

    if m := re.search(r"\b(?:it\s+is|it\'s|they\s+are)\s+([a-z]{3,})\b", low):
        if m.group(1) in (
            "red", "blue", "green", "black", "white", "yellow", "brown", "gray", "grey",
            "orange", "pink", "purple", "beige", "dark", "light",
        ):
            out.append(ParsedFact("color", m.group(1), False, "color_word"))

    if re.search(r"\bnot\s+(?:made\s+of\s+)?(wood|wooden|metal|glass)\b", low):
        mm = re.search(r"\bnot\s+(?:made\s+of\s+)?(wood|wooden|metal|glass|plastic|leather)\b", low)
        if mm:
            mat = mm.group(1).replace("en", "") if mm.group(1) == "wooden" else mm.group(1)
            mat = _plural_singular(mat)
            out.append(ParsedFact("material", mat, True, "negative_material"))

    if m := re.search(r"\b(?:in|inside)\s+(?:the\s+)?(kitchen|bedroom|bathroom|living\s+room|hallway)\b", low):
        loc = m.group(1).replace(" ", "_")
        out.append(ParsedFact("location", loc, False, "in_room"))

    if re.search(r"\bhas\s+(?:a\s+)?drawers?\b", low):
        out.append(ParsedFact("has_drawers", "yes", False, "has_drawers"))
    if re.search(r"\bhas\s+(?:a\s+)?handles?\b", low):
        out.append(ParsedFact("has_handle", "yes", False, "has_handle"))

    if not out:
        out.append(ParsedFact("user_stated", text[:200], False, "fallback"))

    return out

