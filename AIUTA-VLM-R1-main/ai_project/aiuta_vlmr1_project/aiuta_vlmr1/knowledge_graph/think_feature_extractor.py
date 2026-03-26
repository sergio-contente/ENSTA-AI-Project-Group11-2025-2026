"""
think_feature_extractor.py -- Extract and process visual features from VLM descriptions.

Truly open-ended: no hardcoded dictionaries of adjectives or question types.
Uses a single universal template and substring matching for comparison.
"""
from __future__ import annotations

import re

_GENERIC_FEATURES = frozenset({
    "bed", "a bed", "the bed", "chair", "a chair", "the chair",
    "cabinet", "a cabinet", "the cabinet", "couch", "a couch",
    "picture", "a picture", "the object", "an object",
    "the room", "the image", "the scene", "the wall", "the floor",
    "the ceiling", "this image", "the background",
    "unknown", "none", "n/a", "not visible", "not sure",
    "no visible objects",
})

_SPATIAL_PREFIXES = ("near", "against", "beside", "behind", "next to", "in front of")

_META_PREFIXES = (
    "i can", "i see", "the image", "this is", "there is", "it appears",
    "it looks", "it seems", "the photo", "in this",
)

_SYNONYMS: dict[str, str] = {
    "wooden": "wood", "wood": "wooden",
    "metallic": "metal", "metal": "metallic",
    "grey": "gray", "gray": "grey",
    "big": "large", "large": "big",
    "small": "tiny", "tiny": "small",
    "dark-colored": "dark", "dark": "dark-colored",
    "light-colored": "light", "light": "light-colored",
}


def extract_think_features(
    description: str, category: str, max_features: int = 5
) -> list[str]:
    """
    Extract features from a structured VLM description (comma/newline separated).
    Returns cleaned list of 2+ word phrases: ["green mattress", "metal frame"]
    """
    if not description:
        return []

    raw_parts = re.split(r"[,\n]|\band\b", description)

    features: list[str] = []
    seen: set[str] = set()

    for part in raw_parts:
        feat = part.strip().rstrip(".,;:").lstrip("-*\u2022123456789. ")
        feat = feat.strip()
        for prefix in ("a ", "an ", "the ", "its ", "this ", "it has ", "with "):
            if feat.lower().startswith(prefix):
                feat = feat[len(prefix):]
        feat = feat.strip()
        low = feat.lower()

        if len(feat) < 3:
            continue
        if low in _GENERIC_FEATURES or low == category.lower():
            continue
        # Strip category name from the feature phrase to avoid "dresser of dresser"
        cat_lower = category.lower()
        words = low.split()
        words_no_cat = [w for w in words if w != cat_lower]
        if not words_no_cat:
            continue
        if words_no_cat != words:
            feat = " ".join(words_no_cat)
            low = feat.lower()
        if len(feat.split()) < 2 and len(feat) < 5:
            continue
        if len(feat.split()) > 6:
            continue
        if any(low.startswith(p) for p in _META_PREFIXES):
            continue

        if low not in seen:
            seen.add(low)
            features.append(feat)

        if len(features) >= max_features:
            break

    return features


def decompose_feature(feature: str) -> tuple[str, str]:
    """
    Split feature into (qualifier, subject).
    "green mattress" ->("green", "mattress")
    "dark-colored large mattress" ->("dark-colored large", "mattress")
    "near window" ->("window", "")
    """
    words = feature.strip().split()

    for sp in _SPATIAL_PREFIXES:
        sp_words = sp.split()
        if [w.lower() for w in words[: len(sp_words)]] == sp_words:
            rest = " ".join(words[len(sp_words) :])
            return (rest, "")

    if len(words) == 1:
        return (words[0], "")

    subject = words[-1]
    qualifier = " ".join(words[:-1])
    return (qualifier, subject)


def feature_to_question(feature: str, category: str) -> str:
    """
    One universal template -- no classification needed.
    "green mattress" ->"Describe the mattress of the bed."
    "near window"    ->"What is the bed near or next to?"
    """
    qualifier, subject = decompose_feature(feature)
    if not subject:
        return f"What is the {category} near or next to?"
    if subject.lower() == category.lower():
        return f"Can you describe any distinctive features of the {category}?"
    return f"Describe the {subject} of the {category}."


def feature_to_attribute_name(feature: str) -> str:
    """
    Noun-based attribute name.
    "green mattress" ->"think_mattress"
    "near window"    ->"think_near"
    """
    _qualifier, subject = decompose_feature(feature)
    if not subject:
        base = "near"
    else:
        base = re.sub(r"[^a-z0-9]+", "_", subject.lower()).strip("_")
    if len(base) > 30:
        base = base[:30].rstrip("_")
    return f"think_{base}"


def feature_to_qualifier(feature: str) -> str:
    """
    Extract the qualifier (what the detection observed).
    "green mattress" ->"green"
    "near window"    ->"window"
    """
    qualifier, _subject = decompose_feature(feature)
    return qualifier.lower().strip()


def qualifier_matches_response(qualifier: str, oracle_response: str) -> bool:
    """
    Substring + synonym check.
    "green" in "white cotton mattress" ->False
    "white" in "white cotton mattress" ->True
    "wooden" vs "wood frame"           ->True (synonym)
    """
    q = qualifier.lower().strip()
    r = oracle_response.lower().strip()

    if q in r or r in q:
        return True

    syn = _SYNONYMS.get(q)
    if syn and syn in r:
        return True

    return False
