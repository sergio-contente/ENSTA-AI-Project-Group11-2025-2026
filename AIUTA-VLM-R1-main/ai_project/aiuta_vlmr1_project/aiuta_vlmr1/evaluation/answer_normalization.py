"""
Centralized Yes / No / I don't know normalization for IDKVQA-style evaluation.
All benchmarks must use `normalize_yes_no_idk` for comparable metrics.
"""
from __future__ import annotations

import re

# Canonical labels (match IDKVQA ground-truth strings)
LABEL_YES = "Yes"
LABEL_NO = "No"
LABEL_IDK = "I don't know"


def normalize_yes_no_idk(text: str | None) -> str:
    """
    Map free-form model output to one of: Yes, No, I don't know.
    Returns the original stripped string if no pattern matches (caller may treat as invalid).
    """
    if text is None:
        return LABEL_IDK
    s = text.strip()
    if not s:
        return LABEL_IDK

    lower = s.lower()

    # IDK / uncertain first (avoid "no idea" matching "no")
    idk_patterns = (
        r"\bi\s*don'?t\s+know\b",
        r"\bdon'?t\s+know\b",
        r"\b(dunno|unknown|unclear|unsure|uncertain|cannot\s+tell|can'?t\s+tell)\b",
        r"\bnot\s+(?:sure|certain|clear)\b",
        r"\bhard\s+to\s+tell\b",
        r"\bambiguous\b",
        r"(?:^|\s)\?(?:\s|$)",
        r"\bno\s+information\b",
        r"\bcannot\s+determine\b",
        r"\bcan'?t\s+determine\b",
    )
    for pat in idk_patterns:
        if re.search(pat, lower):
            return LABEL_IDK

    # Yes-like
    yes_patterns = (
        r"^\s*yes\b",
        r"^\s*yep\b",
        r"^\s*yeah\b",
        r"\baffirmative\b",
        r"\bpresent\b",
        r"\bvisible\b",
        r"\bthere\s+is\b",
        r"\bit\s+is\s+there\b",
    )
    for pat in yes_patterns:
        if re.search(pat, lower):
            return LABEL_YES

    # No-like (after IDK check)
    no_patterns = (
        r"^\s*no\b",
        r"^\s*nope\b",
        r"\babsent\b",
        r"\bnot\s+present\b",
        r"\bnot\s+visible\b",
        r"\bthere\s+is\s+no\b",
        r"\bdoes\s+not\b",
        r"\bdon'?t\s+see\b",
    )
    for pat in no_patterns:
        if re.search(pat, lower):
            return LABEL_NO

    # Short answers
    if lower in ("y", "true"):
        return LABEL_YES
    if lower in ("n", "false"):
        return LABEL_NO

    return s
