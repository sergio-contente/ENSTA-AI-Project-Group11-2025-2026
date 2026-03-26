"""
Backfill `annotator_answers` in existing IDKVQA result JSON files.

This utility is meant for older runs generated before annotator vote fields were
serialized in `per_sample`.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


_SUFFIXES_TO_STRIP = (
    "You must answer only with Yes, No, or ?=I don't know.",
    "You must answer only with Yes, No, or I don't know.",
)


def normalize_question_text(question: str) -> str:
    """Normalize question text for robust matching across historical prompts."""
    q = question.strip()
    for suffix in _SUFFIXES_TO_STRIP:
        if q.endswith(suffix):
            q = q[: -len(suffix)].strip()
    q = re.sub(r"\s+", " ", q)
    return q


def build_answer_indices(split: str = "val") -> tuple[dict[str, dict[str, int]], dict[str, dict[str, int]]]:
    """
    Return two lookup maps:
    - by sample_id
    - by normalized question text
    """
    from datasets import load_dataset

    ds = load_dataset("ftaioli/IDKVQA", split=split)
    by_id: dict[str, dict[str, int]] = {}
    by_question: dict[str, dict[str, int]] = {}

    for i, row in enumerate(ds):
        answers = row.get("answers")
        if not isinstance(answers, dict):
            continue

        sample_id = str(row.get("id", i))
        by_id[sample_id] = answers

        q = row.get("question", "")
        if isinstance(q, str) and q:
            by_question[normalize_question_text(q)] = answers

    return by_id, by_question


def patch_payload_annotator_answers(
    payload: dict[str, Any],
    *,
    answers_by_id: dict[str, dict[str, int]],
    answers_by_question: dict[str, dict[str, int]],
) -> tuple[int, int]:
    """Patch one loaded result payload in-place. Returns (patched, total)."""
    if not isinstance(payload, dict):
        return 0, 0
    per_sample = payload.get("per_sample")
    if not isinstance(per_sample, list):
        return 0, 0

    patched = 0
    total = len(per_sample)
    for row in per_sample:
        if not isinstance(row, dict):
            continue
        if row.get("annotator_answers"):
            continue

        answers = None
        sample_id = row.get("sample_id")
        if sample_id is not None:
            answers = answers_by_id.get(str(sample_id))

        if answers is None:
            q = row.get("question")
            if isinstance(q, str) and q:
                answers = answers_by_question.get(normalize_question_text(q))

        if answers is not None:
            row["annotator_answers"] = answers
            patched += 1

    return patched, total


def patch_results_dir(results_dir: str, split: str = "val", write_back: bool = True) -> list[dict[str, Any]]:
    """
    Patch all JSON files in a results directory.

    Returns a report list with patch stats per file.
    """
    answers_by_id, answers_by_question = build_answer_indices(split=split)
    out: list[dict[str, Any]] = []
    base = Path(results_dir)
    for path in sorted(base.glob("*.json")):
        try:
            with open(path, encoding="utf-8") as f:
                payload = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue

        patched, total = patch_payload_annotator_answers(
            payload,
            answers_by_id=answers_by_id,
            answers_by_question=answers_by_question,
        )
        if write_back and patched > 0:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False, default=str)

        out.append(
            {
                "file": str(path),
                "patched": patched,
                "total": total,
                "coverage_pct": round((patched / total * 100.0), 2) if total else 0.0,
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill annotator_answers in IDKVQA result JSONs")
    parser.add_argument("--results-dir", type=str, required=True, help="Directory with result JSON files")
    parser.add_argument("--split", type=str, default="val", help="Dataset split to index (default: val)")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute patch stats without writing files",
    )
    args = parser.parse_args()

    report = patch_results_dir(args.results_dir, split=args.split, write_back=not args.dry_run)
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
