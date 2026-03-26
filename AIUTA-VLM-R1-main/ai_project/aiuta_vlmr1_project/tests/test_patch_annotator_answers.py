from __future__ import annotations

from aiuta_vlmr1.evaluation.patch_annotator_answers import (
    normalize_question_text,
    patch_payload_annotator_answers,
)


def test_normalize_question_text_strips_suffix():
    q = "Is the chair red? You must answer only with Yes, No, or ?=I don't know."
    assert normalize_question_text(q) == "Is the chair red?"


def test_patch_payload_prefers_sample_id_then_question():
    payload = {
        "per_sample": [
            {"sample_id": "10", "question": "Q1", "annotator_answers": None},
            {"sample_id": "missing", "question": "Q2  ", "annotator_answers": None},
            {"sample_id": "20", "question": "Q3", "annotator_answers": {"Yes": 5}},
        ]
    }
    by_id = {"10": {"Yes": 3, "No": 1, "I don't know": 1}}
    by_question = {"Q2": {"No": 4, "Yes": 1}}

    patched, total = patch_payload_annotator_answers(
        payload,
        answers_by_id=by_id,
        answers_by_question=by_question,
    )

    assert total == 3
    assert patched == 2
    assert payload["per_sample"][0]["annotator_answers"] == by_id["10"]
    assert payload["per_sample"][1]["annotator_answers"] == by_question["Q2"]
    assert payload["per_sample"][2]["annotator_answers"] == {"Yes": 5}


def test_patch_payload_skips_non_dict_payload():
    patched, total = patch_payload_annotator_answers(
        [{"not": "a benchmark"}],
        answers_by_id={},
        answers_by_question={},
    )
    assert patched == 0 and total == 0
