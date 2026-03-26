"""output_parser.py -- Parse VLM-R1 raw output into structured data.
Refactored from benchmark_ovd.py extract_* functions."""
from __future__ import annotations
import json, re
from dataclasses import dataclass

@dataclass
class ParsedOutput:
    reasoning_text: str | None
    answer_text: str
    bboxes: list[dict] | None
    json_valid: bool
    raw_text: str

class OutputParser:
    _THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
    _ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
    _CODE_BLOCK_RE = re.compile(r"```json\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
    _JSON_LIST_RE = re.compile(r'(\[\s*\{.*?\}\s*\])', re.DOTALL)

    @classmethod
    def extract_think(cls, raw_text: str) -> str | None:
        m = cls._THINK_RE.search(raw_text)
        return m.group(1).strip() if m else None

    @classmethod
    def extract_answer(cls, raw_text: str) -> str:
        m = cls._ANSWER_RE.search(raw_text)
        return m.group(1).strip() if m else raw_text.strip()

    @classmethod
    def extract_json_candidate(cls, raw_text: str) -> str | None:
        answer = cls.extract_answer(raw_text)
        m = cls._CODE_BLOCK_RE.search(answer)
        if m: return m.group(1).strip()
        m = cls._JSON_LIST_RE.search(answer)
        if m: return m.group(1).strip()
        if answer.strip().lower() == "none": return "None"
        return None

    @classmethod
    def parse_bboxes(cls, raw_text: str) -> tuple[bool, list[dict] | None]:
        candidate = cls.extract_json_candidate(raw_text)
        if candidate is None: return False, None
        if candidate == "None": return True, None
        try:
            parsed = json.loads(candidate)
            return (True, parsed) if isinstance(parsed, list) else (False, None)
        except json.JSONDecodeError:
            return False, None

    @classmethod
    def parse_full(cls, raw_text: str) -> ParsedOutput:
        reasoning = cls.extract_think(raw_text)
        answer = cls.extract_answer(raw_text)
        json_valid, bboxes = cls.parse_bboxes(raw_text)
        return ParsedOutput(reasoning_text=reasoning, answer_text=answer,
                            bboxes=bboxes, json_valid=json_valid, raw_text=raw_text)
