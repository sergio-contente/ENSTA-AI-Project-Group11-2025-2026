"""Unit tests for the output parser."""
from aiuta_vlmr1.detector.output_parser import OutputParser

class TestOutputParser:
    def test_extract_think(self):
        assert OutputParser.extract_think("<think>reasoning</think><answer>x</answer>") == "reasoning"

    def test_extract_think_none(self):
        assert OutputParser.extract_think("no tags") is None

    def test_parse_bboxes_valid(self):
        text = '<answer>```json\n[{"bbox_2d": [10,20,100,200], "label": "cat"}]\n```</answer>'
        valid, boxes = OutputParser.parse_bboxes(text)
        assert valid and len(boxes) == 1

    def test_parse_bboxes_none(self):
        valid, boxes = OutputParser.parse_bboxes("<answer>None</answer>")
        assert valid and boxes is None

    def test_parse_full(self):
        text = "<think>A dog in a park.</think><answer>```json\n[{\"bbox_2d\": [50,60,200,300], \"label\": \"dog\"}]\n```</answer>"
        r = OutputParser.parse_full(text)
        assert r.reasoning_text == "A dog in a park."
        assert r.json_valid and len(r.bboxes) == 1
