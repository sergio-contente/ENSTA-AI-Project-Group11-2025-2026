"""prompt_templates.py -- All prompt strings for VLM-R1.
Refactored from benchmark_ovd.py and benchmark_coco.py."""
from __future__ import annotations
import json

SYSTEM_PROMPT = (
    "You are a helpful assistant specialized in visual reasoning and "
    "open-vocabulary object detection. Carefully inspect the image, "
    "reason about whether each queried object is visually present, "
    "and then provide the final answer. "
    "The reasoning process must be enclosed within <think> </think> tags, "
    "and the final answer must be enclosed within <answer> </answer> tags."
)

OUTPUT_FORMAT = (
    'Return each detected object as bounding boxes in JSON format.\n'
    'Format:\n```json\n'
    '[{"bbox_2d": [x1, y1, x2, y2], "label": "object name"}]\n'
    '```\nIf none of the requested objects are present, respond with None.'
)

class PromptBuilder:
    def __init__(self):
        self._categories = []
        self._kg_context = None
        self._scene_type = "general"
        self._hard_negatives = []

    def set_categories(self, c): self._categories = c; return self
    def set_kg_context(self, c): self._kg_context = c; return self
    def set_scene_type(self, s): self._scene_type = s; return self
    def add_hard_negatives(self, n): self._hard_negatives = n; return self

    def build(self) -> tuple[str, str]:
        all_obj = json.dumps(self._categories + self._hard_negatives, ensure_ascii=False)
        parts = []
        if self._kg_context:
            parts.append(f"Context from previous observations:\n{self._kg_context}\n")
        if self._scene_type == "indoor":
            parts.append(f"Carefully inspect this indoor scene and detect: {all_obj}.\n"
                         "For each, describe attributes (color, material, size) and spatial context.\n"
                         "Some objects may not be present.\n")
        else:
            parts.append(f"Please carefully check the image and detect: {all_obj}.\n"
                         "Some may not be present. Only return visible objects.\n")
        parts.append(OUTPUT_FORMAT)
        return SYSTEM_PROMPT, "\n".join(parts)
