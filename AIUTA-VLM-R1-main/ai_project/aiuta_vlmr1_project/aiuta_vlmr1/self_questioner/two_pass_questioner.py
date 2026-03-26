"""
two_pass_questioner.py -- Two-pass self-questioner: detection triples + attribute forward pass.

Pass 1: Extract presence/absence triples from the OVD <think> block (same as VLMr1SelfQuestioner).
Pass 2: Run a second forward pass on the *same* model asking for structured attribute JSON
         about the detected object. Parses the JSON into Attribute objects and populates the KG.

Cost: 2 VLM calls per detection (vs 1 for VLMr1, 5-8 for original AIUTA).
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch

from ..config import Config
from ..detector.base import Detection
from ..knowledge_graph.attribute_parser import parse_attribute_json
from ..knowledge_graph.schema import TargetFacts
from ..knowledge_graph.scene_graph import SceneKnowledgeGraph
from ..knowledge_graph.triple_extractor import TripleExtractor
from ..utils.model_loader import ModelLoader
from .base import AbstractSelfQuestioner, RefinedDescription

logger = logging.getLogger(__name__)

ATTRIBUTE_PROMPT = (
    "You are an embodied agent navigating an indoor environment. "
    "The object detector found a {category} in the image.\n"
    "Carefully examine this {category} and describe ONLY it.\n"
    "Think step by step in <think> tags, then answer in <answer> tags "
    "with this exact JSON:\n"
    '{{\n'
    '  "color": "<primary color>",\n'
    '  "material": "<material if visible, else null>",\n'
    '  "size": "<large/medium/small relative to room>",\n'
    '  "style": "<modern/traditional/minimalist/antique/rustic, else null>",\n'
    '  "features": "<distinctive features: handles, doors, drawers, patterns, etc>",\n'
    '  "location": "<room name: kitchen/bedroom/living room/bathroom, else null>",\n'
    '  "near": "<object it is next to or against, else null>",\n'
    '  "exists": "yes",\n'
    '  "is_open": "<yes/no if applicable, else null>"\n'
    '}}\n'
    "Be specific. If you cannot see an attribute clearly, use null.\n"
    "Focus on attributes that distinguish THIS {category} from others in the same room."
)

ATTRIBUTE_MAX_NEW_TOKENS = 256


class TwoPassSelfQuestioner(AbstractSelfQuestioner):
    """Two-pass questioner: OVD triples + structured attribute extraction."""

    def __init__(self, config: Config) -> None:
        self._config = config
        self._loader = ModelLoader.get_instance(config.model)

    def process(
        self,
        detection: Detection,
        target_facts: TargetFacts,
        kg: SceneKnowledgeGraph,
        timestep: int = 0,
    ) -> RefinedDescription:
        # --- Pass 1: extract triples from existing detection reasoning ---
        if not detection.reasoning:
            return RefinedDescription(object_node=None, text_description="", is_valid=False)

        extraction = TripleExtractor.extract_all(
            reasoning=detection.reasoning,
            category=detection.label,
            queried_objects=[],
            timestep=timestep,
        )

        node = kg.add_object_merged(
            category=detection.label, bbox=detection.bbox, timestep=timestep,
        )
        if hasattr(detection, "image") and detection.image is not None:
            node.detected_crop = np.array(detection.image)

        if extraction.attributes:
            kg.update_attributes(node.obj_id, extraction.attributes)
        for rel in extraction.spatial_relations:
            kg.add_spatial_relation(node.obj_id, rel)

        # --- Pass 2: structured attribute extraction via second forward pass ---
        attr_attributes = self._run_attribute_pass(
            detection=detection,
            category=detection.label,
            timestep=timestep,
            target_facts=target_facts,
        )
        if attr_attributes:
            kg.update_attributes(node.obj_id, attr_attributes)

        return RefinedDescription(
            object_node=node,
            text_description=node.to_natural_language(),
            is_valid=True,
        )

    def _run_attribute_pass(
        self,
        detection: Detection,
        category: str,
        timestep: int,
        target_facts: TargetFacts | None = None,
    ) -> list:
        """Run a second VLM call to get structured attributes for the detected object."""
        return self._generate_attributes(
            self._loader, category=category, timestep=timestep,
            pil_image=getattr(detection, "image", None),
            target_facts=target_facts,
        )

    @staticmethod
    def run_attribute_pass_with_image(
        loader: ModelLoader,
        pil_image: Any,
        category: str,
        timestep: int = 0,
        question_hint: str | None = None,
        target_attr_type: str | None = None,
        existing_attributes: dict[str, str] | None = None,
    ) -> list:
        """
        Standalone attribute pass with an image -- used by idkvqa_eval's two_pass_kg mode.

        Reuses the same ModelLoader singleton so no extra GPU memory is needed.
        """
        return TwoPassSelfQuestioner._generate_attributes(
            loader,
            category=category,
            timestep=timestep,
            pil_image=pil_image,
            question_hint=question_hint,
            target_attr_type=target_attr_type,
            existing_attributes=existing_attributes,
        )

    @staticmethod
    def _contextual_attribute_prompt(
        *,
        category: str,
        question_hint: str | None = None,
        target_attr_type: str | None = None,
        existing_attributes: dict[str, str] | None = None,
        target_facts: TargetFacts | None = None,
    ) -> str:
        prompt = ATTRIBUTE_PROMPT.format(category=category)
        context_lines: list[str] = []
        if question_hint:
            context_lines.append(f"User question to solve: {question_hint}")
        if target_attr_type and target_attr_type != "unknown":
            context_lines.append(f"Prioritize evidence for this attribute type: {target_attr_type}.")
        if existing_attributes:
            shown = ", ".join(f"{k}={v}" for k, v in sorted(existing_attributes.items())[:10])
            if shown:
                context_lines.append(f"Current KG hints from pass-1: {shown}.")
        if target_facts and target_facts.known_attributes:
            facts = ", ".join(f"{k}={v}" for k, v in sorted(target_facts.known_attributes.items())[:10])
            context_lines.append(f"Known target facts: {facts}.")
        if context_lines:
            prompt = (
                f"{prompt}\n\n"
                "Context to disambiguate this object:\n"
                f"{chr(10).join(f'- {ln}' for ln in context_lines)}\n"
                "Ask yourself one concise clarifying question internally, then answer."
            )
        return prompt

    @staticmethod
    def _generate_attributes(
        loader: ModelLoader,
        *,
        category: str,
        timestep: int,
        pil_image: Any | None = None,
        question_hint: str | None = None,
        target_attr_type: str | None = None,
        existing_attributes: dict[str, str] | None = None,
        target_facts: TargetFacts | None = None,
    ) -> list:
        """Shared VLM call for structured attribute extraction (with or without image)."""
        user_text = TwoPassSelfQuestioner._contextual_attribute_prompt(
            category=category,
            question_hint=question_hint,
            target_attr_type=target_attr_type,
            existing_attributes=existing_attributes,
            target_facts=target_facts,
        )
        proc = loader.processor
        model = loader.model

        system = (
            "You are a helpful assistant specialized in visual reasoning for indoor scenes. "
            "The reasoning must be in <think></think> tags, answer in <answer></answer> tags."
        )

        user_content: list[dict[str, Any]] = []
        if pil_image is not None:
            user_content.append({"type": "image", "image": pil_image})
        user_content.append({"type": "text", "text": user_text})

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ]

        text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        proc_kwargs: dict[str, Any] = {"text": [text], "padding": True, "return_tensors": "pt"}
        if pil_image is not None:
            proc_kwargs["images"] = [pil_image]
        inputs = proc(**proc_kwargs).to(loader.device)

        with torch.inference_mode():
            gen_ids = model.generate(**inputs, max_new_tokens=ATTRIBUTE_MAX_NEW_TOKENS, do_sample=False)

        trimmed = [o[len(inp):] for inp, o in zip(inputs.input_ids, gen_ids)]
        raw_output = proc.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        return parse_attribute_json(raw_output, category=category, timestep=timestep)
