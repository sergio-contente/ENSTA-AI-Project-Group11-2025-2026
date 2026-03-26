"""vlmr1_questioner.py -- Self-Questioner using VLM-R1 reasoning + description pass."""
from __future__ import annotations

import numpy as np

from ..detector.base import Detection
from ..knowledge_graph.schema import Attribute, AttributeSource, Certainty, TargetFacts
from ..knowledge_graph.scene_graph import SceneKnowledgeGraph
from ..knowledge_graph.think_feature_extractor import (
    extract_think_features,
    feature_to_attribute_name,
    feature_to_qualifier,
)
from ..knowledge_graph.triple_extractor import TripleExtractor
from .base import AbstractSelfQuestioner, RefinedDescription


class VLMr1SelfQuestioner(AbstractSelfQuestioner):

    @staticmethod
    def _describe_detection(observation, category: str) -> str | None:
        """Single VLM call for a rich visual description of the detected object."""
        tmp_path: str | None = None
        try:
            import os
            import tempfile

            import torch
            from PIL import Image
            from qwen_vl_utils import process_vision_info

            from ..utils.model_loader import ModelLoader

            if not ModelLoader._instances:
                return None
            loader = ModelLoader._instances[next(iter(ModelLoader._instances))]

            if isinstance(observation, str):
                img_url = f"file://{os.path.abspath(observation)}"
            elif isinstance(observation, np.ndarray):
                pil = Image.fromarray(observation.astype(np.uint8))
                fd, tmp_path = tempfile.mkstemp(suffix=".jpg")
                os.close(fd)
                pil.save(tmp_path, format="JPEG", quality=95)
                img_url = f"file://{os.path.abspath(tmp_path)}"
            else:
                return None

            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a visual description assistant. "
                        "List distinctive features as short comma-separated phrases."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": img_url,
                            "min_pixels": 256 * 28 * 28,
                            "max_pixels": 512 * 28 * 28,
                        },
                        {
                            "type": "text",
                            "text": (
                                f"List the 3-5 most distinctive visual features of the "
                                f"{category} in this image. Write each feature as a short "
                                f"phrase (2-4 words), separated by commas. "
                                f"Focus on: colors, patterns, textures, materials, size, "
                                f"nearby objects. "
                                f"Example: blue mattress, wooden frame, near window, "
                                f"striped pillow"
                            ),
                        },
                    ],
                },
            ]

            proc = loader.processor
            text = proc.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            img_in, vid_in = process_vision_info(messages)
            inputs = proc(
                text=[text],
                images=img_in,
                videos=vid_in,
                padding=True,
                return_tensors="pt",
            ).to(loader.device)

            with torch.inference_mode():
                gen = loader.model.generate(
                    **inputs,
                    max_new_tokens=100,
                    do_sample=False,
                    use_cache=False,
                )

            trimmed = [o[len(i) :] for i, o in zip(inputs.input_ids, gen)]
            raw = proc.batch_decode(
                trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0].strip()

            print(
                f"[VLMr1SelfQuestioner] Description pass for {category}: {raw[:200]!r}"
            )
            return raw if raw else None
        except Exception as e:
            print(f"[VLMr1SelfQuestioner] Description pass error: {e}")
            return None
        finally:
            if tmp_path:
                try:
                    import os

                    os.unlink(tmp_path)
                except OSError:
                    pass

    def process(
        self,
        detection: Detection,
        target_facts: TargetFacts,
        kg: SceneKnowledgeGraph,
        timestep: int = 0,
    ) -> RefinedDescription:
        if not detection.reasoning:
            return RefinedDescription(
                object_node=None, text_description="", is_valid=False
            )
        extraction = TripleExtractor.extract_all(
            reasoning=detection.reasoning,
            category=detection.label,
            queried_objects=[],
            timestep=timestep,
        )
        node = kg.add_object_merged(
            category=detection.label, bbox=detection.bbox, timestep=timestep
        )
        if hasattr(detection, "image") and detection.image is not None:
            node.detected_crop = np.array(detection.image)
        if extraction.attributes:
            kg.update_attributes(node.obj_id, extraction.attributes)
        for rel in extraction.spatial_relations:
            kg.add_spatial_relation(node.obj_id, rel)

        # Description pass: 1 VLM call on the CROP (not the full frame)
        obs = getattr(detection, "image", None)
        obs_source = "CROP" if obs is not None else "NONE"
        if obs is None:
            obs_source = "FULL_FRAME_FALLBACK"
            print(
                f"[QUESTIONER_DEBUG] No crop for {detection.label!r} "
                f"bbox={detection.bbox} -- skipping description pass"
            )
        else:
            if isinstance(obs, np.ndarray):
                print(
                    f"[QUESTIONER_DEBUG] detection.label={detection.label!r}, "
                    f"bbox={detection.bbox}, image_source={obs_source}, "
                    f"crop_shape={obs.shape}"
                )
        description = (
            self._describe_detection(obs, detection.label)
            if obs_source == "CROP"
            else None
        )

        if description:
            features = extract_think_features(description, detection.label)
            if features:
                from ..knowledge_graph.scene_graph import (
                    SceneKnowledgeGraph,
                    _classify_value_as_attribute,
                )

                _THINK_TO_CANONICAL: dict[str, str] = {
                    "think_near": "near",
                    "think_size": "size",
                    "think_texture": "texture",
                    "think_fabric": "material",
                    "think_pattern": "pattern",
                    "think_location": "location",
                }

                for feat in features:
                    attr_name = feature_to_attribute_name(feat)
                    qualifier = feature_to_qualifier(feat)
                    qualifier = SceneKnowledgeGraph._normalize_open_answer_value(
                        qualifier
                    )
                    attr = Attribute(
                        name=attr_name,
                        value=qualifier,
                        certainty=Certainty.MEDIUM,
                        source=AttributeSource.VLM_REASONING,
                        timestep=timestep,
                    )
                    kg.update_attributes(node.obj_id, [attr])

                    canonical = (
                        _THINK_TO_CANONICAL.get(attr_name)
                        or _classify_value_as_attribute(qualifier)
                    )
                    if canonical and canonical != attr_name and not node.has_attribute(canonical):
                        canon_attr = Attribute(
                            name=canonical,
                            value=qualifier,
                            certainty=Certainty.MEDIUM,
                            source=AttributeSource.VLM_REASONING,
                            timestep=timestep,
                        )
                        kg.update_attributes(node.obj_id, [canon_attr])
                        print(
                            f"[VLMr1SelfQuestioner] Canonical: "
                            f"{attr_name}={qualifier} -> {canonical}={qualifier}"
                        )

                node._think_features = features  # type: ignore[attr-defined]
                print(
                    f"[VLMr1SelfQuestioner] Extracted features for "
                    f"'{node.obj_id}': {features}"
                )

        return RefinedDescription(
            object_node=node,
            text_description=node.to_natural_language(),
            is_valid=True,
        )
