"""scene_graph.py -- Per-episode scene knowledge graph. Repository pattern."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator

from ..evaluation.coin_metrics import compute_iou
from .schema import (
    Attribute,
    AttributeSource,
    ObjectNode,
    SpatialRelation,
    TargetFacts,
    Certainty,
)

_COLOR_WORDS = frozenset({
    "red", "blue", "green", "black", "white", "yellow", "brown", "gray", "grey",
    "orange", "pink", "purple", "beige", "dark", "light", "tan", "cream",
    "ivory", "maroon", "navy", "teal", "turquoise", "gold", "silver",
})
_MATERIAL_WORDS = frozenset({
    "wood", "wooden", "metal", "metallic", "glass", "plastic", "leather",
    "fabric", "stone", "marble", "ceramic", "steel", "iron",
    "upholstered", "velvet", "cotton", "linen", "wicker", "bamboo",
})
_SIZE_WORDS = frozenset({
    "large", "small", "medium", "big", "tiny", "huge", "compact",
    "oversized", "tall", "short", "wide", "narrow",
})


def _classify_value_as_attribute(value: str) -> str | None:
    """Map a value word to its canonical attribute name (color/material/size)."""
    v = value.strip().lower()
    if v in _COLOR_WORDS:
        return "color"
    parts = v.split()
    if len(parts) == 2 and parts[0] in ("light", "dark", "bright", "pale", "deep"):
        if parts[1] in _COLOR_WORDS or parts[1] in (
            "brown", "blue", "green", "red", "gray", "grey", "pink", "yellow",
        ):
            return "color"
    if v in _MATERIAL_WORDS:
        return "material"
    if v in _SIZE_WORDS:
        return "size"
    return None


class SceneKnowledgeGraph:
    """
    Object deduplication (merge) heuristic:

    When adding a detection of the same ``category`` with bounding boxes whose IoU exceeds
    ``merge_iou_threshold`` and timesteps within ``merge_timestep_window`` of an existing
    node, we update ``timestep_last`` and return the existing node instead of creating a
    duplicate. This reduces redundant nodes from repeated detections of the same instance.
    """

    def __init__(self, merge_iou_threshold: float = 0.5, merge_timestep_window: int = 1):
        self._nodes: dict[str, ObjectNode] = {}
        self._category_index: dict[str, list[str]] = defaultdict(list)
        self._id_counter: dict[str, int] = defaultdict(int)
        self.target_facts = TargetFacts()
        self._merge_iou_threshold = merge_iou_threshold
        self._merge_timestep_window = merge_timestep_window

    def add_object(self, category: str, bbox=None, timestep: int = 0, image_id=None) -> ObjectNode:
        self._id_counter[category] += 1
        obj_id = f"{category}_{self._id_counter[category]:03d}"
        node = ObjectNode(
            obj_id=obj_id,
            category=category,
            bbox=bbox,
            image_id=image_id,
            timestep_first=timestep,
            timestep_last=timestep,
        )
        self._nodes[obj_id] = node
        self._category_index[category].append(obj_id)
        return node

    def add_object_merged(
        self,
        category: str,
        bbox: list[float] | None = None,
        timestep: int = 0,
        image_id=None,
    ) -> ObjectNode:
        """
        Add a new object or merge with an existing same-category node if bbox IoU is high
        and timestep is close (see class docstring).
        """
        if bbox is not None and len(bbox) == 4:
            for oid in self._category_index.get(category, []):
                node = self._nodes.get(oid)
                if node is None or node.bbox is None or len(node.bbox) != 4:
                    continue
                try:
                    iou = compute_iou(bbox, node.bbox)
                except (TypeError, ValueError):
                    continue
                if iou >= self._merge_iou_threshold:
                    if abs(timestep - node.timestep_last) <= self._merge_timestep_window:
                        node.timestep_last = timestep
                        if image_id:
                            node.image_id = image_id
                        return node
        return self.add_object(category, bbox=bbox, timestep=timestep, image_id=image_id)

    def update_attributes(self, obj_id: str, attributes: list[Attribute]) -> None:
        node = self._nodes.get(obj_id)
        if node is None:
            raise KeyError(f"Object {obj_id} not found")
        for attr in attributes:
            existing = node.attributes.get(attr.name)
            if existing is None or attr.certainty > existing.certainty:
                node.attributes[attr.name] = attr

    def add_spatial_relation(self, obj_id: str, relation: SpatialRelation) -> None:
        node = self._nodes.get(obj_id)
        if node is None:
            raise KeyError(f"Object {obj_id} not found")
        node.spatial_relations.append(relation)

    def get_object(self, obj_id: str) -> ObjectNode | None:
        return self._nodes.get(obj_id)

    def get_objects_by_category(self, category: str) -> list[ObjectNode]:
        ids = self._category_index.get(category, [])
        return [self._nodes[oid] for oid in ids if oid in self._nodes]

    def all_objects(self) -> Iterator[ObjectNode]:
        return iter(self._nodes.values())

    @staticmethod
    def _parse_yesno_question(question: str) -> tuple[str, str] | None:
        """Extract (attribute_name, attribute_value) from a yes/no question.

        Returns None if the question is not a recognized yes/no pattern.
        """
        import re

        q = question.strip().lower().rstrip("?").strip()

        m = re.search(r"is the \w+ ([\w\s-]+?) in colou?r", q)
        if m:
            return ("color", m.group(1).strip())
        m = re.search(r"is the .+ made of ([\w\s-]+?)(?:\s+(?:fabric|material))?\s*$", q)
        if m:
            return ("material", m.group(1).strip())
        m = re.search(r"does the .+ have (?:a |an )?([\w\s-]+?) texture", q)
        if m:
            return ("texture", m.group(1).strip())
        m = re.search(r"does the .+ have (?:a |an )?([\w\s-]+?) pattern", q)
        if m:
            return ("pattern", m.group(1).strip())
        m = re.search(r"does the .+ have (?:a |an )?([\w\s-]+?) surface", q)
        if m:
            return ("surface", m.group(1).strip())
        m = re.search(r"does the .+ have ([\w\s-]+?) handles", q)
        if m:
            return ("handles", m.group(1).strip())
        m = re.search(r"is the .+ ([\w\s-]+?) in shape", q)
        if m:
            return ("shape", m.group(1).strip())
        m = re.search(r"is the .+ near (?:a |an |the )?([\w\s-]+?)\s*$", q)
        if m:
            return ("near", m.group(1).strip())
        m = re.search(r"is the .+ in the ([\w\s-]+?)\s*$", q)
        if m:
            return ("location", m.group(1).strip())
        m = re.search(
            r"does the .+ have (?:a |an )?(drawers?|wheels?|doors?|glass doors?|handles?)\s*$",
            q,
        )
        if m:
            feat = m.group(1).strip()
            feat_s = feat.rstrip("s") if feat.endswith("s") and not feat.endswith("ss") else feat
            return (f"has_{feat_s}", "yes")
        m = re.search(r"is the .+ (open|closed)\s*$", q)
        if m:
            return ("is_open", m.group(1))
        m = re.search(r"is the .+ (large|small|medium|big|tiny|huge)\s*$", q)
        if m:
            return ("size", m.group(1))
        m = re.search(r"does the .+ have (?:a |an )?([\w\s-]+?)\s*$", q)
        if m:
            feat = m.group(1).strip()
            canonical = _classify_value_as_attribute(feat)
            if canonical is not None:
                return (canonical, feat)
            attr = re.sub(r"[^a-z0-9]+", "_", feat).strip("_")
            return (f"think_{attr}", feat)
        m = re.search(r"is the \w+ (\w+)\s*$", q)
        if m and m.group(1) not in ("the", "a", "an", "this", "that", "it"):
            adj = m.group(1)
            canonical = _classify_value_as_attribute(adj)
            if canonical is not None:
                return (canonical, adj)
            return ("user_stated", adj)
        return None

    @staticmethod
    def _infer_attribute_from_question(question: str) -> str | None:
        """Reverse-map a question to the attribute name it asks about."""
        import re

        q = question.strip().lower()

        parsed = SceneKnowledgeGraph._parse_yesno_question(question)
        if parsed is not None:
            return parsed[0]

        m = re.search(r"describe the (\w+) of", q)
        if m:
            return f"think_{m.group(1)}"

        patterns = [
            (r"what colou?r ", "color"),
            (r"what material ", "material"),
            (r"is the .+ large or small", "size"),
            (r"in which room ", "location"),
            (r"what is the .+? near", "near"),
            (r"does the .+ have a glass door", "has_glass_door"),
            (r"does the .+ have a handle", "has_handle"),
            (r"does the .+ have drawers", "has_drawer"),
            (r"is the .+ open or closed", "is_open"),
            (r"what is the (\w+) of", None),
        ]
        for pat, attr in patterns:
            if attr is not None:
                if re.search(pat, q):
                    return attr
            else:
                m = re.search(pat, q)
                if m:
                    return m.group(1)
        m = re.search(r"does the .+ have (?:a |an )?([\w\s]+?)\s*\??\s*$", q)
        if m:
            feat = m.group(1).strip()
            return f"think_{re.sub(r'[^a-z0-9]+', '_', feat).strip('_')}"
        return None

    _SYNONYMS: dict[str, str] = {
        "wooden": "wood", "metallic": "metal", "grey": "gray",
        "leather-like": "leather", "dark-colored": "dark",
        "light-colored": "light",
    }

    @staticmethod
    def _normalize_open_answer_value(text: str) -> str:
        """Strip common noise from oracle open answers before storing as attribute value."""
        r = text.strip().lower().rstrip(".")
        for prefix in ("it's ", "it is ", "its ", "the ", "a ", "an "):
            if r.startswith(prefix):
                r = r[len(prefix) :].strip()
        r = SceneKnowledgeGraph._SYNONYMS.get(r, r)
        return r

    def update_target_facts(
        self, user_response: str, timestep: int = 0, question: str | None = None,
    ) -> None:
        r_check = user_response.strip().lower().rstrip(".")
        if r_check in ("i don't know", "i dont know", "unknown", "not sure", "?"):
            print(f"[KG] Skipping IDK response for question: {question!r}")
            return

        src = f"user_t{timestep}"

        # -- YES/NO PATH: value from QUESTION, polarity from RESPONSE --
        if question is not None:
            parsed_yesno = self._parse_yesno_question(question)
            if parsed_yesno is not None:
                attr_name, attr_value = parsed_yesno
                oracle_yes = r_check.startswith("yes")
                oracle_no = r_check.startswith("no")
                if oracle_yes:
                    self.target_facts.add_positive(attr_name, attr_value, f"yesno_confirm|{src}")
                    print(f"[KG] Target fact: {attr_name}={attr_value} (Oracle confirmed)")
                elif oracle_no:
                    self.target_facts.add_negative(attr_name, attr_value, f"yesno_deny|{src}")
                    print(f"[KG] Target fact: NOT {attr_name}={attr_value} (Oracle denied)")
                else:
                    r_val = self._normalize_open_answer_value(user_response)
                    if r_val and r_val not in ("i don't know", "i dont know", "unknown"):
                        self.target_facts.add_positive(attr_name, r_val, f"yesno_open|{src}")
                        print(f"[KG] Target fact: {attr_name}={r_val} (Oracle open answer)")
                self.target_facts.record_question(question)
                return

        # -- OPEN-ENDED PATH: value from RESPONSE --
        from .target_fact_parser import parse_user_response_to_facts

        facts = parse_user_response_to_facts(user_response)
        structured = [f for f in facts if f.provenance != "fallback"]
        if structured:
            for f in structured:
                if f.negative:
                    self.target_facts.add_negative(f.attribute, f.value, f"{f.provenance}|{src}")
                else:
                    self.target_facts.add_positive(f.attribute, f.value, f"{f.provenance}|{src}")
            if question:
                self.target_facts.record_question(question)
            return

        if question is not None:
            attr = self._infer_attribute_from_question(question)
            if attr is not None:
                r = self._normalize_open_answer_value(user_response)
                if r and r not in ("i don't know", "i dont know", "unknown"):
                    self.target_facts.add_positive(attr, r, f"question_attr|{src}")
                    self.target_facts.record_question(question)
                    return

        # -- FINAL FALLBACK --
        r = user_response.strip().lower()
        is_neg = any(r.startswith(p) for p in ("no", "not", "it is not", "it's not"))
        if is_neg:
            clean = r
            for prefix in ("no, ", "not ", "it is not ", "it's not "):
                if clean.startswith(prefix):
                    clean = clean[len(prefix):]
                    break
            self.target_facts.add_negative("user_stated", clean, src)
        else:
            clean = r
            for prefix in ("yes, ", "it is ", "it's ", "it has "):
                if clean.startswith(prefix):
                    clean = clean[len(prefix):]
                    break
            self.target_facts.add_positive("user_stated", clean, src)
        if question:
            self.target_facts.record_question(question)

    def get_kg_context_string(self, category: str) -> str:
        instances = self.get_objects_by_category(category)
        if not instances:
            return ""
        lines = [f"Previously observed {len(instances)} {category}(s):"]
        for node in instances:
            lines.append(f"  - {node.to_natural_language()}")
        if self.target_facts.num_facts > 0:
            lines.append(f"\nTarget: {self.target_facts.to_natural_language()}")
        return "\n".join(lines)

    def get_missing_attributes(self, obj_id: str) -> list[str]:
        node = self._nodes.get(obj_id)
        if node is None:
            return []
        return [a for a in self.target_facts.known_attributes if not node.has_attribute(a)]

    def reset(self):
        self._nodes.clear()
        self._category_index.clear()
        self._id_counter.clear()
        self.target_facts = TargetFacts()

    @property
    def num_objects(self) -> int:
        return len(self._nodes)

    def get_attributes_for_image(self, image_id: str) -> dict[str, str]:
        """Flatten attribute / spatial relation values for all objects in ``image_id``."""
        attrs: dict[str, str] = {}
        for node in self._nodes.values():
            if node.image_id == image_id:
                for name, attr in node.attributes.items():
                    if name not in attrs:
                        attrs[name] = attr.value
                for rel in node.spatial_relations:
                    if rel.relation not in attrs:
                        attrs[rel.relation] = rel.reference
        return attrs

    def get_objects_for_image(self, image_id: str) -> list[ObjectNode]:
        return [n for n in self._nodes.values() if n.image_id == image_id]

    def to_dict(self) -> dict[str, Any]:
        """Serialize full KG state to a JSON-compatible dict."""
        nodes_out: list[dict[str, Any]] = []
        for node in self._nodes.values():
            nodes_out.append(
                {
                    "obj_id": node.obj_id,
                    "category": node.category,
                    "bbox": node.bbox,
                    "image_id": node.image_id,
                    "timestep_first": node.timestep_first,
                    "timestep_last": node.timestep_last,
                    "attributes": {
                        k: {
                            "name": v.name,
                            "value": v.value,
                            "certainty": v.certainty.value,
                            "source": v.source.value,
                            "timestep": v.timestep,
                        }
                        for k, v in node.attributes.items()
                    },
                    "spatial_relations": [
                        {
                            "relation": r.relation,
                            "reference": r.reference,
                            "certainty": r.certainty.value,
                            "timestep": r.timestep,
                        }
                        for r in node.spatial_relations
                    ],
                    "is_target_candidate": node.is_target_candidate,
                    "alignment_score": node.alignment_score,
                }
            )
        tf = self.target_facts
        return {
            "version": 1,
            "merge_iou_threshold": self._merge_iou_threshold,
            "merge_timestep_window": self._merge_timestep_window,
            "nodes": nodes_out,
            "category_index": {k: list(v) for k, v in self._category_index.items()},
            "id_counter": {k: int(v) for k, v in self._id_counter.items()},
            "target_facts": {
                "category": tf.category,
                "known_attributes": dict(tf.known_attributes),
                "negative_attributes": dict(tf.negative_attributes),
                "source_history": list(tf.source_history),
                "fact_provenance": dict(tf.fact_provenance),
                "asked_questions": list(tf.asked_questions),
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SceneKnowledgeGraph:
        """Reconstruct KG from :meth:`to_dict` output."""
        kg = cls(
            merge_iou_threshold=float(data.get("merge_iou_threshold", 0.5)),
            merge_timestep_window=int(data.get("merge_timestep_window", 1)),
        )
        kg._nodes.clear()
        kg._category_index.clear()
        kg._id_counter.clear()

        def _certainty(v: Any) -> Certainty:
            try:
                return Certainty(str(v).lower())
            except ValueError:
                return Certainty.MEDIUM

        def _source(v: Any) -> AttributeSource:
            try:
                return AttributeSource(str(v).lower())
            except ValueError:
                return AttributeSource.VLM_REASONING

        for nd in data.get("nodes") or []:
            attrs: dict[str, Attribute] = {}
            for k, ad in (nd.get("attributes") or {}).items():
                attrs[k] = Attribute(
                    name=ad["name"],
                    value=ad["value"],
                    certainty=_certainty(ad.get("certainty", "medium")),
                    source=_source(ad.get("source", "vlm_reasoning")),
                    timestep=int(ad.get("timestep", 0)),
                )
            spatial: list[SpatialRelation] = []
            for sr in nd.get("spatial_relations") or []:
                spatial.append(
                    SpatialRelation(
                        relation=sr["relation"],
                        reference=sr["reference"],
                        certainty=_certainty(sr.get("certainty", "medium")),
                        timestep=int(sr.get("timestep", 0)),
                    )
                )
            node = ObjectNode(
                obj_id=nd["obj_id"],
                category=nd["category"],
                bbox=nd.get("bbox"),
                image_id=nd.get("image_id"),
                timestep_first=int(nd.get("timestep_first", 0)),
                timestep_last=int(nd.get("timestep_last", 0)),
                attributes=attrs,
                spatial_relations=spatial,
                is_target_candidate=bool(nd.get("is_target_candidate", False)),
                alignment_score=float(nd.get("alignment_score", -1.0)),
            )
            kg._nodes[node.obj_id] = node

        # Rebuild category index from nodes (authoritative)
        for node in kg._nodes.values():
            kg._category_index[node.category].append(node.obj_id)

        for cat, c in (data.get("id_counter") or {}).items():
            kg._id_counter[str(cat)] = int(c)

        tf_raw = data.get("target_facts") or {}
        kg.target_facts = TargetFacts(
            category=str(tf_raw.get("category", "")),
            known_attributes=dict(tf_raw.get("known_attributes") or {}),
            negative_attributes=dict(tf_raw.get("negative_attributes") or {}),
            source_history=list(tf_raw.get("source_history") or []),
            fact_provenance=dict(tf_raw.get("fact_provenance") or {}),
            asked_questions=list(tf_raw.get("asked_questions") or []),
        )

        return kg

    def save_json(self, path: str | Path) -> None:
        """Save KG to a JSON file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False, default=str)

    @classmethod
    def load_json(cls, path: str | Path) -> SceneKnowledgeGraph:
        """Load KG from JSON written by :meth:`save_json`."""
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))
