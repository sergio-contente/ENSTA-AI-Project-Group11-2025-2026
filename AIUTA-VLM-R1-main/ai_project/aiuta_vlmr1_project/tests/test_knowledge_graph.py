"""Unit tests for the Knowledge Graph module."""
import pytest
from aiuta_vlmr1.knowledge_graph.schema import Attribute, Certainty, AttributeSource, TargetFacts
from aiuta_vlmr1.knowledge_graph.scene_graph import SceneKnowledgeGraph
from aiuta_vlmr1.knowledge_graph.graph_matcher import GraphMatcher
from aiuta_vlmr1.knowledge_graph.triple_extractor import TripleExtractor

class TestSceneKnowledgeGraph:
    def test_add_object(self):
        kg = SceneKnowledgeGraph()
        node = kg.add_object("cabinet", bbox=[10, 20, 100, 200], timestep=5)
        assert node.obj_id == "cabinet_001"
        assert kg.num_objects == 1

    def test_multiple_instances(self):
        kg = SceneKnowledgeGraph()
        n1 = kg.add_object("cabinet", timestep=1)
        n2 = kg.add_object("cabinet", timestep=5)
        assert n1.obj_id == "cabinet_001"
        assert n2.obj_id == "cabinet_002"
        assert len(kg.get_objects_by_category("cabinet")) == 2

    def test_update_keeps_higher_certainty(self):
        kg = SceneKnowledgeGraph()
        node = kg.add_object("cabinet")
        kg.update_attributes(node.obj_id, [Attribute("color", "brown", Certainty.MEDIUM)])
        assert node.get_attribute_value("color") == "brown"
        kg.update_attributes(node.obj_id, [Attribute("color", "white", Certainty.LOW)])
        assert node.get_attribute_value("color") == "brown"  # kept higher
        kg.update_attributes(node.obj_id, [Attribute("color", "white", Certainty.HIGH)])
        assert node.get_attribute_value("color") == "white"  # updated

    def test_reset(self):
        kg = SceneKnowledgeGraph()
        kg.add_object("cabinet")
        kg.target_facts.add_positive("color", "white")
        kg.reset()
        assert kg.num_objects == 0
        assert kg.target_facts.num_facts == 0

class TestGraphMatcher:
    def _node(self, attrs):
        from aiuta_vlmr1.knowledge_graph.schema import ObjectNode
        n = ObjectNode(obj_id="test", category="cabinet")
        for k, v in attrs.items():
            n.attributes[k] = Attribute(k, v, Certainty.MEDIUM)
        return n

    def test_no_facts(self):
        assert GraphMatcher.compute_alignment(self._node({"color": "brown"}), TargetFacts()) == -1.0

    def test_perfect_match(self):
        t = TargetFacts(); t.add_positive("color", "white")
        assert GraphMatcher.compute_alignment(self._node({"color": "white"}), t) == 1.0

    def test_contradiction(self):
        t = TargetFacts(); t.add_positive("color", "white")
        assert GraphMatcher.compute_alignment(self._node({"color": "brown"}), t) == 0.0

class TestTripleExtractor:
    def test_color(self):
        attrs = TripleExtractor.extract_attributes("A brown wooden cabinet near the sink.")
        colors = [a for a in attrs if a.name == "color"]
        assert len(colors) == 1 and colors[0].value == "brown"

    def test_absence(self):
        absent = TripleExtractor.extract_absence("A zebra is not present here.", ["zebra", "chair"])
        assert "zebra" in absent

    def test_spatial(self):
        rels = TripleExtractor.extract_spatial("The cabinet is near the kitchen sink.")
        assert any("near" in r.relation for r in rels)
