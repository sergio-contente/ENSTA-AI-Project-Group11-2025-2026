"""Target fact parsing and graph matcher explainability."""
from __future__ import annotations

from aiuta_vlmr1.knowledge_graph.graph_matcher import GraphMatcher
from aiuta_vlmr1.knowledge_graph.schema import Attribute, Certainty, ObjectNode, TargetFacts
from aiuta_vlmr1.knowledge_graph.target_fact_parser import parse_user_response_to_facts


def test_parse_color_and_location():
    facts = parse_user_response_to_facts("it is brown")
    assert any(f.attribute == "color" and f.value == "brown" for f in facts)

    facts2 = parse_user_response_to_facts("it is in the kitchen")
    assert any("location" in f.attribute or "kitchen" in f.value for f in facts2)

    facts3 = parse_user_response_to_facts("it has drawers")
    assert any(f.attribute == "has_drawers" and f.value == "yes" for f in facts3)

    facts4 = parse_user_response_to_facts("it is not wooden")
    assert any(f.negative and f.attribute == "material" for f in facts4)


def test_explain_alignment():
    n = ObjectNode(obj_id="o1", category="cabinet")
    n.attributes["color"] = Attribute("color", "white", Certainty.MEDIUM)
    t = TargetFacts()
    t.add_positive("color", "white")
    exp = GraphMatcher.explain_alignment(n, t)
    assert exp["score"] == 1.0
    assert len(exp["matched"]) >= 1
