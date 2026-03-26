"""Scene graph merge + trigger alignment explanation."""
from __future__ import annotations

from aiuta_vlmr1.interaction_trigger.kg_trigger import KGInteractionTrigger
from aiuta_vlmr1.knowledge_graph.schema import Attribute, Certainty, ObjectNode, TargetFacts
from aiuta_vlmr1.knowledge_graph.scene_graph import SceneKnowledgeGraph
from aiuta_vlmr1.self_questioner.base import RefinedDescription
from aiuta_vlmr1.config import TriggerConfig as TC


def test_merge_same_bbox_same_timestep():
    kg = SceneKnowledgeGraph(merge_iou_threshold=0.5, merge_timestep_window=1)
    b = [0.0, 0.0, 100.0, 100.0]
    n1 = kg.add_object_merged("cabinet", bbox=b, timestep=0)
    n2 = kg.add_object_merged("cabinet", bbox=b, timestep=0)
    assert n1.obj_id == n2.obj_id
    assert kg.num_objects == 1


def test_trigger_includes_alignment_explanation():
    kg = SceneKnowledgeGraph()
    tf = TargetFacts()
    tf.category = "cabinet"
    tf.add_positive("color", "white")
    node = ObjectNode(obj_id="cabinet_001", category="cabinet")
    node.attributes["color"] = Attribute("color", "white", Certainty.HIGH)
    rd = RefinedDescription(object_node=node, text_description="x", is_valid=True)
    trig = KGInteractionTrigger(TC(tau_stop=0.99, tau_skip=0.0, max_interaction_rounds=1))
    action = trig.decide(rd, tf, kg)
    assert action.alignment_explanation.get("score") == 1.0
    assert "matched" in action.alignment_explanation
