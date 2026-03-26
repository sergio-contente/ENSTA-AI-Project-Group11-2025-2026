"""Tests for TwoPassSelfQuestioner and AttributeParser."""
import pytest
from unittest.mock import MagicMock, patch

from aiuta_vlmr1.knowledge_graph.schema import Attribute, Certainty, AttributeSource
from aiuta_vlmr1.knowledge_graph.scene_graph import SceneKnowledgeGraph
from aiuta_vlmr1.knowledge_graph.attribute_parser import parse_attribute_json
from aiuta_vlmr1.detector.base import Detection


# ---------------------------------------------------------------------------
# AttributeParser tests
# ---------------------------------------------------------------------------

class TestAttributeParser:
    def test_valid_json(self):
        text = (
            '<think>The cabinet is brown wood.</think>'
            '<answer>{"color": "brown", "material": "wood", "size": "medium", '
            '"features": "two doors with handles", "location": "near the sink", '
            '"pattern": null}</answer>'
        )
        attrs = parse_attribute_json(text, category="cabinet", timestep=3)
        names = {a.name for a in attrs}
        assert "color" in names
        assert "material" in names
        assert "size" in names
        assert "features" in names
        assert "location" in names
        # pattern is null -> should NOT be present
        assert "pattern" not in names
        # All should be HIGH certainty (structured JSON)
        for a in attrs:
            assert a.certainty == Certainty.HIGH
            assert a.timestep == 3

    def test_null_fields_skipped(self):
        text = '<answer>{"color": "white", "material": null, "size": null, "features": null, "location": null, "pattern": null}</answer>'
        attrs = parse_attribute_json(text)
        assert len(attrs) == 1
        assert attrs[0].name == "color"
        assert attrs[0].value == "white"

    def test_malformed_json_fallback(self):
        text = "The cabinet appears to be brown and wooden, it is large."
        attrs = parse_attribute_json(text, category="cabinet")
        # Should fallback to TripleExtractor and find at least color
        names = {a.name for a in attrs}
        assert "color" in names

    def test_trailing_comma(self):
        text = '<answer>{"color": "red", "material": "metal",}</answer>'
        attrs = parse_attribute_json(text)
        assert len(attrs) == 2
        assert attrs[0].value == "red"

    def test_none_string_treated_as_null(self):
        text = '<answer>{"color": "blue", "material": "none", "size": "large", "features": "None", "location": "corner", "pattern": ""}</answer>'
        attrs = parse_attribute_json(text)
        names = {a.name for a in attrs}
        assert "material" not in names
        assert "features" not in names
        assert "pattern" not in names
        assert "color" in names
        assert "size" in names
        assert "location" in names


# ---------------------------------------------------------------------------
# TwoPassSelfQuestioner tests (mocked model)
# ---------------------------------------------------------------------------

def _make_mock_loader():
    """Create a mock ModelLoader with a generate method that returns attribute JSON."""
    loader = MagicMock()
    loader.device = "cpu"

    # Processor mock
    proc = MagicMock()
    proc.apply_chat_template.return_value = "formatted_text"
    proc.return_value = MagicMock(
        input_ids=[[1, 2, 3]],
        to=MagicMock(return_value=MagicMock(input_ids=[[1, 2, 3]])),
    )
    # Make proc() return something with .to()
    proc_result = MagicMock()
    proc_result.input_ids = [[1, 2, 3]]
    proc_result.to.return_value = proc_result
    proc.side_effect = lambda **kw: proc_result if "text" not in kw else proc_result
    proc.__call__ = lambda self, **kw: proc_result
    loader.processor = proc

    loader.model = MagicMock()

    return loader


class TestTwoPassSelfQuestioner:
    @patch("aiuta_vlmr1.self_questioner.two_pass_questioner.ModelLoader")
    def test_process_populates_kg(self, mock_model_loader_cls):
        """Mock model.generate to return valid attribute JSON, verify KG is populated."""
        mock_loader = MagicMock()
        mock_loader.device = "cpu"
        mock_model_loader_cls.get_instance.return_value = mock_loader

        # Setup processor mock
        proc = MagicMock()
        proc.apply_chat_template.return_value = "text"
        proc_inputs = MagicMock()
        proc_inputs.input_ids = [[1, 2, 3]]
        proc_inputs.to.return_value = proc_inputs
        proc.return_value = proc_inputs
        mock_loader.processor = proc

        # Model generates attribute JSON
        import torch
        gen_output = torch.tensor([[1, 2, 3, 10, 11, 12, 13, 14]])
        mock_loader.model.generate.return_value = gen_output

        attr_json_response = (
            '<think>I see a brown wooden cabinet.</think>'
            '<answer>{"color": "brown", "material": "wood", "size": "large", '
            '"features": "two handles", "location": "near the wall", "pattern": null}</answer>'
        )
        proc.batch_decode.return_value = [attr_json_response]

        from aiuta_vlmr1.self_questioner.two_pass_questioner import TwoPassSelfQuestioner
        from aiuta_vlmr1.config import Config

        config = Config()
        questioner = TwoPassSelfQuestioner(config)

        kg = SceneKnowledgeGraph()
        detection = Detection(
            bbox=[10, 20, 100, 200],
            label="cabinet",
            reasoning="I can see a brown cabinet in the image. It is near the wall.",
        )

        result = questioner.process(detection, kg.target_facts, kg, timestep=0)

        assert result.is_valid
        assert result.object_node is not None
        node = kg.get_object(result.object_node.obj_id)
        assert node is not None

        # Should have attributes from both passes
        assert node.has_attribute("color")
        assert node.get_attribute_value("color") == "brown"
        # From attribute pass (HIGH certainty overrides MEDIUM from pass 1)
        assert node.has_attribute("material")
        assert node.has_attribute("size")

    @patch("aiuta_vlmr1.self_questioner.two_pass_questioner.ModelLoader")
    def test_fallback_on_malformed_json(self, mock_model_loader_cls):
        """If the attribute pass returns garbage, should gracefully fallback."""
        mock_loader = MagicMock()
        mock_loader.device = "cpu"
        mock_model_loader_cls.get_instance.return_value = mock_loader

        proc = MagicMock()
        proc.apply_chat_template.return_value = "text"
        proc_inputs = MagicMock()
        proc_inputs.input_ids = [[1, 2, 3]]
        proc_inputs.to.return_value = proc_inputs
        proc.return_value = proc_inputs
        mock_loader.processor = proc

        import torch
        gen_output = torch.tensor([[1, 2, 3, 10, 11]])
        mock_loader.model.generate.return_value = gen_output

        # Model returns garbage (no valid JSON)
        proc.batch_decode.return_value = ["This is not valid JSON at all, just rambling text"]

        from aiuta_vlmr1.self_questioner.two_pass_questioner import TwoPassSelfQuestioner
        from aiuta_vlmr1.config import Config

        config = Config()
        questioner = TwoPassSelfQuestioner(config)

        kg = SceneKnowledgeGraph()
        detection = Detection(
            bbox=[10, 20, 100, 200],
            label="chair",
            reasoning="There is a red chair in the room.",
        )

        result = questioner.process(detection, kg.target_facts, kg, timestep=0)

        # Should still be valid (pass 1 succeeds even if pass 2 fails)
        assert result.is_valid
        node = kg.get_object(result.object_node.obj_id)
        assert node is not None
        # At minimum, pass 1 should have extracted color from reasoning
        assert node.has_attribute("color")

    def test_no_reasoning_returns_invalid(self):
        """Detection with no reasoning should return is_valid=False."""
        from aiuta_vlmr1.self_questioner.two_pass_questioner import TwoPassSelfQuestioner
        from aiuta_vlmr1.config import Config

        with patch("aiuta_vlmr1.self_questioner.two_pass_questioner.ModelLoader") as mock_cls:
            mock_cls.get_instance.return_value = MagicMock()
            config = Config()
            questioner = TwoPassSelfQuestioner(config)

        kg = SceneKnowledgeGraph()
        detection = Detection(bbox=[0, 0, 1, 1], label="table", reasoning=None)

        result = questioner.process(detection, kg.target_facts, kg, timestep=0)
        assert not result.is_valid
        assert kg.num_objects == 0


class TestRunAttributePassWithImage:
    def test_static_method_returns_attributes(self):
        """run_attribute_pass_with_image should parse attribute JSON from model output."""
        mock_loader = MagicMock()
        mock_loader.device = "cpu"

        proc = MagicMock()
        proc.apply_chat_template.return_value = "text"
        proc_inputs = MagicMock()
        proc_inputs.input_ids = [[1, 2, 3]]
        proc_inputs.to.return_value = proc_inputs
        proc.return_value = proc_inputs
        mock_loader.processor = proc

        import torch
        gen_output = torch.tensor([[1, 2, 3, 10, 11, 12]])
        mock_loader.model.generate.return_value = gen_output

        proc.batch_decode.return_value = [
            '<answer>{"color": "white", "material": "ceramic", "size": "small", '
            '"features": "round shape", "location": "on the counter", "pattern": null}</answer>'
        ]

        from aiuta_vlmr1.self_questioner.two_pass_questioner import TwoPassSelfQuestioner

        fake_image = MagicMock()
        attrs = TwoPassSelfQuestioner.run_attribute_pass_with_image(
            mock_loader, fake_image, category="bowl", timestep=5,
        )

        assert len(attrs) == 5  # pattern is null
        names = {a.name for a in attrs}
        assert names == {"color", "material", "size", "features", "location"}
        for a in attrs:
            assert a.certainty == Certainty.HIGH
            assert a.timestep == 5
