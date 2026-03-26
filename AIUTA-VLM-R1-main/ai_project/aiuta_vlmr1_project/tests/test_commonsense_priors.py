"""Unit tests for common-sense priors."""
from aiuta_vlmr1.knowledge_graph.commonsense_priors import CommonSensePriors

class TestCommonSensePriors:
    def setup_method(self):
        self.priors = CommonSensePriors()

    def test_toilet_in_bathroom(self):
        rooms = self.priors.likely_rooms("toilet")
        assert len(rooms) >= 1
        assert rooms[0].room == "bathroom"
        assert rooms[0].probability >= 0.9

    def test_bed_in_bedroom(self):
        rooms = self.priors.likely_rooms("bed")
        assert rooms[0].room == "bedroom"

    def test_unknown_object(self):
        rooms = self.priors.likely_rooms("flying_saucer")
        assert rooms == []

    def test_room_anchors(self):
        anchors = self.priors.room_anchor_objects("bathroom")
        assert "toilet" in anchors
        assert "bathtub" in anchors

    def test_infer_room_from_objects(self):
        rooms = self.priors.infer_room_type(["sink", "mirror", "towel"])
        assert len(rooms) >= 1
        assert rooms[0][0] == "bathroom"

    def test_infer_kitchen(self):
        rooms = self.priors.infer_room_type(["refrigerator", "oven", "microwave"])
        assert rooms[0][0] == "kitchen"

    def test_cooccurrence(self):
        cooccur = self.priors.cooccurring_objects("toilet")
        names = [c.object_name for c in cooccur]
        assert "sink" in names
        assert "toilet paper" in names

    def test_navigation_context(self):
        ctx = self.priors.get_navigation_context("toilet")
        assert "bathroom" in ctx.lower()
        assert "sink" in ctx.lower() or "bathtub" in ctx.lower()

    def test_frontier_bias(self):
        biases = self.priors.get_frontier_bias("toilet", {
            "frontier_A": ["sink", "mirror", "towel"],     # looks like bathroom
            "frontier_B": ["couch", "tv", "coffee table"],  # looks like living room
            "frontier_C": [],                               # unknown
        })
        assert biases["frontier_A"] > biases["frontier_B"]
        assert biases["frontier_B"] == 0.0 or biases["frontier_A"] > biases["frontier_B"]
