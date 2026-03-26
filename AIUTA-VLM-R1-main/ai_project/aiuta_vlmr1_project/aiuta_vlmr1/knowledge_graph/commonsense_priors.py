"""
commonsense_priors.py -- Room-object association priors for navigation.

Provides structured common-sense knowledge:
  - Which room types typically contain which objects
  - Object co-occurrence (objects usually found together)
  - Spatial priors (where in a room an object is usually located)

This enables the agent to:
  1. Prioritize exploring rooms likely to contain the target
  2. Use visible "anchor objects" to infer room type
  3. Generate better questions ("Is the toilet in the main bathroom
     or the guest bathroom?")

The priors are static (not learned). They're injected into:
  - VLM-R1 prompt via KG context (Phase 4)
  - Navigation policy via frontier scoring bias
  - Question generation via discriminative attribute selection

Usage:
    priors = CommonSensePriors()
    rooms = priors.likely_rooms("toilet")
    # -> [("bathroom", 0.95), ("restroom", 0.85)]

    anchors = priors.room_anchor_objects("bathroom")
    # -> ["sink", "mirror", "bathtub", "shower", "towel rack"]

    cooccur = priors.cooccurring_objects("toilet")
    # -> ["toilet paper", "sink", "mirror", "towel"]
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass
class RoomAssociation:
    """Association between an object and a room type."""
    room: str
    probability: float    # how likely the object is in this room (0-1)


@dataclass
class ObjectCooccurrence:
    """Objects commonly found together."""
    object_name: str
    strength: float       # how often they co-occur (0-1)


# =========================================================================
# Room-Object Associations (from indoor scene statistics / common sense)
# Categories match HM3DSem / GOAT-Bench / CoIN-Bench
# =========================================================================

OBJECT_TO_ROOMS: dict[str, list[RoomAssociation]] = {
    # Bathroom objects
    "toilet": [
        RoomAssociation("bathroom", 0.95),
        RoomAssociation("restroom", 0.90),
    ],
    "bathtub": [
        RoomAssociation("bathroom", 0.95),
    ],
    "shower": [
        RoomAssociation("bathroom", 0.90),
    ],
    "sink": [
        RoomAssociation("bathroom", 0.50),
        RoomAssociation("kitchen", 0.45),
        RoomAssociation("laundry room", 0.20),
    ],
    "mirror": [
        RoomAssociation("bathroom", 0.55),
        RoomAssociation("bedroom", 0.30),
        RoomAssociation("hallway", 0.15),
    ],
    "towel": [
        RoomAssociation("bathroom", 0.85),
        RoomAssociation("kitchen", 0.15),
    ],
    "washbasin": [
        RoomAssociation("bathroom", 0.90),
    ],

    # Kitchen objects
    "refrigerator": [
        RoomAssociation("kitchen", 0.95),
    ],
    "oven": [
        RoomAssociation("kitchen", 0.95),
    ],
    "microwave": [
        RoomAssociation("kitchen", 0.90),
    ],
    "toaster": [
        RoomAssociation("kitchen", 0.90),
    ],
    "kitchen cabinet": [
        RoomAssociation("kitchen", 0.95),
    ],
    "dishwasher": [
        RoomAssociation("kitchen", 0.90),
    ],
    "stove": [
        RoomAssociation("kitchen", 0.95),
    ],

    # Bedroom objects
    "bed": [
        RoomAssociation("bedroom", 0.95),
        RoomAssociation("guest room", 0.80),
    ],
    "wardrobe": [
        RoomAssociation("bedroom", 0.80),
        RoomAssociation("hallway", 0.15),
    ],
    "dresser": [
        RoomAssociation("bedroom", 0.80),
    ],
    "nightstand": [
        RoomAssociation("bedroom", 0.90),
    ],
    "pillow": [
        RoomAssociation("bedroom", 0.60),
        RoomAssociation("living room", 0.35),
    ],
    "blanket": [
        RoomAssociation("bedroom", 0.70),
        RoomAssociation("living room", 0.25),
    ],

    # Living room objects
    "couch": [
        RoomAssociation("living room", 0.80),
        RoomAssociation("family room", 0.60),
        RoomAssociation("office", 0.15),
    ],
    "sofa": [
        RoomAssociation("living room", 0.80),
        RoomAssociation("family room", 0.60),
    ],
    "tv": [
        RoomAssociation("living room", 0.65),
        RoomAssociation("bedroom", 0.25),
        RoomAssociation("family room", 0.40),
    ],
    "coffee table": [
        RoomAssociation("living room", 0.85),
    ],
    "bookshelf": [
        RoomAssociation("living room", 0.40),
        RoomAssociation("office", 0.35),
        RoomAssociation("bedroom", 0.20),
    ],
    "fireplace": [
        RoomAssociation("living room", 0.80),
    ],
    "armchair": [
        RoomAssociation("living room", 0.70),
        RoomAssociation("bedroom", 0.20),
    ],

    # Multi-room objects
    "table": [
        RoomAssociation("dining room", 0.40),
        RoomAssociation("kitchen", 0.25),
        RoomAssociation("living room", 0.20),
        RoomAssociation("office", 0.15),
    ],
    "chair": [
        RoomAssociation("dining room", 0.30),
        RoomAssociation("office", 0.25),
        RoomAssociation("kitchen", 0.20),
        RoomAssociation("bedroom", 0.15),
    ],
    "cabinet": [
        RoomAssociation("kitchen", 0.40),
        RoomAssociation("bathroom", 0.20),
        RoomAssociation("dining room", 0.20),
        RoomAssociation("bedroom", 0.15),
    ],
    "lamp": [
        RoomAssociation("bedroom", 0.30),
        RoomAssociation("living room", 0.30),
        RoomAssociation("office", 0.20),
    ],
    "picture": [
        RoomAssociation("living room", 0.30),
        RoomAssociation("hallway", 0.25),
        RoomAssociation("bedroom", 0.25),
        RoomAssociation("dining room", 0.15),
    ],
    "clock": [
        RoomAssociation("living room", 0.35),
        RoomAssociation("kitchen", 0.30),
        RoomAssociation("hallway", 0.20),
    ],
    "plant": [
        RoomAssociation("living room", 0.35),
        RoomAssociation("hallway", 0.20),
        RoomAssociation("bedroom", 0.15),
        RoomAssociation("kitchen", 0.15),
    ],
    "rug": [
        RoomAssociation("living room", 0.35),
        RoomAssociation("bedroom", 0.30),
        RoomAssociation("hallway", 0.20),
    ],
    "shelf": [
        RoomAssociation("living room", 0.25),
        RoomAssociation("kitchen", 0.25),
        RoomAssociation("bathroom", 0.15),
        RoomAssociation("office", 0.20),
    ],

    # Hallway / entrance objects
    "coat rack": [
        RoomAssociation("hallway", 0.70),
        RoomAssociation("entrance", 0.60),
    ],
    "shoe rack": [
        RoomAssociation("hallway", 0.70),
        RoomAssociation("entrance", 0.60),
    ],
    "hook": [
        RoomAssociation("hallway", 0.40),
        RoomAssociation("bathroom", 0.30),
        RoomAssociation("entrance", 0.25),
    ],

    # Office objects
    "desk": [
        RoomAssociation("office", 0.70),
        RoomAssociation("bedroom", 0.25),
    ],
    "computer": [
        RoomAssociation("office", 0.70),
        RoomAssociation("bedroom", 0.20),
    ],
}


# Room anchor objects: when you see these, you know what room you're in
ROOM_ANCHORS: dict[str, list[str]] = {
    "bathroom": ["toilet", "bathtub", "shower", "sink", "towel rack", "bath mat", "mirror", "towel"],
    "kitchen": ["refrigerator", "oven", "stove", "kitchen counter", "dishwasher", "microwave"],
    "bedroom": ["bed", "nightstand", "dresser", "wardrobe", "alarm clock"],
    "living room": ["couch", "sofa", "coffee table", "tv", "fireplace", "armchair"],
    "dining room": ["dining table", "dining chair", "china cabinet", "chandelier"],
    "office": ["desk", "computer", "office chair", "printer", "filing cabinet"],
    "hallway": ["coat rack", "shoe rack", "console table", "umbrella stand"],
    "laundry room": ["washing machine", "dryer", "laundry basket", "ironing board"],
    "garage": ["car", "workbench", "tool rack", "bicycle"],
}


# Object co-occurrence: objects usually found near each other
COOCCURRENCE: dict[str, list[ObjectCooccurrence]] = {
    "toilet": [
        ObjectCooccurrence("toilet paper", 0.90),
        ObjectCooccurrence("sink", 0.85),
        ObjectCooccurrence("mirror", 0.70),
        ObjectCooccurrence("towel", 0.65),
        ObjectCooccurrence("trash can", 0.50),
    ],
    "bed": [
        ObjectCooccurrence("pillow", 0.95),
        ObjectCooccurrence("nightstand", 0.80),
        ObjectCooccurrence("lamp", 0.70),
        ObjectCooccurrence("blanket", 0.85),
        ObjectCooccurrence("dresser", 0.55),
    ],
    "couch": [
        ObjectCooccurrence("coffee table", 0.70),
        ObjectCooccurrence("tv", 0.65),
        ObjectCooccurrence("cushion", 0.80),
        ObjectCooccurrence("lamp", 0.50),
        ObjectCooccurrence("rug", 0.45),
    ],
    "refrigerator": [
        ObjectCooccurrence("oven", 0.70),
        ObjectCooccurrence("microwave", 0.60),
        ObjectCooccurrence("kitchen counter", 0.80),
        ObjectCooccurrence("sink", 0.75),
    ],
    "desk": [
        ObjectCooccurrence("chair", 0.85),
        ObjectCooccurrence("computer", 0.60),
        ObjectCooccurrence("lamp", 0.50),
        ObjectCooccurrence("bookshelf", 0.40),
    ],
}


class CommonSensePriors:
    """
    Provides common-sense room-object associations for navigation.

    Integration points:
      1. KG context string: "Toilets are typically in bathrooms.
         Anchor objects for bathrooms: sink, mirror, bathtub."
      2. Frontier scoring: bias VLFM value map towards rooms likely
         to contain the target
      3. Room type inference: "I see a sink and mirror ->probably bathroom
         ->good place to look for toilet"
    """

    def likely_rooms(self, object_category: str) -> list[RoomAssociation]:
        """
        Return rooms where this object is likely found, sorted by probability.

        Example: likely_rooms("toilet") -> [("bathroom", 0.95), ("restroom", 0.90)]
        """
        cat = object_category.lower().strip()
        assocs = OBJECT_TO_ROOMS.get(cat, [])
        return sorted(assocs, key=lambda a: -a.probability)

    def room_anchor_objects(self, room_type: str) -> list[str]:
        """
        Return objects that are strong indicators of a room type.

        Example: room_anchor_objects("bathroom") -> ["toilet", "bathtub", ...]
        """
        return ROOM_ANCHORS.get(room_type.lower().strip(), [])

    def cooccurring_objects(self, object_category: str) -> list[ObjectCooccurrence]:
        """
        Return objects commonly found near this object.

        Example: cooccurring_objects("toilet") -> ["toilet paper", "sink", ...]
        """
        cat = object_category.lower().strip()
        return COOCCURRENCE.get(cat, [])

    def infer_room_type(self, visible_objects: list[str]) -> list[tuple[str, float]]:
        """
        Given a list of visible objects, infer the most likely room type.

        Uses anchor object matching: count how many anchors for each room
        type are visible, normalize by total anchors.

        Example:
            infer_room_type(["sink", "mirror", "towel"])
            -> [("bathroom", 0.50), ("kitchen", 0.08)]
        """
        visible_set = {o.lower().strip() for o in visible_objects}
        room_scores: dict[str, float] = {}

        for room, anchors in ROOM_ANCHORS.items():
            if not anchors:
                continue
            matched = sum(1 for a in anchors if a in visible_set)
            if matched > 0:
                room_scores[room] = matched / len(anchors)

        return sorted(room_scores.items(), key=lambda x: -x[1])

    def get_navigation_context(self, target_category: str) -> str:
        """
        Generate a natural language navigation hint for the target.

        This string is injected into VLM-R1's prompt to guide reasoning.

        Example for "toilet":
            "The target object 'toilet' is typically found in: bathroom (95%),
             restroom (90%). When navigating, look for anchor objects that
             indicate a bathroom: sink, bathtub, shower, towel rack.
             Objects commonly found near toilets: toilet paper, sink, mirror."
        """
        rooms = self.likely_rooms(target_category)
        cooccur = self.cooccurring_objects(target_category)

        if not rooms:
            return f"No specific room association known for '{target_category}'."

        parts = [
            f"The target '{target_category}' is typically found in: "
            + ", ".join(f"{r.room} ({r.probability:.0%})" for r in rooms[:3])
            + "."
        ]

        # Add anchor objects for the most likely room
        top_room = rooms[0].room
        anchors = self.room_anchor_objects(top_room)
        if anchors:
            parts.append(
                f"Anchor objects indicating a {top_room}: "
                + ", ".join(anchors[:5]) + "."
            )

        if cooccur:
            parts.append(
                f"Objects commonly near {target_category}: "
                + ", ".join(c.object_name for c in cooccur[:4]) + "."
            )

        return " ".join(parts)

    def get_frontier_bias(
        self,
        target_category: str,
        frontier_visible_objects: dict[str, list[str]],
    ) -> dict[str, float]:
        """
        Compute a bias score for each frontier based on visible objects.

        Args:
            target_category: what we're looking for
            frontier_visible_objects: {frontier_id: [visible_objects]}

        Returns:
            {frontier_id: bias_score} where higher = more promising

        This can be added to VLFM's value map scores to prioritize
        frontiers that look like they lead to the right room.
        """
        target_rooms = self.likely_rooms(target_category)
        if not target_rooms:
            return {fid: 0.0 for fid in frontier_visible_objects}

        target_room_set = {r.room for r in target_rooms}

        biases = {}
        for fid, visible in frontier_visible_objects.items():
            inferred = self.infer_room_type(visible)
            score = 0.0
            for room, room_score in inferred:
                if room in target_room_set:
                    # Weight by both room inference confidence and
                    # room-object association probability
                    room_prob = next(
                        (r.probability for r in target_rooms if r.room == room), 0
                    )
                    score = max(score, room_score * room_prob)
            biases[fid] = score

        return biases
