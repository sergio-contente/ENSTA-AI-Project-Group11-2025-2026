from __future__ import annotations

from typing import List, Tuple, Any

import numpy as np

from vlfm.mapping.frontier_map import FrontierMap


class VLMr1FrontierMap(FrontierMap):
    """
    FrontierMap que troca o encoder de `BLIP2ITMClient` por um adapter local
    com a interface `.cosine(image: np.ndarray, txt: str) -> float`.
    """

    def __init__(self, encoder: Any) -> None:
        # Do not call super().__init__() because it instantiates BLIP2ITMClient.
        self.encoder = encoder
        self.frontiers = []

    def _encode(self, image: np.ndarray, text: str) -> float:
        return float(self.encoder.cosine(image, text))

