from __future__ import annotations

import os
from typing import Any

import numpy as np

from vlfm.policy.itm_policy import ITMPolicyV2
from vlfm.vlm.vlmr1_itm_adapter import VLMr1ITMAdapter


class VLMNavITMPolicy(ITMPolicyV2):
    """
    Version of ITMPolicyV2 that uses local VLM-R1 (via adapter) instead of BLIP2ITM (port 12182).
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        if not getattr(self, "_use_vlmr1", False):
            return

        vlmr1_bridge = getattr(self, "_vlmr1_bridge", None)
        if vlmr1_bridge is None:
            return

        model_config = getattr(vlmr1_bridge.config, "model", None)
        if model_config is None:
            return

        adapter = VLMr1ITMAdapter(model_config=model_config)
        self._itm = adapter

    def _update_value_map(self) -> None:
        """
        Throttled value map update for VLM-R1.
        When there are unconfirmed detections in the detection_cloud, injects a high score
        at the position of those detections to guide exploration toward them.
        """
        update_every = int(os.environ.get("VLMR1_VALUE_MAP_UPDATE_EVERY", "3"))
        update_every = max(1, update_every)

        if getattr(self, "_num_steps", 0) % update_every != 0:
            try:
                self._value_map.update_agent_traj(
                    self._observations_cache["robot_xy"],
                    self._observations_cache["robot_heading"],
                )
            except Exception:
                pass
            return

        # First do the normal update via ITM adapter
        super()._update_value_map()

        # Then inject a high score for unconfirmed detections
        # This ensures the agent explores toward objects seen but not yet confirmed
        try:
            obj_map = getattr(self, "_object_map", None)
            if obj_map is None:
                return
            target = getattr(self, "_target_object", "").split("|")[0].strip().lower()
            if not target:
                return
            detection_cloud = getattr(obj_map, "detection_cloud", {})
            if target not in detection_cloud:
                return
            cloud = detection_cloud[target]
            if cloud is None or len(cloud) == 0:
                return
            # Inject a gaussian over the centroid of the latest detections (last 5000 positions)
            recent_cloud = cloud[-5000:] if len(cloud) > 5000 else cloud
            positions_2d = recent_cloud[:, :2]
            centroid = positions_2d.mean(axis=0)
            map_size = self._value_map.size
            ppm = self._value_map.pixels_per_meter
            origin_arr = self._value_map._episode_pixel_origin
            row_in_map = int(self._value_map._value_map.shape[0] - (int(-centroid[0] * ppm) + origin_arr[0]))
            col_in_map = int(int(-centroid[1] * ppm) + origin_arr[1])
            row_in_map = max(0, min(map_size - 1, row_in_map))
            col_in_map = max(0, min(map_size - 1, col_in_map))
            ys, xs = np.ogrid[0:map_size, 0:map_size]
            dist_px = np.sqrt((xs - col_in_map) ** 2 + (ys - row_in_map) ** 2)
            sigma_px = max(30, int(2.0 * ppm))
            gaussian = 0.85 * np.exp(-(dist_px**2) / (2 * sigma_px**2))
            current = self._value_map._value_map[:, :, 0]
            self._value_map._value_map[:, :, 0] = np.maximum(current, gaussian.astype(np.float32))
        except Exception as e:
            print(f"[VLMr1ITMPolicy] detection_cloud injection error: {e}")

