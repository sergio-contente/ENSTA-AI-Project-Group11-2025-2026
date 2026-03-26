import os

import numpy as np
import pytest


class _FakeStepRes:
    def __init__(self, signal: str):
        class _Sig:
            def __init__(self, v: str):
                self.value = v

        self.signal = _Sig(signal)


class _FakeBridge:
    def __init__(self, signal: str):
        self._signal = signal
        self.calls = 0

    def pipeline_step(self, rgb, timestep: int):
        assert isinstance(rgb, np.ndarray)
        self.calls += 1
        return _FakeStepRes(self._signal)


def _dummy_inputs():
    depth = np.ones((32, 32), dtype=np.float32) * 0.5
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[10:20, 10:20] = 1
    tf = np.eye(4, dtype=np.float32)
    rgb = np.zeros((32, 32, 3), dtype=np.uint8)
    return depth, mask, tf, rgb


def test_object_point_cloud_map_vlmr1_no_crash_and_stop_confirms_target():
    os.environ["COIN_USE_VLMR1"] = "1"
    pytest.importorskip("hydra")
    from vlfm.mapping.object_point_cloud_map import ObjectPointCloudMap

    bridge = _FakeBridge(signal="stop")
    opcm = ObjectPointCloudMap(
        erosion_size=1,
        vlm_agent_brain=None,
        llm_agent_brain=None,
        vlm_oracle=None,
        vlmr1_bridge=bridge,
        use_vlmr1=True,
    )
    depth, mask, tf, rgb = _dummy_inputs()
    ok = opcm.update_map(
        object_name="chair",
        depth_img=depth,
        object_mask=mask,
        tf_camera_to_episodic=tf,
        min_depth=0.1,
        max_depth=5.0,
        fx=200.0,
        fy=200.0,
        llama_promt=None,
        llava_prompt=None,
        rgb_image=rgb,
        target_object="chair",
        total_num_steps=0,
        ep_id=0,
    )
    assert ok is True
    assert bridge.calls == 1
    assert opcm.has_object("chair")


def test_object_point_cloud_map_vlmr1_continue_does_not_confirm_target():
    pytest.importorskip("hydra")
    from vlfm.mapping.object_point_cloud_map import ObjectPointCloudMap

    bridge = _FakeBridge(signal="continue")
    opcm = ObjectPointCloudMap(
        erosion_size=1,
        vlm_agent_brain=None,
        llm_agent_brain=None,
        vlm_oracle=None,
        vlmr1_bridge=bridge,
        use_vlmr1=True,
    )
    depth, mask, tf, rgb = _dummy_inputs()
    ok = opcm.update_map(
        object_name="chair",
        depth_img=depth,
        object_mask=mask,
        tf_camera_to_episodic=tf,
        min_depth=0.1,
        max_depth=5.0,
        fx=200.0,
        fy=200.0,
        llama_promt=None,
        llava_prompt=None,
        rgb_image=rgb,
        target_object="chair",
        total_num_steps=0,
        ep_id=0,
    )
    assert ok is False
    assert bridge.calls == 1
    assert not opcm.has_object("chair")
    assert "chair" in opcm.detection_cloud

