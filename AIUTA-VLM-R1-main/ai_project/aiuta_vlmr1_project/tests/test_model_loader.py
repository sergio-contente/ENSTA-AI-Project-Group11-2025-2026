"""ModelLoader cache keyed by config fingerprint."""
from __future__ import annotations

from aiuta_vlmr1.config import ModelConfig
from aiuta_vlmr1.utils.model_loader import ModelLoader


def test_same_config_same_loader():
    ModelLoader.reset()
    a = ModelConfig(model_id="m1", processor_id="p1")
    b = ModelConfig(model_id="m1", processor_id="p1")
    # Without loading HF, get_instance would load -- we only test fingerprint logic
    from aiuta_vlmr1.utils import model_loader as ml

    assert ml._model_config_fingerprint(a) == ml._model_config_fingerprint(b)


def test_different_config_different_key():
    from aiuta_vlmr1.utils import model_loader as ml

    a = ModelConfig(model_id="m1", processor_id="p1")
    c = ModelConfig(model_id="m2", processor_id="p1")
    assert ml._model_config_fingerprint(a) != ml._model_config_fingerprint(c)
