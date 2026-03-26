"""
model_loader.py -- Config-keyed model loader for VLM-R1.

Refactored from: benchmark_ovd.py::load_model_and_processor()

Multiple instances are allowed: the same ``ModelConfig`` fingerprint always maps to
the same loader; a different config gets a separate loader (for offline
benchmarking with different checkpoints or dtypes).
"""

from __future__ import annotations

import gc
import threading
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from transformers import AutoModelForVision2Seq as Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from ..config import ModelConfig


def _model_config_fingerprint(model_config: ModelConfig) -> str:
    return (
        f"{model_config.model_id}|{model_config.processor_id}|"
        f"{model_config.torch_dtype}|{model_config.device_map}|"
        f"{model_config.max_new_tokens}"
    )


def model_configs_equivalent(a: ModelConfig, b: ModelConfig) -> bool:
    """True if two configs map to the same cached ``ModelLoader`` instance."""
    return _model_config_fingerprint(a) == _model_config_fingerprint(b)


class ModelLoader:
    """
    Holds the VLM-R1 model and processor for one configuration fingerprint.

    Usage:
        loader = ModelLoader.get_instance(config.model)
        model, processor = loader.model, loader.processor
        x = x.to(loader.device)
    """

    _instances: dict[str, ModelLoader] = {}
    _lock = threading.Lock()

    def __init__(self):
        raise RuntimeError("Use ModelLoader.get_instance() instead.")

    @classmethod
    def get_instance(cls, model_config: ModelConfig | None = None) -> ModelLoader:
        """Get or create the loader for this ``ModelConfig`` (keyed by fingerprint)."""
        if model_config is None:
            raise ValueError("model_config is required")
        key = _model_config_fingerprint(model_config)
        if key not in cls._instances:
            with cls._lock:
                if key not in cls._instances:
                    instance = object.__new__(cls)
                    instance._init(model_config)
                    cls._instances[key] = instance
        return cls._instances[key]

    def _init(self, model_config: ModelConfig) -> None:
        from transformers import AutoModelForVision2Seq as Qwen2_5_VLForConditionalGeneration, AutoProcessor

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(model_config.torch_dtype, torch.bfloat16)

        print(f"[ModelLoader] Loading model: {model_config.model_id}")
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_config.model_id,
            torch_dtype=torch_dtype,
            device_map=model_config.device_map,
            trust_remote_code=True,
        )

        print(f"[ModelLoader] Loading processor: {model_config.processor_id}")
        self._processor = AutoProcessor.from_pretrained(
            model_config.processor_id,
            trust_remote_code=True,
        )

        self._config = model_config
        self._device = next(self._model.parameters()).device
        print(f"[ModelLoader] Model loaded successfully (device={self._device}).")

    @property
    def model(self):
        return self._model

    @property
    def processor(self):
        return self._processor

    @property
    def config(self) -> ModelConfig:
        return self._config

    @property
    def device(self) -> torch.device:
        return self._device

    @classmethod
    def reset(cls, model_config: ModelConfig | None = None) -> None:
        """Drop cached loader(s). Frees GPU memory when possible."""
        with cls._lock:
            if model_config is None:
                for inst in cls._instances.values():
                    cls._drop_instance(inst)
                cls._instances.clear()
            else:
                key = _model_config_fingerprint(model_config)
                inst = cls._instances.pop(key, None)
                if inst is not None:
                    cls._drop_instance(inst)

    @staticmethod
    def _drop_instance(inst: ModelLoader) -> None:
        del inst._model
        del inst._processor
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        gc.collect()
