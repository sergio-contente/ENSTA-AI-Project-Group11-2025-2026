"""
config.py -- Global configuration for the AIUTA-VLM-R1 pipeline.

Loaded from YAML config files. Drives the Strategy pattern:
  config.detector_type -> which detector to instantiate
  config.questioner_type -> which self-questioner to use
  config.trigger_type -> which interaction trigger to use
"""

from __future__ import annotations

import yaml
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum


class DetectorType(Enum):
    VLMR1 = "vlmr1"
    GROUNDING_DINO = "grounding_dino"


class QuestionerType(Enum):
    VLMR1 = "vlmr1"          # parse <think> block directly
    TWO_PASS = "two_pass"    # detection triples + attribute forward pass
    ORIGINAL = "original"     # AIUTA multi-call (LLaVA + LLM)


class TriggerType(Enum):
    KG = "kg"                 # graph matching (ours)
    ORIGINAL = "original"     # LLM P_score prompt (baseline)


@dataclass
class ModelConfig:
    """VLM-R1 model configuration."""
    model_id: str = "omlab/VLM-R1-Qwen2.5VL-3B-OVD-0321"
    processor_id: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    torch_dtype: str = "bfloat16"
    device_map: str = "auto"
    max_new_tokens: int = 1024
    temperature: float = 0.0


@dataclass
class KGConfig:
    """Knowledge Graph configuration."""
    certainty_threshold: float = 0.5        # min certainty to keep attribute
    max_instances_per_category: int = 20    # prune older instances if exceeded
    use_llm_fallback: bool = True           # fallback to LLM for triple extraction
    llm_fallback_model: str = ""            # model for fallback (empty = use VLM-R1)


@dataclass
class TriggerConfig:
    """Interaction Trigger thresholds."""
    tau_stop: float = 0.8       # alignment score >= this -> STOP
    tau_skip: float = 0.2       # alignment score < this -> CONTINUE (skip)
    max_interaction_rounds: int = 4
    max_questions_per_episode: int = 6  # budget: force STOP after this many questions


@dataclass
class Config:
    """Top-level experiment configuration."""
    # Component selection (Strategy pattern)
    detector_type: DetectorType = DetectorType.VLMR1
    questioner_type: QuestionerType = QuestionerType.VLMR1
    trigger_type: TriggerType = TriggerType.KG

    # Sub-configs
    model: ModelConfig = field(default_factory=ModelConfig)
    # Optional: dedicated model/processor for second-pass attribute extraction.
    second_pass_model: ModelConfig | None = None
    kg: KGConfig = field(default_factory=KGConfig)
    trigger: TriggerConfig = field(default_factory=TriggerConfig)

    # Evaluation
    coin_bench_path: str = ""
    habitat_scene_dir: str = ""
    output_dir: str = "./results"
    seed: int = 42
    # Optional block from YAML: entropy_threshold, abstention_rule, etc.
    _idkvqa_eval: dict = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        """Load configuration from a YAML file."""
        with open(path, "r") as f:
            raw = yaml.safe_load(f)

        cfg = cls()
        if "detector_type" in raw:
            cfg.detector_type = DetectorType(raw["detector_type"])
        if "questioner_type" in raw:
            cfg.questioner_type = QuestionerType(raw["questioner_type"])
        if "trigger_type" in raw:
            cfg.trigger_type = TriggerType(raw["trigger_type"])

        if "model" in raw:
            for k, v in raw["model"].items():
                if hasattr(cfg.model, k):
                    setattr(cfg.model, k, v)
        if "second_pass_model" in raw and isinstance(raw["second_pass_model"], dict):
            sp = ModelConfig()
            for k, v in raw["second_pass_model"].items():
                if hasattr(sp, k):
                    setattr(sp, k, v)
            cfg.second_pass_model = sp
        if "idkvqa_eval" in raw and isinstance(raw["idkvqa_eval"], dict):
            cfg._idkvqa_eval = dict(raw["idkvqa_eval"])

        if "kg" in raw:
            for k, v in raw["kg"].items():
                if hasattr(cfg.kg, k):
                    setattr(cfg.kg, k, v)

        if "trigger" in raw:
            for k, v in raw["trigger"].items():
                if hasattr(cfg.trigger, k):
                    setattr(cfg.trigger, k, v)

        for key in ("coin_bench_path", "habitat_scene_dir", "output_dir", "seed"):
            if key in raw:
                setattr(cfg, key, raw[key])

        return cfg

    def to_serializable_dict(self) -> dict:
        """Full config snapshot for benchmark JSON (not only a path string)."""
        return {
            "detector_type": self.detector_type.value,
            "questioner_type": self.questioner_type.value,
            "trigger_type": self.trigger_type.value,
            "model": {
                "model_id": self.model.model_id,
                "processor_id": self.model.processor_id,
                "torch_dtype": self.model.torch_dtype,
                "device_map": self.model.device_map,
                "max_new_tokens": self.model.max_new_tokens,
                "temperature": self.model.temperature,
            },
            "second_pass_model": (
                {
                    "model_id": self.second_pass_model.model_id,
                    "processor_id": self.second_pass_model.processor_id,
                    "torch_dtype": self.second_pass_model.torch_dtype,
                    "device_map": self.second_pass_model.device_map,
                    "max_new_tokens": self.second_pass_model.max_new_tokens,
                    "temperature": self.second_pass_model.temperature,
                } if self.second_pass_model is not None else None
            ),
            "kg": {
                "certainty_threshold": self.kg.certainty_threshold,
                "max_instances_per_category": self.kg.max_instances_per_category,
                "use_llm_fallback": self.kg.use_llm_fallback,
                "llm_fallback_model": self.kg.llm_fallback_model,
            },
            "trigger": {
                "tau_stop": self.trigger.tau_stop,
                "tau_skip": self.trigger.tau_skip,
                "max_interaction_rounds": self.trigger.max_interaction_rounds,
            },
            "coin_bench_path": self.coin_bench_path,
            "habitat_scene_dir": self.habitat_scene_dir,
            "output_dir": self.output_dir,
            "seed": self.seed,
            "idkvqa_eval": dict(self._idkvqa_eval),
        }

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to YAML."""
        data = {
            "detector_type": self.detector_type.value,
            "questioner_type": self.questioner_type.value,
            "trigger_type": self.trigger_type.value,
            "model": {
                "model_id": self.model.model_id,
                "processor_id": self.model.processor_id,
                "torch_dtype": self.model.torch_dtype,
                "device_map": self.model.device_map,
                "max_new_tokens": self.model.max_new_tokens,
                "temperature": self.model.temperature,
            },
            "kg": {
                "certainty_threshold": self.kg.certainty_threshold,
                "max_instances_per_category": self.kg.max_instances_per_category,
                "use_llm_fallback": self.kg.use_llm_fallback,
            },
            "trigger": {
                "tau_stop": self.trigger.tau_stop,
                "tau_skip": self.trigger.tau_skip,
                "max_interaction_rounds": self.trigger.max_interaction_rounds,
            },
            "output_dir": self.output_dir,
            "seed": self.seed,
        }
        if self.second_pass_model is not None:
            data["second_pass_model"] = {
                "model_id": self.second_pass_model.model_id,
                "processor_id": self.second_pass_model.processor_id,
                "torch_dtype": self.second_pass_model.torch_dtype,
                "device_map": self.second_pass_model.device_map,
                "max_new_tokens": self.second_pass_model.max_new_tokens,
                "temperature": self.second_pass_model.temperature,
            }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
