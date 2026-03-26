"""base.py -- Abstract detector interface (Port)."""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np

@dataclass
class Detection:
    bbox: list[float]
    label: str
    confidence: float = 1.0
    reasoning: str | None = None
    raw_answer: str | None = None
    attributes: dict[str, str] | None = None
    image: object | None = None  # PIL image for downstream passes (e.g. attribute extraction)

@dataclass
class DetectionResult:
    detections: list[Detection]
    raw_output: str
    reasoning_text: str | None
    json_valid: bool
    latency_sec: float
    preprocess_latency_sec: float = 0.0
    generate_latency_sec: float = 0.0
    parse_latency_sec: float = 0.0

class AbstractDetector(ABC):
    @abstractmethod
    def detect(self, image_path: str, target_categories: list[str],
               kg_context: str | None = None) -> DetectionResult: ...

    @abstractmethod
    def detect_from_observation(self, observation: np.ndarray,
                                 target_categories: list[str],
                                 kg_context: str | None = None) -> DetectionResult: ...
