from abc import ABC, abstractmethod

from dataclasses import dataclass, field
from typing import Any, Dict, List

class BaseModel:
    @abstractmethod
    def load(self):
        return
    
    @abstractmethod
    def batch_generate(self, audio_paths: list[str], prompt: str, batch_size: int):
        return
    
@dataclass
class GenerationResult:
    predictions: List[str]
    total_latency_ms: float
    per_sample_latency_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)