from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


@dataclass
class GenerationResult:
    predictions: List[str]
    total_latency_ms: float
    per_sample_latency_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseModel(ABC):
    def __init__(self, model_config: dict):
        self.model_config = model_config
        self.device = model_config.get("device", "cuda")
        self.generation_cfg = model_config.get("generation", {})
        self.llm = None
        self.default_sampling_params = None
        self.sampling_params_cls = None

    @abstractmethod
    def load(self):
        """Loads model weights, tokenizer/processor, or vLLM engine."""
        pass

    @abstractmethod
    def batch_generate(
        self,
        audio_paths: List[str],
        prompts: Union[str, List[str]],
        batch_size: int = 16,
        **kwargs,
    ) -> GenerationResult:
        """Runs batch inference over a list of audio file paths and prompts."""
        pass

    def _normalize_prompts(
        self, prompts: Union[str, List[str]], count: int
    ) -> List[str]:
        """Ensures prompts is a list matching the number of audio paths."""
        if isinstance(prompts, str):
            return [prompts] * count
        if len(prompts) != count:
            raise ValueError(
                f"Number of prompts ({len(prompts)}) does not match number of audio paths ({count})"
            )
        return prompts

    @staticmethod
    def load_audio_waveform(
        audio_path: str, target_sr: int = 16000
    ) -> Tuple[np.ndarray, int]:
        """Loads audio from disk, resamples to target_sr, converts to mono, and returns numpy array."""
        try:
            import torchaudio
            waveform, sr = torchaudio.load(audio_path)
            if sr != target_sr:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
                waveform = resampler(waveform)
                sr = target_sr
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            return waveform.squeeze().cpu().numpy(), sr
        except Exception as e:
            raise RuntimeError(f"Failed loading audio file '{audio_path}': {e}") from e