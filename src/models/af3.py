import time
from typing import List, Optional, Union
import numpy as np

from src.models.base import BaseModel, GenerationResult


class AudioFlamingo3(BaseModel):
    """
    NVIDIA AudioFlamingo-3 Large Audio-Language Model (LALM)
    using vLLM's multimodal audio engine for accelerated inference.
    """

    def __init__(self, model_config: dict):
        super().__init__(model_config)
        self.llm = None
        self.default_sampling_params = None

    def _resolve_quantization(self) -> Optional[str]:
        quant_cfg = self.model_config.get("quantization", {})
        if not quant_cfg.get("enabled", False):
            return None
        q_type = str(quant_cfg.get("type", "")).lower()
        if q_type in ("nf4", "bnb", "bitsandbytes"):
            return "bitsandbytes"
        return q_type or None

    def load(self):
        try:
            from vllm import LLM, SamplingParams
        except ImportError as e:
            raise ImportError(
                "vLLM is required to run AudioFlamingo3 with the vLLM backend. "
                "Install it via 'pip install vllm'."
            ) from e

        # Model ID defaults to the official HF repository or config override
        model_id = self.model_config.get("pretrained", "nvidia/audio-flamingo-3-hf")
        quant_type = self._resolve_quantization()
        gpu_util = float(self.model_config.get("gpu_memory_utilization", 0.85))
        max_model_len = int(self.model_config.get("max_model_len", 4096))
        max_audio_per_prompt = int(self.model_config.get("max_audio_per_prompt", 5))
        enforce_eager = (quant_type == "bitsandbytes") or self.model_config.get(
            "enforce_eager", False
        )

        llm_kwargs = {
            "model": model_id,
            "trust_remote_code": True,
            "limit_mm_per_prompt": {"audio": max_audio_per_prompt},
            "gpu_memory_utilization": gpu_util,
            "max_model_len": max_model_len,
            "enforce_eager": enforce_eager,
        }
        if quant_type:
            llm_kwargs["quantization"] = quant_type

        self.sampling_params_cls = SamplingParams
        self.llm = LLM(**llm_kwargs)

        top_k = self.generation_cfg.get("top_k")
        self.default_sampling_params = SamplingParams(
            temperature=float(self.generation_cfg.get("temperature", 0.0)),
            top_p=float(self.generation_cfg.get("top_p", 1.0)),
            top_k=int(top_k) if top_k is not None else -1,
            max_tokens=int(self.generation_cfg.get("max_new_tokens", 128)),
        )

    def _format_prompt(self, prompt: str) -> str:
        """Injects audio placeholder token matching AudioFlamingo-3 syntax."""
        if "<sound>" in prompt or "<audio>" in prompt:
            return prompt
        return f"<sound>\n{prompt}"

    def batch_generate(
        self,
        audio_paths: List[str],
        prompts: Union[str, List[str]],
        batch_size: int = 16,
        **kwargs,
    ) -> GenerationResult:
        if self.llm is None:
            raise RuntimeError("Model is not loaded. Call model.load() before batch_generate().")

        t0 = time.perf_counter()
        normalized_prompts = self._normalize_prompts(prompts, len(audio_paths))

        # -------------------------------------------------------------
        # Step 1: Prepare Multimodal Inputs (Audio waveform + prompt)
        # -------------------------------------------------------------
        t_prep_start = time.perf_counter()
        vllm_inputs = []
        for audio_path, prompt in zip(audio_paths, normalized_prompts):
            waveform, sr = self.load_audio_waveform(audio_path, target_sr=16000)
            formatted_prompt = self._format_prompt(prompt)
            vllm_inputs.append({
                "prompt": formatted_prompt,
                "multi_modal_data": {
                    "audio": (waveform, sr)
                }
            })
        prep_time_ms = (time.perf_counter() - t_prep_start) * 1000.0

        # -------------------------------------------------------------
        # Step 2: vLLM Continuous Batch Generation
        # -------------------------------------------------------------
        t_gen_start = time.perf_counter()
        sampling_params = kwargs.get("sampling_params", self.default_sampling_params)
        if self.sampling_params_cls is not None and not isinstance(sampling_params, self.sampling_params_cls):
            top_k = kwargs.get("top_k", self.generation_cfg.get("top_k"))
            sampling_params = self.sampling_params_cls(
                temperature=float(kwargs.get("temperature", self.generation_cfg.get("temperature", 0.0))),
                top_p=float(kwargs.get("top_p", self.generation_cfg.get("top_p", 1.0))),
                top_k=int(top_k) if top_k is not None else -1,
                max_tokens=int(kwargs.get("max_new_tokens", self.generation_cfg.get("max_new_tokens", 128))),
            )

        gen_kwargs = {}
        if sampling_params is not None:
            gen_kwargs["sampling_params"] = sampling_params

        vllm_outputs = self.llm.generate(vllm_inputs, **gen_kwargs)
        gen_time_ms = (time.perf_counter() - t_gen_start) * 1000.0
        total_time_ms = (time.perf_counter() - t0) * 1000.0

        predictions = [out.outputs[0].text.strip() for out in vllm_outputs]
        total_tokens = sum(len(out.outputs[0].token_ids) for out in vllm_outputs)

        return GenerationResult(
            predictions=predictions,
            total_latency_ms=total_time_ms,
            per_sample_latency_ms=total_time_ms / max(len(audio_paths), 1),
            metadata={
                "engine": "vllm",
                "audio_prep_latency_ms": prep_time_ms,
                "vllm_generate_latency_ms": gen_time_ms,
                "generated_tokens": total_tokens,
            },
        )
