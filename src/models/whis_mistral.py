import time
from typing import List, Optional, Union
import numpy as np

from src.models.base import BaseModel, GenerationResult


class WhisperMistral(BaseModel):
    """
    Cascaded Audio-LLM pipeline:
    1. Audio Transcription via WhisperX (CTranslate2 / PyTorch).
    2. Affective Reasoning & Emotion Classification via Mistral using vLLM engine.
    """

    def __init__(self, model_config: dict):
        super().__init__(model_config)
        self.whisper = None
        self.llm = None
        self.default_sampling_params = None

    def _resolve_quantization(self) -> Optional[str]:
        quant_cfg = self.model_config.get("quantization_mistral", {})
        if not quant_cfg.get("enabled", False):
            return None
        q_type = str(quant_cfg.get("type", "")).lower()
        if q_type in ("nf4", "bnb", "bitsandbytes"):
            return "bitsandbytes"
        return q_type or None

    def load(self):
        # 1. Load WhisperX for speech transcription
        try:
            import whisperx
        except ImportError as e:
            raise ImportError(
                "WhisperX is required for WhisperMistral transcription. "
                "Install it via 'pip install whisperx'."
            ) from e

        whisper_model_name = self.model_config.get("pretrained_whisper", "large-v2")
        whisper_device = self.model_config.get("device", "cuda")
        whisper_compute_dtype = (
            self.model_config.get("config_whisper", {}).get("compute_dtype", "float16")
        )

        self.whisper = whisperx.load_model(
            whisper_model_name,
            whisper_device,
            compute_type=whisper_compute_dtype,
        )

        # 2. Load Mistral via vLLM
        try:
            from vllm import LLM, SamplingParams
        except ImportError as e:
            raise ImportError(
                "vLLM is required to run WhisperMistral with the vLLM backend. "
                "Install it via 'pip install vllm'."
            ) from e

        mistral_model_id = self.model_config.get(
            "pretrained_mistral", "mistralai/Mistral-7B-Instruct-v0.3"
        )
        quant_type = self._resolve_quantization()

        # In a cascaded pipeline, reserve VRAM for WhisperX by capping vLLM allocation
        gpu_util = float(self.model_config.get("gpu_memory_utilization", 0.60))
        max_model_len = int(self.model_config.get("max_model_len", 4096))
        enforce_eager = (quant_type == "bitsandbytes") or self.model_config.get(
            "enforce_eager", False
        )

        llm_kwargs = {
            "model": mistral_model_id,
            "gpu_memory_utilization": gpu_util,
            "max_model_len": max_model_len,
            "trust_remote_code": True,
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

    def _format_prompt(self, prompt: str, transcript: str) -> str:
        """Combines original evaluation instruction with the transcribed speech."""
        transcript_text = transcript.strip() if transcript else "[No speech detected]"
        return (
            f"<s>[INST] {prompt}\n\n"
            f"Audio Transcript: \"{transcript_text}\"\n"
            f"Emotion: [/INST]"
        )

    def batch_generate(
        self,
        audio_paths: List[str],
        prompts: Union[str, List[str]],
        batch_size: int = 16,
        **kwargs,
    ) -> GenerationResult:
        if self.whisper is None or self.llm is None:
            raise RuntimeError("Model is not loaded. Call model.load() before batch_generate().")

        import whisperx

        t0 = time.perf_counter()
        normalized_prompts = self._normalize_prompts(prompts, len(audio_paths))

        # Step 1: Transcribe audio inputs via WhisperX
        t_whisper_start = time.perf_counter()
        transcripts: List[str] = []
        for i in range(0, len(audio_paths), batch_size):
            chunk_paths = audio_paths[i : i + batch_size]
            audio_arrays = [whisperx.load_audio(p) for p in chunk_paths]
            for audio_arr in audio_arrays:
                asr_result = self.whisper.transcribe(audio_arr, batch_size=batch_size)
                if isinstance(asr_result, dict):
                    segments = asr_result.get("segments", [])
                    text = " ".join(seg.get("text", "").strip() for seg in segments).strip()
                    if not text and "text" in asr_result:
                        text = str(asr_result["text"]).strip()
                elif isinstance(asr_result, list):
                    text = " ".join(item.get("text", "").strip() for item in asr_result).strip()
                else:
                    text = str(asr_result).strip()
                transcripts.append(text)
        whisper_time_ms = (time.perf_counter() - t_whisper_start) * 1000.0

        # Step 2: Format prompt with transcript
        formatted_prompts = [
            self._format_prompt(p, t)
            for p, t in zip(normalized_prompts, transcripts)
        ]

        # Step 3: Run Continuous-Batch Generation via vLLM
        t_mistral_start = time.perf_counter()
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

        vllm_outputs = self.llm.generate(formatted_prompts, **gen_kwargs)
        mistral_time_ms = (time.perf_counter() - t_mistral_start) * 1000.0
        total_time_ms = (time.perf_counter() - t0) * 1000.0

        predictions = [out.outputs[0].text.strip() for out in vllm_outputs]
        total_generated_tokens = sum(len(out.outputs[0].token_ids) for out in vllm_outputs)

        return GenerationResult(
            predictions=predictions,
            total_latency_ms=total_time_ms,
            per_sample_latency_ms=total_time_ms / max(len(audio_paths), 1),
            metadata={
                "engine": "vllm",
                "whisper_latency_ms": whisper_time_ms,
                "mistral_latency_ms": mistral_time_ms,
                "generated_tokens": total_generated_tokens,
                "transcripts": transcripts,
            },
        )