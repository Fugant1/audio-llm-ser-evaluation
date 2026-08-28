import time
import torch

from src.models.base import BaseModel, GenerationResult

import whisperx
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

class WhisperMistral(BaseModel):
    def __init__(self, model_config: dict):
        super().__init__(model_config)
        self.whisper = None
        self.tokenizer = None
        self.llm = None
        self.model_config = model_config

    def get_whisperx(self):
        return whisperx.load_model(self.model_config["pretrained_whisper"], self.model_config["device"], compute_type=self.model_config["config_whisper"]["compute_dtype"])

    def get_mistral(self):
        model_id = self.model_config["pretrained_mistral"]
        if self.model_config["quantization_mistral"]["enabled"] == True:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type= self.model_config["quantization_mistral"]["type"],
                bnb_4bit_compute_dtype= torch.bfloat16 if self.model_config["quantization_mistral"]["compute_dtype"] == "bfloat16" else None,
                bnb_4bit_use_double_quant=True
                )
        tokenizer = AutoTokenizer(model_id)
        model = AutoModelForCausalLM(
            model_id,
            torch_dtype = torch.bfloat16 if self.model_config["quantization_mistral"]["compute_dtype"] == "bfloat16" else None,
            quantization_config = bnb_config,
            attn_implementation = self.model_config["config_msitral"]["attn_implementation"],
            do_sample = self.model_config["config_mistral"]["do_sample"],
            device_map = self.model_config["config_mistral"]["device_map"]
        )

        return tokenizer, model

    def load(self):
        self.whisper = self.get_whisperx()
        self.tokenizer, self.llm = self.get_mistral()

    def batch_generate(self, audio_paths: list[str], prompts: list[str], batch_size: int = 16) -> GenerationResult:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
    
        t_whisper_start = time.perf_counter()
        audio_arrays = [whisperx.load_audio(p) for p in audio_paths]
        asr_results = self.whisperx.transcribe(audio_arrays, batch_size=batch_size)
        transcripts = [res["text"].strip() for res in asr_results]
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        whisper_time_ms = (time.perf_counter() - t_whisper_start) * 1000

        t_mistral_start = time.perf_counter()
        formatted_inputs = [self._format_prompt(p, t) for p, t in zip(prompts, transcripts)]
        inputs = self.mistral_tokenizer(formatted_inputs, return_tensors="pt", padding=True, truncation=True).to(self.mistral.device)
    
        with torch.inference_mode():
            outputs = self.mistral.generate(**inputs, max_new_tokens=128)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        mistral_time_ms = (time.perf_counter() - t_mistral_start) * 1000
    
        total_time_ms = (time.perf_counter() - t0) * 1000
        decoded_texts = self.mistral_tokenizer.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
    
        return GenerationResult(
                predictions=decoded_texts,
                total_latency_ms=total_time_ms,
                per_sample_latency_ms=total_time_ms / max(len(audio_paths), 1),
                metadata={
                    "whisper_latency_ms": whisper_time_ms,
                    "mistral_latency_ms": mistral_time_ms,
                    "generated_tokens": outputs.shape[1] - inputs.input_ids.shape[1],
                }
            )