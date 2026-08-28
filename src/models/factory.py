from typing import Type
from src.models.base import BaseModel
from src.models.whis_mistral import WhisperMistral
from src.models.af3 import AudioFlamingo3
from src.models.qwenomni import QwenOmni
    
MODEL_REGISTRY: dict[str, Type[BaseModel]] = {
        "whisper_mistral": WhisperMistral,
        "audio_flamingo": AudioFlamingo3,
        "qwen_omni": QwenOmni,
    }
    
def build_model(model_config: dict) -> BaseModel:
        name = model_config.get("name")
        model_cls = MODEL_REGISTRY.get(name)
        if not model_cls:
            raise ValueError(f"Unknown model name '{name}'. Registered: {list(MODEL_REGISTRY.keys())}")
        return model_cls(model_config)

# from src.models.whis_mistral import WhisperMistral
# from src.models.af3 import AudioFlamingo3
# from src.models.qwenomni import QwenOmni

# class Models:
#     def build_whisper_mistral(model_config: dict):
#         #the quantization is being handled inside the specific functions because of the dual model nature of this one
#         return whisper, mistral = WhisperMistral.get_whisperx(model_config), WhisperMistral.get_mistral(model_config)

#     def quantize_model(quantization_config: dict):
#         return BitsAndBytesConfig(
#             load_in_4bit=True,
#             bnb_4bit_quant_type= quantization_config["type"],
#             bnb_4bit_compute_dtype= torch.bfloat16 if quantization_config["compute_dtype"] == "bfloat16" else None,
#             bnb_4bit_use_double_quant=True
#         )

#     def build_af3(model_config: dict):
#         bnb_config = None
#         if model_config["quantization"]["enabled"] == True:
#             bnb_config = quantize_model(model_config["quantization"])
#         return AudioFlamingo3.get_af3(model_config, bnb_config)

#     def build_qwen(model_config: dict):
#         bnb_config = None
#         if model_config["quantization"]["enabled"] == True:
#             bnb_config = quantize_model(model_config["quantization"])
#         return QwenOmni.get_qwen(model_config, bnb_config)

#     def build_model(model_config: dict):
#         if model_config["name"] == "af3":
#             return build_af3(model_config)
#         if model_config["name"] == "wm7":
#             return build_af3(build_whisper_mistral)
#         if model_config["name"] == "qwen_omni":
#             return build_qwen(model_config)