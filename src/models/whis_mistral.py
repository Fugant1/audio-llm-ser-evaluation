import torch
from transformers import BitsAndBytesConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import whisperx

class WhisperMistral:

    def get_whisperx(model_config: dict):
        return whisperx.load_model(model_config["pretrained_whisper"], model_config["device"], compute_type=model_config["config_whisper"]["compute_dtype"])

    def get_mistral(model_config: dict):
        model_id = model_config["pretrained_mistral"]
        if model_config["quantization_mistral"]["enabled"] == True:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type= model_config["quantization_mistral"]["type"],
                bnb_4bit_compute_dtype= torch.bfloat16 if model_config["quantization_mistral"]["compute_dtype"] == "bfloat16" else None,
                bnb_4bit_use_double_quant=True
                )
        tokenizer = AutoTokenizer(model_id)
        model = AutoModelForCausalLM(
            model_id,
            torch_dtype = torch.bfloat16 if model_config["quantization_mistral"]["compute_dtype"] == "bfloat16" else None,
            quantization_config = bnb_config,
            attn_implementation = model_config["config_msitral"]["attn_implementation"],
            do_sample = model_config["config_mistral"]["do_sample"],
            device_map = model_config["config_mistral"]["device_map"]
        )

        return tokenizer, model