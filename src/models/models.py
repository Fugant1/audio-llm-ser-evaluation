from src.models.whis_mistral import WhisperMistral

class Models:
    def build_whisper_mistral(model_config: dict):
        return whisper, mistral = WhisperMistral.get_whisperx(model_config), WhisperMistral.get_mistral(model_config)

    def build_af3():
        pass

    def build_qwen():
        pass

    def build_model(model):
    # TODO: #Add the logic to build each model
        pass