import unittest
from unittest.mock import MagicMock, patch

from src.models.base import BaseModel, GenerationResult
from src.models.factory import MODEL_REGISTRY, build_model
from src.models.whis_mistral import WhisperMistral
from src.models.qwenomni import QwenOmni
from src.models.af3 import AudioFlamingo3


class TestModelsVLLM(unittest.TestCase):
    def test_registry_and_factory(self):
        self.assertIn("whisper_mistral", MODEL_REGISTRY)
        self.assertIn("qwen_omni", MODEL_REGISTRY)
        self.assertIn("audio_flamingo", MODEL_REGISTRY)

        wm = build_model({"name": "whisper_mistral"})
        self.assertIsInstance(wm, WhisperMistral)

        qwen = build_model({"name": "qwen_omni"})
        self.assertIsInstance(qwen, QwenOmni)

        af3 = build_model({"name": "audio_flamingo"})
        self.assertIsInstance(af3, AudioFlamingo3)

        with self.assertRaises(ValueError):
            build_model({"name": "non_existent_model"})

    def test_quantization_resolution(self):
        wm_cfg = {
            "name": "whisper_mistral",
            "quantization_mistral": {"enabled": True, "type": "nf4"},
        }
        wm = WhisperMistral(wm_cfg)
        self.assertEqual(wm._resolve_quantization(), "bitsandbytes")

        wm_disabled = {
            "name": "whisper_mistral",
            "quantization_mistral": {"enabled": False, "type": "nf4"},
        }
        self.assertIsNone(WhisperMistral(wm_disabled)._resolve_quantization())

        qwen_cfg = {
            "name": "qwen_omni",
            "quantization": {"enabled": True, "type": "awq"},
        }
        self.assertEqual(QwenOmni(qwen_cfg)._resolve_quantization(), "awq")

    def test_prompt_normalization(self):
        model = build_model({"name": "qwen_omni"})
        # Single string expanded to count
        prompts = model._normalize_prompts("Classify emotion", 3)
        self.assertEqual(prompts, ["Classify emotion", "Classify emotion", "Classify emotion"])

        # Matching list
        prompts = model._normalize_prompts(["p1", "p2"], 2)
        self.assertEqual(prompts, ["p1", "p2"])

        # Mismatch list length raises ValueError
        with self.assertRaises(ValueError):
            model._normalize_prompts(["p1"], 2)

    def test_prompt_formatting(self):
        wm = build_model({"name": "whisper_mistral"})
        formatted_wm = wm._format_prompt("Classify this", "I am very happy today!")
        self.assertIn("Audio Transcript: \"I am very happy today!\"", formatted_wm)
        self.assertIn("Classify this", formatted_wm)

        qwen = build_model({"name": "qwen_omni"})
        formatted_qwen = qwen._format_prompt("Classify this")
        self.assertIn("<|audio_bos|><|AUDIO|><|audio_eos|>", formatted_qwen)

        af3 = build_model({"name": "audio_flamingo"})
        formatted_af3 = af3._format_prompt("Classify this")
        self.assertIn("<sound>", formatted_af3)

    def test_unloaded_model_raises_runtime_error(self):
        wm = build_model({"name": "whisper_mistral"})
        with self.assertRaises(RuntimeError):
            wm.batch_generate(["fake.wav"], "prompt")

        qwen = build_model({"name": "qwen_omni"})
        with self.assertRaises(RuntimeError):
            qwen.batch_generate(["fake.wav"], "prompt")

        af3 = build_model({"name": "audio_flamingo"})
        with self.assertRaises(RuntimeError):
            af3.batch_generate(["fake.wav"], "prompt")

    def test_whisper_mistral_mocked_batch_generate(self):
        wm = build_model({
            "name": "whisper_mistral",
            "generation": {"temperature": 0.0, "max_new_tokens": 64},
        })
        
        # Mock Whisper and vLLM LLM
        mock_whisper = MagicMock()
        mock_whisper.transcribe.return_value = {"text": "I feel excited!"}
        wm.whisper = mock_whisper

        mock_vllm_output = MagicMock()
        mock_vllm_output.outputs = [MagicMock(text=" excited", token_ids=[1, 2])]
        mock_llm = MagicMock()
        mock_llm.generate.return_value = [mock_vllm_output]
        wm.llm = mock_llm

        with patch("whisperx.load_audio", return_value=[0.0, 0.1]):
            result = wm.batch_generate(["sample.wav"], "Predict emotion")

        self.assertIsInstance(result, GenerationResult)
        self.assertEqual(result.predictions, ["excited"])
        self.assertEqual(result.metadata["engine"], "vllm")
        self.assertEqual(result.metadata["transcripts"], ["I feel excited!"])
        self.assertEqual(result.metadata["generated_tokens"], 2)

    def test_qwen_omni_mocked_batch_generate(self):
        qwen = build_model({
            "name": "qwen_omni",
            "generation": {"temperature": 0.0, "max_new_tokens": 64},
        })

        mock_vllm_output = MagicMock()
        mock_vllm_output.outputs = [MagicMock(text="neutral", token_ids=[10, 11])]
        mock_llm = MagicMock()
        mock_llm.generate.return_value = [mock_vllm_output]
        qwen.llm = mock_llm

        with patch.object(qwen, "load_audio_waveform", return_value=([0.0, 0.0], 16000)):
            result = qwen.batch_generate(["sample.wav"], "Predict emotion")

        self.assertIsInstance(result, GenerationResult)
        self.assertEqual(result.predictions, ["neutral"])
        self.assertEqual(result.metadata["engine"], "vllm")
        self.assertEqual(result.metadata["generated_tokens"], 2)

        # Check prompt formatting called on vLLM inputs
        mock_llm.generate.assert_called_once()
        vllm_input = mock_llm.generate.call_args[0][0][0]
        self.assertIn("<|audio_bos|><|AUDIO|><|audio_eos|>", vllm_input["prompt"])
        self.assertIn("audio", vllm_input["multi_modal_data"])

    def test_af3_mocked_batch_generate(self):
        af3 = build_model({
            "name": "audio_flamingo",
            "generation": {"temperature": 0.0, "max_new_tokens": 64},
        })

        mock_vllm_output = MagicMock()
        mock_vllm_output.outputs = [MagicMock(text="anger", token_ids=[5, 6, 7])]
        mock_llm = MagicMock()
        mock_llm.generate.return_value = [mock_vllm_output]
        af3.llm = mock_llm

        with patch.object(af3, "load_audio_waveform", return_value=([0.0, 0.0], 16000)):
            result = af3.batch_generate(["sample.wav"], "Predict emotion")

        self.assertIsInstance(result, GenerationResult)
        self.assertEqual(result.predictions, ["anger"])
        self.assertEqual(result.metadata["engine"], "vllm")
        self.assertEqual(result.metadata["generated_tokens"], 3)

        # Check prompt formatting called on vLLM inputs
        mock_llm.generate.assert_called_once()
        vllm_input = mock_llm.generate.call_args[0][0][0]
        self.assertIn("<sound>", vllm_input["prompt"])
        self.assertIn("audio", vllm_input["multi_modal_data"])


if __name__ == "__main__":
    unittest.main()
