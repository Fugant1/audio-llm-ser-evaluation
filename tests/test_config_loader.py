import pytest
from src.utils.config_loader import ExperimentConfigFactory, deep_merge

def test_deep_merge():
    base = {"a": 1, "nested": {"x": 10, "y": 20}}
    update = {"b": 2, "nested": {"y": 99, "z": 30}}
    merged = deep_merge(base, update)

    assert merged == {"a": 1, "b": 2, "nested": {"x": 10, "y": 99, "z": 30}}
    # Check immutability of base dict
    assert base["nested"]["y"] == 20


def test_experiment_generation_main():
    factory = ExperimentConfigFactory()
    experiments = list(factory.generate_experiments(group="main"))
    
    # 2 datasets * 3 models * 3 prompts = 18 experiments
    assert len(experiments) == 18
    for exp in experiments:
        assert "project" in exp
        assert "dataset" in exp
        assert "model" in exp
        assert "prompt" in exp
        assert "evaluation" in exp
        assert "run_id" in exp["experiment"]


def test_experiment_quantization_override():
    factory = ExperimentConfigFactory()
    experiments = list(factory.generate_experiments(group="quantization"))
    
    assert len(experiments) > 0
    quant_types = {e["model"]["quantization"]["type"] for e in experiments}
    assert quant_types == {"nf4", "bf16"}


def test_experiment_fine_tuning_override():
    factory = ExperimentConfigFactory()
    experiments = list(factory.generate_experiments(group="fine-tuning"))
    
    assert len(experiments) > 0
    for exp in experiments:
        assert exp.get("training", {}).get("enabled") is True


def test_missing_config_raises():
    factory = ExperimentConfigFactory()
    with pytest.raises(FileNotFoundError):
        factory.load_dataset("non_existent_dataset_123")
