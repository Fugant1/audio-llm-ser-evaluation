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


def test_invalid_group_raises():
    factory = ExperimentConfigFactory()
    with pytest.raises(ValueError, match="Experiment group 'invalid_group_xyz' not found"):
        list(factory.generate_experiments(group="invalid_group_xyz"))


def test_dynamic_label_propagation():
    factory = ExperimentConfigFactory()
    # Check IEMOCAP with zero-shot and cot-json prompts
    iemocap_exps = [e for e in factory.generate_experiments(group="main") if e["dataset"]["name"] == "iemocap"]
    assert len(iemocap_exps) > 0
    for exp in iemocap_exps:
        if exp["prompt"]["strategy"] in ("zero_shot", "cot_json"):
            assert exp["prompt"]["output"]["allowed_labels"] == exp["dataset"]["labels"]
            assert "angry" in exp["prompt"]["output"]["allowed_labels"]
            assert "frustration" in exp["prompt"]["output"]["allowed_labels"]

    # Check MELD
    meld_exps = [e for e in factory.generate_experiments(group="main") if e["dataset"]["name"] == "meld"]
    assert len(meld_exps) > 0
    for exp in meld_exps:
        if exp["prompt"]["strategy"] in ("zero_shot", "cot_json"):
            assert exp["prompt"]["output"]["allowed_labels"] == exp["dataset"]["labels"]
            assert "anger" in exp["prompt"]["output"]["allowed_labels"]
            assert "joy" in exp["prompt"]["output"]["allowed_labels"]


def test_validate_matrix():
    factory = ExperimentConfigFactory()
    report = factory.validate_matrix()
    assert report["total_experiments"] == 37  # 18 main + 6 cot-json + 12 quantization + 1 fine-tuning
    assert "main" in report["groups"]
    assert "cot-json" in report["groups"]
    assert "quantization" in report["groups"]
    assert "fine-tuning" in report["groups"]


def test_list_groups_and_count():
    factory = ExperimentConfigFactory()
    groups = factory.list_groups()
    assert "main" in groups
    assert "quantization" in groups
    assert factory.get_experiment_count("main") == 18
    assert factory.get_experiment_count() == 37


def test_caching():
    factory = ExperimentConfigFactory()
    ds1 = factory.load_dataset("meld")
    ds2 = factory.load_dataset("meld")
    assert ds1 == ds2
    # Verify cached without mutation leakage
    ds1["dataset"]["name"] = "mutated"
    ds3 = factory.load_dataset("meld")
    assert ds3["dataset"]["name"] == "meld"

