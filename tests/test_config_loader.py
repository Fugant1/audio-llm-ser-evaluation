import pytest
from src.utils.config_loader import ExperimentConfigFactory, deep_merge
    
    
def test_deep_merge():
        base = {"a": 1, "nested": {"x": 10, "y": 20}}
        update = {"b": 2, "nested": {"y": 99, "z": 30}}
        merged = deep_merge(base, update)
    
        assert merged == {"a": 1, "b": 2, "nested": {"x": 10, "y": 99, "z": 30}}
        # Check immutability of base
        assert base["nested"]["y"] == 20
    
    
def test_experiment_generation():
        factory = ExperimentConfigFactory()
        experiments = list(factory.generate_experiments(group="main"))
        assert len(experiments) > 0
    
        first_exp = experiments[0]
        assert "project" in first_exp
        assert "dataset" in first_exp
        assert "model" in first_exp
        assert "prompt" in first_exp
        assert "run_id" in first_exp["experiment"]