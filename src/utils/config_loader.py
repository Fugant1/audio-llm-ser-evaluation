import copy
import itertools
from pathlib import Path
from typing import Any, Dict, Iterator, List
import yaml
    
    
def deep_merge(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merges dictionary updates into a base dictionary."""
        result = copy.deepcopy(base)
        for key, value in update.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = deep_merge(result[key], value)
            else:
                result[key] = copy.deepcopy(value)
        return result
    
    
class ExperimentConfigFactory:
        """Loads modular YAMLs and builds resolved experiment configurations."""
    
        def __init__(self, config_root: Path | str = "configs"):
            self.config_root = Path(config_root)
            self.base_cfg = self._load_yaml(self.config_root / "base.yaml")
            self.matrix_cfg = self._load_yaml(self.config_root / "experiments" / "experiments.yaml")
    
        def _load_yaml(self, path: Path) -> Dict[str, Any]:
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
    
        def _normalize_name(self, name: str) -> str:
            """Handles underscores and hyphens in filenames consistently."""
            return name.replace("_", "-")
    
        def load_dataset(self, name: str) -> Dict[str, Any]:
            return self._load_yaml(self.config_root / "datasets" / f"{self._normalize_name(name)}.yaml")
    
        def load_model(self, name: str) -> Dict[str, Any]:
            return self._load_yaml(self.config_root / "models" / f"{self._normalize_name(name)}.yaml")
    
        def load_prompt(self, name: str) -> Dict[str, Any]:
            return self._load_yaml(self.config_root / "prompts" / f"{self._normalize_name(name)}.yaml")
    
        def generate_experiments(self, group: str | None = None) -> Iterator[Dict[str, Any]]:
            """Yields fully resolved experiment dictionaries from the experiment matrix."""
            matrix = self.matrix_cfg.get("matrix", {})
            target_groups = {group: matrix[group]} if group else matrix
    
            for group_name, group_spec in target_groups.items():
                datasets = group_spec.get("datasets", [])
                models = group_spec.get("models", [])
                prompts = group_spec.get("prompts", [])
                quantizations = group_spec.get("quantization", [None])
    
                for ds_name, model_name, prompt_name, quant in itertools.product(
                    datasets, models, prompts, quantizations
                ):
                    # 1. Start from base defaults
                    merged = copy.deepcopy(self.base_cfg)
    
                    # 2. Merge dataset, model, and prompt configs
                    merged = deep_merge(merged, self.load_dataset(ds_name))
                    merged = deep_merge(merged, self.load_model(model_name))
                    merged = deep_merge(merged, self.load_prompt(prompt_name))
    
                    # 3. Apply group-level overrides (e.g. quantization, training)
                    if quant:
                        merged["model"]["quantization"]["type"] = quant
                    if "training" in group_spec:
                        merged["training"] = group_spec["training"]
    
                    # 4. Generate deterministic metadata for logging and reproducibility
                    quant_tag = f"_{quant}" if quant else ""
                    run_id = f"{group_name}_{ds_name}_{model_name}_{prompt_name}{quant_tag}"
                    merged["experiment"] = {
                        "group": group_name,
                        "run_id": run_id,
                        "dataset_name": ds_name,
                        "model_name": model_name,
                        "prompt_name": prompt_name,
                    }
                    yield merged