from __future__ import annotations

import copy
import itertools
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union
import yaml


def deep_merge(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merges dictionary updates into a base dictionary.
    
    Nested dictionaries are merged recursively, while lists and scalar values
    in `update` override the corresponding values in `base`.
    """
    if not isinstance(base, dict):
        return copy.deepcopy(update) if isinstance(update, dict) else update
    if not isinstance(update, dict):
        return copy.deepcopy(base)

    result = copy.deepcopy(base)
    for key, value in update.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


class ExperimentConfigFactory:
    """
    Loads modular YAML configurations, caches parsed files, and builds
    resolved, reproducible experiment configurations from experiment matrices.
    """

    RESERVED_MATRIX_KEYS = {"datasets", "models", "prompts", "quantization"}
    NON_QUANTIZED_PRECISIONS = {"bf16", "fp16", "fp32", "float16", "bfloat16", "float32", "none"}

    def __init__(self, config_root: Union[Path, str] = "configs"):
        self.config_root = Path(config_root)
        self._cache: Dict[Path, Dict[str, Any]] = {}
        self.base_cfg = self._load_yaml(self.config_root / "base.yaml")
        self.matrix_cfg = self._load_yaml(self.config_root / "experiments" / "experiments.yaml")

    def clear_cache(self) -> None:
        """Clears the internal parsed YAML cache."""
        self._cache.clear()

    def _load_yaml(self, path: Path) -> Dict[str, Any]:
        """Loads and caches a YAML file from disk, returning a deep copy."""
        path = path.resolve()
        if path not in self._cache:
            if not path.exists():
                raise FileNotFoundError(f"Configuration file not found: {path}")
            with open(path, "r", encoding="utf-8") as f:
                content = yaml.safe_load(f) or {}
            self._cache[path] = content
        return copy.deepcopy(self._cache[path])

    def _resolve_file(self, folder: str, name: str) -> Path:
        """
        Finds config file matching name with underscores, hyphens, and standard extensions (.yaml, .yml).
        """
        folder_path = self.config_root / folder
        if not folder_path.exists():
            raise FileNotFoundError(f"Configuration folder not found: {folder_path}")

        clean_name = name.strip()
        if clean_name.endswith(".yaml"):
            clean_name = clean_name[:-5]
        elif clean_name.endswith(".yml"):
            clean_name = clean_name[:-4]

        variants = [
            clean_name,
            clean_name.replace("_", "-"),
            clean_name.replace("-", "_"),
            clean_name.replace("-", "").replace("_", ""),
        ]

        candidates: List[Path] = []
        for v in variants:
            candidates.append(folder_path / f"{v}.yaml")
            candidates.append(folder_path / f"{v}.yml")

        for candidate in candidates:
            if candidate.exists():
                return candidate

        raise FileNotFoundError(
            f"No configuration matching '{name}' found in {folder_path}. "
            f"Searched candidate filenames: {[c.name for c in candidates]}"
        )

    def load_dataset(self, name: str) -> Dict[str, Any]:
        """Loads a dataset configuration by name."""
        return self._load_yaml(self._resolve_file("datasets", name))

    def load_model(self, name: str) -> Dict[str, Any]:
        """Loads a model configuration by name."""
        return self._load_yaml(self._resolve_file("models", name))

    def load_prompt(self, name: str) -> Dict[str, Any]:
        """Loads a prompt configuration by name."""
        return self._load_yaml(self._resolve_file("prompts", name))

    def list_groups(self) -> List[str]:
        """Returns the list of available experiment matrix groups."""
        return list(self.matrix_cfg.get("matrix", {}).keys())

    def get_experiment_count(self, group: Optional[str] = None) -> int:
        """Calculates total planned experiments without constructing all dicts."""
        matrix = self.matrix_cfg.get("matrix", {})
        if group and group not in matrix:
            raise ValueError(f"Experiment group '{group}' not found in matrix. Available groups: {list(matrix.keys())}")
        
        target_groups = {group: matrix[group]} if group else matrix
        total = 0
        for g_spec in target_groups.values():
            ds_len = len(g_spec.get("datasets", []))
            m_len = len(g_spec.get("models", []))
            p_len = len(g_spec.get("prompts", []))
            q_len = len(g_spec.get("quantization", [None]))
            total += ds_len * m_len * p_len * max(q_len, 1)
        return total

    def validate_matrix(self) -> Dict[str, Any]:
        """
        Validates that all files referenced across the experiment matrix exist and parse cleanly.
        """
        matrix = self.matrix_cfg.get("matrix", {})
        if not matrix:
            raise ValueError("No 'matrix' section found in experiments.yaml")

        validation_report: Dict[str, Any] = {"groups": {}, "total_experiments": 0}

        for group_name, group_spec in matrix.items():
            datasets = group_spec.get("datasets", [])
            models = group_spec.get("models", [])
            prompts = group_spec.get("prompts", [])
            quantizations = group_spec.get("quantization", [None])

            # Validate dataset configs
            resolved_datasets = [self.load_dataset(d) for d in datasets]
            # Validate model configs
            resolved_models = [self.load_model(m) for m in models]
            # Validate prompt configs
            resolved_prompts = [self.load_prompt(p) for p in prompts]

            exp_count = len(datasets) * len(models) * len(prompts) * max(len(quantizations), 1)
            validation_report["groups"][group_name] = {
                "datasets": [d.get("dataset", {}).get("name", "unknown") for d in resolved_datasets],
                "models": [m.get("model", {}).get("name", "unknown") for m in resolved_models],
                "prompts": [p.get("prompt", {}).get("strategy", "unknown") for p in resolved_prompts],
                "quantizations": quantizations,
                "count": exp_count,
            }
            validation_report["total_experiments"] += exp_count

        return validation_report

    def generate_experiments(self, group: Optional[str] = None) -> Iterator[Dict[str, Any]]:
        """
        Yields fully resolved experiment dictionaries from the experiment matrix.
        
        Args:
            group: Optional name of the experiment group (e.g. 'main', 'quantization').
                   If None, iterates through all groups in the matrix.
        """
        matrix = self.matrix_cfg.get("matrix", {})
        if group and group not in matrix:
            raise ValueError(f"Experiment group '{group}' not found in matrix config. Available groups: {list(matrix.keys())}")

        target_groups = {group: matrix[group]} if group else matrix

        for group_name, group_spec in target_groups.items():
            datasets = group_spec.get("datasets", [])
            models = group_spec.get("models", [])
            prompts = group_spec.get("prompts", [])
            quantizations = group_spec.get("quantization", [None])

            # Extract any group-level overrides (e.g. training, custom params)
            group_overrides = {
                k: v for k, v in group_spec.items() if k not in self.RESERVED_MATRIX_KEYS
            }

            for ds_name, model_name, prompt_name, quant in itertools.product(
                datasets, models, prompts, quantizations
            ):
                # 1. Start from base defaults
                merged = copy.deepcopy(self.base_cfg)

                # 2. Merge dataset, model, and prompt configs
                merged = deep_merge(merged, self.load_dataset(ds_name))
                merged = deep_merge(merged, self.load_model(model_name))
                merged = deep_merge(merged, self.load_prompt(prompt_name))

                # 3. Dynamic dataset label binding for prompts if allowed_labels is null
                prompt_output = merged.get("prompt", {}).get("output", {})
                if isinstance(prompt_output, dict) and prompt_output.get("allowed_labels") is None:
                    if "dataset" in merged and "labels" in merged["dataset"]:
                        merged["prompt"]["output"]["allowed_labels"] = copy.deepcopy(merged["dataset"]["labels"])

                # 4. Apply quantization & precision overrides
                if quant:
                    if "model" not in merged:
                        merged["model"] = {}
                    if "quantization" not in merged["model"]:
                        merged["model"]["quantization"] = {}
                    
                    merged["model"]["quantization"]["type"] = quant
                    is_unquantized = str(quant).lower() in self.NON_QUANTIZED_PRECISIONS
                    merged["model"]["quantization"]["enabled"] = not is_unquantized
                    if is_unquantized:
                        merged["model"]["precision"] = quant

                # 5. Apply any group-level overrides (e.g. training, fine-tuning specs)
                if group_overrides:
                    merged = deep_merge(merged, group_overrides)

                # 6. Generate deterministic metadata for logging and reproducibility
                quant_tag = f"_{quant}" if quant else ""
                run_id = f"{group_name}_{ds_name}_{model_name}_{prompt_name}{quant_tag}"
                merged["experiment"] = {
                    "group": group_name,
                    "run_id": run_id,
                    "dataset_name": ds_name,
                    "model_name": model_name,
                    "prompt_name": prompt_name,
                    "quantization": quant,
                }
                yield merged