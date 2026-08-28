"""
Logging and experiment tracking utilities for AFTA.
"""

from __future__ import annotations

import csv
import json
import logging
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Dict, List, Optional, Sequence, Union


class Logger:
    """
    A unified logger for console, file, and experiment artifact tracking.

    Supports standard logging levels (debug, info, warning, error, critical)
    as well as experiment-specific tracking methods (log_metrics, log_config,
    log_predictions, log_artifact).
    """

    def __init__(
        self,
        name: str = "AFTA",
        log_dir: Optional[Union[str, Path]] = None,
        log_file: Optional[str] = "experiment.log",
        level: Union[int, str] = logging.INFO,
        console: bool = True,
        formatter: Optional[logging.Formatter] = None,
    ):
        """
        Initialize the Logger.

        Args:
            name: Name of the logger (e.g., experiment run_id or module name).
            log_dir: Directory where log files and experiment artifacts will be saved.
            log_file: Name of the log file inside log_dir (default: 'experiment.log').
            level: Logging level (e.g., logging.INFO, 'DEBUG', 'INFO', etc.).
            console: Whether to log to standard output.
            formatter: Optional custom logging.Formatter.
        """
        self.name = name
        self.log_dir = Path(log_dir) if log_dir else None
        self.log_file = log_file

        if isinstance(level, str):
            level = getattr(logging, level.upper(), logging.INFO)
        self.level = level

        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)

        self._logger = logging.getLogger(name)
        self._logger.setLevel(self.level)
        self._logger.propagate = False

        # Clear existing handlers to prevent duplicates
        if self._logger.hasHandlers():
            self._logger.handlers.clear()

        # Default formatter
        self.formatter = formatter or logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Console handler
        if console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(self.level)
            console_handler.setFormatter(self.formatter)
            self._logger.addHandler(console_handler)

        # File handler
        self._file_handler: Optional[logging.FileHandler] = None
        if self.log_dir and self.log_file:
            log_path = self.log_dir / self.log_file
            self._file_handler = logging.FileHandler(log_path, encoding="utf-8")
            self._file_handler.setLevel(self.level)
            self._file_handler.setFormatter(self.formatter)
            self._logger.addHandler(self._file_handler)

    # --- Standard Logging Methods ---

    def debug(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log a DEBUG message."""
        self._logger.debug(msg, *args, **kwargs)

    def info(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log an INFO message."""
        self._logger.info(msg, *args, **kwargs)

    def warning(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log a WARNING message."""
        self._logger.warning(msg, *args, **kwargs)

    def error(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log an ERROR message."""
        self._logger.error(msg, *args, **kwargs)

    def critical(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log a CRITICAL message."""
        self._logger.critical(msg, *args, **kwargs)

    def exception(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log an exception with traceback."""
        self._logger.exception(msg, *args, **kwargs)

    def log(self, level: int, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log a message with a specific integer level."""
        self._logger.log(level, msg, *args, **kwargs)

    # --- Experiment Tracking Methods ---

    def log_config(self, config: Dict[str, Any], filename: str = "resolved_config.json") -> Optional[Path]:
        """
        Save experiment configuration dictionary to JSON file in log_dir.

        Args:
            config: Configuration dictionary to serialize.
            filename: Name of the output JSON file.

        Returns:
            Path to saved configuration file or None if log_dir is not set.
        """
        if not self.log_dir:
            self.warning("Cannot save config: log_dir is not configured.")
            return None

        file_path = self.log_dir / filename
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        self.info(f"Saved configuration to {file_path}")
        return file_path

    def log_metrics(
        self,
        metrics: Dict[str, Any],
        step: Optional[int] = None,
        filename: str = "metrics.json",
    ) -> Optional[Path]:
        """
        Log metrics to console/log file and persist to JSON in log_dir.

        Args:
            metrics: Dictionary of metric names and numeric/string values.
            step: Optional epoch/step/iteration index.
            filename: Name of the metrics JSON file.

        Returns:
            Path to saved metrics file or None if log_dir is not set.
        """
        step_str = f" [Step {step}]" if step is not None else ""
        metrics_repr = ", ".join(
            f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
            for k, v in metrics.items()
        )
        self.info(f"Metrics{step_str}: {metrics_repr}")

        if not self.log_dir:
            return None

        file_path = self.log_dir / filename
        data = {}
        if file_path.exists():
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                data = {}

        if step is not None:
            if "steps" not in data:
                data["steps"] = []
            data["steps"].append({"step": step, **metrics})
            data["latest"] = metrics
        else:
            data.update(metrics)

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        return file_path

    def log_predictions(
        self,
        predictions: Union[Sequence[Dict[str, Any]], Any],
        filename: str = "predictions.csv",
    ) -> Optional[Path]:
        """
        Save predictions to CSV or JSON file in log_dir.

        Args:
            predictions: List of dict records, or pandas DataFrame.
            filename: Target file name (supports .csv and .json).

        Returns:
            Path to saved predictions file or None if log_dir is not set.
        """
        if not self.log_dir:
            self.warning("Cannot save predictions: log_dir is not configured.")
            return None

        file_path = self.log_dir / filename

        # Check if pandas DataFrame
        if hasattr(predictions, "to_csv") and hasattr(predictions, "to_json"):
            if filename.endswith(".json"):
                predictions.to_json(file_path, orient="records", indent=2)
            else:
                predictions.to_csv(file_path, index=False)
            self.info(f"Saved {len(predictions)} predictions to {file_path}")
            return file_path

        # Sequence of dicts
        if isinstance(predictions, Sequence) and predictions:
            if filename.endswith(".json"):
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(list(predictions), f, indent=2)
            else:
                fieldnames = list(predictions[0].keys())
                with open(file_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(predictions)
            self.info(f"Saved {len(predictions)} predictions to {file_path}")
            return file_path
        elif isinstance(predictions, Sequence) and len(predictions) == 0:
            self.warning("Predictions list is empty, nothing saved.")
            return None

        self.warning(f"Unsupported predictions type: {type(predictions)}")
        return None

    def log_artifact(
        self,
        artifact_path: Union[str, Path],
        dest_name: Optional[str] = None,
    ) -> Optional[Path]:
        """
        Copy an artifact file into log_dir.

        Args:
            artifact_path: Path to existing artifact file.
            dest_name: Optional new filename in log_dir.

        Returns:
            Path to copied artifact in log_dir.
        """
        if not self.log_dir:
            self.warning("Cannot save artifact: log_dir is not configured.")
            return None

        src = Path(artifact_path)
        if not src.exists():
            self.error(f"Artifact not found: {src}")
            return None

        dest = self.log_dir / (dest_name or src.name)
        shutil.copy2(src, dest)
        self.info(f"Artifact copied to {dest}")
        return dest

    def close(self) -> None:
        """Close handlers and flush logs."""
        for handler in list(self._logger.handlers):
            handler.flush()
            handler.close()
            self._logger.removeHandler(handler)


def get_logger(
    name: str = "AFTA",
    log_dir: Optional[Union[str, Path]] = None,
    log_file: Optional[str] = "experiment.log",
    level: Union[int, str] = logging.INFO,
    console: bool = True,
) -> Logger:
    """
    Convenience factory function to create a Logger instance.
    """
    return Logger(
        name=name,
        log_dir=log_dir,
        log_file=log_file,
        level=level,
        console=console,
    )
