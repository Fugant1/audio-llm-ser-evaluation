import json
from pathlib import Path
import tempfile
import unittest

from src.tracking.logging import Logger, get_logger
from src.tracking import Logger as TrackingLogger


class TestLogger(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.log_dir = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_imports(self):
        self.assertIs(Logger, TrackingLogger)

    def test_console_and_file_logging(self):
        log_file = "test_run.log"
        logger = Logger(name="test_logger", log_dir=self.log_dir, log_file=log_file, level="DEBUG")
        
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")
        logger.critical("Critical message")
        logger.close()

        log_path = self.log_dir / log_file
        self.assertTrue(log_path.exists())
        content = log_path.read_text(encoding="utf-8")
        self.assertIn("Debug message", content)
        self.assertIn("Info message", content)
        self.assertIn("Warning message", content)
        self.assertIn("Error message", content)
        self.assertIn("Critical message", content)

    def test_log_config(self):
        logger = Logger(name="test_config", log_dir=self.log_dir)
        cfg = {"project": {"name": "aaft"}, "model": {"name": "af3"}}
        saved_path = logger.log_config(cfg)
        logger.close()

        self.assertIsNotNone(saved_path)
        self.assertTrue(saved_path.exists())
        with open(saved_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data, cfg)

    def test_log_metrics(self):
        logger = Logger(name="test_metrics", log_dir=self.log_dir)
        
        # Test direct metrics
        saved_path = logger.log_metrics({"macro_f1": 0.7523, "accuracy": 0.81})
        self.assertIsNotNone(saved_path)
        with open(saved_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertAlmostEqual(data["macro_f1"], 0.7523)
        self.assertAlmostEqual(data["accuracy"], 0.81)

        # Test step-based metrics
        logger.log_metrics({"macro_f1": 0.76}, step=1)
        logger.log_metrics({"macro_f1": 0.78}, step=2)
        logger.close()

        with open(saved_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertIn("steps", data)
        self.assertEqual(len(data["steps"]), 2)
        self.assertEqual(data["steps"][0]["step"], 1)
        self.assertEqual(data["steps"][1]["step"], 2)

    def test_log_predictions(self):
        logger = Logger(name="test_predictions", log_dir=self.log_dir)
        preds = [
            {"sample_id": "s1", "pred": "joy", "truth": "joy"},
            {"sample_id": "s2", "pred": "anger", "truth": "sadness"},
        ]
        csv_path = logger.log_predictions(preds, filename="predictions.csv")
        self.assertIsNotNone(csv_path)
        self.assertTrue(csv_path.exists())

        content = csv_path.read_text(encoding="utf-8")
        self.assertIn("sample_id,pred,truth", content)
        self.assertIn("s1,joy,joy", content)

        json_path = logger.log_predictions(preds, filename="predictions.json")
        self.assertIsNotNone(json_path)
        with open(json_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)
        self.assertEqual(len(json_data), 2)
        logger.close()

    def test_log_artifact(self):
        logger = Logger(name="test_artifact", log_dir=self.log_dir)
        artifact = self.log_dir / "temp_artifact.txt"
        artifact.write_text("matrix output", encoding="utf-8")

        sub_dir = self.log_dir / "artifacts_sub"
        sub_dir.mkdir()
        logger_sub = Logger(name="test_sub", log_dir=sub_dir)
        dest = logger_sub.log_artifact(artifact, "saved_artifact.txt")
        self.assertIsNotNone(dest)
        self.assertTrue(dest.exists())
        self.assertEqual(dest.read_text(encoding="utf-8"), "matrix output")
        logger.close()
        logger_sub.close()

    def test_get_logger(self):
        logger = get_logger(name="factory_logger", log_dir=self.log_dir)
        self.assertIsInstance(logger, Logger)
        self.assertEqual(logger.name, "factory_logger")
        logger.close()


if __name__ == "__main__":
    unittest.main()
