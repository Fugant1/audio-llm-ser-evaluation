import argparse
import json
from pathlib import Path

from src.tracking import Logger, get_logger
from src.utils.config_loader import ExperimentConfigFactory
    
    
def run_single_experiment(cfg: dict) -> None:
        run_id = cfg["experiment"]["run_id"]
        output_dir = Path("results") / cfg["dataset"]["name"] / run_id / f"seed_{str(cfg['project']['seed'])}"
        output_dir.mkdir(parents=True, exist_ok=True)
        logger = get_logger(run_id, Path(output_dir) / "logs")
    
        logger.info(f"Starting Experiment {run_id}")
        # Uber-grade reproducibility: Always snapshot the resolved config with results
        with open(output_dir / "resolved_config.json", "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
        logger.info(f"Resolved configuration saved to {output_dir / 'resolved_config.json'}")
        logger.info(cfg["model"])
        # TODO: Pass cfg to your model inference pipeline and evaluation tracking
        # e.g., model = build_model(cfg["model"])
        #       eval_dataset = build_dataset(cfg["dataset"])
        #       metrics = evaluate(model, eval_dataset, cfg["prompt"], cfg["evaluation"])
    
    
def main():
    parser = argparse.ArgumentParser(description="AAFT Experiment Matrix Runner")
    parser.add_argument("--group", type=str, default=None, help="Experiment group (e.g. 'main', 'quantization')")
    parser.add_argument("--filter-dataset", type=str, default=None, help="Filter by specific dataset")
    parser.add_argument("--run-id", type=str, default=None, help="Run a single specific experiment by its run_id") 
    parser.add_argument("--dry-run", action="store_true", help="Print experiment runs without executing")
    args = parser.parse_args()

    factory = ExperimentConfigFactory()
    experiments = list(factory.generate_experiments(group=args.group))

    if args.filter_dataset:
        experiments = [e for e in experiments if e["dataset"]["name"] == args.filter_dataset]

    if args.run_id:
        experiments = [e for e in experiments if e["experiment"]["run_id"] == args.run_id]
        if not experiments:
            print(f"Erro: Nenhum experimento encontrado com o run_id '{args.run_id}'")
            return

    print(f"Total experiments planned: {len(experiments)}")

    for exp_cfg in experiments:
        if args.dry_run:
            print(f"[DRY-RUN] {exp_cfg['experiment']['run_id']}")
        else:
            run_single_experiment(exp_cfg)
    
    
if __name__ == "__main__":
        main()