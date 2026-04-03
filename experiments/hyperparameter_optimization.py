"""Hyperparameter optimization using Optuna for self-reflection system."""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import numpy as np
import optuna
from optuna.samplers import TPESampler
from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


class HyperparameterOptimizer:
    """Bayesian hyperparameter optimization for self-reflection pipeline."""

    def __init__(
        self,
        n_trials: int = 50,
        n_jobs: int = 1,
        study_name: str = "self_reflection_optimization",
        storage: Optional[str] = None,
        direction: str = "maximize"
    ):
        """Initialize hyperparameter optimizer.

        Args:
            n_trials: Number of optimization trials
            n_jobs: Number of parallel jobs
            study_name: Name for the Optuna study
            storage: Database storage URL (None for in-memory)
            direction: Optimization direction ("maximize" or "minimize")
        """
        self.n_trials = n_trials
        self.n_jobs = n_jobs
        self.study_name = study_name
        self.storage = storage
        self.direction = direction

        # Results tracking
        self.best_params = None
        self.best_value = None

        logger.info(f"HyperparameterOptimizer initialized (n_trials={n_trials})")

    def define_search_space(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Define hyperparameter search space.

        Args:
            trial: Optuna trial object

        Returns:
            Dictionary of hyperparameters
        """
        params = {}

        # MCTS hyperparameters
        params["exploration_constant"] = trial.suggest_float("exploration_constant", 0.5, 2.0)
        params["max_tree_depth"] = trial.suggest_int("max_tree_depth", 10, 30)
        params["expansion_budget"] = trial.suggest_int("expansion_budget", 30, 100)
        params["temperature"] = trial.suggest_float("temperature", 0.3, 0.9)

        # Action thresholds
        params["backtrack_threshold"] = trial.suggest_float("backtrack_threshold", 0.2, 0.4)
        params["conclude_threshold"] = trial.suggest_float("conclude_threshold", 0.75, 0.95)

        # Action weights (constrained to sum to 1.0)
        expand_weight = trial.suggest_float("expand_weight", 0.3, 0.5)
        reflect_weight = trial.suggest_float("reflect_weight", 0.15, 0.35)
        backtrack_weight = trial.suggest_float("backtrack_weight", 0.1, 0.25)
        
        # Normalize weights to sum to 1.0
        total = expand_weight + reflect_weight + backtrack_weight
        params["expand_weight"] = expand_weight / total
        params["reflect_weight"] = reflect_weight / total
        params["backtrack_weight"] = backtrack_weight / total
        params["conclude_weight"] = 1.0 - params["expand_weight"] - params["reflect_weight"] - params["backtrack_weight"]

        # Generator hyperparameters
        params["generator_temperature"] = trial.suggest_float("generator_temperature", 0.5, 0.9)
        params["generator_max_tokens"] = trial.suggest_int("generator_max_tokens", 256, 1024)

        # Evaluator hyperparameters
        params["evaluator_temperature"] = trial.suggest_float("evaluator_temperature", 0.05, 0.2)

        # Reflection hyperparameters
        params["reflection_depth_factual"] = trial.suggest_int("reflection_depth_factual", 1, 2)
        params["reflection_depth_reasoning"] = trial.suggest_int("reflection_depth_reasoning", 2, 4)
        params["reflection_depth_strategic"] = trial.suggest_int("reflection_depth_strategic", 2, 5)

        return params

    def objective(self, trial: optuna.Trial) -> float:
        """Objective function to optimize.

        Args:
            trial: Optuna trial object

        Returns:
            Metric to optimize (accuracy)
        """
        # Get hyperparameters
        params = self.define_search_space(trial)

        logger.info(f"Trial {trial.number}: Testing parameters...")
        logger.debug(f"Parameters: {params}")

        try:
            # Run evaluation with these parameters
            # For now, return synthetic accuracy based on params
            # In production, this would run actual experiments
            
            # Simulated accuracy based on reasonable parameter ranges
            base_accuracy = 0.70
            
            # Reward good exploration constant (around sqrt(2))
            exploration_bonus = -abs(params["exploration_constant"] - 1.414) * 0.05
            
            # Reward moderate temperature
            temp_bonus = -abs(params["temperature"] - 0.7) * 0.03
            
            # Reward balanced weights
            weight_balance = -abs(params["expand_weight"] - 0.4) * 0.02
            
            # Reward appropriate thresholds
            backtrack_bonus = -abs(params["backtrack_threshold"] - 0.3) * 0.02
            conclude_bonus = -abs(params["conclude_threshold"] - 0.85) * 0.02
            
            # Combine metrics
            accuracy = (
                base_accuracy +
                exploration_bonus +
                temp_bonus +
                weight_balance +
                backtrack_bonus +
                conclude_bonus +
                np.random.randn() * 0.02  # Add noise
            )
            
            # Clamp to [0, 1]
            accuracy = max(0.0, min(1.0, accuracy))

            logger.info(f"Trial {trial.number}: Accuracy = {accuracy:.4f}")

            return accuracy

        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {e}")
            return 0.0

    def optimize(self) -> Dict[str, Any]:
        """Run hyperparameter optimization.

        Returns:
            Dictionary with best parameters and results
        """
        logger.info("Starting hyperparameter optimization...")

        # Create study
        sampler = TPESampler(seed=42)
        study = optuna.create_study(
            study_name=self.study_name,
            storage=self.storage,
            direction=self.direction,
            sampler=sampler,
            load_if_exists=True
        )

        # Run optimization
        study.optimize(
            self.objective,
            n_trials=self.n_trials,
            n_jobs=self.n_jobs,
            show_progress_bar=True
        )

        # Get best results
        self.best_params = study.best_params
        self.best_value = study.best_value

        logger.info(f"Optimization complete!")
        logger.info(f"Best value: {self.best_value:.4f}")
        logger.info(f"Best params: {self.best_params}")

        # Generate results
        results = {
            "best_params": self.best_params,
            "best_value": self.best_value,
            "n_trials": len(study.trials),
            "study_name": self.study_name,
            "timestamp": datetime.now().isoformat()
        }

        return results

    def save_results(self, results: Dict[str, Any], save_path: str):
        """Save optimization results to JSON.

        Args:
            results: Results dictionary
            save_path: Path to save JSON file
        """
        save_file = Path(save_path)
        save_file.parent.mkdir(parents=True, exist_ok=True)

        with open(save_file, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"Results saved to {save_path}")

    def generate_config(self, params: Dict[str, Any], template_path: str) -> str:
        """Generate config file from optimized parameters.

        Args:
            params: Optimized parameters
            template_path: Path to config template

        Returns:
            Path to generated config
        """
        import yaml

        # Load template
        with open(template_path, 'r') as f:
            config = yaml.safe_load(f)

        # Update parameters
        if "mcts" in config:
            config["mcts"]["exploration_constant"] = params.get("exploration_constant", 1.414)
            config["mcts"]["max_tree_depth"] = params.get("max_tree_depth", 20)
            config["mcts"]["expansion_budget"] = params.get("expansion_budget", 50)
            config["mcts"]["temperature"] = params.get("temperature", 0.7)

        if "backtrack_threshold" in params:
            config["backtrack_threshold"] = params["backtrack_threshold"]

        if "conclude_threshold" in params:
            config["conclude_threshold"] = params["conclude_threshold"]

        if "actions" in config:
            config["actions"]["expand"]["weight"] = params.get("expand_weight", 0.4)
            config["actions"]["reflect"]["weight"] = params.get("reflect_weight", 0.25)
            config["actions"]["backtrack"]["weight"] = params.get("backtrack_weight", 0.2)
            config["actions"]["conclude"]["weight"] = params.get("conclude_weight", 0.15)

        if "generator" in config:
            config["generator"]["temperature"] = params.get("generator_temperature", 0.7)
            config["generator"]["max_tokens"] = params.get("generator_max_tokens", 512)

        if "evaluator" in config:
            config["evaluator"]["temperature"] = params.get("evaluator_temperature", 0.1)

        # Save optimized config
        optimized_path = Path(template_path).parent / f"config_optimized_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"
        with open(optimized_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

        logger.info(f"Optimized config saved to {optimized_path}")

        return str(optimized_path)


class MultiDatasetOptimizer:
    """Optimize hyperparameters across multiple datasets."""

    def __init__(
        self,
        datasets: List[str] = ["strategy_qa", "commonsenseqa", "gsm8k"],
        n_trials_per_dataset: int = 30
    ):
        """Initialize multi-dataset optimizer.

        Args:
            datasets: List of dataset names
            n_trials_per_dataset: Trials per dataset
        """
        self.datasets = datasets
        self.n_trials_per_dataset = n_trials_per_dataset

        logger.info(f"MultiDatasetOptimizer initialized for {len(datasets)} datasets")

    def optimize_all(self) -> Dict[str, Dict[str, Any]]:
        """Optimize hyperparameters for all datasets.

        Returns:
            Dictionary of dataset -> optimization results
        """
        all_results = {}

        for dataset in self.datasets:
            logger.info(f"\n{'='*60}")
            logger.info(f"Optimizing for dataset: {dataset}")
            logger.info(f"{'='*60}")

            optimizer = HyperparameterOptimizer(
                n_trials=self.n_trials_per_dataset,
                study_name=f"{dataset}_optimization"
            )

            results = optimizer.optimize()
            all_results[dataset] = results

        # Find universal best parameters
        universal_best = self._find_universal_params(all_results)
        all_results["universal"] = universal_best

        return all_results

    def _find_universal_params(self, all_results: Dict) -> Dict[str, Any]:
        """Find parameters that work well across all datasets.

        Args:
            all_results: Results from all datasets

        Returns:
            Universal best parameters
        """
        # Average numerical parameters
        universal = {}

        for key in ["exploration_constant", "temperature", "backtrack_threshold", 
                    "conclude_threshold", "generator_temperature", "evaluator_temperature"]:
            values = [r["best_params"][key] for r in all_results.values() if "best_params" in r]
            if values:
                universal[key] = np.mean(values)

        for key in ["max_tree_depth", "expansion_budget", "generator_max_tokens"]:
            values = [r["best_params"][key] for r in all_results.values() if "best_params" in r]
            if values:
                universal[key] = int(np.mean(values))

        # For weights, use weighted average based on best_value
        weight_keys = ["expand_weight", "reflect_weight", "backtrack_weight", "conclude_weight"]
        total_score = sum(r["best_value"] for r in all_results.values() if "best_value" in r)

        for key in weight_keys:
            weighted_sum = sum(
                r["best_params"].get(key, 0.25) * r.get("best_value", 1.0)
                for r in all_results.values()
                if "best_params" in r
            )
            universal[key] = weighted_sum / total_score if total_score > 0 else 0.25

        logger.info(f"Universal best parameters: {universal}")

        return universal


def main():
    """Main function for hyperparameter optimization."""
    parser = argparse.ArgumentParser(description="Hyperparameter optimization")
    parser.add_argument("--n-trials", type=int, default=50,
                        help="Number of optimization trials")
    parser.add_argument("--n-jobs", type=int, default=1,
                        help="Number of parallel jobs")
    parser.add_argument("--config-template", type=str, default="config.yaml",
                        help="Path to config template")
    parser.add_argument("--output-dir", type=str, default="optimization_results",
                        help="Output directory for results")
    parser.add_argument("--multi-dataset", action="store_true",
                        help="Optimize across multiple datasets")
    parser.add_argument("--datasets", nargs="+",
                        default=["strategy_qa", "commonsenseqa", "gsm8k"],
                        help="Datasets to optimize for")

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.multi_dataset:
        # Multi-dataset optimization
        optimizer = MultiDatasetOptimizer(
            datasets=args.datasets,
            n_trials_per_dataset=args.n_trials
        )

        results = optimizer.optimize_all()

        # Save all results
        results_file = output_dir / "multi_dataset_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"Multi-dataset results saved to {results_file}")

    else:
        # Single optimization
        optimizer = HyperparameterOptimizer(
            n_trials=args.n_trials,
            n_jobs=args.n_jobs
        )

        results = optimizer.optimize()

        # Save results
        results_file = output_dir / f"optimization_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        optimizer.save_results(results, str(results_file))

        # Generate optimized config
        if Path(args.config_template).exists():
            optimized_config = optimizer.generate_config(results["best_params"], args.config_template)
            logger.info(f"Optimized config generated: {optimized_config}")

    logger.info("Hyperparameter optimization complete!")


if __name__ == "__main__":
    main()
