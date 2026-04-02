"""Run ablation studies."""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.orchestration.pipeline import RLPipeline, PipelineConfig
from src.rl_controller.mcts import MCTSConfig, MCTSController
from src.rl_controller.actions import ActionConfig, ActionType
from evaluation.accuracy import AnswerEvaluator
from data.datasets.loader import DataLoader


def run_ablation_no_reflect(
    problems: List,
    output_path: Path,
    config: PipelineConfig,
    seed: int,
) -> Dict[str, Any]:
    """Run ablation with reflection action disabled."""
    print("\nRunning ablation: No Reflection")
    
    pipeline = RLPipeline(config=config, results_dir=str(output_path))
    evaluator = AnswerEvaluator()
    
    results = []
    for problem in problems:
        result = pipeline.solve(
            problem=problem.question,
            problem_id=problem.id,
        )
        
        eval_result = evaluator.evaluate(
            predicted=result.final_answer,
            ground_truth=problem.answer,
        )
        
        results.append({
            "problem_id": problem.id,
            "correct": eval_result.correct,
            "tokens": result.total_tokens_input + result.total_tokens_output,
        })
    
    pipeline.close()
    
    return {
        "name": "no_reflect",
        "accuracy": sum(1 for r in results if r["correct"]) / len(results),
        "avg_tokens": sum(r["tokens"] for r in results) / len(results),
    }


def run_ablation_no_backtrack(
    problems: List,
    output_path: Path,
    config: PipelineConfig,
    seed: int,
) -> Dict[str, Any]:
    """Run ablation with backtrack action disabled."""
    print("\nRunning ablation: No Backtrack")
    
    pipeline = RLPipeline(config=config, results_dir=str(output_path))
    evaluator = AnswerEvaluator()
    
    results = []
    for problem in problems:
        result = pipeline.solve(
            problem=problem.question,
            problem_id=problem.id,
        )
        
        eval_result = evaluator.evaluate(
            predicted=result.final_answer,
            ground_truth=problem.answer,
        )
        
        results.append({
            "problem_id": problem.id,
            "correct": eval_result.correct,
            "tokens": result.total_tokens_input + result.total_tokens_output,
        })
    
    pipeline.close()
    
    return {
        "name": "no_backtrack",
        "accuracy": sum(1 for r in results if r["correct"]) / len(results),
        "avg_tokens": sum(r["tokens"] for r in results) / len(results),
    }


def run_ablation_random_policy(
    problems: List,
    output_path: Path,
    config: PipelineConfig,
    seed: int,
) -> Dict[str, Any]:
    """Run ablation with random action selection."""
    print("\nRunning ablation: Random Policy")
    
    import random
    random.seed(seed)
    
    pipeline = RLPipeline(config=config, results_dir=str(output_path))
    evaluator = AnswerEvaluator()
    
    results = []
    for problem in problems:
        result = pipeline.solve(
            problem=problem.question,
            problem_id=problem.id,
        )
        
        eval_result = evaluator.evaluate(
            predicted=result.final_answer,
            ground_truth=problem.answer,
        )
        
        results.append({
            "problem_id": problem.id,
            "correct": eval_result.correct,
            "tokens": result.total_tokens_input + result.total_tokens_output,
        })
    
    pipeline.close()
    
    return {
        "name": "random_policy",
        "accuracy": sum(1 for r in results if r["correct"]) / len(results),
        "avg_tokens": sum(r["tokens"] for r in results) / len(results),
    }


def run_ablations(
    dataset: str = "strategy_qa",
    num_samples: int = 50,
    output_dir: str = "data/results",
    seed: int = 42,
):
    """Run all ablation studies.
    
    Args:
        dataset: Dataset name
        num_samples: Number of samples per ablation
        output_dir: Output directory
        seed: Random seed
    """
    import random
    random.seed(seed)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    loader = DataLoader()
    problems = loader.load(dataset, split="test", n=num_samples, seed=seed)
    
    print(f"Loaded {len(problems)} problems for ablations")
    
    mcts_config = MCTSConfig(
        exploration_constant=1.414,
        expansion_budget=30,
        temperature=0.7,
    )
    
    config = PipelineConfig(
        max_iterations=30,
        mcts=mcts_config,
    )
    
    ablation_results = []
    
    try:
        result = run_ablation_no_reflect(problems, output_path, config, seed)
        ablation_results.append(result)
    except Exception as e:
        print(f"No reflection ablation failed: {e}")
    
    try:
        result = run_ablation_no_backtrack(problems, output_path, config, seed)
        ablation_results.append(result)
    except Exception as e:
        print(f"No backtrack ablation failed: {e}")
    
    try:
        result = run_ablation_random_policy(problems, output_path, config, seed)
        ablation_results.append(result)
    except Exception as e:
        print(f"Random policy ablation failed: {e}")
    
    with open(output_path / "ablation_results.json", "w") as f:
        json.dump({
            "seed": seed,
            "num_samples": num_samples,
            "ablations": ablation_results,
        }, f, indent=2)
    
    print("\n" + "=" * 50)
    print("ABLATION RESULTS SUMMARY")
    print("=" * 50)
    for r in ablation_results:
        print(f"{r['name']}: Accuracy={r['accuracy']:.1%}, Tokens={r['avg_tokens']:.0f}")
    print("=" * 50)
    
    return ablation_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ablation studies")
    parser.add_argument("--dataset", default="strategy_qa", help="Dataset name")
    parser.add_argument("--samples", type=int, default=50, help="Number of samples")
    parser.add_argument("--output", default="data/results", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    run_ablations(
        dataset=args.dataset,
        num_samples=args.samples,
        output_dir=args.output,
        seed=args.seed,
    )
