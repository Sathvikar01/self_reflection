"""Run baseline zero-shot experiments."""

import os
import sys
import json
import argparse
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.orchestration.baseline import BaselineRunner, BaselineConfig
from evaluation.accuracy import AnswerEvaluator
from data.datasets.loader import DataLoader


def run_baseline(
    dataset: str = "strategy_qa",
    num_samples: int = 100,
    output_dir: str = "data/results",
    seed: int = 42,
):
    """Run baseline experiments.
    
    Args:
        dataset: Dataset name
        num_samples: Number of samples to evaluate
        output_dir: Output directory
        seed: Random seed
    """
    import random
    random.seed(seed)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    loader = DataLoader()
    problems = loader.load(dataset, split="test", n=num_samples, seed=seed)
    
    print(f"Loaded {len(problems)} problems from {dataset}")
    
    config = BaselineConfig(
        model="meta/llama-3.1-8b-instruct",
        temperature=0.7,
        max_tokens=2048,
    )
    
    api_key = os.getenv("NVIDIA_API_KEY")
    runner = BaselineRunner(config=config, results_dir=output_dir, api_key=api_key)
    
    evaluator = AnswerEvaluator()
    
    results = []
    for i, problem in enumerate(problems):
        print(f"\nProcessing problem {i+1}/{len(problems)}: {problem.id}")
        
        result = runner.run_single(
            problem=problem.question,
            problem_id=problem.id,
            ground_truth=problem.answer,
        )
        
        eval_result = evaluator.evaluate(
            predicted=result.answer,
            ground_truth=problem.answer,
            problem_id=problem.id,
        )
        
        result.correct = eval_result.correct
        
        results.append({
            "problem_id": problem.id,
            "question": problem.question,
            "ground_truth": problem.answer,
            "prediction": result.answer,
            "correct": eval_result.correct,
            "tokens": result.input_tokens + result.output_tokens,
            "latency": result.latency_seconds,
        })
    
    runner.save_results("baseline_results.json")
    
    correct_count = sum(1 for r in results if r["correct"])
    accuracy = correct_count / len(results)
    avg_tokens = sum(r["tokens"] for r in results) / len(results)
    avg_latency = sum(r["latency"] for r in results) / len(results)
    
    summary = {
        "dataset": dataset,
        "num_samples": len(results),
        "accuracy": accuracy,
        "correct": correct_count,
        "avg_tokens": avg_tokens,
        "avg_latency": avg_latency,
        "seed": seed,
    }
    
    with open(output_path / "baseline_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "=" * 50)
    print("BASELINE RESULTS SUMMARY")
    print("=" * 50)
    print(f"Dataset: {dataset}")
    print(f"Samples: {len(results)}")
    print(f"Accuracy: {accuracy:.1%} ({correct_count}/{len(results)})")
    print(f"Avg Tokens: {avg_tokens:.0f}")
    print(f"Avg Latency: {avg_latency:.1f}s")
    print("=" * 50)
    
    return results, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run baseline experiments")
    parser.add_argument("--dataset", default="strategy_qa", help="Dataset name")
    parser.add_argument("--samples", type=int, default=100, help="Number of samples")
    parser.add_argument("--output", default="data/results", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    run_baseline(
        dataset=args.dataset,
        num_samples=args.samples,
        output_dir=args.output,
        seed=args.seed,
    )
