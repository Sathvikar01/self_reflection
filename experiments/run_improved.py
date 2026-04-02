"""Run improved RL-guided experiments with all fixes."""

import os
import sys
import json
import argparse
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.orchestration.improved_pipeline import ImprovedRLPipeline, ImprovedPipelineConfig
from evaluation.accuracy import AnswerEvaluator
from data.datasets.loader import DataLoader


def run_improved_experiment(
    dataset: str = "strategy_qa",
    num_samples: int = 10,
    max_iterations: int = 30,
    output_dir: str = "data/results",
    seed: int = 42,
):
    """Run improved experiment with all fixes.
    
    Fixes implemented:
    1. Better PRM with vacuous detection
    2. Answer progress evaluation  
    3. Always-explore backtracking
    4. Different verification model
    5. Credit assignment learning
    6. Path comparison
    """
    import random
    random.seed(seed)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    loader = DataLoader()
    problems = loader.load(dataset, split="test", n=num_samples, seed=seed)
    
    print(f"\n{'='*60}")
    print("IMPROVED RL-GUIDED EXPERIMENT")
    print("="*60)
    print(f"Dataset: {dataset}")
    print(f"Samples: {num_samples}")
    print(f"Max iterations: {max_iterations}")
    print("="*60)
    
    config = ImprovedPipelineConfig(
        max_iterations=max_iterations,
        min_steps_before_conclude=3,
        explore_even_when_good=True,
        base_backtrack_prob=0.25,
        verify_final_answer=True,
        compare_paths=True,
    )
    
    api_key = os.getenv("NVIDIA_API_KEY")
    pipeline = ImprovedRLPipeline(config=config, results_dir=output_dir, api_key=api_key)
    
    evaluator = AnswerEvaluator()
    
    results = []
    for i, problem in enumerate(problems):
        print(f"\n[{i+1}/{len(problems)}] Problem: {problem.id}")
        print(f"Question: {problem.question[:80]}...")
        
        result = pipeline.solve(
            problem=problem.question,
            problem_id=problem.id,
            ground_truth=problem.answer,
        )
        
        eval_result = evaluator.evaluate(
            predicted=result.final_answer,
            ground_truth=problem.answer,
            problem_id=problem.id,
        )
        
        result.correct = eval_result.correct
        
        results.append({
            "problem_id": problem.id,
            "question": problem.question,
            "ground_truth": problem.answer,
            "prediction": result.final_answer,
            "correct": eval_result.correct,
            "score": result.final_score,
            "tokens": result.total_tokens_input + result.total_tokens_output,
            "latency": result.latency_seconds,
            "expansions": result.num_expansions,
            "backtracks": result.num_backtracks,
            "paths_explored": result.paths_explored,
            "verification": result.verification_score,
            "reasoning_path": result.reasoning_path,
        })
        
        print(f"  Answer: {result.final_answer[:50]}...")
        print(f"  Correct: {eval_result.correct}")
        print(f"  Score: {result.final_score:.2f}")
        print(f"  Backtracks: {result.num_backtracks}")
        print(f"  Paths: {result.paths_explored}")
    
    pipeline.save_results("improved_rl_results.json")
    pipeline.close()
    
    correct_count = sum(1 for r in results if r["correct"])
    accuracy = correct_count / len(results) if results else 0
    avg_tokens = sum(r["tokens"] for r in results) / len(results) if results else 0
    avg_latency = sum(r["latency"] for r in results) / len(results) if results else 0
    avg_backtracks = sum(r["backtracks"] for r in results) / len(results) if results else 0
    avg_paths = sum(r["paths_explored"] for r in results) / len(results) if results else 0
    
    summary = {
        "dataset": dataset,
        "num_samples": len(results),
        "accuracy": accuracy,
        "correct": correct_count,
        "avg_tokens": avg_tokens,
        "avg_latency": avg_latency,
        "avg_backtracks": avg_backtracks,
        "avg_paths_explored": avg_paths,
        "max_iterations": max_iterations,
        "seed": seed,
    }
    
    with open(output_path / "improved_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "=" * 60)
    print("IMPROVED RL-GUIDED RESULTS SUMMARY")
    print("=" * 60)
    print(f"Dataset: {dataset}")
    print(f"Samples: {len(results)}")
    print(f"Accuracy: {accuracy:.1%} ({correct_count}/{len(results)})")
    print(f"Avg Tokens: {avg_tokens:.0f}")
    print(f"Avg Latency: {avg_latency:.1f}s")
    print(f"Avg Backtracks: {avg_backtracks:.1f}")
    print(f"Avg Paths Explored: {avg_paths:.1f}")
    print("=" * 60)
    print("\nLearning Summary:")
    print(pipeline.learner.get_learning_summary())
    print("=" * 60)
    
    return results, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run improved RL experiments")
    parser.add_argument("--dataset", default="strategy_qa", help="Dataset name")
    parser.add_argument("--samples", type=int, default=10, help="Number of samples")
    parser.add_argument("--iterations", type=int, default=30, help="Max MCTS iterations")
    parser.add_argument("--output", default="data/results", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    run_improved_experiment(
        dataset=args.dataset,
        num_samples=args.samples,
        max_iterations=args.iterations,
        output_dir=args.output,
        seed=args.seed,
    )
