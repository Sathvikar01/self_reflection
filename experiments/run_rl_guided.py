"""Run RL-guided reasoning experiments."""

import os
import sys
import json
import argparse
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.orchestration.pipeline import RLPipeline, PipelineConfig
from src.rl_controller.mcts import MCTSConfig
from src.rl_controller.actions import ActionConfig
from src.evaluator.prm_client import PRMConfig
from evaluation.accuracy import AnswerEvaluator
from data.datasets.loader import DataLoader


def run_rl_guided(
    dataset: str = "strategy_qa",
    num_samples: int = 100,
    max_iterations: int = 50,
    output_dir: str = "data/results",
    seed: int = 42,
):
    """Run RL-guided experiments.
    
    Args:
        dataset: Dataset name
        num_samples: Number of samples to evaluate
        max_iterations: Maximum MCTS iterations per problem
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
    
    mcts_config = MCTSConfig(
        exploration_constant=1.414,
        max_tree_depth=20,
        expansion_budget=max_iterations,
        temperature=0.7,
    )
    
    action_config = ActionConfig(
        expand_temperature=0.7,
        reflect_temperature=0.5,
        backtrack_threshold=0.3,
    )
    
    prm_config = PRMConfig(
        model="meta/llama-3.3-70b-instruct",
        temperature=0.1,
    )
    
    config = PipelineConfig(
        max_iterations=max_iterations,
        early_stop_score=0.95,
        mcts=mcts_config,
        action=action_config,
        prm=prm_config,
    )
    
    api_key = os.getenv("NVIDIA_API_KEY")
    pipeline = RLPipeline(config=config, results_dir=output_dir, api_key=api_key)
    
    evaluator = AnswerEvaluator()
    
    results = []
    for i, problem in enumerate(problems):
        print(f"\nProcessing problem {i+1}/{len(problems)}: {problem.id}")
        
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
            "reflections": result.num_reflections,
            "backtracks": result.num_backtracks,
            "reasoning_path": result.reasoning_path,
        })
        
        if (i + 1) % 10 == 0:
            pipeline.save_results(f"rl_results_checkpoint_{i+1}.json")
    
    pipeline.save_results("rl_results.json")
    pipeline.close()
    
    correct_count = sum(1 for r in results if r["correct"])
    accuracy = correct_count / len(results)
    avg_tokens = sum(r["tokens"] for r in results) / len(results)
    avg_latency = sum(r["latency"] for r in results) / len(results)
    avg_expansions = sum(r["expansions"] for r in results) / len(results)
    avg_backtracks = sum(r["backtracks"] for r in results) / len(results)
    
    summary = {
        "dataset": dataset,
        "num_samples": len(results),
        "accuracy": accuracy,
        "correct": correct_count,
        "avg_tokens": avg_tokens,
        "avg_latency": avg_latency,
        "avg_expansions": avg_expansions,
        "avg_backtracks": avg_backtracks,
        "max_iterations": max_iterations,
        "seed": seed,
    }
    
    with open(output_path / "rl_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "=" * 50)
    print("RL-GUIDED RESULTS SUMMARY")
    print("=" * 50)
    print(f"Dataset: {dataset}")
    print(f"Samples: {len(results)}")
    print(f"Accuracy: {accuracy:.1%} ({correct_count}/{len(results)})")
    print(f"Avg Tokens: {avg_tokens:.0f}")
    print(f"Avg Latency: {avg_latency:.1f}s")
    print(f"Avg Expansions: {avg_expansions:.1f}")
    print(f"Avg Backtracks: {avg_backtracks:.1f}")
    print("=" * 50)
    
    return results, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RL-guided experiments")
    parser.add_argument("--dataset", default="strategy_qa", help="Dataset name")
    parser.add_argument("--samples", type=int, default=100, help="Number of samples")
    parser.add_argument("--iterations", type=int, default=50, help="Max MCTS iterations")
    parser.add_argument("--output", default="data/results", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    run_rl_guided(
        dataset=args.dataset,
        num_samples=args.samples,
        max_iterations=args.iterations,
        output_dir=args.output,
        seed=args.seed,
    )
