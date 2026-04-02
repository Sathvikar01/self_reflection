"""Run simplified improved experiment."""

import os
import sys
import json
import argparse
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.orchestration.simplified_pipeline import SimplifiedRLPipeline, SimplifiedConfig
from evaluation.accuracy import AnswerEvaluator
from data.datasets.loader import DataLoader


def run_simplified(
    dataset: str = "strategy_qa",
    num_samples: int = 10,
    output_dir: str = "data/results",
    seed: int = 42,
):
    """Run simplified experiment with all fixes."""
    import random
    random.seed(seed)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    loader = DataLoader()
    problems = loader.load(dataset, split="test", n=num_samples, seed=seed)
    
    print("\n" + "=" * 60)
    print("SIMPLIFIED IMPROVED EXPERIMENT")
    print("Fixes: vacuous detection, progress evaluation, backtracking,")
    print("       different verification model, comparative learning")
    print("=" * 60)
    print(f"Dataset: {dataset}")
    print(f"Samples: {num_samples}")
    print("=" * 60)
    
    config = SimplifiedConfig(
        max_steps=5,
        min_steps=3,
        backtrack_probability=0.3,
        verify_final=True,
        explore_alternatives=True,
    )
    
    api_key = os.getenv("NVIDIA_API_KEY")
    pipeline = SimplifiedRLPipeline(api_key=api_key, config=config)
    evaluator = AnswerEvaluator()
    
    results = []
    for i, problem in enumerate(problems):
        print(f"\n[{i+1}/{len(problems)}] {problem.id}")
        print(f"Q: {problem.question[:70]}...")
        
        result = pipeline.solve(
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
            "id": problem.id,
            "question": problem.question,
            "answer": result.answer[:100],
            "correct": eval_result.correct,
            "score": result.final_score,
            "tokens": result.tokens_used,
            "backtracks": result.backtracks,
            "paths": result.paths_explored,
        })
        
        print(f"A: {result.answer[:50]}...")
        print(f"Correct: {eval_result.correct}, Score: {result.final_score:.2f}")
    
    pipeline.save_results(str(output_path / "simplified_results.json"))
    
    correct_count = sum(1 for r in results if r["correct"])
    accuracy = correct_count / len(results) if results else 0
    avg_score = sum(r["score"] for r in results) / len(results) if results else 0
    avg_tokens = sum(r["tokens"] for r in results) / len(results) if results else 0
    
    summary = {
        "dataset": dataset,
        "samples": len(results),
        "accuracy": accuracy,
        "correct": correct_count,
        "avg_score": avg_score,
        "avg_tokens": avg_tokens,
    }
    
    with open(output_path / "simplified_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Accuracy: {accuracy:.1%} ({correct_count}/{len(results)})")
    print(f"Avg Score: {avg_score:.2f}")
    print(f"Avg Tokens: {avg_tokens:.0f}")
    print("=" * 60)
    print("\n" + pipeline.get_summary())
    
    return results, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="strategy_qa")
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--output", default="data/results")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    run_simplified(args.dataset, args.samples, args.output, args.seed)
