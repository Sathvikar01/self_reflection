"""Run comparison experiments with proper cross-validation."""

import os
import sys
import json
import argparse
import random
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.orchestration.baseline import BaselineRunner, BaselineConfig
from src.orchestration.robust_pipeline import RobustRLPipeline, RobustPipelineConfig
from evaluation.accuracy import AnswerEvaluator, AnswerExtractor
from data.datasets.loader import DataLoader


def run_baseline_experiment(
    problems: list,
    output_dir: str,
    seed: int = 42,
) -> dict:
    """Run baseline experiment."""
    
    random.seed(seed)
    
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
        print(f"\n[BASELINE] Problem {i+1}/{len(problems)}: {problem.id}")
        
        result = runner.run_single(
            problem=problem.question,
            problem_id=problem.id,
            ground_truth=problem.answer,
        )
        
        eval_result = evaluator.evaluate(
            predicted=result.answer,
            ground_truth=problem.answer,
            problem_id=problem.id,
            full_response=result.full_response,
        )
        
        result.correct = eval_result.correct
        
        results.append({
            "problem_id": problem.id,
            "question": problem.question,
            "ground_truth": problem.answer,
            "prediction": result.answer,
            "extracted": eval_result.extracted_answer,
            "correct": eval_result.correct,
            "match_type": eval_result.match_type,
            "tokens": result.input_tokens + result.output_tokens,
            "latency": result.latency_seconds,
        })
        
        print(f"  Answer: {eval_result.extracted_answer[:50] if eval_result.extracted_answer else 'N/A'}...")
        print(f"  Correct: {eval_result.correct}")
    
    runner.save_results("baseline_results.json")
    
    correct_count = sum(1 for r in results if r["correct"])
    
    return {
        "results": results,
        "accuracy": correct_count / len(results) if results else 0,
        "correct": correct_count,
        "total": len(results),
        "avg_tokens": sum(r["tokens"] for r in results) / len(results) if results else 0,
        "avg_latency": sum(r["latency"] for r in results) / len(results) if results else 0,
    }


def run_rl_experiment(
    problems: list,
    output_dir: str,
    max_iterations: int = 20,
    seed: int = 42,
) -> dict:
    """Run robust RL experiment."""
    
    random.seed(seed)
    
    config = RobustPipelineConfig(
        max_iterations=max_iterations,
        min_steps_before_conclude=3,
        max_steps=6,
        beam_width=3,
        exploration_temp=0.7,
        conclusion_temp=0.3,
    )
    
    api_key = os.getenv("NVIDIA_API_KEY")
    pipeline = RobustRLPipeline(config=config, results_dir=output_dir, api_key=api_key)
    
    results = []
    
    for i, problem in enumerate(problems):
        print(f"\n[RL] Problem {i+1}/{len(problems)}: {problem.id}")
        print(f"  Question: {problem.question[:60]}...")
        
        result = pipeline.solve(
            problem=problem.question,
            problem_id=problem.id,
            ground_truth=problem.answer,
        )
        
        results.append({
            "problem_id": problem.id,
            "question": problem.question,
            "ground_truth": problem.answer,
            "prediction": result.final_answer,
            "correct": result.correct,
            "score": result.final_score,
            "tokens": result.total_tokens_input + result.total_tokens_output,
            "latency": result.latency_seconds,
            "num_steps": len(result.reasoning_path),
            "paths_explored": result.paths_explored,
        })
        
        print(f"  Answer: {result.final_answer[:50]}...")
        print(f"  Correct: {result.correct}")
        print(f"  Score: {result.final_score:.2f}")
    
    pipeline.save_results("rl_results.json")
    pipeline.close()
    
    correct_count = sum(1 for r in results if r["correct"])
    
    return {
        "results": results,
        "accuracy": correct_count / len(results) if results else 0,
        "correct": correct_count,
        "total": len(results),
        "avg_tokens": sum(r["tokens"] for r in results) / len(results) if results else 0,
        "avg_latency": sum(r["latency"] for r in results) / len(results) if results else 0,
        "avg_score": sum(r["score"] for r in results) / len(results) if results else 0,
    }


def run_cross_validation(
    dataset: str = "strategy_qa",
    n_folds: int = 5,
    n_samples: int = 50,
    output_dir: str = "data/results/cv",
    seed: int = 42,
):
    """Run k-fold cross-validation to ensure no overfitting."""
    
    print(f"\n{'='*60}")
    print(f"CROSS-VALIDATION: {n_folds} folds on {dataset}")
    print(f"{'='*60}")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    loader = DataLoader()
    all_problems = loader.load(dataset, split="test", n=n_samples * n_folds, seed=seed)
    
    random.seed(seed)
    random.shuffle(all_problems)
    
    fold_size = len(all_problems) // n_folds
    
    cv_results = {
        "baseline": {"folds": [], "mean_accuracy": 0, "std_accuracy": 0},
        "rl": {"folds": [], "mean_accuracy": 0, "std_accuracy": 0},
    }
    
    for fold in range(n_folds):
        print(f"\n{'='*60}")
        print(f"FOLD {fold + 1}/{n_folds}")
        print(f"{'='*60}")
        
        start_idx = fold * fold_size
        end_idx = start_idx + fold_size
        fold_problems = all_problems[start_idx:end_idx]
        
        fold_dir = output_path / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nRunning BASELINE for fold {fold + 1}...")
        baseline_result = run_baseline_experiment(
            problems=fold_problems,
            output_dir=str(fold_dir),
            seed=seed + fold,
        )
        cv_results["baseline"]["folds"].append(baseline_result)
        
        print(f"\nRunning RL for fold {fold + 1}...")
        rl_result = run_rl_experiment(
            problems=fold_problems,
            output_dir=str(fold_dir),
            seed=seed + fold,
        )
        cv_results["rl"]["folds"].append(rl_result)
        
        print(f"\nFold {fold + 1} Results:")
        print(f"  Baseline: {baseline_result['accuracy']:.1%} ({baseline_result['correct']}/{baseline_result['total']})")
        print(f"  RL: {rl_result['accuracy']:.1%} ({rl_result['correct']}/{rl_result['total']})")
    
    import statistics
    
    baseline_accs = [f["accuracy"] for f in cv_results["baseline"]["folds"]]
    rl_accs = [f["accuracy"] for f in cv_results["rl"]["folds"]]
    
    cv_results["baseline"]["mean_accuracy"] = statistics.mean(baseline_accs)
    cv_results["baseline"]["std_accuracy"] = statistics.stdev(baseline_accs) if len(baseline_accs) > 1 else 0
    
    cv_results["rl"]["mean_accuracy"] = statistics.mean(rl_accs)
    cv_results["rl"]["std_accuracy"] = statistics.stdev(rl_accs) if len(rl_accs) > 1 else 0
    
    print(f"\n{'='*60}")
    print("CROSS-VALIDATION RESULTS")
    print(f"{'='*60}")
    print(f"Baseline: {cv_results['baseline']['mean_accuracy']:.1%} ± {cv_results['baseline']['std_accuracy']:.1%}")
    print(f"RL: {cv_results['rl']['mean_accuracy']:.1%} ± {cv_results['rl']['std_accuracy']:.1%}")
    
    improvement = cv_results["rl"]["mean_accuracy"] - cv_results["baseline"]["mean_accuracy"]
    print(f"\nImprovement: {improvement:+.1%} ({improvement*100:.1f} percentage points)")
    
    results_file = output_path / "cv_results.json"
    with open(results_file, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "config": {
                "dataset": dataset,
                "n_folds": n_folds,
                "n_samples_per_fold": fold_size,
                "seed": seed,
            },
            "results": cv_results,
        }, f, indent=2, default=str)
    
    return cv_results


def run_single_comparison(
    dataset: str = "strategy_qa",
    n_samples: int = 20,
    output_dir: str = "data/results",
    seed: int = 42,
):
    """Run single comparison (for quick testing)."""
    
    print(f"\n{'='*60}")
    print(f"SINGLE COMPARISON: {dataset} ({n_samples} samples)")
    print(f"{'='*60}")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    loader = DataLoader()
    problems = loader.load(dataset, split="test", n=n_samples, seed=seed)
    
    print(f"\nLoaded {len(problems)} problems")
    
    print(f"\n{'='*60}")
    print("RUNNING BASELINE")
    print(f"{'='*60}")
    baseline_result = run_baseline_experiment(
        problems=problems,
        output_dir=output_dir,
        seed=seed,
    )
    
    print(f"\n{'='*60}")
    print("RUNNING RL")
    print(f"{'='*60}")
    rl_result = run_rl_experiment(
        problems=problems,
        output_dir=output_dir,
        seed=seed,
    )
    
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Baseline: {baseline_result['accuracy']:.1%} ({baseline_result['correct']}/{baseline_result['total']})")
    print(f"RL: {rl_result['accuracy']:.1%} ({rl_result['correct']}/{rl_result['total']})")
    
    improvement = rl_result["accuracy"] - baseline_result["accuracy"]
    print(f"\nImprovement: {improvement:+.1%} ({improvement*100:.1f} percentage points)")
    
    summary = {
        "timestamp": datetime.now().isoformat(),
        "dataset": dataset,
        "n_samples": n_samples,
        "seed": seed,
        "baseline": {
            "accuracy": baseline_result["accuracy"],
            "correct": baseline_result["correct"],
            "total": baseline_result["total"],
            "avg_tokens": baseline_result["avg_tokens"],
            "avg_latency": baseline_result["avg_latency"],
        },
        "rl": {
            "accuracy": rl_result["accuracy"],
            "correct": rl_result["correct"],
            "total": rl_result["total"],
            "avg_tokens": rl_result["avg_tokens"],
            "avg_latency": rl_result["avg_latency"],
            "avg_score": rl_result["avg_score"],
        },
        "improvement": improvement,
        "improvement_pct": improvement * 100,
    }
    
    with open(output_path / "comparison_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run comparison experiments")
    parser.add_argument("--mode", choices=["single", "cv"], default="single", help="Single comparison or cross-validation")
    parser.add_argument("--dataset", default="strategy_qa", help="Dataset name")
    parser.add_argument("--samples", type=int, default=20, help="Number of samples (per fold for CV)")
    parser.add_argument("--folds", type=int, default=5, help="Number of folds for CV")
    parser.add_argument("--output", default="data/results", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    if args.mode == "cv":
        run_cross_validation(
            dataset=args.dataset,
            n_folds=args.folds,
            n_samples=args.samples,
            output_dir=f"{args.output}/cv",
            seed=args.seed,
        )
    else:
        run_single_comparison(
            dataset=args.dataset,
            n_samples=args.samples,
            output_dir=args.output,
            seed=args.seed,
        )
