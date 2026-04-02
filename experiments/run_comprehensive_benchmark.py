"""
Comprehensive benchmark comparing baseline vs self-reflection pipeline.
- Loads all available test problems
- Runs both pipelines
- Performs statistical analysis
"""

import os
import sys
import json
import time
import random
import math
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from src.orchestration.baseline import BaselineRunner
from src.orchestration.self_reflection_pipeline import SelfReflectionPipeline, SelfReflectionConfig
from evaluation.accuracy import AnswerExtractor


def load_all_problems() -> List[Dict[str, Any]]:
    """Load all available test problems from datasets."""
    problems = []
    
    datasets_dir = Path(__file__).parent.parent / "data" / "datasets"
    
    # Load strategyqa_test.json
    strategyqa_test = datasets_dir / "strategyqa_test.json"
    if strategyqa_test.exists():
        with open(strategyqa_test) as f:
            data = json.load(f)
            for item in data:
                problems.append({
                    "id": item.get("id", f"sq_{len(problems)}"),
                    "problem": item.get("question", ""),
                    "answer": item.get("answer", ""),
                })
    
    # Load strategic_test.json
    strategic_test = datasets_dir / "strategic_test.json"
    if strategic_test.exists():
        with open(strategic_test) as f:
            data = json.load(f)
            for item in data:
                problems.append({
                    "id": item.get("id", f"st_{len(problems)}"),
                    "problem": item.get("question", ""),
                    "answer": item.get("answer", ""),
                })
    
    # Load strategyqa_train.json for more samples
    strategyqa_train = datasets_dir / "strategyqa_train.json"
    if strategyqa_train.exists():
        with open(strategyqa_train) as f:
            data = json.load(f)
            for item in data:
                problems.append({
                    "id": item.get("id", f"sqt_{len(problems)}"),
                    "problem": item.get("question", ""),
                    "answer": item.get("answer", ""),
                })
    
    # Load strategic_train.json for more samples
    strategic_train = datasets_dir / "strategic_train.json"
    if strategic_train.exists():
        with open(strategic_train) as f:
            data = json.load(f)
            for item in data:
                problems.append({
                    "id": item.get("id", f"stt_{len(problems)}"),
                    "problem": item.get("question", ""),
                    "answer": item.get("answer", ""),
                })
    
    # Filter out "it depends" answers (not yes/no)
    problems = [p for p in problems if p["answer"].lower() in ["yes", "no"]]
    
    return problems


def run_baseline(problems: List[Dict], results_dir: Path) -> Dict:
    """Run baseline zero-shot on all problems."""
    print("\n" + "="*60)
    print("RUNNING BASELINE (Zero-Shot)")
    print("="*60)
    
    api_key = os.getenv("NVIDIA_API_KEY")
    runner = BaselineRunner(api_key=api_key)
    results = []
    correct = 0
    
    for i, problem in enumerate(problems):
        print(f"\n[{i+1}/{len(problems)}] {problem['problem'][:60]}...")
        
        result = runner.run_single(
            problem=problem["problem"],
            problem_id=problem["id"],
        )
        
        # Check correctness
        is_correct = AnswerExtractor.check_answer(
            result.answer, 
            problem["answer"]
        )
        
        results.append({
            "problem_id": problem["id"],
            "problem": problem["problem"],
            "ground_truth": problem["answer"],
            "baseline_answer": result.answer,
            "baseline_correct": is_correct,
            "baseline_tokens": result.input_tokens + result.output_tokens,
        })
        
        if is_correct:
            correct += 1
            print(f"  [OK] Correct (answer: {result.answer})")
        else:
            print(f"  [X] Wrong (answer: {result.answer}, expected: {problem['answer']})")
    
    accuracy = correct / len(problems) if problems else 0
    
    return {
        "method": "baseline",
        "num_problems": len(problems),
        "correct": correct,
        "accuracy": accuracy,
        "results": results,
    }


def run_self_reflection(problems: List[Dict], results_dir: Path) -> Dict:
    """Run self-reflection pipeline on all problems."""
    print("\n" + "="*60)
    print("RUNNING SELF-REFLECTION PIPELINE")
    print("="*60)
    
    config = SelfReflectionConfig(
        max_iterations=6,
        min_reasoning_steps=2,
        max_reasoning_steps=4,
        reflection_depth=2,
        temperature_reason=0.7,
        temperature_reflect=0.3,
        temperature_conclude=0.2,
    )
    
    pipeline = SelfReflectionPipeline(config=config, results_dir=str(results_dir))
    results = []
    correct = 0
    
    for i, problem in enumerate(problems):
        print(f"\n[{i+1}/{len(problems)}] {problem['problem'][:60]}...")
        
        result = pipeline.solve(
            problem=problem["problem"],
            problem_id=problem["id"],
            ground_truth=problem["answer"],
        )
        
        results.append({
            "problem_id": problem["id"],
            "problem": problem["problem"],
            "ground_truth": problem["answer"],
            "reflection_answer": result.final_answer,
            "reflection_correct": result.correct,
            "reflection_tokens": result.total_tokens,
            "num_reflections": len(result.reflections),
            "num_corrections": len(result.corrections),
        })
        
        if result.correct:
            correct += 1
            print(f"  [OK] Correct (answer: {result.final_answer})")
        else:
            print(f"  [X] Wrong (answer: {result.final_answer}, expected: {problem['answer']})")
    
    accuracy = correct / len(problems) if problems else 0
    
    pipeline.close()
    
    return {
        "method": "self_reflection",
        "num_problems": len(problems),
        "correct": correct,
        "accuracy": accuracy,
        "results": results,
    }


def statistical_analysis(baseline_results: Dict, reflection_results: Dict) -> Dict:
    """Perform statistical analysis on results."""
    import math
    
    n = baseline_results["num_problems"]
    baseline_correct = baseline_results["correct"]
    reflection_correct = reflection_results["correct"]
    
    baseline_acc = baseline_results["accuracy"]
    reflection_acc = reflection_results["accuracy"]
    
    # Paired analysis: count agreement/disagreement
    baseline_outcomes = [r["baseline_correct"] for r in baseline_results["results"]]
    reflection_outcomes = [r["reflection_correct"] for r in reflection_results["results"]]
    
    # Count cases
    both_correct = sum(1 for b, r in zip(baseline_outcomes, reflection_outcomes) if b and r)
    both_wrong = sum(1 for b, r in zip(baseline_outcomes, reflection_outcomes) if not b and not r)
    baseline_only = sum(1 for b, r in zip(baseline_outcomes, reflection_outcomes) if b and not r)
    reflection_only = sum(1 for b, r in zip(baseline_outcomes, reflection_outcomes) if not b and r)
    
    # McNemar's test (for paired binary outcomes)
    # Tests if the discordant pairs are significantly different
    b = baseline_only  # baseline correct, reflection wrong
    c = reflection_only  # baseline wrong, reflection correct
    
    # McNemar's test statistic with continuity correction
    if b + c > 0:
        mcnemar_stat = (abs(b - c) - 1) ** 2 / (b + c)
    else:
        mcnemar_stat = 0
    
    # p-value approximation (chi-square with 1 df)
    # For McNemar's test, we use chi-square distribution
    try:
        from scipy import stats as scipy_stats
        p_value = 1 - scipy_stats.chi2.cdf(mcnemar_stat, 1) if mcnemar_stat > 0 else 1.0
    except ImportError:
        # Approximate p-value using normal distribution
        p_value = 0.5 * (1 + math.erf(-mcnemar_stat / math.sqrt(2))) if mcnemar_stat > 0 else 1.0
    
    # Confidence interval for difference in proportions
    diff = reflection_acc - baseline_acc
    se = math.sqrt((baseline_acc * (1 - baseline_acc) + reflection_acc * (1 - reflection_acc)) / n)
    ci_95_lower = diff - 1.96 * se
    ci_95_upper = diff + 1.96 * se
    
    # Improvement metrics
    improvement = reflection_acc - baseline_acc
    relative_improvement = (reflection_acc - baseline_acc) / baseline_acc if baseline_acc > 0 else 0
    
    return {
        "baseline_accuracy": baseline_acc,
        "reflection_accuracy": reflection_acc,
        "improvement_absolute": improvement,
        "improvement_relative": relative_improvement,
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "baseline_only_correct": baseline_only,
        "reflection_only_correct": reflection_only,
        "mcnemar_statistic": mcnemar_stat,
        "p_value": p_value,
        "ci_95_lower": ci_95_lower,
        "ci_95_upper": ci_95_upper,
        "statistically_significant": p_value < 0.05,
    }


def main():
    """Run comprehensive benchmark."""
    print("="*60)
    print("COMPREHENSIVE BENCHMARK: Baseline vs Self-Reflection")
    print("="*60)
    
    # Setup
    results_dir = Path(__file__).parent.parent / "data" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Load problems
    problems = load_all_problems()
    print(f"\nLoaded {len(problems)} problems")
    
    # Limit to 30 problems for reasonable runtime (API costs)
    max_problems = 30
    if len(problems) > max_problems:
        random.seed(42)  # Reproducibility
        problems = random.sample(problems, max_problems)
        print(f"Sampled {len(problems)} problems for benchmark")
    
    # Run baseline
    baseline_results = run_baseline(problems, results_dir)
    
    # Run self-reflection
    reflection_results = run_self_reflection(problems, results_dir)
    
    # Statistical analysis
    stats = statistical_analysis(baseline_results, reflection_results)
    
    def convert_to_serializable(obj):
        """Convert numpy types to Python native types for JSON serialization."""
        try:
            import numpy as np
            if isinstance(obj, (np.bool_, np.bool)):
                return bool(obj)
            if isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            if isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
        except ImportError:
            pass
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert_to_serializable(v) for v in obj]
        return obj

    # Save combined results
    combined_results = {
        "timestamp": datetime.now().isoformat(),
        "num_problems": len(problems),
        "baseline": baseline_results,
        "self_reflection": reflection_results,
        "statistical_analysis": stats,
    }
    
    combined_results = convert_to_serializable(combined_results)
    
    results_path = results_dir / "comprehensive_benchmark.json"
    with open(results_path, "w") as f:
        json.dump(combined_results, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("BENCHMARK RESULTS")
    print("="*60)
    print(f"\nProblems tested: {len(problems)}")
    print(f"\nBaseline accuracy: {stats['baseline_accuracy']:.1%}")
    print(f"Self-reflection accuracy: {stats['reflection_accuracy']:.1%}")
    print(f"\nImprovement: {stats['improvement_absolute']:+.1%} (absolute)")
    print(f"             {stats['improvement_relative']:+.1%} (relative)")
    print(f"\n95% CI: [{stats['ci_95_lower']:+.1%}, {stats['ci_95_upper']:+.1%}]")
    print(f"\nMcNemar's test: chi2 = {stats['mcnemar_statistic']:.2f}, p = {stats['p_value']:.4f}")
    print(f"Statistically significant (p < 0.05): {stats['statistically_significant']}")
    
    print(f"\nBreakdown:")
    print(f"  Both correct: {stats['both_correct']}")
    print(f"  Both wrong: {stats['both_wrong']}")
    print(f"  Baseline only correct: {stats['baseline_only_correct']}")
    print(f"  Self-reflection only correct: {stats['reflection_only_correct']}")
    
    print(f"\nResults saved to: {results_path}")
    
    return stats


if __name__ == "__main__":
    main()
