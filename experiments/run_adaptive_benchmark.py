"""
Benchmark comparing Fixed vs Adaptive Self-Reflection.
Tests on same questions to verify improvement.
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

from src.orchestration.self_reflection_pipeline import SelfReflectionPipeline, SelfReflectionConfig
from src.orchestration.adaptive_reflection_pipeline import AdaptiveReflectionPipeline, AdaptiveReflectionConfig
from src.orchestration.baseline import BaselineRunner
from evaluation.accuracy import AnswerExtractor


def load_test_problems() -> List[Dict[str, Any]]:
    """Load test problems."""
    problems = []
    
    datasets_dir = Path(__file__).parent.parent / "data" / "datasets"
    
    # Load expanded dataset
    expanded_path = datasets_dir / "expanded_problems.json"
    if expanded_path.exists():
        with open(expanded_path) as f:
            data = json.load(f)
            for item in data:
                problems.append({
                    "id": item.get("id", f"exp_{len(problems)}"),
                    "problem": item.get("question", ""),
                    "answer": item.get("answer", ""),
                })
    
    # Filter yes/no
    problems = [p for p in problems if p["answer"].lower() in ["yes", "no"]]
    
    return problems


def run_baseline(problems: List[Dict]) -> Dict:
    """Run baseline."""
    print("\n" + "="*60)
    print("RUNNING BASELINE")
    print("="*60)
    
    api_key = os.getenv("NVIDIA_API_KEY")
    runner = BaselineRunner(api_key=api_key)
    results = []
    correct = 0
    
    for i, problem in enumerate(problems):
        print(f"\n[{i+1}/{len(problems)}] {problem['problem'][:50]}...")
        
        result = runner.run_single(
            problem=problem["problem"],
            problem_id=problem["id"],
        )
        
        is_correct = AnswerExtractor.check_answer(result.answer, problem["answer"])
        
        results.append({
            "problem_id": problem["id"],
            "problem": problem["problem"],
            "ground_truth": problem["answer"],
            "answer": result.answer,
            "correct": is_correct,
            "tokens": result.input_tokens + result.output_tokens,
        })
        
        if is_correct:
            correct += 1
            print(f"  [OK] Correct")
        else:
            print(f"  [X] Wrong (expected: {problem['answer']})")
    
    return {
        "method": "baseline",
        "num_problems": len(problems),
        "correct": correct,
        "accuracy": correct / len(problems) if problems else 0,
        "results": results,
    }


def run_fixed_reflection(problems: List[Dict]) -> Dict:
    """Run fixed-depth self-reflection."""
    print("\n" + "="*60)
    print("RUNNING FIXED SELF-REFLECTION (Depth=2)")
    print("="*60)
    
    config = SelfReflectionConfig(
        max_iterations=6,
        min_reasoning_steps=2,
        max_reasoning_steps=4,
        reflection_depth=2,  # Fixed depth
        temperature_reason=0.7,
        temperature_reflect=0.3,
        temperature_conclude=0.2,
        enable_selective_reflection=False,  # Disable selective
    )
    
    pipeline = SelfReflectionPipeline(config=config)
    results = []
    correct = 0
    
    for i, problem in enumerate(problems):
        print(f"\n[{i+1}/{len(problems)}] {problem['problem'][:50]}...")
        
        result = pipeline.solve(
            problem=problem["problem"],
            problem_id=problem["id"],
            ground_truth=problem["answer"],
        )
        
        results.append({
            "problem_id": problem["id"],
            "problem": problem["problem"],
            "ground_truth": problem["answer"],
            "answer": result.final_answer,
            "correct": result.correct,
            "tokens": result.total_tokens,
            "num_reflections": len(result.reflections),
        })
        
        if result.correct:
            correct += 1
            print(f"  [OK] Correct")
        else:
            print(f"  [X] Wrong (expected: {problem['answer']})")
    
    pipeline.close()
    
    return {
        "method": "fixed_reflection",
        "num_problems": len(problems),
        "correct": correct,
        "accuracy": correct / len(problems) if problems else 0,
        "results": results,
    }


def run_adaptive_reflection(problems: List[Dict]) -> Dict:
    """Run adaptive self-reflection."""
    print("\n" + "="*60)
    print("RUNNING ADAPTIVE SELF-REFLECTION")
    print("="*60)
    
    config = AdaptiveReflectionConfig(
        min_reflections=1,
        max_reflections=5,
        confidence_threshold_increase=0.7,
        confidence_threshold_stop=0.9,
        degradation_threshold=0.1,
        enable_cross_validation=True,
        validation_samples=3,
        variance_threshold=0.2,
        early_stopping_patience=2,
        temperature_reason=0.7,
        temperature_reflect=0.3,
        temperature_conclude=0.2,
    )
    
    pipeline = AdaptiveReflectionPipeline(config=config)
    results = []
    correct = 0
    
    total_depth = 0
    rollback_count = 0
    overfitting_count = 0
    
    for i, problem in enumerate(problems):
        print(f"\n[{i+1}/{len(problems)}] {problem['problem'][:50]}...")
        
        result = pipeline.solve(
            problem=problem["problem"],
            problem_id=problem["id"],
            ground_truth=problem["answer"],
        )
        
        results.append({
            "problem_id": problem["id"],
            "problem": problem["problem"],
            "ground_truth": problem["answer"],
            "answer": result.final_answer,
            "correct": result.correct,
            "tokens": result.total_tokens,
            "complexity": result.complexity_score,
            "recommended_depth": result.recommended_depth,
            "actual_depth": result.actual_depth,
            "rolled_back": result.rolled_back,
            "overfitting_detected": result.overfitting_detected,
        })
        
        total_depth += result.actual_depth
        if result.rolled_back:
            rollback_count += 1
        if result.overfitting_detected:
            overfitting_count += 1
        
        if result.correct:
            correct += 1
            print(f"  [OK] Correct (depth={result.actual_depth}, rolled_back={result.rolled_back})")
        else:
            print(f"  [X] Wrong (depth={result.actual_depth}, expected: {problem['answer']})")
    
    pipeline.close()
    
    avg_depth = total_depth / len(problems) if problems else 0
    
    return {
        "method": "adaptive_reflection",
        "num_problems": len(problems),
        "correct": correct,
        "accuracy": correct / len(problems) if problems else 0,
        "avg_depth": avg_depth,
        "rollback_count": rollback_count,
        "overfitting_count": overfitting_count,
        "results": results,
    }


def statistical_analysis(baseline: Dict, fixed: Dict, adaptive: Dict) -> Dict:
    """Perform statistical analysis."""
    n = baseline["num_problems"]
    
    # McNemar's test for baseline vs fixed
    b_fixed = sum(1 for b, f in zip(baseline["results"], fixed["results"]) 
                  if b["correct"] and not f["correct"])
    c_fixed = sum(1 for b, f in zip(baseline["results"], fixed["results"]) 
                  if not b["correct"] and f["correct"])
    
    if b_fixed + c_fixed > 0:
        mcnemar_fixed = (abs(b_fixed - c_fixed) - 1) ** 2 / (b_fixed + c_fixed)
    else:
        mcnemar_fixed = 0
    
    # McNemar's test for baseline vs adaptive
    b_adaptive = sum(1 for b, a in zip(baseline["results"], adaptive["results"]) 
                     if b["correct"] and not a["correct"])
    c_adaptive = sum(1 for b, a in zip(baseline["results"], adaptive["results"]) 
                     if not b["correct"] and a["correct"])
    
    if b_adaptive + c_adaptive > 0:
        mcnemar_adaptive = (abs(b_adaptive - c_adaptive) - 1) ** 2 / (b_adaptive + c_adaptive)
    else:
        mcnemar_adaptive = 0
    
    # McNemar's test for fixed vs adaptive
    b_compare = sum(1 for f, a in zip(fixed["results"], adaptive["results"]) 
                    if f["correct"] and not a["correct"])
    c_compare = sum(1 for f, a in zip(fixed["results"], adaptive["results"]) 
                    if not f["correct"] and a["correct"])
    
    if b_compare + c_compare > 0:
        mcnemar_compare = (abs(b_compare - c_compare) - 1) ** 2 / (b_compare + c_compare)
    else:
        mcnemar_compare = 0
    
    # Calculate p-values
    try:
        from scipy import stats
        p_fixed = 1 - stats.chi2.cdf(mcnemar_fixed, 1) if mcnemar_fixed > 0 else 1.0
        p_adaptive = 1 - stats.chi2.cdf(mcnemar_adaptive, 1) if mcnemar_adaptive > 0 else 1.0
        p_compare = 1 - stats.chi2.cdf(mcnemar_compare, 1) if mcnemar_compare > 0 else 1.0
    except ImportError:
        p_fixed = 0.5 * (1 + math.erf(-mcnemar_fixed / math.sqrt(2))) if mcnemar_fixed > 0 else 1.0
        p_adaptive = 0.5 * (1 + math.erf(-mcnemar_adaptive / math.sqrt(2))) if mcnemar_adaptive > 0 else 1.0
        p_compare = 0.5 * (1 + math.erf(-mcnemar_compare / math.sqrt(2))) if mcnemar_compare > 0 else 1.0
    
    return {
        "baseline_accuracy": baseline["accuracy"],
        "fixed_accuracy": fixed["accuracy"],
        "adaptive_accuracy": adaptive["accuracy"],
        "fixed_improvement": fixed["accuracy"] - baseline["accuracy"],
        "adaptive_improvement": adaptive["accuracy"] - baseline["accuracy"],
        "adaptive_vs_fixed": adaptive["accuracy"] - fixed["accuracy"],
        "mcnemar_fixed": {"statistic": mcnemar_fixed, "p_value": p_fixed},
        "mcnemar_adaptive": {"statistic": mcnemar_adaptive, "p_value": p_adaptive},
        "mcnemar_compare": {"statistic": mcnemar_compare, "p_value": p_compare},
        "avg_depth": adaptive.get("avg_depth", 0),
        "rollback_count": adaptive.get("rollback_count", 0),
        "overfitting_count": adaptive.get("overfitting_count", 0),
    }


def main():
    """Run benchmark."""
    print("="*60)
    print("BENCHMARK: Fixed vs Adaptive Self-Reflection")
    print("="*60)
    
    results_dir = Path(__file__).parent.parent / "data" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Load problems
    problems = load_test_problems()
    print(f"\nLoaded {len(problems)} problems")
    
    # Sample for reasonable runtime
    max_problems = min(30, len(problems))
    random.seed(42)
    problems = random.sample(problems, max_problems)
    print(f"Testing on {len(problems)} problems")
    
    # Run all methods
    baseline_results = run_baseline(problems)
    fixed_results = run_fixed_reflection(problems)
    adaptive_results = run_adaptive_reflection(problems)
    
    # Statistical analysis
    stats = statistical_analysis(baseline_results, fixed_results, adaptive_results)
    
    # Save results
    combined = {
        "timestamp": datetime.now().isoformat(),
        "num_problems": len(problems),
        "baseline": baseline_results,
        "fixed_reflection": fixed_results,
        "adaptive_reflection": adaptive_results,
        "statistical_analysis": stats,
    }
    
    results_path = results_dir / "adaptive_vs_fixed_benchmark.json"
    with open(results_path, "w") as f:
        json.dump(combined, f, indent=2, default=str)
    
    # Print summary
    print("\n" + "="*60)
    print("BENCHMARK RESULTS")
    print("="*60)
    
    print(f"\nProblems tested: {len(problems)}")
    print(f"\nBaseline accuracy:     {stats['baseline_accuracy']:.1%}")
    print(f"Fixed reflection:      {stats['fixed_accuracy']:.1%} (improvement: {stats['fixed_improvement']:+.1%})")
    print(f"Adaptive reflection:   {stats['adaptive_accuracy']:.1%} (improvement: {stats['adaptive_improvement']:+.1%})")
    
    print(f"\nAdaptive vs Fixed:     {stats['adaptive_vs_fixed']:+.1%}")
    
    print(f"\nAdaptive metrics:")
    print(f"  Average depth: {stats['avg_depth']:.2f}")
    print(f"  Rollbacks: {stats['rollback_count']}")
    print(f"  Overfitting detected: {stats['overfitting_count']}")
    
    print(f"\nStatistical significance:")
    print(f"  Baseline vs Fixed: p={stats['mcnemar_fixed']['p_value']:.4f}")
    print(f"  Baseline vs Adaptive: p={stats['mcnemar_adaptive']['p_value']:.4f}")
    print(f"  Fixed vs Adaptive: p={stats['mcnemar_compare']['p_value']:.4f}")
    
    print(f"\nResults saved to: {results_path}")
    
    return stats


if __name__ == "__main__":
    main()
