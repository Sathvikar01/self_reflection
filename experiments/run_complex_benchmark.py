"""
Comprehensive benchmark: Baseline vs Fixed vs Adaptive Self-Reflection
on complex reasoning problems requiring multi-step analysis.
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


def load_complex_problems() -> List[Dict[str, Any]]:
    """Load complex reasoning problems."""
    problems = []
    
    datasets_dir = Path(__file__).parent.parent / "data" / "datasets"
    
    # Load complex reasoning dataset
    complex_path = datasets_dir / "complex_reasoning.json"
    if complex_path.exists():
        with open(complex_path, encoding="utf-8") as f:
            data = json.load(f)
            for item in data:
                # Handle both yes/no and numeric answers
                answer = str(item.get("answer", "")).lower()
                problems.append({
                    "id": item.get("id", f"complex_{len(problems)}"),
                    "problem": item.get("question", ""),
                    "answer": answer,
                    "complexity": item.get("complexity", "medium"),
                    "category": item.get("category", "general"),
                    "reasoning_steps": item.get("reasoning_steps", 3),
                })
    
    return problems


def run_baseline(problems: List[Dict]) -> Dict:
    """Run baseline zero-shot."""
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
        
        is_correct = AnswerExtractor.check_answer(result.answer, problem["answer"])
        
        results.append({
            "problem_id": problem["id"],
            "problem": problem["problem"],
            "ground_truth": problem["answer"],
            "complexity": problem["complexity"],
            "category": problem["category"],
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
    """Run fixed-depth self-reflection (depth=2)."""
    print("\n" + "="*60)
    print("RUNNING FIXED SELF-REFLECTION (Depth=2)")
    print("="*60)
    
    config = SelfReflectionConfig(
        max_iterations=6,
        min_reasoning_steps=2,
        max_reasoning_steps=4,
        reflection_depth=2,
        temperature_reason=0.7,
        temperature_reflect=0.3,
        temperature_conclude=0.2,
        enable_selective_reflection=False,
    )
    
    pipeline = SelfReflectionPipeline(config=config)
    results = []
    correct = 0
    total_reflections = 0
    
    for i, problem in enumerate(problems):
        print(f"\n[{i+1}/{len(problems)}] {problem['problem'][:60]}...")
        
        result = pipeline.solve(
            problem=problem["problem"],
            problem_id=problem["id"],
            ground_truth=problem["answer"],
        )
        
        num_reflections = len(result.reflections)
        total_reflections += num_reflections
        
        results.append({
            "problem_id": problem["id"],
            "problem": problem["problem"],
            "ground_truth": problem["answer"],
            "complexity": problem["complexity"],
            "category": problem["category"],
            "answer": result.final_answer,
            "correct": result.correct,
            "tokens": result.total_tokens,
            "num_reflections": num_reflections,
        })
        
        if result.correct:
            correct += 1
            print(f"  [OK] Correct (reflections: {num_reflections})")
        else:
            print(f"  [X] Wrong (reflections: {num_reflections}, expected: {problem['answer']})")
    
    pipeline.close()
    
    avg_reflections = total_reflections / len(problems) if problems else 0
    
    return {
        "method": "fixed_reflection",
        "num_problems": len(problems),
        "correct": correct,
        "accuracy": correct / len(problems) if problems else 0,
        "avg_reflections": avg_reflections,
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
        print(f"\n[{i+1}/{len(problems)}] {problem['problem'][:60]}...")
        
        result = pipeline.solve(
            problem=problem["problem"],
            problem_id=problem["id"],
            ground_truth=problem["answer"],
        )
        
        total_depth += result.actual_depth
        if result.rolled_back:
            rollback_count += 1
        if result.overfitting_detected:
            overfitting_count += 1
        
        results.append({
            "problem_id": problem["id"],
            "problem": problem["problem"],
            "ground_truth": problem["answer"],
            "complexity": problem["complexity"],
            "category": problem["category"],
            "answer": result.final_answer,
            "correct": result.correct,
            "tokens": result.total_tokens,
            "complexity_score": result.complexity_score,
            "recommended_depth": result.recommended_depth,
            "actual_depth": result.actual_depth,
            "rolled_back": result.rolled_back,
            "overfitting_detected": result.overfitting_detected,
        })
        
        if result.correct:
            correct += 1
            print(f"  [OK] Correct (depth: {result.actual_depth}, complexity: {result.complexity_score:.2f})")
        else:
            print(f"  [X] Wrong (depth: {result.actual_depth}, expected: {problem['answer']})")
    
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


def analyze_by_complexity(results: Dict) -> Dict:
    """Analyze accuracy by problem complexity."""
    by_complexity = {"low": [], "medium": [], "high": []}
    
    for r in results["results"]:
        complexity = r.get("complexity", "medium")
        if complexity in by_complexity:
            by_complexity[complexity].append(r["correct"])
    
    analysis = {}
    for level, correct_list in by_complexity.items():
        if correct_list:
            analysis[level] = {
                "total": len(correct_list),
                "correct": sum(correct_list),
                "accuracy": sum(correct_list) / len(correct_list),
            }
    
    return analysis


def statistical_analysis(baseline: Dict, fixed: Dict, adaptive: Dict) -> Dict:
    """Perform statistical analysis."""
    n = baseline["num_problems"]
    
    # McNemar's tests
    def mcnemar_test(r1_results, r2_results):
        b = sum(1 for a, b in zip(r1_results, r2_results) if a["correct"] and not b["correct"])
        c = sum(1 for a, b in zip(r1_results, r2_results) if not a["correct"] and b["correct"])
        if b + c > 0:
            stat = (abs(b - c) - 1) ** 2 / (b + c)
        else:
            stat = 0
        
        try:
            from scipy import stats
            p = 1 - stats.chi2.cdf(stat, 1) if stat > 0 else 1.0
        except ImportError:
            p = 0.5 * (1 + math.erf(-stat / math.sqrt(2))) if stat > 0 else 1.0
        
        return stat, p, b, c
    
    stat_bf, p_bf, b_bf, c_bf = mcnemar_test(baseline["results"], fixed["results"])
    stat_ba, p_ba, b_ba, c_ba = mcnemar_test(baseline["results"], adaptive["results"])
    stat_fa, p_fa, b_fa, c_fa = mcnemar_test(fixed["results"], adaptive["results"])
    
    return {
        "baseline_accuracy": baseline["accuracy"],
        "fixed_accuracy": fixed["accuracy"],
        "adaptive_accuracy": adaptive["accuracy"],
        "baseline_by_complexity": analyze_by_complexity(baseline),
        "fixed_by_complexity": analyze_by_complexity(fixed),
        "adaptive_by_complexity": analyze_by_complexity(adaptive),
        "fixed_improvement": fixed["accuracy"] - baseline["accuracy"],
        "adaptive_improvement": adaptive["accuracy"] - baseline["accuracy"],
        "adaptive_vs_fixed": adaptive["accuracy"] - fixed["accuracy"],
        "mcnemar_baseline_fixed": {"statistic": stat_bf, "p_value": p_bf, "baseline_wins": b_bf, "fixed_wins": c_bf},
        "mcnemar_baseline_adaptive": {"statistic": stat_ba, "p_value": p_ba, "baseline_wins": b_ba, "adaptive_wins": c_ba},
        "mcnemar_fixed_adaptive": {"statistic": stat_fa, "p_value": p_fa, "fixed_wins": b_fa, "adaptive_wins": c_fa},
        "avg_fixed_reflections": fixed.get("avg_reflections", 0),
        "avg_adaptive_depth": adaptive.get("avg_depth", 0),
        "rollback_count": adaptive.get("rollback_count", 0),
        "overfitting_count": adaptive.get("overfitting_count", 0),
    }


def main():
    """Run comprehensive benchmark."""
    print("="*60)
    print("COMPLEX REASONING BENCHMARK")
    print("Baseline vs Fixed vs Adaptive Self-Reflection")
    print("="*60)
    
    results_dir = Path(__file__).parent.parent / "data" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Load problems
    problems = load_complex_problems()
    print(f"\nLoaded {len(problems)} complex reasoning problems")
    
    # Show complexity distribution
    complexity_counts = {}
    for p in problems:
        c = p.get("complexity", "medium")
        complexity_counts[c] = complexity_counts.get(c, 0) + 1
    print(f"Complexity distribution: {complexity_counts}")
    
    # Run all methods
    baseline_results = run_baseline(problems)
    fixed_results = run_fixed_reflection(problems)
    adaptive_results = run_adaptive_reflection(problems)
    
    # Statistical analysis
    stats = statistical_analysis(baseline_results, fixed_results, adaptive_results)
    
    # Save results
    combined = {
        "timestamp": datetime.now().isoformat(),
        "dataset": "complex_reasoning",
        "num_problems": len(problems),
        "baseline": baseline_results,
        "fixed_reflection": fixed_results,
        "adaptive_reflection": adaptive_results,
        "statistical_analysis": stats,
    }
    
    results_path = results_dir / "complex_reasoning_benchmark.json"
    with open(results_path, "w") as f:
        json.dump(combined, f, indent=2, default=str)
    
    # Print summary
    print("\n" + "="*60)
    print("BENCHMARK RESULTS - COMPLEX REASONING")
    print("="*60)
    
    print(f"\nProblems tested: {len(problems)}")
    print(f"\n{'Method':<25} {'Accuracy':<12} {'Correct':<10} {'Improvement'}")
    print("-" * 60)
    print(f"{'Baseline':<25} {stats['baseline_accuracy']:>10.1%}   {baseline_results['correct']:>3}/{len(problems)}   {'-':>10}")
    print(f"{'Fixed (Depth=2)':<25} {stats['fixed_accuracy']:>10.1%}   {fixed_results['correct']:>3}/{len(problems)}   {stats['fixed_improvement']:>+9.1%}")
    print(f"{'Adaptive':<25} {stats['adaptive_accuracy']:>10.1%}   {adaptive_results['correct']:>3}/{len(problems)}   {stats['adaptive_improvement']:>+9.1%}")
    
    print(f"\n{'Efficiency Metrics':<25} {'Fixed':<15} {'Adaptive'}")
    print("-" * 60)
    print(f"{'Avg Reflections/Depth':<25} {stats['avg_fixed_reflections']:>14.2f} {stats['avg_adaptive_depth']:>10.2f}")
    print(f"{'Rollbacks':<25} {'N/A':>14} {stats['rollback_count']:>10}")
    print(f"{'Overfitting Detected':<25} {'N/A':>14} {stats['overfitting_count']:>10}")
    
    print(f"\n{'Statistical Significance':<40} {'p-value':<10} {'Significant'}")
    print("-" * 60)
    print(f"{'Baseline vs Fixed':<40} {stats['mcnemar_baseline_fixed']['p_value']:<10.4f} {'Yes' if stats['mcnemar_baseline_fixed']['p_value'] < 0.05 else 'No':>10}")
    print(f"{'Baseline vs Adaptive':<40} {stats['mcnemar_baseline_adaptive']['p_value']:<10.4f} {'Yes' if stats['mcnemar_baseline_adaptive']['p_value'] < 0.05 else 'No':>10}")
    print(f"{'Fixed vs Adaptive':<40} {stats['mcnemar_fixed_adaptive']['p_value']:<10.4f} {'Yes' if stats['mcnemar_fixed_adaptive']['p_value'] < 0.05 else 'No':>10}")
    
    # By complexity
    print(f"\n{'Accuracy by Complexity'}")
    print("-" * 60)
    for level in ["low", "medium", "high"]:
        if level in stats["baseline_by_complexity"]:
            b = stats["baseline_by_complexity"][level]["accuracy"]
            f = stats["fixed_by_complexity"][level]["accuracy"]
            a = stats["adaptive_by_complexity"][level]["accuracy"]
            n = stats["baseline_by_complexity"][level]["total"]
            print(f"{level.upper():<10} (n={n:>2}): Baseline {b:.0%} -> Fixed {f:.0%} -> Adaptive {a:.0%}")
    
    print(f"\nResults saved to: {results_path}")
    
    return stats


if __name__ == "__main__":
    main()
