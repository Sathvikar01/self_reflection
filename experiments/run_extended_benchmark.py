"""Run comprehensive benchmark on complex extended dataset.

This script evaluates all pipeline configurations on the more complex
extended dataset to test advanced reasoning capabilities.
"""

import os
import sys
import time
import json
import asyncio
from typing import List, Dict, Any
from dataclasses import dataclass, asdict
from pathlib import Path
from loguru import logger
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))

load_dotenv()


@dataclass
class BenchmarkResult:
    config_name: str
    problem_id: str
    question: str
    expected_answer: str
    model_answer: str
    correct: bool
    total_tokens: int
    latency_seconds: float
    num_reflections: int = 0
    num_expansions: int = 0
    num_backtracks: int = 0
    final_score: float = 0.0
    cache_hit_rate: float = 0.0
    complexity: str = "unknown"
    category: str = "unknown"
    error: str = ""


@dataclass
class AggregatedMetrics:
    config_name: str
    total_problems: int
    correct: int
    accuracy: float
    total_tokens: int
    avg_tokens_per_problem: float
    total_latency_seconds: float
    avg_latency_seconds: float
    total_reflections: int
    avg_reflections: float
    total_expansions: int
    avg_expansions: float
    total_backtracks: int
    avg_backtracks: float
    avg_final_score: float
    avg_cache_hit_rate: float
    efficiency: float
    accuracy_by_complexity: Dict[str, float]
    accuracy_by_category: Dict[str, float]


def load_complex_dataset(path: str = "data/datasets/complex_extended.json") -> List[Dict]:
    """Load the complex extended dataset."""
    with open(path, 'r') as f:
        return json.load(f)


def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison."""
    if answer is None:
        return ""
    answer = str(answer).lower().strip()
    answer = answer.replace("$", "").replace(",", "").replace("%", "")
    answer = answer.replace("approximately", "").replace("about", "").strip()
    return answer


def check_answer(model_answer: str, expected: str) -> bool:
    """Check if model answer matches expected."""
    model_norm = normalize_answer(model_answer)
    expected_norm = normalize_answer(expected)
    
    if model_norm == expected_norm:
        return True
    
    if expected_norm in model_norm:
        return True
    
    if model_norm in expected_norm:
        return True
    
    try:
        model_num = float(model_norm.split()[0])
        expected_num = float(expected_norm.split()[0])
        if abs(model_num - expected_num) / max(abs(expected_num), 1e-10) < 0.05:
            return True
    except (ValueError, IndexError):
        pass
    
    yes_variants = ["yes", "true", "correct", "1", "yeah", "yep"]
    no_variants = ["no", "false", "incorrect", "0", "nope", "wrong"]
    
    if expected_norm in yes_variants and any(v in model_norm for v in yes_variants):
        return True
    if expected_norm in no_variants and any(v in model_norm for v in no_variants):
        return True
    
    return False


async def run_baseline_benchmark(
    problems: List[Dict],
    api_key: str,
) -> List[BenchmarkResult]:
    """Run baseline zero-shot benchmark."""
    from src.orchestration.baseline import BaselineRunner, BaselineConfig
    
    logger.info(f"Running Baseline benchmark on {len(problems)} problems...")
    
    config = BaselineConfig(
        temperature=0.7,
        max_tokens=2048,
    )
    
    runner = BaselineRunner(api_key=api_key, config=config)
    results = []
    
    for i, problem in enumerate(problems):
        problem_id = problem.get("id", f"problem_{i}")
        question = problem.get("question", "")
        expected = problem.get("answer", "")
        complexity = problem.get("complexity", "unknown")
        category = problem.get("category", "unknown")
        
        logger.info(f"[{i+1}/{len(problems)}] Baseline: {problem_id}")
        
        try:
            start_time = time.time()
            result = runner.solve(
                problem=question,
                problem_id=problem_id,
            )
            latency = time.time() - start_time
            
            model_answer = result.answer if hasattr(result, 'answer') else str(result)
            correct = check_answer(model_answer, expected)
            
            results.append(BenchmarkResult(
                config_name="Baseline",
                problem_id=problem_id,
                question=question[:100] + "..." if len(question) > 100 else question,
                expected_answer=expected,
                model_answer=model_answer[:200] if model_answer else "",
                correct=correct,
                total_tokens=result.total_tokens if hasattr(result, 'total_tokens') else 0,
                latency_seconds=latency,
                complexity=complexity,
                category=category,
            ))
        except Exception as e:
            logger.error(f"Baseline error for {problem_id}: {e}")
            results.append(BenchmarkResult(
                config_name="Baseline",
                problem_id=problem_id,
                question=question[:100],
                expected_answer=expected,
                model_answer="",
                correct=False,
                total_tokens=0,
                latency_seconds=0,
                complexity=complexity,
                category=category,
                error=str(e),
            ))
    
    runner.close()
    return results


async def run_self_reflection_benchmark(
    problems: List[Dict],
    api_key: str,
) -> List[BenchmarkResult]:
    """Run self-reflection pipeline benchmark."""
    from src.orchestration.self_reflection_pipeline import (
        SelfReflectionPipeline,
        SelfReflectionConfig,
    )
    
    logger.info(f"Running Self-Reflection benchmark on {len(problems)} problems...")
    
    config = SelfReflectionConfig(
        max_iterations=8,
        min_reasoning_steps=2,
        max_reasoning_steps=4,
        reflection_depth=2,
        enable_selective_reflection=True,
    )
    
    pipeline = SelfReflectionPipeline(api_key=api_key, config=config)
    results = []
    
    for i, problem in enumerate(problems):
        problem_id = problem.get("id", f"problem_{i}")
        question = problem.get("question", "")
        expected = problem.get("answer", "")
        complexity = problem.get("complexity", "unknown")
        category = problem.get("category", "unknown")
        
        logger.info(f"[{i+1}/{len(problems)}] Self-Reflection: {problem_id}")
        
        try:
            start_time = time.time()
            result = pipeline.solve(
                problem=question,
                problem_id=problem_id,
            )
            latency = time.time() - start_time
            
            model_answer = result.answer if hasattr(result, 'answer') else str(result)
            correct = check_answer(model_answer, expected)
            
            results.append(BenchmarkResult(
                config_name="Self-Reflection",
                problem_id=problem_id,
                question=question[:100] + "..." if len(question) > 100 else question,
                expected_answer=expected,
                model_answer=model_answer[:200] if model_answer else "",
                correct=correct,
                total_tokens=result.total_tokens if hasattr(result, 'total_tokens') else 0,
                latency_seconds=latency,
                num_reflections=len(result.reflections) if hasattr(result, 'reflections') and result.reflections else 0,
                complexity=complexity,
                category=category,
            ))
        except Exception as e:
            logger.error(f"Self-Reflection error for {problem_id}: {e}")
            results.append(BenchmarkResult(
                config_name="Self-Reflection",
                problem_id=problem_id,
                question=question[:100],
                expected_answer=expected,
                model_answer="",
                correct=False,
                total_tokens=0,
                latency_seconds=0,
                complexity=complexity,
                category=category,
                error=str(e),
            ))
    
    pipeline.close()
    return results


async def run_rl_guided_benchmark(
    problems: List[Dict],
    api_key: str,
) -> List[BenchmarkResult]:
    """Run RL-guided MCTS benchmark."""
    from src.orchestration.pipeline import RLPipeline, PipelineConfig
    from src.rl_controller.mcts import MCTSConfig
    
    logger.info(f"Running RL-Guided benchmark on {len(problems)} problems...")
    
    mcts_config = MCTSConfig(
        exploration_constant=1.414,
        expansion_budget=30,
        use_value_network=False,
    )
    
    config = PipelineConfig(
        max_iterations=30,
        early_stop_score=0.85,
        mcts=mcts_config,
    )
    
    pipeline = RLPipeline(api_key=api_key, config=config)
    results = []
    
    for i, problem in enumerate(problems):
        problem_id = problem.get("id", f"problem_{i}")
        question = problem.get("question", "")
        expected = problem.get("answer", "")
        complexity = problem.get("complexity", "unknown")
        category = problem.get("category", "unknown")
        
        logger.info(f"[{i+1}/{len(problems)}] RL-Guided: {problem_id}")
        
        try:
            start_time = time.time()
            result = pipeline.solve(
                problem=question,
                problem_id=problem_id,
            )
            latency = time.time() - start_time
            
            model_answer = result.answer if hasattr(result, 'answer') else str(result)
            correct = check_answer(model_answer, expected)
            
            action_stats = pipeline.action_executor.get_stats() if hasattr(pipeline, 'action_executor') else {}
            cache_hit_rate = action_stats.get("cache_stats", {}).get("hit_rate", 0)
            
            results.append(BenchmarkResult(
                config_name="RL-Guided",
                problem_id=problem_id,
                question=question[:100] + "..." if len(question) > 100 else question,
                expected_answer=expected,
                model_answer=model_answer[:200] if model_answer else "",
                correct=correct,
                total_tokens=result.total_tokens if hasattr(result, 'total_tokens') else 0,
                latency_seconds=latency,
                num_reflections=result.num_reflections if hasattr(result, 'num_reflections') else 0,
                num_expansions=result.num_expansions if hasattr(result, 'num_expansions') else 0,
                num_backtracks=result.num_backtracks if hasattr(result, 'num_backtracks') else 0,
                final_score=result.final_score if hasattr(result, 'final_score') else 0.0,
                cache_hit_rate=cache_hit_rate,
                complexity=complexity,
                category=category,
            ))
        except Exception as e:
            logger.error(f"RL-Guided error for {problem_id}: {e}")
            results.append(BenchmarkResult(
                config_name="RL-Guided",
                problem_id=problem_id,
                question=question[:100],
                expected_answer=expected,
                model_answer="",
                correct=False,
                total_tokens=0,
                latency_seconds=0,
                complexity=complexity,
                category=category,
                error=str(e),
            ))
    
    pipeline.close()
    return results


def aggregate_results(
    results: List[BenchmarkResult],
) -> AggregatedMetrics:
    """Aggregate benchmark results with detailed breakdowns."""
    if not results:
        return AggregatedMetrics(
            config_name="Unknown",
            total_problems=0,
            correct=0,
            accuracy=0,
            total_tokens=0,
            avg_tokens_per_problem=0,
            total_latency_seconds=0,
            avg_latency_seconds=0,
            total_reflections=0,
            avg_reflections=0,
            total_expansions=0,
            avg_expansions=0,
            total_backtracks=0,
            avg_backtracks=0,
            avg_final_score=0,
            avg_cache_hit_rate=0,
            efficiency=0,
            accuracy_by_complexity={},
            accuracy_by_category={},
        )
    
    config_name = results[0].config_name
    total = len(results)
    correct = sum(1 for r in results if r.correct)
    
    total_tokens = sum(r.total_tokens for r in results)
    total_latency = sum(r.latency_seconds for r in results)
    total_reflections = sum(r.num_reflections for r in results)
    total_expansions = sum(r.num_expansions for r in results)
    total_backtracks = sum(r.num_backtracks for r in results)
    
    avg_score = sum(r.final_score for r in results) / total
    avg_cache = sum(r.cache_hit_rate for r in results) / total
    
    accuracy = correct / total if total > 0 else 0
    avg_tokens = total_tokens / total if total > 0 else 0
    efficiency = accuracy / avg_tokens if avg_tokens > 0 else 0
    
    accuracy_by_complexity = {}
    complexity_groups = {}
    for r in results:
        c = r.complexity
        if c not in complexity_groups:
            complexity_groups[c] = []
        complexity_groups[c].append(r)
    
    for complexity, group_results in complexity_groups.items():
        group_correct = sum(1 for r in group_results if r.correct)
        accuracy_by_complexity[complexity] = group_correct / len(group_results)
    
    accuracy_by_category = {}
    category_groups = {}
    for r in results:
        c = r.category
        if c not in category_groups:
            category_groups[c] = []
        category_groups[c].append(r)
    
    for category, group_results in category_groups.items():
        group_correct = sum(1 for r in group_results if r.correct)
        accuracy_by_category[category] = group_correct / len(group_results)
    
    return AggregatedMetrics(
        config_name=config_name,
        total_problems=total,
        correct=correct,
        accuracy=accuracy,
        total_tokens=total_tokens,
        avg_tokens_per_problem=avg_tokens,
        total_latency_seconds=total_latency,
        avg_latency_seconds=total_latency / total if total > 0 else 0,
        total_reflections=total_reflections,
        avg_reflections=total_reflections / total if total > 0 else 0,
        total_expansions=total_expansions,
        avg_expansions=total_expansions / total if total > 0 else 0,
        total_backtracks=total_backtracks,
        avg_backtracks=total_backtracks / total if total > 0 else 0,
        avg_final_score=avg_score,
        avg_cache_hit_rate=avg_cache,
        efficiency=efficiency,
        accuracy_by_complexity=accuracy_by_complexity,
        accuracy_by_category=accuracy_by_category,
    )


def print_comparison_table(metrics_list: List[AggregatedMetrics]):
    """Print comprehensive comparison table."""
    
    print("\n" + "=" * 140)
    print("COMPLEX EXTENDED DATASET BENCHMARK RESULTS")
    print("=" * 140)
    
    print("\n📊 ACCURACY & PERFORMANCE METRICS")
    print("-" * 140)
    header = f"{'Configuration':<20} {'Accuracy':>10} {'Correct':>8} {'Total':>6} {'Avg Tokens':>12} {'Avg Time':>12} {'Efficiency':>12}"
    print(header)
    print("-" * 140)
    
    for m in metrics_list:
        row = f"{m.config_name:<20} {m.accuracy:>9.1%} {m.correct:>8} {m.total_problems:>6} {m.avg_tokens_per_problem:>12.0f} {m.avg_latency_seconds:>11.2f}s {m.efficiency:>12.4f}"
        print(row)
    
    print("\n📋 REASONING BEHAVIOR METRICS")
    print("-" * 140)
    header = f"{'Configuration':<20} {'Avg Reflect':>12} {'Avg Expand':>12} {'Avg Backtrack':>14} {'Avg Score':>10} {'Cache Hit':>10}"
    print(header)
    print("-" * 140)
    
    for m in metrics_list:
        row = f"{m.config_name:<20} {m.avg_reflections:>12.1f} {m.avg_expansions:>12.1f} {m.avg_backtracks:>14.1f} {m.avg_final_score:>10.2f} {m.avg_cache_hit_rate:>9.1%}"
        print(row)
    
    print("\n📈 ACCURACY BY COMPLEXITY")
    print("-" * 140)
    all_complexities = set()
    for m in metrics_list:
        all_complexities.update(m.accuracy_by_complexity.keys())
    
    complexities = sorted(all_complexities, key=lambda x: {'very_high': 4, 'high': 3, 'medium': 2, 'low': 1}.get(x, 0), reverse=True)
    
    header = f"{'Configuration':<20}"
    for c in complexities[:4]:
        header += f"{c:>15}"
    print(header)
    print("-" * 140)
    
    for m in metrics_list:
        row = f"{m.config_name:<20}"
        for c in complexities[:4]:
            acc = m.accuracy_by_complexity.get(c, 0)
            row += f"{acc:>14.1%}"
        print(row)
    
    print("\n🎯 TOP CATEGORIES BY ACCURACY")
    print("-" * 140)
    
    for m in metrics_list:
        sorted_categories = sorted(
            m.accuracy_by_category.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        print(f"\n{m.config_name}:")
        for category, acc in sorted_categories:
            print(f"  {category:<35} {acc:>6.1%}")
    
    print("\n💰 COST ANALYSIS")
    print("-" * 140)
    header = f"{'Configuration':<20} {'Total Tokens':>12} {'Est. Cost':>12} {'Cost/Problem':>14} {'Cost Reduction':>15}"
    print(header)
    print("-" * 140)
    
    baseline_tokens = metrics_list[0].avg_tokens_per_problem if metrics_list else 1
    
    for m in metrics_list:
        total_cost = m.total_tokens * 0.00015
        cost_per_problem = m.avg_tokens_per_problem * 0.00015
        cost_reduction = 1 - (m.avg_tokens_per_problem / baseline_tokens) if baseline_tokens > 0 else 0
        
        row = f"{m.config_name:<20} {m.total_tokens:>12} ${total_cost:>11.2f} ${cost_per_problem:>13.4f} {cost_reduction:>14.1%}"
        print(row)
    
    print("\n🏆 SUMMARY")
    print("-" * 140)
    
    best_accuracy = max(metrics_list, key=lambda x: x.accuracy)
    fastest = min(metrics_list, key=lambda x: x.avg_latency_seconds)
    most_efficient = max(metrics_list, key=lambda x: x.efficiency)
    
    print(f"Best Accuracy:     {best_accuracy.config_name:<20} ({best_accuracy.accuracy:.1%})")
    print(f"Fastest:           {fastest.config_name:<20} ({fastest.avg_latency_seconds:.2f}s)")
    print(f"Most Efficient:    {most_efficient.config_name:<20} (eff={most_efficient.efficiency:.4f})")
    
    print("\n" + "=" * 140)


def save_results(metrics_list: List[AggregatedMetrics], output_path: str):
    """Save results to JSON file."""
    data = {
        "timestamp": time.time(),
        "results": [asdict(m) for m in metrics_list],
    }
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    
    logger.info(f"Results saved to {output_path}")


async def run_comprehensive_benchmark(
    dataset_path: str = "data/datasets/complex_extended.json",
    n_problems: int = 40,
    output_dir: str = "benchmark_results",
):
    """Run comprehensive benchmark across all configurations."""
    
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        logger.error("NVIDIA_API_KEY not set")
        logger.info("Please set NVIDIA_API_KEY in .env file")
        return None
    
    problems = load_complex_dataset(dataset_path)
    problems = problems[:n_problems]
    
    logger.info(f"Loaded {len(problems)} complex problems")
    logger.info(f"Complexity distribution:")
    
    complexity_counts = {}
    for p in problems:
        c = p.get("complexity", "unknown")
        complexity_counts[c] = complexity_counts.get(c, 0) + 1
    
    for c, count in sorted(complexity_counts.items()):
        logger.info(f"  {c}: {count} problems")
    
    all_results = []
    
    baseline_results = await run_baseline_benchmark(problems, api_key)
    all_results.extend(baseline_results)
    
    self_reflect_results = await run_self_reflection_benchmark(problems, api_key)
    all_results.extend(self_reflect_results)
    
    rl_results = await run_rl_guided_benchmark(problems, api_key)
    all_results.extend(rl_results)
    
    configs = {}
    for result in all_results:
        if result.config_name not in configs:
            configs[result.config_name] = []
        configs[result.config_name].append(result)
    
    metrics_list = [aggregate_results(results) for results in configs.values()]
    
    print_comparison_table(metrics_list)
    
    output_path = f"{output_dir}/complex_benchmark_{int(time.time())}.json"
    save_results(metrics_list, output_path)
    
    return metrics_list


if __name__ == "__main__":
    asyncio.run(run_comprehensive_benchmark())
