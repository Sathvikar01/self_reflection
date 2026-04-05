"""Comprehensive benchmark comparing all pipeline configurations.

This script runs benchmarks for:
1. Baseline (Zero-shot)
2. Self-Reflection Pipeline
3. RL-Guided MCTS Pipeline
4. Adaptive Self-Reflection Pipeline

And outputs a detailed comparison table.
"""

import os
import time
import json
import asyncio
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from loguru import logger
from dotenv import load_dotenv

load_dotenv()


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""
    config_name: str
    problem_id: str
    correct: bool
    total_tokens: int
    latency_seconds: float
    num_reflections: int = 0
    num_expansions: int = 0
    num_backtracks: int = 0
    final_score: float = 0.0
    cache_hit_rate: float = 0.0
    error: str = ""


@dataclass
class AggregatedMetrics:
    """Aggregated metrics for a configuration."""
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
    efficiency: float  # accuracy / avg_tokens


async def run_baseline_benchmark(
    problems: List[Dict],
    api_key: str,
) -> List[BenchmarkResult]:
    """Run baseline zero-shot benchmark."""
    from src.orchestration.baseline import BaselineRunner, BaselineConfig
    
    logger.info("Running Baseline benchmark...")
    
    config = BaselineConfig(
        temperature=0.7,
        max_tokens=2048,
    )
    
    runner = BaselineRunner(api_key=api_key, config=config)
    results = []
    
    for i, problem in enumerate(problems[:5]):  # Limit to 5 for speed
        problem_id = problem.get("id", f"problem_{i}")
        question = problem.get("problem", problem.get("question", ""))
        ground_truth = problem.get("answer", problem.get("ground_truth"))
        
        try:
            start_time = time.time()
            result = runner.solve(
                problem=question,
                problem_id=problem_id,
                ground_truth=ground_truth,
            )
            latency = time.time() - start_time
            
            results.append(BenchmarkResult(
                config_name="Baseline",
                problem_id=problem_id,
                correct=result.correct or False,
                total_tokens=result.total_tokens,
                latency_seconds=latency,
            ))
        except Exception as e:
            logger.error(f"Baseline error for {problem_id}: {e}")
            results.append(BenchmarkResult(
                config_name="Baseline",
                problem_id=problem_id,
                correct=False,
                total_tokens=0,
                latency_seconds=0,
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
    
    logger.info("Running Self-Reflection benchmark...")
    
    config = SelfReflectionConfig(
        max_iterations=8,
        min_reasoning_steps=2,
        max_reasoning_steps=4,
        reflection_depth=2,
        enable_selective_reflection=True,
    )
    
    pipeline = SelfReflectionPipeline(api_key=api_key, config=config)
    results = []
    
    for i, problem in enumerate(problems[:5]):
        problem_id = problem.get("id", f"problem_{i}")
        question = problem.get("problem", problem.get("question", ""))
        ground_truth = problem.get("answer", problem.get("ground_truth"))
        
        try:
            start_time = time.time()
            result = pipeline.solve(
                problem=question,
                problem_id=problem_id,
                ground_truth=ground_truth,
            )
            latency = time.time() - start_time
            
            results.append(BenchmarkResult(
                config_name="Self-Reflection",
                problem_id=problem_id,
                correct=result.correct or False,
                total_tokens=result.total_tokens,
                latency_seconds=latency,
                num_reflections=len(result.reflections) if hasattr(result, 'reflections') else 0,
            ))
        except Exception as e:
            logger.error(f"Self-Reflection error for {problem_id}: {e}")
            results.append(BenchmarkResult(
                config_name="Self-Reflection",
                problem_id=problem_id,
                correct=False,
                total_tokens=0,
                latency_seconds=0,
                error=str(e),
            ))
    
    pipeline.close()
    return results


async def run_rl_guided_benchmark(
    problems: List[Dict],
    api_key: str,
    use_value_network: bool = False,
) -> List[BenchmarkResult]:
    """Run RL-guided MCTS benchmark."""
    from src.orchestration.pipeline import RLPipeline, PipelineConfig
    from src.rl_controller.mcts import MCTSConfig
    
    logger.info(f"Running RL-Guided benchmark (value_network={use_value_network})...")
    
    mcts_config = MCTSConfig(
        exploration_constant=1.414,
        expansion_budget=30,
        use_value_network=use_value_network,
    )
    
    config = PipelineConfig(
        max_iterations=30,
        early_stop_score=0.85,
        mcts=mcts_config,
    )
    
    pipeline = RLPipeline(api_key=api_key, config=config)
    results = []
    
    for i, problem in enumerate(problems[:5]):
        problem_id = problem.get("id", f"problem_{i}")
        question = problem.get("problem", problem.get("question", ""))
        ground_truth = problem.get("answer", problem.get("ground_truth"))
        
        try:
            start_time = time.time()
            result = pipeline.solve(
                problem=question,
                problem_id=problem_id,
                ground_truth=ground_truth,
            )
            latency = time.time() - start_time
            
            # Get cache stats
            action_stats = pipeline.action_executor.get_stats()
            cache_hit_rate = action_stats.get("cache_stats", {}).get("hit_rate", 0)
            
            results.append(BenchmarkResult(
                config_name="RL-Guided" + ("+VN" if use_value_network else ""),
                problem_id=problem_id,
                correct=result.correct or False,
                total_tokens=result.total_tokens,
                latency_seconds=latency,
                num_reflections=result.num_reflections,
                num_expansions=result.num_expansions,
                num_backtracks=result.num_backtracks,
                final_score=result.final_score,
                cache_hit_rate=cache_hit_rate,
            ))
        except Exception as e:
            logger.error(f"RL-Guided error for {problem_id}: {e}")
            results.append(BenchmarkResult(
                config_name="RL-Guided" + ("+VN" if use_value_network else ""),
                problem_id=problem_id,
                correct=False,
                total_tokens=0,
                latency_seconds=0,
                error=str(e),
            ))
    
    pipeline.close()
    return results


def aggregate_results(
    results: List[BenchmarkResult],
) -> AggregatedMetrics:
    """Aggregate benchmark results."""
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
    )


def print_comparison_table(metrics: List[AggregatedMetrics]):
    """Print comprehensive comparison table."""
    
    print("\n" + "=" * 120)
    print("COMPREHENSIVE PIPELINE BENCHMARK COMPARISON")
    print("=" * 120)
    
    # Main metrics table
    print("\n📊 ACCURACY & PERFORMANCE METRICS")
    print("-" * 120)
    header = f"{'Configuration':<20} {'Accuracy':>10} {'Correct':>8} {'Total':>6} {'Avg Tokens':>12} {'Avg Time':>10} {'Efficiency':>12}"
    print(header)
    print("-" * 120)
    
    for m in metrics:
        row = f"{m.config_name:<20} {m.accuracy:>9.1%} {m.correct:>8} {m.total_problems:>6} {m.avg_tokens_per_problem:>12.0f} {m.avg_latency_seconds:>9.2f}s {m.efficiency:>12.4f}"
        print(row)
    
    # Reasoning metrics table
    print("\n📋 REASONING BEHAVIOR METRICS")
    print("-" * 120)
    header = f"{'Configuration':<20} {'Avg Reflect':>12} {'Avg Expand':>12} {'Avg Backtrack':>14} {'Avg Score':>10} {'Cache Hit':>10}"
    print(header)
    print("-" * 120)
    
    for m in metrics:
        row = f"{m.config_name:<20} {m.avg_reflections:>12.1f} {m.avg_expansions:>12.1f} {m.avg_backtracks:>14.1f} {m.avg_final_score:>10.2f} {m.avg_cache_hit_rate:>9.1%}"
        print(row)
    
    # Token breakdown
    print("\n💰 COST ANALYSIS")
    print("-" * 120)
    header = f"{'Configuration':<20} {'Total Tokens':>12} {'Est. Cost':>12} {'Cost/Problem':>14} {'Cost Reduction':>15}"
    print(header)
    print("-" * 120)
    
    baseline_tokens = metrics[0].avg_tokens_per_problem if metrics else 1
    
    for m in metrics:
        total_cost = m.total_tokens * 0.0001  # $0.0001 per token (example)
        cost_per_problem = m.avg_tokens_per_problem * 0.0001
        cost_reduction = 1 - (m.avg_tokens_per_problem / baseline_tokens) if baseline_tokens > 0 else 0
        
        row = f"{m.config_name:<20} {m.total_tokens:>12} ${total_cost:>11.2f} ${cost_per_problem:>13.4f} {cost_reduction:>14.1%}"
        print(row)
    
    # Summary
    print("\n📈 SUMMARY")
    print("-" * 120)
    
    best_accuracy = max(metrics, key=lambda x: x.accuracy)
    fastest = min(metrics, key=lambda x: x.avg_latency_seconds)
    most_efficient = max(metrics, key=lambda x: x.efficiency)
    
    print(f"Best Accuracy:     {best_accuracy.config_name} ({best_accuracy.accuracy:.1%})")
    print(f"Fastest:           {fastest.config_name} ({fastest.avg_latency_seconds:.2f}s)")
    print(f"Most Efficient:    {most_efficient.config_name} (eff={most_efficient.efficiency:.4f})")
    
    print("\n" + "=" * 120)


def save_results(metrics: List[AggregatedMetrics], output_path: str):
    """Save results to JSON file."""
    data = {
        "timestamp": time.time(),
        "results": [asdict(m) for m in metrics],
    }
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    
    logger.info(f"Results saved to {output_path}")


async def run_comprehensive_benchmark(
    dataset: str = "strategy_qa",
    n_problems: int = 20,
    output_dir: str = "benchmark_results",
):
    """Run comprehensive benchmark across all configurations."""
    
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        logger.error("NVIDIA_API_KEY not set")
        return
    
    # Load problems
    from data.datasets.loader import DataLoader
    
    loader = DataLoader()
    problems = loader.load(dataset, split="test", n=n_problems)
    
    logger.info(f"Loaded {len(problems)} problems from {dataset}")
    
    # Run benchmarks
    all_results = []
    
    # 1. Baseline
    baseline_results = await run_baseline_benchmark(problems, api_key)
    all_results.extend(baseline_results)
    
    # 2. Self-Reflection
    self_reflect_results = await run_self_reflection_benchmark(problems, api_key)
    all_results.extend(self_reflect_results)
    
    # 3. RL-Guided
    rl_results = await run_rl_guided_benchmark(problems, api_key, use_value_network=False)
    all_results.extend(rl_results)
    
    # 4. RL-Guided + Value Network (if available)
    # rl_vn_results = await run_rl_guided_benchmark(problems, api_key, use_value_network=True)
    # all_results.extend(rl_vn_results)
    
    # Aggregate by configuration
    configs = {}
    for result in all_results:
        if result.config_name not in configs:
            configs[result.config_name] = []
        configs[result.config_name].append(result)
    
    metrics = [aggregate_results(results) for results in configs.values()]
    
    # Print comparison
    print_comparison_table(metrics)
    
    # Save results
    output_path = f"{output_dir}/benchmark_{int(time.time())}.json"
    save_results(metrics, output_path)
    
    return metrics


if __name__ == "__main__":
    asyncio.run(run_comprehensive_benchmark())
