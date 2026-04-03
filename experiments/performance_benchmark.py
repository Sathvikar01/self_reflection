"""Performance benchmarks for async vs sync processing."""

import asyncio
import time
from typing import List
from loguru import logger
from data.datasets.loader import Problem


class PerformanceBenchmark:
    """Benchmark async vs sync processing."""

    @staticmethod
    async def benchmark_async_batch(
        client,
        problems: List[Problem],
        batch_size: int = 10
    ) -> dict:
        """Benchmark async batch processing.

        Args:
            client: Async client instance
            problems: List of problems to process
            batch_size: Number of concurrent requests

        Returns:
            Benchmark results
        """
        from src.generator.async_nim_client import GenerationConfig

        start_time = time.time()

        config = GenerationConfig()
        requests = [
            ([{"role": "user", "content": p.question}], config)
            for p in problems
        ]

        results = await client.generate_batch(requests, max_concurrent=batch_size)

        elapsed = time.time() - start_time

        return {
            "total_problems": len(problems),
            "batch_size": batch_size,
            "total_time_seconds": elapsed,
            "avg_time_per_problem": elapsed / len(problems) if problems else 0,
            "throughput_problems_per_second": len(problems) / elapsed if elapsed > 0 else 0,
            "results_count": len(results)
        }

    @staticmethod
    def benchmark_sync_sequential(
        client,
        problems: List[Problem]
    ) -> dict:
        """Benchmark sync sequential processing.

        Args:
            client: Sync client instance
            problems: List of problems to process

        Returns:
            Benchmark results
        """
        from src.generator.nim_client import GenerationConfig

        start_time = time.time()

        config = GenerationConfig()
        results = []

        for problem in problems:
            messages = [{"role": "user", "content": problem.question}]
            result = client.generate(messages, config)
            results.append(result)

        elapsed = time.time() - start_time

        return {
            "total_problems": len(problems),
            "total_time_seconds": elapsed,
            "avg_time_per_problem": elapsed / len(problems) if problems else 0,
            "throughput_problems_per_second": len(problems) / elapsed if elapsed > 0 else 0,
            "results_count": len(results)
        }

    @staticmethod
    def compare_benchmarks(async_results: dict, sync_results: dict) -> dict:
        """Compare async vs sync benchmark results.

        Args:
            async_results: Results from async benchmark
            sync_results: Results from sync benchmark

        Returns:
            Comparison metrics
        """
        speedup = (
            sync_results["total_time_seconds"] / async_results["total_time_seconds"]
            if async_results["total_time_seconds"] > 0 else 0
        )

        throughput_improvement = (
            (async_results["throughput_problems_per_second"] - sync_results["throughput_problems_per_second"])
            / sync_results["throughput_problems_per_second"] * 100
            if sync_results["throughput_problems_per_second"] > 0 else 0
        )

        return {
            "speedup_factor": speedup,
            "throughput_improvement_percent": throughput_improvement,
            "async_avg_time": async_results["avg_time_per_problem"],
            "sync_avg_time": sync_results["avg_time_per_problem"],
            "async_throughput": async_results["throughput_problems_per_second"],
            "sync_throughput": sync_results["throughput_problems_per_second"],
            "time_saved_seconds": sync_results["total_time_seconds"] - async_results["total_time_seconds"]
        }


async def run_performance_comparison(
    problems: List[Problem],
    batch_sizes: List[int] = [5, 10, 20]
) -> dict:
    """Run comprehensive performance comparison.

    Args:
        problems: Problems to test with
        batch_sizes: List of batch sizes to test

    Returns:
        Comprehensive benchmark results
    """
    from src.generator.async_nim_client import AsyncNVIDIANIMClient
    from src.generator.nim_client import NVIDIANIMClient
    import os

    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        logger.warning("No API key found, using mock benchmark")
        return {"error": "No API key"}

    results = {
        "async": {},
        "sync": {},
        "comparisons": {}
    }

    # Benchmark async with different batch sizes
    async with AsyncNVIDIANIMClient(api_key=api_key) as async_client:
        for batch_size in batch_sizes:
            logger.info(f"Benchmarking async with batch_size={batch_size}")
            results["async"][batch_size] = await PerformanceBenchmark.benchmark_async_batch(
                async_client,
                problems,
                batch_size=batch_size
            )

    # Benchmark sync
    sync_client = NVIDIANIMClient(api_key=api_key)
    logger.info("Benchmarking sync sequential")
    results["sync"] = PerformanceBenchmark.benchmark_sync_sequential(sync_client, problems)

    # Compare results
    for batch_size in batch_sizes:
        results["comparisons"][batch_size] = PerformanceBenchmark.compare_benchmarks(
            results["async"][batch_size],
            results["sync"]
        )

    # Find best batch size
    best_batch_size = max(
        batch_sizes,
        key=lambda bs: results["comparisons"][bs]["speedup_factor"]
    )

    results["best_batch_size"] = best_batch_size
    results["best_speedup"] = results["comparisons"][best_batch_size]["speedup_factor"]

    logger.info(f"Best batch size: {best_batch_size} with {results['best_speedup']:.2f}x speedup")

    return results


def print_benchmark_report(results: dict):
    """Print formatted benchmark report.

    Args:
        results: Benchmark results from run_performance_comparison
    """
    print("\n" + "="*60)
    print("PERFORMANCE BENCHMARK REPORT")
    print("="*60)

    if "error" in results:
        print(f"\nError: {results['error']}")
        return

    # Sync results
    print("\n--- SYNC (Sequential) ---")
    print(f"Total problems: {results['sync']['total_problems']}")
    print(f"Total time: {results['sync']['total_time_seconds']:.2f}s")
    print(f"Avg time/problem: {results['sync']['avg_time_per_problem']:.2f}s")
    print(f"Throughput: {results['sync']['throughput_problems_per_second']:.2f} problems/s")

    # Async results
    for batch_size, async_result in results["async"].items():
        print(f"\n--- ASYNC (batch_size={batch_size}) ---")
        print(f"Total problems: {async_result['total_problems']}")
        print(f"Total time: {async_result['total_time_seconds']:.2f}s")
        print(f"Avg time/problem: {async_result['avg_time_per_problem']:.2f}s")
        print(f"Throughput: {async_result['throughput_problems_per_second']:.2f} problems/s")

    # Comparisons
    for batch_size, comparison in results["comparisons"].items():
        print(f"\n--- COMPARISON (batch_size={batch_size}) ---")
        print(f"Speedup: {comparison['speedup_factor']:.2f}x")
        print(f"Throughput improvement: {comparison['throughput_improvement_percent']:.1f}%")
        print(f"Time saved: {comparison['time_saved_seconds']:.2f}s")

    # Best configuration
    print("\n" + "="*60)
    print(f"BEST CONFIGURATION: batch_size={results['best_batch_size']}")
    print(f"MAX SPEEDUP: {results['best_speedup']:.2f}x")
    print("="*60 + "\n")


if __name__ == "__main__":
    # Example usage
    from data.datasets.loader import DataLoader

    # Load sample problems
    loader = DataLoader()
    problems = loader.load("strategy_qa", split="test", n=20)

    # Run benchmark
    results = asyncio.run(run_performance_comparison(problems))

    # Print report
    print_benchmark_report(results)
