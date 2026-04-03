"""Async batch processing pipeline for self-reflection."""

import os
import json
import time
import asyncio
from typing import List, Dict, Optional, Any, Callable
from dataclasses import dataclass, field, asdict
from pathlib import Path
from datetime import datetime
from loguru import logger
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from generator.async_nim_client import AsyncNVIDIANIMClient, GenerationConfig
from orchestration.self_reflection_pipeline import (
    SelfReflectionConfig,
    SelfReflectionResult
)
from data.datasets.loader import Problem


@dataclass
class BatchConfig:
    """Configuration for batch processing."""
    max_concurrent: int = 10
    checkpoint_interval: int = 10
    save_intermediate: bool = True
    error_handling: str = "continue"  # "continue", "stop", "raise"
    retry_failed: bool = True
    max_retries: int = 2


@dataclass
class BatchResult:
    """Result from batch processing."""
    total_problems: int
    successful: int
    failed: int
    results: List[SelfReflectionResult]
    errors: List[Dict[str, Any]]
    total_tokens: int
    total_latency_seconds: float
    avg_latency_per_problem: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class AsyncBatchPipeline:
    """Async batch processing for self-reflection pipeline."""

    def __init__(
        self,
        pipeline_config: Optional[SelfReflectionConfig] = None,
        batch_config: Optional[BatchConfig] = None,
        results_dir: str = "data/results",
        api_key: Optional[str] = None
    ):
        self.pipeline_config = pipeline_config or SelfReflectionConfig()
        self.batch_config = batch_config or BatchConfig()
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.api_key = api_key or os.getenv("NVIDIA_API_KEY")
        self.client = AsyncNVIDIANIMClient(
            api_key=self.api_key,
            max_concurrent=self.batch_config.max_concurrent
        )

        logger.info(f"AsyncBatchPipeline initialized (max_concurrent={self.batch_config.max_concurrent})")

    async def solve_batch(
        self,
        problems: List[Problem],
        checkpoint_callback: Optional[Callable] = None
    ) -> BatchResult:
        """Solve a batch of problems concurrently.

        Args:
            problems: List of Problem objects
            checkpoint_callback: Optional callback after each checkpoint

        Returns:
            BatchResult with all results
        """
        start_time = time.time()
        results = []
        errors = []
        successful = 0
        failed = 0

        semaphore = asyncio.Semaphore(self.batch_config.max_concurrent)

        async def solve_with_semaphore(problem: Problem, index: int):
            """Solve single problem with semaphore."""
            async with semaphore:
                try:
                    result = await self._solve_single(problem)
                    return (index, result, None)
                except Exception as e:
                    logger.error(f"Problem {problem.id} failed: {e}")
                    return (index, None, {"problem_id": problem.id, "error": str(e)})

        # Process all problems
        tasks = [
            solve_with_semaphore(problem, i)
            for i, problem in enumerate(problems)
        ]

        # Track progress
        completed = 0

        for coro in asyncio.as_completed(tasks):
            try:
                index, result, error = await coro
                completed += 1

                if error:
                    errors.append(error)
                    failed += 1
                else:
                    results.append((index, result))
                    successful += 1

                # Checkpoint
                if self.batch_config.save_intermediate and completed % self.batch_config.checkpoint_interval == 0:
                    await self._save_checkpoint(results, errors, completed, len(problems))
                    if checkpoint_callback:
                        await checkpoint_callback(completed, len(problems))

                    logger.info(f"Progress: {completed}/{len(problems)} problems processed")

            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                errors.append({"error": str(e)})
                failed += 1

        # Sort results by index to maintain order
        results.sort(key=lambda x: x[0])
        final_results = [r[1] for r in results]

        total_latency = time.time() - start_time
        total_tokens = sum(r.total_tokens for r in final_results)

        batch_result = BatchResult(
            total_problems=len(problems),
            successful=successful,
            failed=failed,
            results=final_results,
            errors=errors,
            total_tokens=total_tokens,
            total_latency_seconds=total_latency,
            avg_latency_per_problem=total_latency / len(problems) if problems else 0
        )

        # Save final results
        await self._save_final_results(batch_result)

        logger.info(
            f"Batch completed: {successful}/{len(problems)} successful, "
            f"{total_latency:.2f}s total, {batch_result.avg_latency_per_problem:.2f}s avg"
        )

        return batch_result

    async def _solve_single(self, problem: Problem) -> SelfReflectionResult:
        """Solve a single problem (simplified async version)."""
        from orchestration.self_reflection_pipeline import SelfReflectionPipeline

        # Use async client in pipeline
        pipeline = SelfReflectionPipeline(
            config=self.pipeline_config,
            api_key=self.api_key
        )

        # For now, run in executor to avoid blocking
        # TODO: Make pipeline fully async
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: pipeline.solve(
                problem=problem.question,
                problem_id=problem.id,
                ground_truth=problem.answer
            )
        )

        return result

    async def _save_checkpoint(
        self,
        results: List[tuple],
        errors: List[Dict],
        completed: int,
        total: int
    ):
        """Save checkpoint of current progress."""
        checkpoint = {
            "timestamp": datetime.now().isoformat(),
            "completed": completed,
            "total": total,
            "results": [asdict(r[1]) if r[1] else None for r in results],
            "errors": errors
        }

        checkpoint_file = self.results_dir / f"checkpoint_{completed}_{total}.json"
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint, f, indent=2, default=str)

        logger.debug(f"Checkpoint saved: {checkpoint_file}")

    async def _save_final_results(self, batch_result: BatchResult):
        """Save final batch results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.results_dir / f"batch_results_{timestamp}.json"

        # Convert to dict for JSON serialization
        result_dict = {
            "total_problems": batch_result.total_problems,
            "successful": batch_result.successful,
            "failed": batch_result.failed,
            "total_tokens": batch_result.total_tokens,
            "total_latency_seconds": batch_result.total_latency_seconds,
            "avg_latency_per_problem": batch_result.avg_latency_per_problem,
            "timestamp": batch_result.timestamp,
            "results": [asdict(r) for r in batch_result.results],
            "errors": batch_result.errors
        }

        with open(results_file, 'w') as f:
            json.dump(result_dict, f, indent=2, default=str)

        logger.info(f"Results saved to {results_file}")

    async def close(self):
        """Close the async client."""
        await self.client.close()

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()


async def run_batch_async(
    problems: List[Problem],
    pipeline_config: Optional[SelfReflectionConfig] = None,
    batch_config: Optional[BatchConfig] = None,
    results_dir: str = "data/results"
) -> BatchResult:
    """Convenience function to run batch processing.

    Args:
        problems: List of problems to solve
        pipeline_config: Pipeline configuration
        batch_config: Batch processing configuration
        results_dir: Directory for results

    Returns:
        BatchResult
    """
    async with AsyncBatchPipeline(
        pipeline_config=pipeline_config,
        batch_config=batch_config,
        results_dir=results_dir
    ) as pipeline:
        return await pipeline.solve_batch(problems)
