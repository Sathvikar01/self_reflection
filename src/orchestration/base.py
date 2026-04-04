"""Base classes for pipeline orchestration.

This module provides abstract base classes to reduce code duplication
across the 8+ pipeline implementations in the codebase.
"""

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Dict, Optional, Any, Generic, TypeVar
from loguru import logger


@dataclass
class BaseResult:
    """Base class for all pipeline results.
    
    All pipeline result dataclasses should inherit from this to ensure
    consistent interface for result handling.
    """
    problem_id: str
    problem: str
    final_answer: str
    correct: Optional[bool] = None
    ground_truth: Optional[str] = None
    total_tokens: int = 0
    latency_seconds: float = 0.0
    reasoning_path: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BaseResult":
        """Create instance from dictionary."""
        return cls(**data)


@dataclass
class BasePipelineConfig:
    """Base configuration for all pipelines.
    
    Common configuration parameters shared across all pipeline types.
    """
    max_iterations: int = 50
    results_dir: str = "data/results"
    checkpoint_interval: int = 10
    save_intermediate: bool = True
    log_verbosity: int = 1  # 0=silent, 1=normal, 2=debug


TResult = TypeVar("TResult", bound=BaseResult)
TConfig = TypeVar("TConfig", bound=BasePipelineConfig)


class BasePipeline(ABC, Generic[TResult, TConfig]):
    """Abstract base class for all reasoning pipelines.
    
    This class provides common functionality shared across all pipeline
    implementations, reducing code duplication from ~500+ lines to
    a single inheritance hierarchy.
    
    Subclasses must implement:
        - solve(): Core problem-solving logic
        - _create_result(): Create the appropriate result type
    """
    
    def __init__(
        self,
        config: TConfig,
        results_dir: str = "data/results",
    ):
        """Initialize base pipeline.
        
        Args:
            config: Pipeline configuration
            results_dir: Directory for saving results
        """
        self.config = config
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self._results: List[TResult] = []
        self._start_time: Optional[float] = None
        
        logger.info(f"{self.__class__.__name__} initialized")
    
    @abstractmethod
    def solve(
        self,
        problem: str,
        problem_id: str = "unknown",
        ground_truth: Optional[str] = None,
    ) -> TResult:
        """Solve a single problem.
        
        Args:
            problem: Problem statement
            problem_id: Unique identifier
            ground_truth: Optional ground truth answer
            
        Returns:
            Result object with answer and metrics
        """
        pass
    
    def solve_batch(
        self,
        problems: List[Dict[str, Any]],
        checkpoint_prefix: str = "batch",
    ) -> List[TResult]:
        """Solve multiple problems with checkpointing.
        
        This method provides common batch solving logic that works
        across all pipeline types.
        
        Args:
            problems: List of dicts with 'id', 'problem', 'answer' keys
            checkpoint_prefix: Prefix for checkpoint files
            
        Returns:
            List of results
        """
        results = []
        
        for i, item in enumerate(problems):
            problem_id = item.get("id", f"problem_{i}")
            problem = item.get("problem", item.get("question", ""))
            ground_truth = item.get("answer", item.get("ground_truth"))
            
            try:
                result = self.solve(
                    problem=problem,
                    problem_id=problem_id,
                    ground_truth=ground_truth,
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Error solving {problem_id}: {e}")
                continue
            
            if self.config.save_intermediate:
                if (i + 1) % self.config.checkpoint_interval == 0:
                    self._save_checkpoint(results, f"{checkpoint_prefix}_step_{i+1}")
        
        return results
    
    def _save_checkpoint(self, results: List[TResult], name: str):
        """Save checkpoint of current results.
        
        Args:
            results: Results to checkpoint
            name: Checkpoint name
        """
        checkpoint_path = self.results_dir / f"{name}.json"
        
        data = {
            "checkpoint_name": name,
            "timestamp": time.time(),
            "num_results": len(results),
            "results": [r.to_dict() for r in results],
        }
        
        with open(checkpoint_path, "w") as f:
            json.dump(data, f, indent=2)
        
        logger.debug(f"Checkpoint saved: {checkpoint_path}")
    
    def save_results(self, filename: str = "results.json"):
        """Save all results to file.
        
        Args:
            filename: Output filename
        """
        results_path = self.results_dir / filename
        
        data = {
            "timestamp": time.time(),
            "config": asdict(self.config),
            "num_problems": len(self._results),
            "results": [r.to_dict() for r in self._results],
            "aggregate_stats": self._compute_aggregate_stats(),
        }
        
        with open(results_path, "w") as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Results saved to {results_path}")
    
    def _compute_aggregate_stats(self) -> Dict[str, Any]:
        """Compute aggregate statistics across all results.
        
        Returns:
            Dictionary of aggregate statistics
        """
        if not self._results:
            return {}
        
        total_tokens = sum(r.total_tokens for r in self._results)
        
        stats = {
            "total_problems": len(self._results),
            "total_tokens": total_tokens,
            "avg_tokens_per_problem": total_tokens / len(self._results),
            "avg_latency": sum(r.latency_seconds for r in self._results) / len(self._results),
        }
        
        correct_count = sum(1 for r in self._results if r.correct)
        if correct_count > 0:
            stats["total_correct"] = correct_count
            stats["accuracy"] = correct_count / len(self._results)
        
        return stats
    
    def _check_answer(self, answer: str, ground_truth: str) -> bool:
        """Check if answer matches ground truth.
        
        Handles various answer formats including yes/no and numeric answers.
        
        Args:
            answer: Extracted answer
            ground_truth: Expected answer
            
        Returns:
            True if answers match
        """
        if not answer or not ground_truth:
            return False
        
        answer_lower = answer.lower().strip()
        truth_lower = ground_truth.lower().strip()
        
        if truth_lower in ["yes", "no"]:
            return truth_lower in answer_lower
        
        return truth_lower in answer_lower or answer_lower == truth_lower
    
    def get_summary(self) -> str:
        """Get summary string of results.
        
        Returns:
            Human-readable summary
        """
        stats = self._compute_aggregate_stats()
        
        if not stats:
            return "No results yet"
        
        lines = [f"Problems: {stats['total_problems']}"]
        
        if "accuracy" in stats:
            lines.append(f"Accuracy: {stats['accuracy']:.1%}")
        
        lines.extend([
            f"Avg Tokens: {stats['avg_tokens_per_problem']:.0f}",
            f"Avg Time: {stats['avg_latency']:.1f}s",
        ])
        
        return "\n".join(lines)
    
    def get_results(self) -> List[TResult]:
        """Get all results.
        
        Returns:
            List of results
        """
        return self._results.copy()
    
    def reset(self):
        """Reset pipeline state."""
        self._results.clear()
        logger.info(f"{self.__class__.__name__} reset")
    
    def close(self):
        """Close pipeline and save final results."""
        if self._results:
            self.save_results()
        logger.info(f"{self.__class__.__name__} closed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False


def convert_to_serializable(obj: Any) -> Any:
    """Convert objects to JSON-serializable format.
    
    Utility function for handling non-serializable types in results.
    
    Args:
        obj: Object to convert
        
    Returns:
        JSON-serializable version of object
    """
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    elif hasattr(obj, "__dict__"):
        return {k: convert_to_serializable(v) for k, v in obj.__dict__.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    else:
        return obj
