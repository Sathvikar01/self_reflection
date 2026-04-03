"""Metrics collection and analysis."""

import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
import time


@dataclass
class ProblemMetrics:
    """Metrics for a single problem."""
    problem_id: str
    solved: bool
    correct: bool
    score: float
    
    input_tokens: int
    output_tokens: int
    latency_seconds: float
    
    num_expansions: int = 0
    num_reflections: int = 0
    num_backtracks: int = 0
    num_api_calls: int = 0
    
    path_length: int = 0
    max_depth: int = 0
    
    ground_truth: Optional[str] = None
    prediction: Optional[str] = None


@dataclass
class ExperimentMetrics:
    """Aggregated metrics for experiment."""
    experiment_name: str
    num_problems: int
    accuracy: float
    avg_score: float
    
    total_tokens: int
    avg_tokens_per_problem: float
    
    total_latency_seconds: float
    avg_latency_seconds: float
    
    total_expansions: int
    total_reflections: int
    total_backtracks: int
    
    avg_path_length: float
    
    cost_ratio: Optional[float] = None
    improvement_over_baseline: Optional[float] = None


class MetricsCollector:
    """Collect and analyze metrics from experiments."""
    
    def __init__(self, name: str = "experiment"):
        self.name = name
        self._problem_metrics: List[ProblemMetrics] = []
        self._timestamps: Dict[str, float] = {}
    
    def start_problem(self, problem_id: str) -> None:
        """Mark start of problem solving."""
        self._timestamps[problem_id] = time.time()
    
    def record_problem(self, metrics: ProblemMetrics) -> None:
        """Record metrics for a solved problem."""
        self._problem_metrics.append(metrics)
    
    def get_aggregate_metrics(self) -> ExperimentMetrics:
        """Compute aggregate metrics."""
        if not self._problem_metrics:
            return self._empty_metrics()
        
        n = len(self._problem_metrics)
        
        total_input = sum(m.input_tokens for m in self._problem_metrics)
        total_output = sum(m.output_tokens for m in self._problem_metrics)
        total_tokens = total_input + total_output
        
        total_latency = sum(m.latency_seconds for m in self._problem_metrics)
        
        correct_count = sum(1 for m in self._problem_metrics if m.correct)
        
        total_expansions = sum(m.num_expansions for m in self._problem_metrics)
        total_reflections = sum(m.num_reflections for m in self._problem_metrics)
        total_backtracks = sum(m.num_backtracks for m in self._problem_metrics)
        
        total_path_length = sum(m.path_length for m in self._problem_metrics)
        
        return ExperimentMetrics(
            experiment_name=self.name,
            num_problems=n,
            accuracy=correct_count / n if n > 0 else 0,
            avg_score=sum(m.score for m in self._problem_metrics) / n,
            total_tokens=total_tokens,
            avg_tokens_per_problem=total_tokens / n if n > 0 else 0,
            total_latency_seconds=total_latency,
            avg_latency_seconds=total_latency / n if n > 0 else 0,
            total_expansions=total_expansions,
            total_reflections=total_reflections,
            total_backtracks=total_backtracks,
            avg_path_length=total_path_length / n if n > 0 else 0,
        )
    
    def _empty_metrics(self) -> ExperimentMetrics:
        """Return empty metrics."""
        return ExperimentMetrics(
            experiment_name=self.name,
            num_problems=0,
            accuracy=0.0,
            avg_score=0.0,
            total_tokens=0,
            avg_tokens_per_problem=0.0,
            total_latency_seconds=0.0,
            avg_latency_seconds=0.0,
            total_expansions=0,
            total_reflections=0,
            total_backtracks=0,
            avg_path_length=0.0,
        )
    
    def compare_with_baseline(
        self,
        baseline_metrics: 'MetricsCollector',
    ) -> Dict[str, Any]:
        """Compare metrics with baseline."""
        ours = self.get_aggregate_metrics()
        theirs = baseline_metrics.get_aggregate_metrics()
        
        if theirs.num_problems == 0:
            return {"error": "No baseline metrics"}
        
        improvement = ours.accuracy - theirs.accuracy
        
        token_increase = (
            (ours.avg_tokens_per_problem - theirs.avg_tokens_per_problem)
            / theirs.avg_tokens_per_problem * 100
            if theirs.avg_tokens_per_problem > 0 else 0
        )
        
        latency_increase = (
            (ours.avg_latency_seconds - theirs.avg_latency_seconds)
            / theirs.avg_latency_seconds * 100
            if theirs.avg_latency_seconds > 0 else 0
        )
        
        return {
            "accuracy_improvement": improvement,
            "accuracy_ours": ours.accuracy,
            "accuracy_baseline": theirs.accuracy,
            "token_increase_percent": token_increase,
            "latency_increase_percent": latency_increase,
            "cost_effectiveness": improvement / max(1, token_increase / 100) if token_increase > 0 else 0,
            "backtracks_per_problem": ours.total_backtracks / max(1, ours.num_problems),
            "reflections_per_problem": ours.total_reflections / max(1, ours.num_problems),
        }
    
    def get_error_analysis(self) -> Dict[str, Any]:
        """Analyze errors in incorrect predictions."""
        incorrect = [m for m in self._problem_metrics if not m.correct]
        
        if not incorrect:
            return {"total_errors": 0}
        
        low_score_errors = sum(1 for m in incorrect if m.score < 0.5)
        high_score_errors = len(incorrect) - low_score_errors
        
        avg_backtracks_incorrect = sum(m.num_backtracks for m in incorrect) / len(incorrect)
        avg_backtracks_correct = 0.0
        correct = [m for m in self._problem_metrics if m.correct]
        if correct:
            avg_backtracks_correct = sum(m.num_backtracks for m in correct) / len(correct)
        
        return {
            "total_errors": len(incorrect),
            "low_score_errors": low_score_errors,
            "high_score_errors": high_score_errors,
            "avg_backtracks_incorrect": avg_backtracks_incorrect,
            "avg_backtracks_correct": avg_backtracks_correct,
            "avg_score_incorrect": sum(m.score for m in incorrect) / len(incorrect),
        }
    
    def get_action_distribution(self) -> Dict[str, int]:
        """Get distribution of actions taken."""
        return {
            "expansions": sum(m.num_expansions for m in self._problem_metrics),
            "reflections": sum(m.num_reflections for m in self._problem_metrics),
            "backtracks": sum(m.num_backtracks for m in self._problem_metrics),
        }
    
    def export_results(self, filepath: str) -> None:
        """Export metrics to JSON file."""
        data = {
            "name": self.name,
            "aggregate": self.get_aggregate_metrics().__dict__,
            "error_analysis": self.get_error_analysis(),
            "action_distribution": self.get_action_distribution(),
            "problems": [m.__dict__ for m in self._problem_metrics],
        }
        
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)
    
    def clear(self) -> None:
        """Clear all metrics."""
        self._problem_metrics.clear()
        self._timestamps.clear()


class ComparisonReport:
    """Generate comparison report between baseline and RL."""
    
    def __init__(
        self,
        baseline_metrics: MetricsCollector,
        rl_metrics: MetricsCollector,
    ):
        self.baseline = baseline_metrics
        self.rl = rl_metrics
    
    def generate_report(self) -> str:
        """Generate text report."""
        baseline = self.baseline.get_aggregate_metrics()
        rl = self.rl.get_aggregate_metrics()
        
        report = []
        report.append("=" * 60)
        report.append("EXPERIMENT COMPARISON REPORT")
        report.append("=" * 60)
        report.append("")
        
        report.append("ACCURACY")
        report.append("-" * 40)
        report.append(f"Baseline accuracy: {baseline.accuracy:.1%}")
        report.append(f"RL accuracy:       {rl.accuracy:.1%}")
        improvement = rl.accuracy - baseline.accuracy
        report.append(f"Improvement:       {improvement:+.1%}")
        report.append("")
        
        report.append("COST ANALYSIS")
        report.append("-" * 40)
        report.append(f"Baseline tokens:   {baseline.avg_tokens_per_problem:.0f} avg")
        report.append(f"RL tokens:         {rl.avg_tokens_per_problem:.0f} avg")
        token_ratio = rl.avg_tokens_per_problem / max(1, baseline.avg_tokens_per_problem)
        report.append(f"Token ratio:       {token_ratio:.2f}x")
        report.append("")
        
        report.append("LATENCY")
        report.append("-" * 40)
        report.append(f"Baseline latency:  {baseline.avg_latency_seconds:.1f}s avg")
        report.append(f"RL latency:        {rl.avg_latency_seconds:.1f}s avg")
        report.append("")
        
        report.append("RL ACTIONS")
        report.append("-" * 40)
        report.append(f"Total expansions:  {rl.total_expansions}")
        report.append(f"Total reflections: {rl.total_reflections}")
        report.append(f"Total backtracks:  {rl.total_backtracks}")
        report.append("")
        
        if rl.num_problems > 0:
            report.append("EFFICIENCY")
            report.append("-" * 40)
            actions_per_problem = (
                rl.total_expansions + rl.total_reflections + rl.total_backtracks
            ) / rl.num_problems
            report.append(f"Actions/problem:   {actions_per_problem:.1f}")
            report.append(f"Backtracks/problem:{rl.total_backtracks / rl.num_problems:.1f}")
        
        report.append("")
        report.append("=" * 60)
        
        return "\n".join(report)
    
    def get_summary_dict(self) -> Dict[str, Any]:
        """Get summary as dictionary."""
        baseline = self.baseline.get_aggregate_metrics()
        rl = self.rl.get_aggregate_metrics()
        
        return {
            "accuracy": {
                "baseline": baseline.accuracy,
                "rl": rl.accuracy,
                "improvement": rl.accuracy - baseline.accuracy,
            },
            "tokens": {
                "baseline_avg": baseline.avg_tokens_per_problem,
                "rl_avg": rl.avg_tokens_per_problem,
                "ratio": rl.avg_tokens_per_problem / max(1, baseline.avg_tokens_per_problem),
            },
            "latency": {
                "baseline_avg": baseline.avg_latency_seconds,
                "rl_avg": rl.avg_latency_seconds,
            },
            "rl_actions": {
                "expansions": rl.total_expansions,
                "reflections": rl.total_reflections,
                "backtracks": rl.total_backtracks,
            },
        }
