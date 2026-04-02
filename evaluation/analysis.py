"""Analysis and result processing."""

import json
import statistics
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict
from datetime import datetime


@dataclass
class ProblemAnalysis:
    """Analysis of a single problem's solving process."""
    problem_id: str
    correct: bool
    score: float
    num_steps: int
    num_backtracks: int
    num_reflections: int
    total_tokens: int
    latency: float
    
    efficiency: float = 0.0
    cost_per_correct: float = 0.0


class ResultAnalyzer:
    """Analyze experiment results."""
    
    def __init__(self):
        self._analyses: List[ProblemAnalysis] = []
    
    def analyze_results(
        self,
        results: List[Dict[str, Any]],
    ) -> List[ProblemAnalysis]:
        """Analyze list of problem results.
        
        Args:
            results: List of result dictionaries
        
        Returns:
            List of ProblemAnalysis
        """
        analyses = []
        
        for r in results:
            total_tokens = r.get("total_tokens_input", 0) + r.get("total_tokens_output", 0)
            
            num_steps = len(r.get("reasoning_path", []))
            
            efficiency = r.get("final_score", 0) / max(1, r.get("num_expansions", 1))
            
            analysis = ProblemAnalysis(
                problem_id=r.get("problem_id", "unknown"),
                correct=r.get("correct", False),
                score=r.get("final_score", 0),
                num_steps=num_steps,
                num_backtracks=r.get("num_backtracks", 0),
                num_reflections=r.get("num_reflections", 0),
                total_tokens=total_tokens,
                latency=r.get("latency_seconds", 0),
                efficiency=efficiency,
            )
            
            analyses.append(analysis)
        
        self._analyses = analyses
        return analyses
    
    def get_statistics(self) -> Dict[str, Any]:
        """Compute aggregate statistics."""
        if not self._analyses:
            return {}
        
        correct = [a for a in self._analyses if a.correct]
        incorrect = [a for a in self._analyses if not a.correct]
        
        return {
            "total_problems": len(self._analyses),
            "correct": len(correct),
            "accuracy": len(correct) / len(self._analyses),
            "avg_score": statistics.mean(a.score for a in self._analyses),
            "avg_tokens": statistics.mean(a.total_tokens for a in self._analyses),
            "avg_latency": statistics.mean(a.latency for a in self._analyses),
            "avg_steps": statistics.mean(a.num_steps for a in self._analyses),
            "avg_backtracks": statistics.mean(a.num_backtracks for a in self._analyses),
            "correct_stats": {
                "avg_tokens": statistics.mean(a.total_tokens for a in correct) if correct else 0,
                "avg_steps": statistics.mean(a.num_steps for a in correct) if correct else 0,
                "avg_backtracks": statistics.mean(a.num_backtracks for a in correct) if correct else 0,
            },
            "incorrect_stats": {
                "avg_tokens": statistics.mean(a.total_tokens for a in incorrect) if incorrect else 0,
                "avg_steps": statistics.mean(a.num_steps for a in incorrect) if incorrect else 0,
                "avg_backtracks": statistics.mean(a.num_backtracks for a in incorrect) if incorrect else 0,
            },
        }
    
    def correlation_analysis(self) -> Dict[str, float]:
        """Analyze correlations between metrics."""
        if len(self._analyses) < 3:
            return {}
        
        scores = [a.score for a in self._analyses]
        tokens = [a.total_tokens for a in self._analyses]
        backtracks = [a.num_backtracks for a in self._analyses]
        steps = [a.num_steps for a in self._analyses]
        
        def pearson(x, y):
            n = len(x)
            if n < 2:
                return 0
            
            mean_x = statistics.mean(x)
            mean_y = statistics.mean(y)
            
            num = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
            den = (sum((xi - mean_x) ** 2 for xi in x) * sum((yi - mean_y) ** 2 for yi in y)) ** 0.5
            
            if den == 0:
                return 0
            return num / den
        
        return {
            "score_tokens": pearson(scores, tokens),
            "score_backtracks": pearson(scores, backtracks),
            "tokens_backtracks": pearson(tokens, backtracks),
            "steps_score": pearson(steps, scores),
        }
    
    def get_error_patterns(self) -> Dict[str, Any]:
        """Identify patterns in errors."""
        incorrect = [a for a in self._analyses if not a.correct]
        
        if not incorrect:
            return {"total_errors": 0}
        
        low_score_errors = [a for a in incorrect if a.score < 0.5]
        high_backtrack_errors = [a for a in incorrect if a.num_backtracks > 3]
        short_path_errors = [a for a in incorrect if a.num_steps < 3]
        
        return {
            "total_errors": len(incorrect),
            "low_score_errors": len(low_score_errors),
            "high_backtrack_errors": len(high_backtrack_errors),
            "short_path_errors": len(short_path_errors),
            "error_rate": len(incorrect) / len(self._analyses),
            "avg_error_score": statistics.mean(a.score for a in incorrect),
            "avg_error_backtracks": statistics.mean(a.num_backtracks for a in incorrect),
        }
    
    def export_analysis(self, filepath: str):
        """Export analysis to JSON."""
        data = {
            "timestamp": datetime.now().isoformat(),
            "statistics": self.get_statistics(),
            "correlations": self.correlation_analysis(),
            "error_patterns": self.get_error_patterns(),
            "problems": [
                {
                    "problem_id": a.problem_id,
                    "correct": a.correct,
                    "score": a.score,
                    "num_steps": a.num_steps,
                    "num_backtracks": a.num_backtracks,
                    "total_tokens": a.total_tokens,
                    "latency": a.latency,
                }
                for a in self._analyses
            ],
        }
        
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)


class BacktrackAnalyzer:
    """Analyze backtracking behavior."""
    
    def __init__(self):
        self._backtrack_events: List[Dict] = []
    
    def record_backtrack(
        self,
        problem_id: str,
        from_score: float,
        to_score: float,
        from_depth: int,
        to_depth: int,
        outcome: str,
    ):
        """Record a backtrack event."""
        self._backtrack_events.append({
            "problem_id": problem_id,
            "from_score": from_score,
            "to_score": to_score,
            "from_depth": from_depth,
            "to_depth": to_depth,
            "outcome": outcome,
            "score_improvement": to_score - from_score,
        })
    
    def analyze(self) -> Dict[str, Any]:
        """Analyze backtrack events."""
        if not self._backtrack_events:
            return {"total_backtracks": 0}
        
        successful = [
            e for e in self._backtrack_events
            if e["outcome"] == "correct" or e["score_improvement"] > 0
        ]
        
        return {
            "total_backtracks": len(self._backtrack_events),
            "successful": len(successful),
            "success_rate": len(successful) / len(self._backtrack_events),
            "avg_score_improvement": statistics.mean(
                e["score_improvement"] for e in self._backtrack_events
            ),
            "avg_depth_change": statistics.mean(
                e["to_depth"] - e["from_depth"] for e in self._backtrack_events
            ),
            "backtracks_by_problem": self._group_by_problem(),
        }
    
    def _group_by_problem(self) -> Dict[str, int]:
        """Group backtrack events by problem."""
        counts = defaultdict(int)
        for e in self._backtrack_events:
            counts[e["problem_id"]] += 1
        return dict(counts)


def generate_comparison_table(
    baseline_stats: Dict[str, Any],
    rl_stats: Dict[str, Any],
) -> str:
    """Generate markdown comparison table.
    
    Args:
        baseline_stats: Baseline statistics
        rl_stats: RL method statistics
    
    Returns:
        Markdown formatted table
    """
    rows = [
        "| Metric | Baseline | RL | Difference |",
        "|--------|----------|-----|------------|",
    ]
    
    metrics = [
        ("Accuracy", "accuracy", "{:.1%}"),
        ("Avg Score", "avg_score", "{:.3f}"),
        ("Avg Tokens", "avg_tokens", "{:.0f}"),
        ("Avg Latency (s)", "avg_latency", "{:.1f}"),
        ("Avg Steps", "avg_steps", "{:.1f}"),
        ("Avg Backtracks", "avg_backtracks", "{:.1f}"),
    ]
    
    for name, key, fmt in metrics:
        baseline_val = baseline_stats.get(key, 0)
        rl_val = rl_stats.get(key, 0)
        
        if isinstance(baseline_val, float) and isinstance(rl_val, float):
            diff = rl_val - baseline_val
            diff_str = f"+{diff:{fmt[2:-1]}}" if diff > 0 else f"{diff:{fmt[2:-1]}}"
        else:
            diff_str = "N/A"
        
        rows.append(f"| {name} | {baseline_val:{fmt[2:-1]}} | {rl_val:{fmt[2:-1]}} | {diff_str} |")
    
    return "\n".join(rows)
