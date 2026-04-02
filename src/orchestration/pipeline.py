"""Main RL-guided reasoning pipeline."""

import os
import json
import time
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from pathlib import Path
from loguru import logger

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ..generator.nim_client import NVIDIANIMClient, GenerationConfig
from ..generator.prompts import PromptBuilder
from ..evaluator.prm_client import PRMEvaluator, PRMConfig, EvaluationResult
from ..rl_controller.tree import StateTree, TreeNode, NodeType
from ..rl_controller.actions import ActionExecutor, ActionType, ActionConfig
from ..rl_controller.mcts import MCTSController, MCTSConfig, MCTSStats


@dataclass
class PipelineConfig:
    """Configuration for the RL pipeline."""
    max_iterations: int = 50
    early_stop_score: float = 0.9
    checkpoint_interval: int = 10
    save_intermediate: bool = True
    log_tree_states: bool = True
    
    mcts: MCTSConfig = field(default_factory=MCTSConfig)
    action: ActionConfig = field(default_factory=ActionConfig)
    prm: PRMConfig = field(default_factory=PRMConfig)


@dataclass
class ProblemResult:
    """Result of solving a single problem."""
    problem_id: str
    problem: str
    final_answer: str
    reasoning_path: List[str]
    final_score: float
    correct: Optional[bool] = None
    ground_truth: Optional[str] = None
    
    total_tokens_input: int = 0
    total_tokens_output: int = 0
    total_api_calls: int = 0
    
    num_expansions: int = 0
    num_reflections: int = 0
    num_backtracks: int = 0
    max_depth_reached: int = 0
    
    latency_seconds: float = 0.0
    convergence_iteration: Optional[int] = None
    
    tree_stats: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class RLPipeline:
    """Main pipeline for RL-guided reasoning."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        config: Optional[PipelineConfig] = None,
        results_dir: str = "data/results",
    ):
        self.config = config or PipelineConfig()
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.generator = NVIDIANIMClient(api_key=api_key)
        self.evaluator = PRMEvaluator(
            client=self.generator,
            config=self.config.prm,
        )
        self.action_executor = ActionExecutor(
            generator=self.generator,
            evaluator=self.evaluator,
            config=self.config.action,
        )
        self.mcts = MCTSController(
            action_executor=self.action_executor,
            config=self.config.mcts,
        )
        
        self._results: List[ProblemResult] = []
        self._current_checkpoint = 0
        
        logger.info("RL Pipeline initialized")
    
    def solve(
        self,
        problem: str,
        problem_id: str = "unknown",
        ground_truth: Optional[str] = None,
    ) -> ProblemResult:
        """Solve a single problem using RL-guided reasoning.
        
        Args:
            problem: Problem statement
            problem_id: Unique identifier
            ground_truth: Optional ground truth answer
        
        Returns:
            ProblemResult with answer and metrics
        """
        start_time = time.time()
        
        logger.info(f"Solving problem {problem_id}: {problem[:100]}...")
        
        answer, score, path = self.mcts.search(
            problem=problem,
            max_iterations=self.config.max_iterations,
            early_stop_threshold=self.config.early_stop_score,
        )
        
        mcts_stats = self.mcts.get_stats()
        gen_stats = self.generator.get_stats()
        action_stats = self.action_executor.get_stats()
        
        result = ProblemResult(
            problem_id=problem_id,
            problem=problem,
            final_answer=answer,
            reasoning_path=path,
            final_score=score,
            ground_truth=ground_truth,
            total_tokens_input=gen_stats["total_input_tokens"],
            total_tokens_output=gen_stats["total_output_tokens"],
            total_api_calls=gen_stats["total_requests"],
            num_expansions=mcts_stats.total_expansions,
            num_reflections=mcts_stats.total_reflections,
            num_backtracks=mcts_stats.total_backtracks,
            max_depth_reached=mcts_stats.max_depth_reached,
            latency_seconds=time.time() - start_time,
            convergence_iteration=mcts_stats.convergence_step,
            tree_stats=mcts_stats.__dict__,
        )
        
        self._results.append(result)
        
        logger.info(
            f"Problem {problem_id} solved: score={score:.2f}, "
            f"tokens={result.total_tokens_input + result.total_tokens_output}, "
            f"time={result.latency_seconds:.1f}s"
        )
        
        return result
    
    def solve_batch(
        self,
        problems: List[Dict[str, Any]],
        checkpoint_prefix: str = "batch",
    ) -> List[ProblemResult]:
        """Solve multiple problems with checkpointing.
        
        Args:
            problems: List of dicts with 'id', 'problem', 'answer' keys
            checkpoint_prefix: Prefix for checkpoint files
        
        Returns:
            List of ProblemResult
        """
        results = []
        
        for i, item in enumerate(problems):
            problem_id = item.get("id", f"problem_{i}")
            problem = item.get("problem", item.get("question", ""))
            ground_truth = item.get("answer", item.get("ground_truth"))
            
            result = self.solve(
                problem=problem,
                problem_id=problem_id,
                ground_truth=ground_truth,
            )
            results.append(result)
            
            if self.config.save_intermediate and (i + 1) % self.config.checkpoint_interval == 0:
                self._save_checkpoint(results, f"{checkpoint_prefix}_step_{i+1}")
        
        return results
    
    def _save_checkpoint(self, results: List[ProblemResult], name: str):
        """Save checkpoint of current results."""
        checkpoint_path = self.results_dir / f"{name}.json"
        
        data = {
            "checkpoint_name": name,
            "timestamp": time.time(),
            "num_results": len(results),
            "results": [asdict(r) for r in results],
        }
        
        with open(checkpoint_path, "w") as f:
            json.dump(data, f, indent=2)
        
        logger.debug(f"Checkpoint saved: {checkpoint_path}")
    
    def save_results(self, filename: str = "results.json"):
        """Save all results to file."""
        results_path = self.results_dir / filename
        
        data = {
            "timestamp": time.time(),
            "config": asdict(self.config),
            "num_problems": len(self._results),
            "results": [asdict(r) for r in self._results],
            "aggregate_stats": self._compute_aggregate_stats(),
        }
        
        with open(results_path, "w") as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Results saved to {results_path}")
    
    def _compute_aggregate_stats(self) -> Dict[str, Any]:
        """Compute aggregate statistics across all results."""
        if not self._results:
            return {}
        
        total_tokens = sum(
            r.total_tokens_input + r.total_tokens_output
            for r in self._results
        )
        
        return {
            "total_problems": len(self._results),
            "total_tokens": total_tokens,
            "avg_tokens_per_problem": total_tokens / len(self._results),
            "avg_score": sum(r.final_score for r in self._results) / len(self._results),
            "avg_latency": sum(r.latency_seconds for r in self._results) / len(self._results),
            "avg_expansions": sum(r.num_expansions for r in self._results) / len(self._results),
            "avg_reflections": sum(r.num_reflections for r in self._results) / len(self._results),
            "avg_backtracks": sum(r.num_backtracks for r in self._results) / len(self._results),
            "total_correct": sum(1 for r in self._results if r.correct),
        }
    
    def get_summary(self) -> str:
        """Get summary string of results."""
        stats = self._compute_aggregate_stats()
        
        if not stats:
            return "No results yet"
        
        return (
            f"Problems: {stats['total_problems']}\n"
            f"Avg Score: {stats['avg_score']:.3f}\n"
            f"Avg Tokens: {stats['avg_tokens_per_problem']:.0f}\n"
            f"Avg Time: {stats['avg_latency']:.1f}s\n"
            f"Avg Expansions: {stats['avg_expansions']:.1f}\n"
            f"Avg Backtracks: {stats['avg_backtracks']:.1f}"
        )
    
    def reset(self):
        """Reset pipeline state."""
        self.generator.reset_stats()
        self.generator.clear_cache()
        self.evaluator.reset_stats()
        self.action_executor.reset_stats()
        self._results.clear()
        logger.info("Pipeline reset")
    
    def close(self):
        """Close pipeline and save final results."""
        self.save_results()
        self.generator.close()
        logger.info("Pipeline closed")
