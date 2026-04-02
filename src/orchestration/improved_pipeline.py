"""Improved pipeline with all fixes integrated."""

import os
import json
import time
import random
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generator.nim_client import NVIDIANIMClient, GenerationConfig
from generator.prompts import PromptBuilder
from evaluator.improved_prm import ImprovedPRM, VerificationResult, ComparativeLearner
from rl_controller.tree import StateTree, TreeNode, NodeType
from rl_controller.actions import ActionExecutor, ActionType, ActionConfig, ActionResult
from rl_controller.improved_mcts import ImprovedMCTSController, ImprovedMCTSConfig


@dataclass
class ImprovedPipelineConfig:
    """Configuration for improved pipeline."""
    max_iterations: int = 30
    min_steps_before_conclude: int = 3
    explore_even_when_good: bool = True
    base_backtrack_prob: float = 0.25
    verify_final_answer: bool = True
    compare_paths: bool = True
    save_all_paths: bool = True


@dataclass
class ImprovedProblemResult:
    """Result of solving a problem with improved pipeline."""
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
    paths_explored: int = 1
    
    latency_seconds: float = 0.0
    
    verification_score: Optional[float] = None
    verification_confidence: Optional[float] = None
    
    learning_applied: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class ImprovedRLPipeline:
    """Improved pipeline with better verification, backtracking, and learning."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        config: Optional[ImprovedPipelineConfig] = None,
        results_dir: str = "data/results",
    ):
        load_dotenv()
        self.config = config or ImprovedPipelineConfig()
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        api_key = api_key or os.getenv("NVIDIA_API_KEY")
        
        self.generator = NVIDIANIMClient(api_key=api_key)
        self.prm = ImprovedPRM(
            generator_model="meta/llama-3.1-8b-instruct",
            verifier_model="meta/llama-3.3-70b-instruct",
        )
        
        action_config = ActionConfig(
            min_steps_before_conclude=self.config.min_steps_before_conclude,
        )
        
        self.action_executor = ActionExecutor(
            generator=self.generator,
            evaluator=self.prm,
            config=action_config,
        )
        
        mcts_config = ImprovedMCTSConfig(
            min_steps_before_conclude=self.config.min_steps_before_conclude,
            base_backtrack_prob=self.config.base_backtrack_prob,
            compare_alternatives=self.config.compare_paths,
        )
        
        self.mcts = ImprovedMCTSController(
            action_executor=self.action_executor,
            verifier=self.prm,
            config=mcts_config,
        )
        
        self.learner = ComparativeLearner()
        
        self._results: List[ImprovedProblemResult] = []
        
        logger.info("Improved RL Pipeline initialized")
    
    def solve(
        self,
        problem: str,
        problem_id: str = "unknown",
        ground_truth: Optional[str] = None,
    ) -> ImprovedProblemResult:
        """Solve a problem with improved reasoning."""
        
        start_time = time.time()
        
        logger.info(f"Solving problem {problem_id}: {problem[:80]}...")
        
        answer, score, reasoning_path = self.mcts.search(
            problem=problem,
            max_iterations=self.config.max_iterations,
        )
        
        verification = None
        if self.config.verify_final_answer:
            verification = self.prm.verify_answer(
                problem=problem,
                reasoning=reasoning_path,
                answer=answer,
            )
            logger.info(f"Verification: score={verification.score:.2f}, category={verification.category}")
            
            if verification.score < 0 and len(reasoning_path) > 2:
                logger.info("Answer verified as incorrect, attempting to find error...")
                error_step, error_reason = self.prm.find_error_step(
                    problem=problem,
                    reasoning=reasoning_path,
                    wrong_answer=answer,
                )
                logger.info(f"Error found at step {error_step}: {error_reason}")
        
        self.learner.record_path(
            problem=problem,
            reasoning=reasoning_path,
            answer=answer,
            prm_scores=[self.prm.evaluate_step(problem, reasoning_path[:i], s).score 
                       for i, s in enumerate(reasoning_path)],
            correct=verification.score > 0 if verification else None,
        )
        
        gen_stats = self.generator.get_stats()
        mcts_stats = self.mcts.get_stats()
        
        correct = None
        if ground_truth:
            correct = self._check_correctness(answer, ground_truth)
        
        result = ImprovedProblemResult(
            problem_id=problem_id,
            problem=problem,
            final_answer=answer,
            reasoning_path=reasoning_path,
            final_score=verification.score if verification else score,
            correct=correct,
            ground_truth=ground_truth,
            total_tokens_input=gen_stats["total_input_tokens"],
            total_tokens_output=gen_stats["total_output_tokens"],
            total_api_calls=gen_stats["total_requests"],
            num_expansions=mcts_stats.get("total_expansions", 0),
            num_reflections=mcts_stats.get("total_reflections", 0),
            num_backtracks=mcts_stats.get("total_backtracks", 0),
            paths_explored=mcts_stats.get("paths_explored", 1),
            latency_seconds=time.time() - start_time,
            verification_score=verification.score if verification else None,
            verification_confidence=verification.confidence if verification else None,
            learning_applied=len(self.learner.successful_paths) > 0,
        )
        
        self._results.append(result)
        
        logger.info(
            f"Problem {problem_id} solved: score={result.final_score:.2f}, "
            f"verified={verification.category if verification else 'N/A'}, "
            f"tokens={result.total_tokens_input + result.total_tokens_output}"
        )
        
        return result
    
    def solve_batch(
        self,
        problems: List[Dict[str, Any]],
        checkpoint_prefix: str = "improved_batch",
    ) -> List[ImprovedProblemResult]:
        """Solve multiple problems with checkpointing."""
        
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
            
            if (i + 1) % 5 == 0:
                logger.info(f"Learning summary: {self.learner.get_learning_summary()}")
        
        return results
    
    def _check_correctness(self, answer: str, ground_truth: str) -> bool:
        """Check if answer matches ground truth."""
        answer_lower = answer.lower().strip()
        truth_lower = ground_truth.lower().strip()
        
        if truth_lower in ["yes", "no"]:
            return truth_lower in answer_lower
        
        return answer_lower == truth_lower or truth_lower in answer_lower
    
    def save_results(self, filename: str = "improved_results.json"):
        """Save all results."""
        results_path = self.results_dir / filename
        
        data = {
            "timestamp": time.time(),
            "config": asdict(self.config),
            "num_problems": len(self._results),
            "results": [asdict(r) for r in self._results],
            "aggregate_stats": self._compute_aggregate_stats(),
            "learning_summary": self.learner.get_learning_summary(),
        }
        
        with open(results_path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.info(f"Results saved to {results_path}")
    
    def _compute_aggregate_stats(self) -> Dict[str, Any]:
        """Compute aggregate statistics."""
        if not self._results:
            return {}
        
        total_tokens = sum(
            r.total_tokens_input + r.total_tokens_output
            for r in self._results
        )
        
        correct_count = sum(1 for r in self._results if r.correct)
        
        return {
            "total_problems": len(self._results),
            "total_tokens": total_tokens,
            "avg_tokens_per_problem": total_tokens / len(self._results),
            "accuracy": correct_count / len(self._results) if self._results else 0,
            "avg_score": sum(r.final_score for r in self._results) / len(self._results),
            "avg_backtracks": sum(r.num_backtracks for r in self._results) / len(self._results),
            "avg_paths_explored": sum(r.paths_explored for r in self._results) / len(self._results),
        }
    
    def get_summary(self) -> str:
        """Get summary string."""
        stats = self._compute_aggregate_stats()
        
        if not stats:
            return "No results yet"
        
        return (
            f"Problems: {stats['total_problems']}\n"
            f"Accuracy: {stats['accuracy']:.1%}\n"
            f"Avg Score: {stats['avg_score']:.2f}\n"
            f"Avg Tokens: {stats['avg_tokens_per_problem']:.0f}\n"
            f"Avg Backtracks: {stats['avg_backtracks']:.1f}\n"
            f"Avg Paths Explored: {stats['avg_paths_explored']:.1f}\n"
            f"\n{self.learner.get_learning_summary()}"
        )
    
    def reset(self):
        """Reset pipeline."""
        self.generator.reset_stats()
        self.generator.clear_cache()
        self._results.clear()
        self.learner = ComparativeLearner()
    
    def close(self):
        """Close pipeline."""
        self.save_results()
        self.generator.close()
        logger.info("Improved pipeline closed")
