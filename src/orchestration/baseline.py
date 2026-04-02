"""Baseline zero-shot runner for comparison."""

import os
import json
import time
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field, asdict
from pathlib import Path
from loguru import logger

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ..generator.nim_client import NVIDIANIMClient, GenerationConfig
from ..generator.prompts import PromptBuilder
from ..generator.mock_client import MockNVIDIANIMClient


@dataclass
class BaselineConfig:
    """Configuration for baseline runner."""
    model: str = "meta/llama-3.1-8b-instruct"
    temperature: float = 0.7
    max_tokens: int = 2048
    top_p: float = 0.95
    num_runs: int = 1
    save_generations: bool = True


@dataclass
class BaselineResult:
    """Result of baseline zero-shot evaluation."""
    problem_id: str
    problem: str
    answer: str
    full_response: str
    ground_truth: Optional[str] = None
    correct: Optional[bool] = None
    
    input_tokens: int = 0
    output_tokens: int = 0
    latency_seconds: float = 0.0
    
    run_number: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaselineRunner:
    """Runner for baseline zero-shot evaluation."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        config: Optional[BaselineConfig] = None,
        results_dir: str = "data/results",
        use_mock: bool = False,
    ):
        self.config = config or BaselineConfig()
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        if use_mock or not api_key:
            self.client = MockNVIDIANIMClient(api_key="mock")
        else:
            self.client = NVIDIANIMClient(api_key=api_key)
        self._results: List[BaselineResult] = []
        
        logger.info(f"Baseline runner initialized with model: {self.config.model}")
    
    def run_single(
        self,
        problem: str,
        problem_id: str = "unknown",
        ground_truth: Optional[str] = None,
        question_type: str = "general",
    ) -> BaselineResult:
        """Run baseline on single problem.
        
        Args:
            problem: Problem statement
            problem_id: Unique identifier
            ground_truth: Optional ground truth answer
            question_type: Type of question for prompt selection
        
        Returns:
            BaselineResult
        """
        start_time = time.time()
        
        logger.info(f"Running baseline for problem {problem_id}")
        
        messages = PromptBuilder.build_baseline_prompt(problem, question_type)
        
        gen_config = GenerationConfig(
            model=self.config.model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            top_p=self.config.top_p,
        )
        
        response = self.client.generate(messages, gen_config)
        
        latency = time.time() - start_time
        
        answer = self._extract_answer(response.text)
        
        result = BaselineResult(
            problem_id=problem_id,
            problem=problem,
            answer=answer,
            full_response=response.text,
            ground_truth=ground_truth,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            latency_seconds=latency,
        )
        
        self._results.append(result)
        
        logger.info(
            f"Baseline complete: tokens={response.input_tokens + response.output_tokens}, "
            f"time={latency:.1f}s"
        )
        
        return result
    
    def _extract_answer(self, response: str) -> str:
        """Extract final answer from response."""
        markers = [
            "therefore,",
            "thus,",
            "the answer is",
            "final answer:",
            "answer:",
            "in conclusion,",
        ]
        
        response_lower = response.lower()
        
        for marker in markers:
            if marker in response_lower:
                idx = response_lower.index(marker)
                answer_part = response[idx + len(marker):].strip()
                lines = answer_part.split("\n")
                if lines:
                    return lines[0].strip()
        
        lines = response.strip().split("\n")
        non_empty = [l.strip() for l in lines if l.strip()]
        
        if non_empty:
            return non_empty[-1]
        
        return response.strip()
    
    def run_batch(
        self,
        problems: List[Dict[str, Any]],
        checkpoint_interval: int = 50,
    ) -> List[BaselineResult]:
        """Run baseline on multiple problems.
        
        Args:
            problems: List of problem dicts
            checkpoint_interval: Save checkpoint every N problems
        
        Returns:
            List of BaselineResult
        """
        results = []
        
        for i, item in enumerate(problems):
            problem_id = item.get("id", f"problem_{i}")
            problem = item.get("problem", item.get("question", ""))
            ground_truth = item.get("answer", item.get("ground_truth"))
            question_type = item.get("type", "general")
            
            result = self.run_single(
                problem=problem,
                problem_id=problem_id,
                ground_truth=ground_truth,
                question_type=question_type,
            )
            results.append(result)
            
            if (i + 1) % checkpoint_interval == 0:
                self._save_checkpoint(results, f"baseline_step_{i+1}")
        
        return results
    
    def run_with_variations(
        self,
        problem: str,
        problem_id: str = "unknown",
        num_variations: int = 3,
        temperatures: Optional[List[float]] = None,
    ) -> List[BaselineResult]:
        """Run baseline with temperature variations for self-consistency.
        
        Args:
            problem: Problem statement
            problem_id: Unique identifier
            num_variations: Number of variations to run
            temperatures: List of temperatures (default: [0.3, 0.7, 1.0])
        
        Returns:
            List of BaselineResult
        """
        temps = temperatures or [0.3, 0.7, 1.0][:num_variations]
        results = []
        
        for i, temp in enumerate(temps):
            old_temp = self.config.temperature
            self.config.temperature = temp
            
            result = self.run_single(
                problem=problem,
                problem_id=f"{problem_id}_var_{i}",
            )
            result.run_number = i
            result.metadata["temperature"] = temp
            results.append(result)
            
            self.config.temperature = old_temp
        
        return results
    
    def _save_checkpoint(self, results: List[BaselineResult], name: str):
        """Save checkpoint."""
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
    
    def save_results(self, filename: str = "baseline_results.json"):
        """Save all results."""
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
        """Compute aggregate statistics."""
        if not self._results:
            return {}
        
        total_tokens = sum(r.input_tokens + r.output_tokens for r in self._results)
        
        return {
            "total_problems": len(self._results),
            "total_tokens": total_tokens,
            "avg_tokens_per_problem": total_tokens / len(self._results),
            "avg_latency": sum(r.latency_seconds for r in self._results) / len(self._results),
            "total_correct": sum(1 for r in self._results if r.correct),
            "accuracy": sum(1 for r in self._results if r.correct) / len(self._results) if self._results else 0,
        }
    
    def get_summary(self) -> str:
        """Get summary string."""
        stats = self._compute_aggregate_stats()
        
        if not stats:
            return "No results yet"
        
        return (
            f"Problems: {stats['total_problems']}\n"
            f"Accuracy: {stats['accuracy']:.1%}\n"
            f"Avg Tokens: {stats['avg_tokens_per_problem']:.0f}\n"
            f"Avg Time: {stats['avg_latency']:.1f}s"
        )
    
    def reset(self):
        """Reset runner state."""
        self.client.reset_stats()
        self.client.clear_cache()
        self._results.clear()
    
    def close(self):
        """Close runner."""
        self.save_results()
        self.client.close()
