"""Simplified improved pipeline that works directly with improved PRM."""

import time
import random
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from dotenv import load_dotenv
from loguru import logger

from .base import BasePipeline, BaseResult, BasePipelineConfig
from ..generator.nim_client import NVIDIANIMClient, GenerationConfig
from ..generator.prompts import PromptBuilder, ReasoningContext
from ..evaluator.improved_prm import ImprovedPRM, VerificationResult, ComparativeLearner


@dataclass
class SimplifiedConfig(BasePipelineConfig):
    """Configuration for simplified pipeline."""
    max_steps: int = 5
    min_steps: int = 3
    backtrack_probability: float = 0.3
    verify_final: bool = True
    explore_alternatives: bool = True


@dataclass
class SimplifiedResult(BaseResult):
    """Result from simplified pipeline."""
    final_score: float = 0.0
    backtracks: int = 0
    paths_explored: int = 1


class SimplifiedRLPipeline(BasePipeline[SimplifiedResult, SimplifiedConfig]):
    """Simplified RL pipeline with all fixes."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        config: Optional[SimplifiedConfig] = None,
        results_dir: str = "data/results",
    ):
        load_dotenv()
        config = config or SimplifiedConfig()
        super().__init__(config=config, results_dir=results_dir)
        
        api_key = api_key or __import__('os').getenv("NVIDIA_API_KEY")
        self.generator = NVIDIANIMClient(api_key=api_key)
        self.prm = ImprovedPRM(
            generator_model="meta/llama-3.1-8b-instruct",
            verifier_model="meta/llama-3.3-70b-instruct",
        )
        self.learner = ComparativeLearner()

    def solve(self, problem: str, problem_id: str = "unknown", ground_truth: Optional[str] = None) -> SimplifiedResult:
        """Solve with exploration and verification."""

        start_time = time.time()
        logger.info(f"Solving {problem_id}: {problem[:60]}...")

        reasoning_chain = []
        best_answer = None
        best_score = -float('inf')
        backtracks = 0
        paths_explored = 0

        for step in range(self.config.max_steps):
            if step < self.config.min_steps - 1:
                current_temperature = 0.7
            else:
                current_temperature = 0.5

            messages = self._build_step_prompt(problem, reasoning_chain)
            gen_config = GenerationConfig(temperature=current_temperature, max_tokens=256)
            response = self.generator.generate(messages, gen_config)
            new_step = response.text.strip()

            if len(new_step.split()) < 5:
                continue

            eval_result = self.prm.evaluate_step(problem, reasoning_chain, new_step, step)

            reasoning_chain.append(new_step)

            if random.random() < self.config.backtrack_probability and len(reasoning_chain) > 1:
                logger.debug(f"Probabilistic backtrack at step {step}")
                backtracks += 1

                alt_reasoning = reasoning_chain[:-1]
                alt_messages = self._build_step_prompt(problem, alt_reasoning)
                alt_response = self.generator.generate(alt_messages, gen_config)
                alt_step = alt_response.text.strip()

                if len(alt_step.split()) >= 5:
                    alt_eval = self.prm.evaluate_step(problem, alt_reasoning, alt_step, step)

                    if alt_eval.score > eval_result.score:
                        reasoning_chain[-1] = alt_step
                        logger.debug(f"Alternative path better: {alt_eval.score:.2f} > {eval_result.score:.2f}")

            if step >= self.config.min_steps - 1:
                answer = self._extract_answer(problem, reasoning_chain)
                verification = self.prm.verify_answer(problem, reasoning_chain, answer)

                paths_explored += 1

                if verification.score > best_score:
                    best_score = verification.score
                    best_answer = answer

                logger.info(f"Step {step+1}: score={verification.score:.2f}, category={verification.category}")

        if best_answer is None:
            best_answer = self._extract_answer(problem, reasoning_chain)
            verification = self.prm.verify_answer(problem, reasoning_chain, best_answer)
            best_score = verification.score

        self.learner.record_path(
            problem=problem,
            reasoning=reasoning_chain,
            answer=best_answer,
            prm_scores=[0.5] * len(reasoning_chain),
            correct=best_score > 0,
        )

        correct = None
        if ground_truth:
            correct = self._check_answer(best_answer, ground_truth)

        stats = self.generator.get_stats()

        result = SimplifiedResult(
            problem_id=problem_id,
            problem=problem,
            final_answer=best_answer,
            reasoning_path=reasoning_chain,
            final_score=best_score,
            correct=correct,
            ground_truth=ground_truth,
            total_tokens=stats["total_tokens"],
            latency_seconds=time.time() - start_time,
            backtracks=backtracks,
            paths_explored=paths_explored,
        )

        self._results.append(result)
        self.generator.reset_stats()

        return result

    def _build_step_prompt(self, problem: str, previous: List[str]) -> List[Dict]:
        """Build prompt for next reasoning step."""
        context = ReasoningContext(problem=problem, previous_steps=previous)
        return PromptBuilder.build_expand_prompt(context)

    def _extract_answer(self, problem: str, reasoning: List[str]) -> str:
        """Extract final answer from reasoning."""
        messages = PromptBuilder.build_conclude_prompt(
            ReasoningContext(problem=problem, previous_steps=reasoning)
        )
        config = GenerationConfig(temperature=0.3, max_tokens=100)
        response = self.generator.generate(messages, config)
        return response.text.strip()

    def get_summary(self) -> str:
        """Get summary of results."""
        if not self._results:
            return "No results yet"

        correct = sum(1 for r in self._results if r.correct)
        avg_score = sum(r.final_score for r in self._results) / len(self._results)
        avg_tokens = sum(r.total_tokens for r in self._results) / len(self._results)
        avg_backtracks = sum(r.backtracks for r in self._results) / len(self._results)

        return (
            f"Accuracy: {correct}/{len(self._results)} ({correct/len(self._results):.1%})\n"
            f"Avg Score: {avg_score:.2f}\n"
            f"Avg Tokens: {avg_tokens:.0f}\n"
            f"Avg Backtracks: {avg_backtracks:.1f}\n"
            f"\n{self.learner.get_learning_summary()}"
        )

    def save_results(self, path: str = "simplified_results.json"):
        """Save results to file."""
        from pathlib import Path
        results_path = self.results_dir / path
        import json
        from dataclasses import asdict
        with open(results_path, "w") as f:
            json.dump([asdict(r) for r in self._results], f, indent=2, default=str)

    def reset(self):
        """Reset pipeline state."""
        self.generator.reset_stats()
        self._results.clear()
        self.learner = ComparativeLearner()

    def close(self):
        """Close pipeline."""
        self.save_results()
        self.generator.close()
