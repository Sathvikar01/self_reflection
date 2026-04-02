"""Robust RL pipeline with problem isolation and better reasoning."""

import os
import json
import time
import random
import hashlib
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generator.nim_client import NVIDIANIMClient, GenerationConfig
from generator.prompts import PromptBuilder, ReasoningContext
from evaluator.improved_prm import ImprovedPRM, VerificationResult
from rl_controller.tree import StateTree, TreeNode, NodeType
from rl_controller.actions import ActionExecutor, ActionType, ActionConfig, ActionResult

load_dotenv()


@dataclass
class RobustPipelineConfig:
    """Configuration for robust pipeline."""
    max_iterations: int = 25
    min_steps_before_conclude: int = 3
    max_steps: int = 8
    exploration_temp: float = 0.8
    conclusion_temp: float = 0.3
    use_beam_search: bool = True
    beam_width: int = 3
    verify_intermediate: bool = True
    max_backtracks: int = 5
    diversity_threshold: float = 0.7


@dataclass
class ReasoningState:
    """State for a single reasoning chain."""
    problem_hash: str
    steps: List[str]
    scores: List[float]
    is_complete: bool = False
    final_answer: str = ""
    final_score: float = 0.0


@dataclass 
class RobustProblemResult:
    """Result of solving a problem with robust pipeline."""
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
    num_backtracks: int = 0
    latency_seconds: float = 0.0
    paths_explored: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)


class ProblemContext:
    """Isolated context for each problem to prevent pollution."""
    
    def __init__(self, problem: str, problem_id: str):
        self.problem = problem
        self.problem_id = problem_id
        self.problem_hash = hashlib.md5(problem.encode()).hexdigest()[:8]
        self.reasoning_states: List[ReasoningState] = []
        self.best_state: Optional[ReasoningState] = None
        self._step_count = 0
        self._backtrack_count = 0
    
    def add_reasoning_step(self, step: str, score: float, state_idx: int = 0):
        """Add a reasoning step to a state."""
        while len(self.reasoning_states) <= state_idx:
            self.reasoning_states.append(ReasoningState(
                problem_hash=self.problem_hash,
                steps=[],
                scores=[],
            ))
        
        self.reasoning_states[state_idx].steps.append(step)
        self.reasoning_states[state_idx].scores.append(score)
        self._step_count += 1
    
    def get_current_state(self) -> Optional[ReasoningState]:
        """Get the best current state."""
        if not self.reasoning_states:
            return None
        return max(self.reasoning_states, key=lambda s: sum(s.scores) / max(1, len(s.scores)) if s.scores else 0)
    
    def record_backtrack(self):
        """Record a backtrack event."""
        self._backtrack_count += 1


class RobustRLPipeline:
    """Pipeline with problem isolation and robust reasoning."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        config: Optional[RobustPipelineConfig] = None,
        results_dir: str = "data/results",
    ):
        load_dotenv()
        self.config = config or RobustPipelineConfig()
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        api_key = api_key or os.getenv("NVIDIA_API_KEY")

        self.generator = NVIDIANIMClient(api_key=api_key)
        self.verifier = NVIDIANIMClient(api_key=api_key)
        
        self.generator_model = "meta/llama-3.1-8b-instruct"
        self.verifier_model = "meta/llama-3.3-70b-instruct"
        
        self._results: List[RobustProblemResult] = []
        self._global_step = 0
        
        logger.info("Robust RL Pipeline initialized")

    def solve(
        self,
        problem: str,
        problem_id: str = "unknown",
        ground_truth: Optional[str] = None,
    ) -> RobustProblemResult:
        """Solve a problem with isolated context."""
        
        start_time = time.time()
        
        self.generator.reset_stats()
        
        context = ProblemContext(problem, problem_id)
        
        logger.info(f"[{problem_id}] Starting reasoning for: {problem[:60]}...")
        
        best_answer, best_reasoning, best_score = self._beam_search_solve(context)
        
        verification_score = self._verify_final(problem, best_reasoning, best_answer)
        
        final_score = (best_score + verification_score) / 2
        
        gen_stats = self.generator.get_stats()
        
        correct = None
        if ground_truth:
            from evaluation.accuracy import AnswerEvaluator
            evaluator = AnswerEvaluator()
            eval_result = evaluator.evaluate(best_answer, ground_truth, problem_id)
            correct = eval_result.correct
        
        result = RobustProblemResult(
            problem_id=problem_id,
            problem=problem,
            final_answer=best_answer,
            reasoning_path=best_reasoning,
            final_score=final_score,
            correct=correct,
            ground_truth=ground_truth,
            total_tokens_input=gen_stats["total_input_tokens"],
            total_tokens_output=gen_stats["total_output_tokens"],
            total_api_calls=gen_stats["total_requests"],
            num_expansions=len(best_reasoning),
            num_backtracks=context._backtrack_count,
            latency_seconds=time.time() - start_time,
            paths_explored=len(context.reasoning_states),
        )
        
        self._results.append(result)
        
        logger.info(
            f"[{problem_id}] Complete: score={final_score:.2f}, "
            f"steps={len(best_reasoning)}, correct={correct}"
        )
        
        return result

    def _beam_search_solve(self, context: ProblemContext) -> Tuple[str, List[str], float]:
        """Solve using beam search with diverse paths."""
        
        beam: List[Tuple[List[str], float]] = [([], 0.0)]
        best_result = ("", [], 0.0)
        
        for step_idx in range(self.config.max_steps):
            candidates = []
            
            for path, path_score in beam:
                for _ in range(self.config.beam_width):
                    new_step, step_score = self._generate_next_step(
                        context.problem, path
                    )
                    
                    if new_step:
                        new_path = path + [new_step]
                        new_score = (path_score * len(path) + step_score) / (len(path) + 1)
                        candidates.append((new_path, new_score))
            
            if not candidates:
                break
            
            candidates.sort(key=lambda x: x[1], reverse=True)
            beam = candidates[:self.config.beam_width]
            
            if step_idx >= self.config.min_steps_before_conclude - 1:
                for path, path_score in beam:
                    if path_score > best_result[2]:
                        answer = self._generate_conclusion(context.problem, path)
                        answer_score = self._verify_final(context.problem, path, answer)
                        combined_score = (path_score + answer_score) / 2
                        
                        if combined_score > best_result[2]:
                            best_result = (answer, path, combined_score)
            
            if best_result[2] > 0.8:
                logger.debug(f"Early termination with high score: {best_result[2]:.2f}")
                break
            
            diverse_beam = self._diversify_beam(beam)
            beam = diverse_beam[:self.config.beam_width]
        
        if best_result[0]:
            return best_result
        
        if beam:
            best_path, best_path_score = beam[0]
            answer = self._generate_conclusion(context.problem, best_path)
            return answer, best_path, best_path_score
        
        return self._fallback_solve(context.problem)

    def _generate_next_step(self, problem: str, previous_steps: List[str]) -> Tuple[str, float]:
        """Generate the next reasoning step."""
        
        if previous_steps:
            prompt = self._build_step_prompt(problem, previous_steps)
        else:
            prompt = self._build_initial_prompt(problem)
        
        messages = [{"role": "user", "content": prompt}]
        config = GenerationConfig(
            model=self.generator_model,
            temperature=self.config.exploration_temp,
            max_tokens=256,
        )
        
        response = self.generator.generate(messages, config)
        step = response.text.strip()
        
        step_score = self._evaluate_step(problem, previous_steps, step)
        
        return step, step_score

    def _build_initial_prompt(self, problem: str) -> str:
        """Build initial reasoning prompt."""
        return f"""Solve this problem step by step. Consider ALL possibilities and edge cases.

Problem: {problem}

First step: What are the key facts or definitions needed? Consider:
- Scientific facts and exceptions
- Special conditions or edge cases
- Nuanced answers like "it depends" or "sometimes"

Provide a concrete fact-based first step."""

    def _build_step_prompt(self, problem: str, previous_steps: List[str]) -> str:
        """Build prompt for next step."""
        steps_text = "\n".join(f"{i+1}. {s}" for i, s in enumerate(previous_steps))
        
        return f"""Problem: {problem}

Reasoning so far:
{steps_text}

Next step: Continue the reasoning logically. Either:
- Make a logical inference from the facts above
- State a concrete conclusion that follows from the reasoning
- If you have enough information, provide the final answer

Provide only the next step, not the full solution."""

    def _generate_conclusion(self, problem: str, reasoning: List[str]) -> str:
        """Generate final conclusion from reasoning."""
        steps_text = "\n".join(f"{i+1}. {s}" for i, s in enumerate(reasoning))
        
        prompt = f"""Problem: {problem}

Complete reasoning:
{steps_text}

Based on the reasoning above, provide a clear final answer.

IMPORTANT CONSIDERATIONS:
1. For yes/no questions, answer "Yes" or "No" directly
2. If the answer depends on conditions (e.g., "at high temperatures", "under certain circumstances"), say "Yes, under certain conditions" or "It depends"
3. For factual questions, provide the specific answer
4. Consider scientific facts and exceptions

Final answer:"""

        messages = [{"role": "user", "content": prompt}]
        config = GenerationConfig(
            model=self.generator_model,
            temperature=self.config.conclusion_temp,
            max_tokens=100,
        )
        
        response = self.generator.generate(messages, config)
        return self._extract_answer(response.text.strip())

    def _extract_answer(self, text: str) -> str:
        """Extract clean answer from text."""
        from evaluation.accuracy import AnswerExtractor
        return AnswerExtractor.extract(text, text)

    def _evaluate_step(self, problem: str, previous: List[str], step: str) -> float:
        """Evaluate the quality of a reasoning step."""
        
        prompt = f"""Rate this reasoning step from -1.0 to 1.0.

Problem: {problem}
{"Previous steps: " + '; '.join(previous[-2:]) if previous else "This is the first step."}
Current step: {step}

Score based on:
+1.0: States a correct, relevant fact
+0.8: Makes a valid logical inference  
+0.5: Provides useful context
0.0: Neutral or meta-commentary
-0.5: Irrelevant or confusing
-1.0: Factually incorrect

Output ONLY the score number:"""

        messages = [{"role": "user", "content": prompt}]
        config = GenerationConfig(
            model=self.verifier_model,
            temperature=0.1,
            max_tokens=10,
        )
        
        response = self.verifier.generate(messages, config)
        
        import re
        numbers = re.findall(r'-?\d+\.?\d*', response.text)
        if numbers:
            try:
                score = float(numbers[0])
                return max(-1.0, min(1.0, score))
            except ValueError:
                pass
        
        return 0.5

    def _verify_final(self, problem: str, reasoning: List[str], answer: str) -> float:
        """Verify the final answer."""
        
        reasoning_text = "\n".join(f"- {s}" for s in reasoning[-5:])
        
        prompt = f"""Verify if this answer is correct.

Problem: {problem}
Reasoning summary:
{reasoning_text}
Proposed answer: {answer}

Is this answer correct and well-supported by the reasoning?
Output format:
VERDICT: [CORRECT/INCORRECT/PARTIALLY_CORRECT]
CONFIDENCE: [0.0 to 1.0]"""

        messages = [{"role": "user", "content": prompt}]
        config = GenerationConfig(
            model=self.verifier_model,
            temperature=0.1,
            max_tokens=50,
        )
        
        response = self.verifier.generate(messages, config)
        text = response.text.upper()
        
        if "CORRECT" in text and "INCORRECT" not in text:
            return 1.0
        elif "PARTIALLY" in text:
            return 0.5
        else:
            return -0.5

    def _diversify_beam(self, beam: List[Tuple[List[str], float]]) -> List[Tuple[List[str], float]]:
        """Ensure diversity in beam candidates."""
        if len(beam) <= 1:
            return beam
        
        diverse = [beam[0]]
        
        for path, score in beam[1:]:
            is_diverse = True
            for existing_path, _ in diverse:
                similarity = self._path_similarity(path, existing_path)
                if similarity > self.config.diversity_threshold:
                    is_diverse = False
                    break
            
            if is_diverse:
                diverse.append((path, score))
        
        while len(diverse) < min(len(beam), self.config.beam_width):
            diverse.append(beam[len(diverse)])
        
        return diverse

    def _path_similarity(self, path1: List[str], path2: List[str]) -> float:
        """Calculate similarity between two reasoning paths."""
        if not path1 or not path2:
            return 0.0
        
        words1 = set(' '.join(path1).lower().split())
        words2 = set(' '.join(path2).lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0

    def _fallback_solve(self, problem: str) -> Tuple[str, List[str], float]:
        """Fallback to simple solution if beam search fails."""
        
        prompt = f"""Answer this question directly and concisely.

Question: {problem}

Provide a clear answer. For yes/no questions, answer "Yes" or "No".

Answer:"""

        messages = [{"role": "user", "content": prompt}]
        config = GenerationConfig(
            model=self.generator_model,
            temperature=0.3,
            max_tokens=100,
        )
        
        response = self.generator.generate(messages, config)
        answer = self._extract_answer(response.text.strip())
        
        return answer, [answer], 0.3

    def solve_batch(
        self,
        problems: List[Dict[str, Any]],
    ) -> List[RobustProblemResult]:
        """Solve multiple problems."""
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
            
            logger.info(f"Progress: {i+1}/{len(problems)}, "
                       f"accuracy so far: {sum(1 for r in results if r.correct)}/{len(results)}")
        
        return results

    def save_results(self, filename: str = "robust_results.json"):
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
            "correct": correct_count,
        }

    def get_summary(self) -> str:
        """Get summary string."""
        stats = self._compute_aggregate_stats()
        
        if not stats:
            return "No results yet"
        
        return (
            f"Problems: {stats['total_problems']}\n"
            f"Accuracy: {stats['accuracy']:.1%} ({stats['correct']}/{stats['total_problems']})\n"
            f"Avg Score: {stats['avg_score']:.2f}\n"
            f"Avg Tokens: {stats['avg_tokens_per_problem']:.0f}"
        )

    def close(self):
        """Close pipeline."""
        self.save_results()
        self.generator.close()
        self.verifier.close()
        logger.info("Robust pipeline closed")
