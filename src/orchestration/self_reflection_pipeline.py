"""Pipeline with TRUE self-reflection - LLM critiques and corrects its own reasoning."""

import os
import time
import random
import hashlib
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger

from .base import BasePipeline, BaseResult, BasePipelineConfig
from ..generator.nim_client import NVIDIANIMClient, GenerationConfig
from evaluation.accuracy import AnswerExtractor


@dataclass
class ReflectionStep:
    """A single reflection step."""
    step_type: str
    content: str
    critique: Optional[str] = None
    is_valid: bool = True
    issues_found: List[str] = field(default_factory=list)


@dataclass
class SelfReflectionConfig(BasePipelineConfig):
    """Configuration for self-reflection pipeline."""
    min_reasoning_steps: int = 2
    max_reasoning_steps: int = 5
    reflection_depth: int = 2
    temperature_reason: float = 0.7
    temperature_reflect: float = 0.3
    temperature_conclude: float = 0.2
    force_reflection: bool = True
    enable_selective_reflection: bool = True
    confidence_threshold_skip: float = 0.9
    reflection_depths: Dict[str, int] = field(default_factory=lambda: {
        "factual": 1,
        "reasoning": 2,
        "strategic": 3
    })


@dataclass
class SelfReflectionResult(BaseResult):
    """Result from self-reflection pipeline."""
    reflections: List[str] = field(default_factory=list)
    corrections: List[str] = field(default_factory=list)
    confidence: float = 0.0
    problem_type: str = "reasoning"
    reflection_depth_used: int = 0
    early_stopped: bool = False


class SelfReflectionPipeline(BasePipeline[SelfReflectionResult, SelfReflectionConfig]):
    """Pipeline where LLM truly reflects on and corrects its own thinking."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        config: Optional[SelfReflectionConfig] = None,
        results_dir: str = "data/results",
    ):
        load_dotenv()
        config = config or SelfReflectionConfig()
        super().__init__(config=config, results_dir=results_dir)
        
        api_key = api_key or os.getenv("NVIDIA_API_KEY")
        self.generator = NVIDIANIMClient(api_key=api_key)
        self.generator_model = "meta/llama-3.1-8b-instruct"

    def solve(
        self,
        problem: str,
        problem_id: str = "unknown",
        ground_truth: Optional[str] = None,
    ) -> SelfReflectionResult:
        """Solve with TRUE self-reflection."""

        start_time = time.time()
        self.generator.reset_stats()

        logger.info(f"[{problem_id}] Starting self-reflection for: {problem[:60]}...")

        reasoning_chain: List[str] = []
        reflections: List[str] = []
        corrections: List[str] = []

        problem_type = "reasoning"
        reflection_depth_used = 0
        early_stopped = False

        if self.config.enable_selective_reflection:
            problem_type = self._classify_problem_type(problem)
            logger.info(f"[{problem_id}] Classified as: {problem_type}")

        logger.info(f"[{problem_id}] Phase 1: Initial reasoning")
        reasoning_chain = self._generate_initial_reasoning(problem)

        baseline_confidence = self._calculate_baseline_confidence(reasoning_chain)
        logger.info(f"[{problem_id}] Baseline confidence: {baseline_confidence:.2f}")

        if self.config.enable_selective_reflection and baseline_confidence > self.config.confidence_threshold_skip:
            logger.info(f"[{problem_id}] High confidence ({baseline_confidence:.2f} > {self.config.confidence_threshold_skip}), skipping reflection")
            early_stopped = True
            reflection_depth_used = 0
        else:
            effective_reflection_depth = self.config.reflection_depth
            if self.config.enable_selective_reflection:
                effective_reflection_depth = self.config.reflection_depths.get(problem_type, self.config.reflection_depth)
            logger.info(f"[{problem_id}] Using reflection depth: {effective_reflection_depth} for type: {problem_type}")

            logger.info(f"[{problem_id}] Phase 2: Self-reflection")
            for i in range(effective_reflection_depth):
                reflection_depth_used = i + 1
                reflection_result = self._self_reflect(problem, reasoning_chain, i)

                if reflection_result["issues_found"]:
                    reflections.append(reflection_result["reflection"])
                    correction = self._apply_correction(
                        problem,
                        reasoning_chain,
                        reflection_result["issues_found"]
                    )
                    corrections.append(correction)
                    reasoning_chain = self._update_reasoning(reasoning_chain, correction)
                else:
                    reflections.append(reflection_result["reflection"])
                    if self.config.enable_selective_reflection and i == 0:
                        logger.info(f"[{problem_id}] No issues found in first pass, early stopping")
                        early_stopped = True
                        break

            logger.info(f"[{problem_id}] Phase 3: Final self-critique")
            final_critique = self._final_self_critique(problem, reasoning_chain)

            if final_critique["needs_revision"]:
                corrections.append(final_critique["correction"])
                reasoning_chain[-1] = final_critique["corrected_step"]

        logger.info(f"[{problem_id}] Phase 4: Final answer")
        final_answer = self._generate_final_answer(problem, reasoning_chain)

        correct = None
        if ground_truth:
            correct = self._check_answer(final_answer, ground_truth)

        gen_stats = self.generator.get_stats()

        result = SelfReflectionResult(
            problem_id=problem_id,
            problem=problem,
            final_answer=final_answer,
            reasoning_path=reasoning_chain,
            reflections=reflections,
            corrections=corrections,
            confidence=self._calculate_confidence(reasoning_chain, reflections),
            correct=correct,
            ground_truth=ground_truth,
            total_tokens=gen_stats["total_input_tokens"] + gen_stats["total_output_tokens"],
            latency_seconds=time.time() - start_time,
            problem_type=problem_type,
            reflection_depth_used=reflection_depth_used,
            early_stopped=early_stopped,
        )

        self._results.append(result)

        logger.info(
            f"[{problem_id}] Complete: answer={final_answer[:30]}..., "
            f"correct={correct}, reflections={len(reflections)}, corrections={len(corrections)}, "
            f"type={problem_type}, depth_used={reflection_depth_used}, early_stopped={early_stopped}"
        )

        return result

    def _generate_initial_reasoning(self, problem: str) -> List[str]:
        """Generate initial reasoning chain."""
        reasoning_chain = []
        for _ in range(self.config.min_reasoning_steps):
            prompt = self._build_reasoning_prompt(problem, reasoning_chain)
            messages = [{"role": "user", "content": prompt}]
            config = GenerationConfig(
                model=self.generator_model,
                temperature=self.config.temperature_reason,
                max_tokens=256,
            )
            response = self.generator.generate(messages, config)
            step = response.text.strip()
            if step:
                reasoning_chain.append(step)
        return reasoning_chain

    def _build_reasoning_prompt(self, problem: str, previous: List[str]) -> str:
        """Build prompt for next reasoning step."""
        if previous:
            steps_text = "\n".join(f"{i+1}. {s}" for i, s in enumerate(previous))
            return f"""Problem: {problem}

Reasoning so far:
{steps_text}

Provide the next logical reasoning step. Focus on facts, logical inferences, or calculations."""
        return f"""Solve this problem step by step.

Problem: {problem}

Provide the first reasoning step. Focus on key facts or definitions."""

    def _self_reflect(self, problem: str, reasoning: List[str], iteration: int) -> Dict[str, Any]:
        """Perform self-reflection on reasoning."""
        reasoning_text = "\n".join(f"{i+1}. {s}" for i, s in enumerate(reasoning))
        
        prompt = f"""Review this reasoning for errors, flaws, or missing considerations.

Problem: {problem}

Reasoning:
{reasoning_text}

Analyze each step critically:
1. Are there any factual errors?
2. Are there logical fallacies?
3. Are there edge cases not considered?
4. Is the reasoning complete?

If you find issues, list them clearly.
Format:
ISSUES:
- [issue 1]
- [issue 2]
...
ANALYSIS: [your analysis]"""

        messages = [{"role": "user", "content": prompt}]
        config = GenerationConfig(
            model=self.generator_model,
            temperature=self.config.temperature_reflect,
            max_tokens=512,
        )
        response = self.generator.generate(messages, config)
        
        issues = []
        analysis = response.text
        if "ISSUES:" in analysis:
            issues_section = analysis.split("ISSUES:")[1].split("ANALYSIS:")[0]
            issues = [line.strip("- ").strip() for line in issues_section.strip().split("\n") if line.strip().startswith("-")]
        
        return {"reflection": response.text, "issues_found": issues}

    def _apply_correction(self, problem: str, reasoning: List[str], issues: List[str]) -> str:
        """Apply correction based on identified issues."""
        reasoning_text = "\n".join(f"{i+1}. {s}" for i, s in enumerate(reasoning))
        issues_text = "\n".join(f"- {issue}" for issue in issues)
        
        prompt = f"""Correct the reasoning based on identified issues.

Problem: {problem}

Original reasoning:
{reasoning_text}

Issues to address:
{issues_text}

Provide a corrected reasoning step that addresses these issues."""

        messages = [{"role": "user", "content": prompt}]
        config = GenerationConfig(
            model=self.generator_model,
            temperature=self.config.temperature_reason,
            max_tokens=256,
        )
        response = self.generator.generate(messages, config)
        return response.text.strip()

    def _update_reasoning(self, reasoning: List[str], correction: str) -> List[str]:
        """Update reasoning chain with correction."""
        return reasoning + [correction]

    def _final_self_critique(self, problem: str, reasoning: List[str]) -> Dict[str, Any]:
        """Final self-critique before answering."""
        reasoning_text = "\n".join(f"{i+1}. {s}" for i, s in enumerate(reasoning))
        
        prompt = f"""Final review before concluding.

Problem: {problem}

Reasoning:
{reasoning_text}

Is this reasoning sound and complete? If there's a critical error in the last step, provide a corrected version.
Format:
NEEDS_REVISION: [yes/no]
CORRECTED_STEP: [corrected version if needed]"""

        messages = [{"role": "user", "content": prompt}]
        config = GenerationConfig(
            model=self.generator_model,
            temperature=self.config.temperature_reflect,
            max_tokens=256,
        )
        response = self.generator.generate(messages, config)
        
        needs_revision = "NEEDS_REVISION: yes" in response.text.upper()
        corrected_step = ""
        if needs_revision and "CORRECTED_STEP:" in response.text:
            corrected_step = response.text.split("CORRECTED_STEP:")[1].strip().split("\n")[0].strip()
        
        return {
            "needs_revision": needs_revision,
            "correction": corrected_step,
            "corrected_step": corrected_step or reasoning[-1] if reasoning else ""
        }

    def _generate_final_answer(self, problem: str, reasoning: List[str]) -> str:
        """Generate final answer from reasoning."""
        reasoning_text = "\n".join(f"{i+1}. {s}" for i, s in enumerate(reasoning))
        
        prompt = f"""Based on the reasoning, provide the final answer.

Problem: {problem}

Reasoning:
{reasoning_text}

Provide ONLY the final answer, concise and clear."""

        messages = [{"role": "user", "content": prompt}]
        config = GenerationConfig(
            model=self.generator_model,
            temperature=self.config.temperature_conclude,
            max_tokens=100,
        )
        response = self.generator.generate(messages, config)
        return AnswerExtractor.extract(response.text, problem)

    def _classify_problem_type(self, problem: str) -> str:
        """Classify problem type for adaptive reflection."""
        problem_lower = problem.lower()
        
        factual_keywords = ["what is", "who", "when", "where", "how many", "define"]
        strategic_keywords = ["should", "would", "could", "might", "best", "worst", "compare", "evaluate"]
        
        if any(kw in problem_lower for kw in factual_keywords):
            return "factual"
        elif any(kw in problem_lower for kw in strategic_keywords):
            return "strategic"
        return "reasoning"

    def _calculate_baseline_confidence(self, reasoning: List[str]) -> float:
        """Calculate baseline confidence in reasoning."""
        if not reasoning:
            return 0.0
        
        total_words = sum(len(step.split()) for step in reasoning)
        if total_words < 20:
            return 0.3
        elif total_words < 50:
            return 0.5
        elif total_words < 100:
            return 0.7
        return 0.85

    def _calculate_confidence(self, reasoning: List[str], reflections: List[str]) -> float:
        """Calculate final confidence score."""
        base_confidence = self._calculate_baseline_confidence(reasoning)
        
        if not reflections:
            return base_confidence
        
        correction_penalty = len(reflections) * 0.1
        return max(0.1, min(1.0, base_confidence - correction_penalty + 0.2))

    def _check_answer(self, answer: str, ground_truth: str) -> bool:
        """Check if answer matches ground truth."""
        answer_lower = answer.lower().strip()
        truth_lower = ground_truth.lower().strip()
        
        if truth_lower in ["yes", "no"]:
            return truth_lower in answer_lower
        
        return truth_lower in answer_lower or answer_lower == truth_lower

    def save_results(self, filename: str = "self_reflection_results.json"):
        """Save all results."""
        super().save_results(filename)

    def reset(self):
        """Reset pipeline state."""
        self.generator.reset_stats()
        self._results.clear()

    def close(self):
        """Close pipeline."""
        self.save_results()
        self.generator.close()
