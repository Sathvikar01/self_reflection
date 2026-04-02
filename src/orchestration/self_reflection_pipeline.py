"""Pipeline with TRUE self-reflection - LLM critiques and corrects its own reasoning."""

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
from evaluation.accuracy import AnswerExtractor

load_dotenv()


@dataclass
class ReflectionStep:
    """A single reflection step."""
    step_type: str  # "reasoning", "reflection", "correction"
    content: str
    critique: Optional[str] = None
    is_valid: bool = True
    issues_found: List[str] = field(default_factory=list)


@dataclass
class SelfReflectionConfig:
    """Configuration for self-reflection pipeline."""
    max_iterations: int = 8
    min_reasoning_steps: int = 2
    max_reasoning_steps: int = 5
    reflection_depth: int = 2 # How many times to reflect
    temperature_reason: float = 0.7
    temperature_reflect: float = 0.3
    temperature_conclude: float = 0.2
    force_reflection: bool = True # Always reflect before concluding
    enable_selective_reflection: bool = True
    confidence_threshold_skip: float = 0.9
    reflection_depths: Dict[str, int] = field(default_factory=lambda: {
        "factual": 1,
        "reasoning": 2,
        "strategic": 3
    })


@dataclass
class SelfReflectionResult:
    """Result from self-reflection pipeline."""
    problem_id: str
    problem: str
    final_answer: str
    reasoning_chain: List[str]
    reflections: List[str] # Self-reflections made
    corrections: List[str] # Self-corrections applied
    confidence: float
    correct: Optional[bool] = None
    ground_truth: Optional[str] = None
    total_tokens: int = 0
    latency_seconds: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    problem_type: str = "reasoning"
    reflection_depth_used: int = 0
    early_stopped: bool = False


class SelfReflectionPipeline:
    """Pipeline where LLM truly reflects on and corrects its own thinking."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        config: Optional[SelfReflectionConfig] = None,
        results_dir: str = "data/results",
    ):
        load_dotenv()
        self.config = config or SelfReflectionConfig()
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        api_key = api_key or os.getenv("NVIDIA_API_KEY")
        self.generator = NVIDIANIMClient(api_key=api_key)
        self.generator_model = "meta/llama-3.1-8b-instruct"
        
        self._results: List[SelfReflectionResult] = []

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

        # Step 1: Generate initial reasoning
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

            # Step 2: Self-reflect on reasoning (this is the KEY part!)
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

        # Step 3: Final self-critique before answering
        logger.info(f"[{problem_id}] Phase 3: Final self-critique")
        final_critique = self._final_self_critique(problem, reasoning_chain)

        if final_critique["needs_revision"]:
            corrections.append(final_critique["correction"])
            reasoning_chain[-1] = final_critique["corrected_step"]

        # Step 4: Generate final answer
        logger.info(f"[{problem_id}] Phase 4: Final answer")
        final_answer = self._generate_final_answer(problem, reasoning_chain)

        # Evaluate
        correct = None
        if ground_truth:
            correct = self._check_answer(final_answer, ground_truth)

        gen_stats = self.generator.get_stats()

        result = SelfReflectionResult(
            problem_id=problem_id,
            problem=problem,
            final_answer=final_answer,
            reasoning_chain=reasoning_chain,
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
        """Generate initial reasoning steps with self-awareness."""
        
        steps = []
        
        # First step: Understand what we're being asked
        prompt = f"""You are solving a problem. Think step by step and be self-aware about your reasoning.

PROBLEM: {problem}

STEP 1: First, understand what the question is really asking. What do we need to determine?

Write your first reasoning step. Be specific and factual. Start with "First, I need to understand..." """

        messages = [{"role": "user", "content": prompt}]
        config = GenerationConfig(
            model=self.generator_model,
            temperature=self.config.temperature_reason,
            max_tokens=200,
        )
        
        response = self.generator.generate(messages, config)
        steps.append(response.text.strip())
        
        # Second step: Gather key facts
        prompt = f"""PROBLEM: {problem}

Current reasoning: {steps[0]}

STEP 2: What are the key facts I need to consider? State concrete facts, definitions, or principles that are relevant.

Write your second reasoning step. Start with "The key facts are..." """

        messages = [{"role": "user", "content": prompt}]
        response = self.generator.generate(messages, config)
        steps.append(response.text.strip())
        
        # Third step: Apply logic
        prompt = f"""PROBLEM: {problem}

Current reasoning:
1. {steps[0]}
2. {steps[1]}

STEP 3: Now apply logic to these facts. What conclusion follows?

Write your third reasoning step. Start with "Therefore..." """

        messages = [{"role": "user", "content": prompt}]
        response = self.generator.generate(messages, config)
        steps.append(response.text.strip())
        
        return steps

    def _classify_problem_type(self, problem: str) -> str:
        prompt = f"""Classify this question into ONE category:

PROBLEM: {problem}

CATEGORIES:
- factual: Questions about scientific facts, definitions, measurements (e.g., "Is the sun brighter than a light bulb?")
- reasoning: Questions requiring logic, cause-effect, or multi-step thinking (e.g., "Do hamsters provide food for any animals?")
- strategic: Questions about optimal decisions, games, planning (e.g., chess questions, "What's the best move?")

Answer with just the category name (factual, reasoning, or strategic):

CATEGORY:"""

        messages = [{"role": "user", "content": prompt}]
        config = GenerationConfig(
            model=self.generator_model,
            temperature=0.1,
            max_tokens=20,
        )

        response = self.generator.generate(messages, config)
        classification = response.text.strip().lower()

        for ptype in ["factual", "reasoning", "strategic"]:
            if ptype in classification:
                return ptype
        return "reasoning"

    def _calculate_baseline_confidence(self, reasoning: List[str]) -> float:
        if not reasoning:
            return 0.5

        confidence = 0.5

        reasoning_text = " ".join(reasoning).lower()

        certainty_markers = ["definitely", "clearly", "obviously", "certainly", "undoubtedly", "always", "never"]
        uncertainty_markers = ["might", "maybe", "perhaps", "possibly", "uncertain", "unclear", "might not"]

        certainty_count = sum(1 for m in certainty_markers if m in reasoning_text)
        uncertainty_count = sum(1 for m in uncertainty_markers if m in reasoning_text)

        confidence += 0.05 * certainty_count
        confidence -= 0.08 * uncertainty_count

        if len(reasoning) >= 3:
            confidence += 0.15

        for step in reasoning:
            if any(word in step.lower() for word in ["because", "therefore", "since", "thus"]):
                confidence += 0.05

        return max(0.3, min(1.0, confidence))

    def _self_reflect(self, problem: str, reasoning: List[str], depth: int) -> Dict:
        """CRITICAL: LLM reflects on its own reasoning and finds flaws."""
        
        reasoning_text = "\n".join(f"{i+1}. {s}" for i, s in enumerate(reasoning))
        
        prompt = f"""You are reflecting on your own reasoning to find potential errors. Be honest and critical.

PROBLEM: {problem}

YOUR CURRENT REASONING:
{reasoning_text}

SELF-REFLECTION: Carefully examine each step of your reasoning above. Ask yourself:
1. Is each fact I stated actually correct? Verify against known scientific facts.
2. Is my logic sound? Does the conclusion really follow from the premises?
3. Did I consider edge cases or exceptions?
4. Am I making any hidden assumptions that might be wrong?
5. Could there be a simpler explanation I'm missing?

Write a critical self-reflection. If you find any issues, list them clearly.
Format your reflection as:
REFLECTION: [your critical analysis]
ISSUES_FOUND: [list each issue or write "none" if no issues]
ISSUE_1: [description]
ISSUE_2: [description] (if applicable)
"""

        messages = [{"role": "user", "content": prompt}]
        config = GenerationConfig(
            model=self.generator_model,
            temperature=self.config.temperature_reflect,
            max_tokens=300,
        )
        
        response = self.generator.generate(messages, config)
        reflection_text = response.text.strip()
        
        # Parse issues
        issues = []
        if "ISSUE_" in reflection_text.upper():
            for line in reflection_text.split("\n"):
                if "ISSUE_" in line.upper() and ":" in line:
                    issue = line.split(":", 1)[1].strip()
                    if issue.lower() != "none" and issue:
                        issues.append(issue)
        
        return {
            "reflection": reflection_text,
            "issues_found": issues,
        }

    def _apply_correction(
        self, 
        problem: str, 
        reasoning: List[str], 
        issues: List[str]
    ) -> str:
        """Apply self-correction based on identified issues."""
        
        reasoning_text = "\n".join(f"{i+1}. {s}" for i, s in enumerate(reasoning))
        issues_text = "\n".join(f"- {issue}" for issue in issues)
        
        prompt = f"""You found issues in your reasoning. Now correct them.

PROBLEM: {problem}

YOUR PREVIOUS REASONING:
{reasoning_text}

ISSUES YOU IDENTIFIED:
{issues_text}

CORRECTION: Fix each issue you found. Provide corrected reasoning steps that address these problems.
Write the corrected version, explaining what was wrong and how you're fixing it.

CORRECTION:"""

        messages = [{"role": "user", "content": prompt}]
        config = GenerationConfig(
            model=self.generator_model,
            temperature=self.config.temperature_reflect,
            max_tokens=300,
        )
        
        response = self.generator.generate(messages, config)
        return response.text.strip()

    def _update_reasoning(self, reasoning: List[str], correction: str) -> List[str]:
        """Update reasoning chain based on correction."""
        
        # Extract corrected steps from correction text
        if len(reasoning) >= 3:
            # Replace the last step with corrected conclusion
            prompt = f"""Based on this correction:
{correction}

Write a single corrected conclusion step (one sentence):
Therefore,"""
            
            messages = [{"role": "user", "content": prompt}]
            config = GenerationConfig(
                model=self.generator_model,
                temperature=self.config.temperature_reflect,
                max_tokens=100,
            )
            
            response = self.generator.generate(messages, config)
            reasoning[-1] = "Therefore, " + response.text.strip().replace("Therefore, ", "")
        
        return reasoning

    def _final_self_critique(self, problem: str, reasoning: List[str]) -> Dict:
        """Final self-critique before generating answer."""
        
        reasoning_text = "\n".join(f"{i+1}. {s}" for i, s in enumerate(reasoning))
        
        prompt = f"""Before finalizing your answer, do one last self-check.

PROBLEM: {problem}

REASONING:
{reasoning_text}

FINAL SELF-CRITIQUE:
1. Does my reasoning directly answer the question asked?
2. Is my conclusion unambiguous and clear?
3. For yes/no questions, did I provide a direct yes or no?

If everything is correct, write "VERIFIED".
If changes are needed, write "NEEDS_REVISION" and explain what needs to change.

VERDICT:"""

        messages = [{"role": "user", "content": prompt}]
        config = GenerationConfig(
            model=self.generator_model,
            temperature=0.1,
            max_tokens=100,
        )
        
        response = self.generator.generate(messages, config)
        critique_text = response.text.strip()
        
        needs_revision = "NEEDS_REVISION" in critique_text.upper()
        
        return {
            "needs_revision": needs_revision,
            "critique": critique_text,
            "correction": critique_text if needs_revision else "",
            "corrected_step": "",
        }

    def _generate_final_answer(self, problem: str, reasoning: List[str]) -> str:
        """Generate final answer after reflection."""
        
        reasoning_text = "\n".join(f"{i+1}. {s}" for i, s in enumerate(reasoning))
        
        prompt = f"""Based on your reflected reasoning, provide a final answer.

PROBLEM: {problem}

REFLECTED REASONING (after self-correction):
{reasoning_text}

Now provide your final answer. Be direct and clear.
- For yes/no questions: answer "Yes" or "No"
- For factual questions: state the specific answer
- If conditions matter: mention them

FINAL ANSWER:"""

        messages = [{"role": "user", "content": prompt}]
        config = GenerationConfig(
            model=self.generator_model,
            temperature=self.config.temperature_conclude,
            max_tokens=50,
        )
        
        response = self.generator.generate(messages, config)
        return AnswerExtractor.extract(response.text.strip(), response.text)

    def _check_answer(self, answer: str, ground_truth: str) -> bool:
        """Check if answer matches ground truth."""
        answer_lower = answer.lower().strip()
        truth_lower = ground_truth.lower().strip()
        
        if truth_lower in ["yes", "no"]:
            return truth_lower in answer_lower
        
        return truth_lower in answer_lower or answer_lower == truth_lower

    def _calculate_confidence(
        self, 
        reasoning: List[str], 
        reflections: List[str]
    ) -> float:
        """Calculate confidence based on reasoning and reflections."""
        base_confidence = 0.7
        
        # More reasoning steps = higher confidence
        if len(reasoning) >= 3:
            base_confidence += 0.1
        
        # Reflections found no issues = higher confidence
        if reflections and "none" in reflections[-1].lower():
            base_confidence += 0.1
        
        # Corrections applied = lower confidence (had issues)
        base_confidence -= 0.05 * len([r for r in reflections if "issue" in r.lower()])
        
        return max(0.3, min(1.0, base_confidence))

    def solve_batch(
        self,
        problems: List[Dict[str, Any]],
    ) -> List[SelfReflectionResult]:
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
            
            # Print detailed output showing reflection
            print(f"\n{'='*60}")
            print(f"Problem {i+1}: {problem}")
            print(f"{'='*60}")
            print(f"Reasoning chain:")
            for j, step in enumerate(result.reasoning_chain):
                print(f"  {j+1}. {step[:100]}...")
            if result.reflections:
                print(f"\nSelf-reflections made: {len(result.reflections)}")
                for j, ref in enumerate(result.reflections):
                    print(f"  Reflection {j+1}: {ref[:100]}...")
            if result.corrections:
                print(f"\nSelf-corrections applied: {len(result.corrections)}")
            print(f"\nFinal answer: {result.final_answer}")
            print(f"Correct: {result.correct}")
            print(f"{'='*60}")
        
        return results

    def save_results(self, filename: str = "self_reflection_results.json"):
        """Save results."""
        results_path = self.results_dir / filename
        
        correct_count = sum(1 for r in self._results if r.correct)

        data = {
            "timestamp": time.time(),
            "config": asdict(self.config),
            "num_problems": len(self._results),
            "accuracy": correct_count / len(self._results) if self._results else 0,
            "correct": correct_count,
            "results": [asdict(r) for r in self._results],
            "selective_reflection_stats": self._get_selective_stats(),
        }
        
        with open(results_path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.info(f"Results saved to {results_path}")

    def close(self):
        """Close pipeline."""
        self.save_results()
        self.generator.close()
        logger.info("Self-reflection pipeline closed")

    def _get_selective_stats(self) -> Dict[str, Any]:
        stats = {
            "by_type": {},
            "early_stops": 0,
            "total_reflection_passes_saved": 0,
        }

        type_counts: Dict[str, int] = {}
        type_correct: Dict[str, int] = {}
        type_avg_depth: Dict[str, List[int]] = {}
        type_early_stopped: Dict[str, int] = {}

        for r in self._results:
            ptype = r.problem_type
            type_counts[ptype] = type_counts.get(ptype, 0) + 1
            if r.correct:
                type_correct[ptype] = type_correct.get(ptype, 0) + 1
            if ptype not in type_avg_depth:
                type_avg_depth[ptype] = []
            type_avg_depth[ptype].append(r.reflection_depth_used)
            if r.early_stopped:
                type_early_stopped[ptype] = type_early_stopped.get(ptype, 0) + 1
                stats["early_stops"] += 1
                default_depth = self.config.reflection_depths.get(ptype, self.config.reflection_depth)
                stats["total_reflection_passes_saved"] += default_depth - r.reflection_depth_used

        for ptype in type_counts:
            stats["by_type"][ptype] = {
                "count": type_counts[ptype],
                "accuracy": type_correct.get(ptype, 0) / type_counts[ptype],
                "avg_reflection_depth": sum(type_avg_depth[ptype]) / len(type_avg_depth[ptype]),
                "early_stops": type_early_stopped.get(ptype, 0),
            }

        return stats
