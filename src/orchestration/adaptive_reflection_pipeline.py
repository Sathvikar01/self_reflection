"""
Adaptive Self-Reflection Pipeline with Rollback and Overfitting Prevention.

Key Features:
1. Query complexity analysis to determine initial reflection depth
2. Adaptive reflection: increases depth if confidence is low
3. Rollback: reverts to previous best answer if reflection degrades quality
4. Overfitting prevention: cross-validation and early stopping
"""

import os
import json
import time
import hashlib
import re
import statistics as stats_module
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger

from ..generator.nim_client import NVIDIANIMClient, GenerationConfig
from ..utils.unified_extractor import UnifiedAnswerExtractor
from ..utils.complexity import QueryComplexityAnalyzer, ComplexityScore
from .base import BasePipeline, BaseResult, BasePipelineConfig

load_dotenv()


@dataclass
class ReflectionCheckpoint:
    """Checkpoint for rollback capability."""
    step: int
    reasoning_chain: List[str]
    answer: str
    confidence: float
    issues_found: List[str]
    timestamp: float


@dataclass
class AdaptiveReflectionConfig(BasePipelineConfig):
    """Configuration for adaptive self-reflection."""
    # Complexity thresholds
    low_complexity_threshold: float = 0.3
    high_complexity_threshold: float = 0.7

    # Reflection bounds
    min_reflections: int = 1
    max_reflections: int = 5

    # Adaptive behavior
    confidence_threshold_increase: float = 0.7  # Increase depth if confidence below
    confidence_threshold_stop: float = 0.9  # Stop if confidence above
    degradation_threshold: float = 0.1  # Rollback if confidence drops by this much

    # Overfitting prevention
    enable_cross_validation: bool = True
    validation_samples: int = 3  # Number of samples for cross-validation
    variance_threshold: float = 0.2  # High variance = overfitting risk

    # Early stopping
    early_stopping_patience: int = 2  # Stop if no improvement for N reflections

    # Temperatures
    temperature_reason: float = 0.7
    temperature_reflect: float = 0.3
    temperature_conclude: float = 0.2


@dataclass
class AdaptiveReflectionResult(BaseResult):
    """Result from adaptive self-reflection."""

    reflections: List[str] = field(default_factory=list)
    corrections: List[str] = field(default_factory=list)
    confidence: float = 0.0
    reasoning_chain: List[str] = field(default_factory=list)

    # Adaptive metrics
    complexity_score: float = 0.0
    recommended_depth: int = 0
    actual_depth: int = 0
    rolled_back: bool = False
    rollback_step: int = 0
    overfitting_detected: bool = False

    # Cross-validation metrics
    cv_answers: List[str] = field(default_factory=list)
    cv_confidence_std: float = 0.0


class AdaptiveReflectionPipeline(BasePipeline[AdaptiveReflectionResult, AdaptiveReflectionConfig]):
    """Pipeline with adaptive self-reflection, rollback, and overfitting prevention."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        config: Optional[AdaptiveReflectionConfig] = None,
        results_dir: str = "data/results",
    ):
        load_dotenv()
        config = config or AdaptiveReflectionConfig()
        super().__init__(config=config, results_dir=results_dir)

        api_key = api_key or os.getenv("NVIDIA_API_KEY")
        self.generator = NVIDIANIMClient(api_key=api_key)
        self.generator_model = "meta/llama-3.1-8b-instruct"

        self.complexity_analyzer = QueryComplexityAnalyzer()

        logger.info("Adaptive Reflection Pipeline initialized")
    
    def solve(
        self,
        problem: str,
        problem_id: str = "unknown",
        ground_truth: Optional[str] = None,
    ) -> AdaptiveReflectionResult:
        """Solve with adaptive self-reflection."""
        start_time = time.time()
        start_tokens = self.generator.get_stats()["total_tokens"]
        
        logger.info(f"[{problem_id}] Starting adaptive reflection for: {problem[:60]}...")
        
        # Step 1: Analyze query complexity
        complexity = self.complexity_analyzer.analyze(problem)
        logger.info(f"[{problem_id}] Complexity: {complexity.overall_score:.2f}, recommended depth: {complexity.recommended_depth}")
        
        # Step 2: Initialize tracking
        checkpoints: List[ReflectionCheckpoint] = []
        reasoning_chain: List[str] = []
        all_reflections: List[str] = []
        all_corrections: List[str] = []
        
        # Step 3: Generate initial reasoning
        initial_reasoning = self._generate_initial_reasoning(problem)
        reasoning_chain.extend(initial_reasoning)
        
        # Step 4: Get initial answer and confidence
        initial_answer, initial_confidence = self._get_answer_with_confidence(
            problem, reasoning_chain
        )
        
        checkpoints.append(ReflectionCheckpoint(
            step=0,
            reasoning_chain=reasoning_chain.copy(),
            answer=initial_answer,
            confidence=initial_confidence,
            issues_found=[],
            timestamp=time.time()
        ))
        
        logger.info(f"[{problem_id}] Initial confidence: {initial_confidence:.2f}")
        
        # Step 5: Adaptive reflection loop
        current_depth = 0
        max_depth = min(complexity.recommended_depth + 1, self.config.max_reflections)
        no_improvement_count = 0
        best_checkpoint = checkpoints[0]
        rolled_back = False
        rollback_step = 0
        
        while current_depth < max_depth:
            current_depth += 1
            
            # Check if we should stop early
            if initial_confidence >= self.config.confidence_threshold_stop:
                logger.info(f"[{problem_id}] High confidence ({initial_confidence:.2f}), stopping early")
                break
            
            # Perform reflection
            reflection_result = self._reflect_and_correct(problem, reasoning_chain, current_depth)
            
            if reflection_result["issues_found"]:
                all_reflections.append(reflection_result["reflection"])
                all_corrections.append(reflection_result["correction"])
                reasoning_chain = reflection_result["updated_reasoning"]
            else:
                all_reflections.append(reflection_result["reflection"])
                # No issues found, can early stop
                if current_depth >= complexity.recommended_depth:
                    logger.info(f"[{problem_id}] No issues found, early stopping at depth {current_depth}")
                    break
            
            # Get new answer and confidence
            new_answer, new_confidence = self._get_answer_with_confidence(
                problem, reasoning_chain
            )
            
            # Create checkpoint
            checkpoint = ReflectionCheckpoint(
                step=current_depth,
                reasoning_chain=reasoning_chain.copy(),
                answer=new_answer,
                confidence=new_confidence,
                issues_found=reflection_result["issues_found"],
                timestamp=time.time()
            )
            checkpoints.append(checkpoint)
            
            logger.info(f"[{problem_id}] Depth {current_depth}: confidence {new_confidence:.2f}")
            
            # Check for degradation (rollback condition)
            confidence_change = new_confidence - best_checkpoint.confidence
            if confidence_change < -self.config.degradation_threshold:
                logger.warning(f"[{problem_id}] Confidence degraded by {abs(confidence_change):.2f}, rolling back")
                rolled_back = True
                rollback_step = current_depth
                break
            
            # Track best checkpoint
            if new_confidence > best_checkpoint.confidence:
                best_checkpoint = checkpoint
                no_improvement_count = 0
            else:
                no_improvement_count += 1
            
            # Early stopping if no improvement
            if no_improvement_count >= self.config.early_stopping_patience:
                logger.info(f"[{problem_id}] No improvement for {no_improvement_count} reflections, stopping")
                break
            
            # Adaptive depth increase
            if new_confidence < self.config.confidence_threshold_increase:
                if current_depth < self.config.max_reflections:
                    logger.info(f"[{problem_id}] Low confidence, continuing reflection")
                    max_depth = min(max_depth + 1, self.config.max_reflections)
        
        # Step 6: Use best checkpoint (rollback if needed)
        if rolled_back:
            final_answer = best_checkpoint.answer
            final_confidence = best_checkpoint.confidence
            reasoning_chain = best_checkpoint.reasoning_chain
        else:
            final_answer = checkpoints[-1].answer
            final_confidence = checkpoints[-1].confidence
        
        # Step 7: Cross-validation for overfitting detection
        cv_answers = []
        cv_confidences = []
        overfitting_detected = False
        cv_confidence_std = 0.0
        
        if self.config.enable_cross_validation:
            cv_answers, cv_confidences = self._cross_validate(problem, reasoning_chain)
            
            if len(cv_confidences) > 1:
                import statistics
                cv_confidence_std = statistics.stdev(cv_confidences)
                
                if cv_confidence_std > self.config.variance_threshold:
                    overfitting_detected = True
                    logger.warning(f"[{problem_id}] Overfitting detected: CV std = {cv_confidence_std:.3f}")
                
                # Use majority vote if overfitting
                if overfitting_detected:
                    from collections import Counter
                    answer_counts = Counter(cv_answers)
                    final_answer = answer_counts.most_common(1)[0][0]
                    logger.info(f"[{problem_id}] Using majority vote answer: {final_answer}")
        
        # Step 8: Check correctness
        correct = None
        if ground_truth:
            correct = self._check_answer(final_answer, ground_truth)
        
        latency = time.time() - start_time
        total_tokens = self.generator.get_stats()["total_tokens"] - start_tokens
        
        result = AdaptiveReflectionResult(
            problem_id=problem_id,
            problem=problem,
            final_answer=final_answer,
            reasoning_chain=reasoning_chain,
            reflections=all_reflections,
            corrections=all_corrections,
            confidence=final_confidence,
            correct=correct,
            ground_truth=ground_truth,
            complexity_score=complexity.overall_score,
            recommended_depth=complexity.recommended_depth,
            actual_depth=current_depth,
            rolled_back=rolled_back,
            rollback_step=rollback_step,
            overfitting_detected=overfitting_detected,
            cv_answers=cv_answers,
            cv_confidence_std=stats_module.stdev(cv_confidences) if len(cv_confidences) > 1 else cv_confidence_std,
            latency_seconds=latency,
            total_tokens=total_tokens,
        )
        
        self._results.append(result)
        
        logger.info(
            f"[{problem_id}] Complete: answer={final_answer}, correct={correct}, "
            f"depth={current_depth}, rolled_back={rolled_back}, overfitting={overfitting_detected}"
        )
        
        return result
    
    def _generate_initial_reasoning(self, problem: str) -> List[str]:
        """Generate initial reasoning steps."""
        prompt = f"""Break down this problem into clear reasoning steps.

Problem: {problem}

Provide 2-4 logical steps that lead to answering this question.
Format each step on a new line starting with "Step X:"."""

        response = self.generator.generate(
            messages=[{"role": "user", "content": prompt}],
            config=GenerationConfig(
                model=self.generator_model,
                temperature=self.config.temperature_reason,
                max_tokens=300,
            )
        )
        
        # Parse steps
        steps = []
        for line in response.text.split("\n"):
            if line.strip().startswith("Step"):
                step_content = line.split(":", 1)[-1].strip() if ":" in line else line.strip()
                if step_content:
                    steps.append(step_content)
        
        if not steps:
            steps = [response.text.strip()]
        
        return steps
    
    def _reflect_and_correct(
        self,
        problem: str,
        reasoning_chain: List[str],
        step_num: int
    ) -> Dict[str, Any]:
        """Perform reflection and apply corrections if needed."""
        reasoning_text = "\n".join(f"Step {i+1}: {s}" for i, s in enumerate(reasoning_chain))
        
        # Reflection prompt
        reflection_prompt = f"""Carefully examine the following reasoning for potential issues.

Problem: {problem}

Current Reasoning:
{reasoning_text}

Analyze each step for:
1. Factual accuracy - are the facts correct?
2. Logical validity - does each step follow from the previous?
3. Completeness - is anything important missing?
4. Relevance - is each step relevant to answering the question?

If you find issues, explain what they are and how to fix them.
If the reasoning is sound, respond with "No issues found."

Format your response as:
Issues: [list any issues or "None"]
Correction: [how to fix or "N/A"]"""

        reflection_response = self.generator.generate(
            messages=[{"role": "user", "content": reflection_prompt}],
            config=GenerationConfig(
                model=self.generator_model,
                temperature=self.config.temperature_reflect,
                max_tokens=400,
            )
        )
        
        reflection_text = reflection_response.text
        
        # Parse reflection
        issues_found = []
        correction = ""
        
        if "no issues found" not in reflection_text.lower():
            # Extract issues
            if "Issues:" in reflection_text or "issues:" in reflection_text:
                issues_part = reflection_text.split("Issues:")[-1] if "Issues:" in reflection_text else reflection_text.split("issues:")[-1]
                issues_part = issues_part.split("Correction:")[0] if "Correction:" in issues_part else issues_part
                issues_found = [i.strip() for i in issues_part.strip().split("\n") if i.strip() and i.strip() != "None"]
            
            # Extract correction
            if "Correction:" in reflection_text or "correction:" in reflection_text:
                correction_part = reflection_text.split("Correction:")[-1] if "Correction:" in reflection_text else reflection_text.split("correction:")[-1]
                correction = correction_part.strip()
        
        # Apply correction if needed
        updated_reasoning = reasoning_chain.copy()
        if issues_found and correction:
            # Add correction as a new step or update reasoning
            correction_prompt = f"""Based on the correction, provide the updated reasoning steps.

Problem: {problem}

Original Reasoning:
{reasoning_text}

Issues Found:
{chr(10).join(issues_found)}

Correction:
{correction}

Provide the corrected reasoning as numbered steps."""

            correction_response = self.generator.generate(
                messages=[{"role": "user", "content": correction_prompt}],
                config=GenerationConfig(
                    model=self.generator_model,
                    temperature=self.config.temperature_reason,
                    max_tokens=300,
                )
            )
            
            # Parse corrected steps
            new_steps = []
            for line in correction_response.text.split("\n"):
                line = line.strip()
                if line and (line.startswith("Step") or (len(line) > 0 and line[0].isdigit())):
                    step_content = line.split(":", 1)[-1].strip() if ":" in line else line
                    if step_content:
                        new_steps.append(step_content)
            
            if new_steps:
                updated_reasoning = new_steps
        
        return {
            "reflection": reflection_text,
            "issues_found": issues_found,
            "correction": correction,
            "updated_reasoning": updated_reasoning,
        }
    
    def _get_answer_with_confidence(
        self,
        problem: str,
        reasoning_chain: List[str]
    ) -> Tuple[str, float]:
        """Generate answer with confidence score.
        
        FIXED: No longer hardcoded to yes/no - now extracts actual answers.
        """
        reasoning_text = "\n".join(f"Step {i+1}: {s}" for i, s in enumerate(reasoning_chain))

        prompt = f"""Based on the reasoning, provide a concise final answer.

Problem: {problem}

Reasoning:
{reasoning_text}

Instructions:
1. Provide ONLY the final answer (number, word, yes/no, or short phrase)
2. Wrap your answer in <answer> tags like: <answer>42</answer>
3. Also provide a confidence score (0.0 to 1.0)

Format:
<answer>[your answer here]</answer>
Confidence: [0.0-1.0]"""

        response = self.generator.generate(
            messages=[{"role": "user", "content": prompt}],
            config=GenerationConfig(
                model=self.generator_model,
                temperature=self.config.temperature_conclude,
                max_tokens=150,
            )
        )

        text = response.text

        # Import unified extractor
        from ..utils.unified_extractor import UnifiedAnswerExtractor
        
        # Extract answer using unified logic
        extracted = UnifiedAnswerExtractor.extract(text)
        answer = extracted.answer if extracted.answer else "unknown"

        # Extract confidence
        confidence = 0.5
        confidence_match = re.search(r"confidence[:\s]+([0-9.]+)", text, re.IGNORECASE)
        if confidence_match:
            try:
                confidence = float(confidence_match.group(1))
                confidence = max(0.0, min(1.0, confidence))
            except ValueError:
                confidence = 0.5
        
        # Higher confidence if we used a reliable extraction method
        if extracted.extraction_method in ["xml_tag", "boxed_latex"]:
            confidence = max(confidence, 0.7)

        return answer, confidence
    
    def _cross_validate(
        self,
        problem: str,
        reasoning_chain: List[str]
    ) -> Tuple[List[str], List[float]]:
        """Perform cross-validation to detect overfitting.
        
        FIXED: No longer hardcoded to yes/no.
        """
        cv_answers = []
        cv_confidences = []

        reasoning_text = "\n".join(f"Step {i+1}: {s}" for i, s in enumerate(reasoning_chain))
        
        from ..utils.unified_extractor import UnifiedAnswerExtractor

        for i in range(self.config.validation_samples):
            temp = self.config.temperature_conclude + (i * 0.1)

            prompt = f"""Answer the question based on the reasoning.

Problem: {problem}

Reasoning:
{reasoning_text}

Wrap your answer in <answer> tags: <answer>your answer</answer>"""

            response = self.generator.generate(
                messages=[{"role": "user", "content": prompt}],
                config=GenerationConfig(
                    model=self.generator_model,
                    temperature=temp,
                    max_tokens=100,
                )
            )

            text = response.text
            extracted = UnifiedAnswerExtractor.extract(text)
            answer = extracted.answer if extracted.answer else "unknown"
            cv_answers.append(answer)
            
            # Confidence based on extraction method reliability
            if extracted.extraction_method in ["xml_tag", "boxed_latex", "explicit_marker"]:
                cv_confidences.append(0.85)
            elif extracted.extraction_method in ["yes_no", "last_sentence"]:
                cv_confidences.append(0.7)
            else:
                cv_confidences.append(0.5)

        return cv_answers, cv_confidences
    
    def _create_result(self, **kwargs) -> AdaptiveReflectionResult:
        """Create an AdaptiveReflectionResult instance."""
        return AdaptiveReflectionResult(**kwargs)

    def close(self):
        """Close the pipeline."""
        super().close()
        self.generator.close()

    def save_results(self, filename: str = "adaptive_reflection_results.json"):
        """Save results."""
        results_path = self.results_dir / filename

        data = {
            "timestamp": time.time(),
            "config": asdict(self.config),
            "num_problems": len(self._results),
            "results": [asdict(r) for r in self._results],
        }

        with open(results_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

        logger.info(f"Results saved to {results_path}")
