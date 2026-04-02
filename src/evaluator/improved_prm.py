"""Improved verification system using different models and better prompts."""

import os
import time
import re
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from loguru import logger
from dotenv import load_dotenv

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generator.nim_client import NVIDIANIMClient, GenerationConfig

load_dotenv()


@dataclass
class VerificationResult:
    """Result of verifying a reasoning step or answer."""
    score: float
    category: str
    reason: str
    confidence: float
    is_vacuous: bool = False
    makes_progress: bool = False
    leads_to_answer: bool = False
    raw_response: str = ""


class ImprovedPRM:
    """Improved Process Reward Model with better prompts and verification."""
    
    VACUOUS_PATTERNS = [
        r"to determine",
        r"let me",
        r"i need to",
        r"first, (i|we) (need to|should|will)",
        r"the next step",
        r"this (step|problem)",
        r"we can see",
        r"it is important",
    ]
    
    PROGRESS_PATTERNS = [
        r"\d+",
        r"therefore",
        r"because",
        r"since",
        r"this means",
        r"which implies",
        r"the answer is",
        r"so (we can conclude|the)",
    ]
    
    def __init__(
        self,
        generator_model: str = "meta/llama-3.1-8b-instruct",
        verifier_model: str = "meta/llama-3.3-70b-instruct",
        use_reward_model: bool = True,
    ):
        self.generator = NVIDIANIMClient(api_key=os.getenv("NVIDIA_API_KEY"))
        self.verifier = NVIDIANIMClient(api_key=os.getenv("NVIDIA_API_KEY"))
        self.generator_model = generator_model
        self.verifier_model = verifier_model
        self.use_reward_model = use_reward_model
        
        self._call_count = 0
        self._total_tokens = 0
    
    def evaluate_step(
        self,
        problem: str,
        previous_steps: List[str],
        current_step: str,
        depth: int = 0,
    ) -> VerificationResult:
        """Evaluate a reasoning step with multiple criteria."""
        
        is_vacuous = self._check_vacuous(current_step)
        makes_progress = self._check_progress(current_step)
        
        quality_score = self._evaluate_step_quality(problem, previous_steps, current_step)
        
        progress_score = self._evaluate_answer_progress(problem, previous_steps, current_step)
        
        combined_score = 0.4 * quality_score + 0.6 * progress_score
        
        if is_vacuous and not makes_progress:
            combined_score -= 0.3
        
        if depth < 2 and makes_progress:
            combined_score += 0.1
        
        category = self._categorize_step(combined_score, is_vacuous, makes_progress)
        
        return VerificationResult(
            score=max(-1.0, min(1.0, combined_score)),
            category=category,
            reason=f"Quality: {quality_score:.2f}, Progress: {progress_score:.2f}",
            confidence=0.8,
            is_vacuous=is_vacuous,
            makes_progress=makes_progress,
            raw_response="",
        )
    
    def verify_answer(
        self,
        problem: str,
        reasoning: List[str],
        answer: str,
    ) -> VerificationResult:
        """Verify if the final answer is correct using different model."""
        
        prompt = f"""You are an answer verification expert. Your task is to verify if the given answer is correct for the problem.

Problem: {problem}

Reasoning:
{chr(10).join(f"- {s}" for s in reasoning)}

Proposed Answer: {answer}

Evaluate:
1. Is the answer directly responsive to the question?
2. Is the reasoning logically sound?
3. Does the answer follow from the reasoning?

Output format:
VERDICT: [CORRECT/INCORRECT/PARTIALLY_CORRECT]
CONFIDENCE: [0.0 to 1.0]
BRIEF_REASON: [one sentence]

Now evaluate:"""
        
        messages = [{"role": "user", "content": prompt}]
        
        config = GenerationConfig(
            model=self.verifier_model,
            temperature=0.1,
            max_tokens=100,
        )
        
        response = self.verifier.generate(messages, config)
        
        verdict, confidence, reason = self._parse_verification(response.text)
        
        score_map = {
            "CORRECT": 1.0,
            "PARTIALLY_CORRECT": 0.5,
            "INCORRECT": -1.0,
        }
        
        self._call_count += 1
        self._total_tokens += response.input_tokens + response.output_tokens
        
        return VerificationResult(
            score=score_map.get(verdict, 0.0) * confidence,
            category=verdict,
            reason=reason,
            confidence=confidence,
            raw_response=response.text,
        )
    
    def find_error_step(
        self,
        problem: str,
        reasoning: List[str],
        wrong_answer: str,
    ) -> Tuple[int, str]:
        """Binary search to find which step caused the error."""
        
        if not reasoning:
            return -1, "No reasoning to analyze"
        
        low, high = 0, len(reasoning) - 1
        error_step = -1
        error_reason = ""
        
        while low <= high:
            mid = (low + high) // 2
            
            partial_reasoning = reasoning[:mid + 1]
            
            error_check = self._check_step_for_error(problem, partial_reasoning)
            
            if error_check["has_error"]:
                error_step = mid
                error_reason = error_check["reason"]
                high = mid - 1
            else:
                low = mid + 1
        
        if error_step == -1:
            error_step = len(reasoning) - 1
            error_reason = "Error in final conclusion"
        
        return error_step, error_reason
    
    def _check_step_for_error(self, problem: str, reasoning: List[str]) -> Dict:
        """Check if error exists in partial reasoning."""
        
        prompt = f"""Analyze this partial reasoning for logical errors or factual mistakes.

Problem: {problem}

Partial Reasoning:
{chr(10).join(f"{i+1}. {s}" for i, s in enumerate(reasoning))}

Check for:
- Factual errors
- Logical fallacies
- Unjustified assumptions
- Missing key information

Output:
HAS_ERROR: [YES/NO]
ERROR_LOCATION: [step number or "none"]
ERROR_TYPE: [factual/logical/assumption/missing/none]

Analyze:"""
        
        messages = [{"role": "user", "content": prompt}]
        config = GenerationConfig(model=self.verifier_model, temperature=0.1, max_tokens=50)
        response = self.verifier.generate(messages, config)
        
        has_error = "YES" in response.text.upper()
        
        error_type = "none"
        for et in ["factual", "logical", "assumption", "missing"]:
            if et.upper() in response.text.upper():
                error_type = et
                break
        
        return {
            "has_error": has_error,
            "reason": error_type if has_error else "",
        }
    
    def _evaluate_step_quality(self, problem: str, previous: List[str], step: str) -> float:
        """Evaluate logical quality of the step."""
        
        prompt = f"""Rate the logical quality of this reasoning step.

Problem: {problem}
Previous steps: {len(previous)}
Current step: {step}

Rate the step on:
- Logical validity: Is the reasoning sound?
- Clarity: Is it clear and unambiguous?
- Relevance: Does it relate to the problem?

Output ONLY a score from -1 (wrong) to 1 (correct):
Score:"""
        
        messages = [{"role": "user", "content": prompt}]
        config = GenerationConfig(model=self.verifier_model, temperature=0.1, max_tokens=10)
        response = self.verifier.generate(messages, config)
        
        score = self._parse_score(response.text)
        return score
    
    def _evaluate_answer_progress(self, problem: str, previous: List[str], step: str) -> float:
        """Evaluate how much the step contributes to the final answer."""
        
        prompt = f"""Rate how much this step contributes to solving the problem.

Problem: {problem}
Current step: {step}

Score based on:
- Does it state a concrete fact? (+0.5)
- Does it make a logical inference? (+0.3)
- Does it directly lead toward the answer? (+0.2)
- Is it just meta-commentary like "Let me think"? (-0.3)
- Does it actually help solve the problem? (+0.5)

Output ONLY a score from -1 (harmful) to 1 (very helpful):
Score:"""
        
        messages = [{"role": "user", "content": prompt}]
        config = GenerationConfig(model=self.verifier_model, temperature=0.1, max_tokens=10)
        response = self.verifier.generate(messages, config)
        
        score = self._parse_score(response.text)
        return score
    
    def _check_vacuous(self, step: str) -> bool:
        """Check if step is vacuous meta-commentary."""
        step_lower = step.lower()
        for pattern in self.VACUOUS_PATTERNS:
            if re.search(pattern, step_lower):
                word_count = len(step.split())
                if word_count < 20:
                    return True
        return False
    
    def _check_progress(self, step: str) -> bool:
        """Check if step makes progress toward answer."""
        step_lower = step.lower()
        for pattern in self.PROGRESS_PATTERNS:
            if re.search(pattern, step_lower):
                return True
        return False
    
    def _categorize_step(self, score: float, is_vacuous: bool, makes_progress: bool) -> str:
        """Categorize the step type."""
        if is_vacuous and not makes_progress:
            return "VACUOUS"
        elif makes_progress and score > 0.5:
            return "PRODUCTIVE"
        elif score > 0.7:
            return "HIGH_QUALITY"
        elif score > 0.3:
            return "MODERATE"
        elif score > 0:
            return "MINOR_PROGRESS"
        else:
            return "PROBLEMATIC"
    
    def _parse_score(self, text: str) -> float:
        """Parse numerical score from response."""
        numbers = re.findall(r"-?\d+\.?\d*", text)
        if numbers:
            try:
                score = float(numbers[0])
                return max(-1.0, min(1.0, score))
            except ValueError:
                pass
        return 0.0
    
    def _parse_verification(self, text: str) -> Tuple[str, float, str]:
        """Parse verification response."""
        text_upper = text.upper()
        
        verdict = "INCORRECT"
        if "CORRECT" in text_upper and "INCORRECT" not in text_upper:
            verdict = "CORRECT"
        elif "PARTIALLY" in text_upper:
            verdict = "PARTIALLY_CORRECT"
        
        confidence = 0.7
        conf_match = re.search(r"CONFIDENCE:\s*([\d.]+)", text_upper)
        if conf_match:
            try:
                confidence = float(conf_match.group(1))
            except ValueError:
                pass
        
        reason = "Verification complete"
        reason_match = re.search(r"BRIEF_REASON:\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
        if reason_match:
            reason = reason_match.group(1).strip()
        
        return verdict, confidence, reason
    
    def get_stats(self) -> Dict:
        """Get usage statistics."""
        return {
            "total_calls": self._call_count,
            "total_tokens": self._total_tokens,
        }


class ComparativeLearner:
    """Learn from comparing multiple reasoning paths."""
    
    def __init__(self):
        self.successful_paths: List[Dict] = []
        self.failed_paths: List[Dict] = []
        self.feature_weights: Dict[str, float] = {}
    
    def record_path(
        self,
        problem: str,
        reasoning: List[str],
        answer: str,
        prm_scores: List[float],
        correct: bool,
    ):
        """Record a reasoning path and its outcome."""
        
        path_data = {
            "problem": problem,
            "reasoning": reasoning,
            "answer": answer,
            "prm_scores": prm_scores,
            "correct": correct,
            "features": self._extract_features(reasoning),
        }
        
        if correct:
            self.successful_paths.append(path_data)
        else:
            self.failed_paths.append(path_data)
        
        if len(self.successful_paths) >= 3 and len(self.failed_paths) >= 3:
            self._update_weights()
    
    def _extract_features(self, reasoning: List[str]) -> Dict[str, float]:
        """Extract features from reasoning chain."""
        
        features = {
            "num_steps": len(reasoning),
            "avg_step_length": sum(len(s.split()) for s in reasoning) / max(1, len(reasoning)),
            "has_numbers": sum(1 for s in reasoning if re.search(r"\d+", s)) / max(1, len(reasoning)),
            "has_because": sum(1 for s in reasoning if "because" in s.lower()) / max(1, len(reasoning)),
            "has_therefore": sum(1 for s in reasoning if "therefore" in s.lower()) / max(1, len(reasoning)),
            "starts_with_fact": 1.0 if reasoning and len(reasoning[0].split()) > 10 else 0.0,
        }
        
        return features
    
    def _update_weights(self):
        """Update feature weights based on successful vs failed paths."""
        
        success_features = {}
        fail_features = {}
        
        for path in self.successful_paths[-10:]:
            for feat, val in path["features"].items():
                success_features[feat] = success_features.get(feat, 0) + val
        
        for path in self.failed_paths[-10:]:
            for feat, val in path["features"].items():
                fail_features[feat] = fail_features.get(feat, 0) + val
        
        for feat in success_features:
            s_avg = success_features.get(feat, 0) / max(1, len(self.successful_paths))
            f_avg = fail_features.get(feat, 0) / max(1, len(self.failed_paths))
            
            self.feature_weights[feat] = s_avg - f_avg
    
    def get_path_quality_prediction(self, reasoning: List[str]) -> float:
        """Predict quality of a reasoning path based on learned weights."""
        
        if not self.feature_weights:
            return 0.5
        
        features = self._extract_features(reasoning)
        
        score = 0.0
        for feat, val in features.items():
            weight = self.feature_weights.get(feat, 0)
            score += val * weight
        
        return (score + 1) / 2
    
    def get_learning_summary(self) -> str:
        """Get summary of what was learned."""
        
        if not self.feature_weights:
            return "Not enough data to learn from yet."
        
        sorted_features = sorted(
            self.feature_weights.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        
        summary_lines = ["Learned patterns for successful reasoning:"]
        for feat, weight in sorted_features[:5]:
            if weight > 0:
                summary_lines.append(f"  + {feat}: helps (+{weight:.3f})")
            else:
                summary_lines.append(f"  - {feat}: hurts ({weight:.3f})")
        
        return "\n".join(summary_lines)
