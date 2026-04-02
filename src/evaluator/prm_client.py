"""Process Reward Model (PRM) Evaluator for scoring reasoning steps."""

import re
import time
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from loguru import logger

import sys
sys.path.insert(0, '..')
from generator.nim_client import NVIDIANIMClient, GenerationConfig
from generator.prompts import PromptBuilder


@dataclass
class EvaluationResult:
    """Result of evaluating a reasoning step."""
    score: float
    raw_response: str
    confidence: float
    parsing_success: bool
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    
    def is_valid(self) -> bool:
        """Check if score is within valid range."""
        return -1.0 <= self.score <= 1.0 and self.parsing_success


@dataclass
class PRMConfig:
    """Configuration for PRM evaluator."""
    model: str = "meta/llama-3.3-70b-instruct"
    temperature: float = 0.1
    max_tokens: int = 64
    score_min: float = -1.0
    score_max: float = 1.0
    retry_on_invalid: bool = True
    max_retries: int = 2


class PRMEvaluator:
    """Process Reward Model evaluator using LLM-as-Judge."""
    
    SCORE_PATTERNS = [
        r'^(-?\d+\.?\d*)$',
        r'score[:\s]*(-?\d+\.?\d*)',
        r'(-?\d+\.?\d*)\s*/\s*1\.?0?',
        r'rating[:\s]*(-?\d+\.?\d*)',
        r'(-?\d+\.?\d*)\s+out\s+of\s+1',
    ]
    
    def __init__(
        self,
        client: Optional[NVIDIANIMClient] = None,
        config: Optional[PRMConfig] = None,
        api_key: Optional[str] = None,
    ):
        if client:
            self.client = client
        else:
            self.client = NVIDIANIMClient(api_key=api_key)
        
        self.config = config or PRMConfig()
        self._evaluation_count = 0
        self._total_scores = 0.0
        self._failed_parses = 0
        
        logger.info(f"PRM Evaluator initialized with model: {self.config.model}")
    
    def _parse_score(self, response: str) -> Tuple[Optional[float], bool]:
        """Parse numerical score from LLM response."""
        response = response.strip()
        
        for pattern in self.SCORE_PATTERNS:
            match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
            if match:
                try:
                    score = float(match.group(1))
                    if self.config.score_min <= score <= self.config.score_max:
                        return score, True
                    else:
                        normalized = (score - self.config.score_min) / (self.config.score_max - self.config.score_min)
                        normalized = normalized * 2 - 1
                        return max(-1.0, min(1.0, normalized)), True
                except ValueError:
                    continue
        
        try:
            score = float(response)
            if self.config.score_min <= score <= self.config.score_max:
                return score, True
        except ValueError:
            pass
        
        return None, False
    
    def _get_generation_config(self) -> GenerationConfig:
        """Get generation config for evaluator."""
        return GenerationConfig(
            model=self.config.model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            top_p=0.95,
            stop_sequences=["\n", "Explanation", "The score"],
        )
    
    def evaluate_step(
        self,
        problem: str,
        previous_steps: List[str],
        current_step: str,
    ) -> EvaluationResult:
        """Evaluate a single reasoning step.
        
        Args:
            problem: The original problem statement
            previous_steps: List of previous reasoning steps
            current_step: The step to evaluate
        
        Returns:
            EvaluationResult with score and metadata
        """
        messages = PromptBuilder.build_evaluation_prompt(
            problem=problem,
            previous_steps=previous_steps,
            current_step=current_step,
        )
        
        gen_config = self._get_generation_config()
        
        for attempt in range(self.config.max_retries + 1):
            start_time = time.time()
            
            try:
                response = self.client.generate(messages, gen_config)
            except Exception as e:
                logger.error(f"Evaluation API call failed: {e}")
                if attempt == self.config.max_retries:
                    return EvaluationResult(
                        score=0.0,
                        raw_response="",
                        confidence=0.0,
                        parsing_success=False,
                    )
                continue
            
            latency_ms = (time.time() - start_time) * 1000
            
            score, parsing_success = self._parse_score(response.text)
            
            if parsing_success or not self.config.retry_on_invalid:
                self._evaluation_count += 1
                self._total_scores += score if score else 0.0
                if not parsing_success:
                    self._failed_parses += 1
                
                confidence = 1.0 if parsing_success else 0.5
                
                return EvaluationResult(
                    score=score if score else 0.0,
                    raw_response=response.text,
                    confidence=confidence,
                    parsing_success=parsing_success,
                    input_tokens=response.input_tokens,
                    output_tokens=response.output_tokens,
                    latency_ms=latency_ms,
                )
            
            logger.warning(f"Failed to parse score, attempt {attempt + 1}/{self.config.max_retries}")
        
        return EvaluationResult(
            score=0.0,
            raw_response="",
            confidence=0.0,
            parsing_success=False,
        )
    
    def evaluate_path(
        self,
        problem: str,
        reasoning_steps: List[str],
    ) -> List[EvaluationResult]:
        """Evaluate all steps in a reasoning path.
        
        Args:
            problem: The original problem statement
            reasoning_steps: List of reasoning steps to evaluate
        
        Returns:
            List of EvaluationResult for each step
        """
        results = []
        
        for i, step in enumerate(reasoning_steps):
            previous = reasoning_steps[:i]
            result = self.evaluate_step(problem, previous, step)
            results.append(result)
        
        return results
    
    def get_aggregate_score(
        self,
        problem: str,
        reasoning_steps: List[str],
        method: str = "mean",
    ) -> float:
        """Get aggregate score for entire reasoning path.
        
        Args:
            problem: The original problem statement
            reasoning_steps: List of reasoning steps
            method: Aggregation method ('mean', 'min', 'prod', 'last')
        
        Returns:
            Aggregated score
        """
        results = self.evaluate_path(problem, reasoning_steps)
        scores = [r.score for r in results if r.parsing_success]
        
        if not scores:
            return 0.0
        
        if method == "mean":
            return sum(scores) / len(scores)
        elif method == "min":
            return min(scores)
        elif method == "prod":
            product = 1.0
            for s in scores:
                product *= (s + 1) / 2
            return product * 2 - 1
        elif method == "last":
            return scores[-1]
        else:
            return sum(scores) / len(scores)
    
    def batch_evaluate(
        self,
        evaluations: List[Tuple[str, List[str], str]],
    ) -> List[EvaluationResult]:
        """Batch evaluate multiple steps.
        
        Args:
            evaluations: List of (problem, previous_steps, current_step) tuples
        
        Returns:
            List of EvaluationResult
        """
        results = []
        for problem, previous_steps, current_step in evaluations:
            result = self.evaluate_step(problem, previous_steps, current_step)
            results.append(result)
        return results
    
    def get_stats(self) -> Dict:
        """Get evaluator statistics."""
        return {
            "total_evaluations": self._evaluation_count,
            "average_score": self._total_scores / max(1, self._evaluation_count),
            "failed_parses": self._failed_parses,
            "parse_success_rate": 1 - (self._failed_parses / max(1, self._evaluation_count)),
        }
    
    def reset_stats(self):
        """Reset evaluator statistics."""
        self._evaluation_count = 0
        self._total_scores = 0.0
        self._failed_parses = 0
