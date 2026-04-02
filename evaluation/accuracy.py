"""Answer evaluation and accuracy computation."""

import re
import json
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
from collections import Counter


@dataclass
class EvaluationResult:
    """Result of evaluating a single answer."""
    problem_id: str
    predicted: str
    ground_truth: str
    correct: bool
    match_type: str
    confidence: float = 1.0
    extracted_answer: str = ""


class AnswerExtractor:
    """Robust answer extraction from various formats."""

    MARKDOWN_PATTERNS = [
        r'\*\*([^*]+)\*\*',
        r'\*([^*]+)\*',
        r'__([^_]+)__',
        r'_([^_]+)_',
    ]

    FINAL_ANSWER_PATTERNS = [
        r'(?:final\s+)?answer\s*[:is]?\s*(.+?)(?:\n|$)',
        r'(?:the\s+)?answer\s+is\s*[:is]?\s*(.+?)(?:\n|$)',
        r'(?:conclusion|therefore|thus|so|hence)[,.]?\s*(.+?)(?:\n|$)',
        r'^\s*(?:yes|no)\s*[.!]?\s*$',
    ]

    YES_NO_PATTERNS = [
        r'\b(yes|no)\b',
        r'^(yes|no)[\s.!]*$',
        r'(?:the\s+answer\s+is\s+|it\s+is\s+)(yes|no)',
        r'(?:^|\s)(yes|no)(?:\s|$|\.|!|,)',
    ]

    @classmethod
    def extract(cls, text: str, full_response: str = "") -> str:
        """Extract clean answer from text or full response."""
        if not text or text.strip() in ['**', '*', '', '****', '__']:
            text = full_response
        if not text:
            return ""
        
        text = text.strip()
        
        for pattern in cls.MARKDOWN_PATTERNS:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                clean = match.strip()
                if clean and clean.lower() in ['yes', 'no'] or len(clean) > 2:
                    text = clean
                    break
        
        for pattern in cls.YES_NO_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                yes_no = match.group(1).lower() if match.lastindex else match.group(0).lower()
                if yes_no in ['yes', 'no']:
                    return yes_no
        
        for pattern in cls.FINAL_ANSWER_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                answer = match.group(1).strip()
                answer = re.sub(r'^[:\s]+', '', answer)
                answer = re.sub(r'[:\s]+$', '', answer)
                if answer:
                    yes_no_match = re.search(r'\b(yes|no)\b', answer, re.IGNORECASE)
                    if yes_no_match:
                        return yes_no_match.group(1).lower()
                    return answer
        
        sentences = re.split(r'[.!?\n]', text)
        if sentences:
            last_sentence = sentences[-1].strip() if sentences[-1].strip() else (sentences[-2].strip() if len(sentences) > 1 else "")
            yes_no = re.search(r'\b(yes|no)\b', last_sentence, re.IGNORECASE)
            if yes_no:
                return yes_no.group(1).lower()
        
        return text

    @classmethod
    def check_answer(cls, predicted: str, ground_truth: str) -> bool:
        """Check if predicted answer matches ground truth."""
        pred_extracted = cls.extract(predicted, predicted)
        truth_extracted = cls.extract(ground_truth, ground_truth)
        
        pred_lower = pred_extracted.lower().strip()
        truth_lower = truth_extracted.lower().strip()
        
        if truth_lower in ["yes", "no"]:
            return truth_lower in pred_lower
        
        return truth_lower in pred_lower or pred_lower == truth_lower


class AnswerEvaluator:
    """Evaluate predicted answers against ground truth."""

    YES_NO_PATTERN = re.compile(r'\b(yes|no)\b', re.IGNORECASE)
    NUMBER_PATTERN = re.compile(r'-?\d+(?:\.\d+)?')

    POSITIVE_INDICATORS = [
        "yes", "correct", "true", "can", "does", "will", "possible",
        "affirmative", "indeed", "absolutely", "certainly", "definitely",
        "possible", "theoretically possible", "it is possible", "is a",
        "is an", "are", "is able", "is indeed"
    ]
    NEGATIVE_INDICATORS = [
        "no", "incorrect", "false", "cannot", "doesn't", "won't",
        "impossible", "never", "negative", "is not", "are not", "do not",
        "does not", "will not", "isn't", "aren't", "doesn't", "don't",
        "not", "is not"
    ]
    DEPENDS_INDICATORS = [
        "depends", "it depends", "sometimes", "may be", "might be", "could be",
        "varies", "conditional", "situation", "uncertain", "it depends on",
        "under certain conditions", "in certain circumstances", "conditionally"
    ]

    COMPARISON_NEGATIVE = [
        "is not", "are not", "isn't", "aren't", "not heavier", "not larger",
        "lower than", "lighter than", "smaller than", "less than",
        "not heavier than", "is not heavier", "is lighter", "lower mass",
        "slightly lower", "slightly lighter", "not"
    ]

    def evaluate(
        self,
        predicted: str,
        ground_truth: str,
        problem_id: str = "unknown",
        full_response: str = "",
    ) -> EvaluationResult:
        """Evaluate a predicted answer.

        Args:
            predicted: Predicted answer (may be extracted)
            ground_truth: Ground truth answer
            problem_id: Problem identifier
            full_response: Full model response for better semantic matching

        Returns:
            EvaluationResult
        """
        extracted = AnswerExtractor.extract(predicted, full_response)
        
        if not extracted:
            return EvaluationResult(
                problem_id=problem_id,
                predicted=predicted,
                ground_truth=ground_truth,
                correct=False,
                match_type="extraction_failed",
                extracted_answer="",
            )

        pred_normalized = self._normalize(extracted)
        truth_normalized = self._normalize(ground_truth)

        exact_match = pred_normalized == truth_normalized

        if exact_match:
            return EvaluationResult(
                problem_id=problem_id,
                predicted=predicted,
                ground_truth=ground_truth,
                correct=True,
                match_type="exact",
                extracted_answer=extracted,
            )

        yes_no_match = self._check_yes_no(pred_normalized, truth_normalized)
        if yes_no_match is not None:
            return EvaluationResult(
                problem_id=problem_id,
                predicted=predicted,
                ground_truth=ground_truth,
                correct=yes_no_match,
                match_type="yes_no",
                extracted_answer=extracted,
            )

        semantic_match = self._check_semantic_match(pred_normalized, truth_normalized)
        if semantic_match is not None:
            return EvaluationResult(
                problem_id=problem_id,
                predicted=predicted,
                ground_truth=ground_truth,
                correct=semantic_match,
                match_type="semantic",
                confidence=0.9,
                extracted_answer=extracted,
            )

        number_match = self._check_numbers(pred_normalized, truth_normalized)
        if number_match is not None:
            return EvaluationResult(
                problem_id=problem_id,
                predicted=predicted,
                ground_truth=ground_truth,
                correct=number_match,
                match_type="number",
                extracted_answer=extracted,
            )

        contains_match = truth_normalized in pred_normalized
        if contains_match:
            return EvaluationResult(
                problem_id=problem_id,
                predicted=predicted,
                ground_truth=ground_truth,
                correct=True,
                match_type="contains",
                confidence=0.8,
                extracted_answer=extracted,
            )

        return EvaluationResult(
            problem_id=problem_id,
            predicted=predicted,
            ground_truth=ground_truth,
            correct=False,
            match_type="none",
            extracted_answer=extracted,
        )

    def _check_semantic_match(self, pred: str, truth: str) -> Optional[bool]:
        """Check semantic match for yes/no/depends answers."""
        truth_type = self._get_answer_type(truth)
        pred_type = self._get_answer_type(pred)
        
        if truth_type and pred_type:
            return truth_type == pred_type
        
        return None

    def _get_answer_type(self, text: str) -> Optional[str]:
        """Determine if text indicates yes, no, or depends."""
        text_lower = text.lower()
        
        for indicator in self.DEPENDS_INDICATORS:
            if indicator in text_lower:
                return "it depends"
        
        for indicator in self.COMPARISON_NEGATIVE:
            if indicator in text_lower:
                return "no"
        
        yes_no_match = self.YES_NO_PATTERN.search(text)
        if yes_no_match:
            return yes_no_match.group(1).lower()
        
        has_positive = any(ind in text_lower for ind in self.POSITIVE_INDICATORS)
        has_negative = any(ind in text_lower for ind in self.NEGATIVE_INDICATORS)
        
        if has_negative and not has_positive:
            return "no"
        if has_positive and not has_negative:
            return "yes"
        
        return None
    
    def _normalize(self, text: str) -> str:
        """Normalize text for comparison."""
        text = text.lower().strip()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def _check_yes_no(self, pred: str, truth: str) -> Optional[bool]:
        """Check yes/no answer match."""
        pred_match = self.YES_NO_PATTERN.search(pred)
        truth_match = self.YES_NO_PATTERN.search(truth)
        
        if pred_match and truth_match:
            return pred_match.group(1).lower() == truth_match.group(1).lower()
        
        return None
    
    def _check_numbers(self, pred: str, truth: str) -> Optional[bool]:
        """Check numeric answer match."""
        pred_numbers = self.NUMBER_PATTERN.findall(pred)
        truth_numbers = self.NUMBER_PATTERN.findall(truth)
        
        if pred_numbers and truth_numbers:
            try:
                pred_num = float(pred_numbers[0])
                truth_num = float(truth_numbers[0])
                return abs(pred_num - truth_num) < 0.001
            except (ValueError, IndexError):
                pass
        
        return None
    
    def evaluate_batch(
        self,
        predictions: List[Tuple[str, str, str]],
    ) -> List[EvaluationResult]:
        """Evaluate batch of predictions.
        
        Args:
            predictions: List of (problem_id, predicted, ground_truth) tuples
        
        Returns:
            List of EvaluationResult
        """
        return [
            self.evaluate(pred, truth, pid)
            for pid, pred, truth in predictions
        ]


class AccuracyCalculator:
    """Calculate various accuracy metrics."""
    
    def __init__(self):
        self._results: List[EvaluationResult] = []
    
    def add_result(self, result: EvaluationResult):
        """Add evaluation result."""
        self._results.append(result)
    
    def add_results(self, results: List[EvaluationResult]):
        """Add multiple results."""
        self._results.extend(results)
    
    def overall_accuracy(self) -> float:
        """Calculate overall accuracy."""
        if not self._results:
            return 0.0
        return sum(1 for r in self._results if r.correct) / len(self._results)
    
    def accuracy_by_type(self) -> Dict[str, float]:
        """Calculate accuracy by match type."""
        by_type: Dict[str, List[EvaluationResult]] = {}
        
        for r in self._results:
            if r.match_type not in by_type:
                by_type[r.match_type] = []
            by_type[r.match_type].append(r)
        
        return {
            match_type: sum(1 for r in results if r.correct) / len(results)
            for match_type, results in by_type.items()
        }
    
    def confidence_distribution(self) -> Dict[str, int]:
        """Get distribution of confidence levels."""
        distribution = Counter()
        
        for r in self._results:
            if r.confidence >= 0.9:
                distribution["high"] += 1
            elif r.confidence >= 0.7:
                distribution["medium"] += 1
            else:
                distribution["low"] += 1
        
        return dict(distribution)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        return {
            "total": len(self._results),
            "correct": sum(1 for r in self._results if r.correct),
            "accuracy": self.overall_accuracy(),
            "by_type": self.accuracy_by_type(),
            "confidence_distribution": self.confidence_distribution(),
        }
    
    def clear(self):
        """Clear all results."""
        self._results.clear()


def compare_methods(
    baseline_results: List[EvaluationResult],
    rl_results: List[EvaluationResult],
) -> Dict[str, Any]:
    """Compare baseline and RL results.
    
    Args:
        baseline_results: Baseline evaluation results
        rl_results: RL method evaluation results
    
    Returns:
        Comparison statistics
    """
    baseline_correct = sum(1 for r in baseline_results if r.correct)
    rl_correct = sum(1 for r in rl_results if r.correct)
    
    n = len(baseline_results)
    
    if n == 0:
        return {"error": "No results to compare"}
    
    baseline_acc = baseline_correct / n
    rl_acc = rl_correct / n
    
    improvement = rl_acc - baseline_acc
    relative_improvement = (rl_acc - baseline_acc) / max(0.001, baseline_acc)
    
    agreed_correct = sum(
        1 for b, r in zip(baseline_results, rl_results)
        if b.correct and r.correct
    )
    agreed_incorrect = sum(
        1 for b, r in zip(baseline_results, rl_results)
        if not b.correct and not r.correct
    )
    
    baseline_only = sum(
        1 for b, r in zip(baseline_results, rl_results)
        if b.correct and not r.correct
    )
    rl_only = sum(
        1 for b, r in zip(baseline_results, rl_results)
        if not b.correct and r.correct
    )
    
    return {
        "baseline_accuracy": baseline_acc,
        "rl_accuracy": rl_acc,
        "absolute_improvement": improvement,
        "relative_improvement": relative_improvement,
        "agreement": (agreed_correct + agreed_incorrect) / n,
        "baseline_correct": baseline_correct,
        "rl_correct": rl_correct,
        "both_correct": agreed_correct,
        "both_incorrect": agreed_incorrect,
        "baseline_only_correct": baseline_only,
        "rl_only_correct": rl_only,
    }
