"""Unified Answer Extraction and Evaluation.

This module provides a single source of truth for answer extraction
and evaluation across all pipelines, ensuring fair comparison.
"""

import re
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class ExtractedAnswer:
    """Result of answer extraction."""
    answer: str
    confidence: float
    extraction_method: str


class UnifiedAnswerExtractor:
    """Universal answer extractor used by ALL pipelines.
    
    This ensures fair evaluation - no pipeline gets special treatment.
    """
    
    XML_ANSWER_PATTERN = re.compile(
        r'<(?:final_)?answer>(.*?)</(?:final_)?answer>',
        re.DOTALL | re.IGNORECASE
    )
    
    BOXED_PATTERN = re.compile(
        r'\\{1,2}boxed\{([^}]+)\}',
        re.DOTALL
    )
    
    FINAL_ANSWER_MARKERS = [
        r'(?:final\s+)?answer\s*[:is]+\s*(.+?)(?:\n|$)',
        r'the\s+answer\s+is\s*[:is]*\s*(.+?)(?:\n|$)',
        r'(?:therefore|thus|so|hence)[,.]?\s*(?:the\s+)?(?:result|answer)\s+is\s*[:is]*\s*(.+?)(?:\n|$)',
        r'(?:therefore|thus|so|hence|conclusion)[,.]?\s*(.+?)(?:\n|$)',
    ]
    
    YES_NO_PATTERN = re.compile(r'\b(yes|no)\b', re.IGNORECASE)
    
    NUMBER_PATTERN = re.compile(r'-?\d+(?:,\d{3})*(?:\.\d+)?(?:\s*(?:million|billion|thousand|k|m|b))?', re.IGNORECASE)
    
    @classmethod
    def extract(cls, text: str, full_response: str = "") -> ExtractedAnswer:
        """Extract answer from text using unified logic.
        
        Priority:
        1. XML tags (<answer> or <final_answer>)
        2. Boxed LaTeX (\\boxed{...})
        3. Explicit markers ("Answer:", "The answer is", etc.)
        4. Yes/No detection
        5. Number extraction (for numeric answers)
        6. Last sentence/line
        
        Args:
            text: The text to extract from (usually the final generation)
            full_response: Full response for fallback context
            
        Returns:
            ExtractedAnswer with extracted text and metadata
        """
        if not text or not text.strip():
            text = full_response
        if not text:
            return ExtractedAnswer("", 0.0, "empty")
        
        text = text.strip()
        
        # Priority 1: XML tags
        match = cls.XML_ANSWER_PATTERN.search(text)
        if match:
            answer = match.group(1).strip()
            return ExtractedAnswer(answer, 1.0, "xml_tag")
        
        # Priority 2: Boxed LaTeX
        match = cls.BOXED_PATTERN.search(text)
        if match:
            answer = match.group(1).strip()
            return ExtractedAnswer(answer, 0.95, "boxed_latex")
        
        # Priority 3: Explicit markers
        for pattern in cls.FINAL_ANSWER_MARKERS:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                answer = match.group(1).strip()
                answer = re.sub(r'^[:\s]+', '', answer)
                answer = re.sub(r'[:\s]+$', '', answer)
                if answer:
                    return ExtractedAnswer(answer, 0.9, "explicit_marker")
        
        # Priority 4: Yes/No detection
        yes_no_match = cls.YES_NO_PATTERN.search(text)
        if yes_no_match:
            return ExtractedAnswer(yes_no_match.group(1).lower(), 0.85, "yes_no")
        
        # Priority 5: Number extraction (look for standalone numbers at end)
        # This is important for math problems
        sentences = re.split(r'[.!?\n]', text)
        for sentence in reversed(sentences):
            sentence = sentence.strip()
            if not sentence:
                continue
            # Look for "is X" or "= X" pattern
            is_match = re.search(r'(?:is|=)\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)\s*\.?\s*$', sentence)
            if is_match:
                return ExtractedAnswer(is_match.group(1).replace(',', ''), 0.85, "number_from_sentence")
        
        # Priority 6: Last meaningful line/sentence
        for sentence in reversed(sentences):
            sentence = sentence.strip()
            if sentence and len(sentence) > 1:
                # Check for markdown bold
                bold_match = re.search(r'\*\*([^*]+)\*\*', sentence)
                if bold_match:
                    return ExtractedAnswer(bold_match.group(1).strip(), 0.75, "markdown_bold")
                return ExtractedAnswer(sentence, 0.6, "last_sentence")
        
        return ExtractedAnswer(text[:200] if len(text) > 200 else text, 0.3, "fallback")
    
    @classmethod
    def check_answer(cls, predicted: str, ground_truth: str) -> bool:
        """Check if predicted answer matches ground truth.
        
        Uses multiple matching strategies:
        1. Exact match (normalized)
        2. Yes/No match
        3. Numeric tolerance match
        4. Contains match (only if ground truth is short)
        
        Args:
            predicted: Predicted answer
            ground_truth: Ground truth answer
            
        Returns:
            True if answers match
        """
        if not predicted or not ground_truth:
            return False
        
        pred_extracted = cls.extract(predicted)
        pred = pred_extracted.answer.lower().strip()
        truth = ground_truth.lower().strip()
        
        # Clean punctuation from end
        pred = pred.rstrip('.,;:')
        truth = truth.rstrip('.,;:')
        
        # Remove common filler words
        filler_words = ['the', 'a', 'an', 'is', 'are', 'was', 'were', 'approximately', 'about']
        pred_clean = ' '.join(w for w in pred.split() if w not in filler_words)
        truth_clean = ' '.join(w for w in truth.split() if w not in filler_words)
        
        # Exact match
        if pred_clean == truth_clean:
            return True
        
        # Yes/No match
        yes_no_match = cls.YES_NO_PATTERN.search(pred)
        truth_yes_no = cls.YES_NO_PATTERN.search(truth)
        if yes_no_match and truth_yes_no:
            return yes_no_match.group(1).lower() == truth_yes_no.group(1).lower()
        
        # Numeric match with tolerance
        pred_numbers = cls.NUMBER_PATTERN.findall(pred)
        truth_numbers = cls.NUMBER_PATTERN.findall(truth)
        if pred_numbers and truth_numbers:
            try:
                pred_num = cls._parse_number(pred_numbers[0])
                truth_num = cls._parse_number(truth_numbers[0])
                if pred_num is not None and truth_num is not None:
                    relative_error = abs(pred_num - truth_num) / max(abs(truth_num), 0.001)
                    return relative_error < 0.05  # 5% tolerance
            except (ValueError, IndexError):
                pass
        
        # Contains match - ONLY if ground truth is short (prevents RL loophole)
        if len(truth_clean) < 20:  # Only allow contains for short ground truths
            if truth_clean in pred_clean:
                return True
            if pred_clean in truth_clean:
                return True
        
        return False
    
    @classmethod
    def _parse_number(cls, num_str: str) -> Optional[float]:
        """Parse number string with multipliers."""
        try:
            num_str = num_str.lower().replace(',', '')
            multipliers = {
                'thousand': 1e3, 'k': 1e3,
                'million': 1e6, 'm': 1e6,
                'billion': 1e9, 'b': 1e9,
            }
            
            multiplier = 1.0
            for word, mult in multipliers.items():
                if word in num_str:
                    multiplier = mult
                    num_str = num_str.replace(word, '').strip()
                    break
            
            return float(num_str) * multiplier
        except (ValueError, AttributeError):
            return None
