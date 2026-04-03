"""Unit tests for answer evaluation and accuracy."""

import pytest
from unittest.mock import Mock
from evaluation.accuracy import AnswerExtractor, AnswerEvaluator
from src.exceptions import AnswerExtractionError


class TestAnswerExtractor:
    """Tests for answer extraction."""
    
    @pytest.mark.unit
    def test_extract_yes(self):
        """Test extracting 'yes' from various formats."""
        test_cases = [
            ("yes", "yes"),
            ("Yes.", "yes"),
            ("**Yes**", "yes"),
            ("The answer is yes", "yes"),
            ("Yes, that is correct.", "yes"),
        ]
        
        for input_text, expected in test_cases:
            result = AnswerExtractor.extract(input_text)
            assert result.lower() == expected, f"Failed for: {input_text}"
    
    @pytest.mark.unit
    def test_extract_no(self):
        """Test extracting 'no' from various formats."""
        test_cases = [
            ("no", "no"),
            ("No.", "no"),
            ("**No**", "no"),
            ("The answer is no", "no"),
            ("No, that is incorrect.", "no"),
        ]
        
        for input_text, expected in test_cases:
            result = AnswerExtractor.extract(input_text)
            assert result.lower() == expected, f"Failed for: {input_text}"
    
    @pytest.mark.unit
    def test_extract_from_markdown(self):
        """Test extracting answers from markdown formatting."""
        assert AnswerExtractor.extract("**yes**") == "yes"
        assert AnswerExtractor.extract("*no*") == "no"
        assert AnswerExtractor.extract("__yes__") == "yes"
    
    @pytest.mark.unit
    def test_extract_from_sentence(self):
        """Test extracting answers from sentences."""
        assert AnswerExtractor.extract("After reasoning, the answer is yes.") == "yes"
        assert AnswerExtractor.extract("Therefore, no is the correct answer.") == "no"
    
    @pytest.mark.unit
    def test_extract_empty_text(self):
        """Test handling of empty text."""
        assert AnswerExtractor.extract("") == ""
        assert AnswerExtractor.extract(None) == ""
    
    @pytest.mark.unit
    def test_extract_numeric_answer(self):
        """Test extracting numeric answers."""
        result1 = AnswerExtractor.extract("The answer is 42")
        assert "42" in result1
        
        result2 = AnswerExtractor.extract("Final: 100.5")
        assert "100.5" in result2
    
    @pytest.mark.unit
    def test_check_answer_exact_match(self):
        """Test exact answer matching."""
        assert AnswerExtractor.check_answer("yes", "yes") is True
        assert AnswerExtractor.check_answer("no", "no") is True
        assert AnswerExtractor.check_answer("yes", "no") is False
    
    @pytest.mark.unit
    def test_check_answer_case_insensitive(self):
        """Test case-insensitive matching."""
        assert AnswerExtractor.check_answer("Yes", "YES") is True
        assert AnswerExtractor.check_answer("No", "no") is True
    
    @pytest.mark.unit
    def test_check_answer_with_extra_text(self):
        """Test matching with extra text."""
        assert AnswerExtractor.check_answer("Yes, that's correct", "yes") is True
        assert AnswerExtractor.check_answer("**No**", "no") is True


class TestAnswerEvaluator:
    """Tests for answer evaluation."""
    
    @pytest.mark.unit
    def test_evaluate_correct_answer(self):
        """Test evaluating correct answer."""
        evaluator = AnswerEvaluator()
        result = evaluator.evaluate("yes", "yes", "test_001")
        
        assert result.correct is True
        assert result.match_type == "exact"
    
    @pytest.mark.unit
    def test_evaluate_incorrect_answer(self):
        """Test evaluating incorrect answer."""
        evaluator = AnswerEvaluator()
        result = evaluator.evaluate("yes", "no", "test_001")
        
        assert result.correct is False
    
    @pytest.mark.unit
    def test_evaluate_semantic_match(self):
        """Test semantic matching."""
        evaluator = AnswerEvaluator()
        
        # Should match yes/no variations
        result1 = evaluator.evaluate("Yes, that is correct", "yes", "test_001")
        assert result1.correct is True
        
        result2 = evaluator.evaluate("The answer is no", "no", "test_002")
        assert result2.correct is True
    
    @pytest.mark.unit
    def test_evaluate_numeric_answer(self):
        """Test evaluating numeric answers."""
        evaluator = AnswerEvaluator()
        
        # Exact numeric match
        result1 = evaluator.evaluate("42", "42", "test_001")
        assert result1.correct is True
        
        # Numeric with tolerance
        result2 = evaluator.evaluate("42.0", "42", "test_002")
        assert result2.correct is True


class TestAnswerExtractorEdgeCases:
    """Tests for edge cases in answer extraction."""
    
    @pytest.mark.unit
    def test_multiple_markdown_patterns(self):
        """Test nested markdown patterns."""
        result = AnswerExtractor.extract("***yes***")
        assert result.lower() == "yes"
    
    @pytest.mark.unit
    def test_whitespace_handling(self):
        """Test handling of extra whitespace."""
        assert AnswerExtractor.extract("  yes  ") == "yes"
        assert AnswerExtractor.extract("\nyes\n") == "yes"
    
    @pytest.mark.unit
    def test_punctuation_handling(self):
        """Test handling of punctuation."""
        assert AnswerExtractor.extract("yes!") == "yes"
        assert AnswerExtractor.extract("yes.") == "yes"
        assert AnswerExtractor.extract("yes,") == "yes"
    
    @pytest.mark.unit
    def test_mixed_case(self):
        """Test mixed case handling."""
        assert AnswerExtractor.extract("YeS") == "yes"
        assert AnswerExtractor.extract("NoO") == "no"
    
    @pytest.mark.unit
    def test_answer_in_middle_of_text(self):
        """Test extracting answer from middle of text."""
        result = AnswerExtractor.extract("Based on reasoning, yes is the answer.")
        assert result.lower() == "yes"
