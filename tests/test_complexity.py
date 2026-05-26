"""Tests for query complexity analyzer."""

import pytest
from src.utils.complexity import QueryComplexityAnalyzer, ComplexityScore


class TestComplexityScore:
    """Tests for ComplexityScore dataclass."""

    def test_creation(self):
        score = ComplexityScore(
            overall_score=0.5,
            factors={"a": 0.3},
            recommended_depth=2,
        )
        assert score.overall_score == 0.5
        assert score.factors == {"a": 0.3}
        assert score.recommended_depth == 2

    def test_defaults(self):
        score = ComplexityScore(
            overall_score=0.0,
            factors={},
            recommended_depth=1,
        )
        assert score.recommended_tree_depth == 0
        assert score.reasoning == ""


class TestQueryComplexityAnalyzer:
    """Tests for QueryComplexityAnalyzer."""

    @pytest.fixture
    def analyzer(self):
        return QueryComplexityAnalyzer()

    def test_factual_query_low_complexity(self, analyzer):
        score = analyzer.analyze("What is the capital of France?")
        assert score.overall_score < 0.5
        assert score.recommended_depth <= 2

    def test_reasoning_query_medium_complexity(self, analyzer):
        score = analyzer.analyze("If A then B, and B then C, therefore what?")
        assert score.overall_score >= 0.3
        assert score.recommended_depth >= 1

    def test_strategic_query_high_complexity(self, analyzer):
        score = analyzer.analyze("What is the best strategy to evaluate pros and cons?")
        assert score.factors["question_type"] == 0.8
        assert score.overall_score >= 0.3

    def test_complex_markers_increase_score(self, analyzer):
        simple = analyzer.analyze("What is 2+2?")
        complex_q = analyzer.analyze("Multiple various factors, however although despite each of the exceptions")
        assert complex_q.factors["complexity_markers"] > simple.factors["complexity_markers"]

    def test_longer_query_increases_length_factor(self, analyzer):
        short = analyzer.analyze("What is 2+2?")
        long_q = analyzer.analyze("What is the result of adding two plus two when considering all possible mathematical frameworks and historical approaches to arithmetic?")
        assert long_q.factors["length"] >= short.factors["length"]

    def test_negation_factor(self, analyzer):
        no_neg = analyzer.analyze("What is the capital of France?")
        with_neg = analyzer.analyze("Why is it not true that the earth is flat?")
        assert with_neg.factors["negation"] > no_neg.factors["negation"]

    def test_multi_part_factor(self, analyzer):
        simple = analyzer.analyze("What is 2+2?")
        multi = analyzer.analyze("What is 2+2, and also what is 3+3?")
        assert multi.factors["multi_part"] >= simple.factors["multi_part"]

    def test_score_factors_present(self, analyzer):
        score = analyzer.analyze("Test query")
        assert "question_type" in score.factors
        assert "complexity_markers" in score.factors
        assert "length" in score.factors
        assert "negation" in score.factors
        assert "multi_part" in score.factors

    def test_recommended_depth_range(self, analyzer):
        queries = [
            "What is 2+2?",
            "If A then B, therefore C?",
            "What is the best optimal strategy to evaluate multiple pros and cons?",
            "However, although despite this, what should we choose?",
        ]
        for q in queries:
            score = analyzer.analyze(q)
            assert 1 <= score.recommended_depth <= 4
            assert 2 <= score.recommended_tree_depth <= 5

    def test_reasoning_string_present(self, analyzer):
        score = analyzer.analyze("What is 2+2?")
        assert len(score.reasoning) > 0

    def test_overall_score_bounds(self, analyzer):
        queries = [
            "Hi",
            "What is the capital of France?",
            "Why because therefore thus if then would could should compare contrast difference between similar best way optimal strategy should i which would choose decide evaluate pros and cons trade-off alternative multiple several various both each all of the none of the some but not all except unless however although despite not never no n't cannot can't and or ; also",
        ]
        for q in queries:
            score = analyzer.analyze(q)
            assert 0.0 <= score.overall_score <= 1.0
