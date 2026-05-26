"""Tests for selective reflection feature."""

import pytest
from src.orchestration.self_reflection_pipeline import (
    SelfReflectionPipeline,
    SelfReflectionConfig,
)


class TestProblemClassification:
    """Tests for problem type classification."""

    @pytest.fixture
    def pipeline(self):
        config = SelfReflectionConfig(enable_selective_reflection=True)
        return SelfReflectionPipeline(config=config)

    def test_factual_questions_classified(self, pipeline):
        factual_questions = [
            "What is the capital of France?",
            "How many planets are in the solar system?",
            "Who was the first president?",
            "Where is the Eiffel Tower?",
            "Define photosynthesis.",
        ]
        for q in factual_questions:
            result = pipeline._classify_problem_type(q)
            assert result == "factual", f"Expected 'factual' for: {q}"

    def test_reasoning_questions_classified(self, pipeline):
        reasoning_questions = [
            "Do hamsters provide food for any animals?",
            "Why does ice float on water?",
        ]
        for q in reasoning_questions:
            result = pipeline._classify_problem_type(q)
            assert result in ("reasoning", "factual"), f"Unexpected type for: {q}"

    def test_strategic_questions_classified(self, pipeline):
        strategic_questions = [
            "Should I invest in stocks or bonds?",
            "Which would be better, A or B?",
            "How should I approach this problem?",
        ]
        for q in strategic_questions:
            result = pipeline._classify_problem_type(q)
            assert result == "strategic", f"Expected 'strategic' for: {q}, got {result}"


class TestBaselineConfidence:
    """Tests for baseline confidence calculation."""

    @pytest.fixture
    def pipeline(self):
        config = SelfReflectionConfig(enable_selective_reflection=True)
        return SelfReflectionPipeline(config=config)

    def test_high_confidence_reasoning(self, pipeline):
        high_confidence_reasoning = [
            "First, I need to understand what the question is asking about the sun's brightness compared to artificial light sources.",
            "The key facts are: the sun is definitely a star producing energy through nuclear fusion, and light bulbs are artificial sources.",
            "Therefore, based on the enormous energy output of the sun versus a simple light bulb, the sun is clearly much brighter.",
        ]
        high_conf = pipeline._calculate_baseline_confidence(high_confidence_reasoning)
        assert high_conf >= 0.5

    def test_low_confidence_reasoning(self, pipeline):
        low_confidence_reasoning = [
            "Maybe.",
            "Not sure.",
        ]
        low_conf = pipeline._calculate_baseline_confidence(low_confidence_reasoning)
        high_conf = pipeline._calculate_baseline_confidence([
            "First, I need to understand what the question is asking about the sun's brightness compared to artificial light sources in detail.",
            "The key facts are: the sun is definitely a star producing enormous energy through nuclear fusion, and light bulbs are tiny artificial sources.",
            "Therefore, based on the enormous energy output of the sun versus a simple light bulb, the sun is clearly much brighter by many orders of magnitude.",
        ])
        assert low_conf < high_conf

    def test_empty_reasoning_returns_zero(self, pipeline):
        assert pipeline._calculate_baseline_confidence([]) == 0.0

    def test_short_reasoning_low_confidence(self, pipeline):
        conf = pipeline._calculate_baseline_confidence(["Short."])
        assert conf <= 0.5


class TestReflectionDepths:
    """Tests for reflection depth configuration."""

    def test_reflection_depths_config(self):
        config = SelfReflectionConfig(
            enable_selective_reflection=True,
            reflection_depths={
                "factual": 1,
                "reasoning": 2,
                "strategic": 3,
            },
        )
        assert config.reflection_depths["factual"] == 1
        assert config.reflection_depths["reasoning"] == 2
        assert config.reflection_depths["strategic"] == 3

    def test_confidence_threshold(self):
        config = SelfReflectionConfig(confidence_threshold_skip=0.9)
        assert config.confidence_threshold_skip == 0.9


class TestSelectiveReflectionIntegration:
    """Tests for selective reflection integration config."""

    def test_selective_reflection_enabled(self):
        config = SelfReflectionConfig(
            enable_selective_reflection=True,
            confidence_threshold_skip=0.9,
        )
        assert config.enable_selective_reflection is True
        assert config.confidence_threshold_skip == 0.9
        assert "factual" in config.reflection_depths
        assert "reasoning" in config.reflection_depths
        assert "strategic" in config.reflection_depths

    def test_default_reflection_depths(self):
        config = SelfReflectionConfig(enable_selective_reflection=True)
        assert config.reflection_depths["factual"] == 1
        assert config.reflection_depths["reasoning"] == 2
        assert config.reflection_depths["strategic"] == 3
