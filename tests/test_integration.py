"""Integration tests for self-reflection pipeline."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from src.orchestration.self_reflection_pipeline import (
    SelfReflectionPipeline,
    SelfReflectionConfig,
    SelfReflectionResult
)


@pytest.mark.integration
class TestSelfReflectionPipeline:
    """Integration tests for the pipeline."""
    
    def test_solve_simple_problem(self, mock_nim_client):
        """Test solving a simple yes/no problem."""
        config = SelfReflectionConfig(
            max_iterations=3,
            min_reasoning_steps=1,
            max_reasoning_steps=2,
            reflection_depth=1
        )
        
        with patch('src.orchestration.self_reflection_pipeline.NVIDIANIMClient', return_value=mock_nim_client):
            pipeline = SelfReflectionPipeline(config=config)
            
            result = pipeline.solve(
                problem="Do hamsters provide food for any animals?",
                problem_id="test_001",
                ground_truth="yes"
            )
            
            assert isinstance(result, SelfReflectionResult)
            assert result.problem_id == "test_001"
            assert result.final_answer is not None
            assert len(result.reasoning_chain) > 0
    
    def test_pipeline_handles_errors_gracefully(self, mock_nim_client):
        """Test pipeline error handling."""
        # Make API calls fail
        mock_nim_client.generate = MagicMock(side_effect=Exception("API Error"))
        
        config = SelfReflectionConfig(max_iterations=1)
        
        with patch('src.orchestration.self_reflection_pipeline.NVIDIANIMClient', return_value=mock_nim_client):
            pipeline = SelfReflectionPipeline(config=config)
            
            # Should handle error gracefully
            result = pipeline.solve(
                problem="Test problem",
                problem_id="test_001"
            )
            
            # Pipeline should return a result even on error
            assert isinstance(result, SelfReflectionResult)
    
    def test_pipeline_respects_config(self):
        """Test that pipeline respects configuration."""
        config = SelfReflectionConfig(
            max_iterations=5,
            min_reasoning_steps=2,
            max_reasoning_steps=3,
            temperature_reason=0.5,
            temperature_reflect=0.3
        )
        
        pipeline = SelfReflectionPipeline(config=config)
        
        assert pipeline.config.max_iterations == 5
        assert pipeline.config.min_reasoning_steps == 2
        assert pipeline.config.max_reasoning_steps == 3
        assert pipeline.config.temperature_reason == 0.5
        assert pipeline.config.temperature_reflect == 0.3


@pytest.mark.integration
class TestEndToEnd:
    """End-to-end tests."""
    
    @pytest.mark.slow
    def test_full_pipeline_execution(self, mock_nim_client, sample_problem):
        """Test full pipeline execution from problem to result."""
        with patch('src.orchestration.self_reflection_pipeline.NVIDIANIMClient', return_value=mock_nim_client):
            pipeline = SelfReflectionPipeline()
            
            result = pipeline.solve(
                problem=sample_problem["question"],
                problem_id=sample_problem["id"],
                ground_truth=sample_problem["answer"]
            )
            
            # Verify result structure
            assert result.problem_id == sample_problem["id"]
            assert result.problem == sample_problem["question"]
            assert result.final_answer is not None
            assert 0.0 <= result.confidence <= 1.0
            assert len(result.reasoning_chain) > 0
    
    @pytest.mark.slow
    def test_pipeline_with_reflection(self, mock_nim_client):
        """Test pipeline with reflection enabled."""
        config = SelfReflectionConfig(
            reflection_depth=2,
            enable_selective_reflection=False
        )
        
        with patch('src.orchestration.self_reflection_pipeline.NVIDIANIMClient', return_value=mock_nim_client):
            pipeline = SelfReflectionPipeline(config=config)
            
            result = pipeline.solve(
                problem="Test problem requiring reflection",
                problem_id="test_reflect_001"
            )
            
            assert isinstance(result, SelfReflectionResult)
            assert len(result.reflections) > 0


@pytest.mark.integration
class TestPipelineMetrics:
    """Tests for pipeline metrics tracking."""
    
    def test_token_tracking(self, mock_nim_client):
        """Test that tokens are tracked."""
        config = SelfReflectionConfig(max_iterations=2)
        
        with patch('src.orchestration.self_reflection_pipeline.NVIDIANIMClient', return_value=mock_nim_client):
            pipeline = SelfReflectionPipeline(config=config)
            
            result = pipeline.solve(
                problem="Test problem",
                problem_id="test_001"
            )
            
            # Should track tokens
            assert result.total_tokens >= 0
    
    def test_latency_tracking(self, mock_nim_client):
        """Test that latency is tracked."""
        config = SelfReflectionConfig(max_iterations=1)
        
        with patch('src.orchestration.self_reflection_pipeline.NVIDIANIMClient', return_value=mock_nim_client):
            pipeline = SelfReflectionPipeline(config=config)
            
            result = pipeline.solve(
                problem="Test problem",
                problem_id="test_001"
            )
            
            # Should track latency
            assert result.latency_seconds >= 0


@pytest.mark.integration
class TestProblemClassification:
    """Tests for problem type classification."""
    
    def test_factual_problem_classification(self, mock_nim_client):
        """Test classification of factual problems."""
        with patch('src.orchestration.self_reflection_pipeline.NVIDIANIMClient', return_value=mock_nim_client):
            pipeline = SelfReflectionPipeline()
            
            result = pipeline.solve(
                problem="What is the capital of France?",
                problem_id="test_factual_001"
            )
            
            assert result.problem_type in ["factual", "reasoning", "strategic"]
    
    def test_reasoning_problem_classification(self, mock_nim_client):
        """Test classification of reasoning problems."""
        with patch('src.orchestration.self_reflection_pipeline.NVIDIANIMClient', return_value=mock_nim_client):
            pipeline = SelfReflectionPipeline()
            
            result = pipeline.solve(
                problem="If A is greater than B and B is greater than C, what is the relationship between A and C?",
                problem_id="test_reasoning_001"
            )
            
            assert result.problem_type in ["factual", "reasoning", "strategic"]
