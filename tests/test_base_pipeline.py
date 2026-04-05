"""Tests for BasePipeline abstract base class."""

import pytest
import json
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any

from src.orchestration.base import (
    BasePipeline,
    BaseResult,
    BasePipelineConfig,
    convert_to_serializable,
)


@dataclass
class TestResult(BaseResult):
    """Test result type for testing."""
    final_score: float = 0.0
    extra_field: str = ""


@dataclass
class TestConfig(BasePipelineConfig):
    """Test config type for testing."""
    custom_param: str = "default"


class TestPipeline(BasePipeline[TestResult, TestConfig]):
    """Test pipeline implementation."""
    
    def solve(
        self,
        problem: str,
        problem_id: str = "unknown",
        ground_truth: Optional[str] = None,
    ) -> TestResult:
        """Simple solve implementation for testing."""
        import time
        start_time = time.time()
        
        result = TestResult(
            problem_id=problem_id,
            problem=problem,
            final_answer=f"Answer to: {problem[:30]}",
            correct=self._check_answer("yes", ground_truth) if ground_truth else None,
            ground_truth=ground_truth,
            total_tokens=100,
            latency_seconds=time.time() - start_time,
            final_score=0.85,
            reasoning_path=["Step 1", "Step 2"],
        )
        
        self._results.append(result)
        return result


class TestBaseResult:
    """Tests for BaseResult dataclass."""
    
    def test_base_result_creation(self):
        """Test creating a base result."""
        result = BaseResult(
            problem_id="test_001",
            problem="Test problem",
            final_answer="Test answer",
            correct=True,
            ground_truth="Test answer",
            total_tokens=100,
            latency_seconds=1.5,
        )
        
        assert result.problem_id == "test_001"
        assert result.problem == "Test problem"
        assert result.final_answer == "Test answer"
        assert result.correct is True
        assert result.total_tokens == 100
    
    def test_base_result_to_dict(self):
        """Test serialization to dictionary."""
        result = TestResult(
            problem_id="test_001",
            problem="Test",
            final_answer="Answer",
            total_tokens=50,
            latency_seconds=0.5,
            final_score=0.9,
        )
        
        data = result.to_dict()
        
        assert data["problem_id"] == "test_001"
        assert data["final_score"] == 0.9
        assert "metadata" in data
    
    def test_base_result_default_values(self):
        """Test default values are set correctly."""
        result = BaseResult(
            problem_id="test",
            problem="Test",
            final_answer="Answer",
        )
        
        assert result.correct is None
        assert result.ground_truth is None
        assert result.total_tokens == 0
        assert result.latency_seconds == 0.0
        assert result.reasoning_path == []
        assert result.metadata == {}


class TestBasePipelineConfig:
    """Tests for BasePipelineConfig."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = BasePipelineConfig()
        
        assert config.max_iterations == 50
        assert config.results_dir == "data/results"
        assert config.checkpoint_interval == 10
        assert config.save_intermediate is True
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = TestConfig(
            max_iterations=100,
            custom_param="custom",
        )
        
        assert config.max_iterations == 100
        assert config.custom_param == "custom"


class TestBasePipeline:
    """Tests for BasePipeline abstract class."""
    
    def test_pipeline_initialization(self, tmp_path):
        """Test pipeline initialization."""
        pipeline = TestPipeline(
            config=TestConfig(),
            results_dir=str(tmp_path),
        )
        
        assert pipeline.config.max_iterations == 50
        assert pipeline.results_dir == tmp_path
        assert pipeline._results == []
    
    def test_solve_single_problem(self, tmp_path):
        """Test solving a single problem."""
        pipeline = TestPipeline(
            config=TestConfig(),
            results_dir=str(tmp_path),
        )
        
        result = pipeline.solve(
            problem="What is 2+2?",
            problem_id="math_001",
            ground_truth="yes",
        )
        
        assert result.problem_id == "math_001"
        assert result.problem == "What is 2+2?"
        assert result.final_answer.startswith("Answer to:")
        assert len(pipeline._results) == 1
    
    def test_solve_batch(self, tmp_path):
        """Test solving multiple problems."""
        pipeline = TestPipeline(
            config=TestConfig(checkpoint_interval=2),
            results_dir=str(tmp_path),
        )
        
        problems = [
            {"id": "p1", "problem": "Question 1"},
            {"id": "p2", "problem": "Question 2"},
            {"id": "p3", "problem": "Question 3"},
        ]
        
        results = pipeline.solve_batch(problems)
        
        assert len(results) == 3
        assert results[0].problem_id == "p1"
        assert results[1].problem_id == "p2"
        assert results[2].problem_id == "p3"
    
    def test_check_answer_yes_no(self, tmp_path):
        """Test answer checking for yes/no questions."""
        pipeline = TestPipeline(
            config=TestConfig(),
            results_dir=str(tmp_path),
        )
        
        # Test yes/no answers
        assert pipeline._check_answer("Yes, that's correct", "yes")
        assert pipeline._check_answer("No, that's wrong", "no")
        assert not pipeline._check_answer("Maybe", "yes")
    
    def test_check_answer_exact_match(self, tmp_path):
        """Test answer checking for exact matches."""
        pipeline = TestPipeline(config=TestConfig(), results_dir=str(tmp_path))
        
        assert pipeline._check_answer("Paris", "Paris")
        assert pipeline._check_answer("paris", "Paris")  # Case insensitive
        assert pipeline._check_answer("The answer is Paris", "Paris")  # Contains
    
    def test_compute_aggregate_stats(self, tmp_path):
        """Test aggregate statistics computation."""
        pipeline = TestPipeline(config=TestConfig(), results_dir=str(tmp_path))
        
        # Add some results
        for i in range(3):
            pipeline._results.append(TestResult(
                problem_id=f"p{i}",
                problem=f"Problem {i}",
                final_answer=f"Answer {i}",
                total_tokens=100 + i * 50,  # 100, 150, 200
                latency_seconds=1.0 + i * 0.5,
                correct=i % 2 == 0,  # 2 correct, 1 incorrect
            ))
        
        stats = pipeline._compute_aggregate_stats()
        
        assert stats["total_problems"] == 3
        assert stats["total_tokens"] == 450  # 100 + 150 + 200
        assert stats["total_correct"] == 2
        assert stats["accuracy"] == pytest.approx(2/3)
    
    def test_save_results(self, tmp_path):
        """Test saving results to file."""
        pipeline = TestPipeline(
            config=TestConfig(),
            results_dir=str(tmp_path),
        )
        
        pipeline.solve("Test problem", "test_001")
        pipeline.save_results("test_results.json")
        
        # Verify file exists
        results_file = tmp_path / "test_results.json"
        assert results_file.exists()
        
        # Verify content
        with open(results_file) as f:
            data = json.load(f)
        
        assert data["num_problems"] == 1
        assert "config" in data
        assert "results" in data
        assert "aggregate_stats" in data
    
    def test_get_summary(self, tmp_path):
        """Test summary generation."""
        pipeline = TestPipeline(config=TestConfig(), results_dir=str(tmp_path))
        
        # Empty results
        assert "No results" in pipeline.get_summary()
        
        # With results
        for i in range(2):
            pipeline._results.append(TestResult(
                problem_id=f"p{i}",
                problem=f"Problem {i}",
                final_answer=f"Answer {i}",
                total_tokens=100,
                latency_seconds=1.0,
                correct=True,
            ))
        
        summary = pipeline.get_summary()
        assert "Problems:" in summary
        assert "Tokens:" in summary
    
    def test_context_manager(self, tmp_path):
        """Test using pipeline as context manager."""
        with TestPipeline(config=TestConfig(), results_dir=str(tmp_path)) as pipeline:
            pipeline.solve("Test", "test_001")
        
        # Results should be saved on exit
        assert (tmp_path / "results.json").exists()
    
    def test_reset(self, tmp_path):
        """Test resetting pipeline state."""
        pipeline = TestPipeline(config=TestConfig(), results_dir=str(tmp_path))
        
        pipeline.solve("Test 1", "p1")
        assert len(pipeline._results) == 1
        
        pipeline.reset()
        assert len(pipeline._results) == 0


class TestConvertToSerializable:
    """Tests for convert_to_serializable utility."""
    
    def test_convert_dict(self):
        """Test converting dictionary."""
        data = {"key": "value", "nested": {"a": 1}}
        result = convert_to_serializable(data)
        
        assert result == data
    
    def test_convert_list(self):
        """Test converting list."""
        data = [{"a": 1}, {"b": 2}]
        result = convert_to_serializable(data)
        
        assert result == data
    
    def test_convert_object_with_to_dict(self):
        """Test converting object with to_dict method."""
        result = TestResult(
            problem_id="test",
            problem="Test",
            final_answer="Answer",
        )
        
        converted = convert_to_serializable(result)
        
        assert converted["problem_id"] == "test"
