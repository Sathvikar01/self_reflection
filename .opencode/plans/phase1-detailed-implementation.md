# Phase 1: Testing Infrastructure - Detailed Implementation Plan

## Overview
**Duration**: Week 1-2 (10 working days)
**Goal**: Establish solid testing foundation with 60%+ coverage for critical components
**Approach**: All components equally + Hybrid testing approach + Flat exception hierarchy

---

## Day 1-2: Setup Testing Infrastructure

### 1.1 Create pytest.ini Configuration
**File**: `pytest.ini` (root directory)

```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    -v
    --strict-markers
    --tb=short
    --cov=src
    --cov-report=term-missing
    --cov-report=html:htmlcov
    --cov-fail-under=60
markers =
    unit: Unit tests (fast, isolated)
    integration: Integration tests (slower, may use real components)
    api: Tests requiring API access (mocked or real)
    slow: Slow tests (>1s)
asyncio_mode = auto
```

**Why**: 
- `-v`: Verbose output for better debugging
- `--strict-markers`: Catch typos in marker names
- `--cov`: Enable coverage reporting
- `--cov-fail-under=60`: Enforce minimum 60% coverage
- `asyncio_mode = auto`: Support async tests

### 1.2 Create conftest.py with Shared Fixtures
**File**: `tests/conftest.py`

```python
"""Shared pytest fixtures and configuration."""
import pytest
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import after path setup
from src.generator.nim_client import NVIDIANIMClient, GenerationConfig, GenerationResponse


@pytest.fixture
def mock_api_response():
    """Mock successful API response."""
    return {
        "choices": [{
            "message": {"content": "Test response"},
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 50
        },
        "model": "meta/llama-3.1-8b-instruct",
        "latency_ms": 1500.0
    }


@pytest.fixture
def mock_nim_client(mock_api_response):
    """Mock NVIDIANIMClient for testing."""
    with patch('src.generator.nim_client.requests.Session') as mock_session:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_api_response
        mock_response.raise_for_status = Mock()
        
        mock_session.return_value.post.return_value = mock_response
        
        client = NVIDIANIMClient(
            api_key="test-api-key",
            cache_enabled=False  # Disable cache for tests
        )
        
        yield client


@pytest.fixture
def sample_problem():
    """Sample problem for testing."""
    return {
        "id": "test_001",
        "question": "Do hamsters provide food for any animals?",
        "answer": "yes",
        "answer_type": "yes_no"
    }


@pytest.fixture
def sample_reasoning_chain():
    """Sample reasoning chain for testing."""
    return [
        "Hamsters are small rodents.",
        "Small rodents are prey for many predators.",
        "Therefore, hamsters provide food for some animals."
    ]


@pytest.fixture
def temp_results_dir(tmp_path):
    """Temporary directory for test results."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    return results_dir
```

### 1.3 Create pyproject.toml (Modern Python Project)
**File**: `pyproject.toml`

```toml
[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "self-reflection"
version = "0.1.0"
description = "RL-Guided Self-Reflection for LLM Reasoning"
requires-python = ">=3.9"

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = [
    "-v",
    "--strict-markers",
    "--tb=short",
    "--cov=src",
    "--cov-report=term-missing",
    "--cov-fail-under=60"
]
markers = [
    "unit: Unit tests",
    "integration: Integration tests",
    "api: Tests requiring API access",
    "slow: Slow tests"
]

[tool.coverage.run]
source = ["src"]
omit = [
    "*/tests/*",
    "*/__pycache__/*",
    "*/site-packages/*"
]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise AssertionError",
    "raise NotImplementedError",
    "if __name__ == .__main__.:",
    "if TYPE_CHECKING:",
]
```

---

## Day 3-4: Create Custom Exception Hierarchy

### 2.1 Create exceptions.py Module
**File**: `src/exceptions.py`

```python
"""Custom exception hierarchy for self-reflection system."""

from typing import Optional, Any


class ReflectionError(Exception):
    """Base exception for all self-reflection errors."""
    
    def __init__(self, message: str, details: Optional[dict] = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)
    
    def __str__(self) -> str:
        if self.details:
            return f"{self.message} | Details: {self.details}"
        return self.message


class APIError(ReflectionError):
    """Errors related to API interactions."""
    
    def __init__(
        self, 
        message: str, 
        status_code: Optional[int] = None,
        api_response: Optional[dict] = None
    ):
        details = {"status_code": status_code, "api_response": api_response}
        super().__init__(message, details)
        self.status_code = status_code
        self.api_response = api_response


class GenerationError(APIError):
    """Errors during text generation."""
    
    def __init__(self, message: str, model: Optional[str] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.model = model


class EvaluationError(ReflectionError):
    """Errors during evaluation/scoring."""
    
    def __init__(self, message: str, step: Optional[str] = None):
        details = {"step": step} if step else {}
        super().__init__(message, details)
        self.step = step


class AnswerExtractionError(ReflectionError):
    """Errors during answer extraction."""
    pass


class DataValidationError(ReflectionError):
    """Errors in data validation."""
    
    def __init__(self, message: str, field: Optional[str] = None, value: Any = None):
        details = {"field": field, "value": value}
        super().__init__(message, details)
        self.field = field
        self.value = value


class ConfigurationError(ReflectionError):
    """Errors in configuration."""
    pass


class PipelineError(ReflectionError):
    """Errors in pipeline execution."""
    
    def __init__(self, message: str, stage: Optional[str] = None, **kwargs):
        details = {"stage": stage, **kwargs}
        super().__init__(message, details)
        self.stage = stage
```

### 2.2 Update nim_client.py to Use Custom Exceptions
**File**: `src/generator/nim_client.py`

**Changes needed**:
1. Import custom exceptions: `from src.exceptions import APIError, GenerationError`
2. Replace `raise ValueError(...)` with `raise ConfigurationError(...)`
3. Replace `raise Exception("Rate limited")` with `raise APIError(...)`
4. Replace generic `except Exception` with specific exception handling

**Specific changes**:

```python
# Line 50: Replace ValueError
raise ConfigurationError("NVIDIA API key not provided. Set NVIDIA_API_KEY environment variable or pass api_key parameter.")

# Line 93: Replace generic Exception
raise APIError(
    f"Rate limited. Retry after {retry_after} seconds",
    status_code=429,
    api_response={"retry_after": retry_after}
)

# Line 95: Add error handling
if response.status_code != 200:
    raise APIError(
        f"API request failed with status {response.status_code}",
        status_code=response.status_code,
        api_response=response.json() if response.content else None
    )
```

---

## Day 5-7: Unit Tests for nim_client.py

### 3.1 Create test_nim_client.py
**File**: `tests/test_nim_client.py`

```python
"""Unit tests for NVIDIANIMClient."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import requests
from src.generator.nim_client import NVIDIANIMClient, GenerationConfig, GenerationResponse
from src.exceptions import APIError, ConfigurationError


class TestNVIDIANIMClientInit:
    """Tests for client initialization."""
    
    def test_init_with_api_key(self):
        """Test initialization with explicit API key."""
        client = NVIDIANIMClient(api_key="test-key")
        assert client.api_key == "test-key"
        assert client.cache_enabled is True
        assert client.timeout == 60
    
    def test_init_with_env_var(self, monkeypatch):
        """Test initialization with environment variable."""
        monkeypatch.setenv("NVIDIA_API_KEY", "env-key")
        client = NVIDIANIMClient()
        assert client.api_key == "env-key"
    
    def test_init_without_api_key_raises(self, monkeypatch):
        """Test that missing API key raises ConfigurationError."""
        monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
        with pytest.raises(ConfigurationError, match="NVIDIA API key not provided"):
            NVIDIANIMClient()
    
    def test_custom_parameters(self):
        """Test initialization with custom parameters."""
        client = NVIDIANIMClient(
            api_key="test-key",
            timeout=120,
            max_retries=5,
            cache_enabled=False
        )
        assert client.timeout == 120
        assert client.max_retries == 5
        assert client.cache_enabled is False


class TestNVIDIANIMClientGenerate:
    """Tests for text generation."""
    
    @pytest.mark.unit
    def test_generate_basic(self, mock_nim_client):
        """Test basic generation without caching."""
        messages = [{"role": "user", "content": "Test prompt"}]
        config = GenerationConfig()
        
        response = mock_nim_client.generate(messages, config)
        
        assert isinstance(response, GenerationResponse)
        assert response.text == "Test response"
        assert response.input_tokens == 100
        assert response.output_tokens == 50
        assert response.cached is False
    
    @pytest.mark.unit
    def test_generate_with_cache_hit(self):
        """Test generation with cache hit."""
        with patch('src.generator.nim_client.requests.Session') as mock_session:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "choices": [{"message": {"content": "Cached response"}}],
                "usage": {"prompt_tokens": 50, "completion_tokens": 30}
            }
            mock_session.return_value.post.return_value = mock_response
            
            client = NVIDIANIMClient(api_key="test-key", cache_enabled=True)
            
            messages = [{"role": "user", "content": "Test"}]
            config = GenerationConfig()
            
            # First call
            response1 = client.generate(messages, config)
            assert response1.cached is False
            
            # Second call (should hit cache)
            response2 = client.generate(messages, config)
            assert response2.cached is True
    
    @pytest.mark.unit
    def test_generate_rate_limit_handling(self):
        """Test rate limit error handling."""
        with patch('src.generator.nim_client.requests.Session') as mock_session:
            mock_response = Mock()
            mock_response.status_code = 429
            mock_response.headers = {"Retry-After": "30"}
            
            mock_session.return_value.post.return_value = mock_response
            
            client = NVIDIANIMClient(api_key="test-key")
            
            messages = [{"role": "user", "content": "Test"}]
            config = GenerationConfig()
            
            with pytest.raises(APIError, match="Rate limited"):
                client.generate(messages, config)
    
    @pytest.mark.unit
    def test_generate_api_error(self):
        """Test API error handling."""
        with patch('src.generator.nim_client.requests.Session') as mock_session:
            mock_response = Mock()
            mock_response.status_code = 500
            mock_response.content = b'{"error": "Internal server error"}'
            mock_response.json.return_value = {"error": "Internal server error"}
            
            mock_session.return_value.post.return_value = mock_response
            
            client = NVIDIANIMClient(api_key="test-key")
            
            messages = [{"role": "user", "content": "Test"}]
            config = GenerationConfig()
            
            with pytest.raises(APIError, match="API request failed"):
                client.generate(messages, config)


class TestGenerationConfig:
    """Tests for GenerationConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = GenerationConfig()
        assert config.model == "meta/llama-3.1-8b-instruct"
        assert config.temperature == 0.7
        assert config.max_tokens == 512
        assert config.top_p == 0.95
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = GenerationConfig(
            model="custom-model",
            temperature=0.5,
            max_tokens=1024
        )
        assert config.model == "custom-model"
        assert config.temperature == 0.5
        assert config.max_tokens == 1024


class TestCacheKeyGeneration:
    """Tests for cache key generation."""
    
    def test_cache_key_deterministic(self):
        """Test that cache keys are deterministic."""
        client = NVIDIANIMClient(api_key="test-key")
        messages = [{"role": "user", "content": "Test"}]
        config = GenerationConfig()
        
        key1 = client._get_cache_key(messages, config)
        key2 = client._get_cache_key(messages, config)
        
        assert key1 == key2
    
    def test_cache_key_different_for_different_messages(self):
        """Test that different messages produce different cache keys."""
        client = NVIDIANIMClient(api_key="test-key")
        config = GenerationConfig()
        
        messages1 = [{"role": "user", "content": "Test1"}]
        messages2 = [{"role": "user", "content": "Test2"}]
        
        key1 = client._get_cache_key(messages1, config)
        key2 = client._get_cache_key(messages2, config)
        
        assert key1 != key2


class TestTokenTracking:
    """Tests for token usage tracking."""
    
    def test_token_tracking(self, mock_nim_client):
        """Test that tokens are tracked correctly."""
        messages = [{"role": "user", "content": "Test"}]
        config = GenerationConfig()
        
        # Generate multiple times
        for _ in range(3):
            mock_nim_client.generate(messages, config)
        
        stats = mock_nim_client.get_stats()
        
        assert stats["total_requests"] == 3
        assert stats["total_input_tokens"] == 300  # 100 * 3
        assert stats["total_output_tokens"] == 150  # 50 * 3
```

---

## Day 8-9: Unit Tests for accuracy.py

### 4.1 Create test_accuracy.py
**File**: `tests/test_accuracy.py`

```python
"""Unit tests for answer evaluation and accuracy."""

import pytest
from evaluation.accuracy import AnswerExtractor, AnswerEvaluator, AccuracyCalculator
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
        assert AnswerExtractor.extract("The answer is 42") == "42"
        assert AnswerExtractor.extract("Final: 100.5") == "100.5"
    
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


class TestAccuracyCalculator:
    """Tests for accuracy calculation."""
    
    @pytest.mark.unit
    def test_calculate_accuracy(self):
        """Test accuracy calculation."""
        calculator = AccuracyCalculator()
        
        results = [
            Mock(problem_id="1", correct=True),
            Mock(problem_id="2", correct=True),
            Mock(problem_id="3", correct=False),
            Mock(problem_id="4", correct=True),
        ]
        
        accuracy = calculator.calculate(results)
        
        assert accuracy["total"] == 4
        assert accuracy["correct"] == 3
        assert accuracy["accuracy"] == 0.75
    
    @pytest.mark.unit
    def test_empty_results(self):
        """Test handling of empty results."""
        calculator = AccuracyCalculator()
        accuracy = calculator.calculate([])
        
        assert accuracy["total"] == 0
        assert accuracy["accuracy"] == 0.0
```

---

## Day 10: Integration Tests and Coverage

### 5.1 Create Integration Tests
**File**: `tests/test_integration.py`

```python
"""Integration tests for self-reflection pipeline."""

import pytest
from unittest.mock import Mock, patch
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
    
    def test_pipeline_handles_errors(self, mock_nim_client):
        """Test pipeline error handling."""
        # Make API calls fail
        mock_nim_client.generate.side_effect = Exception("API Error")
        
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
            assert result.confidence >= 0.0
            assert result.confidence <= 1.0
            assert len(result.reasoning_chain) > 0
```

### 5.2 Run Coverage Analysis

```bash
# Run all tests with coverage
pytest --cov=src --cov-report=html --cov-report=term

# Generate coverage report
pytest --cov=src --cov-report=term-missing

# Check specific file coverage
pytest --cov=src/generator/nim_client --cov-report=term-missing tests/test_nim_client.py
```

---

## Success Criteria

### Coverage Targets
- [ ] `nim_client.py`: ≥70% coverage
- [ ] `accuracy.py`: ≥75% coverage  
- [ ] `self_reflection_pipeline.py`: ≥60% coverage
- [ ] `loader.py`: ≥50% coverage
- [ ] Overall: ≥60% coverage

### Quality Checks
- [ ] All tests pass: `pytest -v`
- [ ] Coverage threshold met: `pytest --cov-fail-under=60`
- [ ] No test warnings: `pytest -W error`
- [ ] Custom exceptions used throughout codebase
- [ ] No bare `except Exception` catches in critical paths

### Documentation
- [ ] All test files have docstrings
- [ ] Test classes documented with test purpose
- [ ] Complex test cases have inline comments

---

## Validation Commands

```bash
# Run all tests
pytest

# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_nim_client.py -v

# Run tests in parallel (faster)
pytest -n auto

# Check coverage threshold
pytest --cov-fail-under=60
```

---

## Files to Create

1. `pytest.ini` - Test configuration
2. `tests/conftest.py` - Shared fixtures
3. `pyproject.toml` - Modern Python project config
4. `src/exceptions.py` - Custom exception hierarchy
5. `tests/test_nim_client.py` - API client tests
6. `tests/test_accuracy.py` - Evaluation tests
7. `tests/test_integration.py` - Integration tests

---

## Files to Modify

1. `src/generator/nim_client.py` - Use custom exceptions
2. `src/orchestration/self_reflection_pipeline.py` - Use custom exceptions
3. `evaluation/accuracy.py` - Use custom exceptions
4. `data/datasets/loader.py` - Add validation, use exceptions

---

## Expected Outcome

After completing Phase 1:
- ✅ Comprehensive test suite with 60%+ coverage
- ✅ Custom exception hierarchy for better error handling
- ✅ Shared test fixtures for easier test writing
- ✅ Coverage reporting integrated with CI/CD
- ✅ Foundation for safe refactoring in subsequent phases

**Time to implement**: ~10 working days
**Next phase**: Async & Batch Processing (Phase 2)
