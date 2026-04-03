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
        mock_response.headers = {}
        
        mock_session.return_value.post.return_value = mock_response
        mock_session.return_value.headers = {}
        
        client = NVIDIANIMClient(
            api_key="test-api-key",
            cache_enabled=False
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


@pytest.fixture
def mock_generation_response():
    """Mock generation response."""
    return GenerationResponse(
        text="Test response text",
        input_tokens=100,
        output_tokens=50,
        latency_ms=1500.0,
        model="meta/llama-3.1-8b-instruct",
        finish_reason="stop",
        cached=False
    )
