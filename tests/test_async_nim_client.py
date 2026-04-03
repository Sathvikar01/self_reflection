"""Unit tests for async NIM client."""

import pytest
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from src.generator.async_nim_client import (
    AsyncNVIDIANIMClient,
    GenerationConfig,
    GenerationResponse
)
from src.exceptions import ConfigurationError, APIError


class TestAsyncNVIDIANIMClientInit:
    """Tests for async client initialization."""

    def test_init_with_api_key(self):
        """Test initialization with explicit API key."""
        client = AsyncNVIDIANIMClient(api_key="test-key")
        assert client.api_key == "test-key"
        assert client.cache_enabled is True
        assert client.timeout == 60
        assert client.max_concurrent == 10

    def test_init_with_env_var(self, monkeypatch):
        """Test initialization with environment variable."""
        monkeypatch.setenv("NVIDIA_API_KEY", "env-key")
        client = AsyncNVIDIANIMClient()
        assert client.api_key == "env-key"

    def test_init_without_api_key_raises(self, monkeypatch):
        """Test that missing API key raises ConfigurationError."""
        monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
        with pytest.raises(ConfigurationError, match="NVIDIA API key not provided"):
            AsyncNVIDIANIMClient()

    def test_custom_parameters(self):
        """Test initialization with custom parameters."""
        client = AsyncNVIDIANIMClient(
            api_key="test-key",
            timeout=120,
            max_concurrent=20,
            cache_enabled=False
        )
        assert client.timeout == 120
        assert client.max_concurrent == 20
        assert client.cache_enabled is False


class TestAsyncNVIDIANIMClientGenerate:
    """Tests for async text generation."""

    @pytest.mark.asyncio
    async def test_generate_basic(self):
        """Test basic async generation."""
        mock_response_data = {
            "choices": [{
                "message": {"content": "Test response"},
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50
            }
        }

        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value=mock_response_data)
            mock_response.headers = {}

            mock_session.post = MagicMock(return_value=mock_response)
            mock_session.__aenter__ = AsyncMock(return_value=mock_response)
            mock_session.__aexit__ = AsyncMock()
            mock_session_class.return_value = mock_session

            client = AsyncNVIDIANIMClient(api_key="test-key")
            messages = [{"role": "user", "content": "Test"}]
            config = GenerationConfig()

            result = await client.generate(messages, config)

            assert isinstance(result, GenerationResponse)
            assert result.text == "Test response"
            assert result.input_tokens == 100
            assert result.output_tokens == 50

            await client.close()

    @pytest.mark.asyncio
    async def test_generate_with_cache_hit(self):
        """Test async generation with cache hit."""
        client = AsyncNVIDIANIMClient(api_key="test-key", cache_enabled=True)

        # Pre-populate cache
        messages = [{"role": "user", "content": "Test"}]
        config = GenerationConfig()
        cache_key = client._get_cache_key(messages, config)
        cached_response = GenerationResponse(
            text="Cached",
            input_tokens=10,
            output_tokens=5,
            latency_ms=100,
            model="test-model",
            finish_reason="stop",
            cached=True
        )
        client._cache[cache_key] = cached_response

        result = await client.generate(messages, config)

        assert result.text == "Cached"
        assert result.cached is True

        await client.close()


class TestAsyncBatchGenerate:
    """Tests for batch generation."""

    @pytest.mark.asyncio
    async def test_generate_batch(self):
        """Test batch generation."""
        mock_response_data = {
            "choices": [{
                "message": {"content": "Batch response"},
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 50,
                "completion_tokens": 30
            }
        }

        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value=mock_response_data)
            mock_response.headers = {}

            mock_session.post = MagicMock(return_value=mock_response)
            mock_session.__aenter__ = AsyncMock(return_value=mock_response)
            mock_session.__aexit__ = AsyncMock()
            mock_session_class.return_value = mock_session

            client = AsyncNVIDIANIMClient(api_key="test-key")

            requests = [
                ([{"role": "user", "content": "Test1"}], GenerationConfig()),
                ([{"role": "user", "content": "Test2"}], GenerationConfig()),
                ([{"role": "user", "content": "Test3"}], GenerationConfig()),
            ]

            results = await client.generate_batch(requests, max_concurrent=2)

            assert len(results) == 3
            for result in results:
                assert isinstance(result, GenerationResponse)
                assert result.text == "Batch response"

            await client.close()


class TestAsyncClientStats:
    """Tests for statistics tracking."""

    def test_get_stats(self):
        """Test stats retrieval."""
        client = AsyncNVIDIANIMClient(api_key="test-key")
        client._total_requests = 10
        client._total_input_tokens = 1000
        client._total_output_tokens = 500
        client._total_errors = 2

        stats = client.get_stats()

        assert stats["total_requests"] == 10
        assert stats["total_input_tokens"] == 1000
        assert stats["total_output_tokens"] == 500
        assert stats["total_errors"] == 2
        assert stats["max_concurrent"] == 10

    def test_reset_stats(self):
        """Test stats reset."""
        client = AsyncNVIDIANIMClient(api_key="test-key")
        client._total_requests = 10
        client._total_errors = 2

        client.reset_stats()

        assert client._total_requests == 0
        assert client._total_errors == 0


class TestAsyncClientCache:
    """Tests for caching."""

    def test_cache_key_generation(self):
        """Test cache key is deterministic."""
        client = AsyncNVIDIANIMClient(api_key="test-key")
        messages = [{"role": "user", "content": "Test"}]
        config = GenerationConfig()

        key1 = client._get_cache_key(messages, config)
        key2 = client._get_cache_key(messages, config)

        assert key1 == key2

    def test_cache_clear(self):
        """Test cache clearing."""
        client = AsyncNVIDIANIMClient(api_key="test-key")
        client._cache["test_key"] = Mock()

        client.clear_cache()

        assert len(client._cache) == 0
