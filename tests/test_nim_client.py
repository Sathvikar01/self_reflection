"""Unit tests for NVIDIANIMClient."""

import pytest
from unittest.mock import Mock, patch
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
                "choices": [{"message": {"content": "Cached response"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 50, "completion_tokens": 30}
            }
            mock_response.raise_for_status = Mock()
            mock_response.headers = {}
            mock_session.return_value.post.return_value = mock_response
            mock_session.return_value.headers = {}
            
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
            mock_response.content = b'{"error": "rate limit"}'
            
            mock_session.return_value.post.return_value = mock_response
            mock_session.return_value.headers = {}
            
            client = NVIDIANIMClient(api_key="test-key", max_retries=1)
            
            messages = [{"role": "user", "content": "Test"}]
            config = GenerationConfig()
            
            with pytest.raises(Exception):  # Will be APIError after refactoring
                client.generate(messages, config)
    
    @pytest.mark.unit
    def test_generate_api_error(self):
        """Test API error handling."""
        with patch('src.generator.nim_client.requests.Session') as mock_session:
            mock_response = Mock()
            mock_response.status_code = 500
            mock_response.content = b'{"error": "Internal server error"}'
            mock_response.json.return_value = {"error": "Internal server error"}
            mock_response.raise_for_status = Mock(side_effect=Exception("500 Error"))
            
            mock_session.return_value.post.return_value = mock_response
            mock_session.return_value.headers = {}
            
            client = NVIDIANIMClient(api_key="test-key", max_retries=1)
            
            messages = [{"role": "user", "content": "Test"}]
            config = GenerationConfig()
            
            with pytest.raises(Exception):
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
    
    def test_token_tracking_disabled_cache(self, mock_nim_client):
        """Test token tracking with cache disabled."""
        messages = [{"role": "user", "content": "Test"}]
        config = GenerationConfig()
        
        # Generate multiple times
        mock_nim_client.generate(messages, config)
        mock_nim_client.generate(messages, config)
        
        stats = mock_nim_client.get_stats()
        
        # Should track all requests even without caching
        assert stats["total_requests"] == 2
