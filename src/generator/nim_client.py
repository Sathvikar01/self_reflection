"""NVIDIA NIM API Client for Base LLM generation."""

import os
import time
import json
import hashlib
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
import requests
from tenacity import retry, stop_after_attempt, wait_exponential
from loguru import logger


@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    model: str = "meta/llama-3.1-8b-instruct"
    temperature: float = 0.7
    max_tokens: int = 512
    top_p: float = 0.95
    stop_sequences: List[str] = field(default_factory=lambda: ["\n\n\n", "Question:", "Problem:"])


@dataclass
class GenerationResponse:
    """Response from generation API."""
    text: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    model: str
    finish_reason: str
    cached: bool = False


class NVIDIANIMClient:
    """Client for NVIDIA NIM API interactions."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://integrate.api.nvidia.com/v1",
        timeout: int = 60,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        cache_enabled: bool = True,
    ):
        self.api_key = api_key or os.getenv("NVIDIA_API_KEY")
        if not self.api_key:
            raise ValueError("NVIDIA API key not provided. Set NVIDIA_API_KEY environment variable or pass api_key parameter.")
        
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.cache_enabled = cache_enabled
        self._cache: Dict[str, GenerationResponse] = {}
        
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        })
        
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._total_requests = 0
        
        logger.info(f"NVIDIA NIM Client initialized with base URL: {base_url}")
    
    def _get_cache_key(self, messages: List[Dict], config: GenerationConfig) -> str:
        """Generate cache key for request."""
        content = json.dumps({"messages": messages, "config": config.__dict__}, sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True
    )
    def _make_request(self, payload: Dict[str, Any]) -> Dict:
        """Make API request with retry logic."""
        response = self._session.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            timeout=self.timeout
        )
        
        if response.status_code == 429:
            retry_after = int(response.headers.get("Retry-After", 60))
            logger.warning(f"Rate limited. Waiting {retry_after} seconds...")
            time.sleep(retry_after)
            raise Exception("Rate limited")
        
        response.raise_for_status()
        return response.json()
    
    def generate(
        self,
        messages: List[Dict[str, str]],
        config: Optional[GenerationConfig] = None,
    ) -> GenerationResponse:
        """Generate text completion using chat API.
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys
            config: Generation configuration
        
        Returns:
            GenerationResponse with generated text and metadata
        """
        if config is None:
            config = GenerationConfig()
        
        if self.cache_enabled:
            cache_key = self._get_cache_key(messages, config)
            if cache_key in self._cache:
                logger.debug("Cache hit for generation request")
                return self._cache[cache_key]
        
        payload = {
            "model": config.model,
            "messages": messages,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "top_p": config.top_p,
            "stop": config.stop_sequences if config.stop_sequences else None,
        }
        
        start_time = time.time()
        
        try:
            response = self._make_request(payload)
        except Exception as e:
            logger.error(f"API request failed: {e}")
            raise
        
        latency_ms = (time.time() - start_time) * 1000
        
        choice = response["choices"][0]
        text = choice["message"]["content"]
        finish_reason = choice.get("finish_reason", "unknown")
        
        usage = response.get("usage", {})
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)
        
        self._total_input_tokens += input_tokens
        self._total_output_tokens += output_tokens
        self._total_requests += 1
        
        result = GenerationResponse(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            model=config.model,
            finish_reason=finish_reason,
            cached=False,
        )
        
        if self.cache_enabled:
            self._cache[cache_key] = result
            result.cached = True
        
        logger.debug(f"Generated {output_tokens} tokens in {latency_ms:.0f}ms")
        return result
    
    def generate_with_system(
        self,
        system_prompt: str,
        user_prompt: str,
        config: Optional[GenerationConfig] = None,
    ) -> GenerationResponse:
        """Generate with separate system and user prompts."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return self.generate(messages, config)
    
    def generate_continuation(
        self,
        conversation_history: List[Dict[str, str]],
        new_content: str,
        config: Optional[GenerationConfig] = None,
    ) -> GenerationResponse:
        """Generate continuation of existing conversation."""
        messages = conversation_history.copy()
        messages.append({"role": "user", "content": new_content})
        return self.generate(messages, config)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        return {
            "total_requests": self._total_requests,
            "total_input_tokens": self._total_input_tokens,
            "total_output_tokens": self._total_output_tokens,
            "total_tokens": self._total_input_tokens + self._total_output_tokens,
            "cache_size": len(self._cache),
        }
    
    def reset_stats(self):
        """Reset usage statistics."""
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._total_requests = 0
    
    def clear_cache(self):
        """Clear response cache."""
        self._cache.clear()
        logger.info("Cache cleared")
    
    def close(self):
        """Close the session."""
        self._session.close()
