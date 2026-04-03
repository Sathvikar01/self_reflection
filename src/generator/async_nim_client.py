"""Async NVIDIA NIM API Client for concurrent requests."""

import os
import time
import json
import hashlib
import asyncio
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
import aiohttp
from tenacity import AsyncRetrying, stop_after_attempt, wait_exponential
from loguru import logger
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from exceptions import ConfigurationError, APIError, GenerationError


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


class AsyncNVIDIANIMClient:
    """Async client for NVIDIA NIM API with connection pooling and concurrent requests."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://integrate.api.nvidia.com/v1",
        timeout: int = 60,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        cache_enabled: bool = True,
        max_concurrent: int = 10,
    ):
        self.api_key = api_key or os.getenv("NVIDIA_API_KEY")
        if not self.api_key:
            raise ConfigurationError(
                "NVIDIA API key not provided. Set NVIDIA_API_KEY environment variable or pass api_key parameter."
            )

        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.cache_enabled = cache_enabled
        self.max_concurrent = max_concurrent
        self._cache: Dict[str, GenerationResponse] = {}

        # Session will be created on first use
        self._session: Optional[aiohttp.ClientSession] = None
        self._semaphore = asyncio.Semaphore(max_concurrent)

        # Stats tracking
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._total_requests = 0
        self._total_errors = 0

        logger.info(f"Async NVIDIA NIM Client initialized (max_concurrent={max_concurrent})")

    async def _ensure_session(self):
        """Ensure aiohttp session is created."""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            connector = aiohttp.TCPConnector(
                limit=self.max_concurrent,
                limit_per_host=self.max_concurrent,
                enable_cleanup_closed=True
            )
            self._session = aiohttp.ClientSession(
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=timeout,
                connector=connector
            )

    def _get_cache_key(self, messages: List[Dict], config: GenerationConfig) -> str:
        """Generate cache key for request."""
        content = json.dumps({"messages": messages, "config": config.__dict__}, sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()

    async def _make_request_with_retry(
        self,
        payload: Dict[str, Any],
        attempt: int = 0
    ) -> Dict:
        """Make API request with retry logic."""
        await self._ensure_session()

        try:
            async with self._semaphore:
                async with self._session.post(
                    f"{self.base_url}/chat/completions",
                    json=payload
                ) as response:
                    if response.status == 429:
                        retry_after = int(response.headers.get("Retry-After", 60))
                        logger.warning(f"Rate limited. Waiting {retry_after} seconds...")
                        await asyncio.sleep(retry_after)
                        raise APIError(
                            f"Rate limited. Retry after {retry_after} seconds",
                            status_code=429,
                            api_response={"retry_after": retry_after}
                        )

                    if response.status != 200:
                        error_body = await response.text()
                        raise APIError(
                            f"API request failed with status {response.status}",
                            status_code=response.status,
                            api_response={"error": error_body}
                        )

                    return await response.json()

        except aiohttp.ClientError as e:
            if attempt < self.max_retries - 1:
                wait_time = self.retry_delay * (2 ** attempt)
                logger.warning(f"Request failed, retrying in {wait_time}s: {e}")
                await asyncio.sleep(wait_time)
                return await self._make_request_with_retry(payload, attempt + 1)
            raise APIError(f"Request failed after {self.max_retries} attempts: {e}")

    async def generate(
        self,
        messages: List[Dict[str, str]],
        config: Optional[GenerationConfig] = None
    ) -> GenerationResponse:
        """Generate text completion using chat API (async).

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            config: Generation configuration

        Returns:
            GenerationResponse with generated text and metadata
        """
        if config is None:
            config = GenerationConfig()

        cache_key = None
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
            response = await self._make_request_with_retry(payload)
        except Exception as e:
            self._total_errors += 1
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
            cached=False
        )

        if self.cache_enabled and cache_key:
            self._cache[cache_key] = result
            result.cached = True

        logger.debug(f"Generated {output_tokens} tokens in {latency_ms:.0f}ms")
        return result

    async def generate_batch(
        self,
        requests: List[tuple],
        max_concurrent: Optional[int] = None
    ) -> List[GenerationResponse]:
        """Generate multiple completions concurrently.

        Args:
            requests: List of (messages, config) tuples
            max_concurrent: Override max concurrent requests

        Returns:
            List of GenerationResponse objects in order
        """
        semaphore = asyncio.Semaphore(max_concurrent or self.max_concurrent)

        async def bounded_generate(messages, config):
            async with semaphore:
                return await self.generate(messages, config)

        tasks = [
            bounded_generate(messages, config)
            for messages, config in requests
        ]

        return await asyncio.gather(*tasks, return_exceptions=False)

    async def generate_with_system(
        self,
        system_prompt: str,
        user_prompt: str,
        config: Optional[GenerationConfig] = None
    ) -> GenerationResponse:
        """Generate with separate system and user prompts."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        return await self.generate(messages, config)

    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        return {
            "total_requests": self._total_requests,
            "total_input_tokens": self._total_input_tokens,
            "total_output_tokens": self._total_output_tokens,
            "total_tokens": self._total_input_tokens + self._total_output_tokens,
            "total_errors": self._total_errors,
            "cache_size": len(self._cache),
            "max_concurrent": self.max_concurrent
        }

    def reset_stats(self):
        """Reset usage statistics."""
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._total_requests = 0
        self._total_errors = 0

    def clear_cache(self):
        """Clear response cache."""
        self._cache.clear()
        logger.info("Cache cleared")

    async def close(self):
        """Close the aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()
            logger.info("Session closed")

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
