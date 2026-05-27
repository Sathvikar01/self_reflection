"""MiMo API Client for Xiaomi's reasoning model."""

import os
import time
import json
import hashlib
from typing import Optional, Dict, Any, List
from collections import OrderedDict
import requests
from tenacity import retry, stop_after_attempt, wait_exponential
from loguru import logger

from ..exceptions import ConfigurationError, APIError
from .types import GenerationConfig, GenerationResponse


class MiMoClient:
    """Client for Xiaomi MiMo API (OpenAI-compatible with reasoning_content support)."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = "mimo-v2.5-pro",
        timeout: int = 600,
        max_retries: int = 3,
        cache_enabled: bool = True,
    ):
        self.api_key = api_key or os.getenv("MIMO_API_KEY")
        if not self.api_key:
            raise ConfigurationError(
                "MiMo API key not provided. Set MIMO_API_KEY environment variable."
            )

        self.base_url = base_url or os.getenv("MIMO_BASE_URL", "https://token-plan-sgp.xiaomimimo.com/v1")
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.cache_enabled = cache_enabled
        self._cache: OrderedDict[str, GenerationResponse] = OrderedDict()
        self._cache_max_size = 1000

        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        })

        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._total_reasoning_tokens = 0
        self._total_requests = 0

        logger.info(f"MiMo Client initialized: {self.base_url}, model={self.model}")

    def _get_cache_key(self, messages: List[Dict], config: GenerationConfig) -> str:
        content = json.dumps({"messages": messages, "config": config.__dict__}, sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True
    )
    def _make_request(self, payload: Dict[str, Any]) -> Dict:
        response = self._session.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            timeout=self.timeout
        )
        if response.status_code == 429:
            retry_after = int(response.headers.get("Retry-After", 60))
            logger.warning(f"Rate limited. Waiting {retry_after}s...")
            time.sleep(retry_after)
            raise APIError(f"Rate limited", status_code=429)
        response.raise_for_status()
        return response.json()

    def generate(
        self, messages: List[Dict[str, str]], config: Optional[GenerationConfig] = None
    ) -> GenerationResponse:
        if config is None:
            config = GenerationConfig(model=self.model)

        cache_key = None
        if self.cache_enabled:
            cache_key = self._get_cache_key(messages, config)
            if cache_key in self._cache:
                self._cache.move_to_end(cache_key)
                return self._cache[cache_key]

        payload = {
            "model": config.model,
            "messages": messages,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "top_p": config.top_p,
        }

        start_time = time.time()
        try:
            response = self._make_request(payload)
        except Exception as e:
            logger.error(f"MiMo API request failed: {e}")
            raise

        latency_ms = (time.time() - start_time) * 1000

        choice = response["choices"][0]
        message = choice["message"]
        text = message.get("content", "")
        reasoning = message.get("reasoning_content", "")
        finish_reason = choice.get("finish_reason", "unknown")

        usage = response.get("usage", {})
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)
        reasoning_tokens = usage.get("completion_tokens_details", {}).get("reasoning_tokens", 0)

        self._total_input_tokens += input_tokens
        self._total_output_tokens += output_tokens
        self._total_reasoning_tokens += reasoning_tokens
        self._total_requests += 1

        # Combine reasoning + content for the full response
        full_text = text
        if reasoning and not text.strip():
            # If content is empty but reasoning exists, use reasoning
            full_text = reasoning
        elif reasoning and text.strip():
            # Prepend reasoning as thinking block
            full_text = f"<thinking>\n{reasoning}\n</thinking>\n{text}"

        result = GenerationResponse(
            text=full_text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            model=config.model,
            finish_reason=finish_reason,
            cached=False,
        )

        if self.cache_enabled and cache_key:
            self._cache[cache_key] = result
            self._cache.move_to_end(cache_key)
            if len(self._cache) > self._cache_max_size:
                self._cache.popitem(last=False)
            result.cached = True

        return result

    def generate_with_system(
        self,
        system_prompt: str,
        user_prompt: str,
        config: Optional[GenerationConfig] = None,
    ) -> GenerationResponse:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return self.generate(messages, config)

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_requests": self._total_requests,
            "total_input_tokens": self._total_input_tokens,
            "total_output_tokens": self._total_output_tokens,
            "total_reasoning_tokens": self._total_reasoning_tokens,
            "total_tokens": self._total_input_tokens + self._total_output_tokens,
            "cache_size": len(self._cache),
        }

    def reset_stats(self):
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._total_reasoning_tokens = 0
        self._total_requests = 0

    def clear_cache(self):
        self._cache.clear()

    def close(self):
        self._session.close()
