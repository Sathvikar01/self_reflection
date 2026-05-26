"""Shared types for generator module."""

from dataclasses import dataclass, field
from typing import List


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
