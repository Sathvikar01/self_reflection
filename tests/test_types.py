"""Tests for generator types."""

import pytest
from src.generator.types import GenerationConfig, GenerationResponse


class TestGenerationConfig:
    """Tests for GenerationConfig dataclass."""

    def test_defaults(self):
        config = GenerationConfig()
        assert config.model == "meta/llama-3.1-8b-instruct"
        assert config.temperature == 0.7
        assert config.max_tokens == 512
        assert config.top_p == 0.95
        assert "\n\n\n" in config.stop_sequences

    def test_custom_values(self):
        config = GenerationConfig(
            model="custom-model",
            temperature=0.5,
            max_tokens=1024,
            top_p=0.9,
            stop_sequences=["STOP"],
        )
        assert config.model == "custom-model"
        assert config.temperature == 0.5
        assert config.max_tokens == 1024
        assert config.top_p == 0.9
        assert config.stop_sequences == ["STOP"]

    def test_stop_sequences_default_factory(self):
        c1 = GenerationConfig()
        c2 = GenerationConfig()
        c1.stop_sequences.append("CUSTOM")
        assert "CUSTOM" not in c2.stop_sequences


class TestGenerationResponse:
    """Tests for GenerationResponse dataclass."""

    def test_creation(self):
        resp = GenerationResponse(
            text="Hello world",
            input_tokens=10,
            output_tokens=5,
            latency_ms=150.0,
            model="test-model",
            finish_reason="stop",
        )
        assert resp.text == "Hello world"
        assert resp.input_tokens == 10
        assert resp.output_tokens == 5
        assert resp.latency_ms == 150.0
        assert resp.model == "test-model"
        assert resp.finish_reason == "stop"
        assert resp.cached is False

    def test_cached_default(self):
        resp = GenerationResponse(
            text="t", input_tokens=1, output_tokens=1,
            latency_ms=1.0, model="m", finish_reason="stop",
        )
        assert resp.cached is False

    def test_cached_true(self):
        resp = GenerationResponse(
            text="t", input_tokens=1, output_tokens=1,
            latency_ms=1.0, model="m", finish_reason="stop", cached=True,
        )
        assert resp.cached is True
