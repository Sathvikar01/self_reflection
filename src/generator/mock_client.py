"""Mock NVIDIA NIM Client for testing without API."""

import time
import random
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field


@dataclass
class MockGenerationResponse:
    """Mock response from generation API."""
    text: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    model: str
    finish_reason: str
    cached: bool = False


class MockNVIDIANIMClient:
    """Mock client for testing without real API."""
    
    SAMPLE_ANSWERS = [
        "Yes, this is correct because the reasoning follows logically from the premises.",
        "No, this cannot be true based on the facts provided.",
        "The answer depends on additional context not provided.",
        "Let me think step by step. First, we need to consider the main factors involved.",
    ]
    
    REASONING_STEPS = [
        "First, I need to identify the key components of this problem.",
        "The relevant facts here are that we're dealing with a comparison scenario.",
        "This step follows from the previous observation about the relationship.",
        "Looking at the evidence, we can conclude that the relationship holds.",
        "The logical next step is to verify this conclusion against the constraints.",
    ]
    
    SCORES = [0.8, 0.6, 0.9, -0.3, 0.5, 0.7, -0.1, 0.85, 0.4, 0.75]
    
    def __init__(self, api_key: Optional[str] = None, **kwargs):
        self.api_key = api_key or "mock_key"
        self._call_count = 0
        self._total_input_tokens = 0
        self._total_output_tokens = 0
    
    def generate(self, messages: List[Dict], config: Any = None) -> MockGenerationResponse:
        """Generate mock response."""
        self._call_count += 1
        
        time.sleep(random.uniform(0.1, 0.3))
        
        last_message = messages[-1]["content"] if messages else ""
        
        if "score" in last_message.lower() or "evaluate" in last_message.lower():
            text = str(random.choice(self.SCORES))
            output_tokens = random.randint(5, 20)
        elif "next" in last_message.lower() or "step" in last_message.lower():
            text = random.choice(self.REASONING_STEPS)
            output_tokens = random.randint(30, 80)
        elif "final" in last_message.lower() or "conclude" in last_message.lower():
            text = random.choice(self.SAMPLE_ANSWERS)
            output_tokens = random.randint(20, 50)
        else:
            text = random.choice(self.SAMPLE_ANSWERS)
            output_tokens = random.randint(50, 150)
        
        input_tokens = sum(len(m["content"].split()) for m in messages)
        
        self._total_input_tokens += input_tokens
        self._total_output_tokens += output_tokens
        
        return MockGenerationResponse(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=random.uniform(100, 500),
            model="mock-model",
            finish_reason="stop",
        )
    
    def generate_with_system(self, system_prompt: str, user_prompt: str, config: Any = None) -> MockGenerationResponse:
        """Generate with separate prompts."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return self.generate(messages, config)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        return {
            "total_requests": self._call_count,
            "total_input_tokens": self._total_input_tokens,
            "total_output_tokens": self._total_output_tokens,
            "total_tokens": self._total_input_tokens + self._total_output_tokens,
        }
    
    def reset_stats(self):
        """Reset statistics."""
        self._call_count = 0
        self._total_input_tokens = 0
        self._total_output_tokens = 0
    
    def clear_cache(self):
        """Clear cache (no-op for mock)."""
        pass
    
    def close(self):
        """Close (no-op for mock)."""
        pass
