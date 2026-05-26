"""Tests for prompt builder module."""

import pytest
from src.generator.prompts import (
    PromptBuilder,
    PromptType,
    ReasoningContext,
)


class TestPromptType:
    """Tests for PromptType enum."""

    def test_values(self):
        assert PromptType.BASELINE.value == "baseline"
        assert PromptType.STEP_EXPAND.value == "step_expand"
        assert PromptType.REFLECT.value == "reflect"
        assert PromptType.CONCLUDE.value == "conclude"
        assert PromptType.EVALUATE_STEP.value == "evaluate_step"
        assert PromptType.SELF_CORRECT.value == "self_correct"


class TestReasoningContext:
    """Tests for ReasoningContext dataclass."""

    def test_creation(self):
        ctx = ReasoningContext(problem="test problem", previous_steps=["step1"])
        assert ctx.problem == "test problem"
        assert ctx.previous_steps == ["step1"]
        assert ctx.current_step is None
        assert ctx.question_type == "general"

    def test_custom_values(self):
        ctx = ReasoningContext(
            problem="math problem",
            previous_steps=["s1", "s2"],
            current_step="s3",
            question_type="math",
        )
        assert ctx.question_type == "math"
        assert ctx.current_step == "s3"


class TestPromptBuilder:
    """Tests for PromptBuilder static methods."""

    def test_get_system_prompt_general(self):
        prompt = PromptBuilder.get_system_prompt("general")
        assert "careful" in prompt.lower() or "methodical" in prompt.lower()

    def test_get_system_prompt_math(self):
        prompt = PromptBuilder.get_system_prompt("math")
        assert "mathematician" in prompt.lower() or "math" in prompt.lower()

    def test_get_system_prompt_reasoning(self):
        prompt = PromptBuilder.get_system_prompt("reasoning")
        assert "reasoning" in prompt.lower()

    def test_build_baseline_prompt(self):
        messages = PromptBuilder.build_baseline_prompt("What is 2+2?")
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert "What is 2+2?" in messages[1]["content"]

    def test_build_baseline_prompt_math_type(self):
        messages = PromptBuilder.build_baseline_prompt("Solve x+1=3", question_type="math")
        assert "mathematician" in messages[0]["content"].lower()

    def test_build_expand_prompt(self):
        ctx = ReasoningContext(
            problem="test",
            previous_steps=["step 1"],
        )
        messages = PromptBuilder.build_expand_prompt(ctx)
        assert len(messages) == 2
        assert "test" in messages[1]["content"]
        assert "step 1" in messages[1]["content"]

    def test_build_expand_prompt_no_steps(self):
        ctx = ReasoningContext(problem="test", previous_steps=[])
        messages = PromptBuilder.build_expand_prompt(ctx)
        assert len(messages) == 2

    def test_build_reflect_prompt(self):
        ctx = ReasoningContext(
            problem="test",
            previous_steps=["step 1", "step 2"],
        )
        messages = PromptBuilder.build_reflect_prompt(ctx)
        assert len(messages) == 2
        assert "step 2" in messages[1]["content"]

    def test_build_reflect_prompt_no_steps_raises(self):
        ctx = ReasoningContext(problem="test", previous_steps=[])
        with pytest.raises(ValueError, match="Cannot reflect"):
            PromptBuilder.build_reflect_prompt(ctx)

    def test_build_conclude_prompt(self):
        ctx = ReasoningContext(
            problem="test",
            previous_steps=["step 1", "step 2"],
        )
        messages = PromptBuilder.build_conclude_prompt(ctx)
        assert len(messages) == 2
        assert "final answer" in messages[1]["content"].lower()

    def test_build_evaluation_prompt(self):
        messages = PromptBuilder.build_evaluation_prompt(
            problem="test",
            previous_steps=["step 1"],
            current_step="step 2",
        )
        assert len(messages) == 2
        assert "step 2" in messages[1]["content"]
        assert "-1.0" in messages[1]["content"]

    def test_build_evaluation_prompt_no_previous(self):
        messages = PromptBuilder.build_evaluation_prompt(
            problem="test",
            previous_steps=[],
            current_step="step 1",
        )
        assert len(messages) == 2

    def test_build_self_correct_prompt(self):
        messages = PromptBuilder.build_self_correct_prompt(
            problem="test",
            previous_steps=["step 1", "step 2"],
            error_step="step 2",
            error_description="Wrong calculation",
        )
        assert len(messages) == 2
        assert "Wrong calculation" in messages[1]["content"]

    def test_build_tree_search_prompt_expand(self):
        messages = PromptBuilder.build_tree_search_prompt(
            problem="test",
            path=["step 1"],
            action="expand",
        )
        assert len(messages) == 2
        assert "next logical step" in messages[1]["content"].lower()

    def test_build_tree_search_prompt_reflect(self):
        messages = PromptBuilder.build_tree_search_prompt(
            problem="test",
            path=["step 1"],
            action="reflect",
        )
        assert "review" in messages[1]["content"].lower() or "errors" in messages[1]["content"].lower()

    def test_build_tree_search_prompt_conclude(self):
        messages = PromptBuilder.build_tree_search_prompt(
            problem="test",
            path=["step 1"],
            action="conclude",
        )
        assert "final answer" in messages[1]["content"].lower()

    def test_build_tree_search_prompt_unknown(self):
        messages = PromptBuilder.build_tree_search_prompt(
            problem="test",
            path=["step 1"],
            action="unknown",
        )
        assert len(messages) == 2

    def test_format_conversation_history(self):
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        formatted = PromptBuilder.format_conversation_history(messages)
        assert "SYSTEM:" in formatted
        assert "USER:" in formatted

    def test_format_conversation_history_truncates(self):
        long_content = "x" * 200
        messages = [{"role": "user", "content": long_content}]
        formatted = PromptBuilder.format_conversation_history(messages)
        assert "..." in formatted
