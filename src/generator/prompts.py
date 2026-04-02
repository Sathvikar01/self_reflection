"""Prompt templates for reasoning tasks."""

from typing import List, Dict, Optional
from dataclasses import dataclass
from enum import Enum


class PromptType(Enum):
    """Types of prompts for different reasoning stages."""
    BASELINE = "baseline"
    STEP_EXPAND = "step_expand"
    REFLECT = "reflect"
    CONCLUDE = "conclude"
    EVALUATE_STEP = "evaluate_step"
    SELF_CORRECT = "self_correct"


@dataclass
class ReasoningContext:
    """Context for building reasoning prompts."""
    problem: str
    previous_steps: List[str]
    current_step: Optional[str] = None
    question_type: str = "general"


class PromptBuilder:
    """Builder for various prompt templates used in reasoning."""
    
    SYSTEM_PROMPT_BASE = """You are a careful and methodical problem solver. You break down complex problems into clear, logical steps. Each step should follow naturally from the previous one and build toward the final answer. Be precise and avoid making unsupported assumptions."""

    SYSTEM_PROMPT_MATH = """You are an expert mathematician. Solve problems by breaking them down into clear algebraic steps. Show your work explicitly, including all calculations. Check each step for errors before proceeding."""

    SYSTEM_PROMPT_REASONING = """You are an expert at multi-step logical reasoning. For each problem, identify the key facts and logical relationships needed. Work through the reasoning step by step, verifying each inference before proceeding to the next."""
    
    EVALUATOR_SYSTEM_PROMPT = """You are a reasoning quality evaluator. Your task is to score the logical validity of a single reasoning step. Output ONLY a single numerical score, no explanation."""

    @staticmethod
    def get_system_prompt(question_type: str = "general") -> str:
        """Get appropriate system prompt based on question type."""
        if "math" in question_type.lower():
            return PromptBuilder.SYSTEM_PROMPT_MATH
        elif "reasoning" in question_type.lower():
            return PromptBuilder.SYSTEM_PROMPT_REASONING
        return PromptBuilder.SYSTEM_PROMPT_BASE
    
    @staticmethod
    def build_baseline_prompt(problem: str, question_type: str = "general") -> List[Dict[str, str]]:
        """Build prompt for baseline zero-shot evaluation."""
        system_prompt = PromptBuilder.get_system_prompt(question_type)
        user_prompt = f"""Solve the following problem step by step. Show your reasoning clearly.

Problem: {problem}

Work through this problem carefully, explaining each step of your reasoning. End with a clear final answer."""
        
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    
    @staticmethod
    def build_expand_prompt(context: ReasoningContext) -> List[Dict[str, str]]:
        """Build prompt for expanding with next reasoning step."""
        system_prompt = PromptBuilder.get_system_prompt("reasoning")
        
        steps_text = ""
        if context.previous_steps:
            steps_text = "\n\nPrevious reasoning steps:\n"
            for i, step in enumerate(context.previous_steps, 1):
                steps_text += f"{i}. {step}\n"
        
        user_prompt = f"""Problem: {context.problem}
{steps_text}
What is the next logical step in solving this problem? Continue from where the reasoning left off. Provide only the next step, not the entire solution."""
        
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    
    @staticmethod
    def build_reflect_prompt(context: ReasoningContext) -> List[Dict[str, str]]:
        """Build prompt for reflection on previous step."""
        system_prompt = PromptBuilder.get_system_prompt("reasoning")
        
        if not context.previous_steps:
            raise ValueError("Cannot reflect without previous steps")
        
        last_step = context.previous_steps[-1]
        previous_steps_text = "\n".join(f"{i+1}. {s}" for i, s in enumerate(context.previous_steps[:-1]))
        
        user_prompt = f"""Problem: {context.problem}

Previous reasoning:
{previous_steps_text}

Last step: {last_step}

Wait, let me carefully review the last step for potential errors or inconsistencies. Is this step logically sound? Are there any hidden assumptions or potential mistakes?

Provide a brief critique of the last step and, if needed, a corrected version."""
        
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    
    @staticmethod
    def build_conclude_prompt(context: ReasoningContext) -> List[Dict[str, str]]:
        """Build prompt for generating final conclusion."""
        system_prompt = PromptBuilder.get_system_prompt("reasoning")
        
        steps_text = "\n".join(f"{i+1}. {step}" for i, step in enumerate(context.previous_steps))
        
        user_prompt = f"""Problem: {context.problem}

Complete reasoning:
{steps_text}

Based on the reasoning above, what is the final answer? Provide a clear, concise answer."""
        
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    
    @staticmethod
    def build_evaluation_prompt(
        problem: str,
        previous_steps: List[str],
        current_step: str,
    ) -> List[Dict[str, str]]:
        """Build prompt for evaluating a single reasoning step."""
        context_text = ""
        if previous_steps:
            context_text = "Previous steps:\n" + "\n".join(f"- {s}" for s in previous_steps)
        
        user_prompt = f"""Evaluate the logical validity of the current reasoning step.

Problem: {problem}

{context_text}

Current step to evaluate: {current_step}

Score this step from -1.0 to 1.0:
- 1.0: Completely correct and logically sound
- 0.5: Mostly correct with minor issues
- 0.0: Neutral or uncertain
- -0.5: Has logical issues or errors
- -1.0: Completely incorrect or irrelevant

Output ONLY the numerical score, nothing else."""
        
        return [
            {"role": "system", "content": PromptBuilder.EVALUATOR_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]
    
    @staticmethod
    def build_self_correct_prompt(
        problem: str,
        previous_steps: List[str],
        error_step: str,
        error_description: str,
    ) -> List[Dict[str, str]]:
        """Build prompt for self-correction after detected error."""
        system_prompt = PromptBuilder.get_system_prompt("reasoning")
        
        steps_before_error = previous_steps[:-1] if previous_steps else []
        context_text = "\n".join(f"{i+1}. {s}" for i, s in enumerate(steps_before_error))
        
        user_prompt = f"""Problem: {problem}

Correct reasoning so far:
{context_text}

An error was identified in this step: {error_step}
Issue: {error_description}

Please provide a corrected version of this step, or an alternative approach if needed."""
        
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    
    @staticmethod
    def build_tree_search_prompt(
        problem: str,
        path: List[str],
        action: str = "expand",
    ) -> List[Dict[str, str]]:
        """Build prompt for tree search with specific action."""
        system_prompt = PromptBuilder.get_system_prompt("reasoning")
        
        path_text = "\n".join(f"Step {i+1}: {step}" for i, step in enumerate(path))
        
        if action == "expand":
            instruction = "What is the next logical step?"
        elif action == "reflect":
            instruction = "Review the last step for potential errors."
        elif action == "conclude":
            instruction = "Based on all steps, what is the final answer?"
        else:
            instruction = "Continue reasoning."
        
        user_prompt = f"""Problem: {problem}

Reasoning path so far:
{path_text}

{instruction}"""
        
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    
    @staticmethod
    def format_conversation_history(
        messages: List[Dict[str, str]]
    ) -> str:
        """Format messages for display or logging."""
        formatted = []
        for msg in messages:
            role = msg["role"].upper()
            content = msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"]
            formatted.append(f"{role}: {content}")
        return "\n".join(formatted)
