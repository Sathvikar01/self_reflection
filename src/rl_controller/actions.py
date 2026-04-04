"""Action execution for MCTS reasoning."""

from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import time
from loguru import logger

from ..generator.nim_client import NVIDIANIMClient, GenerationConfig
from ..generator.prompts import PromptBuilder, ReasoningContext
from ..evaluator.prm_client import PRMEvaluator
from ..evaluator.improved_prm import ImprovedPRM
from ..utils.lru_cache import CachedPRMEvaluator
from .tree import TreeNode, NodeType


class ActionType(Enum):
    """Available actions in MCTS."""
    EXPAND = "expand"
    REFLECT = "reflect"
    BACKTRACK = "backtrack"
    CONCLUDE = "conclude"


@dataclass
class ActionConfig:
    """Configuration for action execution."""
    expand_temperature: float = 0.7
    reflect_temperature: float = 0.5
    conclude_temperature: float = 0.3
    max_expand_tokens: int = 256
    max_reflect_tokens: int = 256
    max_conclude_tokens: int = 128
    min_steps_before_conclude: int = 3
    backtrack_threshold: float = 0.3
    reflect_on_low_score: bool = True
    low_score_threshold: float = 0.2


@dataclass
class ActionResult:
    """Result of executing an action."""
    success: bool
    action_type: ActionType
    content: str
    score: float
    new_node: Optional[TreeNode] = None
    backtracked_to: Optional[TreeNode] = None
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ActionExecutor:
    """Executes actions in the reasoning process."""

    def __init__(
        self,
        generator: NVIDIANIMClient,
        evaluator: Any = None,
        config: Optional[ActionConfig] = None,
        use_cache: bool = True,
        use_persistent_cache: bool = False,
    ):
        self.generator = generator
        self.config = config or ActionConfig()
        
        if use_cache and evaluator is not None:
            self.evaluator = CachedPRMEvaluator(evaluator, use_persistent=use_persistent_cache)
            logger.info("CachedPRMEvaluator enabled for action execution")
        else:
            self.evaluator = evaluator

        self._action_counts = {a: 0 for a in ActionType}
        self._total_tokens = {"input": 0, "output": 0}
    
    def execute(
        self,
        action: ActionType,
        problem: str,
        current_node: TreeNode,
        temperature: Optional[float] = None,
    ) -> ActionResult:
        """Execute an action and return result."""
        self._action_counts[action] += 1
        start_time = time.time()
        
        try:
            if action == ActionType.EXPAND:
                result = self._execute_expand(problem, current_node, temperature)
            elif action == ActionType.REFLECT:
                result = self._execute_reflect(problem, current_node, temperature)
            elif action == ActionType.BACKTRACK:
                result = self._execute_backtrack(problem, current_node)
            elif action == ActionType.CONCLUDE:
                result = self._execute_conclude(problem, current_node, temperature)
            else:
                result = ActionResult(
                    success=False,
                    action_type=action,
                    content="",
                    score=0.0,
                    error=f"Unknown action: {action}",
                )
        except Exception as e:
            logger.error(f"Action {action.value} failed: {e}")
            result = ActionResult(
                success=False,
                action_type=action,
                content="",
                score=0.0,
                error=str(e),
            )
        
        result.latency_ms = (time.time() - start_time) * 1000
        self._total_tokens["input"] += result.input_tokens
        self._total_tokens["output"] += result.output_tokens
        
        return result
    
    def _execute_expand(
        self,
        problem: str,
        current_node: TreeNode,
        temperature: Optional[float] = None,
    ) -> ActionResult:
        """Execute EXPAND action - generate next reasoning step."""
        temp = temperature or self.config.expand_temperature

        context = ReasoningContext(
            problem=problem,
            previous_steps=current_node.path_content[1:],
        )

        messages = PromptBuilder.build_expand_prompt(context)
        gen_config = GenerationConfig(
            temperature=temp,
            max_tokens=self.config.max_expand_tokens,
        )

        response = self.generator.generate(messages, gen_config)
        new_step = response.text.strip()

        previous_steps = current_node.path_content[1:]
        
        if self.evaluator:
            if isinstance(self.evaluator, ImprovedPRM):
                eval_result = self.evaluator.evaluate_step(problem, previous_steps, new_step, current_node.depth)
            else:
                eval_result = self.evaluator.evaluate_step(problem, previous_steps, new_step)
        else:
            from types import SimpleNamespace
            eval_result = SimpleNamespace(
                score=0.5,
                confidence=0.5,
                input_tokens=0,
                output_tokens=0
            )

        new_node = current_node.add_child(
            content=new_step,
            node_type=NodeType.STEP,
            score=eval_result.score,
            action_taken=ActionType.EXPAND.value,
        )

        return ActionResult(
            success=True,
            action_type=ActionType.EXPAND,
            content=new_step,
            score=eval_result.score,
            new_node=new_node,
            input_tokens=response.input_tokens + getattr(eval_result, 'input_tokens', 0),
            output_tokens=response.output_tokens + getattr(eval_result, 'output_tokens', 0),
            metadata={"eval_confidence": getattr(eval_result, 'confidence', 0.5)},
        )
    
    def _execute_reflect(
        self,
        problem: str,
        current_node: TreeNode,
        temperature: Optional[float] = None,
    ) -> ActionResult:
        """Execute REFLECT action - critique previous step."""
        temp = temperature or self.config.reflect_temperature
        
        previous_steps = current_node.path_content[1:]
        if not previous_steps:
            return ActionResult(
                success=False,
                action_type=ActionType.REFLECT,
                content="",
                score=0.0,
                error="Cannot reflect without previous steps",
            )
        
        context = ReasoningContext(
            problem=problem,
            previous_steps=previous_steps,
        )
        
        messages = PromptBuilder.build_reflect_prompt(context)
        gen_config = GenerationConfig(
            temperature=temp,
            max_tokens=self.config.max_reflect_tokens,
        )
        
        response = self.generator.generate(messages, gen_config)
        reflection = response.text.strip()
        
        new_node = current_node.add_child(
            content=reflection,
            node_type=NodeType.REFLECTION,
            score=0.0,
            action_taken=ActionType.REFLECT.value,
        )
        
        return ActionResult(
            success=True,
            action_type=ActionType.REFLECT,
            content=reflection,
            score=0.0,
            new_node=new_node,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            metadata={"reflection_type": "self_critique"},
        )
    
    def _execute_backtrack(
        self,
        problem: str,
        current_node: TreeNode,
    ) -> ActionResult:
        """Execute BACKTRACK action - move to better sibling or parent."""
        if current_node.is_root:
            return ActionResult(
                success=False,
                action_type=ActionType.BACKTRACK,
                content="",
                score=0.0,
                error="Cannot backtrack from root",
            )
        
        best_sibling = current_node.get_best_sibling()
        
        if best_sibling and best_sibling.score > current_node.score:
            target = best_sibling
        elif current_node.parent:
            target = current_node.parent
        else:
            target = current_node
        
        return ActionResult(
            success=True,
            action_type=ActionType.BACKTRACK,
            content=f"Backtracked from depth {current_node.depth} to {target.depth}",
            score=target.score,
            backtracked_to=target,
            metadata={"from_depth": current_node.depth, "to_depth": target.depth},
        )
    
    def _execute_conclude(
        self,
        problem: str,
        current_node: TreeNode,
        temperature: Optional[float] = None,
    ) -> ActionResult:
        """Execute CONCLUDE action - generate final answer."""
        temp = temperature or self.config.conclude_temperature
        
        previous_steps = current_node.path_content[1:]
        
        if len(previous_steps) < self.config.min_steps_before_conclude:
            return ActionResult(
                success=False,
                action_type=ActionType.CONCLUDE,
                content="",
                score=0.0,
                error=f"Need at least {self.config.min_steps_before_conclude} steps before concluding",
            )
        
        context = ReasoningContext(
            problem=problem,
            previous_steps=previous_steps,
        )
        
        messages = PromptBuilder.build_conclude_prompt(context)
        gen_config = GenerationConfig(
            temperature=temp,
            max_tokens=self.config.max_conclude_tokens,
        )
        
        response = self.generator.generate(messages, gen_config)
        conclusion = response.text.strip()
        
        new_node = current_node.add_child(
            content=conclusion,
            node_type=NodeType.CONCLUSION,
            score=current_node.score,
            action_taken=ActionType.CONCLUDE.value,
        )
        new_node.is_terminal = True
        
        return ActionResult(
            success=True,
            action_type=ActionType.CONCLUDE,
            content=conclusion,
            score=current_node.score,
            new_node=new_node,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            metadata={"is_terminal": True},
        )
    
    def get_action_weights(
        self,
        current_node: TreeNode,
        force_action: Optional[ActionType] = None,
    ) -> Dict[ActionType, float]:
        """Get probability weights for each action."""
        if force_action:
            return {a: 1.0 if a == force_action else 0.0 for a in ActionType}
        
        weights = {
            ActionType.EXPAND: 0.4,
            ActionType.REFLECT: 0.25,
            ActionType.BACKTRACK: 0.2,
            ActionType.CONCLUDE: 0.15,
        }
        
        if current_node.is_root:
            weights[ActionType.BACKTRACK] = 0.0
            weights[ActionType.REFLECT] = 0.0
            weights[ActionType.EXPAND] = 0.85
        else:
            if current_node.score < self.config.backtrack_threshold:
                weights[ActionType.BACKTRACK] = 0.5
                weights[ActionType.REFLECT] = 0.3
                weights[ActionType.EXPAND] = 0.15
                weights[ActionType.CONCLUDE] = 0.05
            
            previous_steps = current_node.path_content[1:]
            if len(previous_steps) >= self.config.min_steps_before_conclude:
                weights[ActionType.CONCLUDE] += 0.1
                weights[ActionType.EXPAND] -= 0.1
        
        if current_node.depth > 10:
            weights[ActionType.CONCLUDE] += 0.2
            weights[ActionType.EXPAND] -= 0.2
        
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}
        
        return weights
    
    def get_stats(self) -> Dict:
        """Get action execution statistics."""
        stats = {
            "action_counts": self._action_counts.copy(),
            "total_input_tokens": self._total_tokens["input"],
            "total_output_tokens": self._total_tokens["output"],
            "total_tokens": sum(self._total_tokens.values()),
        }
        
        if hasattr(self.evaluator, 'get_cache_stats'):
            stats["cache_stats"] = self.evaluator.get_cache_stats()
        
        return stats

    def reset_stats(self):
        """Reset statistics."""
        self._action_counts = {a: 0 for a in ActionType}
        self._total_tokens = {"input": 0, "output": 0}
