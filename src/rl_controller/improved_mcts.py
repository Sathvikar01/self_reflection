"""Improved MCTS with always-explore backtracking and better learning."""

import math
import random
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from loguru import logger

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .tree import StateTree, TreeNode, NodeType
from .actions import ActionExecutor, ActionType, ActionResult, ActionConfig


@dataclass
class ImprovedMCTSConfig:
    """Configuration for improved MCTS controller."""
    exploration_constant: float = 1.414
    max_tree_depth: int = 20
    expansion_budget: int = 50
    
    base_backtrack_prob: float = 0.25
    max_depth_bonus: float = 0.15
    score_exploration_factor: float = 0.5
    
    min_steps_before_conclude: int = 3
    min_steps_before_backtrack: int = 1
    
    success_bonus: float = 0.2
    vacuous_penalty: float = 0.3
    
    keep_best_path: bool = True
    compare_alternatives: bool = True


@dataclass
class PathResult:
    """Result of exploring a reasoning path."""
    path: List[TreeNode]
    final_answer: str
    final_score: float
    is_correct: Optional[bool] = None
    prm_scores: List[float] = field(default_factory=list)
    steps: List[str] = field(default_factory=list)


class ImprovedMCTSController:
    """MCTS with always-explore backtracking and proper learning."""
    
    def __init__(
        self,
        action_executor: ActionExecutor,
        verifier: Any = None,
        config: Optional[ImprovedMCTSConfig] = None,
    ):
        self.action_executor = action_executor
        self.verifier = verifier
        self.config = config or ImprovedMCTSConfig()
        
        self._explored_paths: List[PathResult] = []
        self._best_path: Optional[PathResult] = None
        self._stats = {
            "total_expansions": 0,
            "total_backtracks": 0,
            "total_reflections": 0,
            "paths_explored": 0,
            "successful_comparisons": 0,
        }
    
    def search(
        self,
        problem: str,
        max_iterations: int = None,
        early_stop_threshold: float = 0.95,
    ) -> Tuple[str, float, List[str]]:
        """Run improved MCTS with exploration and comparison."""

        max_iter = max_iterations or self.config.expansion_budget

        tree = StateTree(problem, self.config.max_tree_depth)
        current_node = tree.root

        logger.info(f"Starting improved MCTS with {max_iter} iterations")

        for iteration in range(max_iter):
            if current_node.depth >= self.config.min_steps_before_conclude:
                action = ActionType.CONCLUDE
            else:
                action = self._select_action_with_exploration(current_node, iteration)

            if action == ActionType.BACKTRACK:
                backtrack_result = self._execute_backtrack_with_learning(
                    problem, current_node, tree
                )
                if backtrack_result:
                    current_node = backtrack_result
                    self._stats["total_backtracks"] += 1
                continue

            result = self.action_executor.execute(
                action=action,
                problem=problem,
                current_node=current_node,
            )

            self._update_stats(action, result)

            if result.success and result.new_node:
                current_node = result.new_node

            if action == ActionType.CONCLUDE:
                path_result = self._record_path(tree, current_node, problem)

                if self.config.compare_alternatives and iteration < max_iter - 5:
                    logger.info("Exploring alternative paths for comparison")
                    continue
                else:
                    return result.content, result.score, current_node.path_content[1:]
        
        best_result = self._get_best_result(problem)

        if best_result:
            return best_result.final_answer, best_result.final_score, best_result.steps

        best_path = tree.get_best_path()
        best_node = best_path[-1]
        reasoning_steps = [n.content for n in best_path[1:] if n.content]
        
        from generator.prompts import PromptBuilder, ReasoningContext
        from generator.nim_client import GenerationConfig
        
        context = ReasoningContext(problem=problem, previous_steps=reasoning_steps)
        messages = PromptBuilder.build_conclude_prompt(context)
        config = GenerationConfig(temperature=0.3, max_tokens=100)
        answer_response = self.action_executor.generator.generate(messages, config)
        final_answer = answer_response.text.strip()
        
        return final_answer, best_node.score, reasoning_steps
    
    def _select_action_with_exploration(
        self,
        node: TreeNode,
        iteration: int,
    ) -> ActionType:
        """Select action with probabilistic exploration including backtracking."""
        
        weights = self.action_executor.get_action_weights(node)
        
        backtrack_prob = self._calculate_backtrack_probability(node, iteration)
        
        if random.random() < backtrack_prob:
            if node.depth >= self.config.min_steps_before_backtrack:
                logger.debug(f"Probabilistic backtrack triggered (prob={backtrack_prob:.2f})")
                return ActionType.BACKTRACK
        
        if node.depth < self.config.min_steps_before_conclude:
            weights[ActionType.CONCLUDE] = 0.0
        
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}
        
        return self._sample_action(weights)
    
    def _calculate_backtrack_probability(self, node: TreeNode, iteration: int) -> float:
        """Calculate probability of backtracking - KEY INSIGHT: always some probability."""

        base_prob = self.config.base_backtrack_prob

        depth_factor = min(node.depth, 5) * 0.02

        score_factor = self.config.score_exploration_factor * (1.0 - node.score * 0.2)

        iteration_factor = 0.05 if iteration < 5 else 0.02

        if node.depth >= self.config.min_steps_before_conclude:
            depth_factor *= 0.5

        total_prob = base_prob + depth_factor + score_factor + iteration_factor

        return min(0.35, total_prob)
    
    def _execute_backtrack_with_learning(
        self,
        problem: str,
        current_node: TreeNode,
        tree: StateTree,
    ) -> Optional[TreeNode]:
        """Execute backtrack and record what we learn."""
        
        alternatives = []
        
        if current_node.parent:
            for sibling in current_node.parent.children:
                if sibling != current_node:
                    alternatives.append(sibling)
        
        if alternatives:
            best_alternative = max(alternatives, key=lambda n: n.score)
            
            if best_alternative.score > current_node.score - 0.1:
                logger.debug(
                    f"Backtracking: current={current_node.score:.2f}, "
                    f"alternative={best_alternative.score:.2f}"
                )
                return best_alternative
        
        if current_node.parent:
            logger.debug("Backtracking to parent for exploration")
            return current_node.parent
        
        return None
    
    def _record_path(
        self,
        tree: StateTree,
        terminal_node: TreeNode,
        problem: str,
    ) -> PathResult:
        """Record explored path for comparison."""
        
        path = terminal_node.path_to_root
        steps = [n.content for n in path[1:] if n.content]
        prm_scores = [n.score for n in path[1:]]
        
        result = PathResult(
            path=path[1:],
            final_answer=terminal_node.content,
            final_score=terminal_node.score,
            prm_scores=prm_scores,
            steps=steps,
        )
        
        self._explored_paths.append(result)
        
        if self._best_path is None or result.final_score > self._best_path.final_score:
            self._best_path = result
        
        self._stats["paths_explored"] += 1
        
        return result
    
    def _get_best_result(self, problem: str) -> Optional[PathResult]:
        """Get best result from all explored paths."""
        
        if not self._explored_paths:
            return None
        
        if self.verifier and len(self._explored_paths) > 1:
            logger.info("Verifying multiple paths to find best")
            
            for path_result in self._explored_paths:
                verification = self.verifier.verify_answer(
                    problem=problem,
                    reasoning=path_result.steps,
                    answer=path_result.final_answer,
                )
                path_result.is_correct = verification.score > 0.5
            
            correct_paths = [p for p in self._explored_paths if p.is_correct]
            if correct_paths:
                self._stats["successful_comparisons"] += 1
                return max(correct_paths, key=lambda p: p.final_score)
        
        return max(self._explored_paths, key=lambda p: p.final_score)
    
    def _sample_action(self, weights: Dict[ActionType, float]) -> ActionType:
        """Sample action from probability distribution."""
        actions = list(weights.keys())
        probs = list(weights.values())
        
        total = sum(probs)
        if total == 0:
            return ActionType.EXPAND
        
        probs = [p / total for p in probs]
        return random.choices(actions, weights=probs, k=1)[0]
    
    def _update_stats(self, action: ActionType, result: ActionResult):
        """Update statistics."""
        if action == ActionType.EXPAND:
            self._stats["total_expansions"] += 1
        elif action == ActionType.REFLECT:
            self._stats["total_reflections"] += 1
    
    def get_stats(self) -> Dict:
        """Get controller statistics."""
        return self._stats.copy()
    
    def get_learning_summary(self) -> str:
        """Get summary of what was learned from path comparisons."""
        lines = ["MCTS Learning Summary:"]
        lines.append(f"  Paths explored: {self._stats['paths_explored']}")
        lines.append(f"  Total backtracks: {self._stats['total_backtracks']}")
        lines.append(f"  Successful comparisons: {self._stats['successful_comparisons']}")
        
        if self._best_path:
            lines.append(f"  Best path score: {self._best_path.final_score:.2f}")
            lines.append(f"  Best path steps: {len(self._best_path.steps)}")
        
        return "\n".join(lines)
