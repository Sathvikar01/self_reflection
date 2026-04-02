"""Monte Carlo Tree Search for reasoning."""

import math
import random
from typing import List, Dict, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from loguru import logger

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .tree import StateTree, TreeNode, NodeType
from .actions import ActionExecutor, ActionType, ActionConfig, ActionResult


@dataclass
class MCTSConfig:
    """Configuration for MCTS controller."""
    exploration_constant: float = 1.414
    max_tree_depth: int = 20
    max_rollout_depth: int = 5
    expansion_budget: int = 50
    simulation_budget: int = 100
    temperature: float = 0.7
    backtrack_threshold: float = 0.3
    conclude_threshold: float = 0.85
    gamma: float = 0.95
    
    use_value_network: bool = False
    progressive_widening: bool = True
    pw_factor: float = 0.5


@dataclass
class MCTSStats:
    """Statistics for MCTS run."""
    total_expansions: int = 0
    total_simulations: int = 0
    total_backtracks: int = 0
    total_reflections: int = 0
    nodes_created: int = 0
    max_depth_reached: int = 0
    avg_score: float = 0.0
    best_score: float = float('-inf')
    convergence_step: Optional[int] = None


class MCTSController:
    """Monte Carlo Tree Search controller for reasoning."""
    
    def __init__(
        self,
        action_executor: ActionExecutor,
        config: Optional[MCTSConfig] = None,
    ):
        self.action_executor = action_executor
        self.config = config or MCTSConfig()
        self._stats = MCTSStats()
        self._value_network = None
    
    def search(
        self,
        problem: str,
        max_iterations: Optional[int] = None,
        early_stop_threshold: float = 0.9,
    ) -> Tuple[str, float, List[str]]:
        """Run MCTS to find best solution path.
        
        Args:
            problem: The problem to solve
            max_iterations: Maximum iterations (default: expansion_budget)
            early_stop_threshold: Stop if best score exceeds this
        
        Returns:
            Tuple of (final_answer, score, reasoning_path)
        """
        max_iter = max_iterations or self.config.expansion_budget
        self._stats = MCTSStats()
        
        tree = StateTree(problem, self.config.max_tree_depth)
        current_node = tree.root
        
        logger.info(f"Starting MCTS with {max_iter} iterations")
        
        for iteration in range(max_iter):
            if self._should_stop(current_node, early_stop_threshold):
                logger.info(f"Early stopping at iteration {iteration}")
                break
            
            action = self._select_action(current_node, iteration)
            
            result = self.action_executor.execute(
                action=action,
                problem=problem,
                current_node=current_node,
                temperature=self.config.temperature,
            )
            
            self._update_stats(action, result)
            
            if result.success:
                if action == ActionType.BACKTRACK and result.backtracked_to:
                    current_node = result.backtracked_to
                elif result.new_node:
                    current_node = result.new_node
                    
                    if action == ActionType.CONCLUDE:
                        path = current_node.path_content[1:]
                        return result.content, result.score, path
            else:
                logger.debug(f"Action {action.value} failed: {result.error}")
            
            if current_node.is_terminal:
                best_path = tree.get_best_path()
                path_content = [n.content for n in best_path[1:]]
                return current_node.content, current_node.score, path_content
        
        best_path = tree.get_best_path()
        best_node = best_path[-1]
        
        if best_node.node_type != NodeType.CONCLUSION:
            result = self.action_executor.execute(
                action=ActionType.CONCLUDE,
                problem=problem,
                current_node=best_node,
                temperature=0.3,
            )
            if result.success and result.new_node:
                best_node = result.new_node
        
        return best_node.content, best_node.score, [n.content for n in best_path[1:]]
    
    def _should_stop(self, current_node: TreeNode, threshold: float) -> bool:
        """Check if we should early stop."""
        if current_node.depth < 2:
            return False
        if current_node.score >= threshold:
            return True
        if current_node.is_terminal:
            return True
        if current_node.depth >= self.config.max_tree_depth - 1:
            return True
        return False
    
    def _select_action(self, node: TreeNode, iteration: int) -> ActionType:
        """Select next action using UCB1 or policy."""
        weights = self.action_executor.get_action_weights(node)
        
        if random.random() < 0.1:
            available = [a for a, w in weights.items() if w > 0]
            return random.choice(available)
        
        temperature = max(0.1, self.config.temperature * (1 - iteration / 100))
        
        if random.random() < temperature:
            return self._sample_action(weights)
        else:
            return self._select_best_action(weights, node)
    
    def _sample_action(self, weights: Dict[ActionType, float]) -> ActionType:
        """Sample action from probability distribution."""
        actions = list(weights.keys())
        probs = list(weights.values())
        
        total = sum(probs)
        if total == 0:
            return ActionType.EXPAND
        
        probs = [p / total for p in probs]
        return random.choices(actions, weights=probs, k=1)[0]
    
    def _select_best_action(
        self,
        weights: Dict[ActionType, float],
        node: TreeNode,
    ) -> ActionType:
        """Select best action based on weights and node state."""
        if node.score < self.config.backtrack_threshold and not node.is_root:
            if weights[ActionType.BACKTRACK] > 0:
                return ActionType.BACKTRACK
        
        if node.score > self.config.conclude_threshold and node.depth > 2:
            if weights[ActionType.CONCLUDE] > 0:
                return ActionType.CONCLUDE
        
        return ActionType.EXPAND
    
    def _calculate_ucb1(self, node: TreeNode, parent_visits: int) -> float:
        """Calculate UCB1 value for node."""
        if node.visit_count == 0:
            return float('inf')
        
        exploitation = node.score
        exploration = self.config.exploration_constant * math.sqrt(
            math.log(parent_visits) / node.visit_count
        )
        
        return exploitation + exploration
    
    def _update_stats(self, action: ActionType, result: ActionResult):
        """Update MCTS statistics."""
        if action == ActionType.EXPAND:
            self._stats.total_expansions += 1
            if result.new_node:
                self._stats.nodes_created += 1
                self._stats.max_depth_reached = max(
                    self._stats.max_depth_reached,
                    result.new_node.depth
                )
        elif action == ActionType.REFLECT:
            self._stats.total_reflections += 1
        elif action == ActionType.BACKTRACK:
            self._stats.total_backtracks += 1
        
        if result.score > self._stats.best_score:
            self._stats.best_score = result.score
        
        total_actions = sum([
            self._stats.total_expansions,
            self._stats.total_reflections,
            self._stats.total_backtracks,
        ])
        if total_actions > 0:
            self._stats.avg_score = (
                self._stats.avg_score * (total_actions - 1) + result.score
            ) / total_actions
    
    def get_stats(self) -> MCTSStats:
        """Get current MCTS statistics."""
        return self._stats
    
    def solve_with_budget(
        self,
        problem: str,
        token_budget: int = 10000,
    ) -> Tuple[str, float, List[str], Dict]:
        """Solve problem with token budget constraint.
        
        Args:
            problem: Problem to solve
            token_budget: Maximum tokens to use
        
        Returns:
            Tuple of (answer, score, path, usage_stats)
        """
        tokens_used = 0
        tree = StateTree(problem, self.config.max_tree_depth)
        current_node = tree.root
        
        while tokens_used < token_budget:
            action = self._select_action(current_node, tokens_used)
            
            result = self.action_executor.execute(
                action=action,
                problem=problem,
                current_node=current_node,
            )
            
            tokens_used += result.input_tokens + result.output_tokens
            self._update_stats(action, result)
            
            if result.success:
                if action == ActionType.BACKTRACK and result.backtracked_to:
                    current_node = result.backtracked_to
                elif result.new_node:
                    current_node = result.new_node
                    
                    if action == ActionType.CONCLUDE:
                        return (
                            result.content,
                            result.score,
                            current_node.path_content[1:],
                            {"tokens_used": tokens_used, "stats": self._stats},
                        )
            
            if current_node.is_terminal:
                break
        
        best_path = tree.get_best_path()
        best_node = best_path[-1]
        return (
            best_node.content,
            best_node.score,
            [n.content for n in best_path[1:]],
            {"tokens_used": tokens_used, "stats": self._stats},
        )
    
    def run_ablation(
        self,
        problem: str,
        disabled_actions: List[ActionType],
        max_iterations: int = 50,
    ) -> Tuple[str, float, List[str]]:
        """Run MCTS with certain actions disabled for ablation studies."""
        tree = StateTree(problem, self.config.max_tree_depth)
        current_node = tree.root
        
        for _ in range(max_iterations):
            weights = self.action_executor.get_action_weights(current_node)
            
            for action in disabled_actions:
                weights[action] = 0.0
            
            total = sum(weights.values())
            if total > 0:
                weights = {k: v / total for k, v in weights.items()}
            else:
                weights = {ActionType.EXPAND: 1.0}
            
            action = self._sample_action(weights)
            
            result = self.action_executor.execute(
                action=action,
                problem=problem,
                current_node=current_node,
            )
            
            if result.success:
                if action == ActionType.BACKTRACK and result.backtracked_to:
                    current_node = result.backtracked_to
                elif result.new_node:
                    current_node = result.new_node
                    
                    if action == ActionType.CONCLUDE:
                        return result.content, result.score, current_node.path_content[1:]
            
            if current_node.is_terminal:
                break
        
        best_path = tree.get_best_path()
        best_node = best_path[-1]
        return best_node.content, best_node.score, [n.content for n in best_path[1:]]
