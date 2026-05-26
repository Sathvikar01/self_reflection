"""Enhanced Adaptive Self-Reflection Pipeline with Tree Search.

This pipeline combines the best of both worlds:
1. Adaptive depth and complexity analysis from AdaptiveReflectionPipeline
2. Tree expansion and backtracking from RL-Guided MCTS
3. UCB1 action selection for optimal exploration/exploitation

Key Features:
- Query complexity analysis determines initial tree depth
- Tree expansion creates multiple reasoning paths
- UCB1 selection balances exploration vs exploitation
- Backtracking in tree (not just rollback)
- Overfitting prevention via cross-validation
- Adaptive reflection depth
"""

import os
import json
import time
import math
import random
import hashlib
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from pathlib import Path
from enum import Enum
from dotenv import load_dotenv
from loguru import logger

from ..generator.nim_client import NVIDIANIMClient, GenerationConfig
from ..utils.unified_extractor import UnifiedAnswerExtractor
from ..rl_controller.tree import TreeNode, NodeType
from .base import BasePipeline, BaseResult, BasePipelineConfig
from ..utils.complexity import QueryComplexityAnalyzer, ComplexityScore

load_dotenv()


class ActionType(Enum):
    """Actions available in tree search."""
    EXPAND = "expand"
    REFLECT = "reflect"
    BACKTRACK = "backtrack"
    CONCLUDE = "conclude"


@dataclass
class AdaptiveTreeConfig(BasePipelineConfig):
    """Configuration for adaptive tree-based reflection."""

    # Complexity thresholds
    low_complexity_threshold: float = 0.3
    high_complexity_threshold: float = 0.7

    # Tree search parameters
    max_tree_depth: int = 6
    max_tree_width: int = 3
    exploration_constant: float = 1.414

    # Adaptive behavior
    min_reflections: int = 1
    max_reflections: int = 5
    confidence_threshold_increase: float = 0.7
    confidence_threshold_stop: float = 0.9
    degradation_threshold: float = 0.1

    # Backtracking
    backtrack_threshold: float = 0.35
    max_backtracks: int = 5

    # Expansion
    expansion_budget: int = 20
    progressive_widening: bool = True
    pw_factor: float = 0.5

    # Overfitting prevention
    enable_cross_validation: bool = True
    validation_samples: int = 3
    variance_threshold: float = 0.2

    # Early stopping
    early_stopping_patience: int = 2

    # Temperatures
    temperature_reason: float = 0.7
    temperature_reflect: float = 0.3
    temperature_conclude: float = 0.2


@dataclass
class AdaptiveTreeResult(BaseResult):
    """Result from adaptive tree-based reflection."""

    # Tree metrics
    complexity_score: float = 0.0
    recommended_depth: int = 0
    actual_depth: int = 0

    # Tree expansion metrics
    total_expansions: int = 0
    total_backtracks: int = 0
    max_tree_depth_reached: int = 0
    nodes_created: int = 0

    # Adaptive metrics
    rolled_back: bool = False
    rollback_step: int = 0
    overfitting_detected: bool = False

    # Cross-validation
    cv_answers: List[str] = field(default_factory=list)
    cv_confidence_std: float = 0.0

    reflections: List[str] = field(default_factory=list)
    confidence: float = 0.0


class AdaptiveTreeReflectionPipeline(BasePipeline[AdaptiveTreeResult, AdaptiveTreeConfig]):
    """Pipeline combining adaptive reflection with tree search."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        config: Optional[AdaptiveTreeConfig] = None,
        results_dir: str = "data/results",
    ):
        load_dotenv()
        config = config or AdaptiveTreeConfig()
        super().__init__(config=config, results_dir=results_dir)

        api_key = api_key or os.getenv("NVIDIA_API_KEY")
        self.generator = NVIDIANIMClient(api_key=api_key)
        self.generator_model = "meta/llama-3.1-8b-instruct"

        self.complexity_analyzer = QueryComplexityAnalyzer()

        # Tree metrics
        self._total_expansions = 0
        self._total_backtracks = 0

        logger.info("Adaptive Tree Reflection Pipeline initialized")
    
    def solve(
        self,
        problem: str,
        problem_id: str = "unknown",
        ground_truth: Optional[str] = None,
    ) -> AdaptiveTreeResult:
        """Solve with adaptive tree-based reflection."""
        start_time = time.time()
        
        logger.info(f"[{problem_id}] Starting adaptive tree reflection")
        
        # Step 1: Analyze complexity
        complexity = self.complexity_analyzer.analyze(problem)
        logger.info(f"[{problem_id}] Complexity: {complexity.overall_score:.2f}, tree depth: {complexity.recommended_tree_depth}")
        
        # Step 2: Initialize tree
        root = TreeNode(
            id="root",
            content=problem,
            depth=0,
            node_type=NodeType.ROOT,
            created_at=time.time()
        )

        # Step 3: Build initial reasoning
        initial_reasoning = self._generate_initial_reasoning(problem)
        for step in initial_reasoning:
            root.add_child(step, NodeType.STEP)
        
        # Step 4: Tree search with UCB1
        current_node = root.children[-1] if root.children else root
        total_expansions = 0
        total_backtracks = 0
        nodes_created = len(root.children)
        max_depth = len(root.children)
        
        visited_nodes = set()
        max_iterations = self.config.expansion_budget
        
        for iteration in range(max_iterations):
            # Early stopping
            if current_node.score >= self.config.confidence_threshold_stop:
                logger.info(f"[{problem_id}] High confidence reached")
                break
            
            # Check depth limit
            if current_node.depth >= self.config.max_tree_depth:
                # Backtrack if too deep
                if current_node.parent and current_node.parent.parent:
                    current_node = self._backtrack(current_node)
                    total_backtracks += 1
                    continue
            
            # Select action using UCB1
            action = self._select_action_ucb1(current_node, iteration, complexity.overall_score)
            
            # Execute action
            if action == ActionType.EXPAND:
                # Expand: create child nodes
                new_nodes = self._expand_node(problem, current_node)
                if new_nodes:
                    current_node = new_nodes[0]
                    total_expansions += len(new_nodes)
                    nodes_created += len(new_nodes)
                    max_depth = max(max_depth, current_node.depth)
                    
            elif action == ActionType.REFLECT:
                # Reflect: improve current node
                reflection = self._reflect_on_node(problem, current_node)
                current_node.score = reflection.get("score", current_node.score)
                
            elif action == ActionType.BACKTRACK:
                # Backtrack: return to parent
                if current_node.parent:
                    current_node = self._backtrack(current_node)
                    total_backtracks += 1
                    
            elif action == ActionType.CONCLUDE:
                # Conclude: generate final answer
                break
            
            # Visit node
            current_node.visit_count += 1
            visited_nodes.add(current_node.id)
        
        # Step 5: Find best path
        best_leaf = self._find_best_leaf(root)
        best_path = best_leaf.get_path()[1:] if best_leaf != root else []
        
        # Step 6: Generate final answer
        final_answer, confidence = self._generate_final_answer(problem, best_path)
        
        # Step 7: Cross-validation for overfitting
        cv_answers, cv_confidences, overfitting_detected = self._cross_validate(
            problem, best_path
        )
        
        # Use majority vote if overfitting
        if overfitting_detected:
            from collections import Counter
            answer_counts = Counter(cv_answers)
            final_answer = answer_counts.most_common(1)[0][0]
            logger.info(f"[{problem_id}] Using majority vote due to overfitting")
        
        # Check correctness
        correct = None
        if ground_truth:
            correct = self._check_answer(final_answer, ground_truth)

        latency = time.time() - start_time

        result = AdaptiveTreeResult(
            problem_id=problem_id,
            problem=problem,
            final_answer=final_answer,
            reasoning_path=[n.content for n in best_path],
            reflections=[],
            confidence=confidence,
            correct=correct,
            ground_truth=ground_truth,
            complexity_score=complexity.overall_score,
            recommended_depth=complexity.recommended_depth,
            actual_depth=len(best_path),
            total_expansions=total_expansions,
            total_backtracks=total_backtracks,
            max_tree_depth_reached=max_depth,
            nodes_created=nodes_created,
            overfitting_detected=overfitting_detected,
            cv_answers=cv_answers,
            cv_confidence_std=self._calc_std(cv_confidences),
            total_tokens=nodes_created * 100,  # Approximate
            latency_seconds=latency,
        )
        
        self._results.append(result)
        self._total_expansions += total_expansions
        self._total_backtracks += total_backtracks
        
        logger.info(
            f"[{problem_id}] Complete: answer={final_answer}, correct={correct}, "
            f"expansions={total_expansions}, backtracks={total_backtracks}"
        )
        
        return result
    
    def _select_action_ucb1(
        self,
        node: TreeNode,
        iteration: int,
        complexity: float
    ) -> ActionType:
        """Select action using UCB1 algorithm."""
        
        # Calculate UCB1 values for each possible action
        ucb_values = {}
        
        # EXPAND: Good when node has unexplored children
        if node.depth < self.config.max_tree_depth - 1:
            expand_score = 0.5 + 0.3 * (1 - node.depth / self.config.max_tree_depth)
            if complexity > 0.7:
                expand_score += 0.2  # Encourage expansion for complex problems
            
            # UCB1 exploration bonus
            exploration_bonus = self.config.exploration_constant * math.sqrt(
                math.log(iteration + 1) / (node.visit_count + 1)
            )
            ucb_values[ActionType.EXPAND] = expand_score + exploration_bonus
        
        # REFLECT: Good when score is moderate
        if 0.4 < node.score < 0.7:
            reflect_score = node.score
            exploration_bonus = self.config.exploration_constant * math.sqrt(
                math.log(iteration + 1) / (node.visit_count + 1)
            )
            ucb_values[ActionType.REFLECT] = reflect_score + exploration_bonus
        
        # BACKTRACK: Good when score is low
        if node.score < self.config.backtrack_threshold and node.parent:
            backtrack_score = 1.0 - node.score
            ucb_values[ActionType.BACKTRACK] = backtrack_score
        
        # CONCLUDE: Good when score is high and depth is sufficient
        if node.score > self.config.confidence_threshold_stop and node.depth > 2:
            ucb_values[ActionType.CONCLUDE] = node.score * 1.5  # Boost conclusion
        
        # Select action with highest UCB1
        if not ucb_values:
            return ActionType.EXPAND
        
        return max(ucb_values.items(), key=lambda x: x[1])[0]
    
    def _expand_node(self, problem: str, node: TreeNode) -> List[TreeNode]:
        """Expand node by generating child reasoning steps."""
        path = node.get_path()
        reasoning_so_far = [n.content for n in path]
        
        prompt = f"""Given the problem and reasoning so far, generate {self.config.max_tree_width} different next steps.

Problem: {problem}

Reasoning so far:
{self._format_reasoning(reasoning_so_far)}

Generate {self.config.max_tree_width} alternative next steps (one per line, numbered). Each should take the reasoning in a different direction."""

        response = self.generator.generate(
            messages=[{"role": "user", "content": prompt}],
            config=GenerationConfig(
                model=self.generator_model,
                temperature=self.config.temperature_reason,
                max_tokens=300,
            )
        )
        
        # Parse steps
        new_nodes = []
        for line in response.text.split("\n"):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith("Step")):
                content = line.split(".", 1)[-1].strip() if "." in line else line
                if content:
                    child = node.add_child(content, NodeType.STEP)
                    child.score = random.uniform(0.5, 0.8)  # Initial estimate
                    new_nodes.append(child)
        
        return new_nodes
    
    def _reflect_on_node(self, problem: str, node: TreeNode) -> Dict[str, Any]:
        """Reflect on node to improve its score."""
        path = node.get_path()
        reasoning = [n.content for n in path]
        
        prompt = f"""Evaluate this reasoning chain for the problem.

Problem: {problem}

Reasoning:
{self._format_reasoning(reasoning)}

Rate the quality (0.0 to 1.0) and provide brief feedback.

Format:
Score: [0.0-1.0]
Feedback: [brief evaluation]"""

        response = self.generator.generate(
            messages=[{"role": "user", "content": prompt}],
            config=GenerationConfig(
                model=self.generator_model,
                temperature=self.config.temperature_reflect,
                max_tokens=100,
            )
        )
        
        # Parse score
        import re
        score_match = re.search(r"(\d+\.\d+)", response.text)
        score = float(score_match.group(1)) if score_match else 0.5
        
        return {"score": min(1.0, max(0.0, score))}
    
    def _backtrack(self, node: TreeNode) -> TreeNode:
        """Backtrack to parent node."""
        if node.parent:
            logger.debug(f"Backtracking from {node.id} to {node.parent.id}")
            return node.parent
        return node
    
    def _find_best_leaf(self, root: TreeNode) -> TreeNode:
        """Find best leaf node in tree."""
        best = root
        best_score = root.score
        
        stack = [root]
        while stack:
            node = stack.pop()
            if node.score > best_score:
                best = node
                best_score = node.score
            stack.extend(node.children)
        
        return best
    
    def _generate_initial_reasoning(self, problem: str) -> List[str]:
        """Generate initial reasoning steps."""
        prompt = f"""Break down this problem into clear reasoning steps.

Problem: {problem}

Provide 2-4 logical steps. Format each step on a new line starting with "Step X:"."""

        response = self.generator.generate(
            messages=[{"role": "user", "content": prompt}],
            config=GenerationConfig(
                model=self.generator_model,
                temperature=self.config.temperature_reason,
                max_tokens=300,
            )
        )
        
        steps = []
        for line in response.text.split("\n"):
            if line.strip().startswith("Step"):
                content = line.split(":", 1)[-1].strip() if ":" in line else line.strip()
                if content:
                    steps.append(content)
        
        return steps if steps else [response.text.strip()]
    
    def _generate_final_answer(
        self,
        problem: str,
        path: List[TreeNode]
    ) -> Tuple[str, float]:
        """Generate final answer from best path."""
        reasoning = [n.content for n in path]
        
        prompt = f"""Based on the reasoning, answer the question.

Problem: {problem}

Reasoning:
{self._format_reasoning(reasoning)}

Provide:
1. Your final answer (yes/no)
2. Your confidence (0.0 to 1.0)

Format:
Answer: [yes/no]
Confidence: [0.0-1.0]"""

        response = self.generator.generate(
            messages=[{"role": "user", "content": prompt}],
            config=GenerationConfig(
                model=self.generator_model,
                temperature=self.config.temperature_conclude,
                max_tokens=100,
            )
        )
        
        text = response.text.lower()
        
        # Extract answer
        answer = "unknown"
        if "yes" in text:
            answer = "yes"
        elif "no" in text:
            answer = "no"
        
        # Extract confidence
        import re
        confidence_match = re.search(r"confidence[:\s]+([0-9.]+)", text)
        confidence = float(confidence_match.group(1)) if confidence_match else 0.5
        confidence = max(0.0, min(1.0, confidence))
        
        return answer, confidence
    
    def _cross_validate(
        self,
        problem: str,
        path: List[TreeNode]
    ) -> Tuple[List[str], List[float], bool]:
        """Cross-validate to detect overfitting."""
        if not self.config.enable_cross_validation:
            return [], [], False
        
        cv_answers = []
        cv_confidences = []
        
        reasoning = [n.content for n in path]
        
        for i in range(self.config.validation_samples):
            temp = self.config.temperature_conclude + (i * 0.1)
            
            prompt = f"""Answer based on reasoning.

Problem: {problem}

Reasoning:
{self._format_reasoning(reasoning)}

Answer (yes/no):"""

            response = self.generator.generate(
                messages=[{"role": "user", "content": prompt}],
                config=GenerationConfig(
                    model=self.generator_model,
                    temperature=temp,
                    max_tokens=50,
                )
            )
            
            text = response.text.lower()
            if "yes" in text:
                cv_answers.append("yes")
            elif "no" in text:
                cv_answers.append("no")
            else:
                cv_answers.append("unknown")
            
            cv_confidences.append(0.7 if cv_answers[-1] != "unknown" else 0.3)
        
        # Detect overfitting
        overfitting = False
        if len(cv_confidences) > 1:
            std = self._calc_std(cv_confidences)
            overfitting = std > self.config.variance_threshold
        
        return cv_answers, cv_confidences, overfitting
    
    def _create_result(self, **kwargs) -> AdaptiveTreeResult:
        """Create an AdaptiveTreeResult instance."""
        return AdaptiveTreeResult(**kwargs)

    def _format_reasoning(self, steps: List[str]) -> str:
        """Format reasoning steps."""
        return "\n".join(f"Step {i+1}: {s}" for i, s in enumerate(steps))
    
    def _calc_std(self, values: List[float]) -> float:
        """Calculate standard deviation."""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5
    
    def close(self):
        """Close pipeline."""
        super().close()
        self.generator.close()
    
    def save_results(self, filename: str = "adaptive_tree_results.json"):
        """Save results."""
        results_path = self.results_dir / filename
        
        data = {
            "timestamp": time.time(),
            "config": asdict(self.config),
            "total_expansions": self._total_expansions,
            "total_backtracks": self._total_backtracks,
            "num_problems": len(self._results),
            "results": [asdict(r) for r in self._results],
        }
        
        with open(results_path, "w", encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.info(f"Results saved to {results_path}")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        if not self._results:
            return {}
        
        total = len(self._results)
        correct = sum(1 for r in self._results if r.correct)
        
        return {
            "total_problems": total,
            "correct": correct,
            "accuracy": correct / total if total > 0 else 0,
            "avg_expansions": self._total_expansions / total if total > 0 else 0,
            "avg_backtracks": self._total_backtracks / total if total > 0 else 0,
            "avg_depth": sum(r.max_tree_depth_reached for r in self._results) / total,
        }
