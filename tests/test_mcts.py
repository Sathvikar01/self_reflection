"""Tests for MCTS controller."""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rl_controller.mcts import MCTSController, MCTSConfig, MCTSStats
from src.rl_controller.actions import ActionExecutor, ActionType, ActionResult
from src.rl_controller.tree import TreeNode, NodeType


class TestMCTSConfig:
    """Tests for MCTS configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = MCTSConfig()
        
        assert config.exploration_constant == 1.414
        assert config.max_tree_depth == 20
        assert config.expansion_budget == 50
        assert config.temperature == 0.7
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = MCTSConfig(
            exploration_constant=2.0,
            expansion_budget=100,
            temperature=0.5,
        )
        
        assert config.exploration_constant == 2.0
        assert config.expansion_budget == 100
        assert config.temperature == 0.5


class TestMCTSStats:
    """Tests for MCTS statistics."""
    
    def test_initial_stats(self):
        """Test initial statistics."""
        stats = MCTSStats()
        
        assert stats.total_expansions == 0
        assert stats.total_backtracks == 0
        assert stats.best_score == float('-inf')
    
    def test_stat_updates(self):
        """Test statistics are updated."""
        stats = MCTSStats()
        
        stats.total_expansions = 10
        stats.total_backtracks = 3
        stats.best_score = 0.8
        
        assert stats.total_expansions == 10
        assert stats.total_backtracks == 3
        assert stats.best_score == 0.8


class TestMCTSController:
    """Tests for MCTS controller."""
    
    @pytest.fixture
    def mock_executor(self):
        """Create mock action executor."""
        executor = Mock(spec=ActionExecutor)
        
        expand_result = ActionResult(
            success=True,
            action_type=ActionType.EXPAND,
            content="Generated step",
            score=0.5,
            new_node=TreeNode(content="step", score=0.5),
        )
        
        conclude_result = ActionResult(
            success=True,
            action_type=ActionType.CONCLUDE,
            content="Final answer",
            score=0.8,
            new_node=TreeNode(content="answer", node_type=NodeType.CONCLUSION, score=0.8),
        )
        
        executor.execute = Mock(side_effect=[expand_result, conclude_result])
        executor.get_action_weights = Mock(return_value={
            ActionType.EXPAND: 0.4,
            ActionType.REFLECT: 0.25,
            ActionType.BACKTRACK: 0.2,
            ActionType.CONCLUDE: 0.15,
        })
        
        return executor
    
    def test_controller_creation(self, mock_executor):
        """Test controller initialization."""
        config = MCTSConfig(expansion_budget=10)
        controller = MCTSController(mock_executor, config)
        
        assert controller.config == config
        assert controller.action_executor == mock_executor
    
    def test_get_stats(self, mock_executor):
        """Test statistics retrieval."""
        controller = MCTSController(mock_executor)
        stats = controller.get_stats()
        
        assert isinstance(stats, MCTSStats)
        assert stats.total_expansions == 0
    
    def test_action_selection(self, mock_executor):
        """Test action selection logic."""
        controller = MCTSController(mock_executor)
        
        root = TreeNode(content="problem", node_type=NodeType.ROOT)
        
        action = controller._select_action(root, 0)
        
        assert action in [ActionType.EXPAND, ActionType.REFLECT, 
                         ActionType.BACKTRACK, ActionType.CONCLUDE]
    
    def test_should_stop(self, mock_executor):
        """Test early stopping logic."""
        controller = MCTSController(mock_executor)

        root = TreeNode(content="problem")

        assert not controller._should_stop(root, 0.9)

        high_score_node = TreeNode(content="test", score=0.95)
        high_score_node.depth = 3
        assert controller._should_stop(high_score_node, 0.9)


class TestUCBCalculation:
    """Tests for UCB1 calculation."""
    
    def test_ucb_unvisited_node(self):
        """Test UCB for unvisited node returns infinity."""
        config = MCTSConfig()
        controller = MCTSController(Mock(), config)
        
        node = TreeNode(content="test")
        ucb = controller._calculate_ucb1(node, 10)
        
        assert ucb == float('inf')
    
    def test_ucb_visited_node(self):
        """Test UCB for visited node."""
        config = MCTSConfig(exploration_constant=1.0)
        controller = MCTSController(Mock(), config)
        
        node = TreeNode(content="test", score=0.5)
        node.visit_count = 5
        
        ucb = controller._calculate_ucb1(node, 20)
        
        assert isinstance(ucb, float)
        assert ucb > 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
