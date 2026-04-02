"""Tests for tree data structure."""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rl_controller.tree import StateTree, TreeNode, NodeType


class TestTreeNode:
    """Tests for TreeNode class."""
    
    def test_node_creation(self):
        """Test basic node creation."""
        node = TreeNode(content="test content")
        
        assert node.content == "test content"
        assert node.node_type == NodeType.STEP
        assert node.parent is None
        assert len(node.children) == 0
        assert node.score == 0.0
        assert node.visit_count == 0
    
    def test_node_with_parent(self):
        """Test node creation with parent."""
        parent = TreeNode(content="parent")
        child = TreeNode(content="child", parent=parent)
        
        assert child.parent == parent
        assert child.depth == 1
        assert parent.is_root
        assert not child.is_root
    
    def test_add_child(self):
        """Test adding children."""
        parent = TreeNode(content="parent")
        child = parent.add_child("child", score=0.5)
        
        assert len(parent.children) == 1
        assert child.parent == parent
        assert child.content == "child"
        assert child.score == 0.5
    
    def test_path_to_root(self):
        """Test path traversal to root."""
        root = TreeNode(content="root")
        child1 = root.add_child("child1")
        child2 = child1.add_child("child2")
        
        path = child2.path_to_root
        
        assert len(path) == 3
        assert path[0] == root
        assert path[1] == child1
        assert path[2] == child2
    
    def test_is_leaf(self):
        """Test leaf detection."""
        node = TreeNode(content="test")
        assert node.is_leaf
        
        node.add_child("child")
        assert not node.is_leaf
    
    def test_get_siblings(self):
        """Test sibling retrieval."""
        parent = TreeNode(content="parent")
        child1 = parent.add_child("child1")
        child2 = parent.add_child("child2")
        child3 = parent.add_child("child3")
        
        siblings = child1.get_siblings()
        
        assert len(siblings) == 2
        assert child2 in siblings
        assert child3 in siblings
        assert child1 not in siblings


class TestStateTree:
    """Tests for StateTree class."""
    
    def test_tree_creation(self):
        """Test tree initialization."""
        tree = StateTree("test problem")
        
        assert tree.problem == "test problem"
        assert tree.root.content == "test problem"
        assert tree.root.node_type == NodeType.ROOT
    
    def test_add_step(self):
        """Test adding reasoning steps."""
        tree = StateTree("problem")
        step1 = tree.add_step(tree.root, "step 1", score=0.5)
        
        assert step1.content == "step 1"
        assert step1.parent == tree.root
        assert len(tree.root.children) == 1
    
    def test_max_depth(self):
        """Test maximum depth constraint."""
        tree = StateTree("problem", max_depth=3)

        current = tree.root
        for i in range(3):
            current = tree.add_step(current, f"step {i}")

        with pytest.raises(ValueError):
            tree.add_step(current, "step 4")
    
    def test_get_best_path(self):
        """Test best path retrieval."""
        tree = StateTree("problem")
        
        step1 = tree.add_step(tree.root, "step 1", score=0.3)
        step2 = tree.add_step(tree.root, "step 2", score=0.8)
        
        best_path = tree.get_best_path()
        
        assert len(best_path) == 2
        assert best_path[1].score == 0.8
    
    def test_get_stats(self):
        """Test statistics computation."""
        tree = StateTree("problem")
        
        tree.add_step(tree.root, "step 1", score=0.5)
        tree.add_step(tree.root, "step 2", score=0.7)
        
        stats = tree.get_stats()
        
        assert stats["total_nodes"] == 3
        assert stats["num_paths"] == 2
    
    def test_to_json(self):
        """Test JSON serialization."""
        tree = StateTree("problem")
        tree.add_step(tree.root, "step 1", score=0.5)
        
        json_str = tree.to_json()
        
        assert "problem" in json_str
        assert "step 1" in json_str
        assert "stats" in json_str
    
    def test_copy(self):
        """Test tree copying."""
        tree = StateTree("problem")
        tree.add_step(tree.root, "step 1", score=0.5)
        
        tree_copy = tree.copy()
        
        assert tree_copy.problem == tree.problem
        assert tree_copy.root.children[0].score == 0.5
        
        tree_copy.root.children[0].score = 0.9
        assert tree.root.children[0].score == 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
