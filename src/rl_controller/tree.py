"""State tree data structure for MCTS reasoning."""

from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import copy
import json


class NodeType(Enum):
    """Types of nodes in the reasoning tree."""
    ROOT = "root"
    STEP = "step"
    REFLECTION = "reflection"
    CONCLUSION = "conclusion"


@dataclass
class TreeNode:
    """Node in the reasoning tree."""
    
    content: str
    node_type: NodeType = NodeType.STEP
    parent: Optional['TreeNode'] = None
    children: List['TreeNode'] = field(default_factory=list)
    
    score: float = 0.0
    visit_count: int = 0
    cumulative_reward: float = 0.0
    
    action_taken: Optional[str] = None
    depth: int = 0
    is_terminal: bool = False
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    _id: int = field(default_factory=lambda: id(object()))
    
    def __post_init__(self):
        if self.parent is not None:
            self.depth = self.parent.depth + 1
    
    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0
    
    @property
    def is_root(self) -> bool:
        return self.parent is None
    
    @property
    def path_to_root(self) -> List['TreeNode']:
        """Get path from this node to root."""
        path = [self]
        node = self.parent
        while node is not None:
            path.append(node)
            node = node.parent
        return list(reversed(path))
    
    @property
    def path_content(self) -> List[str]:
        """Get content of all nodes from root to this node."""
        return [node.content for node in self.path_to_root]
    
    @property
    def step_count(self) -> int:
        """Count of reasoning steps (excluding reflections)."""
        return sum(1 for node in self.path_to_root if node.node_type == NodeType.STEP)
    
    def add_child(
        self,
        content: str,
        node_type: NodeType = NodeType.STEP,
        score: float = 0.0,
        action_taken: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> 'TreeNode':
        """Add a child node."""
        child = TreeNode(
            content=content,
            node_type=node_type,
            parent=self,
            score=score,
            action_taken=action_taken,
            metadata=metadata or {},
        )
        self.children.append(child)
        return child
    
    def get_best_child(self, criterion: str = "score") -> Optional['TreeNode']:
        """Get best child based on criterion."""
        if not self.children:
            return None
        
        if criterion == "score":
            return max(self.children, key=lambda n: n.score)
        elif criterion == "visit":
            return max(self.children, key=lambda n: n.visit_count)
        elif criterion == "ucb":
            return self.children[0]
        
        return max(self.children, key=lambda n: n.score)
    
    def get_siblings(self) -> List['TreeNode']:
        """Get all sibling nodes."""
        if self.parent is None:
            return []
        return [c for c in self.parent.children if c is not self]
    
    def get_best_sibling(self) -> Optional['TreeNode']:
        """Get best scoring sibling."""
        siblings = self.get_siblings()
        if not siblings:
            return None
        return max(siblings, key=lambda n: n.score)
    
    def update(self, reward: float, discount: float = 0.95):
        """Update node statistics with new reward."""
        self.visit_count += 1
        self.cumulative_reward += reward
        discounted = reward * (discount ** self.depth)
        self.score = self.cumulative_reward / self.visit_count
    
    def to_dict(self) -> Dict:
        """Convert node to dictionary for serialization."""
        return {
            "content": self.content,
            "node_type": self.node_type.value,
            "score": self.score,
            "visit_count": self.visit_count,
            "depth": self.depth,
            "action_taken": self.action_taken,
            "is_terminal": self.is_terminal,
            "num_children": len(self.children),
            "metadata": self.metadata,
        }
    
    def __repr__(self) -> str:
        return f"TreeNode(depth={self.depth}, score={self.score:.2f}, visits={self.visit_count}, type={self.node_type.value})"


class StateTree:
    """Tree structure for managing reasoning states in MCTS."""
    
    def __init__(self, problem: str, max_depth: int = 20):
        self.problem = problem
        self.max_depth = max_depth
        self.root = TreeNode(
            content=problem,
            node_type=NodeType.ROOT,
            depth=0,
        )
        self._node_count = 1
        self._total_depth = 0
    
    def add_step(
        self,
        parent: TreeNode,
        content: str,
        score: float = 0.0,
        action_taken: Optional[str] = None,
        node_type: NodeType = NodeType.STEP,
    ) -> TreeNode:
        """Add a reasoning step as child of parent node."""
        if parent.depth >= self.max_depth:
            parent.is_terminal = True
            raise ValueError(f"Maximum depth {self.max_depth} reached")
        
        child = parent.add_child(
            content=content,
            node_type=node_type,
            score=score,
            action_taken=action_taken,
        )
        self._node_count += 1
        self._total_depth += child.depth
        return child
    
    def get_node_by_path(self, path: List[str]) -> Optional[TreeNode]:
        """Find node by its path content."""
        if not path:
            return self.root
        
        current = self.root
        for content in path[1:]:
            found = False
            for child in current.children:
                if child.content == content:
                    current = child
                    found = True
                    break
            if not found:
                return None
        return current
    
    def get_best_path(self) -> List[TreeNode]:
        """Get path to best terminal node."""
        best_leaf = self._find_best_leaf(self.root)
        if best_leaf is None:
            return [self.root]
        return best_leaf.path_to_root
    
    def _find_best_leaf(self, node: TreeNode) -> Optional[TreeNode]:
        """Recursively find best scoring leaf."""
        if node.is_leaf:
            return node if node.score > 0 else None
        
        best_leaf = None
        best_score = float('-inf')
        
        for child in node.children:
            leaf = self._find_best_leaf(child)
            if leaf and leaf.score > best_score:
                best_leaf = leaf
                best_score = leaf.score
        
        return best_leaf
    
    def get_all_paths(self) -> List[List[TreeNode]]:
        """Get all paths from root to leaves."""
        paths = []
        self._collect_paths(self.root, [], paths)
        return paths
    
    def _collect_paths(
        self,
        node: TreeNode,
        current_path: List[TreeNode],
        paths: List[List[TreeNode]],
    ):
        """Recursively collect all paths to leaves."""
        current_path = current_path + [node]
        
        if node.is_leaf:
            paths.append(current_path)
            return
        
        for child in node.children:
            self._collect_paths(child, current_path, paths)
    
    def prune_below(self, node: TreeNode, keep_best: int = 1):
        """Prune all but the best k children of a node."""
        if len(node.children) <= keep_best:
            return
        
        node.children.sort(key=lambda n: n.score, reverse=True)
        removed = node.children[keep_best:]
        node.children = node.children[:keep_best]
        
        self._node_count -= sum(self._count_subtree(n) for n in removed)
    
    def _count_subtree(self, node: TreeNode) -> int:
        """Count all nodes in subtree."""
        count = 1
        for child in node.children:
            count += self._count_subtree(child)
        return count
    
    def get_stats(self) -> Dict:
        """Get tree statistics."""
        paths = self.get_all_paths()
        depths = [len(p) - 1 for p in paths]
        
        return {
            "total_nodes": self._node_count,
            "num_paths": len(paths),
            "max_depth": max(depths) if depths else 0,
            "avg_depth": sum(depths) / len(depths) if depths else 0,
            "avg_score": sum(n.score for n in self._all_nodes()) / max(1, self._node_count),
        }
    
    def _all_nodes(self) -> List[TreeNode]:
        """Get all nodes in tree."""
        nodes = []
        self._collect_nodes(self.root, nodes)
        return nodes
    
    def _collect_nodes(self, node: TreeNode, nodes: List[TreeNode]):
        """Recursively collect all nodes."""
        nodes.append(node)
        for child in node.children:
            self._collect_nodes(child, nodes)
    
    def to_json(self) -> str:
        """Serialize tree to JSON."""
        def node_to_dict(n):
            return {
                **n.to_dict(),
                "children": [node_to_dict(c) for c in n.children]
            }
        
        return json.dumps({
            "problem": self.problem,
            "root": node_to_dict(self.root),
            "stats": self.get_stats(),
        }, indent=2)
    
    def copy(self) -> 'StateTree':
        """Create a deep copy of the tree."""
        new_tree = StateTree(self.problem, self.max_depth)
        new_tree.root = copy.deepcopy(self.root)
        new_tree._node_count = self._node_count
        return new_tree
    
    def __repr__(self) -> str:
        return f"StateTree(nodes={self._node_count}, max_depth={self.max_depth})"
