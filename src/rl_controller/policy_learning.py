"""Policy learning for adaptive action selection."""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from collections import deque
import random
from loguru import logger
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class PolicyNetwork(nn.Module):
    """Neural network for action policy."""

    def __init__(
        self,
        state_dim: int = 392,
        hidden_dim: int = 128,
        n_actions: int = 4
    ):
        """Initialize policy network.

        Args:
            state_dim: State embedding dimension
            hidden_dim: Hidden layer dimension
            n_actions: Number of actions (expand, reflect, backtrack, conclude)
        """
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, n_actions),
            nn.Softmax(dim=-1)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            state: State tensor (batch_size, state_dim)

        Returns:
            Action probabilities (batch_size, n_actions)
        """
        return self.network(state)


class PolicyLearner:
    """Learn action policies from trajectories."""

    def __init__(
        self,
        state_dim: int = 392,
        hidden_dim: int = 128,
        learning_rate: float = 0.001,
        gamma: float = 0.95
    ):
        """Initialize policy learner.

        Args:
            state_dim: State dimension
            hidden_dim: Hidden dimension
            learning_rate: Learning rate
            gamma: Discount factor
        """
        self.gamma = gamma

        # Policy network
        self.policy = PolicyNetwork(state_dim, hidden_dim)

        # Optimizer
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)

        # Trajectory storage
        self.trajectories: List[Dict] = []

        logger.info("PolicyLearner initialized")

    def select_action(
        self,
        state_embedding: np.ndarray,
        epsilon: float = 0.1
    ) -> int:
        """Select action using policy.

        Args:
            state_embedding: State embedding
            epsilon: Exploration rate

        Returns:
            Action index (0=expand, 1=reflect, 2=backtrack, 3=conclude)
        """
        # Epsilon-greedy exploration
        if random.random() < epsilon:
            return random.randint(0, 3)

        # Use policy
        state_tensor = torch.tensor(state_embedding, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            action_probs = self.policy(state_tensor)

        # Sample from distribution
        action = torch.multinomial(action_probs, 1).item()

        return action

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """Store transition for training.

        Args:
            state: State embedding
            action: Action taken
            reward: Reward received
            next_state: Next state embedding
            done: Episode done flag
        """
        self.trajectories.append({
            "state": state,
            "action": action,
            "reward": reward,
            "next_state": next_state,
            "done": done
        })

    def compute_returns(self) -> List[float]:
        """Compute discounted returns.

        Returns:
            List of returns for each timestep
        """
        returns = []
        R = 0

        for transition in reversed(self.trajectories):
            if transition["done"]:
                R = 0
            R = transition["reward"] + self.gamma * R
            returns.insert(0, R)

        return returns

    def update(self) -> float:
        """Update policy using REINFORCE.

        Returns:
            Loss value
        """
        if len(self.trajectories) == 0:
            return 0.0

        # Compute returns
        returns = self.compute_returns()
        returns = torch.tensor(returns, dtype=torch.float32)

        # Normalize returns
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Compute policy loss
        states = torch.tensor(
            [t["state"] for t in self.trajectories],
            dtype=torch.float32
        )
        actions = torch.tensor([t["action"] for t in self.trajectories])

        # Get log probabilities
        action_probs = self.policy(states)
        log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(1)))

        # Policy gradient loss
        loss = -(log_probs.squeeze() * returns).mean()

        # Update
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Clear trajectories
        self.trajectories = []

        return loss.item()

    def save(self, path: str):
        """Save policy network.

        Args:
            path: Save path
        """
        torch.save({
            "policy_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict()
        }, path)
        logger.info(f"Policy saved to {path}")

    def load(self, path: str):
        """Load policy network.

        Args:
            path: Load path
        """
        checkpoint = torch.load(path, weights_only=True)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        logger.info(f"Policy loaded from {path}")


class AdaptiveActionSelector:
    """Adaptive action selection using learned policy."""

    def __init__(
        self,
        policy_learner: Optional[PolicyLearner] = None,
        initial_weights: Dict[str, float] = None
    ):
        """Initialize adaptive action selector.

        Args:
            policy_learner: Trained policy learner
            initial_weights: Initial action weights
        """
        self.policy = policy_learner

        # Default weights
        self.weights = initial_weights or {
            "expand": 0.4,
            "reflect": 0.25,
            "backtrack": 0.2,
            "conclude": 0.15
        }

        # Action mapping
        self.action_names = ["expand", "reflect", "backtrack", "conclude"]

        logger.info("AdaptiveActionSelector initialized")

    def select_action(
        self,
        state_embedding: Optional[np.ndarray] = None,
        node_score: float = 0.5,
        node_depth: int = 0,
        exploration: bool = True
    ) -> str:
        """Select action adaptively.

        Args:
            state_embedding: State embedding (optional)
            node_score: Current node score
            node_depth: Current node depth
            exploration: Whether to explore

        Returns:
            Action name
        """
        # If policy available and state provided
        if self.policy and state_embedding is not None:
            epsilon = 0.1 if exploration else 0.0
            action_idx = self.policy.select_action(state_embedding, epsilon)
            return self.action_names[action_idx]

        # Otherwise use heuristic
        if node_score < 0.3:
            # Low score - consider backtrack
            if node_depth > 1:
                return "backtrack"
            else:
                return "reflect"

        elif node_score > 0.85:
            # High score - consider conclude
            if node_depth >= 3:
                return "conclude"

        # Default weighted selection
        return self._weighted_selection()

    def _weighted_selection(self) -> str:
        """Weighted random action selection.

        Returns:
            Action name
        """
        total = sum(self.weights.values())
        r = random.random() * total

        cumulative = 0
        for action, weight in self.weights.items():
            cumulative += weight
            if r <= cumulative:
                return action

        return "expand"

    def update_weights(self, performance: Dict[str, float]):
        """Update action weights based on performance.

        Args:
            performance: Performance metrics per action
        """
        # Simple performance-based update
        for action in self.action_names:
            if action in performance:
                # Increase weight for good performance
                self.weights[action] *= (1 + 0.1 * performance[action])

        # Normalize weights
        total = sum(self.weights.values())
        for action in self.weights:
            self.weights[action] /= total

        logger.info(f"Updated weights: {self.weights}")
