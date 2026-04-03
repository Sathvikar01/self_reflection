"""Replay buffer for storing and sampling training data."""

import random
from typing import List, Dict, Any, Optional, Tuple
from collections import deque
import numpy as np
from loguru import logger


class ReplayBuffer:
    """Replay buffer for storing reasoning trajectories."""

    def __init__(self, capacity: int = 10000):
        """Initialize replay buffer.

        Args:
            capacity: Maximum number of experiences to store
        """
        self.capacity = capacity
        self.buffer: deque = deque(maxlen=capacity)

        # Statistics
        self._total_added = 0
        self._total_sampled = 0

        logger.info(f"ReplayBuffer initialized (capacity={capacity})")

    def add(
        self,
        problem: str,
        previous_steps: List[str],
        current_step: str,
        prm_score: float,
        outcome_reward: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Add a single experience to the buffer.

        Args:
            problem: Problem statement
            previous_steps: Previous reasoning steps
            current_step: Current reasoning step
            prm_score: PRM score for this step
            outcome_reward: Final outcome reward (if known)
            metadata: Additional metadata
        """
        experience = {
            "problem": problem,
            "previous_steps": previous_steps.copy(),
            "current_step": current_step,
            "prm_score": prm_score,
            "outcome_reward": outcome_reward,
            "metadata": metadata or {},
            "id": self._total_added
        }

        self.buffer.append(experience)
        self._total_added += 1

    def add_trajectory(
        self,
        problem: str,
        reasoning_chain: List[str],
        prm_scores: List[float],
        final_outcome: Optional[float] = None
    ):
        """Add a complete reasoning trajectory.

        Args:
            problem: Problem statement
            reasoning_chain: List of reasoning steps
            prm_scores: PRM scores for each step
            final_outcome: Final outcome (1.0 for correct, 0.0 for incorrect, None for unknown)
        """
        for i, (step, score) in enumerate(zip(reasoning_chain, prm_scores)):
            previous_steps = reasoning_chain[:i]

            self.add(
                problem=problem,
                previous_steps=previous_steps,
                current_step=step,
                prm_score=score,
                outcome_reward=final_outcome
            )

    def sample(
        self,
        batch_size: int,
        prioritize_positive: bool = False
    ) -> List[Dict[str, Any]]:
        """Sample a batch of experiences.

        Args:
            batch_size: Number of experiences to sample
            prioritize_positive: Whether to prioritize positive outcomes

        Returns:
            List of sampled experiences
        """
        if len(self.buffer) == 0:
            logger.warning("Attempted to sample from empty buffer")
            return []

        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)

        if prioritize_positive:
            # Separate positive and negative outcomes
            positive = [exp for exp in self.buffer if exp.get("outcome_reward", 0.5) > 0.5]
            negative = [exp for exp in self.buffer if exp.get("outcome_reward", 0.5) <= 0.5]

            # Sample 70% positive, 30% negative if available
            n_positive = min(int(batch_size * 0.7), len(positive))
            n_negative = batch_size - n_positive

            if len(positive) < n_positive:
                n_positive = len(positive)
                n_negative = batch_size - n_positive

            sampled = (
                random.sample(positive, n_positive) if n_positive > 0 else [] +
                random.sample(negative, n_negative) if n_negative > 0 else []
            )
        else:
            sampled = random.sample(list(self.buffer), batch_size)

        self._total_sampled += len(sampled)

        return sampled

    def get_all(self) -> List[Dict[str, Any]]:
        """Get all experiences in buffer.

        Returns:
            List of all experiences
        """
        return list(self.buffer)

    def get_by_outcome(self, outcome: float) -> List[Dict[str, Any]]:
        """Get experiences by outcome.

        Args:
            outcome: Outcome value (1.0 or 0.0)

        Returns:
            List of experiences with matching outcome
        """
        return [
            exp for exp in self.buffer
            if exp.get("outcome_reward") == outcome
        ]

    def clear(self):
        """Clear the buffer."""
        self.buffer.clear()
        logger.info("ReplayBuffer cleared")

    def __len__(self) -> int:
        """Get buffer size."""
        return len(self.buffer)

    def get_stats(self) -> Dict[str, Any]:
        """Get buffer statistics.

        Returns:
            Dictionary with statistics
        """
        positive = sum(1 for exp in self.buffer if exp.get("outcome_reward", 0.5) > 0.5)
        negative = sum(1 for exp in self.buffer if exp.get("outcome_reward", 0.5) <= 0.5)

        return {
            "size": len(self.buffer),
            "capacity": self.capacity,
            "total_added": self._total_added,
            "total_sampled": self._total_sampled,
            "positive_outcomes": positive,
            "negative_outcomes": negative,
            "avg_prm_score": np.mean([exp["prm_score"] for exp in self.buffer]) if self.buffer else 0.0
        }


class PrioritizedReplayBuffer(ReplayBuffer):
    """Replay buffer with prioritized sampling based on TD error."""

    def __init__(self, capacity: int = 10000, alpha: float = 0.6, beta: float = 0.4):
        """Initialize prioritized replay buffer.

        Args:
            capacity: Maximum buffer size
            alpha: Priority exponent (0 = uniform, 1 = full prioritization)
            beta: Importance sampling exponent (annealed to 1)
        """
        super().__init__(capacity)
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = 0.001

        # Priority storage
        self.priorities: np.ndarray = np.zeros(capacity, dtype=np.float32)
        self.max_priority = 1.0

    def add(
        self,
        problem: str,
        previous_steps: List[str],
        current_step: str,
        prm_score: float,
        outcome_reward: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
        priority: Optional[float] = None
    ):
        """Add experience with priority.

        Args:
            problem: Problem statement
            previous_steps: Previous steps
            current_step: Current step
            prm_score: PRM score
            outcome_reward: Outcome reward
            metadata: Metadata
            priority: Priority value (uses max_priority if None)
        """
        super().add(problem, previous_steps, current_step, prm_score, outcome_reward, metadata)

        # Set priority
        if priority is None:
            priority = self.max_priority

        # Store priority with exponent
        idx = (self._total_added - 1) % self.capacity
        self.priorities[idx] = priority ** self.alpha

    def sample(self, batch_size: int) -> Tuple[List[Dict], np.ndarray, np.ndarray]:
        """Sample batch with prioritization.

        Args:
            batch_size: Batch size

        Returns:
            Tuple of (experiences, indices, importance weights)
        """
        if len(self.buffer) == 0:
            return [], np.array([]), np.array([])

        # Calculate sampling probabilities
        priorities = self.priorities[:len(self.buffer)]
        probabilities = priorities / priorities.sum()

        # Sample indices
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)

        # Get experiences
        experiences = [self.buffer[i] for i in indices]

        # Calculate importance sampling weights
        weights = (len(self.buffer) * probabilities[indices]) ** (-self.beta)
        weights = weights / weights.max()  # Normalize

        # Anneal beta
        self.beta = min(1.0, self.beta + self.beta_increment)

        return experiences, indices, weights

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """Update priorities for sampled experiences.

        Args:
            indices: Indices of sampled experiences
            priorities: New priority values
        """
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority ** self.alpha
            self.max_priority = max(self.max_priority, priority)


class TrainingDataPipeline:
    """Pipeline for preparing training data from trajectories."""

    def __init__(
        self,
        buffer: Optional[ReplayBuffer] = None,
        embedder = None
    ):
        """Initialize training data pipeline.

        Args:
            buffer: Replay buffer instance
            embedder: State embedder instance
        """
        self.buffer = buffer or ReplayBuffer()
        self.embedder = embedder

        logger.info("TrainingDataPipeline initialized")

    def process_results_file(self, results_file: str):
        """Process results from JSON file.

        Args:
            results_file: Path to results JSON file
        """
        import json
        from pathlib import Path

        results_path = Path(results_file)
        if not results_path.exists():
            logger.error(f"Results file not found: {results_file}")
            return

        with open(results_path, 'r') as f:
            results = json.load(f)

        # Handle different result formats
        if isinstance(results, list):
            for result in results:
                self._process_single_result(result)
        elif isinstance(results, dict):
            if "results" in results:
                for result in results["results"]:
                    self._process_single_result(result)
            else:
                self._process_single_result(results)

        logger.info(f"Processed results from {results_file}, buffer size: {len(self.buffer)}")

    def _process_single_result(self, result: Dict[str, Any]):
        """Process a single result.

        Args:
            result: Result dictionary
        """
        problem = result.get("problem", "")
        reasoning_chain = result.get("reasoning_chain", [])
        correct = result.get("correct")

        # Get PRM scores if available
        prm_scores = result.get("prm_scores", [])

        # If no PRM scores, use placeholder
        if not prm_scores:
            prm_scores = [0.5] * len(reasoning_chain)

        # Determine outcome
        outcome_reward = None
        if correct is not None:
            outcome_reward = 1.0 if correct else 0.0

        # Add to buffer
        self.buffer.add_trajectory(
            problem=problem,
            reasoning_chain=reasoning_chain,
            prm_scores=prm_scores,
            final_outcome=outcome_reward
        )

    def create_training_dataset(
        self,
        batch_size: int = 32,
        include_embeddings: bool = False
    ) -> Dict[str, np.ndarray]:
        """Create training dataset from buffer.

        Args:
            batch_size: Batch size for dataset
            include_embeddings: Whether to compute embeddings

        Returns:
            Dictionary with training data
        """
        experiences = self.buffer.get_all()

        if len(experiences) == 0:
            logger.warning("No experiences in buffer")
            return {}

        # Extract data
        problems = [exp["problem"] for exp in experiences]
        previous_steps_list = [exp["previous_steps"] for exp in experiences]
        current_steps = [exp["current_step"] for exp in experiences]
        prm_scores = np.array([exp["prm_score"] for exp in experiences])

        # Outcome rewards (use PRM score if outcome unknown)
        outcome_rewards = np.array([
            exp.get("outcome_reward", exp["prm_score"])
            for exp in experiences
        ])

        # Combine rewards (weighted average)
        rewards = 0.7 * outcome_rewards + 0.3 * prm_scores

        dataset = {
            "problems": problems,
            "previous_steps": previous_steps_list,
            "current_steps": current_steps,
            "prm_scores": prm_scores,
            "outcome_rewards": outcome_rewards,
            "rewards": rewards
        }

        if include_embeddings and self.embedder:
            logger.info("Computing state embeddings...")
            embeddings = self.embedder.embed_batch(experiences)
            dataset["embeddings"] = embeddings

        return dataset

    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics.

        Returns:
            Statistics dictionary
        """
        return self.buffer.get_stats()
