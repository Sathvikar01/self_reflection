"""Value network evaluator that replaces PRM for cost reduction."""

import torch
from typing import List, Dict, Any, Optional
import numpy as np
from loguru import logger

from ..rl_controller.value_network import ValueNetwork, ValueNetworkConfig, ValueEstimator
from ..rl_controller.state_embedder import StateEmbedder, MockEmbedder


class ValueNetworkEvaluator:
    """Evaluator using trained value network instead of PRM."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = None,
        use_mock_embedder: bool = False
    ):
        """Initialize value network evaluator.

        Args:
            model_path: Path to trained model (uses default if None)
            device: Device for inference
            use_mock_embedder: Whether to use mock embedder for testing
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize embedder
        if use_mock_embedder:
            self.embedder = MockEmbedder(embedding_dim=768)
            embedding_dim = 768 + 8  # Mock dim + metadata
        else:
            self.embedder = StateEmbedder(device=self.device)
            embedding_dim = self.embedder.get_embedding_dim()

        # Initialize model
        config = ValueNetworkConfig(input_dim=embedding_dim)
        self.model = ValueNetwork(config).to(self.device)

        # Load trained weights if available
        if model_path and Path(model_path).exists():
            self.load_model(model_path)
        else:
            logger.warning("No trained model provided, using random weights")

        self.model.eval()

        # Statistics
        self._total_evaluations = 0
        self._total_inference_time = 0.0

        logger.info(f"ValueNetworkEvaluator initialized (device={self.device})")

    def load_model(self, model_path: str):
        """Load trained model weights.

        Args:
            model_path: Path to model checkpoint
        """
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"Loaded model from {model_path}")

    def evaluate_step(
        self,
        problem: str,
        previous_steps: List[str],
        current_step: str
    ) -> float:
        """Evaluate a reasoning step.

        Args:
            problem: Problem statement
            previous_steps: Previous reasoning steps
            current_step: Current step to evaluate

        Returns:
            Value estimate in [-1, 1]
        """
        import time
        start_time = time.time()

        # Embed state
        state_embedding = self.embedder.embed_state(
            problem=problem,
            previous_steps=previous_steps,
            current_step=current_step,
            score=None
        )

        # Convert to tensor
        state_tensor = torch.tensor(state_embedding, dtype=torch.float32, device=self.device).unsqueeze(0)

        # Predict
        with torch.no_grad():
            value = self.model(state_tensor).item()

        self._total_evaluations += 1
        self._total_inference_time += time.time() - start_time

        return value

    def evaluate_batch(
        self,
        states: List[Dict[str, Any]],
        batch_size: int = 32
    ) -> List[float]:
        """Evaluate multiple states in batch.

        Args:
            states: List of state dictionaries
            batch_size: Batch size for processing

        Returns:
            List of value estimates
        """
        import time
        start_time = time.time()

        # Compute embeddings
        embeddings = []
        for state in states:
            emb = self.embedder.embed_state(
                problem=state.get("problem", ""),
                previous_steps=state.get("previous_steps", []),
                current_step=state.get("current_step", ""),
                score=state.get("score")
            )
            embeddings.append(emb)

        # Convert to tensor
        X = torch.tensor(np.array(embeddings), dtype=torch.float32, device=self.device)

        # Batch inference
        values = []
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                batch = X[i:i+batch_size]
                batch_values = self.model(batch).squeeze(-1).tolist()
                values.extend(batch_values if isinstance(batch_values, list) else [batch_values])

        self._total_evaluations += len(states)
        self._total_inference_time += time.time() - start_time

        return values

    def get_stats(self) -> Dict[str, Any]:
        """Get evaluator statistics.

        Returns:
            Dictionary with stats
        """
        avg_inference_time = (
            self._total_inference_time / self._total_evaluations
            if self._total_evaluations > 0 else 0.0
        )

        return {
            "total_evaluations": self._total_evaluations,
            "total_inference_time": self._total_inference_time,
            "avg_inference_time": avg_inference_time,
            "embedder_cache_stats": self.embedder.get_cache_stats()
        }

    def reset_stats(self):
        """Reset statistics."""
        self._total_evaluations = 0
        self._total_inference_time = 0.0
        self.embedder.clear_cache()


class HybridEvaluator:
    """Hybrid evaluator that can use both PRM and value network."""

    def __init__(
        self,
        prm_evaluator,
        value_network_path: Optional[str] = None,
        value_network_weight: float = 0.7,
        device: str = None
    ):
        """Initialize hybrid evaluator.

        Args:
            prm_evaluator: PRM evaluator instance
            value_network_path: Path to trained value network
            value_network_weight: Weight for value network (0-1)
            device: Device for inference
        """
        self.prm_evaluator = prm_evaluator
        self.value_network_weight = value_network_weight

        # Initialize value network evaluator
        self.vn_evaluator = ValueNetworkEvaluator(
            model_path=value_network_path,
            device=device
        )

        # Statistics
        self._prm_calls = 0
        self._vn_calls = 0

        logger.info(f"HybridEvaluator initialized (vn_weight={value_network_weight})")

    def evaluate_step(
        self,
        problem: str,
        previous_steps: List[str],
        current_step: str,
        use_prm: bool = False
    ) -> float:
        """Evaluate step using hybrid approach.

        Args:
            problem: Problem statement
            previous_steps: Previous steps
            current_step: Current step
            use_prm: Force using PRM

        Returns:
            Combined value estimate
        """
        if use_prm or self.value_network_weight == 0:
            # Use PRM only
            prm_score = self.prm_evaluator.evaluate_step(problem, previous_steps, current_step)
            self._prm_calls += 1
            return prm_score

        elif self.value_network_weight == 1:
            # Use value network only
            vn_score = self.vn_evaluator.evaluate_step(problem, previous_steps, current_step)
            self._vn_calls += 1
            return vn_score

        else:
            # Use hybrid combination
            prm_score = self.prm_evaluator.evaluate_step(problem, previous_steps, current_step)
            self._prm_calls += 1

            vn_score = self.vn_evaluator.evaluate_step(problem, previous_steps, current_step)
            self._vn_calls += 1

            # Weighted combination
            combined_score = (
                self.value_network_weight * vn_score +
                (1 - self.value_network_weight) * prm_score
            )

            return combined_score

    def get_stats(self) -> Dict[str, Any]:
        """Get evaluator statistics.

        Returns:
            Statistics dictionary
        """
        return {
            "prm_calls": self._prm_calls,
            "vn_calls": self._vn_calls,
            "prm_percentage": self._prm_calls / (self._prm_calls + self._vn_calls) * 100
            if (self._prm_calls + self._vn_calls) > 0 else 0,
            "vn_stats": self.vn_evaluator.get_stats()
        }


if __name__ == "__main__":
    # Example usage
    evaluator = ValueNetworkEvaluator()

    problem = "Do hamsters provide food for any animals?"
    previous_steps = ["Hamsters are small rodents."]
    current_step = "Small rodents are prey for many predators."

    value = evaluator.evaluate_step(problem, previous_steps, current_step)
    print(f"Value estimate: {value:.4f}")
    print(f"Stats: {evaluator.get_stats()}")
