"""Tests for value network components."""

import pytest
import torch
import numpy as np
from src.rl_controller.value_network import ValueNetwork, ValueNetworkConfig, ValueNetworkTrainer
from src.rl_controller.state_embedder import StateEmbedder, MockEmbedder
from src.rl_controller.replay_buffer import ReplayBuffer
from src.evaluator.value_network_evaluator import ValueNetworkEvaluator


class TestStateEmbedder:
    """Tests for state embedder."""

    def test_embed_text(self):
        """Test text embedding."""
        embedder = MockEmbedder(embedding_dim=384)

        text = "Test reasoning step"
        embedding = embedder.embed_text(text)

        assert embedding.shape[0] == 384
        assert isinstance(embedding, np.ndarray)

    def test_embed_state(self):
        """Test state embedding with metadata."""
        embedder = MockEmbedder(embedding_dim=384)

        problem = "Test problem?"
        previous_steps = ["Step 1", "Step 2"]
        current_step = "Step 3"

        embedding = embedder.embed_state(
            problem=problem,
            previous_steps=previous_steps,
            current_step=current_step,
            score=0.8
        )

        # Should be embedding_dim + metadata_features (8)
        assert embedding.shape[0] == 384 + 8

    def test_cache_hit(self):
        """Test embedding cache."""
        embedder = MockEmbedder(embedding_dim=384)

        text = "Cached text"

        # First call
        emb1 = embedder.embed_text(text)

        # Second call (should hit cache)
        emb2 = embedder.embed_text(text)

        # Should be same
        assert np.allclose(emb1, emb2)

        # Check stats
        stats = embedder.get_cache_stats()
        assert stats["cache_hits"] == 1
        assert stats["cache_misses"] == 1

    def test_get_embedding_dim(self):
        """Test embedding dimension."""
        embedder = MockEmbedder(embedding_dim=384)

        # Base + metadata
        expected_dim = 384 + 8
        assert embedder.get_embedding_dim() == expected_dim


class TestValueNetwork:
    """Tests for value network."""

    def test_forward(self):
        """Test forward pass."""
        config = ValueNetworkConfig(input_dim=392, hidden_dim=128)
        model = ValueNetwork(config)

        batch_size = 10
        x = torch.randn(batch_size, 392)

        output = model(x)

        assert output.shape == (batch_size, 1)
        # Output should be in [-1, 1] due to tanh
        assert output.min() >= -1.0
        assert output.max() <= 1.0

    def test_predict_single(self):
        """Test single prediction."""
        config = ValueNetworkConfig(input_dim=392)
        model = ValueNetwork(config)

        embedding = torch.randn(392)
        value = model.predict(embedding)

        assert isinstance(value, float)
        assert -1.0 <= value <= 1.0

    def test_predict_batch(self):
        """Test batch prediction."""
        config = ValueNetworkConfig(input_dim=392)
        model = ValueNetwork(config)

        batch_size = 5
        embeddings = torch.randn(batch_size, 392)
        values = model.predict_batch(embeddings)

        assert len(values) == batch_size
        for v in values:
            assert -1.0 <= v <= 1.0


class TestValueNetworkTrainer:
    """Tests for value network trainer."""

    def test_train_step(self):
        """Test single training step."""
        config = ValueNetworkConfig(input_dim=392, hidden_dim=128)
        model = ValueNetwork(config)
        trainer = ValueNetworkTrainer(model, config)

        batch_size = 8
        states = torch.randn(batch_size, 392)
        targets = torch.randn(batch_size, 1)

        loss = trainer.train_step(states, targets)

        assert isinstance(loss, float)
        assert loss >= 0

    def test_evaluate(self):
        """Test evaluation."""
        config = ValueNetworkConfig(input_dim=392)
        model = ValueNetwork(config)
        trainer = ValueNetworkTrainer(model, config)

        batch_size = 10
        states = torch.randn(batch_size, 392)
        targets = torch.randn(batch_size, 1)

        metrics = trainer.evaluate(states, targets)

        assert "loss" in metrics
        assert "mae" in metrics
        assert "accuracy" in metrics


class TestReplayBuffer:
    """Tests for replay buffer."""

    def test_add_single(self):
        """Test adding single experience."""
        buffer = ReplayBuffer(capacity=100)

        buffer.add(
            problem="Test problem",
            previous_steps=["Step 1"],
            current_step="Step 2",
            prm_score=0.8
        )

        assert len(buffer) == 1

    def test_add_trajectory(self):
        """Test adding complete trajectory."""
        buffer = ReplayBuffer(capacity=100)

        reasoning_chain = ["Step 1", "Step 2", "Step 3"]
        prm_scores = [0.7, 0.8, 0.9]

        buffer.add_trajectory(
            problem="Test problem",
            reasoning_chain=reasoning_chain,
            prm_scores=prm_scores,
            final_outcome=1.0
        )

        assert len(buffer) == len(reasoning_chain)

    def test_sample(self):
        """Test sampling from buffer."""
        buffer = ReplayBuffer(capacity=100)

        # Add some experiences
        for i in range(10):
            buffer.add(
                problem=f"Problem {i}",
                previous_steps=[],
                current_step=f"Step {i}",
                prm_score=0.5
            )

        # Sample batch
        batch = buffer.sample(batch_size=5)

        assert len(batch) == 5

    def test_get_stats(self):
        """Test buffer statistics."""
        buffer = ReplayBuffer(capacity=100)

        # Add experiences
        for i in range(10):
            buffer.add(
                problem=f"Problem {i}",
                previous_steps=[],
                current_step=f"Step {i}",
                prm_score=0.5,
                outcome_reward=1.0 if i % 2 == 0 else 0.0
            )

        stats = buffer.get_stats()

        assert stats["size"] == 10
        assert stats["positive_outcomes"] == 5
        assert stats["negative_outcomes"] == 5


class TestValueNetworkEvaluator:
    """Tests for value network evaluator."""

    def test_evaluate_step(self):
        """Test step evaluation."""
        evaluator = ValueNetworkEvaluator(use_mock_embedder=True)

        problem = "Test problem?"
        previous_steps = ["Step 1"]
        current_step = "Step 2"

        value = evaluator.evaluate_step(problem, previous_steps, current_step)

        assert isinstance(value, float)
        assert -1.0 <= value <= 1.0

    def test_evaluate_batch(self):
        """Test batch evaluation."""
        evaluator = ValueNetworkEvaluator(use_mock_embedder=True)

        states = [
            {
                "problem": f"Problem {i}",
                "previous_steps": ["Step 1"],
                "current_step": f"Step {i}"
            }
            for i in range(10)
        ]

        values = evaluator.evaluate_batch(states, batch_size=5)

        assert len(values) == 10
        for v in values:
            assert -1.0 <= v <= 1.0

    def test_get_stats(self):
        """Test evaluator statistics."""
        evaluator = ValueNetworkEvaluator(use_mock_embedder=True)

        # Evaluate some steps
        for i in range(5):
            evaluator.evaluate_step(f"Problem {i}", [], f"Step {i}")

        stats = evaluator.get_stats()

        assert stats["total_evaluations"] == 5
        assert stats["avg_inference_time"] > 0
