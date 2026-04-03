"""Training script for value network."""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from rl_controller.value_network import ValueNetwork, ValueNetworkConfig, ValueNetworkTrainer
from rl_controller.state_embedder import StateEmbedder, MockEmbedder
from rl_controller.replay_buffer import ReplayBuffer, TrainingDataPipeline


class ValueNetworkTrainingPipeline:
    """Complete training pipeline for value network."""

    def __init__(
        self,
        embedding_dim: int = 392,  # 384 (base) + 8 (metadata)
        hidden_dim: int = 256,
        num_layers: int = 2,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        device: str = None
    ):
        """Initialize training pipeline.

        Args:
            embedding_dim: Input embedding dimension
            hidden_dim: Hidden layer dimension
            num_layers: Number of hidden layers
            learning_rate: Learning rate for optimizer
            batch_size: Training batch size
            device: Device to use (auto-detected if None)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size

        # Initialize embedder
        self.embedder = StateEmbedder(device=self.device)

        # Initialize value network
        config = ValueNetworkConfig(
            input_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            learning_rate=learning_rate
        )
        self.model = ValueNetwork(config).to(self.device)

        # Initialize trainer
        self.trainer = ValueNetworkTrainer(self.model, config)

        # Initialize data pipeline
        self.buffer = ReplayBuffer(capacity=50000)
        self.data_pipeline = TrainingDataPipeline(self.buffer, self.embedder)

        logger.info(f"TrainingPipeline initialized (device={self.device})")

    def collect_training_data_from_results(self, results_dir: str = "data/results"):
        """Collect training data from existing experiment results.

        Args:
            results_dir: Directory containing result files
        """
        results_path = Path(results_dir)
        if not results_path.exists():
            logger.warning(f"Results directory not found: {results_dir}")
            return

        # Process all JSON files
        json_files = list(results_path.glob("*.json"))
        logger.info(f"Found {len(json_files)} result files")

        for json_file in json_files:
            try:
                self.data_pipeline.process_results_file(str(json_file))
            except Exception as e:
                logger.error(f"Error processing {json_file}: {e}")

        logger.info(f"Collected {len(self.buffer)} training examples")

    def create_synthetic_training_data(self, n_samples: int = 1000):
        """Create synthetic training data for testing.

        Args:
            n_samples: Number of synthetic samples
        """
        problems = [
            "Is the sky blue?",
            "Do dogs bark?",
            "What is 2+2?",
            "Why is water wet?",
            "How do birds fly?"
        ]

        reasoning_templates = [
            ["Consider the question.", "Apply reasoning.", "Reach conclusion."],
            ["First step.", "Second step.", "Third step.", "Final answer."],
            ["Analyze the problem.", "Generate solution.", "Verify answer."]
        ]

        for _ in range(n_samples):
            problem = np.random.choice(problems)
            chain = np.random.choice(reasoning_templates)

            # Random PRM scores
            scores = np.random.uniform(0.3, 0.9, len(chain)).tolist()

            # Random outcome (slightly biased toward correct)
            outcome = np.random.choice([1.0, 0.0], p=[0.6, 0.4])

            self.buffer.add_trajectory(
                problem=problem,
                reasoning_chain=chain,
                prm_scores=scores,
                final_outcome=outcome
            )

        logger.info(f"Created {n_samples} synthetic training samples")

    def prepare_training_data(self) -> tuple:
        """Prepare training data from buffer.

        Returns:
            Tuple of (train_data, val_data)
        """
        # Get all experiences
        experiences = self.buffer.get_all()

        if len(experiences) == 0:
            logger.error("No training data available")
            return None, None

        logger.info(f"Preparing {len(experiences)} samples for training")

        # Compute embeddings
        embeddings = []
        rewards = []

        for i, exp in enumerate(experiences):
            if i % 100 == 0:
                logger.debug(f"Embedding {i}/{len(experiences)}")

            emb = self.embedder.embed_state(
                problem=exp["problem"],
                previous_steps=exp["previous_steps"],
                current_step=exp["current_step"],
                score=exp["prm_score"]
            )

            # Combine PRM score and outcome
            outcome = exp.get("outcome_reward", exp["prm_score"])
            reward = 0.7 * outcome + 0.3 * exp["prm_score"]

            embeddings.append(emb)
            rewards.append(reward)

        # Convert to tensors
        X = torch.tensor(np.array(embeddings), dtype=torch.float32)
        y = torch.tensor(np.array(rewards), dtype=torch.float32).unsqueeze(1)

        # Normalize rewards to [-1, 1]
        y = torch.tanh((y - 0.5) * 2)

        # Split into train/val (80/20)
        n_samples = len(X)
        n_train = int(n_samples * 0.8)

        indices = torch.randperm(n_samples)
        train_indices = indices[:n_train]
        val_indices = indices[n_train:]

        X_train, y_train = X[train_indices], y[train_indices]
        X_val, y_val = X[val_indices], y[val_indices]

        # Move to device
        X_train = X_train.to(self.device)
        y_train = y_train.to(self.device)
        X_val = X_val.to(self.device)
        y_val = y_val.to(self.device)

        logger.info(f"Training data: {X_train.shape}, Validation data: {X_val.shape}")

        return (X_train, y_train), (X_val, y_val)

    def train(
        self,
        n_epochs: int = 20,
        patience: int = 5,
        save_dir: str = "models"
    ):
        """Train the value network.

        Args:
            n_epochs: Maximum number of epochs
            patience: Early stopping patience
            save_dir: Directory to save model
        """
        # Prepare data
        train_data, val_data = self.prepare_training_data()

        if train_data is None:
            logger.error("Training failed: no data")
            return

        X_train, y_train = train_data
        X_val, y_val = val_data

        # Create data loader
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0

        logger.info(f"Starting training for {n_epochs} epochs")

        for epoch in range(n_epochs):
            # Train epoch
            train_losses = []
            for batch_x, batch_y in train_loader:
                loss = self.trainer.train_step(batch_x, batch_y)
                train_losses.append(loss)

            avg_train_loss = np.mean(train_losses)

            # Validate
            val_metrics = self.trainer.evaluate(X_val, y_val)
            val_loss = val_metrics["loss"]

            logger.info(
                f"Epoch {epoch+1}/{n_epochs} - "
                f"Train Loss: {avg_train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}, "
                f"Val MAE: {val_metrics['mae']:.4f}, "
                f"Val Acc: {val_metrics['accuracy']:.2%}"
            )

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0

                # Save best model
                save_path = Path(save_dir)
                save_path.mkdir(parents=True, exist_ok=True)
                model_file = save_path / "best_value_network.pt"
                self.trainer.save(str(model_file))
            else:
                patience_counter += 1

            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

        logger.info(f"Training complete. Best val loss: {best_val_loss:.4f}")

    def save_final_model(self, save_path: str):
        """Save the final trained model.

        Args:
            save_path: Path to save model
        """
        self.trainer.save(save_path)
        logger.info(f"Model saved to {save_path}")

    def load_model(self, load_path: str):
        """Load a pre-trained model.

        Args:
            load_path: Path to model file
        """
        self.trainer.load(load_path)
        logger.info(f"Model loaded from {load_path}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train value network")
    parser.add_argument("--results-dir", type=str, default="data/results",
                        help="Directory containing experiment results")
    parser.add_argument("--n-epochs", type=int, default=20,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Training batch size")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument("--hidden-dim", type=int, default=256,
                        help="Hidden layer dimension")
    parser.add_argument("--save-dir", type=str, default="models",
                        help="Directory to save trained model")
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic training data")
    parser.add_argument("--n-synthetic", type=int, default=1000,
                        help="Number of synthetic samples")

    args = parser.parse_args()

    # Initialize pipeline
    pipeline = ValueNetworkTrainingPipeline(
        hidden_dim=args.hidden_dim,
        learning_rate=args.lr,
        batch_size=args.batch_size
    )

    # Collect training data
    if args.synthetic:
        logger.info("Using synthetic training data")
        pipeline.create_synthetic_training_data(n_samples=args.n_synthetic)
    else:
        logger.info("Collecting training data from results")
        pipeline.collect_training_data_from_results(args.results_dir)

    # Get buffer stats
    stats = pipeline.buffer.get_stats()
    logger.info(f"Buffer stats: {stats}")

    # Train
    pipeline.train(n_epochs=args.n_epochs, save_dir=args.save_dir)

    # Save final model
    final_path = Path(args.save_dir) / "value_network_final.pt"
    pipeline.save_final_model(str(final_path))

    logger.info("Training pipeline complete!")


if __name__ == "__main__":
    main()
