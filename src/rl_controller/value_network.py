"""Optional value network for MCTS leaf evaluation."""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger


@dataclass
class ValueNetworkConfig:
    """Configuration for value network."""
    input_dim: int = 768
    hidden_dim: int = 256
    num_layers: int = 2
    output_dim: int = 1
    dropout: float = 0.1
    learning_rate: float = 0.001
    weight_decay: float = 1e-5


class ValueNetwork(nn.Module):
    """Simple MLP for value estimation.
    
    Takes embedded state representations and outputs value predictions.
    """
    
    def __init__(self, config: Optional[ValueNetworkConfig] = None):
        super().__init__()
        self.config = config or ValueNetworkConfig()
        
        layers = []
        in_dim = self.config.input_dim
        
        for i in range(self.config.num_layers):
            out_dim = self.config.hidden_dim if i < self.config.num_layers - 1 else self.config.output_dim
            layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.ReLU(),
                nn.Dropout(self.config.dropout),
            ])
            in_dim = out_dim
        
        if layers:
            layers = layers[:-2]
        
        self.network = nn.Sequential(*layers)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
        
        Returns:
            Value predictions of shape (batch_size, 1)
        """
        return torch.tanh(self.network(x))
    
    def predict(self, state_embedding: torch.Tensor) -> float:
        """Predict value for single state.
        
        Args:
            state_embedding: State embedding tensor
        
        Returns:
            Predicted value in [-1, 1]
        """
        self.eval()
        with torch.no_grad():
            value = self.forward(state_embedding.unsqueeze(0))
        return value.item()
    
    def predict_batch(self, state_embeddings: torch.Tensor) -> List[float]:
        """Predict values for batch of states.
        
        Args:
            state_embeddings: Batch of state embeddings (batch_size, input_dim)
        
        Returns:
            List of predicted values
        """
        self.eval()
        with torch.no_grad():
            values = self.forward(state_embeddings)
        return values.squeeze(-1).tolist()


class ValueNetworkTrainer:
    """Trainer for the value network."""
    
    def __init__(
        self,
        model: ValueNetwork,
        config: Optional[ValueNetworkConfig] = None,
    ):
        self.model = model
        self.config = config or ValueNetworkConfig()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        self.loss_fn = nn.MSELoss()
        
        self._training_steps = 0
        self._best_loss = float('inf')
    
    def train_step(
        self,
        states: torch.Tensor,
        targets: torch.Tensor,
    ) -> float:
        """Single training step.
        
        Args:
            states: State embeddings (batch_size, input_dim)
            targets: Target values (batch_size, 1)
        
        Returns:
            Loss value
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        predictions = self.model(states)
        loss = self.loss_fn(predictions, targets)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        self._training_steps += 1
        
        return loss.item()
    
    def train_epoch(
        self,
        data_loader: torch.utils.data.DataLoader,
        num_epochs: int = 10,
    ) -> List[float]:
        """Train for multiple epochs.
        
        Args:
            data_loader: DataLoader providing (state, target) pairs
            num_epochs: Number of epochs to train
        
        Returns:
            List of average losses per epoch
        """
        losses = []
        
        for epoch in range(num_epochs):
            epoch_losses = []
            
            for states, targets in data_loader:
                loss = self.train_step(states, targets)
                epoch_losses.append(loss)
            
            avg_loss = sum(epoch_losses) / len(epoch_losses)
            losses.append(avg_loss)
            
            if avg_loss < self._best_loss:
                self._best_loss = avg_loss
            
            logger.debug(f"Epoch {epoch + 1}/{num_epochs}: Loss = {avg_loss:.4f}")
        
        return losses
    
    def evaluate(
        self,
        states: torch.Tensor,
        targets: torch.Tensor,
    ) -> Dict[str, float]:
        """Evaluate model on validation data."""
        self.model.eval()
        
        with torch.no_grad():
            predictions = self.model(states)
            loss = self.loss_fn(predictions, targets)
            
            mae = F.l1_loss(predictions, targets)
            
            within_tolerance = torch.abs(predictions - targets) < 0.1
            accuracy = within_tolerance.float().mean()
        
        return {
            "loss": loss.item(),
            "mae": mae.item(),
            "accuracy": accuracy.item(),
        }
    
    def save(self, path: str):
        """Save model checkpoint."""
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
            "training_steps": self._training_steps,
            "best_loss": self._best_loss,
        }, path)
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, weights_only=True)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self._training_steps = checkpoint.get("training_steps", 0)
        self._best_loss = checkpoint.get("best_loss", float('inf'))
        logger.info(f"Model loaded from {path}")


class MockEmbedder:
    """Mock embedder for testing without a real embedding model."""
    
    def __init__(self, dim: int = 768):
        self.dim = dim
    
    def embed(self, text: str) -> torch.Tensor:
        """Generate mock embedding from text hash."""
        torch.manual_seed(hash(text) % (2**32))
        return torch.randn(self.dim)
    
    def embed_batch(self, texts: List[str]) -> torch.Tensor:
        """Generate mock embeddings for batch."""
        return torch.stack([self.embed(t) for t in texts])


class ValueEstimator:
    """Combines value network with state embedding for MCTS."""
    
    def __init__(
        self,
        model: Optional[ValueNetwork] = None,
        embedder: Optional[MockEmbedder] = None,
    ):
        self.model = model or ValueNetwork()
        self.embedder = embedder or MockEmbedder(self.model.config.input_dim)
    
    def estimate_value(
        self,
        problem: str,
        reasoning_steps: List[str],
    ) -> float:
        """Estimate value of current reasoning state.
        
        Args:
            problem: Original problem
            reasoning_steps: Current reasoning path
        
        Returns:
            Estimated value in [-1, 1]
        """
        state_text = problem + " [SEP] " + " [STEP] ".join(reasoning_steps)
        embedding = self.embedder.embed(state_text)
        return self.model.predict(embedding)
    
    def estimate_values_batch(
        self,
        states: List[Tuple[str, List[str]]],
    ) -> List[float]:
        """Estimate values for multiple states.
        
        Args:
            states: List of (problem, steps) tuples
        
        Returns:
            List of estimated values
        """
        texts = [
            p + " [SEP] " + " [STEP] ".join(s)
            for p, s in states
        ]
        embeddings = self.embedder.embed_batch(texts)
        return self.model.predict_batch(embeddings)
