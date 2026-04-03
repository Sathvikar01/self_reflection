"""Embedding model for state representation in value network."""

import torch
import torch.nn as nn
from typing import List, Optional, Dict, Any
import numpy as np
from loguru import logger
import sys
from pathlib import Path

# Try to import sentence-transformers, fall back to simple embedding if not available
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("sentence-transformers not available, using fallback embedding")


class StateEmbedder:
    """Embeds reasoning state into fixed-size vectors for value network."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        embedding_dim: int = 384,
        device: Optional[str] = None
    ):
        """Initialize the state embedder.

        Args:
            model_name: Name of sentence-transformer model
            embedding_dim: Dimension of embeddings
            device: Device to use (cuda/cpu), auto-detected if None
        """
        self.model_name = model_name
        self.embedding_dim = embedding_dim

        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Initialize embedding model
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.info(f"Loading sentence-transformer model: {model_name}")
            self.model = SentenceTransformer(model_name, device=self.device)
            # Update embedding_dim based on model
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
        else:
            logger.warning("Using fallback random embedding (install sentence-transformers for real embeddings)")
            self.model = None
            self.embedding_dim = embedding_dim

        # Cache for embeddings
        self._cache: Dict[str, np.ndarray] = {}
        self._cache_hits = 0
        self._cache_misses = 0

        logger.info(f"StateEmbedder initialized (dim={self.embedding_dim}, device={self.device})")

    def embed_text(self, text: str) -> np.ndarray:
        """Embed a single text string.

        Args:
            text: Text to embed

        Returns:
            Numpy array of shape (embedding_dim,)
        """
        # Check cache
        cache_key = text
        if cache_key in self._cache:
            self._cache_hits += 1
            return self._cache[cache_key]

        self._cache_misses += 1

        # Generate embedding
        if self.model is not None:
            embedding = self.model.encode(text, convert_to_numpy=True)
        else:
            # Fallback: deterministic hash-based embedding
            embedding = self._fallback_embedding(text)

        # Cache it
        self._cache[cache_key] = embedding

        return embedding

    def embed_state(
        self,
        problem: str,
        previous_steps: List[str],
        current_step: str,
        score: Optional[float] = None
    ) -> np.ndarray:
        """Embed a reasoning state for value network input.

        Args:
            problem: Original problem statement
            previous_steps: List of previous reasoning steps
            current_step: Current reasoning step
            score: Optional PRM score for this state

        Returns:
            State embedding of shape (embedding_dim + metadata_dim,)
        """
        # Combine text for embedding
        combined_text = self._format_state_text(problem, previous_steps, current_step)

        # Get base embedding
        base_embedding = self.embed_text(combined_text)

        # Add metadata features
        metadata_features = self._extract_metadata_features(
            problem, previous_steps, current_step, score
        )

        # Concatenate embedding with metadata
        state_embedding = np.concatenate([base_embedding, metadata_features])

        return state_embedding

    def embed_batch(
        self,
        states: List[Dict[str, Any]],
        batch_size: int = 32
    ) -> np.ndarray:
        """Embed multiple states in batch.

        Args:
            states: List of state dictionaries
            batch_size: Batch size for processing

        Returns:
            Numpy array of shape (n_states, embedding_dim + metadata_dim)
        """
        embeddings = []

        for i in range(0, len(states), batch_size):
            batch = states[i:i+batch_size]
            batch_embeddings = []

            for state in batch:
                emb = self.embed_state(
                    problem=state.get("problem", ""),
                    previous_steps=state.get("previous_steps", []),
                    current_step=state.get("current_step", ""),
                    score=state.get("score")
                )
                batch_embeddings.append(emb)

            embeddings.extend(batch_embeddings)

        return np.array(embeddings)

    def _format_state_text(
        self,
        problem: str,
        previous_steps: List[str],
        current_step: str
    ) -> str:
        """Format state as text for embedding.

        Args:
            problem: Problem statement
            previous_steps: Previous reasoning steps
            current_step: Current step

        Returns:
            Formatted text string
        """
        parts = [f"Problem: {problem}"]

        if previous_steps:
            parts.append("Previous reasoning:")
            for i, step in enumerate(previous_steps, 1):
                parts.append(f"{i}. {step}")

        parts.append(f"Current step: {current_step}")

        return " ".join(parts)

    def _extract_metadata_features(
        self,
        problem: str,
        previous_steps: List[str],
        current_step: str,
        score: Optional[float]
    ) -> np.ndarray:
        """Extract metadata features for state.

        Args:
            problem: Problem statement
            previous_steps: Previous steps
            current_step: Current step
            score: PRM score

        Returns:
            Metadata feature vector
        """
        features = []

        # Number of previous steps
        features.append(len(previous_steps))

        # Length features
        features.append(len(problem))
        features.append(len(current_step))
        features.append(sum(len(s) for s in previous_steps))

        # Average step length
        avg_step_len = np.mean([len(s) for s in previous_steps]) if previous_steps else 0
        features.append(avg_step_len)

        # Score (if available)
        features.append(score if score is not None else 0.0)

        # Has numbers in current step
        has_numbers = any(char.isdigit() for char in current_step)
        features.append(float(has_numbers))

        # Has logical connectives
        logical_words = ["therefore", "because", "so", "thus", "hence", "since"]
        has_logical = any(word in current_step.lower() for word in logical_words)
        features.append(float(has_logical))

        return np.array(features, dtype=np.float32)

    def _fallback_embedding(self, text: str) -> np.ndarray:
        """Generate fallback embedding when sentence-transformers not available.

        Args:
            text: Text to embed

        Returns:
            Deterministic embedding based on text hash
        """
        # Use hash for deterministic embedding
        text_hash = hash(text)

        # Generate pseudo-random but deterministic embedding
        np.random.seed(abs(text_hash) % (2**32))
        embedding = np.random.randn(self.embedding_dim).astype(np.float32)

        # Normalize
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)

        return embedding

    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics.

        Returns:
            Dictionary with cache stats
        """
        total = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total if total > 0 else 0

        return {
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "cache_size": len(self._cache),
            "hit_rate": hit_rate
        }

    def clear_cache(self):
        """Clear embedding cache."""
        self._cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0
        logger.info("Embedding cache cleared")

    def get_embedding_dim(self) -> int:
        """Get total embedding dimension (base + metadata).

        Returns:
            Total embedding dimension
        """
        # Base embedding dim + metadata features (8)
        return self.embedding_dim + 8


class MockEmbedder(StateEmbedder):
    """Mock embedder for testing without sentence-transformers."""

    def __init__(self, embedding_dim: int = 768):
        """Initialize mock embedder.

        Args:
            embedding_dim: Dimension of mock embeddings
        """
        self.embedding_dim = embedding_dim
        self.device = "cpu"
        self.model = None
        self._cache: Dict[str, np.ndarray] = {}
        self._cache_hits = 0
        self._cache_misses = 0
        logger.info(f"MockEmbedder initialized (dim={embedding_dim})")

    def embed_text(self, text: str) -> np.ndarray:
        """Generate mock embedding.

        Args:
            text: Text to embed

        Returns:
            Mock embedding vector
        """
        cache_key = text
        if cache_key in self._cache:
            self._cache_hits += 1
            return self._cache[cache_key]

        self._cache_misses += 1

        # Deterministic random embedding based on text
        text_hash = hash(text)
        np.random.seed(abs(text_hash) % (2**32))
        embedding = np.random.randn(self.embedding_dim).astype(np.float32)
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)

        self._cache[cache_key] = embedding
        return embedding


if __name__ == "__main__":
    # Example usage
    embedder = StateEmbedder()

    problem = "Do hamsters provide food for any animals?"
    previous_steps = ["Hamsters are small rodents."]
    current_step = "Small rodents are prey for many predators."

    state_embedding = embedder.embed_state(
        problem=problem,
        previous_steps=previous_steps,
        current_step=current_step,
        score=0.8
    )

    print(f"State embedding shape: {state_embedding.shape}")
    print(f"Embedding dimension: {embedder.get_embedding_dim()}")
    print(f"Cache stats: {embedder.get_cache_stats()}")
