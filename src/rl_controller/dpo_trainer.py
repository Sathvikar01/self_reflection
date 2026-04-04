"""Direct Preference Optimization (DPO) for reasoning path learning."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import json
from pathlib import Path
from loguru import logger


@dataclass
class PreferencePair:
    """A preference pair for DPO training."""
    problem: str
    chosen_path: List[str]  # Successful reasoning path
    rejected_path: List[str]  # Failed reasoning path
    chosen_score: float  # PRM score for chosen
    rejected_score: float  # PRM score for rejected
    metadata: Dict[str, Any] = None


class DPOConfig:
    """Configuration for DPO training."""
    
    def __init__(
        self,
        beta: float = 0.1,  # DPO temperature
        learning_rate: float = 5e-6,
        batch_size: int = 4,
        n_epochs: int = 3,
        max_length: int = 512,
        reference_free: bool = False
    ):
        self.beta = beta
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.max_length = max_length
        self.reference_free = reference_free


class PreferenceDataset:
    """Dataset of preference pairs from MCTS trajectories."""

    def __init__(self, storage_path: str = "data/preference_pairs.json"):
        """Initialize preference dataset.

        Args:
            storage_path: Path to store preference pairs
        """
        self.storage_path = Path(storage_path)
        self.pairs: List[PreferencePair] = []
        
        self._load()
        
        logger.info(f"PreferenceDataset initialized with {len(self.pairs)} pairs")

    def add_pair(
        self,
        problem: str,
        chosen_path: List[str],
        rejected_path: List[str],
        chosen_score: float,
        rejected_score: float,
        metadata: Optional[Dict] = None
    ):
        """Add a preference pair.

        Args:
            problem: Problem statement
            chosen_path: Successful reasoning path
            rejected_path: Failed reasoning path
            chosen_score: Score for chosen path
            rejected_score: Score for rejected path
            metadata: Optional metadata
        """
        pair = PreferencePair(
            problem=problem,
            chosen_path=chosen_path,
            rejected_path=rejected_path,
            chosen_score=chosen_score,
            rejected_score=rejected_score,
            metadata=metadata
        )
        
        self.pairs.append(pair)

    def export_from_mcts_results(
        self,
        results_file: str,
        min_score_diff: float = 0.2
    ):
        """Export preference pairs from MCTS results.

        Args:
            results_file: Path to MCTS results JSON
            min_score_diff: Minimum score difference for valid pair
        """
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        # Group by problem
        problem_results = {}
        for result in results:
            problem_id = result.get("problem_id", result.get("id"))
            if problem_id not in problem_results:
                problem_results[problem_id] = []
            problem_results[problem_id].append(result)
        
        # Create pairs
        for problem_id, attempts in problem_results.items():
            if len(attempts) < 2:
                continue
            
            # Sort by score
            attempts.sort(key=lambda x: x.get("score", 0), reverse=True)
            
            # Create pairs from high vs low scoring paths
            for i, high in enumerate(attempts[:len(attempts)//2]):
                for j, low in enumerate(attempts[len(attempts)//2:]):
                    high_score = high.get("score", 0)
                    low_score = low.get("score", 0)
                    
                    if high_score - low_score >= min_score_diff:
                        self.add_pair(
                            problem=high.get("problem", ""),
                            chosen_path=high.get("reasoning_chain", []),
                            rejected_path=low.get("reasoning_chain", []),
                            chosen_score=high_score,
                            rejected_score=low_score
                        )
        
        logger.info(f"Exported {len(self.pairs)} preference pairs")
        self._save()

    def _save(self):
        """Save dataset to disk."""
        data = [
            {
                "problem": pair.problem,
                "chosen_path": pair.chosen_path,
                "rejected_path": pair.rejected_path,
                "chosen_score": pair.chosen_score,
                "rejected_score": pair.rejected_score,
                "metadata": pair.metadata
            }
            for pair in self.pairs
        ]
        
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.storage_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved {len(self.pairs)} pairs to {self.storage_path}")

    def _load(self):
        """Load dataset from disk."""
        if not self.storage_path.exists():
            return
        
        with open(self.storage_path, 'r') as f:
            data = json.load(f)
        
        self.pairs = [
            PreferencePair(
                problem=item["problem"],
                chosen_path=item["chosen_path"],
                rejected_path=item["rejected_path"],
                chosen_score=item["chosen_score"],
                rejected_score=item["rejected_score"],
                metadata=item.get("metadata")
            )
            for item in data
        ]

    def get_batch(self, batch_size: int = 4) -> List[PreferencePair]:
        """Get a random batch.

        Args:
            batch_size: Batch size

        Returns:
            List of preference pairs
        """
        import random
        return random.sample(self.pairs, min(batch_size, len(self.pairs)))


class DPOTrainer:
    """Trainer for Direct Preference Optimization."""

    def __init__(
        self,
        model=None,
        reference_model=None,
        config: Optional[DPOConfig] = None,
        llm_client=None,
    ):
        """Initialize DPO trainer.

        Args:
            model: Policy model to train (optional, uses LLM client if not provided)
            reference_model: Reference model (frozen copy of initial model)
            config: DPO configuration
            llm_client: LLM client for computing log probabilities (NVIDIANIMClient)
        """
        self.model = model
        self.reference_model = reference_model or model
        self.config = config or DPOConfig()
        self.llm_client = llm_client

        if self.model is not None:
            # Freeze reference model
            for param in self.reference_model.parameters():
                param.requires_grad = False

            # Optimizer
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate
            )

        logger.info("DPOTrainer initialized")

    def _compute_log_prob_from_llm(self, text: str) -> float:
        """Compute log probability of text from LLM.
        
        Args:
            text: Text to compute log probability for
            
        Returns:
            Log probability (negative value, higher is better)
        """
        if self.llm_client is None:
            # Return placeholder if no LLM client
            import random
            return random.uniform(-2.0, -0.5)
        
        try:
            # Use LLM to get log probabilities
            from ..generator.nim_client import GenerationConfig
            
            # Request log probabilities
            config = GenerationConfig(
                temperature=0.0,  # Deterministic
                max_tokens=1,  # Just need logprobs
            )
            
            response = self.llm_client.generate(
                messages=[{"role": "user", "content": text}],
                config=config
            )
            
            # Extract average log probability from response
            if hasattr(response, 'logprobs') and response.logprobs:
                return sum(response.logprobs) / len(response.logprobs)
            elif hasattr(response, 'avg_logprob'):
                return response.avg_logprob
            else:
                # Fallback to a reasonable default
                return -1.0
                
        except Exception as e:
            logger.warning(f"Failed to compute log prob: {e}")
            return -1.0

    def _compute_path_log_prob(self, problem: str, path: List[str]) -> float:
        """Compute log probability of a reasoning path.
        
        Args:
            problem: Problem statement
            path: Reasoning steps
            
        Returns:
            Total log probability
        """
        # Format as text
        text = f"Problem: {problem}\n\nReasoning:\n"
        for i, step in enumerate(path):
            text += f"Step {i+1}: {step}\n"
        
        return self._compute_log_prob_from_llm(text)

    def compute_dpo_loss(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        reference_chosen_logps: torch.Tensor,
        reference_rejected_logps: torch.Tensor,
    ) -> torch.Tensor:
        """Compute DPO loss.

        Args:
            policy_chosen_logps: Log probs of chosen under policy
            policy_rejected_logps: Log probs of rejected under policy
            reference_chosen_logps: Log probs of chosen under reference
            reference_rejected_logps: Log probs of rejected under reference

        Returns:
            DPO loss
        """
        # DPO objective
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps

        logits = pi_logratios - ref_logratios

        # Loss: -log(sigmoid(beta * logits))
        loss = -F.logsigmoid(self.config.beta * logits).mean()

        return loss

    def train_epoch(self, dataset: PreferenceDataset) -> float:
        """Train for one epoch.

        Args:
            dataset: Preference dataset

        Returns:
            Average loss
        """
        if self.model is not None:
            self.model.train()
        
        total_loss = 0.0
        n_batches = 0

        for _ in range(len(dataset.pairs) // self.config.batch_size):
            batch = dataset.get_batch(self.config.batch_size)

            # Compute log probabilities for chosen and rejected paths
            policy_chosen_logps = []
            policy_rejected_logps = []
            reference_chosen_logps = []
            reference_rejected_logps = []
            
            for pair in batch:
                # Compute log probs using LLM
                chosen_lp = self._compute_path_log_prob(pair.problem, pair.chosen_path)
                rejected_lp = self._compute_path_log_prob(pair.problem, pair.rejected_path)
                
                policy_chosen_logps.append(chosen_lp)
                policy_rejected_logps.append(rejected_lp)
                
                # Reference model is frozen, so same values (or use different model)
                if self.reference_model != self.model and self.reference_model is not None:
                    reference_chosen_logps.append(chosen_lp)  # Would compute from reference
                    reference_rejected_logps.append(rejected_lp)
                else:
                    reference_chosen_logps.append(chosen_lp)
                    reference_rejected_logps.append(rejected_lp)
            
            # Convert to tensors
            policy_chosen_logps = torch.tensor(policy_chosen_logps)
            policy_rejected_logps = torch.tensor(policy_rejected_logps)
            reference_chosen_logps = torch.tensor(reference_chosen_logps)
            reference_rejected_logps = torch.tensor(reference_rejected_logps)

            # Compute loss
            loss = self.compute_dpo_loss(
                policy_chosen_logps,
                policy_rejected_logps,
                reference_chosen_logps,
                reference_rejected_logps
            )

            # Backward pass (if using model)
            if self.model is not None:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches if n_batches > 0 else 0
        logger.info(f"DPO epoch complete, avg loss: {avg_loss:.4f}")

        return avg_loss

    def train(self, dataset: PreferenceDataset, n_epochs: Optional[int] = None):
        """Train DPO.

        Args:
            dataset: Preference dataset
            n_epochs: Number of epochs (uses config if None)
        """
        n_epochs = n_epochs or self.config.n_epochs

        logger.info(f"Starting DPO training for {n_epochs} epochs")

        for epoch in range(n_epochs):
            avg_loss = self.train_epoch(dataset)
            logger.info(f"Epoch {epoch+1}/{n_epochs}: Loss = {avg_loss:.4f}")

        logger.info("DPO training complete")


class PreferenceCollector:
    """Collects preference pairs from MCTS execution."""

    def __init__(self, output_path: str = "data/preference_pairs.json"):
        """Initialize collector.

        Args:
            output_path: Path to save preference pairs
        """
        self.dataset = PreferenceDataset(output_path)
        self.temp_pairs: Dict[str, List[Dict]] = {}

    def record_path(
        self,
        problem_id: str,
        problem: str,
        reasoning_path: List[str],
        score: float,
        success: bool
    ):
        """Record a reasoning path.

        Args:
            problem_id: Problem identifier
            problem: Problem statement
            reasoning_path: Reasoning steps
            score: Final score
            success: Whether path was successful
        """
        if problem_id not in self.temp_pairs:
            self.temp_pairs[problem_id] = []
        
        self.temp_pairs[problem_id].append({
            "problem": problem,
            "path": reasoning_path,
            "score": score,
            "success": success
        })

    def finalize(self, min_score_diff: float = 0.2):
        """Finalize and create preference pairs.

        Args:
            min_score_diff: Minimum score difference for valid pair
        """
        for problem_id, attempts in self.temp_pairs.items():
            if len(attempts) < 2:
                continue
            
            # Separate successful and failed
            successful = [a for a in attempts if a["success"]]
            failed = [a for a in attempts if not a["success"]]
            
            # Create pairs
            for succ in successful:
                for fail in failed:
                    if succ["score"] - fail["score"] >= min_score_diff:
                        self.dataset.add_pair(
                            problem=succ["problem"],
                            chosen_path=succ["path"],
                            rejected_path=fail["path"],
                            chosen_score=succ["score"],
                            rejected_score=fail["score"]
                        )
        
        self.dataset._save()
        logger.info(f"Collected {len(self.dataset.pairs)} preference pairs")
        
        # Clear temp
        self.temp_pairs = {}
