"""Score aggregation and normalization utilities."""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np


class AggregationMethod(Enum):
    """Methods for aggregating step scores."""
    MEAN = "mean"
    WEIGHTED_MEAN = "weighted_mean"
    MIN = "min"
    MAX = "max"
    PRODUCT = "product"
    EXPONENTIAL = "exponential"
    LAST = "last"
    CUMULATIVE = "cumulative"


@dataclass
class AggregatedScore:
    """Result of score aggregation."""
    score: float
    method: AggregationMethod
    num_steps: int
    individual_scores: List[float]
    confidence: float


class ScoreNormalizer:
    """Normalizes scores to standard range."""
    
    def __init__(
        self,
        target_min: float = -1.0,
        target_max: float = 1.0,
        source_min: Optional[float] = None,
        source_max: Optional[float] = None,
    ):
        self.target_min = target_min
        self.target_max = target_max
        self.source_min = source_min
        self.source_max = source_max
        self._fitted = source_min is not None and source_max is not None
        self._min_seen = float('inf')
        self._max_seen = float('-inf')
    
    def fit(self, scores: List[float]) -> 'ScoreNormalizer':
        """Fit normalizer to observed scores."""
        if scores:
            self._min_seen = min(scores)
            self._max_seen = max(scores)
            if self.source_min is None:
                self.source_min = self._min_seen
            if self.source_max is None:
                self.source_max = self._max_seen
            self._fitted = True
        return self
    
    def normalize(self, score: float) -> float:
        """Normalize a single score."""
        if not self._fitted:
            return score
        
        source_range = self.source_max - self.source_min
        if source_range == 0:
            return (self.target_min + self.target_max) / 2
        
        normalized = (score - self.source_min) / source_range
        return self.target_min + normalized * (self.target_max - self.target_min)
    
    def normalize_batch(self, scores: List[float]) -> List[float]:
        """Normalize a batch of scores."""
        return [self.normalize(s) for s in scores]
    
    def denormalize(self, score: float) -> float:
        """Convert back to source range."""
        if not self._fitted:
            return score
        
        target_range = self.target_max - self.target_min
        if target_range == 0:
            return self.source_min
        
        normalized = (score - self.target_min) / target_range
        return self.source_min + normalized * (self.source_max - self.source_min)


class ScoreAggregator:
    """Aggregates step scores using various methods."""
    
    WEIGHT_DECAY = 0.95
    EXPONENTIAL_BASE = 2.0
    
    def __init__(
        self,
        method: AggregationMethod = AggregationMethod.WEIGHTED_MEAN,
        discount_factor: float = 0.95,
    ):
        self.method = method
        self.discount_factor = discount_factor
    
    def aggregate(
        self,
        scores: List[float],
        weights: Optional[List[float]] = None,
    ) -> AggregatedScore:
        """Aggregate scores using configured method.
        
        Args:
            scores: List of step scores
            weights: Optional list of weights for each step
        
        Returns:
            AggregatedScore with result and metadata
        """
        if not scores:
            return AggregatedScore(
                score=0.0,
                method=self.method,
                num_steps=0,
                individual_scores=[],
                confidence=0.0,
            )
        
        if weights and len(weights) != len(scores):
            weights = None
        
        aggregated = self._compute_aggregate(scores, weights)
        confidence = self._compute_confidence(scores)
        
        return AggregatedScore(
            score=aggregated,
            method=self.method,
            num_steps=len(scores),
            individual_scores=scores,
            confidence=confidence,
        )
    
    def _compute_aggregate(
        self,
        scores: List[float],
        weights: Optional[List[float]] = None,
    ) -> float:
        """Compute aggregated score."""
        if self.method == AggregationMethod.MEAN:
            return np.mean(scores)
        
        elif self.method == AggregationMethod.WEIGHTED_MEAN:
            if weights is None:
                weights = [self.WEIGHT_DECAY ** i for i in range(len(scores))]
            weights = np.array(weights)
            weights = weights / weights.sum()
            return np.average(scores, weights=weights)
        
        elif self.method == AggregationMethod.MIN:
            return min(scores)
        
        elif self.method == AggregationMethod.MAX:
            return max(scores)
        
        elif self.method == AggregationMethod.PRODUCT:
            normalized = [(s + 1) / 2 for s in scores]
            product = np.prod(normalized)
            return product * 2 - 1
        
        elif self.method == AggregationMethod.EXPONENTIAL:
            weights = [self.EXPONENTIAL_BASE ** (-i) for i in range(len(scores))]
            weights = np.array(weights)
            weights = weights / weights.sum()
            return np.average(scores, weights=weights)
        
        elif self.method == AggregationMethod.LAST:
            return scores[-1]
        
        elif self.method == AggregationMethod.CUMULATIVE:
            cumulative = 0.0
            for i, score in enumerate(scores):
                cumulative = score + self.discount_factor * cumulative
            return cumulative
        
        else:
            return np.mean(scores)
    
    def _compute_confidence(self, scores: List[float]) -> float:
        """Compute confidence based on score consistency."""
        if len(scores) <= 1:
            return 1.0
        
        variance = np.var(scores)
        max_variance = 1.0
        confidence = 1.0 - min(variance / max_variance, 1.0)
        return confidence
    
    def compute_step_rewards(
        self,
        scores: List[float],
        gamma: float = 0.95,
    ) -> List[float]:
        """Compute discounted rewards for each step.
        
        Uses temporal difference to compute value estimates.
        """
        if not scores:
            return []
        
        rewards = []
        cumulative = 0.0
        
        for i in range(len(scores) - 1, -1, -1):
            cumulative = scores[i] + gamma * cumulative
            rewards.insert(0, cumulative)
        
        return rewards
    
    @staticmethod
    def interpolate_score(
        score: float,
        from_range: Tuple[float, float] = (-1.0, 1.0),
        to_range: Tuple[float, float] = (0.0, 1.0),
    ) -> float:
        """Interpolate score from one range to another."""
        from_min, from_max = from_range
        to_min, to_max = to_range
        
        normalized = (score - from_min) / (from_max - from_min)
        return to_min + normalized * (to_max - to_min)
    
    @staticmethod
    def softmax_scores(scores: List[float], temperature: float = 1.0) -> List[float]:
        """Apply softmax to scores for probability distribution."""
        exp_scores = np.exp(np.array(scores) / temperature)
        return list(exp_scores / exp_scores.sum())


class RollingScoreTracker:
    """Tracks rolling statistics of scores."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.scores: List[float] = []
        self._sum = 0.0
        self._sum_sq = 0.0
    
    def add(self, score: float):
        """Add a score to the tracker."""
        if len(self.scores) >= self.window_size:
            removed = self.scores.pop(0)
            self._sum -= removed
            self._sum_sq -= removed * removed
        
        self.scores.append(score)
        self._sum += score
        self._sum_sq += score * score
    
    def mean(self) -> float:
        """Get mean of tracked scores."""
        if not self.scores:
            return 0.0
        return self._sum / len(self.scores)
    
    def std(self) -> float:
        """Get standard deviation of tracked scores."""
        if len(self.scores) < 2:
            return 0.0
        variance = (self._sum_sq - self._sum ** 2 / len(self.scores)) / (len(self.scores) - 1)
        return max(0, variance) ** 0.5
    
    def recent(self, n: int = 10) -> List[float]:
        """Get n most recent scores."""
        return self.scores[-n:]
    
    def clear(self):
        """Clear all tracked scores."""
        self.scores.clear()
        self._sum = 0.0
        self._sum_sq = 0.0
