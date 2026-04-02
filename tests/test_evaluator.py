"""Tests for evaluator components."""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluator.prm_client import PRMEvaluator, PRMConfig, EvaluationResult
from src.evaluator.scoring import ScoreAggregator, ScoreNormalizer, AggregationMethod


class TestPRMEvaluator:
    """Tests for PRM evaluator."""
    
    def test_config_creation(self):
        """Test PRM configuration."""
        config = PRMConfig(
            model="meta/llama-3.3-70b-instruct",
            temperature=0.1,
        )
        
        assert config.model == "meta/llama-3.3-70b-instruct"
        assert config.temperature == 0.1
    
    def test_score_parsing(self):
        """Test score parsing from responses."""
        config = PRMConfig()
        
        with patch('src.evaluator.prm_client.NVIDIANIMClient'):
            evaluator = PRMEvaluator(config=config)
            
            score, success = evaluator._parse_score("0.75")
            assert success
            assert score == 0.75
            
            score, success = evaluator._parse_score("-0.5")
            assert success
            assert score == -0.5
            
            score, success = evaluator._parse_score("Score: 0.8")
            assert success
            assert score == 0.8
    
    def test_evaluation_result(self):
        """Test evaluation result creation."""
        result = EvaluationResult(
            score=0.75,
            raw_response="0.75",
            confidence=0.9,
            parsing_success=True,
        )
        
        assert result.score == 0.75
        assert result.is_valid()
    
    def test_invalid_result(self):
        """Test invalid evaluation result."""
        result = EvaluationResult(
            score=2.0,
            raw_response="invalid",
            confidence=0.5,
            parsing_success=False,
        )
        
        assert not result.is_valid()


class TestScoreAggregator:
    """Tests for score aggregator."""
    
    def test_mean_aggregation(self):
        """Test mean aggregation."""
        aggregator = ScoreAggregator(method=AggregationMethod.MEAN)
        
        result = aggregator.aggregate([0.5, 0.7, 0.9])
        
        assert result.score == pytest.approx(0.7, abs=0.01)
        assert result.method == AggregationMethod.MEAN
    
    def test_min_aggregation(self):
        """Test min aggregation."""
        aggregator = ScoreAggregator(method=AggregationMethod.MIN)
        
        result = aggregator.aggregate([0.5, 0.7, 0.9])
        
        assert result.score == 0.5
    
    def test_max_aggregation(self):
        """Test max aggregation."""
        aggregator = ScoreAggregator(method=AggregationMethod.MAX)
        
        result = aggregator.aggregate([0.5, 0.7, 0.9])
        
        assert result.score == 0.9
    
    def test_last_aggregation(self):
        """Test last aggregation."""
        aggregator = ScoreAggregator(method=AggregationMethod.LAST)
        
        result = aggregator.aggregate([0.5, 0.7, 0.9])
        
        assert result.score == 0.9
    
    def test_empty_scores(self):
        """Test aggregation with no scores."""
        aggregator = ScoreAggregator()
        
        result = aggregator.aggregate([])
        
        assert result.score == 0.0
        assert result.num_steps == 0
    
    def test_weighted_mean(self):
        """Test weighted mean aggregation."""
        aggregator = ScoreAggregator(method=AggregationMethod.WEIGHTED_MEAN)
        
        result = aggregator.aggregate([0.5, 0.7, 0.9])
        
        assert result.score > 0.5
        assert result.score < 0.9


class TestScoreNormalizer:
    """Tests for score normalizer."""
    
    def test_normalization(self):
        """Test score normalization."""
        normalizer = ScoreNormalizer(
            target_min=-1.0,
            target_max=1.0,
        )
        normalizer.fit([0.0, 10.0])
        
        normalized = normalizer.normalize(5.0)
        
        assert -1.0 <= normalized <= 1.0
    
    def test_denormalization(self):
        """Test score denormalization."""
        normalizer = ScoreNormalizer(
            target_min=-1.0,
            target_max=1.0,
            source_min=0.0,
            source_max=10.0,
        )
        normalizer._fitted = True
        
        original = normalizer.denormalize(0.0)
        
        assert original == pytest.approx(5.0, abs=0.01)
    
    def test_batch_normalization(self):
        """Test batch normalization."""
        normalizer = ScoreNormalizer()
        normalizer.fit([0.0, 100.0])
        
        normalized = normalizer.normalize_batch([25.0, 50.0, 75.0])
        
        assert len(normalized) == 3


class TestDiscountedRewards:
    """Tests for discounted reward computation."""
    
    def test_compute_step_rewards(self):
        """Test step reward computation."""
        aggregator = ScoreAggregator()
        
        rewards = aggregator.compute_step_rewards([1.0, 1.0, 1.0], gamma=0.9)
        
        assert len(rewards) == 3
        assert rewards[0] > rewards[1] > rewards[2]
    
    def test_softmax_scores(self):
        """Test softmax transformation."""
        aggregator = ScoreAggregator()
        
        probs = aggregator.softmax_scores([0.5, 0.7, 0.9])
        
        assert abs(sum(probs) - 1.0) < 0.001
        assert all(0 <= p <= 1 for p in probs)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
