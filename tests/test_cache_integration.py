"""Tests for cache integration with ActionExecutor."""

import pytest
from unittest.mock import Mock, MagicMock, patch
from dataclasses import dataclass

from src.utils.lru_cache import LRUCache, PersistentCache, CachedPRMEvaluator


class TestCachedPRMEvaluator:
    """Tests for CachedPRMEvaluator integration."""
    
    def test_initialization_with_lru_cache(self):
        """Test initialization with in-memory LRU cache."""
        mock_evaluator = Mock()
        cached = CachedPRMEvaluator(mock_evaluator, use_persistent=False)
        
        assert cached.prm_evaluator == mock_evaluator
        assert isinstance(cached.cache, LRUCache)
    
    def test_initialization_with_persistent_cache(self, tmp_path):
        """Test initialization with persistent cache."""
        mock_evaluator = Mock()
        cache_path = str(tmp_path / "test_cache.db")
        
        cached = CachedPRMEvaluator(
            mock_evaluator,
            use_persistent=True,
        )
        
        assert cached.prm_evaluator == mock_evaluator
    
    def test_evaluate_step_caches_result(self):
        """Test that evaluate_step caches results."""
        # Create mock evaluator that returns EvaluationResult-like object
        mock_result = Mock()
        mock_result.score = 0.85
        mock_result.confidence = 0.9
        
        mock_evaluator = Mock()
        mock_evaluator.evaluate_step = Mock(return_value=mock_result)
        
        cached = CachedPRMEvaluator(mock_evaluator, use_persistent=False)
        
        # First call - should call underlying evaluator
        result1 = cached.evaluate_step(
            problem="Test problem",
            previous_steps=["Step 1"],
            current_step="Step 2",
        )
        
        assert result1.score == 0.85
        assert mock_evaluator.evaluate_step.call_count == 1
        
        # Second call with same args - should use cache
        result2 = cached.evaluate_step(
            problem="Test problem",
            previous_steps=["Step 1"],
            current_step="Step 2",
        )
        
        # Cache hit - call count should still be 1
        assert mock_evaluator.evaluate_step.call_count == 1
        
        # Check cache stats
        stats = cached.get_cache_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
    
    def test_evaluate_step_different_problems(self):
        """Test that different problems are cached separately."""
        mock_evaluator = Mock()
        mock_evaluator.evaluate_step = Mock(side_effect=lambda p, ps, cs, d=0: Mock(score=0.5))
        
        cached = CachedPRMEvaluator(mock_evaluator, use_persistent=False)
        
        # Different problems
        cached.evaluate_step("Problem A", ["Step 1"], "Step 2")
        cached.evaluate_step("Problem B", ["Step 1"], "Step 2")
        
        # Should have called underlying evaluator twice
        assert mock_evaluator.evaluate_step.call_count == 2
        
        stats = cached.get_cache_stats()
        assert stats["misses"] == 2
    
    def test_hash_key_generation(self):
        """Test that cache keys are generated correctly."""
        cached = CachedPRMEvaluator(Mock(), use_persistent=False)
        
        key1 = cached._hash_key("Problem A", ("step1",), "step2", depth=0)
        key2 = cached._hash_key("Problem A", ("step1",), "step2", depth=0)
        key3 = cached._hash_key("Problem B", ("step1",), "step2", depth=0)
        
        # Same args should produce same key
        assert key1 == key2
        
        # Different args should produce different keys
        assert key1 != key3
    
    def test_cache_stats_tracking(self):
        """Test that cache statistics are tracked."""
        mock_evaluator = Mock()
        mock_evaluator.evaluate_step = Mock(return_value=Mock(score=0.5))
        
        cached = CachedPRMEvaluator(mock_evaluator, use_persistent=False)
        
        # Initial stats
        stats = cached.get_cache_stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["hit_rate"] == 0
        
        # One miss
        cached.evaluate_step("Problem", [], "Step")
        stats = cached.get_cache_stats()
        assert stats["misses"] == 1
        
        # One hit
        cached.evaluate_step("Problem", [], "Step")
        stats = cached.get_cache_stats()
        assert stats["hits"] == 1
        assert stats["hit_rate"] == pytest.approx(0.5)


class TestActionExecutorCaching:
    """Tests for ActionExecutor caching integration."""
    
    def test_action_executor_uses_cache(self):
        """Test that ActionExecutor can use cached evaluator."""
        from src.rl_controller.actions import ActionExecutor, ActionConfig
        from src.generator.mock_client import MockNVIDIANIMClient
        
        mock_evaluator = Mock()
        mock_evaluator.evaluate_step = Mock(return_value=Mock(score=0.7))
        
        generator = MockNVIDIANIMClient(api_key="mock")
        
        executor = ActionExecutor(
            generator=generator,
            evaluator=mock_evaluator,
            config=ActionConfig(),
            use_cache=True,
        )
        
        # Check that evaluator is wrapped
        assert hasattr(executor.evaluator, 'get_cache_stats')
    
    def test_action_executor_without_cache(self):
        """Test ActionExecutor without caching."""
        from src.rl_controller.actions import ActionExecutor, ActionConfig
        from src.generator.mock_client import MockNVIDIANIMClient
        
        mock_evaluator = Mock()
        generator = MockNVIDIANIMClient(api_key="mock")
        
        executor = ActionExecutor(
            generator=generator,
            evaluator=mock_evaluator,
            config=ActionConfig(),
            use_cache=False,
        )
        
        # Evaluator should not be wrapped
        assert executor.evaluator == mock_evaluator
    
    def test_get_stats_includes_cache_stats(self):
        """Test that get_stats includes cache statistics."""
        from src.rl_controller.actions import ActionExecutor, ActionConfig
        from src.generator.mock_client import MockNVIDIANIMClient
        
        mock_evaluator = Mock()
        mock_evaluator.evaluate_step = Mock(return_value=Mock(score=0.7))
        
        generator = MockNVIDIANIMClient(api_key="mock")
        
        executor = ActionExecutor(
            generator=generator,
            evaluator=mock_evaluator,
            config=ActionConfig(),
            use_cache=True,
        )
        
        stats = executor.get_stats()
        
        # Should have cache_stats if wrapped
        if hasattr(executor.evaluator, 'get_cache_stats'):
            assert "cache_stats" in stats


class TestLRUCachePerformance:
    """Tests for LRU cache performance characteristics."""
    
    def test_cache_eviction(self):
        """Test that cache evicts old entries when full."""
        cache = LRUCache(capacity=3, ttl_seconds=3600)
        
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        cache.put("key3", "value3")
        
        # Cache is full
        stats = cache.get_stats()
        assert stats["size"] == 3
        
        # Add one more - should evict oldest
        cache.put("key4", "value4")
        
        # key1 should be evicted
        assert cache.get("key1") is None
        assert cache.get("key2") == "value2"
        assert cache.get("key3") == "value3"
        assert cache.get("key4") == "value4"
    
    def test_cache_lru_order(self):
        """Test that LRU order is maintained correctly."""
        cache = LRUCache(capacity=3, ttl_seconds=3600)
        
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        cache.put("key3", "value3")
        
        # Access key1 - makes it most recently used
        cache.get("key1")
        
        # Add new item - should evict key2 (now oldest)
        cache.put("key4", "value4")
        
        assert cache.get("key1") == "value1"  # Still there
        assert cache.get("key2") is None      # Evicted
        assert cache.get("key3") == "value3"
        assert cache.get("key4") == "value4"
    
    def test_cache_ttl_expiration(self):
        """Test that cache respects TTL."""
        import time
        
        cache = LRUCache(capacity=10, ttl_seconds=1)
        
        cache.put("key1", "value1")
        
        # Immediately available
        assert cache.get("key1") == "value1"
        
        # Wait for TTL
        time.sleep(1.1)
        
        # Should be expired
        assert cache.get("key1") is None
    
    def test_cache_hit_rate_calculation(self):
        """Test hit rate calculation."""
        cache = LRUCache(capacity=10, ttl_seconds=3600)
        
        cache.put("key1", "value1")
        
        # 1 miss (key1 put), 1 hit (key1 get)
        cache.get("key1")
        
        # 1 miss (key2 not found)
        cache.get("key2")
        
        stats = cache.get_stats()
        # hits=1, misses=2, hit_rate=0.33
        assert stats["hits"] == 1
        assert stats["misses"] == 2
        assert stats["hit_rate"] == pytest.approx(1/3, rel=0.01)


class TestPersistentCache:
    """Tests for persistent SQLite cache."""
    
    def test_persistent_cache_basic(self, tmp_path):
        """Test basic persistent cache operations."""
        cache_path = str(tmp_path / "test_cache.db")
        cache = PersistentCache(db_path=cache_path, ttl_hours=24)
        
        # Put and get
        cache.put("key1", {"data": "value1"})
        
        result = cache.get("key1")
        assert result == {"data": "value1"}
        
        stats = cache.get_stats()
        assert stats["hits"] == 1
    
    def test_persistent_cache_persistence(self, tmp_path):
        """Test that cache persists across instances."""
        cache_path = str(tmp_path / "test_cache.db")
        
        # Create and populate
        cache1 = PersistentCache(db_path=cache_path, ttl_hours=24)
        cache1.put("key1", "value1")
        
        # Create new instance
        cache2 = PersistentCache(db_path=cache_path, ttl_hours=24)
        
        # Should have data
        result = cache2.get("key1")
        assert result == "value1"
    
    def test_persistent_cache_cleanup(self, tmp_path):
        """Test expired entry cleanup."""
        import time
        
        cache_path = str(tmp_path / "test_cache.db")
        cache = PersistentCache(db_path=cache_path, ttl_hours=0.00001)  # Very short TTL
        
        cache.put("key1", "value1")
        
        # Wait for expiration
        time.sleep(0.1)
        
        cache.cleanup_expired()
        
        # Should be cleaned up
        result = cache.get("key1")
        assert result is None
