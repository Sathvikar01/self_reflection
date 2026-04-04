"""LRU Cache for API responses and reasoning step evaluations."""

import hashlib
import json
import time
from typing import Any, Optional, Dict, Tuple
from collections import OrderedDict
from threading import Lock
import sqlite3
from pathlib import Path
from loguru import logger


class LRUCache:
    """Thread-safe LRU cache with TTL support."""

    def __init__(self, capacity: int = 10000, ttl_seconds: int = 86400):
        """Initialize LRU cache.

        Args:
            capacity: Maximum number of items
            ttl_seconds: Time-to-live in seconds (default: 24 hours)
        """
        self.capacity = capacity
        self.ttl_seconds = ttl_seconds
        self.cache: OrderedDict[str, Tuple[Any, float]] = OrderedDict()
        self.lock = Lock()
        
        # Statistics
        self.hits = 0
        self.misses = 0

        logger.info(f"LRUCache initialized (capacity={capacity}, ttl={ttl_seconds}s)")

    def _generate_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments.

        Returns:
            MD5 hash of serialized arguments
        """
        key_data = {"args": args, "kwargs": kwargs}
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()

    def get(self, key: str) -> Optional[Any]:
        """Get item from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found/expired
        """
        with self.lock:
            if key not in self.cache:
                self.misses += 1
                return None

            value, timestamp = self.cache[key]
            
            # Check TTL
            if time.time() - timestamp > self.ttl_seconds:
                del self.cache[key]
                self.misses += 1
                return None

            # Move to end (most recently used)
            self.cache.move_to_end(key)
            self.hits += 1
            return value

    def put(self, key: str, value: Any):
        """Put item in cache.

        Args:
            key: Cache key
            value: Value to cache
        """
        with self.lock:
            if key in self.cache:
                # Update existing
                self.cache.move_to_end(key)
            else:
                # Add new
                if len(self.cache) >= self.capacity:
                    # Evict oldest
                    self.cache.popitem(last=False)
            
            self.cache[key] = (value, time.time())

    def get_or_compute(self, key: str, compute_fn) -> Any:
        """Get from cache or compute if missing.

        Args:
            key: Cache key
            compute_fn: Function to compute value if not cached

        Returns:
            Cached or computed value
        """
        cached = self.get(key)
        if cached is not None:
            return cached

        # Compute and cache
        value = compute_fn()
        self.put(key, value)
        return value

    def clear(self):
        """Clear cache."""
        with self.lock:
            self.cache.clear()
            logger.info("LRU cache cleared")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Statistics dictionary
        """
        with self.lock:
            total = self.hits + self.misses
            hit_rate = self.hits / total if total > 0 else 0
            
            return {
                "size": len(self.cache),
                "capacity": self.capacity,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": hit_rate,
                "total_requests": total
            }


class PersistentCache:
    """SQLite-based persistent cache for API responses."""

    def __init__(self, db_path: str = "cache/api_cache.db", ttl_hours: int = 168):
        """Initialize persistent cache.

        Args:
            db_path: Path to SQLite database
            ttl_hours: Time-to-live in hours (default: 1 week)
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.ttl_seconds = ttl_hours * 3600
        
        # Initialize database
        self._init_db()
        
        # Statistics
        self.hits = 0
        self.misses = 0

        logger.info(f"PersistentCache initialized (db={db_path})")

    def _init_db(self):
        """Initialize SQLite database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cache (
                key TEXT PRIMARY KEY,
                value TEXT,
                timestamp REAL,
                metadata TEXT
            )
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp ON cache(timestamp)
        """)
        
        conn.commit()
        conn.close()

    def _hash_key(self, *args, **kwargs) -> str:
        """Generate hash key."""
        key_data = {"args": args, "kwargs": kwargs}
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()

    def get(self, key: str) -> Optional[Any]:
        """Get from persistent cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT value, timestamp FROM cache WHERE key = ?",
            (key,)
        )
        
        result = cursor.fetchone()
        conn.close()
        
        if result is None:
            self.misses += 1
            return None
        
        value_json, timestamp = result
        
        # Check TTL
        if time.time() - timestamp > self.ttl_seconds:
            self._delete(key)
            self.misses += 1
            return None
        
        self.hits += 1
        return json.loads(value_json)

    def put(self, key: str, value: Any, metadata: Optional[Dict] = None):
        """Put into persistent cache.

        Args:
            key: Cache key
            value: Value to cache
            metadata: Optional metadata
        """
        value_json = json.dumps(value)
        metadata_json = json.dumps(metadata or {})
        timestamp = time.time()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            """INSERT OR REPLACE INTO cache (key, value, timestamp, metadata)
               VALUES (?, ?, ?, ?)""",
            (key, value_json, timestamp, metadata_json)
        )
        
        conn.commit()
        conn.close()

    def _delete(self, key: str):
        """Delete from cache."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM cache WHERE key = ?", (key,))
        conn.commit()
        conn.close()

    def cleanup_expired(self):
        """Remove expired entries."""
        cutoff_time = time.time() - self.ttl_seconds
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM cache WHERE timestamp < ?", (cutoff_time,))
        deleted = cursor.rowcount
        conn.commit()
        conn.close()
        
        if deleted > 0:
            logger.info(f"Cleaned up {deleted} expired cache entries")

    def clear(self):
        """Clear all cached entries."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM cache")
        conn.commit()
        conn.close()
        logger.info("Persistent cache cleared")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM cache")
        size = cursor.fetchone()[0]
        
        conn.close()
        
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        
        return {
            "size": size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "total_requests": total
        }


class CachedPRMEvaluator:
    """PRM evaluator with LRU caching."""

    def __init__(self, prm_evaluator, use_persistent: bool = False):
        """Initialize cached evaluator.

        Args:
            prm_evaluator: PRM evaluator instance
            use_persistent: Use persistent cache instead of in-memory
        """
        self.prm_evaluator = prm_evaluator
        self._use_persistent = use_persistent

        if use_persistent:
            self.cache = PersistentCache()
        else:
            self.cache = LRUCache(capacity=10000, ttl_seconds=3600)

        logger.info("CachedPRMEvaluator initialized")

    def _hash_key(self, problem: str, previous_steps: tuple, current_step: str, depth: int = 0) -> str:
        """Generate cache key."""
        key_data = {
            "problem": problem,
            "previous_steps": previous_steps,
            "current_step": current_step,
            "depth": depth
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()

    def evaluate_step(
        self,
        problem: str,
        previous_steps: list,
        current_step: str,
        depth: int = 0
    ):
        """Evaluate step with caching.

        Args:
            problem: Problem statement
            previous_steps: Previous reasoning steps
            current_step: Current step to evaluate
            depth: Optional depth parameter for ImprovedPRM

        Returns:
            Evaluation result (same as underlying evaluator)
        """
        cache_key = self._hash_key(problem, tuple(previous_steps), current_step, depth)

        def compute():
            if hasattr(self.prm_evaluator, 'evaluate_step'):
                sig = self.prm_evaluator.evaluate_step.__code__.co_varnames
                if 'depth' in sig:
                    return self.prm_evaluator.evaluate_step(problem, previous_steps, current_step, depth)
                return self.prm_evaluator.evaluate_step(problem, previous_steps, current_step)
            return self.prm_evaluator.evaluate_step(problem, previous_steps, current_step)

        if isinstance(self.cache, LRUCache):
            return self.cache.get_or_compute(cache_key, compute)
        else:
            cached = self.cache.get(cache_key)
            if cached is not None:
                if isinstance(cached, dict) and hasattr(self.prm_evaluator, 'evaluate_step'):
                    from types import SimpleNamespace
                    return SimpleNamespace(**cached)
                return cached

            value = compute()
            if hasattr(value, '__dict__'):
                self.cache.put(cache_key, value.__dict__)
            else:
                self.cache.put(cache_key, value)
            return value

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self.cache.get_stats()
        """Get cache statistics."""
        return self.cache.get_stats()
