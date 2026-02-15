"""Multi-Layer Caching System - Ready to use!

Usage:
    from app.core.cache import cache, cached
    
    # Initialize in main.py
    await cache.initialize(redis_url)
    
    # Use decorator
    @cached(prefix="my_function", ttl=3600)
    async def my_expensive_function(arg1, arg2):
        # This will be cached for 1 hour
        ...
"""

from functools import wraps
import hashlib
import json
import pickle
from typing import Any, Callable, Optional
import redis.asyncio as aioredis
import structlog

logger = structlog.get_logger(__name__)


class CacheLayer:
    """Multi-layer caching: Memory (LRU) → Redis → Database."""
    
    def __init__(self):
        self.memory_cache = {}  # Simple in-memory cache
        self.max_memory_items = 1000
        self.redis_client = None
        self.stats = {"hits": 0, "misses": 0, "memory_hits": 0, "redis_hits": 0}
    
    async def initialize(self, redis_url: str):
        """Initialize Redis connection."""
        try:
            self.redis_client = aioredis.from_url(
                redis_url,
                decode_responses=False,
                max_connections=50,
            )
            await self.redis_client.ping()
            logger.info("cache_initialized", redis_url=redis_url)
        except Exception as e:
            logger.error("cache_init_failed", error=str(e))
            self.redis_client = None
    
    def _generate_key(self, prefix: str, *args, **kwargs) -> str:
        """Generate cache key from function args."""
        key_data = json.dumps({"args": args, "kwargs": kwargs}, sort_keys=True)
        key_hash = hashlib.sha256(key_data.encode()).hexdigest()[:16]
        return f"{prefix}:{key_hash}"
    
    async def get(self, key: str) -> Optional[Any]:
        """Get from memory → Redis → None."""
        # Layer 1: Memory cache
        if key in self.memory_cache:
            self.stats["hits"] += 1
            self.stats["memory_hits"] += 1
            logger.debug("cache_hit_memory", key=key)
            return self.memory_cache[key]
        
        # Layer 2: Redis cache
        if self.redis_client:
            try:
                data = await self.redis_client.get(key)
                if data:
                    obj = pickle.loads(data)
                    self.memory_cache[key] = obj
                    self._evict_if_needed()
                    self.stats["hits"] += 1
                    self.stats["redis_hits"] += 1
                    logger.debug("cache_hit_redis", key=key)
                    return obj
            except Exception as e:
                logger.warning("redis_get_failed", key=key, error=str(e))
        
        self.stats["misses"] += 1
        logger.debug("cache_miss", key=key)
        return None
    
    async def set(self, key: str, value: Any, ttl: int = 3600):
        """Set in both memory and Redis."""
        # Layer 1: Memory
        self.memory_cache[key] = value
        self._evict_if_needed()
        
        # Layer 2: Redis
        if self.redis_client:
            try:
                await self.redis_client.setex(key, ttl, pickle.dumps(value))
                logger.debug("cache_set", key=key, ttl=ttl)
            except Exception as e:
                logger.warning("redis_set_failed", key=key, error=str(e))
    
    def _evict_if_needed(self):
        """LRU eviction when memory cache is full."""
        if len(self.memory_cache) > self.max_memory_items:
            # Remove oldest 20% of items
            items_to_remove = len(self.memory_cache) // 5
            for key in list(self.memory_cache.keys())[:items_to_remove]:
                del self.memory_cache[key]
            logger.info("cache_evicted", items_removed=items_to_remove)
    
    async def invalidate(self, pattern: str):
        """Invalidate cache by pattern."""
        # Clear memory cache
        keys_to_delete = [k for k in self.memory_cache.keys() if k.startswith(pattern)]
        for key in keys_to_delete:
            del self.memory_cache[key]
        
        # Clear Redis cache
        if self.redis_client:
            try:
                count = 0
                async for key in self.redis_client.scan_iter(match=f"{pattern}*"):
                    await self.redis_client.delete(key)
                    count += 1
                logger.info("cache_invalidated", pattern=pattern, keys_deleted=count)
            except Exception as e:
                logger.warning("redis_invalidate_failed", pattern=pattern, error=str(e))
    
    def get_stats(self) -> dict:
        """Get cache statistics."""
        total = self.stats["hits"] + self.stats["misses"]
        hit_rate = (self.stats["hits"] / total * 100) if total > 0 else 0
        return {
            **self.stats,
            "hit_rate": round(hit_rate, 2),
            "memory_size": len(self.memory_cache),
        }
    
    async def close(self):
        """Close Redis connection."""
        if self.redis_client:
            await self.redis_client.close()


# Global cache instance
cache = CacheLayer()


def cached(prefix: str, ttl: int = 3600):
    """Decorator for caching function results.
    
    Args:
        prefix: Cache key prefix (e.g., 'trends', 'user_profile')
        ttl: Time to live in seconds (default: 1 hour)
    
    Example:
        @cached(prefix="trends", ttl=3600)
        async def analyze_trends(genre: str) -> dict:
            # Expensive operation
            return {"trending": [...]}
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate cache key
            key = cache._generate_key(prefix, *args, **kwargs)
            
            # Try cache first
            result = await cache.get(key)
            if result is not None:
                return result
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Cache result
            if result is not None:
                await cache.set(key, result, ttl)
            
            return result
        return wrapper
    return decorator
