"""
High-Performance Caching System for Code Generation Results
"""
import asyncio
import hashlib
import json
import logging
import pickle
import time
from typing import Any, Dict, Optional, Union, List
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import OrderedDict
import diskcache as dc

from config import settings


logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Represents a cached item with metadata"""
    key: str
    value: Any
    created_at: float
    accessed_at: float
    access_count: int
    size_bytes: int
    tags: List[str] = None
    
    def is_expired(self, ttl: int) -> bool:
        """Check if cache entry has expired"""
        return time.time() - self.created_at > ttl
    
    def touch(self):
        """Update access metadata"""
        self.accessed_at = time.time()
        self.access_count += 1


class MemoryCache:
    """High-performance in-memory LRU cache"""
    
    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        self.max_size = max_size
        self.ttl = ttl
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = asyncio.Lock()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get item from cache"""
        async with self._lock:
            if key not in self._cache:
                return None
            
            entry = self._cache[key]
            
            # Check expiration
            if entry.is_expired(self.ttl):
                del self._cache[key]
                return None
            
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            entry.touch()
            
            return entry.value
    
    async def set(self, key: str, value: Any, tags: List[str] = None) -> bool:
        """Set item in cache"""
        async with self._lock:
            # Calculate size
            try:
                size = len(pickle.dumps(value))
            except:
                size = 1024  # Estimate for non-serializable objects
            
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=time.time(),
                accessed_at=time.time(),
                access_count=1,
                size_bytes=size,
                tags=tags or []
            )
            
            # Add to cache
            self._cache[key] = entry
            self._cache.move_to_end(key)
            
            # Evict if over capacity
            while len(self._cache) > self.max_size:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
            
            return True
    
    async def delete(self, key: str) -> bool:
        """Delete item from cache"""
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False
    
    async def clear(self):
        """Clear all cache entries"""
        async with self._lock:
            self._cache.clear()
    
    async def stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        async with self._lock:
            total_size = sum(entry.size_bytes for entry in self._cache.values())
            return {
                "entries": len(self._cache),
                "max_size": self.max_size,
                "total_size_bytes": total_size,
                "hit_rate": 0,  # Would need hit/miss tracking
                "avg_access_count": sum(e.access_count for e in self._cache.values()) / max(len(self._cache), 1)
            }


class DiskCache:
    """Persistent disk-based cache using diskcache"""
    
    def __init__(self, cache_dir: Path, max_size: int = 10 * 1024 * 1024 * 1024):  # 10GB
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache = dc.Cache(str(cache_dir), size_limit=max_size)
    
    async def get(self, key: str) -> Optional[Any]:
        """Get item from disk cache"""
        try:
            return await asyncio.to_thread(self.cache.get, key)
        except Exception as e:
            logger.error(f"Disk cache get error: {e}")
            return None
    
    async def set(self, key: str, value: Any, ttl: int = 3600, tags: List[str] = None) -> bool:
        """Set item in disk cache"""
        try:
            return await asyncio.to_thread(self.cache.set, key, value, expire=ttl, tag=tags[0] if tags else None)
        except Exception as e:
            logger.error(f"Disk cache set error: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete item from disk cache"""
        try:
            return await asyncio.to_thread(self.cache.delete, key)
        except Exception as e:
            logger.error(f"Disk cache delete error: {e}")
            return False
    
    async def clear(self):
        """Clear all cache entries"""
        try:
            await asyncio.to_thread(self.cache.clear)
        except Exception as e:
            logger.error(f"Disk cache clear error: {e}")
    
    async def stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        try:
            stats = await asyncio.to_thread(self.cache.stats)
            return {
                "hits": stats.get("cache_hits", 0),
                "misses": stats.get("cache_misses", 0),
                "entries": len(self.cache),
                "size_bytes": stats.get("size", 0)
            }
        except Exception as e:
            logger.error(f"Disk cache stats error: {e}")
            return {}


class RedisCache:
    """Redis-based distributed cache"""
    
    def __init__(self, redis_url: str):
        try:
            import redis.asyncio as redis
            self.redis = redis.from_url(redis_url)
        except ImportError:
            logger.error("Redis not available. Install with: pip install redis")
            self.redis = None
    
    async def get(self, key: str) -> Optional[Any]:
        """Get item from Redis cache"""
        if not self.redis:
            return None
        
        try:
            data = await self.redis.get(key)
            if data:
                return pickle.loads(data)
            return None
        except Exception as e:
            logger.error(f"Redis cache get error: {e}")
            return None
    
    async def set(self, key: str, value: Any, ttl: int = 3600, tags: List[str] = None) -> bool:
        """Set item in Redis cache"""
        if not self.redis:
            return False
        
        try:
            data = pickle.dumps(value)
            await self.redis.set(key, data, ex=ttl)
            
            # Store tags if provided
            if tags:
                for tag in tags:
                    await self.redis.sadd(f"tag:{tag}", key)
            
            return True
        except Exception as e:
            logger.error(f"Redis cache set error: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete item from Redis cache"""
        if not self.redis:
            return False
        
        try:
            result = await self.redis.delete(key)
            return result > 0
        except Exception as e:
            logger.error(f"Redis cache delete error: {e}")
            return False
    
    async def clear(self):
        """Clear all cache entries"""
        if not self.redis:
            return
        
        try:
            await self.redis.flushdb()
        except Exception as e:
            logger.error(f"Redis cache clear error: {e}")
    
    async def stats(self) -> Dict[str, Any]:
        """Get Redis cache statistics"""
        if not self.redis:
            return {}
        
        try:
            info = await self.redis.info("memory")
            return {
                "memory_used": info.get("used_memory", 0),
                "memory_peak": info.get("used_memory_peak", 0),
                "keys": await self.redis.dbsize()
            }
        except Exception as e:
            logger.error(f"Redis cache stats error: {e}")
            return {}


class SmartCache:
    """Intelligent multi-tier cache system"""
    
    def __init__(self):
        self.config = settings.cache
        
        # Initialize cache backends
        self.memory_cache = MemoryCache(
            max_size=self.config.max_size,
            ttl=self.config.ttl
        )
        
        self.disk_cache = DiskCache(
            cache_dir=Path("cache") / "disk",
            max_size=10 * 1024 * 1024 * 1024  # 10GB
        )
        
        self.redis_cache = None
        if self.config.redis_url:
            self.redis_cache = RedisCache(self.config.redis_url)
        
        # Statistics
        self.stats = {
            "hits": 0,
            "misses": 0,
            "memory_hits": 0,
            "disk_hits": 0,
            "redis_hits": 0
        }
    
    def _generate_key(self, data: Union[str, Dict, List]) -> str:
        """Generate cache key from input data"""
        if isinstance(data, str):
            content = data
        else:
            content = json.dumps(data, sort_keys=True)
        
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    async def get(self, key: Union[str, Dict, List], use_tiers: List[str] = None) -> Optional[Any]:
        """Get item from cache with multi-tier lookup"""
        if not self.config.enabled:
            return None
        
        cache_key = self._generate_key(key) if not isinstance(key, str) else key
        use_tiers = use_tiers or ["memory", "disk", "redis"]
        
        # Try memory cache first
        if "memory" in use_tiers:
            value = await self.memory_cache.get(cache_key)
            if value is not None:
                self.stats["hits"] += 1
                self.stats["memory_hits"] += 1
                return value
        
        # Try Redis cache second
        if "redis" in use_tiers and self.redis_cache:
            value = await self.redis_cache.get(cache_key)
            if value is not None:
                # Promote to memory cache
                await self.memory_cache.set(cache_key, value)
                self.stats["hits"] += 1
                self.stats["redis_hits"] += 1
                return value
        
        # Try disk cache last
        if "disk" in use_tiers:
            value = await self.disk_cache.get(cache_key)
            if value is not None:
                # Promote to higher tiers
                await self.memory_cache.set(cache_key, value)
                if self.redis_cache:
                    await self.redis_cache.set(cache_key, value, self.config.ttl)
                
                self.stats["hits"] += 1
                self.stats["disk_hits"] += 1
                return value
        
        self.stats["misses"] += 1
        return None
    
    async def set(self, key: Union[str, Dict, List], value: Any, ttl: int = None, tags: List[str] = None) -> bool:
        """Set item in cache across all tiers"""
        if not self.config.enabled:
            return False
        
        cache_key = self._generate_key(key) if not isinstance(key, str) else key
        ttl = ttl or self.config.ttl
        
        success = True
        
        # Store in memory cache
        success &= await self.memory_cache.set(cache_key, value, tags)
        
        # Store in Redis cache
        if self.redis_cache:
            success &= await self.redis_cache.set(cache_key, value, ttl, tags)
        
        # Store in disk cache
        success &= await self.disk_cache.set(cache_key, value, ttl, tags)
        
        return success
    
    async def delete(self, key: Union[str, Dict, List]) -> bool:
        """Delete item from all cache tiers"""
        if not self.config.enabled:
            return False
        
        cache_key = self._generate_key(key) if not isinstance(key, str) else key
        
        success = True
        success &= await self.memory_cache.delete(cache_key)
        success &= await self.disk_cache.delete(cache_key)
        
        if self.redis_cache:
            success &= await self.redis_cache.delete(cache_key)
        
        return success
    
    async def clear(self):
        """Clear all cache tiers"""
        await self.memory_cache.clear()
        await self.disk_cache.clear()
        
        if self.redis_cache:
            await self.redis_cache.clear()
        
        # Reset statistics
        self.stats = {key: 0 for key in self.stats}
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        memory_stats = await self.memory_cache.stats()
        disk_stats = await self.disk_cache.stats()
        
        redis_stats = {}
        if self.redis_cache:
            redis_stats = await self.redis_cache.stats()
        
        total_hits = self.stats["hits"]
        total_requests = total_hits + self.stats["misses"]
        hit_rate = (total_hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            "hit_rate": round(hit_rate, 2),
            "total_hits": total_hits,
            "total_misses": self.stats["misses"],
            "memory_hits": self.stats["memory_hits"],
            "disk_hits": self.stats["disk_hits"],
            "redis_hits": self.stats["redis_hits"],
            "memory_cache": memory_stats,
            "disk_cache": disk_stats,
            "redis_cache": redis_stats
        }


# Global cache instance
cache = SmartCache()
