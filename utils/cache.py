"""
Cache manager for ML service using Redis
"""

import redis.asyncio as redis
import json
import pickle
import hashlib
from typing import Any, Optional, Union, Dict, List
from datetime import datetime, timedelta
import asyncio
from utils.logger import log_cache_operation, ml_logger

class CacheManager:
    """
    Async Redis cache manager for ML operations
    """
    
    def __init__(self, redis_url: str, default_ttl: int = 300):
        """
        Initialize cache manager
        
        Args:
            redis_url: Redis connection URL
            default_ttl: Default TTL in seconds
        """
        self.redis_url = redis_url
        self.default_ttl = default_ttl
        self.client: Optional[redis.Redis] = None
        self._connected = False
    
    async def connect(self):
        """Connect to Redis"""
        try:
            self.client = redis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=False,  # We'll handle encoding ourselves
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
                max_connections=20
            )
            
            # Test connection
            await self.client.ping()
            self._connected = True
            ml_logger.info("Cache manager connected to Redis")
            
        except Exception as e:
            ml_logger.error(f"Failed to connect to Redis: {e}")
            self._connected = False
            raise
    
    async def close(self):
        """Close Redis connection"""
        if self.client:
            await self.client.close()
            self._connected = False
            ml_logger.info("Cache manager disconnected from Redis")
    
    async def is_connected(self) -> bool:
        """Check if connected to Redis"""
        if not self._connected or not self.client:
            return False
        
        try:
            await self.client.ping()
            return True
        except:
            self._connected = False
            return False
    
    def _generate_key(self, prefix: str, *args, **kwargs) -> str:
        """
        Generate cache key from prefix and arguments
        
        Args:
            prefix: Key prefix
            *args: Positional arguments
            **kwargs: Keyword arguments
        
        Returns:
            Generated cache key
        """
        # Create a string from all arguments
        key_parts = [str(arg) for arg in args]
        key_parts.extend([f"{k}:{v}" for k, v in sorted(kwargs.items())])
        key_string = ":".join(key_parts)
        
        # Hash long keys to avoid Redis key length limits
        if len(key_string) > 100:
            key_hash = hashlib.md5(key_string.encode()).hexdigest()
            return f"{prefix}:hash:{key_hash}"
        else:
            return f"{prefix}:{key_string}" if key_string else prefix
    
    async def get(
        self,
        key: str,
        deserialize: bool = True,
        use_pickle: bool = False
    ) -> Optional[Any]:
        """
        Get value from cache
        
        Args:
            key: Cache key
            deserialize: Whether to deserialize JSON
            use_pickle: Whether to use pickle for deserialization
        
        Returns:
            Cached value or None
        """
        if not await self.is_connected():
            return None
        
        try:
            value = await self.client.get(key)
            
            if value is None:
                log_cache_operation("get", key, hit=False)
                return None
            
            log_cache_operation("get", key, hit=True, size=len(value))
            
            if not deserialize:
                return value
            
            if use_pickle:
                return pickle.loads(value)
            else:
                return json.loads(value.decode('utf-8'))
                
        except Exception as e:
            ml_logger.error(f"Cache get error for key {key}: {e}")
            return None
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        serialize: bool = True,
        use_pickle: bool = False
    ) -> bool:
        """
        Set value in cache
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
            serialize: Whether to serialize as JSON
            use_pickle: Whether to use pickle for serialization
        
        Returns:
            True if successful, False otherwise
        """
        if not await self.is_connected():
            return False
        
        try:
            if serialize:
                if use_pickle:
                    serialized_value = pickle.dumps(value)
                else:
                    serialized_value = json.dumps(value, ensure_ascii=False).encode('utf-8')
            else:
                serialized_value = value
            
            ttl = ttl or self.default_ttl
            result = await self.client.setex(key, ttl, serialized_value)
            
            log_cache_operation("set", key, size=len(serialized_value))
            return result
            
        except Exception as e:
            ml_logger.error(f"Cache set error for key {key}: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """
        Delete key from cache
        
        Args:
            key: Cache key to delete
        
        Returns:
            True if key was deleted, False otherwise
        """
        if not await self.is_connected():
            return False
        
        try:
            result = await self.client.delete(key)
            log_cache_operation("delete", key)
            return result > 0
            
        except Exception as e:
            ml_logger.error(f"Cache delete error for key {key}: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """
        Check if key exists in cache
        
        Args:
            key: Cache key to check
        
        Returns:
            True if key exists, False otherwise
        """
        if not await self.is_connected():
            return False
        
        try:
            result = await self.client.exists(key)
            return result > 0
            
        except Exception as e:
            ml_logger.error(f"Cache exists error for key {key}: {e}")
            return False
    
    async def get_many(self, keys: List[str], deserialize: bool = True) -> Dict[str, Any]:
        """
        Get multiple values from cache
        
        Args:
            keys: List of cache keys
            deserialize: Whether to deserialize JSON
        
        Returns:
            Dictionary of key-value pairs
        """
        if not await self.is_connected() or not keys:
            return {}
        
        try:
            values = await self.client.mget(keys)
            result = {}
            
            for key, value in zip(keys, values):
                if value is not None:
                    try:
                        if deserialize:
                            result[key] = json.loads(value.decode('utf-8'))
                        else:
                            result[key] = value
                        log_cache_operation("mget", key, hit=True)
                    except Exception as e:
                        ml_logger.error(f"Error deserializing cached value for {key}: {e}")
                        log_cache_operation("mget", key, hit=False)
                else:
                    log_cache_operation("mget", key, hit=False)
            
            return result
            
        except Exception as e:
            ml_logger.error(f"Cache mget error: {e}")
            return {}
    
    async def set_many(
        self,
        data: Dict[str, Any],
        ttl: Optional[int] = None,
        serialize: bool = True
    ) -> bool:
        """
        Set multiple values in cache
        
        Args:
            data: Dictionary of key-value pairs
            ttl: Time to live in seconds
            serialize: Whether to serialize as JSON
        
        Returns:
            True if successful, False otherwise
        """
        if not await self.is_connected() or not data:
            return False
        
        try:
            pipe = self.client.pipeline()
            ttl = ttl or self.default_ttl
            
            for key, value in data.items():
                if serialize:
                    serialized_value = json.dumps(value, ensure_ascii=False).encode('utf-8')
                else:
                    serialized_value = value
                
                pipe.setex(key, ttl, serialized_value)
                log_cache_operation("mset", key)
            
            await pipe.execute()
            return True
            
        except Exception as e:
            ml_logger.error(f"Cache mset error: {e}")
            return False
    
    async def increment(self, key: str, amount: int = 1, ttl: Optional[int] = None) -> Optional[int]:
        """
        Increment a counter in cache
        
        Args:
            key: Cache key
            amount: Amount to increment by
            ttl: Time to live for new keys
        
        Returns:
            New value or None if error
        """
        if not await self.is_connected():
            return None
        
        try:
            pipe = self.client.pipeline()
            pipe.incrby(key, amount)
            
            if ttl:
                pipe.expire(key, ttl)
            
            results = await pipe.execute()
            log_cache_operation("incr", key)
            return results[0]
            
        except Exception as e:
            ml_logger.error(f"Cache increment error for key {key}: {e}")
            return None
    
    async def get_stats(self) -> Optional[Dict[str, Any]]:
        """
        Get cache statistics
        
        Returns:
            Cache statistics dictionary
        """
        if not await self.is_connected():
            return None
        
        try:
            info = await self.client.info("memory")
            keyspace = await self.client.info("keyspace")
            
            return {
                "memory_usage_bytes": info.get("used_memory", 0),
                "memory_usage_human": info.get("used_memory_human", "0B"),
                "total_connections": info.get("connected_clients", 0),
                "keyspace": keyspace,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            ml_logger.error(f"Error getting cache stats: {e}")
            return None
    
    async def clear_pattern(self, pattern: str) -> int:
        """
        Clear keys matching pattern
        
        Args:
            pattern: Redis key pattern (e.g., "ml:intent:*")
        
        Returns:
            Number of keys deleted
        """
        if not await self.is_connected():
            return 0
        
        try:
            keys = []
            async for key in self.client.scan_iter(match=pattern):
                keys.append(key)
            
            if keys:
                deleted = await self.client.delete(*keys)
                ml_logger.info(f"Cleared {deleted} keys matching pattern: {pattern}")
                return deleted
            
            return 0
            
        except Exception as e:
            ml_logger.error(f"Error clearing pattern {pattern}: {e}")
            return 0

# ML-specific cache methods
class MLCacheManager(CacheManager):
    """
    Extended cache manager with ML-specific methods
    """
    
    async def cache_intent_prediction(
        self,
        text: str,
        intent: str,
        confidence: float,
        context: Optional[Dict] = None,
        ttl: int = 300
    ):
        """Cache intent prediction result"""
        key = self._generate_key("ml:intent", text, context or {})
        value = {
            "intent": intent,
            "confidence": confidence,
            "timestamp": datetime.now().isoformat()
        }
        await self.set(key, value, ttl=ttl)
    
    async def get_cached_intent(
        self,
        text: str,
        context: Optional[Dict] = None
    ) -> Optional[Dict[str, Any]]:
        """Get cached intent prediction"""
        key = self._generate_key("ml:intent", text, context or {})
        return await self.get(key)
    
    async def cache_entities(
        self,
        text: str,
        entities: Dict[str, Any],
        intent: Optional[str] = None,
        ttl: int = 300
    ):
        """Cache entity extraction result"""
        key = self._generate_key("ml:entities", text, intent or "")
        value = {
            "entities": entities,
            "timestamp": datetime.now().isoformat()
        }
        await self.set(key, value, ttl=ttl)
    
    async def get_cached_entities(
        self,
        text: str,
        intent: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Get cached entity extraction"""
        key = self._generate_key("ml:entities", text, intent or "")
        result = await self.get(key)
        return result.get("entities") if result else None
    
    async def cache_translation(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        translation: str,
        ttl: int = 3600
    ):
        """Cache translation result"""
        key = self._generate_key("ml:translation", text, source_lang, target_lang)
        value = {
            "translation": translation,
            "source_lang": source_lang,
            "target_lang": target_lang,
            "timestamp": datetime.now().isoformat()
        }
        await self.set(key, value, ttl=ttl)
    
    async def get_cached_translation(
        self,
        text: str,
        source_lang: str,
        target_lang: str
    ) -> Optional[str]:
        """Get cached translation"""
        key = self._generate_key("ml:translation", text, source_lang, target_lang)
        result = await self.get(key)
        return result.get("translation") if result else None
    
    async def cache_language_detection(
        self,
        text: str,
        language: str,
        confidence: float,
        ttl: int = 1800
    ):
        """Cache language detection result"""
        key = self._generate_key("ml:language", text)
        value = {
            "language": language,
            "confidence": confidence,
            "timestamp": datetime.now().isoformat()
        }
        await self.set(key, value, ttl=ttl)
    
    async def get_cached_language(self, text: str) -> Optional[Dict[str, Any]]:
        """Get cached language detection"""
        key = self._generate_key("ml:language", text)
        return await self.get(key)