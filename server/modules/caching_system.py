"""
Redis Caching System for Medical AI Pipeline

Implements intelligent caching for:
- Entity recognition results
- Knowledge graph queries
- Disease predictions
- Document retrieval results
- Evidence fusion outcomes
"""

import redis
import json
import hashlib
import logging
import time
import pickle
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class CacheConfig:
    """Configuration for caching system"""
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None
    default_ttl: int = 3600  # 1 hour
    max_cache_size: int = 1000000  # 1MB per key
    enable_compression: bool = True
    cache_hit_threshold: float = 0.8  # Minimum similarity for cache hit

@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    data: Any
    timestamp: float
    hit_count: int
    similarity_threshold: float
    source_component: str

class MedicalCacheManager:
    """
    Intelligent caching system for medical AI pipeline components.
    
    Provides component-specific caching with similarity-based retrieval,
    TTL management, and performance optimization features.
    """
    
    def __init__(self, config: Optional[CacheConfig] = None):
        """
        Initialize the caching system
        
        Args:
            config: Cache configuration settings
        """
        self.config = config or CacheConfig()
        self.redis_client = None
        self.cache_stats = {
            'hits': 0,
            'misses': 0,
            'stores': 0,
            'errors': 0,
            'evictions': 0
        }
        self._initialize_redis()
    
    def _initialize_redis(self):
        """Initialize Redis connection"""
        try:
            self.redis_client = redis.Redis(
                host=self.config.redis_host,
                port=self.config.redis_port,
                db=self.config.redis_db,
                password=self.config.redis_password,
                decode_responses=False,  # We'll handle encoding ourselves
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True
            )
            
            # Test connection
            self.redis_client.ping()
            logger.info(f"Connected to Redis at {self.config.redis_host}:{self.config.redis_port}")
            
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {str(e)}")
            self.redis_client = None
    
    def _generate_cache_key(self, component: str, input_data: Union[str, Dict, List]) -> str:
        """Generate a consistent cache key for input data"""
        # Convert input to string representation
        if isinstance(input_data, str):
            data_str = input_data.lower().strip()
        elif isinstance(input_data, (dict, list)):
            data_str = json.dumps(input_data, sort_keys=True)
        else:
            data_str = str(input_data)
        
        # Create hash
        hash_object = hashlib.sha256(data_str.encode())
        hash_hex = hash_object.hexdigest()[:16]  # Use first 16 characters
        
        return f"medical:{component}:{hash_hex}"
    
    def _serialize_data(self, data: Any) -> bytes:
        """Serialize data for storage"""
        try:
            if self.config.enable_compression:
                import gzip
                serialized = pickle.dumps(data)
                return gzip.compress(serialized)
            else:
                return pickle.dumps(data)
        except Exception as e:
            logger.error(f"Serialization error: {str(e)}")
            return None
    
    def _deserialize_data(self, data: bytes) -> Any:
        """Deserialize data from storage"""
        try:
            if self.config.enable_compression:
                import gzip
                decompressed = gzip.decompress(data)
                return pickle.loads(decompressed)
            else:
                return pickle.loads(data)
        except Exception as e:
            logger.error(f"Deserialization error: {str(e)}")
            return None
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text inputs for fuzzy matching"""
        from difflib import SequenceMatcher
        return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()
    
    def get_cached_result(self, component: str, input_data: Union[str, Dict, List], 
                         fuzzy_match: bool = True) -> Optional[Any]:
        """
        Retrieve cached result for component input
        
        Args:
            component: Component name (e.g., 'ned', 'rag', 'neuro_symbolic')
            input_data: Input data to look up
            fuzzy_match: Enable fuzzy matching for similar inputs
            
        Returns:
            Cached result if found, None otherwise
        """
        if not self.redis_client:
            return None
        
        try:
            # Try exact match first
            cache_key = self._generate_cache_key(component, input_data)
            cached_data = self.redis_client.get(cache_key)
            
            if cached_data:
                result = self._deserialize_data(cached_data)
                if result:
                    self.cache_stats['hits'] += 1
                    self._update_hit_count(cache_key)
                    logger.debug(f"Cache hit for {component}: {cache_key}")
                    return result
            
            # Try fuzzy matching if enabled and input is text
            if fuzzy_match and isinstance(input_data, str):
                similar_result = self._find_similar_cached_result(component, input_data)
                if similar_result:
                    self.cache_stats['hits'] += 1
                    return similar_result
            
            self.cache_stats['misses'] += 1
            return None
            
        except Exception as e:
            logger.error(f"Cache retrieval error: {str(e)}")
            self.cache_stats['errors'] += 1
            return None
    
    def _find_similar_cached_result(self, component: str, input_text: str) -> Optional[Any]:
        """Find similar cached results using fuzzy matching"""
        try:
            # Get all keys for this component
            pattern = f"medical:{component}:*"
            keys = self.redis_client.keys(pattern)
            
            best_similarity = 0
            best_result = None
            
            for key in keys[:50]:  # Limit search to avoid performance issues
                # Try to reconstruct original input (this is approximate)
                cached_data = self.redis_client.get(key)
                if cached_data:
                    # For now, we'll skip fuzzy matching and rely on exact matches
                    # This can be enhanced with additional metadata storage
                    pass
            
            return best_result if best_similarity >= self.config.cache_hit_threshold else None
            
        except Exception as e:
            logger.error(f"Fuzzy matching error: {str(e)}")
            return None
    
    def store_result(self, component: str, input_data: Union[str, Dict, List], 
                    result: Any, ttl: Optional[int] = None) -> bool:
        """
        Store result in cache
        
        Args:
            component: Component name
            input_data: Input data used as key
            result: Result to cache
            ttl: Time to live in seconds
            
        Returns:
            True if stored successfully, False otherwise
        """
        if not self.redis_client:
            return False
        
        try:
            cache_key = self._generate_cache_key(component, input_data)
            serialized_data = self._serialize_data(result)
            
            if not serialized_data:
                return False
            
            # Check size limit
            if len(serialized_data) > self.config.max_cache_size:
                logger.warning(f"Cache entry too large: {len(serialized_data)} bytes")
                return False
            
            # Store with TTL
            ttl = ttl or self.config.default_ttl
            success = self.redis_client.setex(cache_key, ttl, serialized_data)
            
            if success:
                self.cache_stats['stores'] += 1
                logger.debug(f"Cached result for {component}: {cache_key}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Cache storage error: {str(e)}")
            self.cache_stats['errors'] += 1
            return False
    
    def _update_hit_count(self, cache_key: str):
        """Update hit count for cache entry"""
        try:
            hit_key = f"{cache_key}:hits"
            self.redis_client.incr(hit_key)
            self.redis_client.expire(hit_key, self.config.default_ttl)
        except Exception as e:
            logger.debug(f"Failed to update hit count: {str(e)}")
    
    def invalidate_cache(self, component: Optional[str] = None, 
                        pattern: Optional[str] = None) -> int:
        """
        Invalidate cache entries
        
        Args:
            component: Specific component to invalidate
            pattern: Custom pattern to match keys
            
        Returns:
            Number of keys deleted
        """
        if not self.redis_client:
            return 0
        
        try:
            if pattern:
                keys = self.redis_client.keys(pattern)
            elif component:
                keys = self.redis_client.keys(f"medical:{component}:*")
            else:
                keys = self.redis_client.keys("medical:*")
            
            if keys:
                deleted = self.redis_client.delete(*keys)
                self.cache_stats['evictions'] += deleted
                logger.info(f"Invalidated {deleted} cache entries")
                return deleted
            
            return 0
            
        except Exception as e:
            logger.error(f"Cache invalidation error: {str(e)}")
            return 0
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        total_requests = self.cache_stats['hits'] + self.cache_stats['misses']
        hit_rate = self.cache_stats['hits'] / total_requests if total_requests > 0 else 0
        
        redis_info = {}
        if self.redis_client:
            try:
                redis_info = self.redis_client.info()
                redis_info = {
                    'used_memory': redis_info.get('used_memory_human', 'N/A'),
                    'connected_clients': redis_info.get('connected_clients', 0),
                    'total_commands_processed': redis_info.get('total_commands_processed', 0),
                    'keyspace_hits': redis_info.get('keyspace_hits', 0),
                    'keyspace_misses': redis_info.get('keyspace_misses', 0)
                }
            except Exception as e:
                logger.error(f"Failed to get Redis info: {str(e)}")
        
        return {
            'cache_stats': self.cache_stats,
            'hit_rate': hit_rate,
            'total_requests': total_requests,
            'redis_info': redis_info,
            'config': asdict(self.config)
        }
    
    def warm_cache(self, component: str, input_samples: List[Union[str, Dict, List]], 
                   processor_func: callable) -> int:
        """
        Warm up cache with common queries
        
        Args:
            component: Component name
            input_samples: Sample inputs to pre-process
            processor_func: Function to process inputs
            
        Returns:
            Number of entries cached
        """
        if not self.redis_client:
            return 0
        
        cached_count = 0
        
        for input_data in input_samples:
            try:
                # Check if already cached
                if self.get_cached_result(component, input_data, fuzzy_match=False):
                    continue
                
                # Process and cache
                result = processor_func(input_data)
                if result and self.store_result(component, input_data, result):
                    cached_count += 1
                    
            except Exception as e:
                logger.error(f"Cache warming error for {input_data}: {str(e)}")
        
        logger.info(f"Warmed cache with {cached_count} entries for {component}")
        return cached_count
    
    def cleanup_expired_entries(self) -> int:
        """Clean up expired cache entries and return count"""
        if not self.redis_client:
            return 0
        
        try:
            # Redis handles TTL automatically, but we can clean up hit counters
            pattern = "medical:*:hits"
            hit_keys = self.redis_client.keys(pattern)
            
            expired_count = 0
            for key in hit_keys:
                ttl = self.redis_client.ttl(key)
                if ttl == -1:  # No expiration set
                    self.redis_client.expire(key, self.config.default_ttl)
                elif ttl == -2:  # Key doesn't exist
                    expired_count += 1
            
            return expired_count
            
        except Exception as e:
            logger.error(f"Cleanup error: {str(e)}")
            return 0

class ComponentCacheWrapper:
    """
    Wrapper class that adds caching functionality to medical pipeline components
    """
    
    def __init__(self, component_name: str, cache_manager: MedicalCacheManager):
        self.component_name = component_name
        self.cache_manager = cache_manager
    
    def cached_call(self, func: callable, input_data: Any, *args, **kwargs) -> Any:
        """
        Execute function with caching support
        
        Args:
            func: Function to execute
            input_data: Primary input used for cache key
            *args, **kwargs: Additional function arguments
            
        Returns:
            Function result (from cache or fresh execution)
        """
        # Try to get from cache
        cached_result = self.cache_manager.get_cached_result(self.component_name, input_data)
        if cached_result is not None:
            return cached_result
        
        # Execute function
        start_time = time.time()
        result = func(input_data, *args, **kwargs)
        execution_time = time.time() - start_time
        
        # Cache result if execution was successful and took reasonable time
        if result is not None and execution_time > 0.1:  # Cache if took more than 100ms
            self.cache_manager.store_result(self.component_name, input_data, result)
        
        return result
    
    def batch_cached_call(self, func: callable, input_list: List[Any], *args, **kwargs) -> List[Any]:
        """Execute function on batch of inputs with caching"""
        results = []
        uncached_inputs = []
        uncached_indices = []
        
        # Check cache for each input
        for i, input_data in enumerate(input_list):
            cached_result = self.cache_manager.get_cached_result(self.component_name, input_data)
            if cached_result is not None:
                results.append(cached_result)
            else:
                results.append(None)  # Placeholder
                uncached_inputs.append(input_data)
                uncached_indices.append(i)
        
        # Process uncached inputs
        if uncached_inputs:
            uncached_results = func(uncached_inputs, *args, **kwargs)
            
            # Store results and update final results
            for i, (input_data, result) in enumerate(zip(uncached_inputs, uncached_results)):
                original_index = uncached_indices[i]
                results[original_index] = result
                
                # Cache the result
                if result is not None:
                    self.cache_manager.store_result(self.component_name, input_data, result)
        
        return results