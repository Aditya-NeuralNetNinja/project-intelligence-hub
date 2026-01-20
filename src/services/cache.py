"""
Query result caching for improved performance.

Provides in-memory caching with TTL for query results,
reducing latency and API costs for repeated queries.
"""

from __future__ import annotations

import hashlib
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from langchain.schema import Document


@dataclass
class CacheEntry:
    """A single cache entry with TTL support."""
    value: Any
    timestamp: float
    ttl: float
    hits: int = 0

    def is_expired(self) -> bool:
        """Check if entry has expired.

        Returns:
            True if entry is past its TTL.
        """
        return time.time() - self.timestamp > self.ttl

    def access(self) -> Any:
        """Access the cached value and increment hit counter.

        Returns:
            Cached value.
        """
        self.hits += 1
        return self.value


class QueryCache:
    """In-memory cache for query results with TTL."""

    def __init__(
        self,
        default_ttl: float = 3600,
        max_size: int = 1000,
    ) -> None:
        self._cache: Dict[str, CacheEntry] = {}
        self._lock = threading.RLock()
        self.default_ttl = default_ttl
        self.max_size = max_size
        self._total_hits = 0
        self._total_misses = 0

    def _make_key(self, query: str, context_hash: str = "") -> str:
        """Create cache key from query and context.

        Args:
            query: Query string.
            context_hash: Optional context identifier.

        Returns:
            MD5 hash key.
        """
        combined = f"{query.lower().strip()}:{context_hash}"
        return hashlib.md5(combined.encode()).hexdigest()

    def _evict_if_needed(self) -> None:
        """Evict oldest entries if cache is at capacity."""
        if len(self._cache) < self.max_size:
            return

        # Remove expired entries first
        self.cleanup_expired()

        # If still over capacity, remove oldest entries
        if len(self._cache) >= self.max_size:
            sorted_keys = sorted(
                self._cache.keys(),
                key=lambda k: self._cache[k].timestamp
            )
            # Remove oldest 10%
            to_remove = max(1, len(sorted_keys) // 10)
            for key in sorted_keys[:to_remove]:
                del self._cache[key]

    def get(
        self,
        query: str,
        context_hash: str = ""
    ) -> Optional[Any]:
        """Get cached result if exists and not expired.

        Args:
            query: Query string.
            context_hash: Optional context identifier.

        Returns:
            Cached value or None if not found/expired.
        """
        key = self._make_key(query, context_hash)

        with self._lock:
            entry = self._cache.get(key)

            if entry is None:
                self._total_misses += 1
                return None

            if entry.is_expired():
                del self._cache[key]
                self._total_misses += 1
                return None

            self._total_hits += 1
            return entry.access()

    def set(
        self,
        query: str,
        context_hash: str,
        value: Any,
        ttl: Optional[float] = None,
    ) -> None:
        """Cache a result.

        Args:
            query: Query string.
            context_hash: Context identifier.
            value: Value to cache.
            ttl: Optional TTL override.
        """
        key = self._make_key(query, context_hash)

        with self._lock:
            self._evict_if_needed()
            self._cache[key] = CacheEntry(
                value=value,
                timestamp=time.time(),
                ttl=ttl or self.default_ttl,
            )

    def invalidate(self, query: str, context_hash: str = "") -> bool:
        """Invalidate a specific cache entry.

        Args:
            query: Query string.
            context_hash: Context identifier.

        Returns:
            True if entry was found and removed.
        """
        key = self._make_key(query, context_hash)

        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    def invalidate_all(self) -> int:
        """Clear entire cache.

        Returns:
            Number of entries cleared.
        """
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            return count

    def cleanup_expired(self) -> int:
        """Remove expired entries.

        Returns:
            Number of entries removed.
        """
        with self._lock:
            expired_keys = [
                k for k, v in self._cache.items() if v.is_expired()
            ]
            for key in expired_keys:
                del self._cache[key]
            return len(expired_keys)

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache metrics.
        """
        with self._lock:
            total_requests = self._total_hits + self._total_misses
            hit_rate = (
                self._total_hits / total_requests
                if total_requests > 0
                else 0.0
            )

            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "total_hits": self._total_hits,
                "total_misses": self._total_misses,
                "hit_rate": round(hit_rate, 3),
                "default_ttl": self.default_ttl,
            }


class AnswerCache(QueryCache):
    """Specialized cache for GraphRAG answers.

    Extends QueryCache with answer-specific functionality like
    caching both the answer and supporting documents.
    """

    @dataclass
    class AnswerEntry:
        """Cached answer with supporting documents."""
        answer: str
        documents: List[Document] = field(default_factory=list)
        cypher_result: str = ""
        metadata: Dict[str, Any] = field(default_factory=dict)

    def set_answer(
        self,
        query: str,
        answer: str,
        documents: Optional[List[Document]] = None,
        cypher_result: str = "",
        context_hash: str = "",
        ttl: Optional[float] = None,
    ) -> None:
        """Cache a complete answer with metadata.

        Args:
            query: User query.
            answer: Generated answer.
            documents: Supporting documents.
            cypher_result: Cypher query result if any.
            context_hash: Context identifier.
            ttl: Optional TTL override.
        """
        entry = self.AnswerEntry(
            answer=answer,
            documents=documents or [],
            cypher_result=cypher_result,
            metadata={"cached_at": time.time()},
        )
        self.set(query, context_hash, entry, ttl)

    def get_answer(
        self,
        query: str,
        context_hash: str = ""
    ) -> Optional[AnswerEntry]:
        """Get cached answer.

        Args:
            query: User query.
            context_hash: Context identifier.

        Returns:
            AnswerEntry or None if not found.
        """
        result = self.get(query, context_hash)
        if isinstance(result, self.AnswerEntry):
            return result
        return None


# Global cache instances
_query_cache: Optional[QueryCache] = None
_answer_cache: Optional[AnswerCache] = None


def get_query_cache(
    default_ttl: float = 3600,
    max_size: int = 1000,
) -> QueryCache:
    """Get or create the global query cache.

    Args:
        default_ttl: Default TTL for entries.
        max_size: Maximum cache size.

    Returns:
        QueryCache singleton instance.
    """
    global _query_cache
    if _query_cache is None:
        _query_cache = QueryCache(default_ttl=default_ttl, max_size=max_size)
    return _query_cache


def get_answer_cache(
    default_ttl: float = 3600,
    max_size: int = 500,
) -> AnswerCache:
    """Get or create the global answer cache.

    Args:
        default_ttl: Default TTL for entries.
        max_size: Maximum cache size.

    Returns:
        AnswerCache singleton instance.
    """
    global _answer_cache
    if _answer_cache is None:
        _answer_cache = AnswerCache(default_ttl=default_ttl, max_size=max_size)
    return _answer_cache
