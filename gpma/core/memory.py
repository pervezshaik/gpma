"""
Memory Module

Provides memory systems for agents to store and retrieve information.

KEY CONCEPTS:
1. Short-Term Memory - Recent context, cleared after task completion
2. Long-Term Memory - Persistent storage across sessions
3. Semantic Memory - Vector-based similarity search (optional advanced feature)

WHY MEMORY?
- Agents need context to make good decisions
- Previous results inform current tasks
- Learning from past interactions
- Maintaining conversation history

LEARNING POINTS:
- This is a simplified in-memory implementation
- Production systems might use Redis, SQLite, or vector databases
- Memory retrieval can be key-based or semantic (similarity-based)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from collections import OrderedDict
import json
import logging
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class MemoryEntry:
    """
    A single memory item with metadata.

    Memories have:
    - Key: Unique identifier
    - Value: The stored data
    - Timestamp: When it was stored
    - TTL: How long to keep it (optional)
    - Tags: For categorization and retrieval
    """
    key: str
    value: Any
    timestamp: datetime = field(default_factory=datetime.now)
    ttl: Optional[timedelta] = None  # Time to live
    tags: List[str] = field(default_factory=list)
    access_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_expired(self) -> bool:
        """Check if this memory has expired."""
        if self.ttl is None:
            return False
        return datetime.now() > (self.timestamp + self.ttl)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "key": self.key,
            "value": self.value,
            "timestamp": self.timestamp.isoformat(),
            "ttl_seconds": self.ttl.total_seconds() if self.ttl else None,
            "tags": self.tags,
            "access_count": self.access_count,
            "metadata": self.metadata
        }


class Memory(ABC):
    """
    Abstract base class for memory implementations.

    All memory types support:
    - store: Save a value
    - retrieve: Get a value
    - search: Find values by criteria
    - forget: Remove values
    """

    @abstractmethod
    def store(self, key: str, value: Any, **kwargs) -> None:
        """Store a value in memory."""
        pass

    @abstractmethod
    def retrieve(self, key: str) -> Optional[Any]:
        """Retrieve a value by key."""
        pass

    @abstractmethod
    def search(self, query: str = None, tags: List[str] = None) -> List[MemoryEntry]:
        """Search for memories matching criteria."""
        pass

    @abstractmethod
    def forget(self, key: str) -> bool:
        """Remove a memory by key."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all memories."""
        pass


class ShortTermMemory(Memory):
    """
    Fast, limited-capacity memory for recent context.

    Uses LRU (Least Recently Used) eviction when capacity is reached.

    TYPICAL USE CASES:
    - Current conversation context
    - Recent task results
    - Temporary variables during processing

    EXAMPLE:
        stm = ShortTermMemory(capacity=100)
        stm.store("user_query", "What's the weather?")
        stm.store("api_response", {"temp": 72, "condition": "sunny"})

        query = stm.retrieve("user_query")  # "What's the weather?"
    """

    def __init__(self, capacity: int = 100, default_ttl: timedelta = None):
        """
        Initialize short-term memory.

        Args:
            capacity: Maximum number of items to store
            default_ttl: Default time-to-live for items
        """
        self.capacity = capacity
        self.default_ttl = default_ttl or timedelta(hours=1)
        self._storage: OrderedDict[str, MemoryEntry] = OrderedDict()

    def store(self, key: str, value: Any, ttl: timedelta = None, tags: List[str] = None) -> None:
        """
        Store a value in short-term memory.

        If capacity is exceeded, oldest items are evicted (LRU).
        """
        # Remove expired entries first
        self._cleanup_expired()

        # Evict oldest if at capacity
        while len(self._storage) >= self.capacity:
            oldest_key = next(iter(self._storage))
            del self._storage[oldest_key]
            logger.debug(f"STM evicted: {oldest_key}")

        entry = MemoryEntry(
            key=key,
            value=value,
            ttl=ttl or self.default_ttl,
            tags=tags or []
        )
        self._storage[key] = entry
        self._storage.move_to_end(key)  # Mark as recently used

    def retrieve(self, key: str) -> Optional[Any]:
        """Retrieve a value, returning None if not found or expired."""
        if key not in self._storage:
            return None

        entry = self._storage[key]
        if entry.is_expired():
            del self._storage[key]
            return None

        entry.access_count += 1
        self._storage.move_to_end(key)  # Mark as recently used
        return entry.value

    def search(self, query: str = None, tags: List[str] = None) -> List[MemoryEntry]:
        """
        Search memories by text query or tags.

        Simple implementation - production might use full-text search.
        """
        self._cleanup_expired()
        results = []

        for entry in self._storage.values():
            # Tag matching
            if tags:
                if not any(t in entry.tags for t in tags):
                    continue

            # Simple text matching in key or value
            if query:
                query_lower = query.lower()
                key_match = query_lower in entry.key.lower()
                value_match = query_lower in str(entry.value).lower()
                if not (key_match or value_match):
                    continue

            results.append(entry)

        return results

    def forget(self, key: str) -> bool:
        """Remove a specific memory."""
        if key in self._storage:
            del self._storage[key]
            return True
        return False

    def clear(self) -> None:
        """Clear all short-term memory."""
        self._storage.clear()

    def _cleanup_expired(self) -> None:
        """Remove all expired entries."""
        expired_keys = [k for k, v in self._storage.items() if v.is_expired()]
        for key in expired_keys:
            del self._storage[key]

    def get_recent(self, limit: int = 10) -> List[MemoryEntry]:
        """Get the most recent memories."""
        self._cleanup_expired()
        items = list(self._storage.values())
        return items[-limit:]

    def __len__(self) -> int:
        return len(self._storage)


class LongTermMemory(Memory):
    """
    Persistent memory that survives across sessions.

    In this implementation, we use file-based storage.
    Production might use SQLite, PostgreSQL, or a vector database.

    TYPICAL USE CASES:
    - User preferences
    - Learned patterns
    - Historical data
    - Knowledge base

    EXAMPLE:
        ltm = LongTermMemory("./agent_memory.json")
        ltm.store("user_preferences", {"theme": "dark", "language": "en"})

        # Later, even after restart:
        prefs = ltm.retrieve("user_preferences")  # {"theme": "dark", ...}
    """

    def __init__(self, storage_path: str = None):
        """
        Initialize long-term memory.

        Args:
            storage_path: Path to JSON file for persistence (optional)
        """
        self.storage_path = storage_path
        self._storage: Dict[str, MemoryEntry] = {}

        # Load existing data if path provided
        if storage_path:
            self._load()

    def store(self, key: str, value: Any, tags: List[str] = None, metadata: Dict[str, Any] = None) -> None:
        """Store a value in long-term memory."""
        entry = MemoryEntry(
            key=key,
            value=value,
            tags=tags or [],
            metadata=metadata or {}
        )
        self._storage[key] = entry

        # Persist to disk
        if self.storage_path:
            self._save()

    def retrieve(self, key: str) -> Optional[Any]:
        """Retrieve a value by key."""
        if key not in self._storage:
            return None
        entry = self._storage[key]
        entry.access_count += 1
        return entry.value

    def search(self, query: str = None, tags: List[str] = None) -> List[MemoryEntry]:
        """Search memories by query or tags."""
        results = []

        for entry in self._storage.values():
            # Tag matching
            if tags:
                if not any(t in entry.tags for t in tags):
                    continue

            # Text matching
            if query:
                query_lower = query.lower()
                key_match = query_lower in entry.key.lower()
                value_match = query_lower in str(entry.value).lower()
                if not (key_match or value_match):
                    continue

            results.append(entry)

        return results

    def forget(self, key: str) -> bool:
        """Remove a memory and persist the change."""
        if key in self._storage:
            del self._storage[key]
            if self.storage_path:
                self._save()
            return True
        return False

    def clear(self) -> None:
        """Clear all long-term memory."""
        self._storage.clear()
        if self.storage_path:
            self._save()

    def _save(self) -> None:
        """Persist memory to disk."""
        try:
            data = {k: v.to_dict() for k, v in self._storage.items()}
            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save memory: {e}")

    def _load(self) -> None:
        """Load memory from disk."""
        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
                for key, item in data.items():
                    self._storage[key] = MemoryEntry(
                        key=item["key"],
                        value=item["value"],
                        timestamp=datetime.fromisoformat(item["timestamp"]),
                        tags=item.get("tags", []),
                        access_count=item.get("access_count", 0),
                        metadata=item.get("metadata", {})
                    )
        except FileNotFoundError:
            pass  # Start with empty memory
        except Exception as e:
            logger.error(f"Failed to load memory: {e}")


class CompositeMemory:
    """
    Combines short-term and long-term memory for complete agent memory.

    This is what agents typically use - it provides both fast recent
    access and persistent storage.

    USAGE:
        memory = CompositeMemory(
            stm_capacity=100,
            ltm_path="./memory.json"
        )

        # Store in short-term (default)
        memory.store("current_task", "fetch weather")

        # Store in long-term explicitly
        memory.store("user_name", "Alice", long_term=True)

        # Retrieve searches both
        value = memory.retrieve("user_name")
    """

    def __init__(self, stm_capacity: int = 100, ltm_path: str = None):
        self.short_term = ShortTermMemory(capacity=stm_capacity)
        self.long_term = LongTermMemory(storage_path=ltm_path)

    def store(self, key: str, value: Any, long_term: bool = False, **kwargs) -> None:
        """
        Store a value in memory.

        Args:
            key: Unique identifier
            value: Data to store
            long_term: If True, store in persistent memory
            **kwargs: Additional arguments (tags, ttl, etc.)
        """
        if long_term:
            self.long_term.store(key, value, **kwargs)
        else:
            self.short_term.store(key, value, **kwargs)

    def retrieve(self, key: str) -> Optional[Any]:
        """
        Retrieve a value, checking short-term first, then long-term.
        """
        # Check short-term first (faster, more recent)
        value = self.short_term.retrieve(key)
        if value is not None:
            return value

        # Fall back to long-term
        return self.long_term.retrieve(key)

    def search(self, query: str = None, tags: List[str] = None) -> List[MemoryEntry]:
        """Search across both memory systems."""
        stm_results = self.short_term.search(query, tags)
        ltm_results = self.long_term.search(query, tags)

        # Combine and deduplicate by key (prefer STM for duplicates)
        seen_keys = {e.key for e in stm_results}
        combined = stm_results + [e for e in ltm_results if e.key not in seen_keys]

        return combined

    def forget(self, key: str) -> bool:
        """Remove from both memory systems."""
        stm_removed = self.short_term.forget(key)
        ltm_removed = self.long_term.forget(key)
        return stm_removed or ltm_removed

    def get_context(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent context for task processing.

        Returns the most recent memories as a list of dicts.
        """
        recent = self.short_term.get_recent(limit)
        return [
            {"key": e.key, "value": e.value, "timestamp": e.timestamp.isoformat()}
            for e in recent
        ]

    def promote_to_long_term(self, key: str) -> bool:
        """
        Move a short-term memory to long-term storage.

        Useful for important information that should persist.
        """
        value = self.short_term.retrieve(key)
        if value is not None:
            self.long_term.store(key, value)
            self.short_term.forget(key)
            return True
        return False
