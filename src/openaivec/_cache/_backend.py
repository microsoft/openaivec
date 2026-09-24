"""Cache backends for BatchCache.

Defines the ``CacheBackend`` runtime-checkable protocol and the default
``InMemoryCacheBackend`` implementation. The persistent DuckDB implementation
lives in ``openaivec.duckdb_ext._cache``.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Hashable, Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Generic, Protocol, TypeVar, runtime_checkable

if TYPE_CHECKING:
    from openaivec.duckdb_ext._cache import DuckDBCacheBackend  # noqa: F401

__all__: list[str] = []

S = TypeVar("S", bound=Hashable)
T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class CacheBackend(Protocol[S, T]):
    """Protocol that cache backends must satisfy.

    ``BatchCache`` and ``AsyncBatchCache`` accept any object
    that implements these methods.  Two built-in implementations are
    provided: ``InMemoryCacheBackend`` (default) and ``DuckDBCacheBackend``.
    """

    def __contains__(self, key: S) -> bool: ...
    def __getitem__(self, key: S) -> T: ...
    def __setitem__(self, key: S, value: T) -> None: ...
    def __len__(self) -> int: ...
    def __iter__(self) -> Iterator[S]: ...
    def get(self, key: S, default: T | None = None) -> T | None: ...
    def move_to_end(self, key: S) -> None: ...
    def pop_oldest(self) -> tuple[S, T]: ...
    def clear(self) -> None: ...
    def keys(self) -> list[S]: ...
    def close(self) -> None: ...


# ---------------------------------------------------------------------------
# In-memory backend (default)
# ---------------------------------------------------------------------------


@dataclass
class InMemoryCacheBackend(Generic[S, T]):
    """In-memory cache backend wrapping an ``OrderedDict``.

    This is the default backend used by ``BatchCache`` /
    ``AsyncBatchCache``.  Replace with ``DuckDBCacheBackend`` for
    persistent cross-session caching.
    """

    _data: OrderedDict[S, T] = field(default_factory=OrderedDict, init=False, repr=False)

    def __contains__(self, key: S) -> bool:
        return key in self._data

    def __getitem__(self, key: S) -> T:
        return self._data[key]

    def __setitem__(self, key: S, value: T) -> None:
        self._data[key] = value

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self) -> Iterator[S]:
        return iter(self._data)

    def get(self, key: S, default: T | None = None) -> T | None:
        """Return cached value or *default*."""
        return self._data.get(key, default)

    def move_to_end(self, key: S) -> None:
        """Mark *key* as most-recently used (LRU bookkeeping)."""
        self._data.move_to_end(key)

    def pop_oldest(self) -> tuple[S, T]:
        """Remove and return the least-recently used ``(key, value)`` pair."""
        return self._data.popitem(last=False)

    def clear(self) -> None:
        """Remove all entries."""
        self._data.clear()

    def keys(self) -> list[S]:
        """Return all cache keys in insertion order."""
        return list(self._data.keys())

    def close(self) -> None:
        """Release any external resources (no-op for in-memory backend)."""


def __getattr__(name: str) -> object:
    """Keep the historical DuckDB cache import available."""
    if name == "DuckDBCacheBackend":
        from openaivec.duckdb_ext._cache import DuckDBCacheBackend

        return DuckDBCacheBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
