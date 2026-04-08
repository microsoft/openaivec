"""Cache backends for BatchCache.

Defines the ``CacheBackend`` runtime-checkable protocol and two concrete
implementations:

* ``InMemoryCacheBackend`` – default in-memory ``OrderedDict`` store.
* ``DuckDBCacheBackend`` – persistent DuckDB-backed store.

Both satisfy the same protocol so the batching proxy can swap backends
transparently.
"""

from __future__ import annotations

import pickle
from collections import OrderedDict
from collections.abc import Hashable, Iterator
from dataclasses import dataclass, field
from typing import Generic, Protocol, TypeVar, runtime_checkable

import duckdb

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


# ---------------------------------------------------------------------------
# DuckDB persistent backend
# ---------------------------------------------------------------------------

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS {table} (
    key TEXT PRIMARY KEY,
    value BLOB NOT NULL,
    accessed_at TIMESTAMP DEFAULT now()
)
"""

_UPSERT_SQL = """
INSERT INTO {table} (key, value, accessed_at)
VALUES ($1, $2, now())
ON CONFLICT (key) DO UPDATE SET value = excluded.value, accessed_at = now()
"""

_TOUCH_SQL = "UPDATE {table} SET accessed_at = now() WHERE key = $1"
_SELECT_SQL = "SELECT value FROM {table} WHERE key = $1"
_DELETE_OLDEST_SQL = """
DELETE FROM {table}
WHERE key IN (
    SELECT key FROM {table} ORDER BY accessed_at ASC LIMIT $1
)
"""
_COUNT_SQL = "SELECT count(*) FROM {table}"
_CONTAINS_SQL = "SELECT 1 FROM {table} WHERE key = $1 LIMIT 1"
_ALL_KEYS_SQL = "SELECT key FROM {table} ORDER BY accessed_at ASC"
_DELETE_KEY_SQL = "DELETE FROM {table} WHERE key = $1"
_DELETE_ALL_SQL = "DELETE FROM {table}"


@dataclass
class DuckDBCacheBackend:
    """Persistent cache backend backed by a DuckDB database.

    The caller is responsible for creating and injecting the DuckDB connection.
    Use the ``of`` classmethod for a convenient factory that handles connection
    creation.

    Attributes:
        conn (duckdb.DuckDBPyConnection): An open DuckDB connection.
        table (str): Table name used for cache storage.

    Example:
        >>> from openaivec._cache._backend import DuckDBCacheBackend
        >>> cache = DuckDBCacheBackend.of(":memory:")
        >>> cache["hello"] = [1.0, 2.0, 3.0]
        >>> cache["hello"]
        [1.0, 2.0, 3.0]
        >>> len(cache)
        1
        >>> cache.close()
    """

    conn: duckdb.DuckDBPyConnection
    table: str = "openaivec_cache"

    @classmethod
    def of(cls, database: str = ":memory:", *, table: str = "openaivec_cache") -> DuckDBCacheBackend:
        """Create a backend with a new DuckDB connection.

        Opens a connection, ensures the cache table exists, and returns
        the ready-to-use backend.

        Args:
            database (str): Path to the DuckDB database file.
            table (str): Cache table name.

        Returns:
            DuckDBCacheBackend: A new backend instance.
        """
        conn = duckdb.connect(database)
        conn.execute(_CREATE_TABLE_SQL.format(table=table))
        return cls(conn=conn, table=table)

    def __contains__(self, key: object) -> bool:
        result = self.conn.execute(_CONTAINS_SQL.format(table=self.table), [str(key)]).fetchone()
        return result is not None

    def __getitem__(self, key: object) -> T:
        result = self.conn.execute(_SELECT_SQL.format(table=self.table), [str(key)]).fetchone()
        if result is None:
            raise KeyError(key)
        self.conn.execute(_TOUCH_SQL.format(table=self.table), [str(key)])
        return pickle.loads(result[0])

    def __setitem__(self, key: object, value: T) -> None:
        blob = pickle.dumps(value)
        self.conn.execute(_UPSERT_SQL.format(table=self.table), [str(key), blob])

    def __len__(self) -> int:
        result = self.conn.execute(_COUNT_SQL.format(table=self.table)).fetchone()
        return result[0] if result else 0

    def __iter__(self) -> Iterator:
        rows = self.conn.execute(_ALL_KEYS_SQL.format(table=self.table)).fetchall()
        return iter(row[0] for row in rows)

    def get(self, key: object, default: T | None = None) -> T | None:
        """Return cached value or *default*."""
        try:
            return self[key]
        except KeyError:
            return default

    def move_to_end(self, key: object) -> None:
        """Refresh the ``accessed_at`` timestamp for LRU bookkeeping."""
        self.conn.execute(_TOUCH_SQL.format(table=self.table), [str(key)])

    def pop_oldest(self) -> tuple[str, T]:
        """Remove and return the least-recently used ``(key, value)`` pair."""
        row = self.conn.execute(f"SELECT key, value FROM {self.table} ORDER BY accessed_at ASC LIMIT 1").fetchone()
        if row is None:
            raise KeyError("cache is empty")
        key, blob = row
        self.conn.execute(_DELETE_KEY_SQL.format(table=self.table), [key])
        return key, pickle.loads(blob)

    def keys(self) -> list[str]:
        """Return all cache keys ordered by access time (oldest first)."""
        rows = self.conn.execute(_ALL_KEYS_SQL.format(table=self.table)).fetchall()
        return [row[0] for row in rows]

    def clear(self) -> None:
        """Remove all entries from the cache table."""
        self.conn.execute(_DELETE_ALL_SQL.format(table=self.table))

    def close(self) -> None:
        """Close the underlying DuckDB connection."""
        self.conn.close()
