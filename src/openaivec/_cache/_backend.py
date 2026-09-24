"""Cache backends for BatchCache.

Defines the ``CacheBackend`` runtime-checkable protocol and two concrete
implementations:

* ``InMemoryCacheBackend`` – default in-memory ``OrderedDict`` store.
* ``DuckDBCacheBackend`` – persistent DuckDB-backed store.

Both satisfy the same protocol so the batching proxy can swap backends
transparently.
"""

from __future__ import annotations

import base64
import json
import math
import pickle
import re
import threading
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

_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z_0-9]*\Z")
_KEY_FORMAT = "typed-json-v1"
_SQL_CHUNK_SIZE = 500


def _quote_table(table: str) -> str:
    """Validate a single, unqualified SQL identifier and quote it."""
    if not isinstance(table, str) or not _IDENTIFIER.fullmatch(table):
        raise ValueError("table must be a single SQL identifier (letters, digits, underscores; no leading digit)")
    return f'"{table}"'


def _key_data(key: object) -> list:
    if isinstance(key, bool):
        return ["bool", key]
    if isinstance(key, str):
        return ["str", key]
    if isinstance(key, int):
        return ["int", str(key)]
    if isinstance(key, float):
        if math.isnan(key):
            raise TypeError("NaN is not a supported DuckDB cache key")
        return ["float", key.hex()]
    if isinstance(key, bytes):
        return ["bytes", base64.b64encode(key).decode("ascii")]
    if isinstance(key, tuple):
        return ["tuple", [_key_data(item) for item in key]]
    raise TypeError(f"unsupported DuckDB cache key type: {type(key).__name__}")


def _encode_key(key: object) -> str:
    return json.dumps(_key_data(key), ensure_ascii=False, separators=(",", ":"))


def _decode_key(encoded: str) -> Hashable:
    tag, value = json.loads(encoded)
    if tag == "bool":
        return bool(value)
    if tag == "str":
        return str(value)
    if tag == "int":
        return int(value)
    if tag == "float":
        return float.fromhex(value)
    if tag == "bytes":
        return base64.b64decode(value)
    if tag == "tuple":
        return tuple(_decode_key(json.dumps(item)) for item in value)
    raise ValueError(f"unknown DuckDB cache key encoding: {tag}")


@dataclass
class DuckDBCacheBackend(Generic[T]):
    """Persistent cache backend backed by a DuckDB database.

    The caller is responsible for creating and injecting the DuckDB connection.
    Use the ``of`` classmethod for a convenient factory that handles connection
    creation.

    Attributes:
        conn (duckdb.DuckDBPyConnection): An open DuckDB connection.
        table (str): Unqualified SQL identifier for cache storage.

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
    _sql_table: str = field(init=False, repr=False)
    _next_seq: int = field(init=False, repr=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)

    def __post_init__(self) -> None:
        self._sql_table = _quote_table(self.table)
        self.conn.execute(
            f"CREATE TABLE IF NOT EXISTS {self._sql_table} ("
            "key TEXT PRIMARY KEY, value BLOB NOT NULL, access_seq BIGINT NOT NULL, "
            f"key_format TEXT NOT NULL DEFAULT '{_KEY_FORMAT}')"
        )
        columns = {row[1] for row in self.conn.execute(f"PRAGMA table_info({self._sql_table})").fetchall()}
        if not {"key", "value", "access_seq", "key_format"} <= columns:
            raise ValueError(
                f"cache table {self.table!r} uses an incompatible schema; "
                "choose a new table name or migrate the old entries explicitly"
            )
        row = self.conn.execute(
            f"SELECT max(access_seq), count(*) FILTER (WHERE key_format != ?) FROM {self._sql_table}",
            [_KEY_FORMAT],
        ).fetchone()
        if row is None or row[1]:
            raise ValueError(f"cache table {self.table!r} contains unsupported key-format rows")
        self._next_seq = row[0] or 0

    @classmethod
    def of(cls, database: str = ":memory:", *, table: str = "openaivec_cache") -> DuckDBCacheBackend:
        """Create a backend with a new DuckDB connection.

        Opens a connection, ensures the cache table exists, and returns
        the ready-to-use backend.

        Args:
            database (str): Path to the DuckDB database file.
            table (str): Single unqualified SQL identifier (letters, digits,
                underscores; cannot begin with a digit). SQL reserved words
                are supported.

        Returns:
            DuckDBCacheBackend: A new backend instance.
        """
        conn = duckdb.connect(database)
        try:
            return cls(conn=conn, table=table)
        except Exception:
            conn.close()
            raise

    def __contains__(self, key: object) -> bool:
        with self._lock:
            row = self.conn.execute(
                f"SELECT 1 FROM {self._sql_table} WHERE key = ? LIMIT 1", [_encode_key(key)]
            ).fetchone()
            return row is not None

    def __getitem__(self, key: object) -> T:
        encoded = _encode_key(key)
        with self._lock:
            row = self.conn.execute(f"SELECT value FROM {self._sql_table} WHERE key = ?", [encoded]).fetchone()
            if row is None:
                raise KeyError(key)
            self.touch_many([key])
            return pickle.loads(row[0])

    def __setitem__(self, key: object, value: T) -> None:
        self.put_many([(key, value)])

    def __len__(self) -> int:
        with self._lock:
            row = self.conn.execute(f"SELECT count(*) FROM {self._sql_table}").fetchone()
            return row[0] if row is not None else 0

    def __iter__(self) -> Iterator:
        return iter(self.keys())

    def get(self, key: object, default: T | None = None) -> T | None:
        """Return cached value or *default*."""
        try:
            return self[key]
        except KeyError:
            return default

    def move_to_end(self, key: object) -> None:
        """Refresh the access sequence for LRU bookkeeping."""
        self.touch_many([key])

    def pop_oldest(self) -> tuple[Hashable, T]:
        """Remove and return the least-recently used ``(key, value)`` pair."""
        with self._lock:
            self.conn.execute("BEGIN TRANSACTION")
            try:
                row = self.conn.execute(
                    f"SELECT key, value FROM {self._sql_table} ORDER BY access_seq ASC LIMIT 1"
                ).fetchone()
                if row is None:
                    raise KeyError("cache is empty")
                self.conn.execute(f"DELETE FROM {self._sql_table} WHERE key = ?", [row[0]])
                self.conn.execute("COMMIT")
            except Exception:
                self.conn.execute("ROLLBACK")
                raise
            return _decode_key(row[0]), pickle.loads(row[1])

    def keys(self) -> list[Hashable]:
        """Return all cache keys ordered by access time (oldest first)."""
        with self._lock:
            rows = self.conn.execute(f"SELECT key FROM {self._sql_table} ORDER BY access_seq ASC").fetchall()
            return [_decode_key(row[0]) for row in rows]

    def get_many(self, keys: list[Hashable]) -> dict[Hashable, T]:
        """Fetch a batch without updating LRU order; use ``touch_many`` after consumption."""
        encoded = list(dict.fromkeys(_encode_key(key) for key in keys))
        if len({_decode_key(key) for key in encoded}) != len(encoded):
            raise ValueError("bulk lookup cannot represent different key types that compare equal in a dict")
        if not encoded:
            return {}
        with self._lock:
            blobs: dict[str, bytes] = {}
            for start in range(0, len(encoded), _SQL_CHUNK_SIZE):
                rows = self.conn.execute(
                    f"SELECT key, value FROM {self._sql_table} WHERE key IN (SELECT unnest(?))",
                    [encoded[start : start + _SQL_CHUNK_SIZE]],
                ).fetchall()
                blobs.update(rows)
            return {_decode_key(key): pickle.loads(blobs[key]) for key in encoded if key in blobs}

    def put_many(self, items: list[tuple[Hashable, T]]) -> None:
        """Upsert entries in order in bounded SQL batches and one transaction."""
        encoded = list({_encode_key(key): pickle.dumps(value) for key, value in items}.items())
        if not encoded:
            return
        with self._lock:
            self.conn.execute("BEGIN TRANSACTION")
            try:
                row = self.conn.execute(f"SELECT max(access_seq) FROM {self._sql_table}").fetchone()
                self._next_seq = max(self._next_seq, row[0] or 0) if row is not None else self._next_seq
                for start in range(0, len(encoded), _SQL_CHUNK_SIZE):
                    chunk = encoded[start : start + _SQL_CHUNK_SIZE]
                    params: list[object] = []
                    for key, blob in chunk:
                        self._next_seq += 1
                        params.extend((key, blob, self._next_seq))
                    placeholders = ", ".join(["(?, ?, ?)"] * len(chunk))
                    self.conn.execute(
                        f"INSERT INTO {self._sql_table} (key, value, access_seq) VALUES {placeholders} "
                        "ON CONFLICT (key) DO UPDATE SET value = excluded.value, access_seq = excluded.access_seq",
                        params,
                    )
                self.conn.execute("COMMIT")
            except Exception:
                self.conn.execute("ROLLBACK")
                raise

    def touch_many(self, keys: list[Hashable]) -> None:
        """Update LRU order in one transactional SQL statement per chunk."""
        encoded = list(dict.fromkeys(_encode_key(key) for key in keys))
        if not encoded:
            return
        with self._lock:
            self.conn.execute("BEGIN TRANSACTION")
            try:
                row = self.conn.execute(f"SELECT max(access_seq) FROM {self._sql_table}").fetchone()
                self._next_seq = max(self._next_seq, row[0] or 0) if row is not None else self._next_seq
                for start in range(0, len(encoded), _SQL_CHUNK_SIZE):
                    chunk = encoded[start : start + _SQL_CHUNK_SIZE]
                    params: list[object] = []
                    for key in chunk:
                        self._next_seq += 1
                        params.extend((key, self._next_seq))
                    placeholders = ", ".join(["(?, ?)"] * len(chunk))
                    self.conn.execute(
                        f"UPDATE {self._sql_table} AS t SET access_seq = v.seq "
                        f"FROM (VALUES {placeholders}) AS v(key, seq) WHERE t.key = v.key",
                        params,
                    )
                self.conn.execute("COMMIT")
            except Exception:
                self.conn.execute("ROLLBACK")
                raise

    def clear(self) -> None:
        """Remove all entries from the cache table."""
        with self._lock:
            self.conn.execute(f"DELETE FROM {self._sql_table}")

    def close(self) -> None:
        """Close the underlying DuckDB connection."""
        with self._lock:
            self.conn.close()
