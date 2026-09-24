"""Tests for DuckDB cache backend."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from time import perf_counter
from unittest.mock import MagicMock
from uuid import uuid4

import duckdb
import pytest
from pydantic import BaseModel

from openaivec._cache._backend import DuckDBCacheBackend, InMemoryCacheBackend


# Module-level models for pickle compatibility
class _Sentiment(BaseModel):
    label: str
    score: float


class _Detail(BaseModel):
    reason: str


class _NestedResult(BaseModel):
    sentiment: str
    detail: _Detail


class _Priority(int, Enum):
    LOW = 1
    HIGH = 2


class _TaskResult(BaseModel):
    priority: _Priority


class _UnicodeResult(BaseModel):
    text: str


class _ScoreResult(BaseModel):
    value: float


class _SentimentLabel(BaseModel):
    label: str


class TestInMemoryCacheBackend:
    """Verify the in-memory InMemoryCacheBackend used as default by proxies."""

    def test_set_and_get(self):
        c: InMemoryCacheBackend[str, int] = InMemoryCacheBackend()
        c["a"] = 1
        assert c["a"] == 1

    def test_contains(self):
        c: InMemoryCacheBackend[str, int] = InMemoryCacheBackend()
        assert "x" not in c
        c["x"] = 10
        assert "x" in c

    def test_len(self):
        c: InMemoryCacheBackend[str, int] = InMemoryCacheBackend()
        assert len(c) == 0
        c["a"] = 1
        c["b"] = 2
        assert len(c) == 2

    def test_get_default(self):
        c: InMemoryCacheBackend[str, int] = InMemoryCacheBackend()
        assert c.get("missing") is None
        assert c.get("missing", 42) == 42

    def test_pop_oldest(self):
        c: InMemoryCacheBackend[str, int] = InMemoryCacheBackend()
        c["a"] = 1
        c["b"] = 2
        key, val = c.pop_oldest()
        assert key == "a"
        assert val == 1
        assert len(c) == 1

    def test_move_to_end(self):
        c: InMemoryCacheBackend[str, int] = InMemoryCacheBackend()
        c["a"] = 1
        c["b"] = 2
        c.move_to_end("a")
        key, _ = c.pop_oldest()
        assert key == "b"

    def test_clear(self):
        c: InMemoryCacheBackend[str, int] = InMemoryCacheBackend()
        c["a"] = 1
        c.clear()
        assert len(c) == 0

    def test_keys(self):
        c: InMemoryCacheBackend[str, int] = InMemoryCacheBackend()
        c["b"] = 2
        c["a"] = 1
        assert c.keys() == ["b", "a"]

    def test_iter(self):
        c: InMemoryCacheBackend[str, int] = InMemoryCacheBackend()
        c["x"] = 10
        c["y"] = 20
        assert list(c) == ["x", "y"]


class TestDuckDBCacheBackend:
    """Test the DuckDB-backed persistent cache."""

    def test_set_and_get(self):
        c = DuckDBCacheBackend.of(":memory:")
        c["hello"] = "world"
        assert c["hello"] == "world"
        c.close()

    def test_contains(self):
        c = DuckDBCacheBackend.of(":memory:")
        assert "x" not in c
        c["x"] = 42
        assert "x" in c
        c.close()

    def test_len(self):
        c = DuckDBCacheBackend.of(":memory:")
        assert len(c) == 0
        c["a"] = 1
        c["b"] = 2
        assert len(c) == 2
        c.close()

    def test_get_default(self):
        c = DuckDBCacheBackend.of(":memory:")
        assert c.get("missing") is None
        assert c.get("missing", 99) == 99
        c.close()

    def test_getitem_missing_raises(self):
        c = DuckDBCacheBackend.of(":memory:")
        with pytest.raises(KeyError):
            c["nonexistent"]
        c.close()

    def test_overwrite(self):
        c = DuckDBCacheBackend.of(":memory:")
        c["key"] = "v1"
        c["key"] = "v2"
        assert c["key"] == "v2"
        assert len(c) == 1
        c.close()

    def test_pop_oldest(self):
        c = DuckDBCacheBackend.of(":memory:")
        c["first"] = 100
        c["second"] = 200
        key, val = c.pop_oldest()
        assert key == "first"
        assert val == 100
        assert len(c) == 1
        c.close()

    def test_pop_oldest_empty_raises(self):
        c = DuckDBCacheBackend.of(":memory:")
        with pytest.raises(KeyError):
            c.pop_oldest()
        c.close()

    def test_clear(self):
        c = DuckDBCacheBackend.of(":memory:")
        c["a"] = 1
        c["b"] = 2
        c.clear()
        assert len(c) == 0
        c.close()

    def test_keys(self):
        c = DuckDBCacheBackend.of(":memory:")
        c["b"] = 2
        c["a"] = 1
        keys = c.keys()
        assert set(keys) == {"a", "b"}
        c.close()

    def test_complex_values(self):
        """Verify pickle handles complex Python objects."""
        import numpy as np

        c = DuckDBCacheBackend.of(":memory:")
        vec = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        c["vec"] = vec
        restored = c["vec"]
        np.testing.assert_array_equal(restored, vec)
        c.close()

    def test_custom_table_name(self):
        c = DuckDBCacheBackend.of(":memory:", table="my_cache")
        c["k"] = "v"
        assert c["k"] == "v"
        c.close()

    @pytest.mark.parametrize("table", ["select", "from", "MixedCase"])
    def test_reserved_and_quoted_table_names(self, table):
        c = DuckDBCacheBackend.of(":memory:", table=table)
        c["key"] = 3
        assert c["key"] == 3
        c.clear()
        assert len(c) == 0
        c.close()
        direct = DuckDBCacheBackend(conn=duckdb.connect(), table=table)
        direct["key"] = 4
        assert direct["key"] == 4
        direct.close()

    @pytest.mark.parametrize("table", ["", "1invalid", "schema.cache", "a b", "x\x00y"])
    def test_invalid_table_names_rejected_in_both_constructors(self, table):
        with pytest.raises(ValueError, match="single SQL identifier"):
            DuckDBCacheBackend.of(":memory:", table=table)
        conn = duckdb.connect()
        with pytest.raises(ValueError, match="single SQL identifier"):
            DuckDBCacheBackend(conn=conn, table=table)
        conn.close()

    def test_injection_payload_does_not_execute(self):
        conn = duckdb.connect()
        conn.execute("CREATE TABLE important (value INTEGER)")
        with pytest.raises(ValueError, match="single SQL identifier"):
            DuckDBCacheBackend(conn=conn, table="cache; DROP TABLE important; --")
        assert conn.execute("SELECT count(*) FROM important").fetchone()[0] == 0
        conn.close()

    def test_distinct_type_tagged_keys_and_order(self):
        c = DuckDBCacheBackend.of(":memory:")
        entries = [(1, "integer"), ("1", "string"), (True, "boolean"), (1.0, "float"), (b"1", "bytes"), ((1,), "tuple")]
        for key, value in entries:
            c[key] = value
        assert len(c) == len(entries)
        for key, value in entries:
            assert c[key] == value
        assert c.keys() == [key for key, _ in entries]
        assert [c.pop_oldest()[1] for _ in entries] == [value for _, value in entries]
        c.close()

    def test_unsupported_keys_fail_clearly(self):
        c = DuckDBCacheBackend.of(":memory:")
        with pytest.raises(TypeError, match="unsupported DuckDB cache key type: frozenset"):
            c[frozenset({"a"})] = "value"
        with pytest.raises(TypeError, match="unsupported DuckDB cache key type: frozenset"):
            _ = frozenset({"a"}) in c
        with pytest.raises(TypeError, match="NaN is not a supported"):
            c[float("nan")] = "value"
        c.close()

    def test_bulk_lookup_rejects_python_equal_distinct_keys(self):
        c = DuckDBCacheBackend.of(":memory:")
        c.put_many([(1, "integer"), (True, "boolean")])
        assert c[1] == "integer"
        assert c[True] == "boolean"
        with pytest.raises(ValueError, match="different key types that compare equal"):
            c.get_many([1, True])
        c.close()

    def test_reopen_persists_typed_keys(self):
        database = Path("artifacts") / f"cache-{uuid4().hex}.duckdb"
        database.parent.mkdir(exist_ok=True)
        try:
            first = DuckDBCacheBackend.of(str(database))
            first[1] = "integer"
            first["1"] = "string"
            first.close()
            second = DuckDBCacheBackend.of(str(database))
            assert second[1] == "integer"
            assert second["1"] == "string"
            second.close()
        finally:
            database.unlink(missing_ok=True)
            database.with_suffix(".duckdb.wal").unlink(missing_ok=True)

    def test_live_instances_keep_access_sequence_monotonic(self):
        conn = duckdb.connect()
        first = DuckDBCacheBackend(conn=conn)
        second = DuckDBCacheBackend(conn=conn)
        first["a"] = 1
        second["b"] = 2
        first.move_to_end("a")
        assert first.pop_oldest() == ("b", 2)
        first.close()

    def test_old_schema_requires_explicit_migration(self):
        conn = duckdb.connect()
        conn.execute("CREATE TABLE legacy (key TEXT PRIMARY KEY, value BLOB NOT NULL, accessed_at TIMESTAMP)")
        conn.execute("INSERT INTO legacy VALUES ('1', '\\x01', now())")
        with pytest.raises(ValueError, match="choose a new table name or migrate"):
            DuckDBCacheBackend(conn=conn, table="legacy")
        assert conn.execute("SELECT key FROM legacy").fetchone()[0] == "1"
        conn.close()

    def test_bulk_methods_bound_query_count_and_lru(self):
        conn = MagicMock(wraps=duckdb.connect())
        c = DuckDBCacheBackend(conn=conn)
        entries = [(f"k{i}", i) for i in range(1000)]
        conn.execute.reset_mock()
        c.put_many(entries)
        assert conn.execute.call_count <= 5
        conn.execute.reset_mock()
        assert c.get_many([key for key, _ in entries]) == dict(entries)
        assert conn.execute.call_count <= 2
        conn.execute.reset_mock()
        c.touch_many(["k0", "k0", "k1"])
        assert conn.execute.call_count <= 4
        assert c.pop_oldest() == ("k2", 2)
        c.close()

    def test_move_to_end_updates_access(self):
        c = DuckDBCacheBackend.of(":memory:")
        c["a"] = 1
        c["b"] = 2
        c.move_to_end("a")
        key, _ = c.pop_oldest()
        assert key == "b"
        c.close()

    def test_iter(self):
        c = DuckDBCacheBackend.of(":memory:")
        c["x"] = 10
        c["y"] = 20
        assert set(c) == {"x", "y"}
        c.close()


class TestDuckDBCacheBackendWithProxy:
    """Test DuckDBCacheBackend as a drop-in for BatchCache."""

    @pytest.mark.asyncio
    async def test_async_1000_key_bulk_round_trips(self):
        from openaivec._cache import AsyncBatchCache

        conn = MagicMock(wraps=duckdb.connect())
        backend = DuckDBCacheBackend(conn=conn)
        proxy: AsyncBatchCache[str, str] = AsyncBatchCache(
            batch_size=1000, max_concurrency=2, cache=backend, show_progress=False
        )
        items = [f"k{i}" for i in range(1000)]

        async def mapper(xs: list[str]) -> list[str]:
            return [f"result:{key}" for key in xs]

        conn.execute.reset_mock()
        expected = [f"result:{key}" for key in items]
        assert await proxy.map(items, mapper) == expected
        assert conn.execute.call_count <= 24
        conn.execute.reset_mock()
        assert await proxy.map(items, mapper) == expected
        assert conn.execute.call_count <= 12
        backend.close()

    def test_1000_key_cold_warm_bulk_round_trips(self):
        from openaivec._cache import BatchCache

        conn = MagicMock(wraps=duckdb.connect())
        backend = DuckDBCacheBackend(conn=conn)
        proxy: BatchCache[str, str] = BatchCache(batch_size=1000, cache=backend, show_progress=False)
        items = [f"k{i}" for i in range(1000)]
        calls = 0

        def mapper(xs: list[str]) -> list[str]:
            nonlocal calls
            calls += 1
            return [f"result:{key}" for key in xs]

        conn.execute.reset_mock()
        start = perf_counter()
        expected = [f"result:{key}" for key in items]
        assert proxy.map(items, mapper) == expected
        cold = perf_counter() - start
        cold_queries = conn.execute.call_count
        conn.execute.reset_mock()
        start = perf_counter()
        assert proxy.map(items, mapper) == expected
        warm = perf_counter() - start
        warm_queries = conn.execute.call_count
        assert calls == 1
        assert cold_queries <= 24
        assert warm_queries <= 12
        print(f"DuckDB 1,000 keys: cold={cold:.3f}s/{cold_queries} SQL; warm={warm:.3f}s/{warm_queries} SQL")
        backend.close()

    def test_proxy_with_duckdb_backend(self):
        from openaivec._cache import BatchCache

        backend = DuckDBCacheBackend.of(":memory:")
        proxy: BatchCache[str, str] = BatchCache(
            batch_size=2,
            cache=backend,
        )

        calls: list[list[str]] = []

        def mapper(xs: list[str]) -> list[str]:
            calls.append(xs[:])
            return [f"result:{x}" for x in xs]

        out = proxy.map(["a", "b", "c"], mapper)
        assert out == ["result:a", "result:b", "result:c"]

        # Second call: a and b should be cached
        out2 = proxy.map(["a", "b", "d"], mapper)
        assert out2 == ["result:a", "result:b", "result:d"]
        # Only "d" should have been called
        assert calls[-1] == ["d"]

        backend.close()

    def test_proxy_dedup_with_duckdb(self):
        from openaivec._cache import BatchCache

        backend = DuckDBCacheBackend.of(":memory:")
        proxy: BatchCache[str, str] = BatchCache(
            batch_size=10,
            cache=backend,
        )

        calls: list[list[str]] = []

        def mapper(xs: list[str]) -> list[str]:
            calls.append(xs[:])
            return [x.upper() for x in xs]

        out = proxy.map(["a", "b", "a", "c", "b"], mapper)
        assert out == ["A", "B", "A", "C", "B"]
        # Dedup: only unique items processed
        assert calls == [["a", "b", "c"]]

        backend.close()

    @pytest.mark.asyncio
    async def test_async_proxy_with_duckdb_backend(self):
        from openaivec._cache import AsyncBatchCache

        backend = DuckDBCacheBackend.of(":memory:")
        proxy: AsyncBatchCache[str, str] = AsyncBatchCache(
            batch_size=2,
            max_concurrency=2,
            cache=backend,
        )

        async def mapper(xs: list[str]) -> list[str]:
            return [f"async:{x}" for x in xs]

        out = await proxy.map(["x", "y", "z"], mapper)
        assert out == ["async:x", "async:y", "async:z"]

        # Cached
        out2 = await proxy.map(["x", "w"], mapper)
        assert out2 == ["async:x", "async:w"]

        backend.close()


class TestDuckDBCacheResponsesEdgeCases:
    """Edge cases for caching Pydantic model responses in DuckDB."""

    def test_pydantic_model_roundtrip(self):
        c = DuckDBCacheBackend.of(":memory:")
        original = _Sentiment(label="positive", score=0.95)
        c["text1"] = original
        restored = c["text1"]
        assert isinstance(restored, _Sentiment)
        assert restored.label == "positive"
        assert restored.score == 0.95
        c.close()

    def test_nested_pydantic_model_roundtrip(self):
        c = DuckDBCacheBackend.of(":memory:")
        original = _NestedResult(sentiment="negative", detail=_Detail(reason="broken"))
        c["text2"] = original
        restored = c["text2"]
        assert restored.detail.reason == "broken"
        c.close()

    def test_enum_field_roundtrip(self):
        c = DuckDBCacheBackend.of(":memory:")
        c["t"] = _TaskResult(priority=_Priority.HIGH)
        restored = c["t"]
        assert restored.priority == _Priority.HIGH
        assert restored.priority.value == 2
        c.close()

    def test_none_cached_value(self):
        c = DuckDBCacheBackend.of(":memory:")
        c["null_result"] = None
        assert c["null_result"] is None
        c.close()

    def test_unicode_key_and_model(self):
        c = DuckDBCacheBackend.of(":memory:")
        c["\u65e5\u672c\u8a9e\u306e\u30ec\u30d3\u30e5\u30fc"] = _UnicodeResult(text="\u30dd\u30b8\u30c6\u30a3\u30d6")
        assert c["\u65e5\u672c\u8a9e\u306e\u30ec\u30d3\u30e5\u30fc"].text == "\u30dd\u30b8\u30c6\u30a3\u30d6"
        c.close()

    def test_overwrite_preserves_latest_model(self):
        c = DuckDBCacheBackend.of(":memory:")
        c["k"] = _ScoreResult(value=0.5)
        c["k"] = _ScoreResult(value=0.9)
        assert c["k"].value == 0.9
        assert len(c) == 1
        c.close()

    def test_special_chars_in_key(self):
        c = DuckDBCacheBackend.of(":memory:")
        keys = [
            "line1\nline2",
            "tab\there",
            'quote"inside',
            "single'quote",
            "backslash\\path",
            "",
        ]
        for i, k in enumerate(keys):
            c[k] = f"val{i}"
        for i, k in enumerate(keys):
            assert c[k] == f"val{i}"
        c.close()

    @pytest.mark.asyncio
    async def test_async_cache_dedup_with_pydantic(self):
        from openaivec._cache import AsyncBatchCache

        backend = DuckDBCacheBackend.of(":memory:")
        cache: AsyncBatchCache[str, _SentimentLabel] = AsyncBatchCache(
            batch_size=10, max_concurrency=2, cache=backend, show_progress=False
        )
        call_count = 0

        async def mock_parse(texts: list[str]) -> list[_SentimentLabel]:
            nonlocal call_count
            call_count += 1
            return [_SentimentLabel(label="pos") for _ in texts]

        r1 = await cache.map(["a", "b", "a"], mock_parse)
        assert len(r1) == 3
        assert r1[0] == r1[2]
        assert call_count == 1

        r2 = await cache.map(["a", "c"], mock_parse)
        assert r2[0].label == "pos"
        assert call_count == 2

        backend.close()
