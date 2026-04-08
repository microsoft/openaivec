"""Tests for DuckDB cache backend."""

from __future__ import annotations

from enum import Enum

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
