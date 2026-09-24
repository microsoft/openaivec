"""Caching utilities used across OpenAIVec."""

from typing import TYPE_CHECKING

from ._backend import CacheBackend, InMemoryCacheBackend
from .optimize import BatchSizeSuggester, PerformanceMetric
from .proxy import AsyncBatchCache, BatchCache, BatchCacheBase

if TYPE_CHECKING:
    from openaivec.duckdb_ext._cache import DuckDBCacheBackend

__all__ = [
    "AsyncBatchCache",
    "BatchSizeSuggester",
    "BatchCache",
    "CacheBackend",
    "DuckDBCacheBackend",
    "InMemoryCacheBackend",
    "PerformanceMetric",
    "BatchCacheBase",
]


def __getattr__(name: str) -> object:
    """Keep the historical DuckDB cache import available."""
    if name == "DuckDBCacheBackend":
        from openaivec.duckdb_ext._cache import DuckDBCacheBackend

        return DuckDBCacheBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
