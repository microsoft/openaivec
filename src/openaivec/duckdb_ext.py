"""DuckDB integration for openaivec.

Provides helpers that bridge openaivec's batched AI capabilities with DuckDB:

- **UDF registration** – register ``responses``, ``embeddings`` and ``task``
  functions directly as DuckDB scalar UDFs so SQL queries can invoke the
  OpenAI API transparently.
- **Persistent caching** – pass ``DuckDBCacheBackend`` as the ``_cache`` field
  of ``BatchCache`` for cross-session cache persistence.
- **Vector similarity** – ``similarity_search`` performs top-k cosine similarity
  queries against an embedding table using DuckDB's built-in
  ``list_cosine_similarity``.
- **Schema → DDL** – ``pydantic_to_duckdb_ddl`` converts a Pydantic model to a
  ``CREATE TABLE`` statement for immediate SQL analysis of structured-output
  results.

## Quick Start

```python
import duckdb
from openaivec.duckdb_ext import register_responses_udf, register_embeddings_udf

conn = duckdb.connect()
register_responses_udf(conn, "translate", instructions="Translate to French")
register_embeddings_udf(conn, "embed")

conn.sql("SELECT translate(review) FROM products")
conn.sql("SELECT text, embed(text) FROM documents")
```
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Coroutine
from typing import Any, TypeVar

import duckdb
import numpy as np
import pyarrow as pa
from openai import AsyncOpenAI
from pydantic import BaseModel

from openaivec._cache import AsyncBatchCache
from openaivec._cache.proxy import DEFAULT_MANAGED_CACHE_SIZE
from openaivec._embeddings import AsyncBatchEmbeddings
from openaivec._model import EmbeddingsModelName, PreparedTask, ResponseFormat, ResponsesModelName
from openaivec._provider import CONTAINER
from openaivec._responses import AsyncBatchResponses

__all__ = [
    "pydantic_to_duckdb_ddl",
    "register_embeddings_udf",
    "register_responses_udf",
    "register_task_udf",
    "similarity_search",
]

_LOGGER = logging.getLogger(__name__)
_T = TypeVar("_T")


def _run_async(coro: Coroutine[Any, Any, _T]) -> _T:
    """Run a coroutine from any context, including inside a running event loop.

    When called from a Jupyter notebook (or any environment that already has a
    running asyncio loop), ``asyncio.run`` / ``loop.run_until_complete`` would
    raise ``RuntimeError``.  This helper spawns a dedicated background thread
    with its own event loop so the call always succeeds.
    """
    result: _T | None = None
    exc: BaseException | None = None

    def _target() -> None:
        nonlocal result, exc
        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(coro)
        except BaseException as e:
            exc = e
        finally:
            loop.close()

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()
    thread.join()
    if exc is not None:
        raise exc
    return result  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# DuckDB UDF registration
# ---------------------------------------------------------------------------


def _pydantic_to_struct_type(model: type[BaseModel]) -> duckdb.DuckDBPyType:
    """Convert a Pydantic model to a DuckDB STRUCT type for UDF return values."""
    fields: dict[str, str] = {}
    for field_name, field_info in model.model_fields.items():
        fields[field_name] = _python_type_to_duckdb(field_info.annotation) if field_info.annotation else "VARCHAR"
    return duckdb.struct_type(fields)


def _model_to_dict(result: BaseModel) -> dict:
    """Recursively convert a Pydantic model to a plain dict for DuckDB STRUCT."""
    out = {}
    for key, value in result.model_dump().items():
        out[key] = value
    return out


def register_responses_udf(
    conn: duckdb.DuckDBPyConnection,
    name: str,
    *,
    instructions: str,
    response_format: type = str,
    model_name: str | None = None,
    batch_size: int = 64,
    max_concurrency: int = 8,
    **api_kwargs: Any,
) -> None:
    """Register a DuckDB Arrow-based UDF that calls the OpenAI Responses API.

    The UDF processes rows in vectorized batches via ``AsyncBatchResponses``,
    leveraging deduplication and concurrency for throughput.

    When ``response_format`` is a Pydantic ``BaseModel``, the UDF returns a
    DuckDB ``STRUCT`` whose fields match the model, allowing direct field
    access in SQL (e.g. ``SELECT udf(text).sentiment FROM ...``).
    When ``response_format`` is ``str``, the UDF returns ``VARCHAR``.

    Args:
        conn (duckdb.DuckDBPyConnection): An open DuckDB connection.
        name (str): UDF name visible in SQL.
        instructions (str): System prompt for the model.
        response_format (type): ``str`` for plain text or a Pydantic ``BaseModel``
            for structured output as a DuckDB STRUCT. Defaults to ``str``.
        model_name (str | None): Model or deployment name. Defaults to the
            container-registered ``ResponsesModelName``.
        batch_size (int): Rows per API batch. Defaults to 64.
        max_concurrency (int): Maximum concurrent API requests. Defaults to 8.
        **api_kwargs: Extra parameters forwarded to the OpenAI API.

    Example:
        >>> import duckdb
        >>> from pydantic import BaseModel
        >>> from openaivec.duckdb_ext import register_responses_udf
        >>> class Sentiment(BaseModel):
        ...     label: str
        ...     score: float
        >>> conn = duckdb.connect()
        >>> register_responses_udf(conn, "sentiment", instructions="Analyze sentiment", response_format=Sentiment)
        >>> # conn.sql("SELECT sentiment(text).label, sentiment(text).score FROM docs")
    """

    _model_name = model_name or CONTAINER.resolve(ResponsesModelName).value
    async_client = CONTAINER.resolve(AsyncOpenAI)

    cache: AsyncBatchCache = AsyncBatchCache(
        batch_size=batch_size,
        max_concurrency=max_concurrency,
        max_cache_size=DEFAULT_MANAGED_CACHE_SIZE,
        show_progress=False,
    )
    batch_client = AsyncBatchResponses(
        client=async_client,
        model_name=_model_name,
        system_message=instructions,
        response_format=response_format,
        cache=cache,
        api_kwargs=api_kwargs,
    )

    is_structured = isinstance(response_format, type) and issubclass(response_format, BaseModel)
    return_type = _pydantic_to_struct_type(response_format) if is_structured else duckdb.sqltype("VARCHAR")

    def _batch_udf(arrow_batch: pa.Array) -> pa.Array:
        texts = arrow_batch.to_pylist()
        non_null_indices = [i for i, t in enumerate(texts) if t is not None]
        non_null_texts = [texts[i] for i in non_null_indices]

        if not non_null_texts:
            if is_structured:
                return pa.nulls(len(texts))
            return pa.array([None] * len(texts), type=pa.string())

        results = _run_async(batch_client.parse(non_null_texts))

        out = [None] * len(texts)
        for idx, result in zip(non_null_indices, results):
            if is_structured and isinstance(result, BaseModel):
                out[idx] = _serialize_for_duckdb(result.model_dump())
            elif result is not None:
                out[idx] = str(result)

        return pa.array(out)

    conn.create_function(name, _batch_udf, [duckdb.sqltype("VARCHAR")], return_type, type="arrow")


def register_embeddings_udf(
    conn: duckdb.DuckDBPyConnection,
    name: str,
    *,
    model_name: str | None = None,
    batch_size: int = 128,
    max_concurrency: int = 8,
    **api_kwargs: Any,
) -> None:
    """Register a DuckDB Arrow-based UDF that returns embedding vectors.

    The UDF processes rows in vectorized batches via ``AsyncBatchEmbeddings``,
    leveraging deduplication and concurrency for throughput.

    Args:
        conn (duckdb.DuckDBPyConnection): An open DuckDB connection.
        name (str): UDF name visible in SQL.
        model_name (str | None): Embeddings model or deployment name.
        batch_size (int): Rows per API batch. Defaults to 128.
        max_concurrency (int): Maximum concurrent API requests. Defaults to 8.
        **api_kwargs: Extra parameters forwarded to the OpenAI API.

    Example:
        >>> import duckdb
        >>> from openaivec.duckdb_ext import register_embeddings_udf
        >>> conn = duckdb.connect()
        >>> register_embeddings_udf(conn, "embed")
        >>> # conn.sql("SELECT embed(text) FROM docs")
    """

    _model_name = model_name or CONTAINER.resolve(EmbeddingsModelName).value
    async_client = CONTAINER.resolve(AsyncOpenAI)

    cache: AsyncBatchCache[str, np.ndarray] = AsyncBatchCache(
        batch_size=batch_size,
        max_concurrency=max_concurrency,
        max_cache_size=DEFAULT_MANAGED_CACHE_SIZE,
        show_progress=False,
    )
    batch_client = AsyncBatchEmbeddings(
        client=async_client,
        model_name=_model_name,
        cache=cache,
        api_kwargs=api_kwargs,
    )

    def _batch_udf(arrow_batch: pa.Array) -> pa.Array:
        texts = arrow_batch.to_pylist()
        non_null_indices = [i for i, t in enumerate(texts) if t is not None]
        non_null_texts = [texts[i] for i in non_null_indices]

        if not non_null_texts:
            return pa.array([None] * len(texts))

        results = _run_async(batch_client.create(non_null_texts))

        out: list[list[float] | None] = [None] * len(texts)
        for idx, vec in zip(non_null_indices, results):
            out[idx] = vec.tolist()

        return pa.array(out, type=pa.list_(pa.float32()))

    conn.create_function(name, _batch_udf, [duckdb.sqltype("VARCHAR")], duckdb.list_type("FLOAT"), type="arrow")


def register_task_udf(
    conn: duckdb.DuckDBPyConnection,
    name: str,
    *,
    task: PreparedTask[ResponseFormat],
    model_name: str | None = None,
    batch_size: int = 64,
    max_concurrency: int = 8,
    **api_kwargs: Any,
) -> None:
    """Register a DuckDB UDF backed by a ``PreparedTask``.

    Args:
        conn (duckdb.DuckDBPyConnection): An open DuckDB connection.
        name (str): UDF name visible in SQL.
        task (PreparedTask): Pre-configured task with instructions and response format.
        model_name (str | None): Model or deployment name.
        batch_size (int): Rows per API batch. Defaults to 64.
        max_concurrency (int): Maximum concurrent API requests. Defaults to 8.
        **api_kwargs: Extra parameters forwarded to the OpenAI API.
    """
    register_responses_udf(
        conn,
        name,
        instructions=task.instructions,
        response_format=task.response_format,
        model_name=model_name,
        batch_size=batch_size,
        max_concurrency=max_concurrency,
        **api_kwargs,
    )


# ---------------------------------------------------------------------------
# Vector similarity search
# ---------------------------------------------------------------------------


def similarity_search(
    conn: duckdb.DuckDBPyConnection,
    target_table: str,
    query_table: str,
    *,
    target_column: str = "embedding",
    query_column: str = "embedding",
    target_text_column: str = "text",
    query_text_column: str = "text",
    top_k: int = 10,
) -> duckdb.DuckDBPyRelation:
    """Perform top-k cosine similarity search between two DuckDB tables.

    Uses DuckDB's built-in ``list_cosine_similarity`` for efficient
    vector comparison without leaving SQL.

    Args:
        conn (duckdb.DuckDBPyConnection): An open DuckDB connection.
        target_table (str): Table containing candidate embeddings.
        query_table (str): Table containing query embeddings.
        target_column (str): Embedding column in *target_table*.
        query_column (str): Embedding column in *query_table*.
        target_text_column (str): Text identifier column in *target_table*.
        query_text_column (str): Text identifier column in *query_table*.
        top_k (int): Number of results per query.

    Returns:
        duckdb.DuckDBPyRelation: A DuckDB relation with columns ``query_text``,
        ``target_text``, ``score`` ordered by descending similarity.

    Example:
        >>> import duckdb
        >>> from openaivec.duckdb_ext import similarity_search
        >>> conn = duckdb.connect()
        >>> # (after populating docs and queries tables with embeddings)
        >>> results = similarity_search(conn, "docs", "queries", top_k=5)
        >>> results.df()
    """
    sql = f"""
        SELECT
            q.{query_text_column} AS query_text,
            t.{target_text_column} AS target_text,
            list_cosine_similarity(
                t.{target_column}::FLOAT[],
                q.{query_column}::FLOAT[]
            ) AS score
        FROM {query_table} q
        CROSS JOIN {target_table} t
        QUALIFY row_number() OVER (
            PARTITION BY q.{query_text_column}
            ORDER BY list_cosine_similarity(
                t.{target_column}::FLOAT[],
                q.{query_column}::FLOAT[]
            ) DESC
        ) <= {top_k}
        ORDER BY q.{query_text_column}, score DESC
    """
    return conn.sql(sql)


# ---------------------------------------------------------------------------
# Pydantic → DuckDB DDL
# ---------------------------------------------------------------------------

_PYDANTIC_TO_DUCKDB_TYPE: dict[type, str] = {
    str: "VARCHAR",
    int: "INTEGER",
    float: "DOUBLE",
    bool: "BOOLEAN",
    bytes: "BLOB",
}


def _python_type_to_duckdb(py_type: type) -> str:
    """Map a Python/Pydantic type to its DuckDB column type string."""
    from enum import Enum

    if py_type in _PYDANTIC_TO_DUCKDB_TYPE:
        return _PYDANTIC_TO_DUCKDB_TYPE[py_type]

    origin = getattr(py_type, "__origin__", None)

    # Enum → map based on value type
    if isinstance(py_type, type) and issubclass(py_type, Enum):
        if issubclass(py_type, int):
            return "INTEGER"
        if issubclass(py_type, float):
            return "DOUBLE"
        return "VARCHAR"

    # list[T] → T[]
    if origin is list:
        args = getattr(py_type, "__args__", ())
        inner = args[0] if args else Any
        return f"{_python_type_to_duckdb(inner)}[]"

    # dict → JSON
    if origin is dict or py_type is dict:
        return "JSON"

    # Optional / Union with None
    if origin is type(int | str):  # types.UnionType for X | Y
        args = [a for a in py_type.__args__ if a is not type(None)]
        if len(args) == 1:
            return _python_type_to_duckdb(args[0])
        return "VARCHAR"

    # Pydantic BaseModel → STRUCT
    if isinstance(py_type, type) and issubclass(py_type, BaseModel):
        fields = []
        for field_name, field_info in py_type.model_fields.items():
            col_type = _python_type_to_duckdb(field_info.annotation) if field_info.annotation else "VARCHAR"
            fields.append(f"{field_name} {col_type}")
        return f"STRUCT({', '.join(fields)})"

    # Literal → VARCHAR
    if hasattr(py_type, "__args__") and hasattr(py_type, "__origin__"):
        import typing

        if py_type.__origin__ is typing.Literal:
            return "VARCHAR"

    return "VARCHAR"


def _serialize_for_duckdb(value: Any) -> Any:
    """Recursively convert Enum values to their primitives for DuckDB."""
    from enum import Enum

    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {k: _serialize_for_duckdb(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_serialize_for_duckdb(v) for v in value]
    return value


def pydantic_to_duckdb_ddl(model: type[BaseModel], table_name: str) -> str:
    """Generate a ``CREATE TABLE`` DDL statement from a Pydantic model.

    Args:
        model (type[BaseModel]): The Pydantic model class.
        table_name (str): Name for the DuckDB table.

    Returns:
        str: A ``CREATE TABLE IF NOT EXISTS`` statement.

    Example:
        >>> from pydantic import BaseModel
        >>> from openaivec.duckdb_ext import pydantic_to_duckdb_ddl
        >>> class Review(BaseModel):
        ...     sentiment: str
        ...     rating: int
        ...     tags: list[str]
        >>> print(pydantic_to_duckdb_ddl(Review, "reviews"))
        CREATE TABLE IF NOT EXISTS reviews (
            sentiment VARCHAR,
            rating INTEGER,
            tags VARCHAR[]
        )
    """
    columns: list[str] = []
    for field_name, field_info in model.model_fields.items():
        col_type = _python_type_to_duckdb(field_info.annotation) if field_info.annotation else "VARCHAR"
        columns.append(f"    {field_name} {col_type}")
    body = ",\n".join(columns)
    return f"CREATE TABLE IF NOT EXISTS {table_name} (\n{body}\n)"
