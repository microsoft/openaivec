"""Arrow UDFs for batched responses, embeddings, and prepared tasks."""

from __future__ import annotations

from typing import Any

import duckdb
import numpy as np
import pyarrow as pa
from duckdb.func import FunctionNullHandling, PythonUDFType
from openai import AsyncOpenAI
from pydantic import BaseModel

from openaivec._cache import AsyncBatchCache
from openaivec._cache.proxy import DEFAULT_MANAGED_CACHE_SIZE
from openaivec._embeddings import AsyncBatchEmbeddings, EmbeddingLimits
from openaivec._model import EmbeddingsModelName, PreparedTask, ResponseFormat, ResponsesModelName
from openaivec._provider import CONTAINER
from openaivec._responses import AsyncBatchResponses
from openaivec._retry import RetryPolicy
from openaivec._util import run_async
from openaivec.duckdb_ext._types import _pydantic_to_struct_type, _serialize_for_duckdb

__all__ = ["responses_udf", "embeddings_udf", "task_udf"]

def responses_udf(
    conn: duckdb.DuckDBPyConnection,
    name: str,
    *,
    instructions: str,
    response_format: type = str,
    model_name: str | None = None,
    batch_size: int | None = 64,
    max_concurrency: int = 8,
    multimodal: bool = False,
    max_validation_retries: int = 3,
    retry_policy: RetryPolicy | None = None,
    **api_kwargs: Any,
) -> None:
    """Register a DuckDB Arrow-based UDF that calls the OpenAI Responses API.

    The UDF processes rows in vectorized batches via ``AsyncBatchResponses``,
    leveraging deduplication and concurrency for throughput.

    When ``response_format`` is a Pydantic ``BaseModel``, the UDF returns a
    DuckDB ``STRUCT`` whose fields match the model, allowing direct field
    access in SQL (e.g. ``SELECT udf(text).sentiment FROM ...``).
    When ``response_format`` is ``str``, the UDF returns ``VARCHAR``.

    SQL NULL inputs are not sent to the API. Missing parsed responses remain
    SQL NULL, including structured outputs; all-NULL Arrow batches retain the
    declared return type.

    Args:
        conn (duckdb.DuckDBPyConnection): An open DuckDB connection.
        name (str): UDF name visible in SQL.
        instructions (str): System prompt for the model.
        response_format (type): ``str`` for plain text or a Pydantic ``BaseModel``
            for structured output as a DuckDB STRUCT. Defaults to ``str``.
        model_name (str | None): Model or deployment name. Defaults to the
            container-registered ``ResponsesModelName``.
        batch_size (int | None): Rows per API batch; None enables auto-tuning.
            Defaults to 64.
        max_concurrency (int): Maximum concurrent API requests. Defaults to 8.
        max_validation_retries (int): Additional schema/ID corrections per batch.
            Defaults to 3; 0 disables correction. Must be nonnegative.
        retry_policy (RetryPolicy | None): Transport limits. ``None`` preserves SDK retries.
        **api_kwargs: Extra parameters forwarded to the OpenAI API.

    Example:
        >>> import duckdb
        >>> from pydantic import BaseModel
        >>> from openaivec.duckdb_ext import responses_udf
        >>> class Sentiment(BaseModel):
        ...     label: str
        ...     score: float
        >>> conn = duckdb.connect()
        >>> responses_udf(conn, "sentiment", instructions="Analyze sentiment", response_format=Sentiment)
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
        max_validation_retries=max_validation_retries,
        retry_policy=retry_policy,
        api_kwargs=api_kwargs,
        multimodal=multimodal,
    )

    is_structured = isinstance(response_format, type) and issubclass(response_format, BaseModel)
    return_type = _pydantic_to_struct_type(response_format) if is_structured else duckdb.sqltype("VARCHAR")
    arrow_type = (
        conn.sql("SELECT NULL")
        .select(duckdb.ConstantExpression(None).cast(return_type))
        .limit(0)
        .to_arrow_table()
        .schema.field(0)
        .type
    )

    def _batch_udf(arrow_batch: pa.Array) -> pa.Array:
        texts = arrow_batch.to_pylist()
        non_null_indices = [i for i, t in enumerate(texts) if t is not None]
        non_null_texts = [texts[i] for i in non_null_indices]

        if not non_null_texts:
            return pa.nulls(len(texts), type=arrow_type)

        results = run_async(batch_client.parse(non_null_texts))

        out: list[Any] = [None] * len(texts)
        for idx, result in zip(non_null_indices, results):
            if is_structured and isinstance(result, BaseModel):
                out[idx] = _serialize_for_duckdb(result.model_dump())
            elif result is not None:
                out[idx] = str(result)

        if all(value is None for value in out):
            return pa.nulls(len(out), type=arrow_type)
        return pa.array(out)

    conn.create_function(
        name,
        _batch_udf,
        [duckdb.sqltype("VARCHAR")],
        return_type,
        type=PythonUDFType.ARROW,
        null_handling=FunctionNullHandling.SPECIAL,
    )


def embeddings_udf(
    conn: duckdb.DuckDBPyConnection,
    name: str,
    *,
    model_name: str | None = None,
    batch_size: int = 128,
    max_concurrency: int = 8,
    limits: EmbeddingLimits | None = None,
    retry_policy: RetryPolicy | None = None,
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
        limits (EmbeddingLimits | None): Hard provider limits; None uses OpenAI defaults.
        retry_policy (RetryPolicy | None): Transport limits. ``None`` preserves SDK retries.
        **api_kwargs: Extra parameters forwarded to the OpenAI API.

    Example:
        >>> import duckdb
        >>> from openaivec.duckdb_ext import embeddings_udf
        >>> conn = duckdb.connect()
        >>> embeddings_udf(conn, "embed")
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
        limits=limits if limits is not None else EmbeddingLimits(),
        retry_policy=retry_policy,
    )

    def _batch_udf(arrow_batch: pa.Array) -> pa.Array:
        texts = arrow_batch.to_pylist()
        non_null_indices = [i for i, t in enumerate(texts) if t is not None]
        non_null_texts = [texts[i] for i in non_null_indices]

        if not non_null_texts:
            return pa.array([None] * len(texts))

        results = run_async(batch_client.create(non_null_texts))

        out: list[list[float] | None] = [None] * len(texts)
        for idx, vec in zip(non_null_indices, results):
            out[idx] = vec.tolist()

        return pa.array(out, type=pa.list_(pa.float32()))

    conn.create_function(
        name, _batch_udf, [duckdb.sqltype("VARCHAR")], duckdb.list_type("FLOAT"), type=PythonUDFType.ARROW
    )


def task_udf(
    conn: duckdb.DuckDBPyConnection,
    name: str,
    *,
    task: PreparedTask[ResponseFormat],
    model_name: str | None = None,
    batch_size: int | None = 64,
    max_concurrency: int = 8,
    multimodal: bool = False,
    max_validation_retries: int = 3,
    retry_policy: RetryPolicy | None = None,
    **api_kwargs: Any,
) -> None:
    """Register a DuckDB UDF backed by a ``PreparedTask``.

    Args:
        conn (duckdb.DuckDBPyConnection): An open DuckDB connection.
        name (str): UDF name visible in SQL.
        task (PreparedTask): Pre-configured task with instructions and response format.
        model_name (str | None): Model or deployment name.
        batch_size (int | None): Rows per API batch; None enables auto-tuning.
            Defaults to 64.
        max_concurrency (int): Maximum concurrent API requests. Defaults to 8.
        max_validation_retries (int): Additional schema/ID corrections per batch.
            Defaults to 3; 0 disables correction. Must be nonnegative.
        multimodal (bool): When ``True``, file paths and URLs are sent as
            multimodal content. Defaults to ``False``.
        retry_policy (RetryPolicy | None): Transport limits. ``None`` preserves SDK retries.
        **api_kwargs: Extra parameters forwarded to the OpenAI API.
    """
    responses_udf(
        conn,
        name,
        instructions=task.instructions,
        response_format=task.response_format,
        model_name=model_name,
        batch_size=batch_size,
        max_concurrency=max_concurrency,
        multimodal=multimodal,
        max_validation_retries=max_validation_retries,
        retry_policy=retry_policy,
        **api_kwargs,
    )
