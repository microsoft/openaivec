"""DuckDB integration for openaivec.

Provides helpers that bridge openaivec's batched AI capabilities with DuckDB:

- **UDF registration** – register ``responses``, ``embeddings``, ``task``,
  ``parse`` and ``count_tokens``
  functions directly as DuckDB scalar UDFs for SQL queries.
- **Schema inference** – infer a Pydantic response model once from a bounded
  table sample before registering a typed ``parse`` UDF.
- **Persistent caching** – pass ``DuckDBCacheBackend`` as the ``cache`` field
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
from openaivec.duckdb_ext import responses_udf, embeddings_udf

conn = duckdb.connect()
responses_udf(conn, "translate", instructions="Translate to French", reasoning={"effort": "none"})
embeddings_udf(conn, "embed")

conn.sql("SELECT translate(review) FROM products")
conn.sql("SELECT text, embed(text) FROM documents")
```
"""

from __future__ import annotations

import logging
import typing
from datetime import date, datetime, time
from decimal import Decimal
from enum import Enum
from typing import Any
from uuid import UUID

import duckdb
import numpy as np
import pyarrow as pa
from duckdb.func import FunctionNullHandling, PythonUDFType
from duckdb.sqltypes import DuckDBPyType
from openai import AsyncOpenAI
from pydantic import BaseModel

from openaivec._cache import AsyncBatchCache
from openaivec._cache.proxy import DEFAULT_MANAGED_CACHE_SIZE
from openaivec._duckdb_tokens import count_tokens_udf
from openaivec._embeddings import AsyncBatchEmbeddings, EmbeddingLimits
from openaivec._model import EmbeddingsModelName, PreparedTask, ResponseFormat, ResponsesModelName
from openaivec._provider import CONTAINER
from openaivec._responses import AsyncBatchResponses
from openaivec._retry import RetryPolicy
from openaivec._schema import SchemaInferenceInput, SchemaInferenceOutput, SchemaInferer
from openaivec._util import run_async

__all__ = [
    "count_tokens_udf",
    "infer_schema",
    "parse_udf",
    "pydantic_to_duckdb_ddl",
    "embeddings_udf",
    "responses_udf",
    "task_udf",
    "similarity_search",
]

_LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DuckDB UDF registration
# ---------------------------------------------------------------------------


def _pydantic_to_struct_type(model: type[BaseModel]) -> DuckDBPyType:
    """Convert a Pydantic model to a DuckDB STRUCT type for UDF return values."""
    fields: dict[str, str] = {}
    for field_name, field_info in model.model_fields.items():
        fields[field_name] = _python_type_to_duckdb(field_info.annotation) if field_info.annotation else "VARCHAR"
    return duckdb.struct_type(fields)


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


def _quote_identifier(name: str) -> str:
    """Quote a single DuckDB column name, including embedded double quotes."""
    if not isinstance(name, str) or not name:
        raise ValueError("example_field_name must be a non-empty string")
    return '"' + name.replace('"', '""') + '"'


def infer_schema(
    conn: duckdb.DuckDBPyConnection,
    instructions: str,
    example_table_name: str,
    example_field_name: str,
    max_examples: int = 100,
    *,
    max_retries: int = 8,
    retry_policy: RetryPolicy | None = None,
    **api_kwargs: Any,
) -> SchemaInferenceOutput:
    """Infer a response schema from a bounded sample of a DuckDB column.

    The table is resolved through DuckDB's relation API, which accepts regular,
    schema-qualified, and quoted table names without interpolating the name in
    SQL. NULL values are skipped before the sample limit is applied.

    Args:
        conn (duckdb.DuckDBPyConnection): Connection containing the example table.
        instructions (str): Description of the information to extract.
        example_table_name (str): Table or view containing examples.
        example_field_name (str): Column containing example text.
        max_examples (int): Maximum number of non-NULL examples. Must be positive.
        max_retries (int): Maximum schema inference attempts. Must be positive.
        retry_policy (RetryPolicy | None): Transport retry policy.
        **api_kwargs: Extra parameters for schema inference API calls.

    Returns:
        SchemaInferenceOutput: Inferred prompt and Pydantic model.

    Raises:
        ValueError: If limits are invalid or the sample has no non-NULL values.

    Example:
        >>> import duckdb
        >>> conn = duckdb.connect()
        >>> conn.sql("CREATE TABLE reviews(text VARCHAR)")
        >>> # infer_schema(conn, "Extract sentiment", "reviews", "text")
    """
    if max_examples < 1:
        raise ValueError("max_examples must be >= 1")
    if max_retries < 1:
        raise ValueError("max_retries must be >= 1")
    if not isinstance(example_table_name, str) or not example_table_name:
        raise ValueError("example_table_name must be a non-empty string")

    column = _quote_identifier(example_field_name)
    examples = [
        row[0]
        for row in (
            conn.table(example_table_name)
            .filter(f"{column} IS NOT NULL")
            .select(f"CAST({column} AS VARCHAR)")
            .limit(max_examples)
            .fetchall()
        )
    ]
    if not examples:
        raise ValueError("No non-NULL examples found in the selected column")

    inference_input = SchemaInferenceInput(examples=examples, instructions=instructions)
    inferer = CONTAINER.resolve(SchemaInferer)
    return inferer.infer_schema(
        inference_input,
        max_retries=max_retries,
        retry_policy=retry_policy,
        **api_kwargs,
    )


def parse_udf(
    conn: duckdb.DuckDBPyConnection,
    name: str,
    *,
    instructions: str,
    response_format: type[ResponseFormat] | None = None,
    example_table_name: str | None = None,
    example_field_name: str | None = None,
    max_examples: int = 100,
    model_name: str | None = None,
    batch_size: int | None = 64,
    max_concurrency: int = 8,
    multimodal: bool = False,
    max_retries: int = 8,
    max_validation_retries: int = 3,
    retry_policy: RetryPolicy | None = None,
    **api_kwargs: Any,
) -> None:
    """Register a DuckDB parse UDF with an explicit or inferred output schema.

    DuckDB fixes UDF return types at registration. If ``response_format`` is
    omitted, the schema is inferred once from ``example_table_name`` and
    ``example_field_name`` before registering the UDF. The generated UDF then
    returns a typed DuckDB ``STRUCT`` for each non-NULL input.

    Args:
        conn (duckdb.DuckDBPyConnection): Connection on which to register the UDF.
        name (str): UDF name visible in SQL.
        instructions (str): Description of the information to parse.
        response_format (type[ResponseFormat] | None): ``str`` or a Pydantic
            ``BaseModel`` subclass. If None, infer the schema from examples.
        example_table_name (str | None): Source table for schema inference.
        example_field_name (str | None): Text column for schema inference.
        max_examples (int): Maximum non-NULL examples to inspect.
        model_name (str | None): Model or deployment name.
        batch_size (int | None): Inputs per API batch; None enables auto-tuning.
        max_concurrency (int): Maximum concurrent API requests.
        multimodal (bool): Whether file paths and URLs are multimodal inputs.
        max_retries (int): Maximum schema inference attempts.
        max_validation_retries (int): Additional extraction corrections.
        retry_policy (RetryPolicy | None): Transport retry policy.
        **api_kwargs: Extra parameters for inference and parsing API calls.

    Raises:
        ValueError: If limits are invalid or inference inputs are missing.
        TypeError: If an explicit response format is unsupported.

    Example:
        >>> import duckdb
        >>> from pydantic import BaseModel
        >>> class Parsed(BaseModel):
        ...     label: str
        >>> conn = duckdb.connect()
        >>> parse_udf(conn, "parse_label", instructions="Extract label", response_format=Parsed)
    """
    if max_examples < 1:
        raise ValueError("max_examples must be >= 1")
    if max_retries < 1:
        raise ValueError("max_retries must be >= 1")
    if max_validation_retries < 0:
        raise ValueError("max_validation_retries must be >= 0")

    if response_format is None:
        if not example_table_name or not example_field_name:
            raise ValueError("Both example_table_name and example_field_name are required for schema inference")
        schema = infer_schema(
            conn,
            instructions,
            example_table_name,
            example_field_name,
            max_examples=max_examples,
            max_retries=max_retries,
            retry_policy=retry_policy,
            **api_kwargs,
        )
        resolved_instructions = schema.inference_prompt
        resolved_response_format = schema.model
    elif response_format is str or (isinstance(response_format, type) and issubclass(response_format, BaseModel)):
        resolved_instructions = instructions
        resolved_response_format = response_format
    else:
        raise TypeError("response_format must be str or a Pydantic BaseModel subclass")

    responses_udf(
        conn,
        name,
        instructions=resolved_instructions,
        response_format=resolved_response_format,
        model_name=model_name,
        batch_size=batch_size,
        max_concurrency=max_concurrency,
        multimodal=multimodal,
        max_validation_retries=max_validation_retries,
        retry_policy=retry_policy,
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

_PRIMITIVE_TYPE_MAP: dict[type, str] = {
    str: "VARCHAR",
    int: "INTEGER",
    float: "DOUBLE",
    bool: "BOOLEAN",
    bytes: "BLOB",
    datetime: "TIMESTAMP",
    date: "DATE",
    time: "TIME",
    Decimal: "DECIMAL",
    UUID: "UUID",
}


def _python_type_to_duckdb(py_type: Any) -> str:
    """Map a Python/Pydantic type to its DuckDB column type string."""
    if py_type in _PRIMITIVE_TYPE_MAP:
        return _PRIMITIVE_TYPE_MAP[py_type]

    origin = getattr(py_type, "__origin__", None)

    if isinstance(py_type, type) and issubclass(py_type, Enum):
        if issubclass(py_type, int):
            return "INTEGER"
        if issubclass(py_type, float):
            return "DOUBLE"
        return "VARCHAR"

    if origin is list:
        args = getattr(py_type, "__args__", ())
        inner = args[0] if args else Any
        return f"{_python_type_to_duckdb(inner)}[]"

    if origin is dict or py_type is dict:
        return "JSON"

    if origin is type(int | str):  # types.UnionType
        args = [a for a in py_type.__args__ if a is not type(None)]
        return _python_type_to_duckdb(args[0]) if len(args) == 1 else "VARCHAR"

    if isinstance(py_type, type) and issubclass(py_type, BaseModel):
        fields = [
            f"{name} {_python_type_to_duckdb(info.annotation) if info.annotation else 'VARCHAR'}"
            for name, info in py_type.model_fields.items()
        ]
        return f"STRUCT({', '.join(fields)})"

    if hasattr(py_type, "__origin__") and py_type.__origin__ is typing.Literal:
        return "VARCHAR"

    return "VARCHAR"


def _serialize_for_duckdb(value: Any) -> Any:
    """Recursively convert Enum values to their primitives for DuckDB."""
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
