"""Token-count UDF for DuckDB."""

from __future__ import annotations

import duckdb
import pyarrow as pa
import tiktoken
from duckdb.func import FunctionNullHandling, PythonUDFType

from openaivec._provider import CONTAINER

__all__ = ["count_tokens_udf"]


def count_tokens_udf(conn: duckdb.DuckDBPyConnection, name: str) -> None:
    """Register a vectorized DuckDB UDF that counts tokens in text values.

    The tokenizer is the same configured ``tiktoken.Encoding`` used by
    ``pandas.Series.ai.count_tokens``. SQL NULL values remain NULL.

    Args:
        conn (duckdb.DuckDBPyConnection): Connection on which to register the UDF.
        name (str): Function name visible in SQL.

    Example:
        >>> import duckdb
        >>> from openaivec.duckdb_ext import count_tokens_udf
        >>> conn = duckdb.connect()
        >>> count_tokens_udf(conn, "token_count")
        >>> conn.sql("SELECT token_count('hello')").fetchone()[0] > 0
        True
    """
    encoding = CONTAINER.resolve(tiktoken.Encoding)

    def _batch_udf(arrow_batch: pa.Array) -> pa.Array:
        texts = arrow_batch.to_pylist()
        non_null_texts = [text for text in texts if text is not None]
        if not non_null_texts:
            return pa.nulls(len(texts), type=pa.int64())

        counts = iter(len(tokens) for tokens in encoding.encode_batch(non_null_texts))
        return pa.array([next(counts) if text is not None else None for text in texts], type=pa.int64())

    conn.create_function(
        name,
        _batch_udf,
        [duckdb.sqltype("VARCHAR")],
        duckdb.sqltype("BIGINT"),
        type=PythonUDFType.ARROW,
        null_handling=FunctionNullHandling.SPECIAL,
    )
