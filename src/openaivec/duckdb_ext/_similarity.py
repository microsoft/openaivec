"""Top-k cosine similarity search over DuckDB relations."""

from __future__ import annotations

import duckdb

from openaivec.duckdb_ext._identifiers import _quote_identifier, _quote_table_name

__all__ = ["similarity_search"]


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
        duckdb.DuckDBPyRelation: A DuckDB relation with ``query_id`` (1-based
        query row position), ``query_text``, ``target_text``, and ``score``,
        ordered by query row and descending similarity.

    Raises:
        ValueError: If an identifier or ``top_k`` is invalid.

    Example:
        >>> import duckdb
        >>> from openaivec.duckdb_ext import similarity_search
        >>> conn = duckdb.connect()
        >>> # (after populating docs and queries tables with embeddings)
        >>> results = similarity_search(conn, "docs", "queries", top_k=5)
        >>> results.df()
    """
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    target = _quote_table_name(target_table)
    query = _quote_table_name(query_table)
    target_vector = _quote_identifier(target_column)
    query_vector = _quote_identifier(query_column)
    target_text = _quote_identifier(target_text_column)
    query_text = _quote_identifier(query_text_column)

    sql = f"""
        WITH queries AS (
            SELECT row_number() OVER () AS query_id,
                   q.{query_text} AS query_text,
                   q.{query_vector} AS query_vector
            FROM {query} AS q
        ), candidates AS (
            SELECT q.query_id, q.query_text, t.{target_text} AS target_text,
                   list_cosine_similarity(t.{target_vector}::FLOAT[], q.query_vector::FLOAT[]) AS score
            FROM queries AS q
            CROSS JOIN {target} AS t
        )
        SELECT
            query_id, query_text, target_text, score
        FROM candidates
        QUALIFY row_number() OVER (
            PARTITION BY query_id
            ORDER BY score DESC
        ) <= ?
        ORDER BY query_id, score DESC
    """
    return conn.sql(sql, params=[top_k])
