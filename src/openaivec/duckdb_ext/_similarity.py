"""Top-k cosine similarity search over DuckDB relations."""

from __future__ import annotations

import duckdb

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
