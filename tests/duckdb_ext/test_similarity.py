"""DuckDB similarity search tests."""

import duckdb
import pytest

from openaivec.duckdb_ext import similarity_search


class TestSimilaritySearch:
    def test_basic_similarity(self):
        conn = duckdb.connect(":memory:")
        conn.execute("""
            CREATE TABLE docs (text VARCHAR, embedding FLOAT[3]);
            INSERT INTO docs VALUES
                ('doc_a', [1.0, 0.0, 0.0]),
                ('doc_b', [0.0, 1.0, 0.0]),
                ('doc_c', [0.7, 0.7, 0.0]);
        """)
        conn.execute("""
            CREATE TABLE queries (text VARCHAR, embedding FLOAT[3]);
            INSERT INTO queries VALUES ('q1', [1.0, 0.0, 0.0]);
        """)

        result = similarity_search(conn, "docs", "queries", top_k=2)
        df = result.df()

        assert len(df) == 2
        assert df.iloc[0]["target_text"] == "doc_a"
        assert df.iloc[0]["score"] == pytest.approx(1.0, abs=1e-5)
        conn.close()

    def test_multiple_queries(self):
        conn = duckdb.connect(":memory:")
        conn.execute("""
            CREATE TABLE targets (text VARCHAR, embedding FLOAT[2]);
            INSERT INTO targets VALUES ('t1', [1.0, 0.0]), ('t2', [0.0, 1.0]);
        """)
        conn.execute("""
            CREATE TABLE q (text VARCHAR, embedding FLOAT[2]);
            INSERT INTO q VALUES ('q1', [1.0, 0.0]), ('q2', [0.0, 1.0]);
        """)

        result = similarity_search(conn, "targets", "q", top_k=1)
        df = result.df()

        assert len(df) == 2
        # q1 should match t1, q2 should match t2
        q1_row = df[df["query_text"] == "q1"].iloc[0]
        assert q1_row["target_text"] == "t1"

        q2_row = df[df["query_text"] == "q2"].iloc[0]
        assert q2_row["target_text"] == "t2"
        conn.close()

    def test_custom_columns(self):
        conn = duckdb.connect(":memory:")
        conn.execute("""
            CREATE TABLE corpus (id VARCHAR, vec FLOAT[2]);
            INSERT INTO corpus VALUES ('c1', [1.0, 0.0]);
        """)
        conn.execute("""
            CREATE TABLE search (id VARCHAR, vec FLOAT[2]);
            INSERT INTO search VALUES ('s1', [1.0, 0.0]);
        """)

        result = similarity_search(
            conn,
            "corpus",
            "search",
            target_column="vec",
            query_column="vec",
            target_text_column="id",
            query_text_column="id",
            top_k=1,
        )
        df = result.df()
        assert len(df) == 1
        assert df.iloc[0]["query_text"] == "s1"
        assert df.iloc[0]["target_text"] == "c1"
        conn.close()
