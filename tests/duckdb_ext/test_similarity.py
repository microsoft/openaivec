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

    @pytest.mark.parametrize("top_k", [1, 2])
    def test_duplicate_query_text_has_independent_ranking(self, top_k):
        conn = duckdb.connect(":memory:")
        conn.execute("CREATE TABLE docs (text VARCHAR, embedding FLOAT[2])")
        conn.execute("INSERT INTO docs VALUES ('first', [1, 0]), ('second', [0, 1])")
        conn.execute("CREATE TABLE queries (text VARCHAR, embedding FLOAT[2])")
        conn.execute("INSERT INTO queries VALUES ('same', [1, 0]), ('same', [0, 1])")

        rows = similarity_search(conn, "docs", "queries", top_k=top_k).fetchall()
        assert len(rows) == 2 * top_k
        assert [row[0] for row in rows] == [1] * top_k + [2] * top_k
        assert rows[0][2] == "first"
        assert rows[top_k][2] == "second"
        conn.close()

    def test_quoted_and_qualified_identifiers(self):
        conn = duckdb.connect(":memory:")
        conn.execute('CREATE SCHEMA "my schema"')
        conn.execute('CREATE TABLE "my schema"."select" ("from" VARCHAR, "vec tor" FLOAT[2])')
        conn.execute('CREATE TABLE "my schema"."queries" ("query text" VARCHAR, "vec tor" FLOAT[2])')
        conn.execute('INSERT INTO "my schema"."select" VALUES (\'doc\', [1, 0])')
        conn.execute('INSERT INTO "my schema"."queries" VALUES (\'q\', [1, 0])')

        rows = similarity_search(
            conn,
            '"my schema"."select"',
            '"my schema"."queries"',
            target_column="vec tor",
            query_column="vec tor",
            target_text_column="from",
            query_text_column="query text",
            top_k=1,
        ).fetchall()
        assert rows[0][:3] == (1, "q", "doc")
        conn.close()

    def test_literal_dot_in_quoted_table_name(self):
        conn = duckdb.connect(":memory:")
        conn.execute('CREATE TABLE "doc.set" (text VARCHAR, embedding FLOAT[2])')
        conn.execute("INSERT INTO \"doc.set\" VALUES ('doc', [1, 0])")
        conn.execute("CREATE TABLE queries (text VARCHAR, embedding FLOAT[2])")
        conn.execute("INSERT INTO queries VALUES ('q', [1, 0])")

        assert similarity_search(conn, '"doc.set"', "queries", top_k=1).fetchall()[0][2] == "doc"
        conn.close()

    @pytest.mark.parametrize("top_k", [0, -1, 1.5, True, "1; DROP TABLE docs"])
    def test_invalid_top_k(self, top_k):
        conn = duckdb.connect(":memory:")
        with pytest.raises(ValueError, match="top_k"):
            similarity_search(conn, "docs", "queries", top_k=top_k)

    @pytest.mark.parametrize(
        "identifier", ["", "docs; DROP TABLE docs", "docs -- comment", "schema..docs", "docs\nWHERE 1=1"]
    )
    def test_invalid_table_identifier(self, identifier):
        conn = duckdb.connect(":memory:")
        with pytest.raises(ValueError, match="identifier"):
            similarity_search(conn, identifier, "queries")

    def test_invalid_column_identifier(self):
        conn = duckdb.connect(":memory:")
        with pytest.raises(ValueError, match="identifier"):
            similarity_search(conn, "docs", "queries", query_column="embedding) OR true --")

    def test_identifier_payload_cannot_change_sql_structure(self):
        conn = duckdb.connect(":memory:")
        conn.execute("CREATE TABLE docs (text VARCHAR)")
        with pytest.raises(ValueError, match="identifier"):
            similarity_search(conn, "docs; DROP TABLE docs; --", "queries")
        assert conn.sql("SELECT count(*) FROM docs").fetchone() == (0,)
        conn.close()
