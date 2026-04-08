"""Tests for openaivec.duckdb_ext (DDL, similarity, struct UDF type mapping)."""

from __future__ import annotations

from enum import Enum

import duckdb
import numpy as np
import pandas as pd
import pytest
from pydantic import BaseModel

from openaivec.duckdb_ext import (
    _pydantic_to_struct_type,
    _python_type_to_duckdb,
    _serialize_for_duckdb,
    pydantic_to_duckdb_ddl,
    similarity_search,
)

# ---------------------------------------------------------------------------
# pydantic_to_duckdb_ddl
# ---------------------------------------------------------------------------


class SimpleModel(BaseModel):
    name: str
    age: int
    score: float
    active: bool


class NestedModel(BaseModel):
    label: str
    tags: list[str]


class ModelWithOptional(BaseModel):
    required_field: str
    optional_field: str | None = None


class ModelWithNested(BaseModel):
    info: NestedModel
    count: int


class TestPydanticToDuckDBDDL:
    def test_simple_model(self):
        ddl = pydantic_to_duckdb_ddl(SimpleModel, "simple")
        assert "CREATE TABLE IF NOT EXISTS simple" in ddl
        assert "name VARCHAR" in ddl
        assert "age INTEGER" in ddl
        assert "score DOUBLE" in ddl
        assert "active BOOLEAN" in ddl

    def test_list_field(self):
        ddl = pydantic_to_duckdb_ddl(NestedModel, "nested")
        assert "label VARCHAR" in ddl
        assert "tags VARCHAR[]" in ddl

    def test_optional_field(self):
        ddl = pydantic_to_duckdb_ddl(ModelWithOptional, "opt")
        assert "required_field VARCHAR" in ddl
        assert "optional_field VARCHAR" in ddl

    def test_nested_struct(self):
        ddl = pydantic_to_duckdb_ddl(ModelWithNested, "with_nested")
        assert "info STRUCT" in ddl
        assert "count INTEGER" in ddl

    def test_ddl_is_executable(self):
        """Verify the generated DDL runs without error in DuckDB."""
        conn = duckdb.connect(":memory:")
        ddl = pydantic_to_duckdb_ddl(SimpleModel, "test_table")
        conn.execute(ddl)
        result = conn.execute("SELECT * FROM test_table").fetchall()
        assert result == []
        conn.close()

    def test_list_field_ddl_is_executable(self):
        conn = duckdb.connect(":memory:")
        ddl = pydantic_to_duckdb_ddl(NestedModel, "tag_table")
        conn.execute(ddl)
        conn.execute("INSERT INTO tag_table VALUES ('test', ['a', 'b'])")
        result = conn.execute("SELECT * FROM tag_table").fetchone()
        assert result[0] == "test"
        assert result[1] == ["a", "b"]
        conn.close()


# ---------------------------------------------------------------------------
# similarity_search
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Pydantic → DuckDB STRUCT type mapping
# ---------------------------------------------------------------------------


class SentimentResult(BaseModel):
    label: str
    score: float


class ReviewResult(BaseModel):
    sentiment: str
    rating: int
    tags: list[str]


class TestPydanticToStructType:
    def test_simple_struct(self):
        st = _pydantic_to_struct_type(SentimentResult)
        st_str = str(st)
        assert "VARCHAR" in st_str
        assert "DOUBLE" in st_str

    def test_struct_with_list(self):
        st = _pydantic_to_struct_type(ReviewResult)
        st_str = str(st)
        assert "INTEGER" in st_str
        assert "VARCHAR[]" in st_str

    def test_struct_udf_field_access(self):
        conn = duckdb.connect(":memory:")
        st = _pydantic_to_struct_type(SentimentResult)

        def mock_fn(x: str) -> dict:
            return {"label": "positive", "score": 0.95}

        conn.create_function("get_sentiment", mock_fn, [duckdb.sqltype("VARCHAR")], st)

        row = conn.sql("SELECT get_sentiment('hello').label AS label, get_sentiment('hello').score AS score").fetchone()
        assert row[0] == "positive"
        assert row[1] == pytest.approx(0.95)
        conn.close()

    def test_struct_udf_in_table(self):
        conn = duckdb.connect(":memory:")
        st = _pydantic_to_struct_type(SentimentResult)

        def mock_fn(x: str) -> dict:
            return {"label": "pos" if "good" in x else "neg", "score": 0.9}

        conn.create_function("classify", mock_fn, [duckdb.sqltype("VARCHAR")], st)
        conn.execute("CREATE TABLE texts (t VARCHAR)")
        conn.execute("INSERT INTO texts VALUES ('good day'), ('bad day')")

        df = conn.sql("SELECT t, classify(t).label AS label FROM texts").df()
        assert len(df) == 2
        assert df[df["t"] == "good day"].iloc[0]["label"] == "pos"
        assert df[df["t"] == "bad day"].iloc[0]["label"] == "neg"
        conn.close()


# ---------------------------------------------------------------------------
# Enum / Literal type mapping
# ---------------------------------------------------------------------------


class StrColor(str, Enum):
    RED = "red"
    GREEN = "green"


class IntPriority(int, Enum):
    LOW = 1
    HIGH = 2


class WithEnums(BaseModel):
    label: str
    color: StrColor
    priority: IntPriority


class TestEnumTypeMapping:
    def test_str_enum_maps_to_varchar(self):
        assert _python_type_to_duckdb(StrColor) == "VARCHAR"

    def test_int_enum_maps_to_integer(self):
        assert _python_type_to_duckdb(IntPriority) == "INTEGER"

    def test_struct_with_enums(self):
        st = _pydantic_to_struct_type(WithEnums)
        st_str = str(st)
        assert "INTEGER" in st_str
        assert "VARCHAR" in st_str

    def test_serialize_converts_enum_values(self):
        raw = {"label": "test", "color": StrColor.RED, "priority": IntPriority.HIGH}
        result = _serialize_for_duckdb(raw)
        assert result == {"label": "test", "color": "red", "priority": 2}

    def test_serialize_handles_nested_enums(self):
        raw = {"items": [StrColor.RED, StrColor.GREEN], "nested": {"p": IntPriority.LOW}}
        result = _serialize_for_duckdb(raw)
        assert result == {"items": ["red", "green"], "nested": {"p": 1}}

    def test_enum_struct_udf(self):
        conn = duckdb.connect(":memory:")
        st = _pydantic_to_struct_type(WithEnums)

        def mock(x: str) -> dict:
            return _serialize_for_duckdb({"label": x, "color": StrColor.GREEN, "priority": IntPriority.HIGH})

        conn.create_function("with_enums", mock, [duckdb.sqltype("VARCHAR")], st)
        row = conn.sql("SELECT with_enums('test').priority, with_enums('test').color").fetchone()
        assert row == (2, "green")
        conn.close()


# ---------------------------------------------------------------------------
# count_tokens batch optimization
# ---------------------------------------------------------------------------


class TestCountTokensBatch:
    def test_count_tokens_returns_correct_counts(self):

        from openaivec import pandas_ext  # noqa: F401 — registers .ai accessor
        from openaivec._provider import ensure_default_registrations

        ensure_default_registrations()

        s = pd.Series(["hello", "hello world", ""])
        result = s.ai.count_tokens()
        assert len(result) == 3
        assert result.iloc[0] > 0
        assert result.iloc[1] > result.iloc[0]
        assert result.iloc[2] == 0

    def test_count_tokens_preserves_index(self):

        from openaivec import pandas_ext  # noqa: F401 — registers .ai accessor
        from openaivec._provider import ensure_default_registrations

        ensure_default_registrations()

        s = pd.Series(["a", "bb", "ccc"], index=[10, 20, 30])
        result = s.ai.count_tokens()
        assert list(result.index) == [10, 20, 30]
        assert result.name == "num_tokens"


# ---------------------------------------------------------------------------
# Arrow embedding edge cases
# ---------------------------------------------------------------------------


class TestArrowEmbeddings:
    def test_embeddings_to_series_creates_arrow_dtype(self):

        from openaivec.pandas_ext._common import _embeddings_to_series

        vecs = [np.array([1.0, 2.0], dtype=np.float32), np.array([3.0, 4.0], dtype=np.float32)]
        s = _embeddings_to_series(vecs, index=pd.RangeIndex(2))
        assert "pyarrow" in str(s.dtype) or "arrow" in str(s.dtype).lower()
        assert len(s) == 2

    def test_embeddings_to_series_empty(self):

        from openaivec.pandas_ext._common import _embeddings_to_series

        s = _embeddings_to_series([], index=pd.RangeIndex(0))
        assert len(s) == 0

    def test_embedding_series_to_matrix_arrow(self):

        from openaivec.pandas_ext._common import _embedding_series_to_matrix, _embeddings_to_series

        vecs = [np.array([1.0, 0.0, 0.0], dtype=np.float32), np.array([0.0, 1.0, 0.0], dtype=np.float32)]
        s = _embeddings_to_series(vecs, index=pd.RangeIndex(2))
        matrix = _embedding_series_to_matrix(s)
        assert matrix.shape == (2, 3)
        assert matrix.dtype == np.float32
        np.testing.assert_array_almost_equal(matrix[0], [1.0, 0.0, 0.0])

    def test_embedding_series_to_matrix_object(self):

        from openaivec.pandas_ext._common import _embedding_series_to_matrix

        s = pd.Series([np.array([1.0, 2.0]), np.array([3.0, 4.0])])
        matrix = _embedding_series_to_matrix(s)
        assert matrix.shape == (2, 2)

    def test_similarity_with_arrow_embeddings(self):

        from openaivec import pandas_ext  # noqa: F401
        from openaivec.pandas_ext._common import _embeddings_to_series

        v1 = [np.array([1.0, 0.0], dtype=np.float32), np.array([0.0, 1.0], dtype=np.float32)]
        v2 = [np.array([1.0, 0.0], dtype=np.float32), np.array([1.0, 0.0], dtype=np.float32)]
        idx = pd.RangeIndex(2)
        df = pd.DataFrame({"a": _embeddings_to_series(v1, index=idx), "b": _embeddings_to_series(v2, index=idx)})
        sim = df.ai.similarity("a", "b")
        assert len(sim) == 2
        assert sim.iloc[0] == pytest.approx(1.0, abs=1e-5)
        assert sim.iloc[1] == pytest.approx(0.0, abs=1e-5)

    def test_arrow_embeddings_to_duckdb(self):

        from openaivec.pandas_ext._common import _embeddings_to_series

        vecs = [np.array([1.0, 0.0, 0.0], dtype=np.float32), np.array([0.7, 0.7, 0.0], dtype=np.float32)]
        s = _embeddings_to_series(vecs, index=pd.RangeIndex(2))
        emb_df = pd.DataFrame({"text": ["a", "b"], "emb": s})  # noqa: F841 — referenced by DuckDB SQL

        conn = duckdb.connect(":memory:")
        result = conn.sql("SELECT text, list_cosine_similarity(emb, [1,0,0]::FLOAT[3]) AS sim FROM emb_df").df()
        assert result.iloc[0]["sim"] == pytest.approx(1.0, abs=1e-5)
        conn.close()
