"""pandas Arrow embedding conversion and DuckDB interoperability tests."""

import duckdb
import numpy as np
import pandas as pd
import pytest

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
