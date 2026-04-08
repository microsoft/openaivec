"""Tests for the synchronous pandas DataFrame accessor (``.ai``)."""

import json

import numpy as np
import pandas as pd
import pytest
from pydantic import BaseModel

from openaivec import pandas_ext


@pytest.mark.requires_api
class TestDataFrameSync:
    def test_dataframe_responses(self, sample_dataframe):
        names_fr = sample_dataframe.ai.responses("translate to French")
        assert all(isinstance(x, str) for x in names_fr)
        assert names_fr.shape == (3,)
        assert names_fr.index.equals(sample_dataframe.index)

    def test_dataframe_parse(self, review_dataframe):
        results = review_dataframe.ai.parse(
            instructions="Extract sentiment from the review", batch_size=2, show_progress=False
        )
        assert len(results) == 3
        assert results.index.equals(review_dataframe.index)
        assert all(isinstance(result, (dict, BaseModel)) for result in results)

    def test_dataframe_infer_schema(self):
        df = pd.DataFrame(
            [
                {"product": "laptop", "review": "Great performance", "rating": 5},
                {"product": "mouse", "review": "Poor quality", "rating": 2},
                {"product": "keyboard", "review": "Average product", "rating": 3},
            ]
        )
        schema = df.ai.infer_schema(instructions="Extract product analysis metrics", max_examples=2)
        assert schema is not None
        assert schema.model is not None
        assert schema.task is not None
        assert schema.object_spec.fields is not None
        assert isinstance(schema.object_spec.fields, list)
        assert len(schema.object_spec.fields) > 0

    def test_dataframe_task(self):
        from openaivec._model import PreparedTask

        task = PreparedTask(instructions="Extract the animal name from the data", response_format=str)
        df = pd.DataFrame([{"animal": "cat", "legs": 4}, {"animal": "dog", "legs": 4}])
        results = df.ai.task(task=task, batch_size=2, show_progress=False, temperature=0.0, top_p=1.0)
        assert len(results) == 2
        assert results.index.equals(df.index)
        assert all(isinstance(result, str) for result in results)

    def test_dataframe_similarity(self, vector_dataframe):
        similarity_scores = vector_dataframe.ai.similarity("vector1", "vector2")
        expected_scores = [1.0, 1.0, 0.0]
        assert np.allclose(similarity_scores, expected_scores)

    def test_dataframe_similarity_invalid_vectors(self):
        df = pd.DataFrame(
            {
                "vector1": [np.array([1, 0]), "invalid", np.array([1, 1])],
                "vector2": [np.array([1, 0]), np.array([0, 1]), np.array([1, -1])],
            }
        )
        with pytest.raises(TypeError):
            df.ai.similarity("vector1", "vector2")

    def test_dataframe_fillna(self):
        df_complete = pd.DataFrame(
            {
                "name": ["Alice", "Bob", "Charlie", "David"],
                "age": [25, 30, 35, 40],
                "city": ["Tokyo", "Osaka", "Kyoto", "Tokyo"],
            }
        )
        result_df = df_complete.ai.fillna("name")
        pd.testing.assert_frame_equal(result_df, df_complete)

        df_custom_index = pd.DataFrame(
            {"name": ["Alice", "Bob", "Charlie"], "score": [85, 90, 78]},
            index=["student_1", "student_2", "student_3"],
        )
        result_df = df_custom_index.ai.fillna("name")
        pd.testing.assert_index_equal(result_df.index, df_custom_index.index)
        assert result_df.shape == df_custom_index.shape

    def test_dataframe_extract_pydantic(self, fruit_model):
        sample_df = pd.DataFrame(
            [
                {"name": "apple", "fruit": fruit_model(name="apple", color="red", taste="crunchy")},
                {"name": "banana", "fruit": fruit_model(name="banana", color="yellow", taste="soft")},
                {"name": "cherry", "fruit": fruit_model(name="cherry", color="red", taste="tart")},
            ]
        ).ai.extract("fruit")
        expected_columns = ["name", "fruit_name", "fruit_color", "fruit_taste"]
        assert list(sample_df.columns) == expected_columns

    def test_dataframe_extract_dict(self):
        sample_df = pd.DataFrame(
            [
                {"fruit": {"name": "apple", "color": "red", "flavor": "sweet", "taste": "crunchy"}},
                {"fruit": {"name": "banana", "color": "yellow", "flavor": "sweet", "taste": "soft"}},
                {"fruit": {"name": "cherry", "color": "red", "flavor": "sweet", "taste": "tart"}},
            ]
        ).ai.extract("fruit")
        expected_columns = ["fruit_name", "fruit_color", "fruit_flavor", "fruit_taste"]
        assert list(sample_df.columns) == expected_columns

    def test_dataframe_extract_dict_with_none(self):
        sample_df = pd.DataFrame(
            [
                {"fruit": {"name": "apple", "color": "red", "flavor": "sweet", "taste": "crunchy"}},
                {"fruit": None},
                {"fruit": {"name": "cherry", "color": "red", "flavor": "sweet", "taste": "tart"}},
            ]
        ).ai.extract("fruit")
        expected_columns = ["fruit_name", "fruit_color", "fruit_flavor", "fruit_taste"]
        assert list(sample_df.columns) == expected_columns
        assert sample_df.iloc[1].isna().all()

    def test_dataframe_extract_with_invalid_row(self):
        sample_df = pd.DataFrame(
            [
                {"fruit": {"name": "apple", "color": "red", "flavor": "sweet", "taste": "crunchy"}},
                {"fruit": 123},
                {"fruit": {"name": "cherry", "color": "red", "flavor": "sweet", "taste": "tart"}},
            ]
        )
        expected_columns = ["fruit"]
        assert list(sample_df.columns) == expected_columns

    def test_parse_with_cache_methods(self):
        from openaivec._cache import BatchingMapProxy

        df = pd.DataFrame(
            [
                {"review": "Great product", "rating": 5},
                {"review": "Poor quality", "rating": 1},
            ]
        )
        cache = BatchingMapProxy(batch_size=2)
        df_results = df.ai.parse_with_cache(instructions="Analyze sentiment", cache=cache)
        assert len(df_results) == 2
        assert all(isinstance(result, (dict, BaseModel)) for result in df_results)

    def test_empty_dataframe_handling(self):
        empty_df = pd.DataFrame()
        assert empty_df.empty

    def test_fillna_task_creation(self):
        from openaivec.task.table import fillna

        df_with_missing = pd.DataFrame(
            {
                "name": ["Alice", "Bob", None, "David"],
                "age": [25, 30, 35, 40],
                "city": ["Tokyo", "Osaka", "Kyoto", "Tokyo"],
            }
        )
        task = fillna(df_with_missing, "name")
        assert task is not None
        assert isinstance(task.instructions, str)
        assert task.response_format.__name__ == "FillNaResponse"
        with pytest.raises(AttributeError):
            _ = getattr(task, "api_kwargs")

    def test_fillna_task_validation(self):
        from openaivec.task.table import fillna

        empty_df = pd.DataFrame()
        with pytest.raises(ValueError):
            fillna(empty_df, "nonexistent")
        df = pd.DataFrame({"name": ["Alice", "Bob"]})
        with pytest.raises(ValueError):
            fillna(df, "nonexistent")
        df_all_null = pd.DataFrame({"name": [None, None, None], "age": [25, 30, 35]})
        with pytest.raises(ValueError):
            fillna(df_all_null, "name")
        df_valid = pd.DataFrame({"name": ["Alice", None, "Bob"], "age": [25, 30, 35]})
        with pytest.raises(ValueError):
            fillna(df_valid, "name", max_examples=0)
        with pytest.raises(ValueError):
            fillna(df_valid, "name", max_examples=-1)

    def test_fillna_missing_rows_detection(self):
        df_with_missing = pd.DataFrame(
            {
                "name": ["Alice", "Bob", None, "David", None],
                "age": [25, 30, 35, 40, 45],
                "city": ["Tokyo", "Osaka", "Kyoto", "Tokyo", "Nagoya"],
            }
        )
        missing_rows = df_with_missing[df_with_missing["name"].isna()]
        assert len(missing_rows) == 2
        assert missing_rows.index.tolist() == [2, 4]


def test_dataframe_similarity_zero_norm_returns_nan():
    from openaivec import pandas_ext  # noqa: F401

    df = pd.DataFrame(
        {
            "left": [np.array([0.0, 0.0]), np.array([1.0, 0.0])],
            "right": [np.array([1.0, 0.0]), np.array([0.0, 1.0])],
        }
    )
    result = df.ai.similarity("left", "right")
    assert np.isnan(result.iloc[0])
    assert result.iloc[1] == pytest.approx(0.0)


def test_df_rows_to_json_series_serializes_timestamp():
    from openaivec.pandas_ext._common import _df_rows_to_json_series

    df = pd.DataFrame(
        {"ts": [pd.Timestamp("2024-01-01T12:34:56"), pd.Timestamp("2024-01-02T00:00:00")], "x": [1, 2]},
        index=["a", "b"],
    )
    out = _df_rows_to_json_series(df)
    assert out.index.equals(df.index)
    assert out.name == "record"
    assert json.loads(out.iloc[0]) == {"ts": "2024-01-01T12:34:56", "x": 1}
    assert json.loads(out.iloc[1]) == {"ts": "2024-01-02T00:00:00", "x": 2}


def test_df_rows_to_json_series_serializes_numpy_scalars():
    from openaivec.pandas_ext._common import _df_rows_to_json_series

    df = pd.DataFrame({"x": [np.int64(7)], "y": [np.float32(1.5)]})
    out = _df_rows_to_json_series(df)
    assert json.loads(out.iloc[0]) == {"x": 7, "y": pytest.approx(1.5)}


def test_df_rows_to_json_series_serializes_string_values():
    from openaivec.pandas_ext._common import _df_rows_to_json_series

    df = pd.DataFrame({"s1": ["hello"], "s2": [np.str_("world")]})
    out = _df_rows_to_json_series(df)
    assert json.loads(out.iloc[0]) == {"s1": "hello", "s2": "world"}


def test_dataframe_fillna_no_missing_skips_task_construction(monkeypatch):
    def fail_fillna(*args, **kwargs):
        raise AssertionError("fillna task builder must not be called when there are no missing rows")

    monkeypatch.setattr(pandas_ext, "fillna", fail_fillna)
    df = pd.DataFrame({"name": ["Alice", "Bob"], "age": [20, 30]})
    result = df.ai.fillna("name")
    pd.testing.assert_frame_equal(result, df)


def test_dataframe_fillna_fills_only_missing_rows_in_index_order(monkeypatch):
    from openaivec._model import PreparedTask
    from openaivec.task.table import FillNaResponse

    def fake_fillna(df, target_column_name, max_examples=500):
        assert target_column_name == "name"
        assert max_examples == 500
        return PreparedTask(instructions="stub fillna", response_format=FillNaResponse)

    def fake_task(self, task, batch_size=None, show_progress=True, **api_kwargs):
        assert task.instructions == "stub fillna"
        assert self._obj.index.tolist() == [10, 40]
        return [
            FillNaResponse(index=999, output="Carol"),
            FillNaResponse(index=888, output=None),
        ]

    monkeypatch.setattr(pandas_ext, "fillna", fake_fillna)
    monkeypatch.setattr(pandas_ext.OpenAIVecDataFrameAccessor, "task", fake_task)

    df = pd.DataFrame(
        {"name": [None, "Bob", "Alice", None], "age": [18, 25, 31, 44]},
        index=[10, 20, 30, 40],
    )
    result = df.ai.fillna("name")
    assert pd.isna(df.loc[10, "name"])
    assert pd.isna(df.loc[40, "name"])
    assert result.loc[10, "name"] == "Carol"
    assert pd.isna(result.loc[40, "name"])
    assert result.loc[20, "name"] == "Bob"
    assert result.loc[30, "name"] == "Alice"
