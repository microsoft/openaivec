"""Tests for the asynchronous pandas DataFrame accessor (``.aio``)."""

import asyncio

import pandas as pd
import pytest
from pydantic import BaseModel

from openaivec import pandas_ext


@pytest.mark.requires_api
class TestDataFrameAsync:
    def test_dataframe_aio_responses(self, sample_dataframe):
        async def run():
            return await sample_dataframe.aio.responses("translate the 'name' field to French")

        names_fr = asyncio.run(run())
        assert all(isinstance(x, str) for x in names_fr)
        assert names_fr.shape == (3,)
        assert names_fr.index.equals(sample_dataframe.index)

    def test_dataframe_aio_parse(self):
        async def run_test():
            df = pd.DataFrame(
                [
                    {"text": "Happy customer", "score": 5},
                    {"text": "Unhappy customer", "score": 1},
                    {"text": "Neutral feedback", "score": 3},
                ]
            )
            return await df.aio.parse(
                instructions="Analyze the sentiment", batch_size=2, max_concurrency=2, show_progress=False
            )

        results = asyncio.run(run_test())
        assert len(results) == 3
        assert all(isinstance(result, (dict, BaseModel)) for result in results)

    def test_dataframe_aio_task(self):
        from openaivec._model import PreparedTask

        async def run_test():
            task = PreparedTask(instructions="Describe the animal", response_format=str)
            df = pd.DataFrame([{"name": "fluffy", "type": "cat"}, {"name": "buddy", "type": "dog"}])
            return await df.aio.task(
                task=task,
                batch_size=2,
                max_concurrency=2,
                show_progress=False,
                temperature=0.0,
                top_p=1.0,
            )

        results = asyncio.run(run_test())
        assert len(results) == 2
        assert all(isinstance(result, str) for result in results)

    def test_dataframe_aio_fillna(self):
        async def run_test():
            df_with_missing = pd.DataFrame(
                {
                    "name": ["Alice", "Bob", "Charlie"],
                    "age": [25, 30, 35],
                    "city": ["Tokyo", "Osaka", "Kyoto"],
                }
            )
            return await df_with_missing.aio.fillna("name")

        result, original = (
            asyncio.run(run_test()),
            pd.DataFrame(
                {
                    "name": ["Alice", "Bob", "Charlie"],
                    "age": [25, 30, 35],
                    "city": ["Tokyo", "Osaka", "Kyoto"],
                }
            ),
        )
        pd.testing.assert_frame_equal(result, original)

    def test_dataframe_aio_pipe(self):
        async def run_test():
            df = pd.DataFrame({"name": ["apple", "banana", "cherry"], "color": ["red", "yellow", "red"]})

            def add_column(df):
                df = df.copy()
                df["processed"] = df["name"] + "_processed"
                return df

            result1 = await df.aio.pipe(add_column)

            async def add_async_column(df):
                await asyncio.sleep(0.01)
                df = df.copy()
                df["async_processed"] = df["name"] + "_async"
                return df

            result2 = await df.aio.pipe(add_async_column)
            return result1, result2, df

        result1, result2, original_df = asyncio.run(run_test())
        assert "processed" in result1.columns
        assert len(result1) == 3
        assert result1["processed"].str.endswith("_processed").all()
        assert "async_processed" in result2.columns
        assert len(result2) == 3
        assert result2["async_processed"].str.endswith("_async").all()
        assert "processed" not in original_df.columns
        assert "async_processed" not in original_df.columns

    def test_dataframe_aio_assign(self):
        async def run_test():
            df = pd.DataFrame({"name": ["alice", "bob", "charlie"], "age": [25, 30, 35]})

            def compute_category(df):
                return ["young" if age < 30 else "adult" for age in df["age"]]

            result1 = await df.aio.assign(category=compute_category)

            async def compute_async_score(df):
                await asyncio.sleep(0.01)
                return [age * 2 for age in df["age"]]

            result2 = await df.aio.assign(score=compute_async_score)
            return result1, result2, df

        result1, result2, original_df = asyncio.run(run_test())
        assert "category" in result1.columns
        assert list(result1["category"]) == ["young", "adult", "adult"]
        assert "score" in result2.columns
        assert list(result2["score"]) == [50, 60, 70]
        assert "category" not in original_df.columns
        assert "score" not in original_df.columns


@pytest.mark.asyncio
async def test_dataframe_aio_fillna_fills_only_missing_rows_in_index_order(monkeypatch):
    from openaivec._model import PreparedTask
    from openaivec.task.table import FillNaResponse

    def fake_fillna(df, target_column_name, max_examples=500):
        assert target_column_name == "name"
        return PreparedTask(instructions="stub fillna", response_format=FillNaResponse)

    async def fake_task(self, task, batch_size=None, max_concurrency=8, show_progress=True, **api_kwargs):
        assert task.instructions == "stub fillna"
        assert self._obj.index.tolist() == [1, 3]
        return [
            FillNaResponse(index=100, output="Charlie"),
            FillNaResponse(index=200, output="Dana"),
        ]

    monkeypatch.setattr(pandas_ext, "fillna", fake_fillna)
    monkeypatch.setattr(pandas_ext.AsyncOpenAIVecDataFrameAccessor, "task", fake_task)

    df = pd.DataFrame(
        {"name": [None, "Bob", None], "age": [21, 32, 45]},
        index=[1, 2, 3],
    )
    result = await df.aio.fillna("name")
    assert result.loc[1, "name"] == "Charlie"
    assert result.loc[2, "name"] == "Bob"
    assert result.loc[3, "name"] == "Dana"
    assert pd.isna(df.loc[1, "name"])
    assert pd.isna(df.loc[3, "name"])


@pytest.mark.asyncio
async def test_dataframe_aio_assign_supports_chained_column_dependencies():
    from openaivec import pandas_ext  # noqa: F401

    df = pd.DataFrame({"base": [1, 2, 3]})

    async def make_double(current: pd.DataFrame):
        await asyncio.sleep(0)
        return current["base"] * 2

    def make_plus_one(current: pd.DataFrame):
        return current["double"] + 1

    out = await df.aio.assign(double=make_double, plus_one=make_plus_one)
    assert out["double"].tolist() == [2, 4, 6]
    assert out["plus_one"].tolist() == [3, 5, 7]
