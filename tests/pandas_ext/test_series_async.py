"""Tests for the asynchronous pandas Series accessor (``.aio``)."""

import asyncio
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from pydantic import BaseModel

from openaivec import pandas_ext


@pytest.mark.requires_api
class TestSeriesAsync:
    @pytest.mark.asyncio
    async def test_series_aio_embeddings(self, sample_dataframe):
        embeddings = await sample_dataframe["name"].aio.embeddings()
        assert all(isinstance(embedding, np.ndarray) for embedding in embeddings)
        assert embeddings.shape == (3,)
        assert embeddings.index.equals(sample_dataframe.index)

    @pytest.mark.asyncio
    async def test_series_aio_responses(self, sample_dataframe):
        names_fr = await sample_dataframe["name"].aio.responses("translate to French")
        assert all(isinstance(x, str) for x in names_fr)
        assert names_fr.shape == (3,)
        assert names_fr.index.equals(sample_dataframe.index)

    def test_series_aio_parse(self):
        async def run_test():
            reviews = pd.Series(["Excellent service!", "Poor experience.", "Okay product."])
            return await reviews.aio.parse(
                instructions="Extract sentiment and rating", batch_size=2, max_concurrency=2, show_progress=False
            )

        results = asyncio.run(run_test())
        assert len(results) == 3
        assert all(isinstance(result, (dict, BaseModel)) for result in results)

    def test_series_aio_task(self):
        from openaivec._model import PreparedTask

        async def run_test():
            task = PreparedTask(
                instructions="Classify sentiment as positive or negative",
                response_format=str,
            )
            series = pd.Series(["I love this!", "This is terrible"])
            return await series.aio.task(
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

    def test_shared_cache_async(self):
        from openaivec._cache import AsyncBatchingMapProxy

        async def run_test():
            shared_cache = AsyncBatchingMapProxy(batch_size=32, max_concurrency=4)
            series1 = pd.Series(["cat", "dog", "elephant"])
            series2 = pd.Series(["dog", "elephant", "lion"])
            result1 = await series1.aio.responses_with_cache(instructions="translate to French", cache=shared_cache)
            result2 = await series2.aio.responses_with_cache(instructions="translate to French", cache=shared_cache)
            return result1, result2, series1, series2

        result1, result2, series1, series2 = asyncio.run(run_test())
        assert all(isinstance(x, str) for x in result1)
        assert all(isinstance(x, str) for x in result2)
        assert len(result1) == 3
        assert len(result2) == 3
        dog_idx1 = series1.loc[series1 == "dog"].index[0]
        dog_idx2 = series2.loc[series2 == "dog"].index[0]
        elephant_idx1 = series1.loc[series1 == "elephant"].index[0]
        elephant_idx2 = series2.loc[series2 == "elephant"].index[0]
        assert result1[dog_idx1] == result2[dog_idx2]
        assert result1[elephant_idx1] == result2[elephant_idx2]


@pytest.mark.asyncio
async def test_series_aio_parse_with_cache_forwards_api_kwargs_to_schema_inference(monkeypatch):
    captured: dict[str, object] = {}

    def fake_infer_schema(self, instructions: str, max_examples: int = 100, **api_kwargs):
        captured["infer_kwargs"] = dict(api_kwargs)
        return SimpleNamespace(inference_prompt="inferred prompt", model=str)

    async def fake_responses_with_cache(self, instructions: str, cache, response_format=str, **api_kwargs):
        captured["responses_kwargs"] = dict(api_kwargs)
        captured["instructions"] = instructions
        captured["response_format"] = response_format
        return pd.Series(["ok"] * len(self._obj), index=self._obj.index, name=self._obj.name)

    monkeypatch.setattr(pandas_ext.OpenAIVecSeriesAccessor, "infer_schema", fake_infer_schema)
    monkeypatch.setattr(pandas_ext.AsyncOpenAIVecSeriesAccessor, "responses_with_cache", fake_responses_with_cache)

    series = pd.Series(["a", "b"])
    cache = pandas_ext.AsyncBatchingMapProxy[str, str](batch_size=2, max_concurrency=1, show_progress=False)

    out = await series.aio.parse_with_cache(
        instructions="extract something",
        cache=cache,
        response_format=None,
        temperature=0.4,
        top_p=0.2,
    )

    assert out.tolist() == ["ok", "ok"]
    assert captured["infer_kwargs"] == {"temperature": 0.4, "top_p": 0.2}
    assert captured["responses_kwargs"] == {"temperature": 0.4, "top_p": 0.2}
    assert captured["instructions"] == "inferred prompt"
    assert captured["response_format"] is str
