from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import numpy as np
import pandas as pd
import pytest
import tiktoken
from openai import AsyncOpenAI, OpenAI

from openaivec import EmbeddingLimits, duckdb_ext, pandas_ext
from openaivec._cache import AsyncBatchCache, BatchCache
from openaivec._cache.optimize import BatchSizeSuggester
from openaivec._embeddings import AsyncBatchEmbeddings, BatchEmbeddings, _plan_embedding_batches
from openaivec._provider import CONTAINER


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("batch_size", [None, -1, 0, 128, 4096])
async def test_embedding_item_limits_preserve_order(monkeypatch, asynchronous, batch_size):
    monkeypatch.setattr(BatchSizeSuggester, "suggest_batch_size", lambda self: 4096)
    texts = [f"text {index}" for index in range(2049)]
    await _assert_embedding_requests(texts, asynchronous, batch_size, "text-embedding-3-small")


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("model_name", ["text-embedding-3-small", "custom-deployment"])
async def test_embedding_token_budget(asynchronous, model_name):
    texts = [" a" * 8100 + f" {index}" for index in range(38)]
    await _assert_embedding_requests(texts, asynchronous, 0, model_name)


async def _assert_embedding_requests(texts, asynchronous, batch_size, model_name):
    encoding = tiktoken.get_encoding("cl100k_base")
    values = {text: index for index, text in enumerate(texts)}

    def respond(**kwargs):
        inputs = kwargs["input"]
        assert 0 < len(inputs) <= 2048
        assert sum(len(encoding.encode_ordinary(text)) for text in inputs) <= 300000
        assert kwargs["dimensions"] == 1
        return SimpleNamespace(
            data=[
                SimpleNamespace(index=index, embedding=[values[text]])
                for index, text in reversed(list(enumerate(inputs)))
            ]
        )

    create = AsyncMock(side_effect=respond) if asynchronous else Mock(side_effect=respond)
    client = Mock(spec=AsyncOpenAI if asynchronous else OpenAI)
    client.embeddings = Mock(create=create)
    wrapper = AsyncBatchEmbeddings if asynchronous else BatchEmbeddings
    embedder = wrapper.of(client, model_name, batch_size=batch_size, dimensions=1)
    inputs = texts + [texts[0], texts[-1]]
    result = embedder.create(inputs)
    if asynchronous:
        result = await result
    assert len(result) == len(inputs)
    assert [row.tolist() for row in result] == [[values[text]] for text in inputs]
    assert all(row.dtype == np.float32 for row in result)
    assert create.call_count >= 2
    assert sum(len(call.kwargs["input"]) for call in create.call_args_list) == len(texts)


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("invalid", ["", " a" * 8193], ids=["empty", "too-long"])
async def test_embedding_invalid_input_is_rejected_before_request(asynchronous, invalid):
    create = AsyncMock() if asynchronous else Mock()
    client = Mock(spec=AsyncOpenAI if asynchronous else OpenAI)
    client.embeddings = Mock(create=create)
    wrapper = AsyncBatchEmbeddings if asynchronous else BatchEmbeddings
    embedder = wrapper.of(client, "text-embedding-3-small", batch_size=0)
    with pytest.raises(ValueError, match="input.*(empty|8192)"):
        result = embedder.create(["valid", invalid])
        if asynchronous:
            await result
    create.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("max_inputs,max_tokens,expected_sizes", [(2, 10, [2, 2, 1]), (10, 3, [1] * 5)])
async def test_embedding_provider_limits(asynchronous, max_inputs, max_tokens, expected_sizes):
    def respond(**kwargs):
        assert "limits" not in kwargs
        return SimpleNamespace(
            data=[SimpleNamespace(index=index, embedding=[1.0]) for index in range(len(kwargs["input"]))]
        )

    create = AsyncMock(side_effect=respond) if asynchronous else Mock(side_effect=respond)
    client = Mock(spec=AsyncOpenAI if asynchronous else OpenAI)
    client.embeddings = Mock(create=create)
    wrapper = AsyncBatchEmbeddings if asynchronous else BatchEmbeddings
    limits = EmbeddingLimits(
        max_inputs=max_inputs, max_input_tokens=8, max_request_tokens=max_tokens, encoding_name="cl100k_base"
    )
    embedder = wrapper.of(client, "custom-provider", batch_size=0, limits=limits)
    result = embedder.create([f"value{index}" for index in range(5)])
    if asynchronous:
        result = await result
    assert len(result) == 5
    assert [len(call.kwargs["input"]) for call in create.call_args_list] == expected_sizes


@pytest.mark.parametrize("field_name", ["max_inputs", "max_input_tokens", "max_request_tokens"])
@pytest.mark.parametrize("invalid", [0, -1])
def test_embedding_limits_require_positive_values(field_name, invalid):
    with pytest.raises(ValueError, match=field_name):
        EmbeddingLimits(**{field_name: invalid})


@pytest.mark.parametrize("field_name", ["max_inputs", "max_input_tokens", "max_request_tokens"])
@pytest.mark.parametrize("invalid", [True, 2.5])
def test_embedding_limits_require_integers(field_name, invalid):
    with pytest.raises(TypeError, match=field_name):
        EmbeddingLimits(**{field_name: invalid})


@pytest.mark.parametrize("count,sizes", [(2048, [2048]), (2049, [2048, 1])])
def test_embedding_exact_item_boundary(count, sizes):
    batches = _plan_embedding_batches(
        [f"value{index}" for index in range(count)], "text-embedding-3-small", EmbeddingLimits()
    )
    assert [len(batch) for batch in batches] == sizes


def test_embedding_exact_token_boundaries():
    texts = [" a" * 8192] * 36 + [" a" * 5088, " a"]
    batches = _plan_embedding_batches(texts, "text-embedding-3-small", EmbeddingLimits())
    assert batches == [texts[:-1], texts[-1:]]


def test_embedding_variable_input_lengths():
    texts = [" a" * count for count in [2, 8, 1, 9]]
    batches = _plan_embedding_batches(texts, "text-embedding-3-small", EmbeddingLimits(max_request_tokens=10))
    assert batches == [texts[:2], texts[2:]]
    with pytest.raises(ValueError, match="exceeds 10"):
        _plan_embedding_batches([" a" * 11], "text-embedding-3-small", EmbeddingLimits(max_request_tokens=10))


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("indices", [[0, 0], [0], [0, 2], [0, 1, 2]])
async def test_embedding_invalid_response_indices(asynchronous, indices):
    invalid = SimpleNamespace(data=[SimpleNamespace(index=index, embedding=[index]) for index in indices])
    valid = SimpleNamespace(data=[SimpleNamespace(index=index, embedding=[index]) for index in [0, 1]])
    create = AsyncMock(side_effect=[invalid, valid]) if asynchronous else Mock(side_effect=[invalid, valid])
    client = Mock(spec=AsyncOpenAI if asynchronous else OpenAI)
    client.embeddings = Mock(create=create)
    wrapper = AsyncBatchEmbeddings if asynchronous else BatchEmbeddings
    embedder = wrapper.of(client, "text-embedding-3-small", batch_size=0)
    with pytest.raises(ValueError, match="indices"):
        result = embedder.create(["first", "second"])
        if asynchronous:
            await result
    result = embedder.create(["first", "second"])
    if asynchronous:
        result = await result
    assert [row.tolist() for row in result] == [[0], [1]]
    assert create.call_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("with_cache", [False, True])
async def test_pandas_embedding_provider_limits(monkeypatch, asynchronous, with_cache):
    values = {"first": 1, "second": 2, "third": 3}

    def respond(**kwargs):
        assert "limits" not in kwargs
        return SimpleNamespace(
            data=[SimpleNamespace(index=index, embedding=[values[text]]) for index, text in enumerate(kwargs["input"])]
        )

    create = AsyncMock(side_effect=respond) if asynchronous else Mock(side_effect=respond)
    client = Mock(spec=AsyncOpenAI if asynchronous else OpenAI)
    client.embeddings = Mock(create=create)
    resolve = CONTAINER.resolve
    monkeypatch.setattr(CONTAINER, "resolve", lambda kind: client if kind in (OpenAI, AsyncOpenAI) else resolve(kind))
    series = pd.Series(["first", "second", "third", "first"], index=[9, 3, 7, 2], name="value")
    accessor_type = pandas_ext.AsyncOpenAIVecSeriesAccessor if asynchronous else pandas_ext.OpenAIVecSeriesAccessor
    accessor = accessor_type(series)
    options = {"limits": EmbeddingLimits(max_inputs=2)}
    if with_cache:
        cache_type = AsyncBatchCache if asynchronous else BatchCache
        options["cache"] = cache_type(batch_size=0, show_progress=False)
    else:
        options.update(batch_size=0, show_progress=False)
    method = accessor.embeddings_with_cache if with_cache else accessor.embeddings
    result = method(**options)
    if asynchronous:
        result = await result
    assert result.index.equals(series.index)
    assert result.name == series.name
    assert result.tolist() == [[1], [2], [3], [1]]
    assert [len(call.kwargs["input"]) for call in create.call_args_list] == [2, 1]


@pytest.mark.parametrize("integration", ["duckdb", "spark"])
def test_udf_embedding_provider_limits(monkeypatch, integration):
    def respond(**kwargs):
        assert "limits" not in kwargs
        return SimpleNamespace(
            data=[SimpleNamespace(index=index, embedding=[1.0]) for index in range(len(kwargs["input"]))]
        )

    create = AsyncMock(side_effect=respond)
    client = Mock(spec=AsyncOpenAI)
    client.embeddings = Mock(create=create)
    resolve = CONTAINER.resolve
    monkeypatch.setattr(CONTAINER, "resolve", lambda kind: client if kind is AsyncOpenAI else resolve(kind))
    limits = EmbeddingLimits(max_inputs=2)
    if integration == "duckdb":
        import duckdb

        with duckdb.connect() as connection:
            duckdb_ext.embeddings_udf(connection, "embed", batch_size=0, limits=limits)
            connection.execute("CREATE TABLE inputs(value VARCHAR)")
            connection.execute("INSERT INTO inputs VALUES ('first'), ('second'), ('third'), ('first')")
            assert connection.sql("SELECT embed(value) FROM inputs").fetchall() == [([1.0],)] * 4
    else:
        pytest.importorskip("pyspark")
        from openaivec import spark_ext

        monkeypatch.setattr(spark_ext, "pandas_udf", lambda **kwargs: lambda function: function)
        udf = spark_ext.embeddings_udf(batch_size=0, limits=limits)
        result = list(udf(iter([pd.Series(["first", "second", "third", "first"])])))
        assert len(result) == 1
        assert len(result[0]) == 4
    assert [len(call.kwargs["input"]) for call in create.call_args_list] == [2, 1]
