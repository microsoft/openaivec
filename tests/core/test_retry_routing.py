import inspect

import duckdb
import httpx
import pandas as pd
import pytest
from openai import AsyncOpenAI, OpenAI, RateLimitError

from openaivec import PreparedTask, RetryPolicy, SchemaInferer, duckdb_ext
from openaivec import pandas_ext as pandas_ext
from openaivec._cache import AsyncBatchCache, BatchCache
from openaivec._provider import CONTAINER

_METHODS = ["responses", "responses_with_cache", "task", "task_with_cache", "parse", "parse_with_cache"]
_PANDAS_CASES = (
    [
        (asynchronous, dataframe, method)
        for asynchronous in (False, True)
        for dataframe in (False, True)
        for method in _METHODS
    ]
    + [
        (asynchronous, False, method)
        for asynchronous in (False, True)
        for method in ("embeddings", "embeddings_with_cache", "infer_schema")
    ]
    + [(False, True, "infer_schema")]
)


def _assert_policy_parameter(method):
    parameter = inspect.signature(method).parameters["retry_policy"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is None
    assert "retry_policy (RetryPolicy | None)" in inspect.getdoc(method)


def _configure_client(monkeypatch, client):
    resolve = CONTAINER.resolve

    def resolve_client(kind):
        if kind in (OpenAI, AsyncOpenAI):
            return client
        if kind is SchemaInferer:
            return SchemaInferer(client=client, model_name="test-model")
        return resolve(kind)

    monkeypatch.setattr(CONTAINER, "resolve", resolve_client)


def _options(method, *, inferred=False):
    options = {"retry_policy": RetryPolicy(max_attempts=2, initial_delay=0)}
    if method.startswith("task"):
        options["task"] = PreparedTask(instructions="echo", response_format=str)
    elif not method.startswith("embeddings"):
        options["instructions"] = "echo"
        if method != "infer_schema" and not inferred:
            options["response_format"] = str
    return options


@pytest.fixture
def failed_transport():
    requests = []

    def handler(request):
        requests.append(request)
        assert b"retry_policy" not in request.content
        return httpx.Response(429, json={"error": {"message": "forced failure"}})

    return httpx.MockTransport(handler), requests


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous,dataframe,method_name", _PANDAS_CASES)
async def test_pandas_retry_policy(monkeypatch, failed_transport, asynchronous, dataframe, method_name):
    transport, requests = failed_transport
    client = (
        AsyncOpenAI(api_key="test", max_retries=7, http_client=httpx.AsyncClient(transport=transport))
        if asynchronous
        else OpenAI(api_key="test", max_retries=7, http_client=httpx.Client(transport=transport))
    )
    _configure_client(monkeypatch, client)
    data = pd.Series(["input"], index=[4])
    if dataframe:
        data = data.to_frame("value")
    method = getattr(data.aio if asynchronous else data.ai, method_name)
    options = _options(method_name)
    if method_name.endswith("with_cache"):
        options["cache"] = (AsyncBatchCache if asynchronous else BatchCache)(batch_size=0, show_progress=False)
    try:
        _assert_policy_parameter(method)
        with pytest.raises(RateLimitError):
            result = method(**options)
            if asynchronous:
                await result
        assert len(requests) == 2
        assert client.max_retries == 7
        assert not client.is_closed()
    finally:
        if asynchronous:
            await client.close()
        else:
            client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("dataframe", [False, True])
async def test_inferred_pandas_parse_policy(monkeypatch, failed_transport, asynchronous, dataframe):
    transport, requests = failed_transport
    client = (
        AsyncOpenAI(api_key="test", http_client=httpx.AsyncClient(transport=transport))
        if asynchronous
        else OpenAI(api_key="test", http_client=httpx.Client(transport=transport))
    )
    _configure_client(monkeypatch, client)
    data = pd.Series(["input"])
    if dataframe:
        data = data.to_frame("value")
    accessor = data.aio if asynchronous else data.ai
    try:
        with pytest.raises(RateLimitError):
            result = accessor.parse(**_options("parse", inferred=True))
            if asynchronous:
                await result
        assert len(requests) == 2
    finally:
        if asynchronous:
            await client.close()
        else:
            client.close()


@pytest.mark.parametrize("method_name", ["responses_udf", "task_udf", "embeddings_udf"])
def test_duckdb_retry_policy(monkeypatch, failed_transport, method_name):
    transport, requests = failed_transport
    client = AsyncOpenAI(api_key="test", http_client=httpx.AsyncClient(transport=transport))
    resolve = CONTAINER.resolve
    monkeypatch.setattr(CONTAINER, "resolve", lambda kind: client if kind is AsyncOpenAI else resolve(kind))
    method = getattr(duckdb_ext, method_name)
    try:
        _assert_policy_parameter(method)
        with duckdb.connect() as connection:
            method(connection, "reply", **_options(method_name))
            connection.execute("CREATE TABLE inputs(value VARCHAR)")
            connection.execute("INSERT INTO inputs VALUES ('input')")
            with pytest.raises(duckdb.InvalidInputException, match="RateLimitError"):
                connection.sql("SELECT reply(value) FROM inputs").fetchall()
        assert len(requests) == 2
    finally:
        import asyncio

        asyncio.run(client.close())


@pytest.mark.parametrize("method_name", ["responses_udf", "task_udf", "embeddings_udf", "parse_udf"])
def test_spark_retry_policy(monkeypatch, failed_transport, method_name):
    pytest.importorskip("pyspark")
    from openaivec import spark_ext

    transport, requests = failed_transport
    client = AsyncOpenAI(api_key="test", http_client=httpx.AsyncClient(transport=transport))
    resolve = CONTAINER.resolve
    monkeypatch.setattr(CONTAINER, "resolve", lambda kind: client if kind is AsyncOpenAI else resolve(kind))
    monkeypatch.setattr(spark_ext, "pandas_udf", lambda **kwargs: lambda function: function)
    method = getattr(spark_ext, method_name)
    try:
        _assert_policy_parameter(method)
        udf = method(**_options(method_name))
        with pytest.raises(RateLimitError):
            list(udf(iter([pd.Series(["input"])])))
        assert len(requests) == 2
    finally:
        import asyncio

        asyncio.run(client.close())
