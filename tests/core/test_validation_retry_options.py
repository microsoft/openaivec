import inspect
from unittest.mock import AsyncMock, Mock

import duckdb
import pandas as pd
import pytest
from openai import AsyncOpenAI, OpenAI
from pydantic import ValidationError, create_model

from openaivec import PreparedTask, duckdb_ext, pandas_ext
from openaivec._cache import AsyncBatchCache, BatchCache
from openaivec._provider import CONTAINER

_METHODS = ["responses", "responses_with_cache", "task", "task_with_cache", "parse", "parse_with_cache"]
_ACCESSORS = [
    pandas_ext.OpenAIVecSeriesAccessor,
    pandas_ext.AsyncOpenAIVecSeriesAccessor,
    pandas_ext.OpenAIVecDataFrameAccessor,
    pandas_ext.AsyncOpenAIVecDataFrameAccessor,
]


def _validation_error():
    return ValidationError.from_exception_data(
        "Response", [{"type": "missing", "loc": ("assistant_messages",), "input": {}}]
    )


@pytest.mark.parametrize("accessor", _ACCESSORS)
@pytest.mark.parametrize("method_name", _METHODS)
def test_pandas_validation_retry_signature(accessor, method_name):
    parameter = inspect.signature(getattr(accessor, method_name)).parameters["max_validation_retries"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("dataframe", [False, True])
@pytest.mark.parametrize("method_name", _METHODS)
@pytest.mark.parametrize("retries", [-1, 0, 1])
async def test_pandas_validation_retry_budget(monkeypatch, asynchronous, dataframe, method_name, retries):
    parse = AsyncMock(side_effect=_validation_error()) if asynchronous else Mock(side_effect=_validation_error())
    client = AsyncOpenAI(api_key="test") if asynchronous else OpenAI(api_key="test")
    monkeypatch.setattr(client.responses, "parse", parse)
    resolve = CONTAINER.resolve
    monkeypatch.setattr(CONTAINER, "resolve", lambda kind: client if kind in (OpenAI, AsyncOpenAI) else resolve(kind))
    series = pd.Series(["first", "second", "first"], index=[9, 3, 7])
    data = series.to_frame("value") if dataframe else series
    accessor = data.aio if asynchronous else data.ai
    options = {"max_validation_retries": retries}
    if method_name.startswith("task"):
        options["task"] = PreparedTask(instructions="echo", response_format=str)
    else:
        options.update(instructions="echo", response_format=str)
    if method_name.endswith("with_cache"):
        cache_type = AsyncBatchCache if asynchronous else BatchCache
        options["cache"] = cache_type(batch_size=2, show_progress=False)
    try:
        error = ValueError if retries < 0 else ValidationError
        with pytest.raises(error):
            result = getattr(accessor, method_name)(**options)
            if asynchronous:
                await result
        assert parse.call_count == max(retries + 1, 0)
        for call in parse.call_args_list:
            assert "max_validation_retries" not in call.kwargs
    finally:
        if asynchronous:
            await client.close()
        else:
            client.close()


@pytest.mark.parametrize("method_name", ["responses_udf", "task_udf"])
@pytest.mark.parametrize("retries", [-1, 0, 1])
def test_duckdb_validation_retry_budget(monkeypatch, method_name, retries):
    parse = AsyncMock(side_effect=_validation_error())
    client = Mock(spec=AsyncOpenAI)
    client.responses = Mock(parse=parse)
    resolve = CONTAINER.resolve
    monkeypatch.setattr(CONTAINER, "resolve", lambda kind: client if kind is AsyncOpenAI else resolve(kind))
    method = getattr(duckdb_ext, method_name)
    parameter = inspect.signature(method).parameters["max_validation_retries"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default == 3
    options = {"max_validation_retries": retries}
    if method_name == "task_udf":
        options["task"] = PreparedTask(instructions="echo", response_format=str)
    else:
        options["instructions"] = "echo"
    with duckdb.connect() as connection:
        if retries < 0:
            with pytest.raises(ValueError, match="max_validation_retries"):
                method(connection, "reply", **options)
        else:
            method(connection, "reply", **options)
            connection.execute("CREATE TABLE inputs(value VARCHAR)")
            connection.execute("INSERT INTO inputs VALUES ('first')")
            with pytest.raises(duckdb.InvalidInputException, match="ValidationError"):
                connection.sql("SELECT reply(value) FROM inputs").fetchall()
    assert parse.call_count == max(retries + 1, 0)
    for call in parse.call_args_list:
        assert "max_validation_retries" not in call.kwargs


@pytest.mark.parametrize("method_name", ["responses_udf", "task_udf", "parse_udf"])
@pytest.mark.parametrize("retries", [-1, 0, 1])
@pytest.mark.parametrize("structured", [False, True])
def test_spark_validation_retry_budget(monkeypatch, method_name, retries, structured):
    pytest.importorskip("pyspark")
    from openaivec import spark_ext

    monkeypatch.setattr(spark_ext, "pandas_udf", lambda **kwargs: lambda function: function)
    parse = AsyncMock(side_effect=_validation_error())
    client = Mock(spec=AsyncOpenAI)
    client.responses = Mock(parse=parse)
    resolve = CONTAINER.resolve
    monkeypatch.setattr(CONTAINER, "resolve", lambda kind: client if kind is AsyncOpenAI else resolve(kind))
    method = getattr(spark_ext, method_name)
    parameter = inspect.signature(method).parameters["max_validation_retries"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default == 3
    options = {"max_validation_retries": retries}
    response_format = create_model("Result", value=(str, ...)) if structured else str
    if method_name == "task_udf":
        options["task"] = PreparedTask(instructions="echo", response_format=response_format)
    else:
        options.update(instructions="echo", response_format=response_format)
    if retries < 0:
        with pytest.raises(ValueError, match="max_validation_retries"):
            method(**options)
    else:
        udf = method(**options)
        with pytest.raises(ValidationError):
            list(udf(iter([pd.Series(["first"])])))
    assert parse.call_count == max(retries + 1, 0)
    for call in parse.call_args_list:
        assert "max_validation_retries" not in call.kwargs
