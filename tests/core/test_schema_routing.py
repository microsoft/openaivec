import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pandas as pd
import pytest
from openai import AsyncOpenAI, OpenAI
from pydantic import ValidationError

import openaivec
from openaivec import SchemaInferer, pandas_ext  # noqa: F401
from openaivec._cache import AsyncBatchCache, BatchCache
from openaivec._di import ProviderError
from openaivec._provider import CONTAINER, set_default_registrations
from openaivec._schema import SchemaInferenceOutput
from openaivec._schema.spec import FieldSpec, ObjectSpec


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("change", ["model", "client"])
async def test_pandas_inference_uses_updated_configuration(monkeypatch, asynchronous, change):
    # Preserve existing registrations while exercising the default provider graph.
    monkeypatch.setattr(CONTAINER, "_providers", CONTAINER._providers.copy())
    monkeypatch.setattr(CONTAINER, "_instances", CONTAINER._instances.copy())
    set_default_registrations()
    schema = SchemaInferenceOutput(
        instructions="extract",
        examples_summary="names",
        examples_instructions_alignment="extract names",
        object_spec=ObjectSpec(name="Result", fields=[FieldSpec(name="name", type="string", description="Name")]),
        inference_prompt="extract name",
    )
    client_type = AsyncOpenAI if asynchronous else OpenAI
    first = client_type(api_key="test")
    second = client_type(api_key="test")
    mock_type = AsyncMock if asynchronous else Mock
    first_parse = mock_type(return_value=SimpleNamespace(output_parsed=schema))
    second_parse = mock_type(return_value=SimpleNamespace(output_parsed=schema))
    monkeypatch.setattr(first.responses, "parse", first_parse)
    monkeypatch.setattr(second.responses, "parse", second_parse)
    set_client = openaivec.set_async_client if asynchronous else openaivec.set_client
    set_client(first)
    openaivec.set_responses_model("gpt-6-luna")
    series = pd.Series(["first"])
    accessor = series.aio if asynchronous else series.ai

    async def infer():
        result = accessor.infer_schema("extract", reasoning={"effort": "none"}, store=False)
        return await result if asynchronous else result

    try:
        assert await infer() is schema
        assert first_parse.call_args.kwargs["model"] == "gpt-6-luna"
        if change == "model":
            openaivec.set_responses_model("gpt-6-sol")
        else:
            set_client(second)
        assert await infer() is schema
        current_parse = first_parse if change == "model" else second_parse
        assert current_parse.call_args.kwargs["model"] == ("gpt-6-sol" if change == "model" else "gpt-6-luna")
        assert current_parse.call_args.kwargs["reasoning"] == {"effort": "none"}
        assert current_parse.call_args.kwargs["store"] is False
        assert first_parse.call_count == (2 if change == "model" else 1)
        assert second_parse.call_count == (0 if change == "model" else 1)
        assert not first.is_closed()
        assert not second.is_closed()
    finally:
        if asynchronous:
            await first.close()
            await second.close()
        else:
            first.close()
            second.close()


@pytest.mark.parametrize("max_retries", [1, 2])
def test_pandas_inference_forwards_request_options(monkeypatch, max_retries):
    with OpenAI(api_key="test") as client:
        parse = Mock(return_value=SimpleNamespace(output_parsed=None))
        monkeypatch.setattr(client.responses, "parse", parse)
        inferer = SchemaInferer(client=client, model_name="gpt-4.1-mini")
        monkeypatch.setattr(CONTAINER, "resolve", Mock(return_value=inferer))
        with pytest.raises(ValueError, match=f"after {max_retries} attempts"):
            pd.Series(["first", "second"]).ai.infer_schema(
                "extract",
                max_examples=1,
                max_retries=max_retries,
                temperature=0,
                max_output_tokens=64,
                timeout=1,
                store=False,
            )
        assert parse.call_count == max_retries
        for call in parse.call_args_list:
            options = call.kwargs
            assert options["temperature"] == 0
            assert options["max_output_tokens"] == 64
            assert options["timeout"] == 1
            assert options["store"] is False
            assert "max_retries" not in options
            payload = json.loads(options["input"])
            assert set(payload) == {"instructions", "examples"}
            assert len(payload["examples"]) == 1


@pytest.mark.asyncio
async def test_async_parse_uses_only_configured_async_client(monkeypatch):
    schema = SchemaInferenceOutput(
        instructions="extract",
        examples_summary="names",
        examples_instructions_alignment="extract names",
        object_spec=ObjectSpec(name="Result", fields=[FieldSpec(name="name", type="string", description="Name")]),
        inference_prompt="extract name",
    )
    async with AsyncOpenAI(api_key="test", base_url="https://example.invalid/v1/") as client:

        def respond(**kwargs):
            if kwargs["text_format"] is SchemaInferenceOutput:
                return SimpleNamespace(output_parsed=schema)
            data = json.loads(kwargs["input"])
            parsed = kwargs["text_format"].model_validate(
                {
                    "assistant_messages": [
                        {"id": message["id"], "body": {"name": message["body"]}} for message in data["user_messages"]
                    ]
                }
            )
            return SimpleNamespace(output_parsed=parsed)

        parse = AsyncMock(side_effect=respond)
        monkeypatch.setattr(client.responses, "parse", parse)
        resolve = CONTAINER.resolve
        sync_calls = Mock(side_effect=ProviderError("Synchronous client must not be resolved"))

        def resolve_client(kind):
            if kind is AsyncOpenAI:
                return client
            if kind is OpenAI or kind is SchemaInferer:
                return sync_calls()
            return resolve(kind)

        monkeypatch.setattr(CONTAINER, "resolve", resolve_client)
        series = pd.Series(["first", "second", "first"], index=[9, 3, 7], name="names")
        result = await series.aio.parse("extract", max_examples=2, store=False, temperature=0)
        assert result.index.equals(series.index)
        assert [value.name for value in result] == series.tolist()
        assert parse.call_count == 2
        sync_calls.assert_not_called()
        for call in parse.call_args_list:
            assert call.kwargs["store"] is False
            assert call.kwargs["temperature"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("max_retries", [1, 2])
async def test_async_schema_retry_budget_and_options(monkeypatch, max_retries):
    async with AsyncOpenAI(api_key="test") as client:
        parse = AsyncMock(return_value=SimpleNamespace(output_parsed=None))
        monkeypatch.setattr(client.responses, "parse", parse)
        inferer = openaivec.AsyncSchemaInferer(client=client, model_name="gpt-4.1-mini")
        data = openaivec.SchemaInferenceInput(examples=["first"], instructions="extract")
        with pytest.raises(ValueError, match=f"after {max_retries} attempts"):
            await inferer.infer_schema(data, max_retries=max_retries, store=False, timeout=1)
        assert parse.call_count == max_retries
        for call in parse.call_args_list:
            assert call.kwargs["store"] is False
            assert call.kwargs["timeout"] == 1
            assert "max_retries" not in call.kwargs
        if max_retries > 1:
            assert "PRIOR VALIDATION FEEDBACK" in parse.call_args.kwargs["instructions"]


@pytest.mark.asyncio
async def test_async_schema_retries_sdk_validation_error(monkeypatch):
    with pytest.raises(ValidationError) as error:
        openaivec.SchemaInferenceInput.model_validate({})
    async with AsyncOpenAI(api_key="test") as client:
        parse = AsyncMock(side_effect=[error.value, SimpleNamespace(output_parsed=None)])
        monkeypatch.setattr(client.responses, "parse", parse)
        inferer = openaivec.AsyncSchemaInferer(client=client, model_name="gpt-4.1-mini")
        data = openaivec.SchemaInferenceInput(examples=["first"], instructions="extract")
        with pytest.raises(ValueError, match="after 2 attempts"):
            await inferer.infer_schema(data, max_retries=2)
        assert parse.call_count == 2
        assert "PRIOR VALIDATION FEEDBACK" in parse.call_args.kwargs["instructions"]
        assert "instructions" in parse.call_args.kwargs["instructions"]
        parse.side_effect = ValueError("transport configuration")
        parse.reset_mock()
        with pytest.raises(ValueError, match="transport configuration"):
            await inferer.infer_schema(data)
        assert parse.call_count == 1


@pytest.mark.asyncio
async def test_async_schema_cancellation_stops_request(monkeypatch):
    started = asyncio.Event()
    stopped = asyncio.Event()

    async def respond(**kwargs):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()

    async with AsyncOpenAI(api_key="test") as client:
        parse = AsyncMock(side_effect=respond)
        monkeypatch.setattr(client.responses, "parse", parse)
        inferer = openaivec.AsyncSchemaInferer(client=client, model_name="gpt-4.1-mini")
        data = openaivec.SchemaInferenceInput(examples=["first"], instructions="extract")
        task = asyncio.create_task(inferer.infer_schema(data))
        try:
            await asyncio.wait_for(started.wait(), timeout=2)
        finally:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        assert stopped.is_set()
        assert parse.call_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("dataframe", [False, True])
@pytest.mark.parametrize("with_cache", [False, True])
@pytest.mark.parametrize("infer_schema", [False, True])
async def test_parse_separates_inference_and_extraction_limits(
    monkeypatch, asynchronous, dataframe, with_cache, infer_schema
):
    schema = SimpleNamespace(model=str, inference_prompt="extract")
    inference = AsyncMock(return_value=schema) if asynchronous else Mock(return_value=schema)
    accessor_type = pandas_ext.AsyncOpenAIVecSeriesAccessor if asynchronous else pandas_ext.OpenAIVecSeriesAccessor
    monkeypatch.setattr(accessor_type, "infer_schema", inference)
    client = AsyncOpenAI(api_key="test") if asynchronous else OpenAI(api_key="test")
    parse = (
        AsyncMock(return_value=SimpleNamespace(output_parsed=None))
        if asynchronous
        else Mock(return_value=SimpleNamespace(output_parsed=None))
    )
    monkeypatch.setattr(client.responses, "parse", parse)
    resolve = CONTAINER.resolve
    monkeypatch.setattr(CONTAINER, "resolve", lambda kind: client if kind in (OpenAI, AsyncOpenAI) else resolve(kind))
    series = pd.Series(["first", "second", "first"], index=[9, 3, 7])
    data = series.to_frame("value") if dataframe else series
    accessor = data.aio if asynchronous else data.ai
    options = {
        "instructions": "extract",
        "response_format": None if infer_schema else str,
        "max_retries": 1,
        "max_validation_retries": 0,
        "store": False,
    }
    method_name = "parse_with_cache" if with_cache else "parse"
    if with_cache:
        cache_type = AsyncBatchCache if asynchronous else BatchCache
        options["cache"] = cache_type(batch_size=2, show_progress=False)
    try:
        result = getattr(accessor, method_name)(**options)
        if asynchronous:
            result = await result
        assert result.index.equals(series.index)
        assert result.tolist() == [None, None, None]
        assert parse.call_count == 1
        assert "max_retries" not in parse.call_args.kwargs
        assert "max_validation_retries" not in parse.call_args.kwargs
        assert parse.call_args.kwargs["store"] is False
        assert inference.call_count == int(infer_schema)
        if infer_schema:
            assert inference.call_args.kwargs["max_retries"] == 1
            assert "max_validation_retries" not in inference.call_args.kwargs
    finally:
        if asynchronous:
            await client.close()
        else:
            client.close()


@pytest.mark.parametrize("extract", [False, True])
def test_spark_inference_controls_do_not_reach_extraction(monkeypatch, extract):
    pytest.importorskip("pyspark")
    from openaivec import spark_ext

    schema = SimpleNamespace(model=str, inference_prompt="extract")
    inference = Mock(return_value=schema)
    spark = Mock()
    spark.table.return_value.rdd.map.return_value.takeSample.return_value = ["first"]
    parse = AsyncMock(return_value=SimpleNamespace(output_parsed=None))
    client = Mock(spec=AsyncOpenAI)
    client.responses = Mock(parse=parse)
    resolve = CONTAINER.resolve

    def resolve_dependency(kind):
        if kind is spark_ext.SparkSession:
            return spark
        if kind is SchemaInferer:
            return SimpleNamespace(infer_schema=inference)
        if kind is AsyncOpenAI:
            return client
        return resolve(kind)

    monkeypatch.setattr(CONTAINER, "resolve", resolve_dependency)
    monkeypatch.setattr(spark_ext, "pandas_udf", lambda **kwargs: lambda function: function)
    method = spark_ext.parse_udf if extract else spark_ext.infer_schema
    result = method("extract", example_table_name="inputs", example_field_name="value", max_retries=1, store=False)
    if extract:
        list(result(iter([pd.Series(["first"])])))
        assert parse.call_count == 1
        assert "max_retries" not in parse.call_args.kwargs
        assert parse.call_args.kwargs["store"] is False
    assert inference.call_count == 1
    assert inference.call_args.kwargs == {"max_retries": 1, "retry_policy": None, "store": False}
    assert inference.call_args.args[0].examples == ["first"]
