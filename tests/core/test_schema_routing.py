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
from openaivec._di import ProviderError
from openaivec._provider import CONTAINER
from openaivec._schema import SchemaInferenceOutput
from openaivec._schema.spec import FieldSpec, ObjectSpec


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
