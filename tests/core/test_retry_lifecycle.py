import asyncio
import json
from functools import partial
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pandas as pd
import pytest
from openai import AsyncOpenAI, OpenAI, RateLimitError

from openaivec import (
    AsyncBatchEmbeddings,
    AsyncBatchResponses,
    AsyncSchemaInferer,
    BatchEmbeddings,
    BatchResponses,
    EmbeddingLimits,
    RetryPolicy,
    SchemaInferenceInput,
    SchemaInferer,
    _retry,
)
from openaivec import pandas_ext as pandas_ext
from openaivec._provider import CONTAINER
from openaivec._schema.spec import FieldSpec, ObjectSpec


@pytest.mark.parametrize(
    "options,error_type",
    [
        ({"max_attempts": 0}, ValueError),
        ({"max_attempts": -1}, ValueError),
        ({"max_attempts": True}, TypeError),
        ({"max_attempts": 1.5}, TypeError),
        ({"initial_delay": -1}, ValueError),
        ({"initial_delay": float("nan")}, ValueError),
        ({"initial_delay": True}, TypeError),
        ({"max_delay": float("inf")}, ValueError),
        ({"max_delay": "1"}, TypeError),
        ({"max_elapsed": 0}, ValueError),
        ({"max_elapsed": -1}, ValueError),
        ({"max_elapsed": float("inf")}, ValueError),
        ({"max_elapsed": False}, TypeError),
    ],
)
def test_retry_policy_rejects_invalid_limits(options, error_type):
    with pytest.raises(error_type):
        RetryPolicy(**options)


def _response(payload):
    return httpx.Response(
        200,
        json={
            "id": "resp_test",
            "object": "response",
            "created_at": 0,
            "model": "test-model",
            "output": [
                {
                    "id": "msg_test",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": json.dumps(payload), "annotations": []}],
                }
            ],
        },
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("endpoint", ["responses", "schema", "embeddings"])
async def test_deadline_spans_corrections_and_subrequests(monkeypatch, asynchronous, endpoint):
    elapsed = SimpleNamespace(value=0.0)
    requests = []

    def handler(request):
        requests.append(request)
        elapsed.value += 3
        if endpoint == "embeddings":
            return httpx.Response(200, json={"data": [{"index": 0, "embedding": [1.0]}]})
        return _response({"assistant_messages": []} if endpoint == "responses" else {})

    monkeypatch.setattr(_retry, "time", SimpleNamespace(monotonic=lambda: elapsed.value))
    transport = httpx.MockTransport(handler)
    client = (
        AsyncOpenAI(api_key="test", http_client=httpx.AsyncClient(transport=transport))
        if asynchronous
        else OpenAI(api_key="test", http_client=httpx.Client(transport=transport))
    )
    policy = RetryPolicy(max_attempts=1, max_elapsed=5)
    try:
        if endpoint == "schema":
            inferer = (AsyncSchemaInferer if asynchronous else SchemaInferer)(client, "test-model")
            data = SchemaInferenceInput(examples=["first"], instructions="extract")
            method = partial(inferer.infer_schema, data, max_retries=8, retry_policy=policy)
        elif endpoint == "responses":
            wrapper = (AsyncBatchResponses if asynchronous else BatchResponses).of(
                client, "test-model", "echo", retry_policy=policy, max_validation_retries=4
            )
            wrapper.cache.show_progress = False
            method = partial(wrapper.parse, ["first"])
        else:
            wrapper = (AsyncBatchEmbeddings if asynchronous else BatchEmbeddings).of(
                client, "test-model", batch_size=0, limits=EmbeddingLimits(max_inputs=1), retry_policy=policy
            )
            wrapper.cache.show_progress = False
            method = partial(wrapper.create, ["first", "second", "third"])
        with pytest.raises(TimeoutError):
            result = method()
            if asynchronous:
                await result
        assert len(requests) == 2
        assert requests[0].extensions["timeout"]["read"] == 5
        assert requests[1].extensions["timeout"]["read"] == 2
    finally:
        if asynchronous:
            await client.close()
        else:
            client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("mode", ["batch", "multimodal_text", "multimodal_structured"])
async def test_transport_retry_does_not_reset_validation_budget(asynchronous, mode):
    requests = []

    def handler(request):
        requests.append(request)
        if len(requests) % 2:
            return httpx.Response(429, json={"error": {"message": "forced failure"}})
        if mode == "multimodal_structured":
            return _response({"instructions": "extract", "examples": ["input"]})
        return _response({"assistant_messages": []})

    transport = httpx.MockTransport(handler)
    client = (
        AsyncOpenAI(api_key="test", http_client=httpx.AsyncClient(transport=transport))
        if asynchronous
        else OpenAI(api_key="test", http_client=httpx.Client(transport=transport))
    )
    wrapper = (AsyncBatchResponses if asynchronous else BatchResponses).of(
        client,
        "test-model",
        "echo",
        response_format=SchemaInferenceInput if mode == "multimodal_structured" else str,
        retry_policy=RetryPolicy(max_attempts=2, initial_delay=0),
        max_validation_retries=1,
    )
    wrapper.cache.show_progress = False
    try:
        if mode != "batch":
            result = wrapper._request_multimodal([{"role": "user", "content": "input"}])
            if asynchronous:
                result = await result
            if mode == "multimodal_structured":
                assert result == SchemaInferenceInput(instructions="extract", examples=["input"])
            else:
                assert result == '{"assistant_messages": []}'
            assert len(requests) == 2
        else:
            from pydantic import ValidationError

            with pytest.raises(ValidationError):
                result = wrapper.parse(["input"])
                if asynchronous:
                    await result
            assert len(requests) == 4
            assert "VALIDATION" in json.loads(requests[-1].content)["instructions"]
        assert not client.is_closed()
        assert client.max_retries == 2
    finally:
        if asynchronous:
            await client.close()
        else:
            client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("adapter", ["pandas", "pandas_async", "spark"])
async def test_inferred_parse_retries_inference_and_extraction(monkeypatch, adapter):
    from openaivec import SchemaInferenceOutput

    schema = SchemaInferenceOutput(
        instructions="extract",
        examples_summary="names",
        examples_instructions_alignment="extract names",
        object_spec=ObjectSpec(name="Result", fields=[FieldSpec(name="name", type="string", description="Name")]),
        inference_prompt="extract name",
    )
    requests = []

    def handler(request):
        requests.append(request)
        assert b"retry_policy" not in request.content
        if len(requests) == 2:
            return _response(schema.model_dump())
        return httpx.Response(429, json={"error": {"message": "forced failure"}})

    transport = httpx.MockTransport(handler)
    with OpenAI(api_key="test", http_client=httpx.Client(transport=transport)) as client:
        async with AsyncOpenAI(api_key="test", http_client=httpx.AsyncClient(transport=transport)) as async_client:
            resolve = CONTAINER.resolve
            dependencies = {
                OpenAI: client,
                AsyncOpenAI: async_client,
                SchemaInferer: SchemaInferer(client, "test-model"),
            }
            monkeypatch.setattr(
                CONTAINER, "resolve", lambda kind: dependencies[kind] if kind in dependencies else resolve(kind)
            )
            policy = RetryPolicy(max_attempts=2, initial_delay=0)
            if adapter == "spark":
                pytest.importorskip("pyspark")
                from openaivec import spark_ext

                spark = Mock()
                spark.table.return_value.rdd.map.return_value.takeSample.return_value = ["input"]
                dependencies[spark_ext.SparkSession] = spark
                monkeypatch.setattr(spark_ext, "pandas_udf", lambda **kwargs: lambda function: function)
                udf = spark_ext.parse_udf(
                    "extract",
                    example_table_name="inputs",
                    example_field_name="value",
                    retry_policy=policy,
                    batch_size=0,
                )
                assert len(requests) == 2
                with pytest.raises(RateLimitError):
                    await asyncio.to_thread(list, udf(iter([pd.Series(["input"])])))
            else:
                data = pd.Series(["input"])
                with pytest.raises(RateLimitError):
                    if adapter == "pandas_async":
                        await data.aio.parse("extract", retry_policy=policy, batch_size=0, show_progress=False)
                    else:
                        data.ai.parse("extract", retry_policy=policy, batch_size=0, show_progress=False)
            assert len(requests) == 4
            assert not client.is_closed()
            assert not async_client.is_closed()


@pytest.mark.asyncio
@pytest.mark.parametrize("stage,termination", [("request", "cancel"), ("request", "deadline"), ("backoff", "cancel")])
async def test_async_termination_drains_and_allows_reuse(monkeypatch, stage, termination):
    started = asyncio.Event()
    stopped = asyncio.Event()
    recovered = False
    requests = []

    async def wait():
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()

    async def handler(request):
        requests.append(request)
        if recovered:
            return httpx.Response(200, json={"data": [{"index": 0, "embedding": [2.0]}]})
        if stage == "request":
            await wait()
        return httpx.Response(429, json={"error": {"message": "forced failure"}})

    async def backoff(delay):
        await wait()

    if stage == "backoff":
        monkeypatch.setattr(_retry, "asyncio", SimpleNamespace(sleep=backoff, wait_for=asyncio.wait_for))
    async with AsyncOpenAI(
        api_key="test", http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
    ) as client:
        wrapper = AsyncBatchEmbeddings.of(
            client,
            "test-model",
            batch_size=0,
            retry_policy=RetryPolicy(max_elapsed=0.1 if termination == "deadline" else None),
        )
        wrapper.cache.show_progress = False
        pending = asyncio.create_task(wrapper.create(["input"]))
        try:
            await asyncio.wait_for(started.wait(), 2)
            if termination == "cancel":
                pending.cancel()
            with pytest.raises(asyncio.CancelledError if termination == "cancel" else TimeoutError):
                await asyncio.wait_for(pending, 2)
        finally:
            pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)
        assert stopped.is_set()
        assert len(requests) == 1
        assert not wrapper.cache._inflight
        assert not client.is_closed()
        recovered = True
        result = await asyncio.wait_for(wrapper.create(["input"]), 2)
        assert result[0].tolist() == [2.0]
        direct = await client.embeddings.create(model="test-model", input=["direct"])
        assert direct.data[0].embedding == [2.0]
        assert len(requests) == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_server_no_retry_header_is_respected(asynchronous):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(429, headers={"x-should-retry": "false"}, json={"error": {"message": "stop"}})

    transport = httpx.MockTransport(handler)
    client = (
        AsyncOpenAI(api_key="test", http_client=httpx.AsyncClient(transport=transport))
        if asynchronous
        else OpenAI(api_key="test", http_client=httpx.Client(transport=transport))
    )
    wrapper = (AsyncBatchEmbeddings if asynchronous else BatchEmbeddings).of(
        client, "test-model", retry_policy=RetryPolicy(max_attempts=5, initial_delay=0)
    )
    wrapper.cache.show_progress = False
    try:
        with pytest.raises(RateLimitError):
            result = wrapper.create(["input"])
            if asynchronous:
                await result
        assert len(requests) == 1
    finally:
        if asynchronous:
            await client.close()
        else:
            client.close()
