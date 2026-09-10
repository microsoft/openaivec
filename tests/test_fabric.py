import asyncio
import json
import os
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import httpx
import pandas as pd
import pytest
from openai import AsyncAzureOpenAI, AzureOpenAI
from openai.lib.azure import API_KEY_SENTINEL
from pydantic import BaseModel

import openaivec
from openaivec._di import ProviderError
from openaivec._fabric import provide_async_fabric_client, provide_fabric_client
from openaivec._provider import CONTAINER, set_default_registrations
from openaivec._schema import SchemaInferer


@pytest.fixture
def fabric_runtime(monkeypatch, reset_environment):
    requests: list[httpx.Request] = []

    def handle_request(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path.endswith("/embeddings"):
            inputs = json.loads(request.content)["input"]
            return httpx.Response(
                200,
                json={
                    "object": "list",
                    "model": "text-embedding-ada-002",
                    "data": [
                        {"object": "embedding", "index": index, "embedding": [float(index), 1.0]}
                        for index in range(len(inputs))
                    ],
                },
            )
        output = []
        if credentials.response_text is not None:
            output = [
                {
                    "id": "msg_test",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": credentials.response_text, "annotations": []}],
                }
            ]
        return httpx.Response(
            200,
            json={
                "id": "resp_test",
                "object": "response",
                "created_at": 0,
                "model": "gpt-5.1",
                "status": "completed",
                "output": output,
            },
        )

    sync_http_client = httpx.Client(transport=httpx.MockTransport(handle_request))
    async_http_client = httpx.AsyncClient(transport=httpx.MockTransport(handle_request))
    credentials = ModuleType("synapse.ml.fabric.credentials")
    credentials.requests = requests
    credentials.response_text = None
    credentials.get_openai_httpx_sync_client = Mock(return_value=sync_http_client)
    credentials.get_openai_httpx_async_client = Mock(return_value=async_http_client)
    discovery = ModuleType("synapse.ml.fabric.service_discovery")
    discovery.get_fabric_env_config = Mock(
        return_value=SimpleNamespace(fabric_env_config=SimpleNamespace(ml_workload_endpoint="https://fabric.example/"))
    )
    notebookutils = ModuleType("notebookutils")
    notebookutils.credentials = SimpleNamespace(
        getSecret=Mock(side_effect=AssertionError("Unexpected Key Vault access"))
    )
    monkeypatch.setitem(sys.modules, credentials.__name__, credentials)
    monkeypatch.setitem(sys.modules, discovery.__name__, discovery)
    monkeypatch.setitem(sys.modules, notebookutils.__name__, notebookutils)
    set_default_registrations()
    yield credentials
    sync_http_client.close()
    asyncio.run(async_http_client.aclose())
    set_default_registrations()


def test_setup_fabric_uses_runtime_transport_and_models(fabric_runtime):
    openaivec.setup_fabric()

    fabric_runtime.get_openai_httpx_sync_client.assert_not_called()
    fabric_runtime.get_openai_httpx_async_client.assert_not_called()
    assert openaivec.get_responses_model() == "gpt-5.1"
    assert openaivec.get_embeddings_model() == "text-embedding-ada-002"
    client = openaivec.get_client()
    assert isinstance(client, AzureOpenAI)
    assert str(client.base_url) == "https://fabric.example/cognitive/openai/openai/"
    assert client.default_query["api-version"] == "2025-04-01-preview"
    assert openaivec.get_client() is client
    fabric_runtime.get_openai_httpx_sync_client.assert_called_once_with()


@pytest.mark.asyncio
async def test_setup_fabric_supports_async_runtime_transport(fabric_runtime):
    openaivec.setup_fabric()

    client = openaivec.get_async_client()
    assert isinstance(client, AsyncAzureOpenAI)
    assert str(client.base_url) == "https://fabric.example/cognitive/openai/openai/"
    fabric_runtime.get_openai_httpx_sync_client.assert_not_called()
    fabric_runtime.get_openai_httpx_async_client.assert_called_once_with()
    await client.close()


def test_setup_fabric_isolates_existing_azure_credentials(fabric_runtime, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "unrelated-openai-key")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "unrelated-azure-key")
    monkeypatch.setenv("AZURE_OPENAI_AD_TOKEN", "unrelated-token")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://other.openai.azure.com/")
    monkeypatch.setenv("AZURE_OPENAI_BASE_URL", "https://other.openai.azure.com/openai/v1/")
    openaivec.setup_fabric()

    client = openaivec.get_client()
    assert client.api_key == API_KEY_SENTINEL
    assert str(client.base_url) == "https://fabric.example/cognitive/openai/openai/"
    assert "unrelated-token" not in client.default_headers.values()
    client.responses.create(model="gpt-5.1", input="test")
    request = fabric_runtime.requests[-1]
    assert all("unrelated" not in value for value in request.headers.values())


def test_setup_fabric_overrides_models_and_cached_schema_inferer(fabric_runtime, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    set_default_registrations()
    previous_inferer = CONTAINER.resolve(SchemaInferer)

    openaivec.setup_fabric(responses_model="gpt-5-mini", embeddings_model="custom-embedding")

    inferer = CONTAINER.resolve(SchemaInferer)
    assert inferer is not previous_inferer
    assert inferer.client is openaivec.get_client()
    assert inferer.model_name == "gpt-5-mini"
    assert openaivec.get_embeddings_model() == "custom-embedding"


def test_setup_fabric_requires_runtime_without_replacing_existing_client(monkeypatch, reset_environment):
    monkeypatch.setattr("openaivec._fabric.is_fabric_environment", lambda: False)
    existing = Mock()
    openaivec.set_client(existing)

    with pytest.raises(RuntimeError, match="Fabric notebook"):
        openaivec.setup_fabric()

    assert openaivec.get_client() is existing


def test_setup_fabric_reports_missing_async_helper(fabric_runtime):
    del fabric_runtime.get_openai_httpx_async_client
    openaivec.setup_fabric()

    with pytest.raises(ProviderError, match="async.*Fabric"):
        openaivec.get_async_client()

    assert isinstance(openaivec.get_client(), AzureOpenAI)


@pytest.mark.asyncio
@pytest.mark.parametrize("async_client", [False, True])
async def test_fabric_worker_client_uses_helpers_without_notebook(fabric_runtime, monkeypatch, async_client):
    monkeypatch.setattr("openaivec._fabric.is_fabric_environment", lambda: False)

    if async_client:
        client = provide_async_fabric_client(api_version="2025-04-01-preview")
        try:
            result = await client.embeddings.create(model="text-embedding-ada-002", input=["a", "b"])
        finally:
            await client.close()
    else:
        client = provide_fabric_client(api_version="2025-04-01-preview")
        try:
            result = client.embeddings.create(model="text-embedding-ada-002", input=["a", "b"])
        finally:
            client.close()

    assert len(result.data) == 2
    assert len(fabric_runtime.requests) == 1
    assert fabric_runtime.requests[0].url.host == "fabric.example"


@pytest.mark.parametrize("operation", ["create", "parse"])
def test_fabric_responses_are_stateless(fabric_runtime, operation):
    class Result(BaseModel):
        value: str

    openaivec.setup_fabric()
    client = openaivec.get_client()
    kwargs = {"text_format": Result} if operation == "parse" else {}

    getattr(client.responses, operation)(model="gpt-5.1", input="test", **kwargs)

    request = fabric_runtime.requests[-1]
    assert request.url.path == "/cognitive/openai/openai/responses"
    assert json.loads(request.content)["store"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["create", "parse"])
async def test_async_fabric_responses_are_stateless(fabric_runtime, operation):
    class Result(BaseModel):
        value: str

    openaivec.setup_fabric()
    client = openaivec.get_async_client()
    kwargs = {"text_format": Result} if operation == "parse" else {}

    await getattr(client.responses, operation)(model="gpt-5.1", input="test", **kwargs)

    assert json.loads(fabric_runtime.requests[-1].content)["store"] is False


@pytest.mark.parametrize(
    "kwargs",
    [
        {"store": True},
        {"previous_response_id": "resp_previous"},
        {"extra_body": {"store": True}},
        {"extra_body": {"previous_response_id": "resp_previous"}},
    ],
)
@pytest.mark.parametrize("async_client", [False, True])
def test_fabric_rejects_stateful_requests_before_sending(fabric_runtime, kwargs, async_client):
    openaivec.setup_fabric()

    with pytest.raises(ValueError, match="Fabric.*(store|previous_response_id)"):
        if async_client:
            asyncio.run(openaivec.get_async_client().responses.create(model="gpt-5.1", input="test", **kwargs))
        else:
            openaivec.get_client().responses.create(model="gpt-5.1", input="test", **kwargs)

    assert fabric_runtime.requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize("async_client", [False, True])
async def test_fabric_batch_responses_preserve_order_and_deduplicate(fabric_runtime, async_client):
    openaivec.setup_fabric()
    fabric_runtime.response_text = json.dumps({"assistant_messages": [{"id": 1, "body": "B"}, {"id": 0, "body": "A"}]})
    inputs = ["a", "b", "a"]

    if async_client:
        responses = openaivec.AsyncBatchResponses.of(
            client=openaivec.get_async_client(),
            model_name=openaivec.get_responses_model(),
            system_message="Repeat each input.",
            batch_size=2,
        )
        results = await responses.parse(inputs)
    else:
        responses = openaivec.BatchResponses.of(
            client=openaivec.get_client(),
            model_name=openaivec.get_responses_model(),
            system_message="Repeat each input.",
            batch_size=2,
        )
        results = responses.parse(inputs)

    assert results == ["A", "B", "A"]
    assert len(fabric_runtime.requests) == 1
    payload = json.loads(fabric_runtime.requests[0].content)
    assert payload["store"] is False
    assert [message["body"] for message in json.loads(payload["input"])["user_messages"]] == ["a", "b"]


@pytest.mark.asyncio
@pytest.mark.parametrize("async_client", [False, True])
async def test_fabric_batch_embeddings_use_deployment_endpoint(fabric_runtime, async_client):
    openaivec.setup_fabric()
    inputs = ["a", "b", "a"]

    if async_client:
        embeddings = openaivec.AsyncBatchEmbeddings.of(
            client=openaivec.get_async_client(), model_name=openaivec.get_embeddings_model(), batch_size=2
        )
        results = await embeddings.create(inputs)
    else:
        embeddings = openaivec.BatchEmbeddings.of(
            client=openaivec.get_client(), model_name=openaivec.get_embeddings_model(), batch_size=2
        )
        results = embeddings.create(inputs)

    assert [embedding.tolist() for embedding in results] == [[0.0, 1.0], [1.0, 1.0], [0.0, 1.0]]
    assert len(fabric_runtime.requests) == 1
    request = fabric_runtime.requests[0]
    assert request.url.path == "/cognitive/openai/openai/deployments/text-embedding-ada-002/embeddings"
    payload = json.loads(request.content)
    assert payload["input"] == ["a", "b"]
    assert "store" not in payload


def test_fabric_schema_inference_is_stateless(fabric_runtime):
    openaivec.setup_fabric()
    inferer = CONTAINER.resolve(SchemaInferer)
    data = openaivec.SchemaInferenceInput(examples=["orange"], instructions="Extract the item name.")

    with pytest.raises(ValueError, match="Schema inference returned no parsed output"):
        inferer.infer_schema(data, max_retries=1)

    assert len(fabric_runtime.requests) == 1
    assert json.loads(fabric_runtime.requests[0].content)["store"] is False


def test_spark_setup_fabric_keeps_authentication_runtime_managed(fabric_runtime, monkeypatch):
    from openaivec import spark_ext

    monkeypatch.setenv("OPENAI_API_KEY", "unrelated-openai-key")
    spark = SimpleNamespace(sparkContext=SimpleNamespace(environment={"existing": "value"}))
    original_environment = dict(os.environ)

    spark_ext.setup_fabric(spark, responses_model_name="gpt-5-mini", api_version="2025-04-01-preview")

    assert spark.sparkContext.environment == {"existing": "value"}
    assert dict(os.environ) == original_environment
    assert openaivec.get_responses_model() == "gpt-5-mini"
    assert openaivec.get_embeddings_model() == "text-embedding-ada-002"
    fabric_runtime.get_openai_httpx_sync_client.assert_not_called()
    fabric_runtime.get_openai_httpx_async_client.assert_not_called()


@pytest.mark.parametrize("operation", ["string", "structured", "embeddings", "task", "parse"])
def test_spark_fabric_clients_are_partition_local(fabric_runtime, monkeypatch, operation):
    from openaivec import spark_ext

    class Result(BaseModel):
        value: str

    monkeypatch.setattr(spark_ext, "pandas_udf", lambda **kwargs: lambda function: function)
    transport = fabric_runtime.get_openai_httpx_async_client.return_value._transport
    created_loops = []
    closed_loops = []
    clients = []

    def create_http_client():
        created_loops.append(asyncio.get_running_loop())
        client = httpx.AsyncClient(transport=transport)
        close_client = client.aclose

        async def close():
            closed_loops.append(asyncio.get_running_loop())
            await close_client()

        monkeypatch.setattr(client, "aclose", close)
        clients.append(client)
        return client

    fabric_runtime.get_openai_httpx_async_client.side_effect = create_http_client
    structured = operation in {"structured", "task", "parse"}
    fabric_runtime.response_text = json.dumps(
        {
            "assistant_messages": [
                {"id": 1, "body": {"value": "B"} if structured else "B"},
                {"id": 0, "body": {"value": "A"} if structured else "A"},
            ]
        }
    )
    spark = SimpleNamespace(sparkContext=SimpleNamespace(environment={}))
    spark_ext.setup_fabric(spark)
    if operation == "embeddings":
        udf = spark_ext.embeddings_udf(batch_size=2)
    elif operation == "task":
        udf = spark_ext.task_udf(openaivec.PreparedTask(instructions="Echo.", response_format=Result), batch_size=2)
    elif operation == "parse":
        udf = spark_ext.parse_udf("Echo.", response_format=Result, batch_size=2)
    else:
        udf = spark_ext.responses_udf("Echo.", response_format=Result if structured else str, batch_size=2)
    fabric_runtime.get_openai_httpx_async_client.assert_not_called()
    set_default_registrations()
    monkeypatch.setattr("openaivec._fabric.is_fabric_environment", lambda: False)
    monkeypatch.setattr(spark_ext, "get_async_client", Mock(side_effect=AssertionError("Unexpected default client")))

    for _ in range(2):
        results = list(udf(iter([pd.Series(["a", "b", "a"]), pd.Series(["b", "a"])])))
        if structured:
            assert [result.to_dict("records") for result in results] == [
                [{"value": "A"}, {"value": "B"}, {"value": "A"}],
                [{"value": "B"}, {"value": "A"}],
            ]
        elif operation == "embeddings":
            assert [[list(vector) for vector in result] for result in results] == [
                [[0.0, 1.0], [1.0, 1.0], [0.0, 1.0]],
                [[1.0, 1.0], [0.0, 1.0]],
            ]
        else:
            assert [result.tolist() for result in results] == [["A", "B", "A"], ["B", "A"]]

    assert len(fabric_runtime.requests) == 2
    assert len(created_loops) == 2
    assert created_loops[0] is not created_loops[1]
    assert created_loops == closed_loops
    assert all(loop.is_closed() for loop in created_loops)
    assert all(client.is_closed for client in clients)


@pytest.mark.parametrize("structured", [False, True])
def test_spark_fabric_closes_client_when_request_fails(fabric_runtime, monkeypatch, structured):
    from openaivec import spark_ext

    class Result(BaseModel):
        value: str

    monkeypatch.setattr(spark_ext, "pandas_udf", lambda **kwargs: lambda function: function)
    spark_ext.setup_fabric(SimpleNamespace(sparkContext=SimpleNamespace(environment={})))
    udf = spark_ext.responses_udf("Echo.", response_format=Result if structured else str, store=True)

    with pytest.raises(ValueError, match="store=False"):
        list(udf(iter([pd.Series(["a"])])))

    assert fabric_runtime.get_openai_httpx_async_client.return_value.is_closed
    assert fabric_runtime.requests == []


def test_spark_fabric_closes_client_when_iteration_stops(fabric_runtime, monkeypatch):
    from openaivec import spark_ext

    monkeypatch.setattr(spark_ext, "pandas_udf", lambda **kwargs: lambda function: function)
    spark_ext.setup_fabric(SimpleNamespace(sparkContext=SimpleNamespace(environment={})))
    udf = spark_ext.embeddings_udf(batch_size=2)
    outputs = udf(iter([pd.Series(["a"]), pd.Series(["b"])]))

    next(outputs)
    outputs.close()

    assert fabric_runtime.get_openai_httpx_async_client.return_value.is_closed
    assert len(fabric_runtime.requests) == 1


@pytest.mark.parametrize("setup_name", ["setup", "setup_azure", "setup_entra_id"])
def test_spark_setup_can_replace_fabric_authentication(fabric_runtime, monkeypatch, setup_name):
    from openaivec import spark_ext

    monkeypatch.setattr(spark_ext, "pandas_udf", lambda **kwargs: lambda function: function)
    spark = SimpleNamespace(sparkContext=SimpleNamespace(environment={}))
    spark_ext.setup_fabric(spark)
    kwargs = {
        "setup": {"api_key": "openai-key"},
        "setup_azure": {"api_key": "azure-key", "base_url": "https://example.openai.azure.com/openai/v1/"},
        "setup_entra_id": {
            "base_url": "https://example.openai.azure.com/openai/v1/",
            "tenant_id": "tenant",
            "client_id": "client",
            "client_secret": "secret",
        },
    }
    getattr(spark_ext, setup_name)(spark, **kwargs[setup_name])
    default_client = Mock(side_effect=RuntimeError("Default authentication selected"))
    monkeypatch.setattr(spark_ext, "get_async_client", default_client)
    udf = spark_ext.embeddings_udf()

    with pytest.raises(RuntimeError, match="Default authentication selected"):
        list(udf(iter([pd.Series(["a"])])))

    default_client.assert_called_once_with()
    fabric_runtime.get_openai_httpx_async_client.assert_not_called()


def test_partition_event_loop_closes_when_cleanup_fails():
    from openaivec._util import run_partition_async

    loops = []

    async def run_part(part):
        loops.append(asyncio.get_running_loop())
        return part

    async def cleanup():
        raise RuntimeError("Cleanup failed")

    with pytest.raises(RuntimeError, match="Cleanup failed"):
        list(run_partition_async(iter(["a"]), run_part, cleanup))

    assert loops[0].is_closed()
