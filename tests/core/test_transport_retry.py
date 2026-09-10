import httpx
import pytest
from openai import AsyncOpenAI, InternalServerError, OpenAI, RateLimitError

from openaivec import AsyncBatchEmbeddings, AsyncBatchResponses, BatchEmbeddings, BatchResponses, _util


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("endpoint", ["embeddings", "responses"])
@pytest.mark.parametrize("sdk_retries", [0, 2])
@pytest.mark.parametrize("attempts", [None, 1, 3])
@pytest.mark.parametrize("status_code,error_type", [(429, RateLimitError), (500, InternalServerError)])
async def test_default_transport_has_one_retry_owner(
    monkeypatch, asynchronous, endpoint, sdk_retries, attempts, status_code, error_type
):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(status_code, json={"error": {"message": "forced transient failure"}})

    monkeypatch.setattr(_util, "get_exponential_with_cutoff", lambda scale: 0)
    monkeypatch.setattr(_util.time, "sleep", lambda delay: None)
    transport = httpx.MockTransport(handler)
    if asynchronous:
        client = AsyncOpenAI(
            api_key="test", max_retries=sdk_retries, http_client=httpx.AsyncClient(transport=transport)
        )
        wrapper_type = AsyncBatchEmbeddings if endpoint == "embeddings" else AsyncBatchResponses
    else:
        client = OpenAI(api_key="test", max_retries=sdk_retries, http_client=httpx.Client(transport=transport))
        wrapper_type = BatchEmbeddings if endpoint == "embeddings" else BatchResponses
    monkeypatch.setattr(client, "_calculate_retry_timeout", lambda *args, **kwargs: 0)
    options = {} if endpoint == "embeddings" else {"system_message": "echo"}
    if attempts is not None:
        from openaivec import RetryPolicy

        options["retry_policy"] = RetryPolicy(max_attempts=attempts, initial_delay=0)
    wrapper = wrapper_type.of(client, "test-model", batch_size=0, **options)
    wrapper.cache.show_progress = False
    method = wrapper.create if endpoint == "embeddings" else wrapper.parse
    try:
        with pytest.raises(error_type):
            result = method(["input"])
            if asynchronous:
                await result
        assert len(requests) == (sdk_retries + 1 if attempts is None else attempts)
        assert client.max_retries == sdk_retries
        assert not client.is_closed()
    finally:
        if asynchronous:
            await client.close()
        else:
            client.close()
