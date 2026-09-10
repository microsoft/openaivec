import asyncio
from dataclasses import dataclass, field
from types import SimpleNamespace

import httpx
import pytest
from openai import AsyncOpenAI, BadRequestError, OpenAI, RateLimitError

from openaivec import AsyncBatchEmbeddings, BatchEmbeddings, RetryPolicy, _retry


@dataclass
class Clock:
    now: float = 0
    delays: list[float] = field(default_factory=list)

    def monotonic(self):
        return self.now

    def sleep(self, delay):
        self.delays.append(delay)
        self.now += delay

    async def asleep(self, delay):
        self.sleep(delay)


@pytest.fixture
def clock(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(_retry, "time", SimpleNamespace(monotonic=clock.monotonic, sleep=clock.sleep))
    monkeypatch.setattr(_retry, "asyncio", SimpleNamespace(sleep=clock.asleep, wait_for=asyncio.wait_for))
    monkeypatch.setattr(_retry.random, "uniform", lambda low, high: high)
    return clock


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("deadline", [None, 5.0])
async def test_retry_delay_and_deadline(clock, asynchronous, deadline):
    requests = []

    def handler(request):
        requests.append(request)
        clock.now += 1
        return httpx.Response(429, json={"error": {"message": "forced failure"}})

    transport = httpx.MockTransport(handler)
    if asynchronous:
        client = AsyncOpenAI(api_key="test", http_client=httpx.AsyncClient(transport=transport))
        wrapper_type = AsyncBatchEmbeddings
    else:
        client = OpenAI(api_key="test", http_client=httpx.Client(transport=transport))
        wrapper_type = BatchEmbeddings
    policy = RetryPolicy(max_attempts=5, initial_delay=2, max_delay=3, max_elapsed=deadline)
    wrapper = wrapper_type.of(client, "test-model", batch_size=0, retry_policy=policy)
    wrapper.cache.show_progress = False
    try:
        with pytest.raises(TimeoutError if deadline is not None else RateLimitError):
            result = wrapper.create(["input"])
            if asynchronous:
                await result
        assert len(requests) == (2 if deadline is not None else 5)
        assert clock.delays == ([2] if deadline is not None else [2, 3, 3, 3])
        assert client.max_retries == 2
        assert not client.is_closed()
    finally:
        if asynchronous:
            await client.close()
        else:
            client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("request_timeout", [None, 30.0])
@pytest.mark.parametrize("elapsed", [1.0, 6.0])
async def test_retry_deadline_caps_timeout_and_rejects_late_result(clock, asynchronous, request_timeout, elapsed):
    requests = []

    def handler(request):
        requests.append(request)
        clock.now += elapsed
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [1.0]}]})

    transport = httpx.MockTransport(handler)
    timeout = httpx.Timeout(10, connect=2)
    if asynchronous:
        client = AsyncOpenAI(api_key="test", timeout=timeout, http_client=httpx.AsyncClient(transport=transport))
        wrapper_type = AsyncBatchEmbeddings
    else:
        client = OpenAI(api_key="test", timeout=timeout, http_client=httpx.Client(transport=transport))
        wrapper_type = BatchEmbeddings
    options = {} if request_timeout is None else {"timeout": request_timeout}
    wrapper = wrapper_type.of(client, "test-model", retry_policy=RetryPolicy(max_elapsed=5), **options)
    wrapper.cache.show_progress = False
    try:
        if elapsed > 5:
            with pytest.raises(TimeoutError):
                result = wrapper.create(["input"])
                if asynchronous:
                    await result
        else:
            result = wrapper.create(["input"])
            if asynchronous:
                result = await result
            assert result[0].tolist() == [1.0]
        assert len(requests) == 1
        assert requests[0].extensions["timeout"] == {
            "connect": 2 if request_timeout is None else 5,
            "read": 5,
            "write": 5,
            "pool": 5,
        }
        assert client.timeout == timeout
        assert client.max_retries == 2
        assert not client.is_closed()
    finally:
        if asynchronous:
            await client.close()
        else:
            client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_nonretryable_error_is_not_retried(clock, asynchronous):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(400, json={"error": {"message": "invalid request"}})

    transport = httpx.MockTransport(handler)
    if asynchronous:
        client = AsyncOpenAI(api_key="test", http_client=httpx.AsyncClient(transport=transport))
        wrapper_type = AsyncBatchEmbeddings
    else:
        client = OpenAI(api_key="test", http_client=httpx.Client(transport=transport))
        wrapper_type = BatchEmbeddings
    wrapper = wrapper_type.of(client, "test-model", retry_policy=RetryPolicy())
    wrapper.cache.show_progress = False
    try:
        with pytest.raises(BadRequestError):
            result = wrapper.create(["input"])
            if asynchronous:
                await result
        assert len(requests) == 1
        assert clock.delays == []
    finally:
        if asynchronous:
            await client.close()
        else:
            client.close()
