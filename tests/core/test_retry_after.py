import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import format_datetime
from types import SimpleNamespace

import httpx
import pytest
from openai import APIStatusError, AsyncOpenAI, OpenAI

from openaivec import RetryPolicy, _retry


@dataclass
class Clock:
    now: float = 0
    delays: list[float] = field(default_factory=list)

    def monotonic(self):
        return self.now

    def time(self):
        return 1_700_000_000 + self.now

    def sleep(self, delay):
        self.delays.append(delay)
        self.now += delay

    async def asleep(self, delay):
        self.sleep(delay)


@pytest.fixture
def clock(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(
        _retry, "time", SimpleNamespace(monotonic=clock.monotonic, time=clock.time, sleep=clock.sleep)
    )
    monkeypatch.setattr(_retry, "asyncio", SimpleNamespace(sleep=clock.asleep, wait_for=asyncio.wait_for))
    monkeypatch.setattr(_retry.random, "uniform", lambda low, high: high)
    return clock


async def _invoke(handler, policy, asynchronous, *, deadline=None):
    transport = httpx.MockTransport(handler)
    client = (
        AsyncOpenAI(api_key="test", http_client=httpx.AsyncClient(transport=transport))
        if asynchronous
        else OpenAI(api_key="test", http_client=httpx.Client(transport=transport))
    )
    try:
        call = _retry.call_with_retry_async if asynchronous else _retry.call_with_retry
        result = call(
            client,
            policy,
            lambda request_client, options: request_client.embeddings.create(
                model="test-model", input=["input"], **options
            ),
            {},
            deadline=deadline,
        )
        return await result if asynchronous else result
    finally:
        assert client.max_retries == 2
        assert not client.is_closed()
        if asynchronous:
            await client.close()
        else:
            client.close()


def _error_response(status, retry_after, **headers):
    if retry_after is not None:
        headers["Retry-After"] = retry_after
    return httpx.Response(status, headers=headers, json={"error": {"message": "retry later"}})


def _success_response():
    return httpx.Response(200, json={"data": [{"index": 0, "embedding": [1.0]}]})


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("status", [429, 503])
@pytest.mark.parametrize("header_type", ["seconds", "date"])
async def test_retry_after_respects_server_wait(clock, asynchronous, status, header_type):
    requests = []
    retry_after = (
        "3"
        if header_type == "seconds"
        else format_datetime(datetime.fromtimestamp(clock.time() + 3, timezone.utc), usegmt=True)
    )

    def handler(request):
        requests.append(clock.now)
        return _error_response(status, retry_after) if len(requests) == 1 else _success_response()

    result = await _invoke(handler, RetryPolicy(initial_delay=0.5, max_delay=3), asynchronous)

    assert result.data[0].embedding == [1.0]
    assert requests == [0, 3]
    assert clock.delays == [3]


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("retry_after", [None, "invalid", "", "nan", "inf", "-inf", "-1", "1e999"])
async def test_retry_after_invalid_or_missing_uses_capped_jitter(clock, asynchronous, retry_after):
    requests = []

    def handler(request):
        requests.append(clock.now)
        return _error_response(429, retry_after) if len(requests) <= 3 else _success_response()

    await _invoke(handler, RetryPolicy(max_attempts=4, initial_delay=1, max_delay=2), asynchronous)

    assert requests == [0, 1, 3, 5]
    assert clock.delays == [1, 2, 2]


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("retry_after", ["0", "0.25", "past_date"])
async def test_retry_after_shorter_than_local_delay(clock, asynchronous, retry_after):
    requests = []
    expected = 0.25 if retry_after == "0.25" else 0
    if retry_after == "past_date":
        retry_after = format_datetime(datetime.fromtimestamp(clock.time() - 1, timezone.utc), usegmt=True)

    def handler(request):
        requests.append(clock.now)
        return _error_response(503, retry_after) if len(requests) == 1 else _success_response()

    await _invoke(handler, RetryPolicy(initial_delay=1), asynchronous)

    assert requests == [0, expected]
    assert clock.delays == [expected]


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("status", [429, 503])
@pytest.mark.parametrize("max_delay", [0, 2])
async def test_retry_after_above_delay_cap_propagates_original_error(clock, asynchronous, status, max_delay):
    requests = []

    def handler(request):
        requests.append(request)
        return _error_response(status, "3")

    with pytest.raises(APIStatusError) as caught:
        await _invoke(handler, RetryPolicy(max_delay=max_delay), asynchronous)

    assert caught.value.status_code == status
    assert caught.value.response.headers["Retry-After"] == "3"
    assert len(requests) == 1
    assert clock.delays == []


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("server_wait", ["2", "3"])
async def test_retry_after_respects_shared_deadline(clock, asynchronous, server_wait):
    requests = []

    def handler(request):
        requests.append(request)
        clock.now += 1
        return _error_response(429, server_wait)

    # An existing batch deadline must take precedence over a new 60-second budget.
    with pytest.raises(TimeoutError, match="Next transport retry would exceed the deadline") as caught:
        await _invoke(handler, RetryPolicy(max_elapsed=60), asynchronous, deadline=3)

    assert isinstance(caught.value.__cause__, APIStatusError)
    assert len(requests) == 1
    assert clock.delays == []


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("stop_reason", ["exhausted", "nonretryable", "server_opt_out"])
async def test_retry_after_does_not_override_stop_conditions(clock, asynchronous, stop_reason):
    requests = []
    status = 400 if stop_reason == "nonretryable" else 429
    headers = {"x-should-retry": "false"} if stop_reason == "server_opt_out" else {}

    def handler(request):
        requests.append(request)
        return _error_response(status, "1", **headers)

    policy = RetryPolicy(max_attempts=1 if stop_reason == "exhausted" else 3)
    with pytest.raises(APIStatusError):
        await _invoke(handler, policy, asynchronous)

    assert len(requests) == 1
    assert clock.delays == []
