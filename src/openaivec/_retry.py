import asyncio
import math
import random
import time
from asyncio import TimeoutError as AsyncTimeoutError
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from logging import getLogger
from typing import Any, TypeVar

import httpx
from openai import APIConnectionError, APIStatusError, AsyncOpenAI, NotGiven, OpenAI, Timeout

__all__ = []

Result = TypeVar("Result")
_LOGGER = getLogger(__name__)


@dataclass(frozen=True)
class RetryPolicy:
    """Explicit transport retry limits, replacing SDK retries for each request.

    Attributes:
        max_attempts (int): Total HTTP attempts, including the initial attempt.
            Set to 1 for fail-fast behavior. Defaults to 3.
        initial_delay (float): Initial full-jitter delay ceiling in seconds.
            Defaults to 0.5. Set to zero to disable retry delays.
        max_delay (float): Fixed ceiling for each retry delay in seconds.
            Defaults to 8.0.
        max_elapsed (float | None): Optional elapsed-time budget in seconds.
            Async calls are cancelled at the deadline. Sync calls cap HTTP
            timeouts and reject late results but cannot forcibly interrupt
            a blocking transport. Defaults to None.
    """

    max_attempts: int = 3
    initial_delay: float = 0.5
    max_delay: float = 8.0
    max_elapsed: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.max_attempts, int) or isinstance(self.max_attempts, bool):
            raise TypeError("max_attempts must be an integer")
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        for name in ("initial_delay", "max_delay", "max_elapsed"):
            value = getattr(self, name)
            if name == "max_elapsed" and value is None:
                continue
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise TypeError(f"{name} must be a number")
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and >= 0")
            if name == "max_elapsed" and value == 0:
                raise ValueError("max_elapsed must be > 0")


def retry_deadline(policy: RetryPolicy | None) -> float | None:
    return time.monotonic() + policy.max_elapsed if policy is not None and policy.max_elapsed is not None else None


def _remaining(deadline: float | None) -> float | None:
    if deadline is None:
        return None
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("Transport retry deadline exceeded")
    return remaining


def _request_options(client: OpenAI | AsyncOpenAI, options: dict[str, Any], remaining: float | None) -> dict[str, Any]:
    if remaining is None:
        return options
    timeout = options.get("timeout", client.timeout)
    if isinstance(timeout, NotGiven):
        timeout = client.timeout
    if not isinstance(timeout, (httpx.Timeout, Timeout)):
        timeout = httpx.Timeout(timeout)
    bounded = {name: remaining if value is None else min(value, remaining) for name, value in timeout.as_dict().items()}
    return {**options, "timeout": httpx.Timeout(**bounded)}


def _retry_delay(
    policy: RetryPolicy, delay: float, attempt: int, error: APIConnectionError | APIStatusError, deadline: float | None
) -> float:
    if not _is_retryable(error):
        raise error
    remaining = _remaining(deadline)
    if attempt + 1 == policy.max_attempts:
        _LOGGER.warning("Transport retries exhausted after %d attempt(s): %s", attempt + 1, type(error).__name__)
        raise error
    interval = random.uniform(0, delay)
    if remaining is not None and interval >= remaining:
        raise TimeoutError("Next transport retry would exceed the deadline") from error
    return interval


def _is_retryable(error: APIConnectionError | APIStatusError) -> bool:
    if isinstance(error, APIConnectionError):
        return True
    if error.response.headers.get("x-should-retry") == "false":
        return False
    return error.status_code in (408, 409, 429) or error.status_code >= 500


def call_with_retry(
    client: OpenAI,
    policy: RetryPolicy | None,
    operation: Callable[[OpenAI, dict[str, Any]], Result],
    options: dict[str, Any],
    *,
    deadline: float | None = None,
) -> Result:
    if policy is None:
        return operation(client, options)
    if deadline is None:
        deadline = retry_deadline(policy)
    request_client = client.with_options(max_retries=0)
    delay = min(policy.initial_delay, policy.max_delay)
    for attempt in range(policy.max_attempts):
        try:
            result = operation(request_client, _request_options(request_client, options, _remaining(deadline)))
            _remaining(deadline)
            return result
        except (APIConnectionError, APIStatusError) as error:
            time.sleep(_retry_delay(policy, delay, attempt, error, deadline))
            delay = min(policy.max_delay, delay * 2)
    raise RuntimeError("unreachable transport retry loop state")


async def call_with_retry_async(
    client: AsyncOpenAI,
    policy: RetryPolicy | None,
    operation: Callable[[AsyncOpenAI, dict[str, Any]], Awaitable[Result]],
    options: dict[str, Any],
    *,
    deadline: float | None = None,
) -> Result:
    if policy is None:
        return await operation(client, options)
    if deadline is None:
        deadline = retry_deadline(policy)
    request_client = client.with_options(max_retries=0)
    delay = min(policy.initial_delay, policy.max_delay)
    for attempt in range(policy.max_attempts):
        try:
            remaining = _remaining(deadline)
            pending = operation(request_client, _request_options(request_client, options, remaining))
            result = await pending if remaining is None else await asyncio.wait_for(pending, remaining)
            _remaining(deadline)
            return result
        except AsyncTimeoutError as error:
            raise TimeoutError("Transport retry deadline exceeded") from error
        except (APIConnectionError, APIStatusError) as error:
            await asyncio.sleep(_retry_delay(policy, delay, attempt, error, deadline))
            delay = min(policy.max_delay, delay * 2)
    raise RuntimeError("unreachable transport retry loop state")
