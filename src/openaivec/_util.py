from __future__ import annotations

import asyncio
import functools
import re
import threading
import time
from collections.abc import Awaitable, Callable, Iterator
from dataclasses import dataclass
from typing import ParamSpec, TypeVar

import numpy as np
import tiktoken

__all__ = []

T = TypeVar("T")
U = TypeVar("U")
R = TypeVar("R")
P = ParamSpec("P")


# ---------------------------------------------------------------------------
# Async-to-sync bridge utilities
# ---------------------------------------------------------------------------


class BackgroundLoop:
    """Persistent event loop running on a daemon thread.

    Provides a ``run`` method that submits a coroutine and blocks until it
    completes.  Safe to call from any context, including Jupyter notebooks
    that already have a running event loop.  A single loop is reused for
    all calls to avoid per-invocation thread/loop creation overhead.
    """

    _instance: BackgroundLoop | None = None
    _lock: threading.Lock = threading.Lock()

    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()

    @classmethod
    def instance(cls) -> BackgroundLoop:
        """Return the singleton instance, creating it on first call."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def run(self, coro: Awaitable[T]) -> T:
        """Submit *coro* to the background loop and block for the result."""
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()


def run_async(coro: Awaitable[T]) -> T:
    """Run a coroutine from any context, including inside a running event loop.

    Delegates to the singleton ``BackgroundLoop``.
    """
    return BackgroundLoop.instance().run(coro)


def create_event_loop() -> asyncio.AbstractEventLoop:
    """Create a new event loop and set it as the current loop for this thread."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    return loop


def close_event_loop(loop: asyncio.AbstractEventLoop) -> None:
    """Gracefully shut down and close an event loop."""
    try:
        loop.run_until_complete(loop.shutdown_asyncgens())
        if hasattr(loop, "shutdown_default_executor"):
            loop.run_until_complete(loop.shutdown_default_executor())
    finally:
        asyncio.set_event_loop(None)
        loop.close()


PartitionResult = TypeVar("PartitionResult")


def run_partition_async(
    parts: Iterator[R],
    runner: Callable[[R], Awaitable[PartitionResult]],
    cleanup: Callable[[], Awaitable[None]],
) -> Iterator[PartitionResult]:
    """Run async partition work on a single reusable event loop.

    Creates a dedicated loop, iterates *parts* calling *runner* for each,
    then awaits *cleanup* and closes the loop.
    """
    loop = create_event_loop()
    try:
        for part in parts:
            yield loop.run_until_complete(runner(part))
    finally:
        loop.run_until_complete(cleanup())
        close_event_loop(loop)


def get_exponential_with_cutoff(scale: float) -> float:
    """Sample an exponential random variable with an upper cutoff.

    A value is repeatedly drawn from an exponential distribution with rate
    ``1/scale`` until it is smaller than ``3 * scale``.

    Args:
        scale (float): Scale parameter of the exponential distribution.

    Returns:
        float: Sampled value bounded by ``3 * scale``.
    """
    gen = np.random.default_rng()

    while True:
        v = gen.exponential(scale)
        if v < scale * 3:
            return v


def backoff(
    exceptions: list[type[Exception]],
    scale: float | None = None,
    max_retries: int | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator implementing exponential back‑off retry logic.

    Args:
        exceptions (list[type[Exception]]): List of exception types that trigger a retry.
        scale (int | None): Initial scale parameter for the exponential jitter.
            This scale is used as the mean for the first delay's exponential
            distribution and doubles with each subsequent retry. If ``None``,
            an initial scale of 1.0 is used.
        max_retries (int | None): Maximum number of retries. ``None`` means
            retry indefinitely.

    Returns:
        Callable[..., V]: A decorated function that retries on the specified
            exceptions with exponential back‑off.

    Raises:
        Exception: Re‑raised when the maximum number of retries is exceeded.
    """

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            attempt = 0
            # Initialize the scale for the exponential backoff. This scale will double with each retry.
            # If the input 'scale' is None, default to 1.0. This 'scale' is the mean of the exponential distribution.
            current_jitter_scale = float(scale) if scale is not None else 1.0

            while True:
                try:
                    return func(*args, **kwargs)
                except tuple(exceptions):
                    attempt += 1
                    if max_retries is not None and attempt >= max_retries:
                        raise

                    # Get the sleep interval with exponential jitter, using the current scale
                    interval = get_exponential_with_cutoff(current_jitter_scale)
                    time.sleep(interval)

                    # Double the scale for the next potential retry
                    current_jitter_scale *= 2

        return wrapper

    return decorator


def backoff_async(
    exceptions: list[type[Exception]],
    scale: float | None = None,
    max_retries: int | None = None,
) -> Callable[[Callable[P, Awaitable[R]]], Callable[P, Awaitable[R]]]:
    """Asynchronous version of the backoff decorator.

    Args:
        exceptions (list[type[Exception]]): List of exception types that trigger a retry.
        scale (int | None): Initial scale parameter for the exponential jitter.
            This scale is used as the mean for the first delay's exponential
            distribution and doubles with each subsequent retry. If ``None``,
            an initial scale of 1.0 is used.
        max_retries (int | None): Maximum number of retries. ``None`` means
            retry indefinitely.

    Returns:
        Callable[..., Awaitable[V]]: A decorated asynchronous function that
            retries on the specified exceptions with exponential back‑off.

    Raises:
        Exception: Re‑raised when the maximum number of retries is exceeded.
    """

    def decorator(func: Callable[P, Awaitable[R]]) -> Callable[P, Awaitable[R]]:
        @functools.wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            attempt = 0
            # Initialize the scale for the exponential backoff. This scale will double with each retry.
            # If the input 'scale' is None, default to 1.0. This 'scale' is the mean of the exponential distribution.
            current_jitter_scale = float(scale) if scale is not None else 1.0

            while True:
                try:
                    return await func(*args, **kwargs)
                except tuple(exceptions):
                    attempt += 1
                    if max_retries is not None and attempt >= max_retries:
                        raise

                    # Get the sleep interval with exponential jitter, using the current scale
                    interval = get_exponential_with_cutoff(current_jitter_scale)
                    await asyncio.sleep(interval)

                    # Double the scale for the next potential retry
                    current_jitter_scale *= 2

        return wrapper

    return decorator


@dataclass(frozen=True)
class TextChunker:
    """Utility for splitting text into token‑bounded chunks."""

    enc: tiktoken.Encoding

    def split(self, original: str, max_tokens: int, sep: list[str]) -> list[str]:
        """Token‑aware sentence segmentation.

        The text is first split by the given separators, then greedily packed
        into chunks whose token counts do not exceed ``max_tokens``.

        Args:
            original (str): Original text to split.
            max_tokens (int): Maximum number of tokens allowed per chunk.
            sep (list[str]): List of literal separator strings used for splitting.

        Returns:
            list[str]: List of text chunks respecting the ``max_tokens`` limit.

        Raises:
            ValueError: If any individual split segment exceeds ``max_tokens``.
        """
        pattern = f"({'|'.join(re.escape(s) for s in sep)})"
        sentences = re.split(pattern, original)
        sentences = [s.strip() for s in sentences if s.strip()]
        sentences = [(s, len(self.enc.encode(s))) for s in sentences]

        chunks = []
        sentence = ""
        token_count = 0
        for s, n in sentences:
            if n > max_tokens:
                raise ValueError(f"Segment exceeds max_tokens ({n} > {max_tokens}): {s!r}")
            if token_count + n > max_tokens:
                if sentence:
                    chunks.append(sentence)
                sentence = ""
                token_count = 0

            sentence += s
            token_count += n

        if sentence:
            chunks.append(sentence)

        return chunks
