import asyncio
import functools
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TypeVar

import numpy as np
import tiktoken

__all__ = []

T = TypeVar("T")
U = TypeVar("U")
V = TypeVar("V")


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
    scale: int | None = None,
    max_retries: int | None = None,
) -> Callable[..., V]:
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

    def decorator(func: Callable[..., V]) -> Callable[..., V]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> V:
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

    return decorator  # type: ignore[return-value]


def backoff_async(
    exceptions: list[type[Exception]],
    scale: int | None = None,
    max_retries: int | None = None,
) -> Callable[..., Awaitable[V]]:
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

    def decorator(func: Callable[..., Awaitable[V]]) -> Callable[..., Awaitable[V]]:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> V:
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

    return decorator  # type: ignore[return-value]


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
            sep (list[str]): List of separator patterns used by
                :pyfunc:`re.split`.

        Returns:
            list[str]: List of text chunks respecting the ``max_tokens`` limit.
        """
        sentences = re.split(f"({'|'.join(sep)})", original)
        sentences = [s.strip() for s in sentences if s.strip()]
        sentences = [(s, len(self.enc.encode(s))) for s in sentences]

        chunks = []
        sentence = ""
        token_count = 0
        for s, n in sentences:
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
