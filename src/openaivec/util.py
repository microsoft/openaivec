import asyncio
import functools
import re
import time
from dataclasses import dataclass
from typing import Awaitable, Callable, Dict, List, TypeVar

import numpy as np
import tiktoken


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


def backoff(exception: type[Exception], scale: int | None = None, max_retries: int | None = None) -> Callable[..., V]:
    """Decorator implementing exponential back‑off retry logic.

    Args:
        exception (type[Exception]): Exception type that triggers a retry.
        scale (int | None): Initial scale parameter for the exponential jitter.
            This scale is used as the mean for the first delay's exponential
            distribution and doubles with each subsequent retry. If ``None``,
            an initial scale of 1.0 is used.
        max_retries (Optional[int]): Maximum number of retries. ``None`` means
            retry indefinitely.

    Returns:
        Callable[..., V]: A decorated function that retries on the specified
            exception with exponential back‑off.

    Raises:
        exception: Re‑raised when the maximum number of retries is exceeded.
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
                except exception:
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
    exception: type[Exception], scale: int | None = None, max_retries: int | None = None
) -> Callable[..., Awaitable[V]]:
    """Asynchronous version of the backoff decorator.

    Args:
        exception (type[Exception]): Exception type that triggers a retry.
        scale (int | None): Initial scale parameter for the exponential jitter.
            This scale is used as the mean for the first delay's exponential
            distribution and doubles with each subsequent retry. If ``None``,
            an initial scale of 1.0 is used.
        max_retries (int | None): Maximum number of retries. ``None`` means
            retry indefinitely.

    Returns:
        Callable[..., Awaitable[V]]: A decorated asynchronous function that
            retries on the specified exception with exponential back‑off.

    Raises:
        exception: Re‑raised when the maximum number of retries is exceeded.
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
                except exception:
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

    def split(self, original: str, max_tokens: int, sep: List[str]) -> List[str]:
        """Token‑aware sentence segmentation.

        The text is first split by the given separators, then greedily packed
        into chunks whose token counts do not exceed ``max_tokens``.

        Args:
            original (str): Original text to split.
            max_tokens (int): Maximum number of tokens allowed per chunk.
            sep (List[str]): List of separator patterns used by
                :pyfunc:`re.split`.

        Returns:
            List[str]: List of text chunks respecting the ``max_tokens`` limit.
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


async def map_async(inputs: List[T], f: Callable[[List[T]], Awaitable[List[U]]], batch_size: int = 128) -> List[U]:
    """Asynchronously map a function `f` over a list of inputs in batches.

    This function divides the input list into smaller batches and applies the
    asynchronous function `f` to each batch concurrently. It gathers the results
    and returns them in the same order as the original inputs.

    Args:
        inputs (List[T]): List of inputs to be processed.
        f (Callable[[List[T]], Awaitable[List[U]]]): Asynchronous function to apply.
            It takes a batch of inputs (List[T]) and must return a list of
            corresponding outputs (List[U]) of the same size.
        batch_size (int): Size of each batch for processing.

    Returns:
        List[U]: List of outputs corresponding to the original inputs, in order.
    """
    original_hashes: List[int] = [hash(str(v)) for v in inputs]  # Use str(v) for hash if T is not hashable
    hash_inputs: Dict[int, T] = {k: v for k, v in zip(original_hashes, inputs)}
    unique_hashes: List[int] = list(hash_inputs.keys())
    unique_inputs: List[T] = list(hash_inputs.values())
    input_batches: List[List[T]] = [unique_inputs[i : i + batch_size] for i in range(0, len(unique_inputs), batch_size)]
    # Ensure f is awaited correctly within gather
    tasks = [f(batch) for batch in input_batches]
    output_batches: List[List[U]] = await asyncio.gather(*tasks)
    unique_outputs: List[U] = [u for batch in output_batches for u in batch]
    if len(unique_hashes) != len(unique_outputs):
        raise ValueError(
            f"Number of unique inputs ({len(unique_hashes)}) does not match number of unique outputs ({len(unique_outputs)}). Check the function f."
        )
    hash_outputs: Dict[int, U] = {k: v for k, v in zip(unique_hashes, unique_outputs)}
    outputs: List[U] = [hash_outputs[k] for k in original_hashes]
    return outputs


def map(inputs: List[T], f: Callable[[List[T]], List[U]], batch_size: int = 128) -> List[U]:
    """Map a function `f` over a list of inputs in batches.

    This function divides the input list into smaller batches and applies the
    function `f` to each batch. It gathers the results and returns them in the
    same order as the original inputs.

    Args:
        inputs (List[T]): List of inputs to be processed.
        f (Callable[[List[T]], List[U]]): Function to apply. It takes a batch of
            inputs (List[T]) and must return a list of corresponding outputs
            (List[U]) of the same size.
        batch_size (int): Size of each batch for processing.

    Returns:
        List[U]: List of outputs corresponding to the original inputs, in order.
    """
    original_hashes: List[int] = [hash(str(v)) for v in inputs]  # Use str(v) for hash if T is not hashable
    hash_inputs: Dict[int, T] = {k: v for k, v in zip(original_hashes, inputs)}
    unique_hashes: List[int] = list(hash_inputs.keys())
    unique_inputs: List[T] = list(hash_inputs.values())
    input_batches: List[List[T]] = [unique_inputs[i : i + batch_size] for i in range(0, len(unique_inputs), batch_size)]
    output_batches: List[List[U]] = [f(batch) for batch in input_batches]
    unique_outputs: List[U] = [u for batch in output_batches for u in batch]
    if len(unique_hashes) != len(unique_outputs):
        raise ValueError(
            f"Number of unique inputs ({len(unique_hashes)}) does not match number of unique outputs ({len(unique_outputs)}). Check the function f."
        )
    hash_outputs: Dict[int, U] = {k: v for k, v in zip(unique_hashes, unique_outputs)}
    outputs: List[U] = [hash_outputs[k] for k in original_hashes]
    return outputs
