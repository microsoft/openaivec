from dataclasses import dataclass, field
from logging import Logger, getLogger
from typing import Any

import numpy as np
import tiktoken
from numpy.typing import NDArray
from openai import AsyncOpenAI, InternalServerError, OpenAI, RateLimitError
from openai.types import Embedding

from openaivec._cache import AsyncBatchCache, BatchCache
from openaivec._cache.proxy import DEFAULT_MANAGED_CACHE_SIZE
from openaivec._log import observe
from openaivec._util import backoff, backoff_async

__all__ = []

_LOGGER: Logger = getLogger(__name__)


@dataclass(frozen=True)
class EmbeddingLimits:
    """Hard request limits for the selected embedding provider.

    Attributes:
        max_inputs (int): Maximum strings per request. Defaults to 2048.
        max_input_tokens (int): Maximum tokens per string. Defaults to 8192.
        max_request_tokens (int): Maximum aggregate tokens. Defaults to 300000.
        encoding_name (str | None): Explicit tiktoken encoding for a custom
            provider. None selects the model encoding, with cl100k_base for
            deployment aliases. All numeric limits must be positive.
    """

    max_inputs: int = 2048
    max_input_tokens: int = 8192
    max_request_tokens: int = 300000
    encoding_name: str | None = None

    def __post_init__(self) -> None:
        for name in ("max_inputs", "max_input_tokens", "max_request_tokens"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{name} must be an integer")
            if value < 1:
                raise ValueError(f"{name} must be > 0")


def _plan_embedding_batches(inputs: list[str], model_name: str, limits: EmbeddingLimits) -> list[list[str]]:
    if limits.encoding_name is not None:
        encoding = tiktoken.get_encoding(limits.encoding_name)
    else:
        try:
            encoding = tiktoken.encoding_for_model(model_name)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")
    batches: list[list[str]] = []
    batch: list[str] = []
    batch_tokens = 0
    for index, text in enumerate(inputs):
        if not isinstance(text, str):
            raise TypeError(f"Embedding input at index {index} must be a string")
        token_count = len(encoding.encode_ordinary(text))
        if not token_count:
            raise ValueError(f"Embedding input at index {index} is empty")
        input_limit = min(limits.max_input_tokens, limits.max_request_tokens)
        if token_count > input_limit:
            raise ValueError(f"Embedding input at index {index} exceeds {input_limit} tokens ({token_count})")
        if len(batch) == limits.max_inputs or batch_tokens + token_count > limits.max_request_tokens:
            batches.append(batch)
            batch = []
            batch_tokens = 0
        batch.append(text)
        batch_tokens += token_count
    if batch:
        batches.append(batch)
    return batches


def _ordered_embedding_rows(data: list[Embedding], expected_count: int) -> list[NDArray[np.float32]]:
    if len(data) != expected_count or {item.index for item in data} != set(range(expected_count)):
        raise ValueError("Embedding response indices must match the requested inputs exactly")
    return _as_float32_rows([item.embedding for item in sorted(data, key=lambda item: item.index)])


def _as_float32_rows(raw_embeddings: list[list[float]]) -> list[NDArray[np.float32]]:
    """Convert a batch of embedding payloads into float32 row views.

    The returned arrays share one contiguous float32 backing matrix, which
    avoids one allocation per embedding while keeping the public return type
    as ``list[np.ndarray]``.
    """
    if not raw_embeddings:
        return []
    matrix = np.asarray(raw_embeddings, dtype=np.float32)
    return list(matrix)


@dataclass(frozen=True)
class BatchEmbeddings:
    """Thin wrapper around the OpenAI embeddings endpoint (synchronous).

    API requests are limited to 2,048 inputs and 300,000 total tokens, even
    with automatic or nonpositive batch sizes. Empty inputs and inputs over
    8,192 tokens are rejected before sending the affected cache batch. Text
    is never truncated. Deployment aliases use the ``cl100k_base`` tokenizer
    shared by the supported OpenAI embedding models.

    Attributes:
        client (OpenAI): Configured OpenAI client.
        model_name (str): For Azure OpenAI, use your deployment name. For OpenAI, use the model name
            (e.g., ``"text-embedding-3-small"``).
        cache (BatchCache[str, NDArray[np.float32]]): Batching proxy for
            ordered, cached mapping. Library-managed instances use bounded
            retention by default.
        api_kwargs (dict[str, Any]): Additional OpenAI API parameters stored at initialization.
        limits (EmbeddingLimits): Provider request limits, independent of cache batch size.
    """

    client: OpenAI
    model_name: str
    cache: BatchCache[str, NDArray[np.float32]] = field(
        default_factory=lambda: BatchCache(batch_size=None, max_cache_size=DEFAULT_MANAGED_CACHE_SIZE)
    )
    api_kwargs: dict[str, Any] = field(default_factory=dict)
    limits: EmbeddingLimits = field(default_factory=EmbeddingLimits)

    @classmethod
    def of(
        cls,
        client: OpenAI,
        model_name: str,
        batch_size: int | None = None,
        *,
        limits: EmbeddingLimits | None = None,
        **api_kwargs,
    ) -> "BatchEmbeddings":
        """Factory constructor.

        Args:
            client (OpenAI): OpenAI client.
            model_name (str): For Azure OpenAI, use your deployment name. For OpenAI, use the model name.
            batch_size (int | None, optional): Max unique inputs per API call. Defaults to None
                (automatic batch size optimization). Set to a positive integer for fixed batch size.
            **api_kwargs: Additional OpenAI API parameters (e.g., dimensions for text-embedding-3 models).
            limits (EmbeddingLimits | None): Provider-specific hard limits. None uses OpenAI defaults.

        Returns:
            BatchEmbeddings: Configured instance backed by a batching proxy.
        """
        return cls(
            client=client,
            model_name=model_name,
            cache=BatchCache(batch_size=batch_size, max_cache_size=DEFAULT_MANAGED_CACHE_SIZE),
            api_kwargs=api_kwargs,
            limits=limits if limits is not None else EmbeddingLimits(),
        )

    @observe(_LOGGER)
    def _embed_chunk(self, inputs: list[str]) -> list[NDArray[np.float32]]:
        """Embed one minibatch of strings.

        This private helper is the unit of work used by the map/parallel
        utilities.  Exponential back‑off is applied automatically when
        ``openai.RateLimitError`` is raised.

        Args:
            inputs (list[str]): Input strings to be embedded. Duplicates allowed.

        Returns:
            list[NDArray[np.float32]]: Embedding vectors aligned to ``inputs``.
        """
        rows: list[NDArray[np.float32]] = []
        for batch in _plan_embedding_batches(inputs, self.model_name, self.limits):
            rows.extend(self._request_embeddings(batch))
        return rows

    @backoff(exceptions=[RateLimitError, InternalServerError], scale=1, max_retries=12)
    def _request_embeddings(self, inputs: list[str]) -> list[NDArray[np.float32]]:
        responses = self.client.embeddings.create(input=inputs, model=self.model_name, **self.api_kwargs)
        return _ordered_embedding_rows(responses.data, len(inputs))

    @observe(_LOGGER)
    def create(self, inputs: list[str]) -> list[NDArray[np.float32]]:
        """Generate embeddings for inputs using cached, ordered batching.

        Args:
            inputs (list[str]): Input strings. Duplicates allowed.

        Returns:
            list[NDArray[np.float32]]: Embedding vectors aligned to ``inputs``.
        """
        return self.cache.map(inputs, self._embed_chunk)


@dataclass(frozen=True)
class AsyncBatchEmbeddings:
    """Thin wrapper around the OpenAI embeddings endpoint (asynchronous).

    This class provides an asynchronous interface for generating embeddings using
    OpenAI models. It manages concurrency, handles rate limits automatically,
    and efficiently processes batches of inputs, including de-duplication.

    Request limits and input validation match ``BatchEmbeddings``. Splitting
    a cache batch does not increase concurrency; subrequests run sequentially
    within the existing cache worker.

    Example:
        ```python
        import asyncio
        import numpy as np
        from openai import AsyncOpenAI
        from openaivec import AsyncBatchEmbeddings

        # Assuming openai_async_client is an initialized AsyncOpenAI client
        openai_async_client = AsyncOpenAI() # Replace with your actual client initialization

        embedder = AsyncBatchEmbeddings.of(
            client=openai_async_client,
            model_name="text-embedding-3-small",
            batch_size=128,
            max_concurrency=8,
        )
        texts = ["This is the first document.", "This is the second document.", "This is the first document."]

        # Asynchronous call
        async def main():
            embeddings = await embedder.create(texts)
            # embeddings will be a list of numpy arrays (float32)
            # The embedding for the third text will be identical to the first
            # due to automatic de-duplication.
            print(f"Generated {len(embeddings)} embeddings.")
            print(f"Shape of first embedding: {embeddings[0].shape}")
            assert np.array_equal(embeddings[0], embeddings[2])

        # Run the async function
        asyncio.run(main())
        ```

    Attributes:
        client (AsyncOpenAI): Configured OpenAI async client.
        model_name (str): For Azure OpenAI, use your deployment name. For OpenAI, use the model name.
        cache (AsyncBatchCache[str, NDArray[np.float32]]): Async batching
            proxy. Library-managed instances use bounded retention by default.
        api_kwargs (dict): Additional OpenAI API parameters stored at initialization.
        limits (EmbeddingLimits): Provider request limits, independent of cache batch size.
    """

    client: AsyncOpenAI
    model_name: str
    cache: AsyncBatchCache[str, NDArray[np.float32]] = field(
        default_factory=lambda: AsyncBatchCache(
            batch_size=None,
            max_concurrency=8,
            max_cache_size=DEFAULT_MANAGED_CACHE_SIZE,
        )
    )
    api_kwargs: dict[str, Any] = field(default_factory=dict)
    limits: EmbeddingLimits = field(default_factory=EmbeddingLimits)

    @classmethod
    def of(
        cls,
        client: AsyncOpenAI,
        model_name: str,
        batch_size: int | None = None,
        max_concurrency: int = 8,
        *,
        limits: EmbeddingLimits | None = None,
        **api_kwargs,
    ) -> "AsyncBatchEmbeddings":
        """Factory constructor.

        Args:
            client (AsyncOpenAI): OpenAI async client.
            model_name (str): For Azure OpenAI, use your deployment name. For OpenAI, use the model name.
            batch_size (int | None, optional): Max unique inputs per API call. Defaults to None
                (automatic batch size optimization). Set to a positive integer for fixed batch size.
            max_concurrency (int, optional): Max concurrent API calls. Defaults to 8.
            limits (EmbeddingLimits | None): Provider-specific hard limits. None uses OpenAI defaults.
            **api_kwargs: Additional OpenAI API parameters (e.g., dimensions for text-embedding-3 models).

        Returns:
            AsyncBatchEmbeddings: Configured instance with an async batching proxy.
        """
        return cls(
            client=client,
            model_name=model_name,
            cache=AsyncBatchCache(
                batch_size=batch_size,
                max_concurrency=max_concurrency,
                max_cache_size=DEFAULT_MANAGED_CACHE_SIZE,
            ),
            api_kwargs=api_kwargs,
            limits=limits if limits is not None else EmbeddingLimits(),
        )

    @observe(_LOGGER)
    async def _embed_chunk(self, inputs: list[str]) -> list[NDArray[np.float32]]:
        """Embed one minibatch of strings asynchronously.

        This private helper handles the actual API call for a batch of inputs.
        Exponential back-off is applied automatically when ``openai.RateLimitError``
        is raised.

        Args:
            inputs (list[str]): Input strings to be embedded. Duplicates allowed.

        Returns:
            list[NDArray[np.float32]]: Embedding vectors aligned to ``inputs``.

        Raises:
            RateLimitError: Propagated if retries are exhausted.
        """
        rows: list[NDArray[np.float32]] = []
        for batch in _plan_embedding_batches(inputs, self.model_name, self.limits):
            rows.extend(await self._request_embeddings(batch))
        return rows

    @backoff_async(exceptions=[RateLimitError, InternalServerError], scale=1, max_retries=12)
    async def _request_embeddings(self, inputs: list[str]) -> list[NDArray[np.float32]]:
        responses = await self.client.embeddings.create(input=inputs, model=self.model_name, **self.api_kwargs)
        return _ordered_embedding_rows(responses.data, len(inputs))

    @observe(_LOGGER)
    async def create(self, inputs: list[str]) -> list[NDArray[np.float32]]:
        """Generate embeddings for inputs using proxy batching (async).

        Args:
            inputs (list[str]): Input strings. Duplicates allowed.

        Returns:
            list[NDArray[np.float32]]: Embedding vectors aligned to ``inputs``.
        """
        return await self.cache.map(inputs, self._embed_chunk)  # type: ignore[arg-type]
