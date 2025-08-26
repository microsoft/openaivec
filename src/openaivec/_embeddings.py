from dataclasses import dataclass, field
from logging import Logger, getLogger

import numpy as np
from numpy.typing import NDArray
from openai import AsyncOpenAI, InternalServerError, OpenAI, RateLimitError

from openaivec._log import observe
from openaivec._proxy import AsyncBatchingMapProxy, BatchingMapProxy
from openaivec._util import backoff, backoff_async

__all__ = [
    "BatchEmbeddings",
    "AsyncBatchEmbeddings",
]

_LOGGER: Logger = getLogger(__name__)


@dataclass(frozen=True)
class BatchEmbeddings:
    """Thin wrapper around the OpenAI embeddings endpoint (synchronous).

    Attributes:
        client (OpenAI): Configured OpenAI client.
        model_name (str): For Azure OpenAI, use your deployment name. For OpenAI, use the model name
            (e.g., ``"text-embedding-3-small"``).
        cache (BatchingMapProxy[str, NDArray[np.float32]]): Batching proxy for ordered, cached mapping.
        api_kwargs (dict[str, Any]): Additional OpenAI API parameters stored at initialization.
    """

    client: OpenAI
    model_name: str
    cache: BatchingMapProxy[str, NDArray[np.float32]] = field(default_factory=lambda: BatchingMapProxy(batch_size=None))
    api_kwargs: dict[str, int | float | str | bool] = field(default_factory=dict)

    @classmethod
    def of(cls, client: OpenAI, model_name: str, batch_size: int | None = None, **api_kwargs) -> "BatchEmbeddings":
        """Factory constructor.

        Args:
            client (OpenAI): OpenAI client.
            model_name (str): For Azure OpenAI, use your deployment name. For OpenAI, use the model name.
            batch_size (int | None, optional): Max unique inputs per API call. Defaults to None
                (automatic batch size optimization). Set to a positive integer for fixed batch size.
            **api_kwargs: Additional OpenAI API parameters (e.g., dimensions for text-embedding-3 models).

        Returns:
            BatchEmbeddings: Configured instance backed by a batching proxy.
        """
        return cls(
            client=client,
            model_name=model_name,
            cache=BatchingMapProxy(batch_size=batch_size),
            api_kwargs=api_kwargs,
        )

    @observe(_LOGGER)
    @backoff(exceptions=[RateLimitError, InternalServerError], scale=1, max_retries=12)
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
        responses = self.client.embeddings.create(input=inputs, model=self.model_name, **self.api_kwargs)
        return [np.array(d.embedding, dtype=np.float32) for d in responses.data]

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
        cache (AsyncBatchingMapProxy[str, NDArray[np.float32]]): Async batching proxy.
        api_kwargs (dict): Additional OpenAI API parameters stored at initialization.
    """

    client: AsyncOpenAI
    model_name: str
    cache: AsyncBatchingMapProxy[str, NDArray[np.float32]] = field(
        default_factory=lambda: AsyncBatchingMapProxy(batch_size=None, max_concurrency=8)
    )
    api_kwargs: dict[str, int | float | str | bool] = field(default_factory=dict)

    @classmethod
    def of(
        cls,
        client: AsyncOpenAI,
        model_name: str,
        batch_size: int | None = None,
        max_concurrency: int = 8,
        **api_kwargs,
    ) -> "AsyncBatchEmbeddings":
        """Factory constructor.

        Args:
            client (AsyncOpenAI): OpenAI async client.
            model_name (str): For Azure OpenAI, use your deployment name. For OpenAI, use the model name.
            batch_size (int | None, optional): Max unique inputs per API call. Defaults to None
                (automatic batch size optimization). Set to a positive integer for fixed batch size.
            max_concurrency (int, optional): Max concurrent API calls. Defaults to 8.
            **api_kwargs: Additional OpenAI API parameters (e.g., dimensions for text-embedding-3 models).

        Returns:
            AsyncBatchEmbeddings: Configured instance with an async batching proxy.
        """
        return cls(
            client=client,
            model_name=model_name,
            cache=AsyncBatchingMapProxy(batch_size=batch_size, max_concurrency=max_concurrency),
            api_kwargs=api_kwargs,
        )

    @backoff_async(exceptions=[RateLimitError, InternalServerError], scale=1, max_retries=12)
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
        responses = await self.client.embeddings.create(input=inputs, model=self.model_name, **self.api_kwargs)
        return [np.array(d.embedding, dtype=np.float32) for d in responses.data]

    @observe(_LOGGER)
    async def create(self, inputs: list[str]) -> list[NDArray[np.float32]]:
        """Generate embeddings for inputs using proxy batching (async).

        Args:
            inputs (list[str]): Input strings. Duplicates allowed.

        Returns:
            list[NDArray[np.float32]]: Embedding vectors aligned to ``inputs``.
        """
        return await self.cache.map(inputs, self._embed_chunk)  # type: ignore[arg-type]
