import asyncio
from dataclasses import dataclass, field
from logging import Logger, getLogger
from typing import List

import numpy as np
from numpy.typing import NDArray
from openai import AsyncOpenAI, OpenAI, RateLimitError

from .log import observe
from .util import backoff, backoff_async, map, map_async

__all__ = [
    "BatchEmbeddings",
    "AsyncBatchEmbeddings",
]

_LOGGER: Logger = getLogger(__name__)


@dataclass(frozen=True)
class BatchEmbeddings:
    """Thin wrapper around the OpenAI /embeddings endpoint.

    Attributes:
        client: An already‑configured ``openai.OpenAI`` client.
        model_name: The model identifier, e.g. ``"text-embedding-3-small"``.
    """

    client: OpenAI
    model_name: str

    @observe(_LOGGER)
    @backoff(exception=RateLimitError, scale=15, max_retries=8)
    def _embed_chunk(self, inputs: List[str]) -> List[NDArray[np.float32]]:
        """Embed one minibatch of sentences.

        This private helper is the unit of work used by the map/parallel
        utilities.  Exponential back‑off is applied automatically when
        ``openai.RateLimitError`` is raised.

        Args:
            inputs: Input strings to be embedded.  Duplicates are allowed; the
                implementation may decide to de‑duplicate internally.

        Returns:
            List of embedding vectors with the same ordering as *sentences*.
        """
        responses = self.client.embeddings.create(input=inputs, model=self.model_name)
        return [np.array(d.embedding, dtype=np.float32) for d in responses.data]

    @observe(_LOGGER)
    def create(self, inputs: List[str], batch_size: int) -> List[NDArray[np.float32]]:
        """See ``VectorizedEmbeddings.create`` for contract details.

        The call is internally delegated to either ``map_unique_minibatch`` or
        its parallel counterpart depending on *is_parallel*.

        Args:
            inputs: A list of input strings. Duplicates are allowed; the
                implementation may decide to de‑duplicate internally.
            batch_size: Maximum number of sentences to be sent to the underlying
                model in one request.

        Returns:
            A list of ``np.ndarray`` objects (dtype ``float32``) where each entry
                is the embedding of the corresponding sentence in *sentences*.

        Raises:
            openai.RateLimitError: Propagated if retries are exhausted.
        """
        return map(inputs, self._embed_chunk, batch_size)


@dataclass(frozen=True)
class AsyncBatchEmbeddings:
    """Thin wrapper around the OpenAI /embeddings endpoint using async operations.

    This class provides an asynchronous interface for generating embeddings using
    OpenAI models. It manages concurrency, handles rate limits automatically,
    and efficiently processes batches of inputs, including de-duplication.

    Example:
        ```python
        import asyncio
        import numpy as np
        from openai import AsyncOpenAI
        from openaivec.aio.embeddings import AsyncBatchEmbeddings

        # Assuming openai_async_client is an initialized AsyncOpenAI client
        openai_async_client = AsyncOpenAI() # Replace with your actual client initialization

        embedder = AsyncBatchEmbeddings(
            client=openai_async_client,
            model_name="text-embedding-3-small",
            max_concurrency=8  # Limit concurrent requests
        )
        texts = ["This is the first document.", "This is the second document.", "This is the first document."]

        # Asynchronous call
        async def main():
            embeddings = await embedder.create(texts, batch_size=128)
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
        client: An already‑configured ``openai.AsyncOpenAI`` client.
        model_name: The model identifier, e.g. ``"text-embedding-3-small"``.
        max_concurrency: Maximum number of concurrent requests to the OpenAI API.
    """

    client: AsyncOpenAI
    model_name: str
    max_concurrency: int = 8  # Default concurrency limit
    _semaphore: asyncio.Semaphore = field(init=False, repr=False)

    def __post_init__(self):
        # Initialize the semaphore after the object is created
        # Use object.__setattr__ because the dataclass is frozen
        object.__setattr__(self, "_semaphore", asyncio.Semaphore(self.max_concurrency))

    @observe(_LOGGER)
    @backoff_async(exception=RateLimitError, scale=15, max_retries=8)
    async def _embed_chunk(self, inputs: List[str]) -> List[NDArray[np.float32]]:
        """Embed one minibatch of sentences asynchronously, respecting concurrency limits.

        This private helper handles the actual API call for a batch of inputs.
        Exponential back-off is applied automatically when ``openai.RateLimitError``
        is raised.

        Args:
            inputs: Input strings to be embedded. Duplicates are allowed.

        Returns:
            List of embedding vectors (``np.ndarray`` with dtype ``float32``)
            in the same order as *inputs*.

        Raises:
            openai.RateLimitError: Propagated if retries are exhausted.
        """
        # Acquire semaphore before making the API call
        async with self._semaphore:
            responses = await self.client.embeddings.create(input=inputs, model=self.model_name)
            return [np.array(d.embedding, dtype=np.float32) for d in responses.data]

    @observe(_LOGGER)
    async def create(self, inputs: List[str], batch_size: int) -> List[NDArray[np.float32]]:
        """Asynchronous public API: generate embeddings for a list of inputs.

        Uses ``openaivec.util.map_async`` to efficiently handle batching and de-duplication.

        Args:
            inputs: A list of input strings. Duplicates are handled efficiently.
            batch_size: Maximum number of unique inputs per API call.

        Returns:
            A list of ``np.ndarray`` objects (dtype ``float32``) where each entry
            is the embedding of the corresponding string in *inputs*.

        Raises:
            openai.RateLimitError: Propagated if retries are exhausted during API calls.
        """

        return await map_async(inputs, self._embed_chunk, batch_size)
