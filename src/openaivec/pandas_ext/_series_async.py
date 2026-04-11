"""Asynchronous pandas Series accessor (``.aio``)."""

import asyncio
from typing import cast

import numpy as np
import pandas as pd
from openai import AsyncOpenAI

from openaivec._cache import AsyncBatchCache
from openaivec._cache.proxy import DEFAULT_MANAGED_CACHE_SIZE
from openaivec._embeddings import AsyncBatchEmbeddings
from openaivec._model import EmbeddingsModelName, PreparedTask, ResponseFormat, ResponsesModelName
from openaivec._provider import CONTAINER
from openaivec._responses import AsyncBatchResponses
from openaivec._schema import SchemaInferenceOutput
from openaivec.pandas_ext._common import _embeddings_to_series


@pd.api.extensions.register_series_accessor("aio")
class AsyncOpenAIVecSeriesAccessor:
    """pandas Series accessor (``.aio``) that adds OpenAI helpers."""

    def __init__(self, series_obj: pd.Series):
        self._obj = series_obj

    async def responses_with_cache(
        self,
        instructions: str,
        cache: AsyncBatchCache[str, ResponseFormat],
        response_format: type[ResponseFormat] = str,
        multimodal: bool = False,
        **api_kwargs,
    ) -> pd.Series:
        """Call an LLM once for every Series element using a provided cache (asynchronously).

        This method allows external control over caching behavior by accepting
        a pre-configured AsyncBatchCache instance, enabling cache sharing
        across multiple operations or custom batch size management. The concurrency
        is controlled by the cache instance itself.

        Example:
            ```python
            result = await series.aio.responses_with_cache(
                "classify",
                cache=shared,
                max_output_tokens=256,
                frequency_penalty=0.2,
            )
            ```

        Args:
            instructions (str): System prompt prepended to every user message.
            cache (AsyncBatchCache[str, ResponseFormat]): Pre-configured cache
                instance for managing API call batching and deduplication.
                Set cache.batch_size=None to enable automatic batch size optimization.
            response_format (type[ResponseFormat], optional): Pydantic model or built‑in
                type the assistant should return. Defaults to ``str``.
            **api_kwargs: Additional OpenAI API parameters (e.g. ``temperature``,
                ``top_p``, ``max_output_tokens``) forwarded verbatim to the
                underlying client.

        Returns:
            pandas.Series: Series whose values are instances of ``response_format``.

        Note:
            This is an asynchronous method and must be awaited.
        """
        client: AsyncBatchResponses = AsyncBatchResponses(
            client=CONTAINER.resolve(AsyncOpenAI),
            model_name=CONTAINER.resolve(ResponsesModelName).value,
            system_message=instructions,
            response_format=response_format,
            cache=cache,
            api_kwargs=api_kwargs,
            multimodal=multimodal,
        )

        results = await client.parse(self._obj.tolist())
        return pd.Series(results, index=self._obj.index, name=self._obj.name)

    async def responses(
        self,
        instructions: str,
        response_format: type[ResponseFormat] = str,
        batch_size: int | None = None,
        max_concurrency: int = 8,
        show_progress: bool = True,
        multimodal: bool = False,
        **api_kwargs,
    ) -> pd.Series:
        """Call an LLM once for every Series element (asynchronously).

        Example:
            ```python
            animals = pd.Series(["cat", "dog", "elephant"])
            # Must be awaited
            results = await animals.aio.responses("translate to French")

            # With progress bar for large datasets
            large_series = pd.Series(["data"] * 1000)
            results = await large_series.aio.responses(
                "analyze this data",
                batch_size=32,
                max_concurrency=4,
                show_progress=True
            )
            ```

        Args:
            instructions (str): System prompt prepended to every user message.
            response_format (type[ResponseFormat], optional): Pydantic model or built‑in
                type the assistant should return. Defaults to ``str``.
            batch_size (int | None, optional): Number of prompts grouped into a single
                request. Defaults to ``None`` (automatic batch size optimization
                based on execution time). Set to a positive integer for fixed batch size.
            max_concurrency (int, optional): Maximum number of concurrent
                requests. Defaults to ``8``.
            show_progress (bool, optional): Show progress bar in Jupyter notebooks. Defaults to ``True``.
            **api_kwargs: Additional OpenAI API parameters (e.g. ``temperature``,
                ``top_p``, ``max_output_tokens``) forwarded verbatim to the
                underlying client.

        Returns:
            pandas.Series: Series whose values are instances of ``response_format``.

        Note:
            This is an asynchronous method and must be awaited.
        """
        return await self.responses_with_cache(
            instructions=instructions,
            cache=AsyncBatchCache(
                batch_size=batch_size,
                max_concurrency=max_concurrency,
                max_cache_size=DEFAULT_MANAGED_CACHE_SIZE,
                show_progress=show_progress,
            ),
            response_format=response_format,
            multimodal=multimodal,
            **api_kwargs,
        )

    async def embeddings_with_cache(
        self,
        cache: AsyncBatchCache[str, np.ndarray],
        **api_kwargs,
    ) -> pd.Series:
        """Compute OpenAI embeddings for every Series element using a provided cache (asynchronously).

        This method allows external control over caching behavior by accepting
        a pre-configured AsyncBatchCache instance, enabling cache sharing
        across multiple operations or custom batch size management. The concurrency
        is controlled by the cache instance itself.

        Example:
            ```python
            from openaivec._cache import AsyncBatchCache
            import numpy as np

            # Create a shared cache with custom batch size and concurrency
            shared_cache = AsyncBatchCache[str, np.ndarray](
                batch_size=64, max_concurrency=4
            )

            animals = pd.Series(["cat", "dog", "elephant"])
            # Must be awaited
            embeddings = await animals.aio.embeddings_with_cache(cache=shared_cache)
            ```

        Args:
            cache (AsyncBatchCache[str, np.ndarray]): Pre-configured cache
                instance for managing API call batching and deduplication.
                Set cache.batch_size=None to enable automatic batch size optimization.
            **api_kwargs: Additional OpenAI API parameters (e.g. ``temperature``,
                ``top_p``, ``max_output_tokens``) forwarded verbatim to the
                underlying client.

        Returns:
            pandas.Series: Series whose values are ``np.ndarray`` objects
                (dtype ``float32``).

        Note:
            This is an asynchronous method and must be awaited.
        """
        client: AsyncBatchEmbeddings = AsyncBatchEmbeddings(
            client=CONTAINER.resolve(AsyncOpenAI),
            model_name=CONTAINER.resolve(EmbeddingsModelName).value,
            cache=cache,
            api_kwargs=api_kwargs,
        )

        # Await the async operation
        results = await client.create(self._obj.tolist())

        return _embeddings_to_series(
            results,
            index=self._obj.index,
            name=self._obj.name,
        )

    async def embeddings(
        self, batch_size: int | None = None, max_concurrency: int = 8, show_progress: bool = True, **api_kwargs
    ) -> pd.Series:
        """Compute OpenAI embeddings for every Series element (asynchronously).

        Example:
            ```python
            animals = pd.Series(["cat", "dog", "elephant"])
            # Must be awaited
            embeddings = await animals.aio.embeddings()

            # With progress bar for large datasets
            large_texts = pd.Series(["text"] * 5000)
            embeddings = await large_texts.aio.embeddings(
                batch_size=100,
                max_concurrency=4,
                show_progress=True
            )
            ```

        Args:
            batch_size (int | None, optional): Number of inputs grouped into a
                single request. Defaults to ``None`` (automatic batch size optimization
                based on execution time). Set to a positive integer for fixed batch size.
            max_concurrency (int, optional): Maximum number of concurrent
                requests. Defaults to ``8``.
            show_progress (bool, optional): Show progress bar in Jupyter notebooks. Defaults to ``True``.
            **api_kwargs: Additional OpenAI API parameters (e.g. ``temperature``,
                ``top_p``, ``max_output_tokens``) forwarded verbatim to the
                underlying client.

        Returns:
            pandas.Series: Series whose values are ``np.ndarray`` objects
                (dtype ``float32``).

        Note:
            This is an asynchronous method and must be awaited.
        """
        return await self.embeddings_with_cache(
            cache=AsyncBatchCache(
                batch_size=batch_size,
                max_concurrency=max_concurrency,
                max_cache_size=DEFAULT_MANAGED_CACHE_SIZE,
                show_progress=show_progress,
            ),
            **api_kwargs,
        )

    async def task_with_cache(
        self,
        task: PreparedTask[ResponseFormat],
        cache: AsyncBatchCache[str, ResponseFormat],
        multimodal: bool = False,
        **api_kwargs,
    ) -> pd.Series:
        """Execute a prepared task on every Series element using a provided cache (asynchronously).

        This method allows external control over caching behavior by accepting
        a pre-configured AsyncBatchCache instance, enabling cache sharing
        across multiple operations or custom batch size management. The concurrency
        is controlled by the cache instance itself.

        Example:
            ```python
            from openaivec._model import PreparedTask
            from openaivec._cache import AsyncBatchCache

            shared_cache = AsyncBatchCache(batch_size=64, max_concurrency=4)
            sentiment_task = PreparedTask(...)
            reviews = pd.Series(["Great product!", "Not satisfied", "Amazing quality"])
            results = await reviews.aio.task_with_cache(sentiment_task, cache=shared_cache)
            ```

        Args:
            task (PreparedTask): A pre-configured task containing instructions,
                response format for processing the inputs.
            cache (AsyncBatchCache[str, ResponseFormat]): Pre-configured cache
                instance for managing API call batching and deduplication.
                Set cache.batch_size=None to enable automatic batch size optimization.
            **api_kwargs: Additional OpenAI API parameters (e.g. ``temperature``,
                ``top_p``, ``max_output_tokens``) forwarded verbatim to the
                underlying client.

        Returns:
            pandas.Series: Series whose values are instances of the task's
                response format, aligned with the original Series index.

        Note:
            This is an asynchronous method and must be awaited.
        """
        client = AsyncBatchResponses(
            client=CONTAINER.resolve(AsyncOpenAI),
            model_name=CONTAINER.resolve(ResponsesModelName).value,
            system_message=task.instructions,
            response_format=task.response_format,
            cache=cache,
            api_kwargs=api_kwargs,
            multimodal=multimodal,
        )
        results = await client.parse(self._obj.tolist())

        return pd.Series(results, index=self._obj.index, name=self._obj.name)

    async def task(
        self,
        task: PreparedTask,
        batch_size: int | None = None,
        max_concurrency: int = 8,
        show_progress: bool = True,
        multimodal: bool = False,
        **api_kwargs,
    ) -> pd.Series:
        """Execute a prepared task on every Series element (asynchronously).

        Example:
            ```python
            from openaivec._model import PreparedTask

            # Assume you have a prepared task for sentiment analysis
            sentiment_task = PreparedTask(...)

            reviews = pd.Series(["Great product!", "Not satisfied", "Amazing quality"])
            # Must be awaited
            results = await reviews.aio.task(sentiment_task)

            # With progress bar for large datasets
            large_reviews = pd.Series(["review text"] * 2000)
            results = await large_reviews.aio.task(
                sentiment_task,
                batch_size=50,
                max_concurrency=4,
                show_progress=True
            )
            ```

        Args:
            task (PreparedTask): A pre-configured task containing instructions,
                response format for processing the inputs.
            batch_size (int | None, optional): Number of prompts grouped into a single
                request to optimize API usage. Defaults to ``None`` (automatic batch size
                optimization based on execution time). Set to a positive integer for fixed batch size.
            max_concurrency (int, optional): Maximum number of concurrent
                requests. Defaults to 8.
            show_progress (bool, optional): Show progress bar in Jupyter notebooks. Defaults to ``True``.
            **api_kwargs: Additional OpenAI API parameters (e.g. ``temperature``,
                ``top_p``, ``max_output_tokens``) forwarded verbatim to the
                underlying client.

        Returns:
            pandas.Series: Series whose values are instances of the task's
                response format, aligned with the original Series index.

        Note:
            This is an asynchronous method and must be awaited.
        """
        return await self.task_with_cache(
            task=task,
            cache=AsyncBatchCache(
                batch_size=batch_size,
                max_concurrency=max_concurrency,
                max_cache_size=DEFAULT_MANAGED_CACHE_SIZE,
                show_progress=show_progress,
            ),
            multimodal=multimodal,
            **api_kwargs,
        )

    async def parse_with_cache(
        self,
        instructions: str,
        cache: AsyncBatchCache[str, ResponseFormat],
        response_format: type[ResponseFormat] | None = None,
        max_examples: int = 100,
        multimodal: bool = False,
        **api_kwargs,
    ) -> pd.Series:
        """Parse Series values into structured data using an LLM with a provided cache (asynchronously).

        This async method provides external cache control while parsing Series
        content into structured data. Automatic schema inference is performed
        when no response format is specified.

        Example:
            ```python
            from openaivec._cache import AsyncBatchCache

            shared = AsyncBatchCache(batch_size=64, max_concurrency=4)
            result = await series.aio.parse_with_cache(
                "Extract dates and amounts",
                cache=shared,
                response_format=None,
            )
            ```

        Args:
            instructions (str): Plain language description of what to extract
                (e.g., "Extract dates, amounts, and descriptions from receipts").
                Guides both extraction and schema inference.
            cache (AsyncBatchCache[str, ResponseFormat]): Pre-configured
                async cache for managing concurrent API calls and deduplication.
                Set cache.batch_size=None for automatic optimization.
            response_format (type[ResponseFormat] | None, optional): Target
                structure for parsed data. Can be a Pydantic model, built-in
                type, or None for automatic inference. Defaults to None.
            max_examples (int, optional): Maximum values to analyze for schema
                inference (when response_format is None). Defaults to 100.
            **api_kwargs: Additional OpenAI API parameters (e.g. ``temperature``,
                ``top_p``, ``max_output_tokens``) forwarded verbatim to the
                underlying client.

        Returns:
            pandas.Series: Series containing parsed structured data aligned
                with the original index.

        Note:
            This is an asynchronous method and must be awaited.
        """
        schema: SchemaInferenceOutput | None = None
        if response_format is None:
            inferred_schema = await asyncio.to_thread(
                self._obj.ai.infer_schema,
                instructions=instructions,
                max_examples=max_examples,
                **api_kwargs,
            )
            schema = inferred_schema
            resolved_response_format = cast(type[ResponseFormat], inferred_schema.model)
        else:
            resolved_response_format = response_format

        return await self.responses_with_cache(
            instructions=schema.inference_prompt if schema else instructions,
            cache=cache,
            response_format=resolved_response_format,
            multimodal=multimodal,
            **api_kwargs,
        )

    async def parse(
        self,
        instructions: str,
        response_format: type[ResponseFormat] | None = None,
        max_examples: int = 100,
        batch_size: int | None = None,
        max_concurrency: int = 8,
        show_progress: bool = True,
        multimodal: bool = False,
        **api_kwargs,
    ) -> pd.Series:
        """Parse Series values into structured data using an LLM (asynchronously).

        Async version of the parse method, extracting structured information
        from unstructured text with automatic schema inference when needed.

        Args:
            instructions (str): Plain language extraction goals (e.g., "Extract
                product names, prices, and categories from descriptions").
            response_format (type[ResponseFormat] | None, optional): Target
                structure. None triggers automatic schema inference. Defaults to None.
            max_examples (int, optional): Maximum values for schema inference.
                Defaults to 100.
            batch_size (int | None, optional): Requests per batch. None for
                automatic optimization. Defaults to None.
            max_concurrency (int, optional): Maximum concurrent API requests.
                Defaults to 8.
            show_progress (bool, optional): Show progress bar. Defaults to True.
            **api_kwargs: Additional OpenAI API parameters (e.g. ``temperature``,
                ``top_p``, ``max_output_tokens``) forwarded verbatim to the
                underlying client.

        Returns:
            pandas.Series: Parsed structured data indexed like the original Series.

        Example:
            ```python
            emails = pd.Series([
                "Meeting tomorrow at 3pm with John about Q4 planning",
                "Lunch with Sarah on Friday to discuss new project"
            ])

            # Async extraction with schema inference
            parsed = await emails.aio.parse(
                "Extract meeting details including time, person, and topic"
            )
            ```

        Note:
            This is an asynchronous method and must be awaited.
        """
        return await self.parse_with_cache(
            instructions=instructions,
            cache=AsyncBatchCache(
                batch_size=batch_size,
                max_concurrency=max_concurrency,
                max_cache_size=DEFAULT_MANAGED_CACHE_SIZE,
                show_progress=show_progress,
            ),
            response_format=response_format,
            max_examples=max_examples,
            multimodal=multimodal,
            **api_kwargs,
        )
