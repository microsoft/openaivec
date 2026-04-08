"""Synchronous pandas Series accessor (``.ai``)."""

from typing import cast

import numpy as np
import pandas as pd
import tiktoken
from openai import OpenAI

from openaivec._cache import BatchingMapProxy
from openaivec._cache.proxy import DEFAULT_MANAGED_CACHE_SIZE
from openaivec._embeddings import BatchEmbeddings
from openaivec._model import EmbeddingsModelName, PreparedTask, ResponseFormat, ResponsesModelName
from openaivec._provider import CONTAINER
from openaivec._responses import BatchResponses
from openaivec._schema import SchemaInferenceInput, SchemaInferenceOutput, SchemaInferer
from openaivec.pandas_ext._common import _extract_value


@pd.api.extensions.register_series_accessor("ai")
class OpenAIVecSeriesAccessor:
    """pandas Series accessor (``.ai``) that adds OpenAI helpers."""

    def __init__(self, series_obj: pd.Series):
        self._obj = series_obj

    def responses_with_cache(
        self,
        instructions: str,
        cache: BatchingMapProxy[str, ResponseFormat],
        response_format: type[ResponseFormat] = str,
        **api_kwargs,
    ) -> pd.Series:
        """Call an LLM once for every Series element using a provided cache.

        This is a lower-level method that allows explicit cache management for advanced
        use cases. Most users should use the standard ``responses`` method instead.

        Args:
            instructions (str): System prompt prepended to every user message.
            cache (BatchingMapProxy[str, ResponseFormat]): Explicit cache instance for
                batching and deduplication control.
            response_format (type[ResponseFormat], optional): Pydantic model or built-in
                type the assistant should return. Defaults to ``str``.
            **api_kwargs: Arbitrary OpenAI Responses API parameters (e.g. ``temperature``,
                ``top_p``, ``frequency_penalty``, ``presence_penalty``, ``seed``, etc.) are
                forwarded verbatim to the underlying client.

        Returns:
            pandas.Series: Series whose values are instances of ``response_format``.
        """

        client: BatchResponses = BatchResponses(
            client=CONTAINER.resolve(OpenAI),
            model_name=CONTAINER.resolve(ResponsesModelName).value,
            system_message=instructions,
            response_format=response_format,
            cache=cache,
            api_kwargs=api_kwargs,
        )

        return pd.Series(client.parse(self._obj.tolist()), index=self._obj.index, name=self._obj.name)

    def responses(
        self,
        instructions: str,
        response_format: type[ResponseFormat] = str,
        batch_size: int | None = None,
        show_progress: bool = True,
        **api_kwargs,
    ) -> pd.Series:
        """Call an LLM once for every Series element.

        Example:
            ```python
            animals = pd.Series(["cat", "dog", "elephant"])
            # Basic usage
            animals.ai.responses("translate to French")

            # With progress bar in Jupyter notebooks
            large_series = pd.Series(["data"] * 1000)
            large_series.ai.responses(
                "analyze this data",
                batch_size=32,
                show_progress=True
            )

            # With custom temperature
            animals.ai.responses(
                "translate creatively",
                temperature=0.8
            )
            ```

        Args:
            instructions (str): System prompt prepended to every user message.
            response_format (type[ResponseFormat], optional): Pydantic model or built‑in
                type the assistant should return. Defaults to ``str``.
            batch_size (int | None, optional): Number of prompts grouped into a single
                request. Defaults to ``None`` (automatic batch size optimization
                based on execution time). Set to a positive integer for fixed batch size.
            show_progress (bool, optional): Show progress bar in Jupyter notebooks. Defaults to ``True``.
            **api_kwargs: Additional OpenAI API parameters (temperature, top_p, etc.).

        Returns:
            pandas.Series: Series whose values are instances of ``response_format``.
        """
        return self.responses_with_cache(
            instructions=instructions,
            cache=BatchingMapProxy(
                batch_size=batch_size,
                max_cache_size=DEFAULT_MANAGED_CACHE_SIZE,
                show_progress=show_progress,
            ),
            response_format=response_format,
            **api_kwargs,
        )

    def embeddings_with_cache(
        self,
        cache: BatchingMapProxy[str, np.ndarray],
        **api_kwargs,
    ) -> pd.Series:
        """Compute OpenAI embeddings for every Series element using a provided cache.

        This method allows external control over caching behavior by accepting
        a pre-configured BatchingMapProxy instance, enabling cache sharing
        across multiple operations or custom batch size management.

        Example:
            ```python
            from openaivec._cache import BatchingMapProxy
            import numpy as np

            # Create a shared cache with custom batch size
            shared_cache = BatchingMapProxy[str, np.ndarray](batch_size=64)

            animals = pd.Series(["cat", "dog", "elephant"])
            embeddings = animals.ai.embeddings_with_cache(cache=shared_cache)
            ```

        Args:
            cache (BatchingMapProxy[str, np.ndarray]): Pre-configured cache
                instance for managing API call batching and deduplication.
                Set cache.batch_size=None to enable automatic batch size optimization.
            **api_kwargs: Additional keyword arguments to pass to the OpenAI API.

        Returns:
            pandas.Series: Series whose values are ``np.ndarray`` objects
                (dtype ``float32``).
        """
        client: BatchEmbeddings = BatchEmbeddings(
            client=CONTAINER.resolve(OpenAI),
            model_name=CONTAINER.resolve(EmbeddingsModelName).value,
            cache=cache,
            api_kwargs=api_kwargs,
        )

        return pd.Series(
            client.create(self._obj.tolist()),
            index=self._obj.index,
            name=self._obj.name,
        )

    def embeddings(self, batch_size: int | None = None, show_progress: bool = True, **api_kwargs) -> pd.Series:
        """Compute OpenAI embeddings for every Series element.

        Example:
            ```python
            animals = pd.Series(["cat", "dog", "elephant"])
            # Basic usage
            animals.ai.embeddings()

            # With progress bar for large datasets
            large_texts = pd.Series(["text"] * 5000)
            embeddings = large_texts.ai.embeddings(
                batch_size=100,
                show_progress=True
            )
            ```

        Args:
            batch_size (int | None, optional): Number of inputs grouped into a
                single request. Defaults to ``None`` (automatic batch size optimization
                based on execution time). Set to a positive integer for fixed batch size.
            show_progress (bool, optional): Show progress bar in Jupyter notebooks. Defaults to ``True``.
            **api_kwargs: Additional OpenAI API parameters (e.g., dimensions for text-embedding-3 models).

        Returns:
            pandas.Series: Series whose values are ``np.ndarray`` objects
                (dtype ``float32``).
        """
        return self.embeddings_with_cache(
            cache=BatchingMapProxy(
                batch_size=batch_size,
                max_cache_size=DEFAULT_MANAGED_CACHE_SIZE,
                show_progress=show_progress,
            ),
            **api_kwargs,
        )

    def task_with_cache(
        self,
        task: PreparedTask[ResponseFormat],
        cache: BatchingMapProxy[str, ResponseFormat],
        **api_kwargs,
    ) -> pd.Series:
        """Execute a prepared task on every Series element using a provided cache.

        This mirrors ``responses_with_cache`` but uses the task's stored instructions
        and response format. A supplied ``BatchingMapProxy`` enables cross‑operation
        deduplicated reuse and external batch size / progress control.

        Example:
            ```python
            from openaivec._cache import BatchingMapProxy
            shared_cache = BatchingMapProxy(batch_size=64)
            reviews.ai.task_with_cache(sentiment_task, cache=shared_cache)
            ```

        Args:
            task (PreparedTask): Prepared task (instructions + response_format).
            cache (BatchingMapProxy[str, ResponseFormat]): Pre‑configured cache instance.
            **api_kwargs: Additional OpenAI API parameters forwarded to the Responses API.

        Note:
            Core routing keys (``model``, system instructions, user input) are managed
            internally and cannot be overridden.

        Returns:
            pandas.Series: Task results aligned with the original Series index.
        """
        client: BatchResponses = BatchResponses(
            client=CONTAINER.resolve(OpenAI),
            model_name=CONTAINER.resolve(ResponsesModelName).value,
            system_message=task.instructions,
            response_format=task.response_format,
            cache=cache,
            api_kwargs=api_kwargs,
        )
        return pd.Series(client.parse(self._obj.tolist()), index=self._obj.index, name=self._obj.name)

    def task(
        self,
        task: PreparedTask,
        batch_size: int | None = None,
        show_progress: bool = True,
        **api_kwargs,
    ) -> pd.Series:
        """Execute a prepared task on every Series element.

        Example:
            ```python
            from openaivec._model import PreparedTask

            # Assume you have a prepared task for sentiment analysis
            sentiment_task = PreparedTask(...)

            reviews = pd.Series(["Great product!", "Not satisfied", "Amazing quality"])
            # Basic usage
            results = reviews.ai.task(sentiment_task)

            # With progress bar for large datasets
            large_reviews = pd.Series(["review text"] * 2000)
            results = large_reviews.ai.task(
                sentiment_task,
                batch_size=50,
                show_progress=True
            )
            ```

        Args:
            task (PreparedTask): A pre-configured task containing instructions,
                response format for processing the inputs.
            batch_size (int | None, optional): Number of prompts grouped into a single
                request to optimize API usage. Defaults to ``None`` (automatic batch size
                optimization based on execution time). Set to a positive integer for fixed batch size.
            show_progress (bool, optional): Show progress bar in Jupyter notebooks. Defaults to ``True``.
            **api_kwargs: Additional OpenAI API parameters forwarded to the Responses API.

        Note:
            Core batching / routing keys (``model``, ``instructions`` / system message,
            user ``input``) are managed by the library and cannot be overridden.

        Returns:
            pandas.Series: Series whose values are instances of the task's response format.
        """
        return self.task_with_cache(
            task=task,
            cache=BatchingMapProxy(
                batch_size=batch_size,
                max_cache_size=DEFAULT_MANAGED_CACHE_SIZE,
                show_progress=show_progress,
            ),
            **api_kwargs,
        )

    def parse_with_cache(
        self,
        instructions: str,
        cache: BatchingMapProxy[str, ResponseFormat],
        response_format: type[ResponseFormat] | None = None,
        max_examples: int = 100,
        **api_kwargs,
    ) -> pd.Series:
        """Parse Series values using an LLM with a provided cache.

        This method allows external control over caching behavior while parsing
        Series content into structured data. If no response format is provided,
        the method automatically infers an appropriate schema by analyzing the
        data patterns.

        Args:
            instructions (str): Plain language description of what information
                to extract (e.g., "Extract customer information including name
                and contact details"). This guides both the extraction process
                and schema inference.
            cache (BatchingMapProxy[str, ResponseFormat]): Pre-configured cache
                instance for managing API call batching and deduplication.
                Set cache.batch_size=None to enable automatic batch size optimization.
            response_format (type[ResponseFormat] | None, optional): Target structure
                for the parsed data. Can be a Pydantic model class, built-in type
                (str, int, float, bool, list, dict), or None. If None, the method
                infers an appropriate schema based on the instructions and data.
                Defaults to None.
            max_examples (int, optional): Maximum number of Series values to
                analyze when inferring the schema. Only used when response_format
                is None. Defaults to 100.
            **api_kwargs: Additional OpenAI API parameters (temperature, top_p,
                frequency_penalty, presence_penalty, seed, etc.) forwarded to
                the underlying API calls.

        Returns:
            pandas.Series: Series containing parsed structured data. Each value
                is an instance of the specified response_format or the inferred
                schema model, aligned with the original Series index.
        """

        schema: SchemaInferenceOutput | None = None
        if response_format is None:
            schema = self.infer_schema(instructions=instructions, max_examples=max_examples, **api_kwargs)
            resolved_response_format = cast(type[ResponseFormat], schema.model)
        else:
            resolved_response_format = response_format

        return self.responses_with_cache(
            instructions=schema.inference_prompt if schema else instructions,
            cache=cache,
            response_format=resolved_response_format,
            **api_kwargs,
        )

    def parse(
        self,
        instructions: str,
        response_format: type[ResponseFormat] | None = None,
        max_examples: int = 100,
        batch_size: int | None = None,
        show_progress: bool = True,
        **api_kwargs,
    ) -> pd.Series:
        """Parse Series values into structured data using an LLM.

        This method extracts structured information from unstructured text in
        the Series. When no response format is provided, it automatically
        infers an appropriate schema by analyzing patterns in the data.

        Args:
            instructions (str): Plain language description of what information
                to extract (e.g., "Extract product details including price,
                category, and availability"). This guides both the extraction
                process and schema inference.
            response_format (type[ResponseFormat] | None, optional): Target
                structure for the parsed data. Can be a Pydantic model class,
                built-in type (str, int, float, bool, list, dict), or None.
                If None, automatically infers a schema. Defaults to None.
            max_examples (int, optional): Maximum number of Series values to
                analyze when inferring schema. Only used when response_format
                is None. Defaults to 100.
            batch_size (int | None, optional): Number of requests to process
                per batch. None enables automatic optimization. Defaults to None.
            show_progress (bool, optional): Display progress bar in Jupyter
                notebooks. Defaults to True.
            **api_kwargs: Additional OpenAI API parameters (temperature, top_p,
                frequency_penalty, presence_penalty, seed, etc.).

        Returns:
            pandas.Series: Series containing parsed structured data as instances
                of response_format or the inferred schema model.

        Example:
            ```python
            # With explicit schema
            from pydantic import BaseModel
            class Product(BaseModel):
                name: str
                price: float
                in_stock: bool

            descriptions = pd.Series([
                "iPhone 15 Pro - $999, available now",
                "Samsung Galaxy S24 - $899, out of stock"
            ])
            products = descriptions.ai.parse(
                "Extract product information",
                response_format=Product
            )

            # With automatic schema inference
            reviews = pd.Series([
                "Great product! 5 stars. Fast shipping.",
                "Poor quality. 2 stars. Slow delivery."
            ])
            parsed = reviews.ai.parse(
                "Extract review rating and shipping feedback"
            )
            ```
        """
        return self.parse_with_cache(
            instructions=instructions,
            cache=BatchingMapProxy(
                batch_size=batch_size,
                max_cache_size=DEFAULT_MANAGED_CACHE_SIZE,
                show_progress=show_progress,
            ),
            response_format=response_format,
            max_examples=max_examples,
            **api_kwargs,
        )

    def infer_schema(self, instructions: str, max_examples: int = 100, **api_kwargs) -> SchemaInferenceOutput:
        """Infer a structured data schema from Series content using AI.

        This method analyzes a sample of Series values to automatically generate
        a Pydantic model that captures the relevant information structure. The
        inferred schema supports both flat and hierarchical (nested) structures,
        making it suitable for complex data extraction tasks.

        Args:
            instructions (str): Plain language description of the extraction goal
                (e.g., "Extract customer information for CRM system", "Parse
                event details for calendar integration"). This guides which
                fields to include and their purpose.
            max_examples (int, optional): Maximum number of Series values to
                analyze for pattern detection. The method samples randomly up
                to this limit. Higher values may improve schema quality but
                increase inference time. Defaults to 100.
            **api_kwargs: Additional OpenAI API parameters for fine-tuning
                the inference process.

        Returns:
            InferredSchema: A comprehensive schema object containing:
                - instructions: Refined extraction objective statement
                - fields: Hierarchical field specifications with names, types,
                  descriptions, and nested structures where applicable
                - inference_prompt: Optimized prompt for consistent extraction
                - model: Dynamically generated Pydantic model class supporting
                  both flat and nested structures
                - task: PreparedTask configured for batch extraction using
                  the inferred schema

        Example:
            ```python
            # Simple flat structure
            reviews = pd.Series([
                "5 stars! Great product, fast shipping to NYC.",
                "2 stars. Product broke, slow delivery to LA."
            ])
            schema = reviews.ai.infer_schema(
                "Extract review ratings and shipping information"
            )

            # Hierarchical structure
            orders = pd.Series([
                "Order #123: John Doe, 123 Main St, NYC. Items: iPhone ($999), Case ($29)",
                "Order #456: Jane Smith, 456 Oak Ave, LA. Items: iPad ($799)"
            ])
            schema = orders.ai.infer_schema(
                "Extract order details including customer and items"
            )
            # Inferred schema may include nested structures like:
            # - customer: {name: str, address: str, city: str}
            # - items: [{product: str, price: float}]

            # Apply the schema for extraction
            extracted = orders.ai.task(schema.task)
            ```

        Note:
            The inference process uses multiple AI iterations to ensure schema
            validity. Nested structures are automatically detected when the
            data contains hierarchical relationships. The generated Pydantic
            model ensures type safety and validation for all extracted data.
        """
        inferer = CONTAINER.resolve(SchemaInferer)

        input: SchemaInferenceInput = SchemaInferenceInput(
            examples=self._obj.sample(n=min(max_examples, len(self._obj))).tolist(),
            instructions=instructions,
            **api_kwargs,
        )
        return inferer.infer_schema(input)

    def count_tokens(self) -> pd.Series:
        """Count `tiktoken` tokens per row.

        Example:
            ```python
            animals = pd.Series(["cat", "dog", "elephant"])
            animals.ai.count_tokens()
            ```
            This method uses the `tiktoken` library to count tokens based on the
            model name configured via `set_responses_model`.

        Returns:
            pandas.Series: Token counts for each element.
        """
        encoding: tiktoken.Encoding = CONTAINER.resolve(tiktoken.Encoding)
        return self._obj.map(encoding.encode).map(len).rename("num_tokens")

    def extract(self) -> pd.DataFrame:
        """Expand a Series of Pydantic models/dicts into columns.

        Example:
            ```python
            animals = pd.Series([
                {"name": "cat", "legs": 4},
                {"name": "dog", "legs": 4},
                {"name": "elephant", "legs": 4},
            ])
            animals.ai.extract()
            ```
            This method returns a DataFrame with the same index as the Series,
            where each column corresponds to a key in the dictionaries.
            If the Series has a name, extracted columns are prefixed with it.

        Returns:
            pandas.DataFrame: Expanded representation.
        """
        extracted = pd.DataFrame(
            self._obj.map(lambda x: _extract_value(x, self._obj.name)).tolist(),
            index=self._obj.index,
        )

        if self._obj.name:
            # If the Series has a name and all elements are dict or BaseModel, use it as the prefix for the columns
            extracted.columns = [f"{self._obj.name}_{col}" for col in extracted.columns]
        return extracted
