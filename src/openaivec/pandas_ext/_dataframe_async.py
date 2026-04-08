"""Asynchronous pandas DataFrame accessor (``.aio``)."""

import inspect
from collections.abc import Awaitable, Callable

import pandas as pd

from openaivec._cache import AsyncBatchingMapProxy
from openaivec._cache.proxy import DEFAULT_MANAGED_CACHE_SIZE
from openaivec._model import PreparedTask, ResponseFormat
from openaivec.pandas_ext._common import T, _df_rows_to_json_series
from openaivec.task.table import FillNaResponse


@pd.api.extensions.register_dataframe_accessor("aio")
class AsyncOpenAIVecDataFrameAccessor:
    """pandas DataFrame accessor (``.aio``) that adds OpenAI helpers."""

    def __init__(self, df_obj: pd.DataFrame):
        self._obj = df_obj

    async def responses_with_cache(
        self,
        instructions: str,
        cache: AsyncBatchingMapProxy[str, ResponseFormat],
        response_format: type[ResponseFormat] = str,
        **api_kwargs,
    ) -> pd.Series:
        """Generate a response for each row after serializing it to JSON using a provided cache (asynchronously).

        This method allows external control over caching behavior by accepting
        a pre-configured AsyncBatchingMapProxy instance, enabling cache sharing
        across multiple operations or custom batch size management. The concurrency
        is controlled by the cache instance itself.

        Example:
            ```python
            from openaivec._cache import AsyncBatchingMapProxy

            # Create a shared cache with custom batch size and concurrency
            shared_cache = AsyncBatchingMapProxy(batch_size=64, max_concurrency=4)

            df = pd.DataFrame([
                {"name": "cat", "legs": 4},
                {"name": "dog", "legs": 4},
                {"name": "elephant", "legs": 4},
            ])
            # Must be awaited
            result = await df.aio.responses_with_cache(
                "what is the animal's name?",
                cache=shared_cache
            )
            ```

        Args:
            instructions (str): System prompt for the assistant.
            cache (AsyncBatchingMapProxy[str, ResponseFormat]): Pre-configured cache
                instance for managing API call batching and deduplication.
                Set cache.batch_size=None to enable automatic batch size optimization.
            response_format (type[ResponseFormat], optional): Desired Python type of the
                responses. Defaults to ``str``.
            **api_kwargs: Additional OpenAI API parameters (temperature, top_p, etc.).

        Returns:
            pandas.Series: Responses aligned with the DataFrame's original index.

        Note:
            This is an asynchronous method and must be awaited.
        """
        # Await the call to the async Series method using .aio
        return await _df_rows_to_json_series(self._obj).aio.responses_with_cache(
            instructions=instructions,
            cache=cache,
            response_format=response_format,
            **api_kwargs,
        )

    async def responses(
        self,
        instructions: str,
        response_format: type[ResponseFormat] = str,
        batch_size: int | None = None,
        max_concurrency: int = 8,
        show_progress: bool = True,
        **api_kwargs,
    ) -> pd.Series:
        """Generate a response for each row after serializing it to JSON (asynchronously).

        Example:
            ```python
            df = pd.DataFrame([
                {"name": "cat", "legs": 4},
                {"name": "dog", "legs": 4},
                {"name": "elephant", "legs": 4},
            ])
            # Must be awaited
            results = await df.aio.responses("what is the animal's name?")

            # With progress bar for large datasets
            large_df = pd.DataFrame({"id": list(range(1000))})
            results = await large_df.aio.responses(
                "generate a name for this ID",
                batch_size=20,
                max_concurrency=4,
                show_progress=True
            )
            ```

        Args:
            instructions (str): System prompt for the assistant.
            response_format (type[ResponseFormat], optional): Desired Python type of the
                responses. Defaults to ``str``.
            batch_size (int | None, optional): Number of requests sent in one batch.
                Defaults to ``None`` (automatic batch size optimization
                based on execution time). Set to a positive integer for fixed batch size.
            **api_kwargs: Additional OpenAI API parameters (temperature, top_p, etc.).
            max_concurrency (int, optional): Maximum number of concurrent
                requests. Defaults to ``8``.
            show_progress (bool, optional): Show progress bar in Jupyter notebooks. Defaults to ``True``.

        Returns:
            pandas.Series: Responses aligned with the DataFrame's original index.

        Note:
            This is an asynchronous method and must be awaited.
        """
        return await self.responses_with_cache(
            instructions=instructions,
            cache=AsyncBatchingMapProxy(
                batch_size=batch_size,
                max_concurrency=max_concurrency,
                max_cache_size=DEFAULT_MANAGED_CACHE_SIZE,
                show_progress=show_progress,
            ),
            response_format=response_format,
            **api_kwargs,
        )

    async def task_with_cache(
        self,
        task: PreparedTask[ResponseFormat],
        cache: AsyncBatchingMapProxy[str, ResponseFormat],
        **api_kwargs,
    ) -> pd.Series:
        """Execute a prepared task on each DataFrame row using a provided cache (asynchronously).

        After serializing each row to JSON, this method executes the prepared task.

        Args:
            task (PreparedTask): Prepared task (instructions + response_format).
            cache (AsyncBatchingMapProxy[str, ResponseFormat]): Pre‑configured async cache instance.
            **api_kwargs: Additional OpenAI API parameters forwarded to the Responses API.

        Note:
            Core routing keys are managed internally.

        Returns:
            pandas.Series: Task results aligned with the DataFrame's original index.

        Note:
            This is an asynchronous method and must be awaited.
        """
        return await _df_rows_to_json_series(self._obj).aio.task_with_cache(
            task=task,
            cache=cache,
            **api_kwargs,
        )

    async def task(
        self,
        task: PreparedTask,
        batch_size: int | None = None,
        max_concurrency: int = 8,
        show_progress: bool = True,
        **api_kwargs,
    ) -> pd.Series:
        """Execute a prepared task on each DataFrame row after serializing it to JSON (asynchronously).

        Example:
            ```python
            from openaivec._model import PreparedTask

            # Assume you have a prepared task for data analysis
            analysis_task = PreparedTask(...)

            df = pd.DataFrame([
                {"name": "cat", "legs": 4},
                {"name": "dog", "legs": 4},
                {"name": "elephant", "legs": 4},
            ])
            # Must be awaited
            results = await df.aio.task(analysis_task)

            # With progress bar for large datasets
            large_df = pd.DataFrame({"id": list(range(1000))})
            results = await large_df.aio.task(
                analysis_task,
                batch_size=50,
                max_concurrency=4,
                show_progress=True
            )
            ```

        Args:
            task (PreparedTask): A pre-configured task containing instructions,
                response format for processing the inputs.
            batch_size (int | None, optional): Number of requests sent in one batch
                to optimize API usage. Defaults to ``None`` (automatic batch size
                optimization based on execution time). Set to a positive integer for fixed batch size.
            max_concurrency (int, optional): Maximum number of concurrent
                requests. Defaults to 8.
            show_progress (bool, optional): Show progress bar in Jupyter notebooks. Defaults to ``True``.
            **api_kwargs: Additional OpenAI API parameters forwarded to the Responses API.

        Note:
            Core batching / routing keys (``model``, ``instructions`` / system message, user ``input``)
            are managed by the library and cannot be overridden.

        Returns:
            pandas.Series: Series whose values are instances of the task's
                response format, aligned with the DataFrame's original index.

        Note:
            This is an asynchronous method and must be awaited.
        """
        # Await the call to the async Series method using .aio
        return await _df_rows_to_json_series(self._obj).aio.task(
            task=task,
            batch_size=batch_size,
            max_concurrency=max_concurrency,
            show_progress=show_progress,
            **api_kwargs,
        )

    async def parse_with_cache(
        self,
        instructions: str,
        cache: AsyncBatchingMapProxy[str, ResponseFormat],
        response_format: type[ResponseFormat] | None = None,
        max_examples: int = 100,
        **api_kwargs,
    ) -> pd.Series:
        """Parse DataFrame rows into structured data using an LLM with cache (asynchronously).

        Async method for parsing DataFrame rows (as JSON) with external cache
        control, enabling deduplication across operations and concurrent processing.

        Args:
            instructions (str): Plain language extraction goals (e.g., "Extract
                invoice details including items, quantities, and totals").
            cache (AsyncBatchingMapProxy[str, ResponseFormat]): Pre-configured
                async cache for concurrent API call management.
            response_format (type[ResponseFormat] | None, optional): Target
                structure. None triggers automatic schema inference. Defaults to None.
            max_examples (int, optional): Maximum rows for schema inference.
                Defaults to 100.
            **api_kwargs: Additional OpenAI API parameters.

        Returns:
            pandas.Series: Parsed structured data indexed like the original DataFrame.

        Note:
            This is an asynchronous method and must be awaited.
        """
        return await _df_rows_to_json_series(self._obj).aio.parse_with_cache(
            instructions=instructions,
            cache=cache,
            response_format=response_format,
            max_examples=max_examples,
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
        **api_kwargs,
    ) -> pd.Series:
        """Parse DataFrame rows into structured data using an LLM (asynchronously).

        Async version for extracting structured information from DataFrame rows,
        with automatic schema inference when no format is specified.

        Args:
            instructions (str): Plain language extraction goals (e.g., "Extract
                customer details, order items, and payment information").
            response_format (type[ResponseFormat] | None, optional): Target
                structure. None triggers automatic inference. Defaults to None.
            max_examples (int, optional): Maximum rows for schema inference.
                Defaults to 100.
            batch_size (int | None, optional): Rows per batch. None for
                automatic optimization. Defaults to None.
            max_concurrency (int, optional): Maximum concurrent requests.
                Defaults to 8.
            show_progress (bool, optional): Show progress bar. Defaults to True.
            **api_kwargs: Additional OpenAI API parameters.

        Returns:
            pandas.Series: Parsed structured data indexed like the original DataFrame.

        Example:
            ```python
            df = pd.DataFrame({
                'raw_data': [
                    'Customer: John Doe, Order: 2 laptops @ $1200 each',
                    'Customer: Jane Smith, Order: 5 phones @ $800 each'
                ]
            })

            # Async parsing with automatic schema inference
            parsed = await df.aio.parse(
                "Extract customer name, product, quantity, and unit price"
            )
            ```

        Note:
            This is an asynchronous method and must be awaited.
        """
        return await self.parse_with_cache(
            instructions=instructions,
            cache=AsyncBatchingMapProxy(
                batch_size=batch_size,
                max_concurrency=max_concurrency,
                max_cache_size=DEFAULT_MANAGED_CACHE_SIZE,
                show_progress=show_progress,
            ),
            response_format=response_format,
            max_examples=max_examples,
            **api_kwargs,
        )

    async def pipe(self, func: Callable[[pd.DataFrame], Awaitable[T] | T]) -> T:
        """Apply a function to the DataFrame, supporting both synchronous and asynchronous functions.

        This method allows chaining operations on the DataFrame, similar to pandas' `pipe` method,
        but with support for asynchronous functions.

        Example:
            ```python
            async def process_data(df):
                # Simulate an asynchronous computation
                await asyncio.sleep(1)
                return df.dropna()

            df = pd.DataFrame({"col": [1, 2, None, 4]})
            # Must be awaited
            result = await df.aio.pipe(process_data)
            ```

        Args:
            func (Callable[[pd.DataFrame], Awaitable[T] | T]): A function that takes a DataFrame
                as input and returns either a result or an awaitable result.

        Returns:
            T: The result of applying the function, either directly or after awaiting it.

        Note:
            This is an asynchronous method and must be awaited if the function returns an awaitable.
        """
        result = func(self._obj)
        if inspect.isawaitable(result):
            return await result
        else:
            return result

    async def assign(self, **kwargs) -> pd.DataFrame:
        """Asynchronously assign new columns to the DataFrame, evaluating sequentially.

        This method extends pandas' `assign` method by supporting asynchronous
        functions as column values and evaluating assignments sequentially, allowing
        later assignments to refer to columns created earlier in the same call.

        For each key-value pair in `kwargs`:
        - If the value is a callable, it is invoked with the current state of the DataFrame
          (including columns created in previous steps of this `assign` call).
          If the result is awaitable, it is awaited; otherwise, it is used directly.
        - If the value is not callable, it is assigned directly to the new column.

        Example:
            ```python
            async def compute_column(df):
                # Simulate an asynchronous computation
                await asyncio.sleep(1)
                return df["existing_column"] * 2

            async def use_new_column(df):
                # Access the column created in the previous step
                await asyncio.sleep(1)
                return df["new_column"] + 5


            df = pd.DataFrame({"existing_column": [1, 2, 3]})
            # Must be awaited
            df = await df.aio.assign(
                new_column=compute_column,
                another_column=use_new_column
            )
            ```

        Args:
            **kwargs: Column names as keys and either static values or callables
                (synchronous or asynchronous) as values.

        Returns:
            pandas.DataFrame: A new DataFrame with the assigned columns.

        Note:
            This is an asynchronous method and must be awaited.
        """
        df_current = self._obj.copy()
        for key, value in kwargs.items():
            if callable(value):
                result = value(df_current)
                if inspect.isawaitable(result):
                    column_data = await result
                else:
                    column_data = result
            else:
                column_data = value

            df_current[key] = column_data

        return df_current

    async def fillna(
        self,
        target_column_name: str,
        max_examples: int = 500,
        batch_size: int | None = None,
        max_concurrency: int = 8,
        show_progress: bool = True,
    ) -> pd.DataFrame:
        """Fill missing values in a DataFrame column using AI-powered inference (asynchronously).

        This method uses machine learning to intelligently fill missing (NaN) values
        in a specified column by analyzing patterns from non-missing rows in the DataFrame.
        It creates a prepared task that provides examples of similar rows to help the AI
        model predict appropriate values for the missing entries.

        Args:
            target_column_name (str): The name of the column containing missing values
                that need to be filled.
            max_examples (int, optional): The maximum number of example rows to use
                for context when predicting missing values. Higher values may improve
                accuracy but increase API costs and processing time. Defaults to 500.
            batch_size (int | None, optional): Number of requests sent in one batch
                to optimize API usage. Defaults to ``None`` (automatic batch size
                optimization based on execution time). Set to a positive integer for fixed batch size.
            max_concurrency (int, optional): Maximum number of concurrent
                requests. Defaults to 8.
            show_progress (bool, optional): Show progress bar in Jupyter notebooks. Defaults to ``True``.

        Returns:
            pandas.DataFrame: A new DataFrame with missing values filled in the target
                column. The original DataFrame is not modified.

        Example:
            ```python
            df = pd.DataFrame({
                'name': ['Alice', 'Bob', None, 'David'],
                'age': [25, 30, 35, None],
                'city': ['Tokyo', 'Osaka', 'Kyoto', 'Tokyo']
            })

            # Fill missing values in the 'name' column (must be awaited)
            filled_df = await df.aio.fillna('name')

            # With progress bar for large datasets
            large_df = pd.DataFrame({'name': [None] * 1000, 'age': list(range(1000))})
            filled_df = await large_df.aio.fillna(
                'name',
                batch_size=32,
                max_concurrency=4,
                show_progress=True
            )
            ```

        Note:
            This is an asynchronous method and must be awaited.
            If the target column has no missing values, the original DataFrame
            is returned unchanged.
        """

        missing_rows = self._obj[self._obj[target_column_name].isna()]
        if missing_rows.empty:
            return self._obj
        import openaivec.pandas_ext as _pkg

        task: PreparedTask = _pkg.fillna(self._obj, target_column_name, max_examples)

        filled_values: list[FillNaResponse] = await missing_rows.aio.task(
            task=task,
            batch_size=batch_size,
            max_concurrency=max_concurrency,
            show_progress=show_progress,
        )

        # get deep copy of the DataFrame to avoid modifying the original
        df = self._obj.copy()

        # Get the actual indices of missing rows to map the results correctly
        missing_indices = missing_rows.index.tolist()

        for i, result in enumerate(filled_values):
            if result.output is not None:
                # Use the actual index from the original DataFrame, not the relative index from result
                actual_index = missing_indices[i]
                df.at[actual_index, target_column_name] = result.output

        return df
