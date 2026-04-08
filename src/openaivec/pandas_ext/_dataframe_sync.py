"""Synchronous pandas DataFrame accessor (``.ai``)."""

import numpy as np
import pandas as pd

from openaivec._cache import BatchCache
from openaivec._cache.proxy import DEFAULT_MANAGED_CACHE_SIZE
from openaivec._model import PreparedTask, ResponseFormat
from openaivec._schema import SchemaInferenceOutput
from openaivec.pandas_ext._common import _df_rows_to_json_series, _embedding_series_to_matrix
from openaivec.task.table import FillNaResponse


@pd.api.extensions.register_dataframe_accessor("ai")
class OpenAIVecDataFrameAccessor:
    """pandas DataFrame accessor (``.ai``) that adds OpenAI helpers."""

    def __init__(self, df_obj: pd.DataFrame):
        self._obj = df_obj

    def responses_with_cache(
        self,
        instructions: str,
        cache: BatchCache[str, ResponseFormat],
        response_format: type[ResponseFormat] = str,
        **api_kwargs,
    ) -> pd.Series:
        """Call an LLM once for every DataFrame row using a provided cache.

        This method allows external control over caching behavior by accepting
        a pre-configured BatchCache instance, enabling cache sharing
        across multiple operations or custom batch size management.

        Example:
            ```python
            from openaivec._cache import BatchCache

            # Create a shared cache with custom batch size
            shared_cache = BatchCache(batch_size=64)

            df = pd.DataFrame([
                {"name": "cat", "legs": 4},
                {"name": "dog", "legs": 4},
                {"name": "elephant", "legs": 4},
            ])
            result = df.ai.responses_with_cache(
                "what is the animal's name?",
                cache=shared_cache
            )
            ```

        Args:
            instructions (str): System prompt prepended to every user message.
            cache (BatchCache[str, ResponseFormat]): Pre-configured cache
                instance for managing API call batching and deduplication.
                Set cache.batch_size=None to enable automatic batch size optimization.
            response_format (type[ResponseFormat], optional): Pydantic model or built‑in
                type the assistant should return. Defaults to ``str``.
            **api_kwargs: Additional OpenAI API parameters (e.g. ``temperature``,
                ``top_p``, ``max_output_tokens``) forwarded verbatim to the
                underlying client.

        Returns:
            pandas.Series: Responses aligned with the DataFrame's original index.
        """
        return _df_rows_to_json_series(self._obj).ai.responses_with_cache(
            instructions=instructions,
            cache=cache,
            response_format=response_format,
            **api_kwargs,
        )

    def responses(
        self,
        instructions: str,
        response_format: type[ResponseFormat] = str,
        batch_size: int | None = None,
        show_progress: bool = True,
        **api_kwargs,
    ) -> pd.Series:
        """Call an LLM once for every DataFrame row.

        Example:
            ```python
            df = pd.DataFrame([
                {"name": "cat", "legs": 4},
                {"name": "dog", "legs": 4},
                {"name": "elephant", "legs": 4},
            ])
            # Basic usage
            df.ai.responses("what is the animal's name?")

            # With progress bar for large datasets
            large_df = pd.DataFrame({"id": list(range(1000))})
            large_df.ai.responses(
                "generate a name for this ID",
                batch_size=20,
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
            show_progress (bool, optional): Show progress bar in Jupyter notebooks. Defaults to ``True``.
            **api_kwargs: Additional OpenAI API parameters (e.g. ``temperature``,
                ``top_p``, ``max_output_tokens``) forwarded verbatim to the
                underlying client.

        Returns:
            pandas.Series: Responses aligned with the DataFrame's original index.
        """
        return self.responses_with_cache(
            instructions=instructions,
            cache=BatchCache(
                batch_size=batch_size,
                max_cache_size=DEFAULT_MANAGED_CACHE_SIZE,
                show_progress=show_progress,
            ),
            response_format=response_format,
            **api_kwargs,
        )

    def task_with_cache(
        self,
        task: PreparedTask[ResponseFormat],
        cache: BatchCache[str, ResponseFormat],
        **api_kwargs,
    ) -> pd.Series:
        """Execute a prepared task on every DataFrame row using a provided cache.

        Example:
            ```python
            from openaivec._model import PreparedTask
            from openaivec._cache import BatchCache

            shared_cache = BatchCache(batch_size=64)
            analysis_task = PreparedTask(...)
            df = pd.DataFrame([
                {"name": "cat", "legs": 4},
                {"name": "dog", "legs": 4},
                {"name": "elephant", "legs": 4},
            ])
            results = df.ai.task_with_cache(analysis_task, cache=shared_cache)
            ```

        Args:
            task (PreparedTask): A pre-configured task containing instructions,
                response format for processing the inputs.
            cache (BatchCache[str, ResponseFormat]): Pre-configured cache
                instance for managing API call batching and deduplication.
                Set cache.batch_size=None to enable automatic batch size optimization.
            **api_kwargs: Additional OpenAI API parameters (e.g. ``temperature``,
                ``top_p``, ``max_output_tokens``) forwarded verbatim to the
                underlying client.

        Returns:
            pandas.Series: Series whose values are instances of the task's
                response format, aligned with the DataFrame's original index.

        Note:
            Core routing keys are managed internally.
        """
        return _df_rows_to_json_series(self._obj).ai.task_with_cache(
            task=task,
            cache=cache,
            **api_kwargs,
        )

    def task(
        self,
        task: PreparedTask,
        batch_size: int | None = None,
        show_progress: bool = True,
        **api_kwargs,
    ) -> pd.Series:
        """Execute a prepared task on every DataFrame row.

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
            # Basic usage
            results = df.ai.task(analysis_task)

            # With progress bar for large datasets
            large_df = pd.DataFrame({"id": list(range(1000))})
            results = large_df.ai.task(
                analysis_task,
                batch_size=50,
                show_progress=True
            )
            ```

        Args:
            task (PreparedTask): A pre-configured task containing instructions,
                response format for processing the inputs.
            batch_size (int | None, optional): Number of requests sent in one batch
                to optimize API usage. Defaults to ``None`` (automatic batch size
                optimization based on execution time). Set to a positive integer for fixed batch size.
            show_progress (bool, optional): Show progress bar in Jupyter notebooks. Defaults to ``True``.
            **api_kwargs: Additional OpenAI API parameters (e.g. ``temperature``,
                ``top_p``, ``max_output_tokens``) forwarded verbatim to the
                underlying client.

        Returns:
            pandas.Series: Series whose values are instances of the task's
                response format, aligned with the DataFrame's original index.

        Note:
            Core batching / routing keys (``model``, ``instructions`` / system message, user ``input``)
            are managed by the library and cannot be overridden.
        """
        return _df_rows_to_json_series(self._obj).ai.task(
            task=task,
            batch_size=batch_size,
            show_progress=show_progress,
            **api_kwargs,
        )

    def parse_with_cache(
        self,
        instructions: str,
        cache: BatchCache[str, ResponseFormat],
        response_format: type[ResponseFormat] | None = None,
        max_examples: int = 100,
        **api_kwargs,
    ) -> pd.Series:
        """Parse DataFrame rows into structured data using an LLM with a provided cache.

        This method processes each DataFrame row (converted to JSON) and extracts
        structured information using an LLM. External cache control enables
        deduplication across operations and custom batch management.

        Example:
            ```python
            from openaivec._cache import BatchCache

            shared = BatchCache(batch_size=64)
            result = df.ai.parse_with_cache(
                "Extract shipping details",
                cache=shared,
                response_format=None,
            )
            ```

        Args:
            instructions (str): Plain language description of what information
                to extract from each row (e.g., "Extract shipping details and
                order status"). Guides both extraction and schema inference.
            cache (BatchCache[str, ResponseFormat]): Pre-configured cache
                instance for managing API call batching and deduplication.
                Set cache.batch_size=None to enable automatic batch size optimization.
            response_format (type[ResponseFormat] | None, optional): Target
                structure for parsed data. Can be a Pydantic model, built-in
                type, or None for automatic schema inference. Defaults to None.
            max_examples (int, optional): Maximum rows to analyze when inferring
                schema (only used when response_format is None). Defaults to 100.
            **api_kwargs: Additional OpenAI API parameters (e.g. ``temperature``,
                ``top_p``, ``max_output_tokens``) forwarded verbatim to the
                underlying client.

        Returns:
            pandas.Series: Series containing parsed structured data as instances
                of response_format or the inferred schema model, indexed like
                the original DataFrame.
        """
        return _df_rows_to_json_series(self._obj).ai.parse_with_cache(
            instructions=instructions,
            cache=cache,
            response_format=response_format,
            max_examples=max_examples,
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
        """Parse DataFrame rows into structured data using an LLM.

        Each row is converted to JSON and processed to extract structured
        information. When no response format is provided, the method
        automatically infers an appropriate schema from the data.

        Args:
            instructions (str): Plain language description of extraction goals
                (e.g., "Extract transaction details including amount, date,
                and merchant"). Guides extraction and schema inference.
            response_format (type[ResponseFormat] | None, optional): Target
                structure for parsed data. Can be a Pydantic model, built-in
                type, or None for automatic inference. Defaults to None.
            max_examples (int, optional): Maximum rows to analyze for schema
                inference (when response_format is None). Defaults to 100.
            batch_size (int | None, optional): Rows per API batch. None
                enables automatic optimization. Defaults to None.
            show_progress (bool, optional): Show progress bar in Jupyter
                notebooks. Defaults to True.
            **api_kwargs: Additional OpenAI API parameters (e.g. ``temperature``,
                ``top_p``, ``max_output_tokens``) forwarded verbatim to the
                underlying client.

        Returns:
            pandas.Series: Parsed structured data indexed like the original
                DataFrame.

        Example:
            ```python
            df = pd.DataFrame({
                'log': [
                    '2024-01-01 10:00 ERROR Database connection failed',
                    '2024-01-01 10:05 INFO Service started successfully'
                ]
            })

            # With automatic schema inference
            parsed = df.ai.parse("Extract timestamp, level, and message")
            # Returns Series with inferred structure like:
            # {timestamp: str, level: str, message: str}
            ```
        """
        return self.parse_with_cache(
            instructions=instructions,
            cache=BatchCache(
                batch_size=batch_size,
                max_cache_size=DEFAULT_MANAGED_CACHE_SIZE,
                show_progress=show_progress,
            ),
            response_format=response_format,
            max_examples=max_examples,
            **api_kwargs,
        )

    def infer_schema(self, instructions: str, max_examples: int = 100, **api_kwargs) -> SchemaInferenceOutput:
        """Infer a structured data schema from DataFrame rows using AI.

        This method analyzes a sample of DataFrame rows to automatically infer
        a structured schema that can be used for consistent data extraction.
        Each row is converted to JSON format and analyzed to identify patterns,
        field types, and potential categorical values.

        Args:
            instructions (str): Plain language description of how the extracted
                structured data will be used (e.g., "Extract operational metrics
                for dashboard", "Parse customer attributes for segmentation").
                This guides field relevance and helps exclude irrelevant information.
            max_examples (int): Maximum number of rows to analyze from the
                DataFrame. The method will sample randomly up to this limit.
                Defaults to 100.
            **api_kwargs: Additional OpenAI API parameters (e.g. ``temperature``,
                ``top_p``, ``max_output_tokens``) forwarded verbatim to the
                underlying client.

        Returns:
            InferredSchema: An object containing:
                - instructions: Normalized statement of the extraction objective
                - fields: List of field specifications with names, types, and descriptions
                - inference_prompt: Reusable prompt for future extractions
                - model: Dynamically generated Pydantic model for parsing
                - task: PreparedTask for batch extraction operations

        Example:
            ```python
            df = pd.DataFrame({
                'text': [
                    "Order #123: Shipped to NYC, arriving Tuesday",
                    "Order #456: Delayed due to weather, new ETA Friday",
                    "Order #789: Delivered to customer in LA"
                ],
                'timestamp': ['2024-01-01', '2024-01-02', '2024-01-03']
            })

            # Infer schema for logistics tracking
            schema = df.ai.infer_schema(
                instructions="Extract shipping status and location data for logistics tracking"
            )

            # Apply the schema to extract structured data
            extracted_df = df.ai.task(schema.task)
            ```

        Note:
            Each row is converted to JSON before analysis. The inference
            process automatically detects hierarchical relationships and
            creates appropriate nested structures when present. The generated
            Pydantic model ensures type safety and validation.
        """
        return _df_rows_to_json_series(self._obj).ai.infer_schema(
            instructions=instructions,
            max_examples=max_examples,
            **api_kwargs,
        )

    def extract(self, column: str) -> pd.DataFrame:
        """Flatten one column of Pydantic models or dicts into top-level columns.

        The target column should contain Pydantic models or dicts. The source
        column is dropped from the result.

        Example:
            ```python
            df = pd.DataFrame([
                {"animal": {"name": "cat", "legs": 4}},
                {"animal": {"name": "dog", "legs": 4}},
                {"animal": {"name": "elephant", "legs": 4}},
            ])
            df.ai.extract("animal")
            ```
            This method returns a DataFrame with the same index as the original,
            where each column corresponds to a key in the dictionaries.
            The source column is dropped.

        Args:
            column (str): Column to expand.

        Returns:
            pandas.DataFrame: Original DataFrame with the extracted columns; the source column is dropped.
        """
        if column not in self._obj.columns:
            raise ValueError(f"Column '{column}' does not exist in the DataFrame.")

        return (
            self._obj.pipe(lambda df: df.reset_index(drop=True))
            .pipe(lambda df: df.join(df[column].ai.extract()))
            .pipe(lambda df: df.set_index(self._obj.index))
            .pipe(lambda df: df.drop(columns=[column]))
        )

    def fillna(
        self,
        target_column_name: str,
        max_examples: int = 500,
        batch_size: int | None = None,
        show_progress: bool = True,
    ) -> pd.DataFrame:
        """Fill missing values in a DataFrame column using AI-powered inference.

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

            # Fill missing values in the 'name' column
            filled_df = df.ai.fillna('name')

            # With progress bar for large datasets
            large_df = pd.DataFrame({'name': [None] * 1000, 'age': list(range(1000))})
            filled_df = large_df.ai.fillna('name', batch_size=32, show_progress=True)
            ```

        Note:
            If the target column has no missing values, the original DataFrame
            is returned unchanged.
        """

        missing_rows = self._obj[self._obj[target_column_name].isna()]
        if missing_rows.empty:
            return self._obj
        import openaivec.pandas_ext as _pkg

        task: PreparedTask = _pkg.fillna(self._obj, target_column_name, max_examples)

        filled_values: list[FillNaResponse] = missing_rows.ai.task(
            task=task, batch_size=batch_size, show_progress=show_progress
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

    def similarity(self, col1: str, col2: str) -> pd.Series:
        """Compute cosine similarity between two columns containing embedding vectors.

        This method calculates the cosine similarity between vectors stored in
        two columns of the DataFrame. The vectors should be numpy arrays or
        array-like objects that support dot product operations.

        Example:
            ```python
            df = pd.DataFrame({
                'vec1': [np.array([1, 0, 0]), np.array([0, 1, 0])],
                'vec2': [np.array([1, 0, 0]), np.array([1, 1, 0])]
            })
            similarities = df.ai.similarity('vec1', 'vec2')
            ```

        Args:
            col1 (str): Name of the first column containing embedding vectors.
            col2 (str): Name of the second column containing embedding vectors.

        Returns:
            pandas.Series: Series containing cosine similarity scores between
                corresponding vectors in col1 and col2, with values ranging
                from -1 to 1, where 1 indicates identical direction.
        """
        try:
            left = _embedding_series_to_matrix(self._obj[col1])
            right = _embedding_series_to_matrix(self._obj[col2])
        except Exception as exc:
            raise TypeError("Both columns must contain numeric vectors with consistent dimensions.") from exc

        numerator = np.einsum("ij,ij->i", left, right)
        denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)

        with np.errstate(divide="ignore", invalid="ignore"):
            similarity = np.divide(
                numerator,
                denominator,
                out=np.full(numerator.shape, np.nan, dtype=float),
                where=denominator != 0,
            )

        return pd.Series(similarity, index=self._obj.index, name="similarity")
