"""Pandas Series / DataFrame extension for OpenAI.

## Setup
```python
from openai import OpenAI, AzureOpenAI, AsyncOpenAI, AsyncAzureOpenAI
from openaivec import pandas_ext

# Option 1: Use environment variables (automatic detection)
# Set OPENAI_API_KEY or Azure OpenAI environment variables
# (AZURE_OPENAI_API_KEY, AZURE_OPENAI_BASE_URL, AZURE_OPENAI_API_VERSION)
# No explicit setup needed - clients are automatically created

# Option 2: Use an existing OpenAI client instance
client = OpenAI(api_key="your-api-key")
pandas_ext.use(client)

# Option 3: Use an existing Azure OpenAI client instance
azure_client = AzureOpenAI(
    api_key="your-azure-key",
    base_url="https://YOUR-RESOURCE-NAME.services.ai.azure.com/openai/v1/",
    api_version="preview"
)
pandas_ext.use(azure_client)

# Option 4: Use async Azure OpenAI client instance
async_azure_client = AsyncAzureOpenAI(
    api_key="your-azure-key",
    base_url="https://YOUR-RESOURCE-NAME.services.ai.azure.com/openai/v1/",
    api_version="preview"
)
pandas_ext.use_async(async_azure_client)

# Set up model names (optional, defaults shown)
pandas_ext.responses_model("gpt-4.1-mini")
pandas_ext.embeddings_model("text-embedding-3-small")
```

This module provides `.ai` and `.aio` accessors for pandas Series and DataFrames
to easily interact with OpenAI APIs for tasks like generating responses or embeddings.
"""

import inspect
import json
import logging
from typing import Awaitable, Callable, List, Type, TypeVar

import numpy as np
import pandas as pd
import tiktoken
from openai import AsyncOpenAI, OpenAI

__all__ = [
    "embeddings_model",
    "responses_model",
    "use",
    "use_async",
]
from pydantic import BaseModel

from openaivec._embeddings import AsyncBatchEmbeddings, BatchEmbeddings
from openaivec._model import EmbeddingsModelName, PreparedTask, ResponseFormat, ResponsesModelName
from openaivec._provider import CONTAINER, _check_azure_v1_api_url
from openaivec._proxy import AsyncBatchingMapProxy, BatchingMapProxy
from openaivec._responses import AsyncBatchResponses, BatchResponses
from openaivec.task.table import FillNaResponse, fillna

__all__ = [
    "use",
    "use_async",
    "responses_model",
    "embeddings_model",
]

_LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers (not exported)
# ---------------------------------------------------------------------------
def _df_rows_to_json_series(df: pd.DataFrame) -> pd.Series:
    """Return a Series of JSON strings (UTF-8, no ASCII escaping) representing DataFrame rows.

    Each element is the JSON serialisation of the corresponding row as a dict. Index and
    name are preserved so downstream operations retain alignment. This consolidates the
    previously duplicated inline pipeline used by responses*/task* DataFrame helpers.
    """
    return pd.Series(df.to_dict(orient="records"), index=df.index, name="record").map(
        lambda x: json.dumps(x, ensure_ascii=False)
    )


T = TypeVar("T")  # For pipe function return type


def use(client: OpenAI) -> None:
    """Register a custom OpenAI‑compatible client.

    Args:
        client (OpenAI): A pre‑configured `openai.OpenAI` or
            `openai.AzureOpenAI` instance.
            The same instance is reused by every helper in this module.
    """
    # Check Azure v1 API URL if using AzureOpenAI client
    if client.__class__.__name__ == "AzureOpenAI" and hasattr(client, "base_url"):
        _check_azure_v1_api_url(str(client.base_url))

    CONTAINER.register(OpenAI, lambda: client)


def use_async(client: AsyncOpenAI) -> None:
    """Register a custom asynchronous OpenAI‑compatible client.

    Args:
        client (AsyncOpenAI): A pre‑configured `openai.AsyncOpenAI` or
            `openai.AsyncAzureOpenAI` instance.
            The same instance is reused by every helper in this module.
    """
    # Check Azure v1 API URL if using AsyncAzureOpenAI client
    if client.__class__.__name__ == "AsyncAzureOpenAI" and hasattr(client, "base_url"):
        _check_azure_v1_api_url(str(client.base_url))

    CONTAINER.register(AsyncOpenAI, lambda: client)


def responses_model(name: str) -> None:
    """Override the model used for text responses.

    Args:
        name (str): For Azure OpenAI, use your deployment name. For OpenAI, use the model name
            (for example, ``gpt-4.1-mini``).
    """
    CONTAINER.register(ResponsesModelName, lambda: ResponsesModelName(name))


def embeddings_model(name: str) -> None:
    """Override the model used for text embeddings.

    Args:
        name (str): For Azure OpenAI, use your deployment name. For OpenAI, use the model name,
            e.g. ``text-embedding-3-small``.
    """
    CONTAINER.register(EmbeddingsModelName, lambda: EmbeddingsModelName(name))


def _extract_value(x, series_name):
    """Return a homogeneous ``dict`` representation of any Series value.

    Args:
        x (Any): Single element taken from the Series.
        series_name (str): Name of the Series (used for logging).

    Returns:
        dict: A dictionary representation or an empty ``dict`` if ``x`` cannot
            be coerced.
    """
    if x is None:
        return {}
    elif isinstance(x, BaseModel):
        return x.model_dump()
    elif isinstance(x, dict):
        return x

    _LOGGER.warning(
        f"The value '{x}' in the series '{series_name}' is not a dict or BaseModel. Returning an empty dict."
    )
    return {}


@pd.api.extensions.register_series_accessor("ai")
class OpenAIVecSeriesAccessor:
    """pandas Series accessor (``.ai``) that adds OpenAI helpers."""

    def __init__(self, series_obj: pd.Series):
        self._obj = series_obj

    def responses_with_cache(
        self,
        instructions: str,
        cache: BatchingMapProxy[str, ResponseFormat],
        response_format: Type[ResponseFormat] = str,
        temperature: float | None = 0.0,
        top_p: float = 1.0,
        **api_kwargs,
    ) -> pd.Series:
        client: BatchResponses = BatchResponses(
            client=CONTAINER.resolve(OpenAI),
            model_name=CONTAINER.resolve(ResponsesModelName).value,
            system_message=instructions,
            response_format=response_format,
            cache=cache,
            temperature=temperature,
            top_p=top_p,
        )

        # Forward any extra kwargs to the underlying Responses API.
        return pd.Series(client.parse(self._obj.tolist(), **api_kwargs), index=self._obj.index, name=self._obj.name)

    def embeddings_with_cache(
        self,
        cache: BatchingMapProxy[str, np.ndarray],
    ) -> pd.Series:
        """Compute OpenAI embeddings for every Series element using a provided cache.

        This method allows external control over caching behavior by accepting
        a pre-configured BatchingMapProxy instance, enabling cache sharing
        across multiple operations or custom batch size management.

        Args:
            cache (BatchingMapProxy[str, np.ndarray]): Pre-configured cache
                instance for managing API call batching and deduplication.
                Set cache.batch_size=None to enable automatic batch size optimization.

        Returns:
            pandas.Series: Series whose values are ``np.ndarray`` objects
                (dtype ``float32``).

        Example:
            ```python
            from openaivec._proxy import BatchingMapProxy
            import numpy as np

            # Create a shared cache with custom batch size
            shared_cache = BatchingMapProxy[str, np.ndarray](batch_size=64)

            animals = pd.Series(["cat", "dog", "elephant"])
            embeddings = animals.ai.embeddings_with_cache(cache=shared_cache)
            ```
        """
        client: BatchEmbeddings = BatchEmbeddings(
            client=CONTAINER.resolve(OpenAI),
            model_name=CONTAINER.resolve(EmbeddingsModelName).value,
            cache=cache,
        )

        return pd.Series(
            client.create(self._obj.tolist()),
            index=self._obj.index,
            name=self._obj.name,
        )

    def responses(
        self,
        instructions: str,
        response_format: Type[ResponseFormat] = str,
        batch_size: int | None = None,
        temperature: float | None = 0.0,
        top_p: float = 1.0,
        show_progress: bool = False,
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
            ```

        Args:
            instructions (str): System prompt prepended to every user message.
            response_format (Type[ResponseFormat], optional): Pydantic model or built‑in
                type the assistant should return. Defaults to ``str``.
            batch_size (int | None, optional): Number of prompts grouped into a single
                request. Defaults to ``None`` (automatic batch size optimization
                based on execution time). Set to a positive integer for fixed batch size.
            temperature (float, optional): Sampling temperature. Defaults to ``0.0``.
            top_p (float, optional): Nucleus sampling parameter. Defaults to ``1.0``.
            show_progress (bool, optional): Show progress bar in Jupyter notebooks. Defaults to ``False``.

        Returns:
            pandas.Series: Series whose values are instances of ``response_format``.
        """
        return self.responses_with_cache(
            instructions=instructions,
            cache=BatchingMapProxy(batch_size=batch_size, show_progress=show_progress),
            response_format=response_format,
            temperature=temperature,
            top_p=top_p,
            **api_kwargs,
        )

    def task_with_cache(
        self,
        task: PreparedTask[ResponseFormat],
        cache: BatchingMapProxy[str, ResponseFormat],
        **api_kwargs,
    ) -> pd.Series:
        """Execute a prepared task on every Series element using a provided cache.

        This mirrors ``responses_with_cache`` but uses the task's stored instructions,
        response format, temperature and top_p. A supplied ``BatchingMapProxy`` enables
        cross‑operation deduplicated reuse and external batch size / progress control.

        Args:
            task (PreparedTask): Prepared task (instructions + response_format + sampling params).
            cache (BatchingMapProxy[str, ResponseFormat]): Pre‑configured cache instance.

        Additional Keyword Args:
            Arbitrary OpenAI Responses API parameters (e.g. ``frequency_penalty``, ``presence_penalty``,
            ``seed``, etc.) forwarded verbatim to the underlying client. Core routing keys
            (``model``, system instructions, user input) are managed internally and cannot be overridden.

        Returns:
            pandas.Series: Task results aligned with the original Series index.

        Example:
            ```python
            from openaivec._proxy import BatchingMapProxy
            shared_cache = BatchingMapProxy(batch_size=64)
            reviews.ai.task_with_cache(sentiment_task, cache=shared_cache)
            ```
        """
        client: BatchResponses = BatchResponses(
            client=CONTAINER.resolve(OpenAI),
            model_name=CONTAINER.resolve(ResponsesModelName).value,
            system_message=task.instructions,
            response_format=task.response_format,
            cache=cache,
            temperature=task.temperature,
            top_p=task.top_p,
        )
        return pd.Series(client.parse(self._obj.tolist(), **api_kwargs), index=self._obj.index, name=self._obj.name)

    def task(
        self,
        task: PreparedTask,
        batch_size: int | None = None,
        show_progress: bool = False,
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
                response format, and other parameters for processing the inputs.
            batch_size (int | None, optional): Number of prompts grouped into a single
                request to optimize API usage. Defaults to ``None`` (automatic batch size
                optimization based on execution time). Set to a positive integer for fixed batch size.
            show_progress (bool, optional): Show progress bar in Jupyter notebooks. Defaults to ``False``.

        Additional Keyword Args:
            Arbitrary OpenAI Responses API parameters (e.g. ``frequency_penalty``, ``presence_penalty``,
            ``seed``, etc.) are forwarded verbatim to the underlying client. Core batching / routing
            keys (``model``, ``instructions`` / system message, user ``input``) are managed by the
            library and cannot be overridden.

        Returns:
            pandas.Series: Series whose values are instances of the task's response format.
        """
        return self.task_with_cache(
            task=task,
            cache=BatchingMapProxy(batch_size=batch_size, show_progress=show_progress),
            **api_kwargs,
        )

    def embeddings(self, batch_size: int | None = None, show_progress: bool = False) -> pd.Series:
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
            show_progress (bool, optional): Show progress bar in Jupyter notebooks. Defaults to ``False``.

        Returns:
            pandas.Series: Series whose values are ``np.ndarray`` objects
                (dtype ``float32``).
        """
        return self.embeddings_with_cache(
            cache=BatchingMapProxy(batch_size=batch_size, show_progress=show_progress),
        )

    def count_tokens(self) -> pd.Series:
        """Count `tiktoken` tokens per row.

        Example:
            ```python
            animals = pd.Series(["cat", "dog", "elephant"])
            animals.ai.count_tokens()
            ```
            This method uses the `tiktoken` library to count tokens based on the
            model name set by `responses_model`.

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


@pd.api.extensions.register_dataframe_accessor("ai")
class OpenAIVecDataFrameAccessor:
    """pandas DataFrame accessor (``.ai``) that adds OpenAI helpers."""

    def __init__(self, df_obj: pd.DataFrame):
        self._obj = df_obj

    def extract(self, column: str) -> pd.DataFrame:
        """Flatten one column of Pydantic models/dicts into top‑level columns.

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
            .pipe(lambda df: df.drop(columns=[column], axis=1))
        )

    def responses_with_cache(
        self,
        instructions: str,
        cache: BatchingMapProxy[str, ResponseFormat],
        response_format: Type[ResponseFormat] = str,
        temperature: float | None = 0.0,
        top_p: float = 1.0,
        **api_kwargs,
    ) -> pd.Series:
        """Generate a response for each row after serialising it to JSON using a provided cache.

        This method allows external control over caching behavior by accepting
        a pre-configured BatchingMapProxy instance, enabling cache sharing
        across multiple operations or custom batch size management.

        Args:
            instructions (str): System prompt for the assistant.
            cache (BatchingMapProxy[str, ResponseFormat]): Pre-configured cache
                instance for managing API call batching and deduplication.
                Set cache.batch_size=None to enable automatic batch size optimization.
            response_format (Type[ResponseFormat], optional): Desired Python type of the
                responses. Defaults to ``str``.
            temperature (float, optional): Sampling temperature. Defaults to ``0.0``.
            top_p (float, optional): Nucleus sampling parameter. Defaults to ``1.0``.

        Returns:
            pandas.Series: Responses aligned with the DataFrame's original index.

        Example:
            ```python
            from openaivec._proxy import BatchingMapProxy

            # Create a shared cache with custom batch size
            shared_cache = BatchingMapProxy(batch_size=64)

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
        """
        return _df_rows_to_json_series(self._obj).ai.responses_with_cache(
            instructions=instructions,
            cache=cache,
            response_format=response_format,
            temperature=temperature,
            top_p=top_p,
            **api_kwargs,
        )

    def responses(
        self,
        instructions: str,
        response_format: Type[ResponseFormat] = str,
        batch_size: int | None = None,
        temperature: float | None = 0.0,
        top_p: float = 1.0,
        show_progress: bool = False,
        **api_kwargs,
    ) -> pd.Series:
        """Generate a response for each row after serialising it to JSON.

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
            instructions (str): System prompt for the assistant.
            response_format (Type[ResponseFormat], optional): Desired Python type of the
                responses. Defaults to ``str``.
            batch_size (int | None, optional): Number of requests sent in one batch.
                Defaults to ``None`` (automatic batch size optimization
                based on execution time). Set to a positive integer for fixed batch size.
            temperature (float, optional): Sampling temperature. Defaults to ``0.0``.
            top_p (float, optional): Nucleus sampling parameter. Defaults to ``1.0``.
            show_progress (bool, optional): Show progress bar in Jupyter notebooks. Defaults to ``False``.

        Returns:
            pandas.Series: Responses aligned with the DataFrame's original index.
        """
        return self.responses_with_cache(
            instructions=instructions,
            cache=BatchingMapProxy(batch_size=batch_size, show_progress=show_progress),
            response_format=response_format,
            temperature=temperature,
            top_p=top_p,
            **api_kwargs,
        )

    def task(
        self,
        task: PreparedTask,
        batch_size: int | None = None,
        show_progress: bool = False,
        **api_kwargs,
    ) -> pd.Series:
        """Execute a prepared task on each DataFrame row after serialising it to JSON.

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
                response format, and other parameters for processing the inputs.
            batch_size (int | None, optional): Number of requests sent in one batch
                to optimize API usage. Defaults to ``None`` (automatic batch size
                optimization based on execution time). Set to a positive integer for fixed batch size.
            show_progress (bool, optional): Show progress bar in Jupyter notebooks. Defaults to ``False``.

        Additional Keyword Args:
            Arbitrary OpenAI Responses API parameters (e.g. ``frequency_penalty``, ``presence_penalty``,
            ``seed``, etc.) are forwarded verbatim to the underlying client. Core batching / routing
            keys (``model``, ``instructions`` / system message, user ``input``) are managed by the
            library and cannot be overridden.

        Returns:
            pandas.Series: Series whose values are instances of the task's
                response format, aligned with the DataFrame's original index.
        """
        return _df_rows_to_json_series(self._obj).ai.task(
            task=task,
            batch_size=batch_size,
            show_progress=show_progress,
            **api_kwargs,
        )

    def task_with_cache(
        self,
        task: PreparedTask[ResponseFormat],
        cache: BatchingMapProxy[str, ResponseFormat],
        **api_kwargs,
    ) -> pd.Series:
        """Execute a prepared task on each DataFrame row after serializing it to JSON using a provided cache.

        Args:
            task (PreparedTask): Prepared task (instructions + response_format + sampling params).
            cache (BatchingMapProxy[str, ResponseFormat]): Pre‑configured cache instance.

        Additional Keyword Args:
            Arbitrary OpenAI Responses API parameters (e.g. ``frequency_penalty``, ``presence_penalty``,
            ``seed``) forwarded verbatim. Core routing keys are managed internally.

        Returns:
            pandas.Series: Task results aligned with the DataFrame's original index.
        """
        return _df_rows_to_json_series(self._obj).ai.task_with_cache(
            task=task,
            cache=cache,
            **api_kwargs,
        )

    def fillna(
        self,
        target_column_name: str,
        max_examples: int = 500,
        batch_size: int | None = None,
        show_progress: bool = False,
        **api_kwargs,
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
            show_progress (bool, optional): Show progress bar in Jupyter notebooks. Defaults to ``False``.

        Additional Keyword Args:
            Arbitrary OpenAI Responses API parameters (e.g. ``frequency_penalty``, ``presence_penalty``,
            ``seed``, etc.) are forwarded verbatim to the underlying task execution.

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

        task: PreparedTask = fillna(self._obj, target_column_name, max_examples)
        missing_rows = self._obj[self._obj[target_column_name].isna()]
        if missing_rows.empty:
            return self._obj

        filled_values: List[FillNaResponse] = missing_rows.ai.task(
            task=task, batch_size=batch_size, show_progress=show_progress, **api_kwargs
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

        Args:
            col1 (str): Name of the first column containing embedding vectors.
            col2 (str): Name of the second column containing embedding vectors.

        Returns:
            pandas.Series: Series containing cosine similarity scores between
                corresponding vectors in col1 and col2, with values ranging
                from -1 to 1, where 1 indicates identical direction.

        Example:
            ```python
            df = pd.DataFrame({
                'vec1': [np.array([1, 0, 0]), np.array([0, 1, 0])],
                'vec2': [np.array([1, 0, 0]), np.array([1, 1, 0])]
            })
            similarities = df.ai.similarity('vec1', 'vec2')
            ```
        """
        return self._obj.apply(
            lambda row: np.dot(row[col1], row[col2]) / (np.linalg.norm(row[col1]) * np.linalg.norm(row[col2])),
            axis=1,
        ).rename("similarity")  # type: ignore[arg-type]


@pd.api.extensions.register_series_accessor("aio")
class AsyncOpenAIVecSeriesAccessor:
    """pandas Series accessor (``.aio``) that adds OpenAI helpers."""

    def __init__(self, series_obj: pd.Series):
        self._obj = series_obj

    async def responses_with_cache(
        self,
        instructions: str,
        cache: AsyncBatchingMapProxy[str, ResponseFormat],
        response_format: Type[ResponseFormat] = str,
        temperature: float | None = 0.0,
        top_p: float = 1.0,
        **api_kwargs,
    ) -> pd.Series:
        """Call an LLM once for every Series element using a provided cache (asynchronously).

        This method allows external control over caching behavior by accepting
        a pre-configured AsyncBatchingMapProxy instance, enabling cache sharing
        across multiple operations or custom batch size management. The concurrency
        is controlled by the cache instance itself.

        Args:
            instructions (str): System prompt prepended to every user message.
            cache (AsyncBatchingMapProxy[str, ResponseFormat]): Pre-configured cache
                instance for managing API call batching and deduplication.
                Set cache.batch_size=None to enable automatic batch size optimization.
            response_format (Type[ResponseFormat], optional): Pydantic model or built‑in
                type the assistant should return. Defaults to ``str``.
            temperature (float | None, optional): Sampling temperature. ``None`` omits the
                parameter (recommended for reasoning models). Defaults to ``0.0``.
            top_p (float, optional): Nucleus sampling parameter. Defaults to ``1.0``.
            **api_kwargs: Additional keyword arguments forwarded verbatim to
                ``AsyncOpenAI.responses.parse`` (e.g. ``max_output_tokens``, penalties,
                future parameters). Core batching keys (model, instructions, input,
                text_format) are protected and silently ignored if provided.

        Returns:
            pandas.Series: Series whose values are instances of ``response_format``.

        Example:
            ```python
            result = await series.aio.responses_with_cache(
                "classify",
                cache=shared,
                max_output_tokens=256,
                frequency_penalty=0.2,
            )
            ```

        Note:
            This is an asynchronous method and must be awaited.
        """
        client: AsyncBatchResponses = AsyncBatchResponses(
            client=CONTAINER.resolve(AsyncOpenAI),
            model_name=CONTAINER.resolve(ResponsesModelName).value,
            system_message=instructions,
            response_format=response_format,
            cache=cache,
            temperature=temperature,
            top_p=top_p,
        )
        results = await client.parse(self._obj.tolist(), **api_kwargs)
        return pd.Series(results, index=self._obj.index, name=self._obj.name)

    async def embeddings_with_cache(
        self,
        cache: AsyncBatchingMapProxy[str, np.ndarray],
    ) -> pd.Series:
        """Compute OpenAI embeddings for every Series element using a provided cache (asynchronously).

        This method allows external control over caching behavior by accepting
        a pre-configured AsyncBatchingMapProxy instance, enabling cache sharing
        across multiple operations or custom batch size management. The concurrency
        is controlled by the cache instance itself.

        Args:
            cache (AsyncBatchingMapProxy[str, np.ndarray]): Pre-configured cache
                instance for managing API call batching and deduplication.
                Set cache.batch_size=None to enable automatic batch size optimization.

        Returns:
            pandas.Series: Series whose values are ``np.ndarray`` objects
                (dtype ``float32``).

        Example:
            ```python
            from openaivec._proxy import AsyncBatchingMapProxy
            import numpy as np

            # Create a shared cache with custom batch size and concurrency
            shared_cache = AsyncBatchingMapProxy[str, np.ndarray](
                batch_size=64, max_concurrency=4
            )

            animals = pd.Series(["cat", "dog", "elephant"])
            # Must be awaited
            embeddings = await animals.aio.embeddings_with_cache(cache=shared_cache)
            ```

        Note:
            This is an asynchronous method and must be awaited.
        """
        client: AsyncBatchEmbeddings = AsyncBatchEmbeddings(
            client=CONTAINER.resolve(AsyncOpenAI),
            model_name=CONTAINER.resolve(EmbeddingsModelName).value,
            cache=cache,
        )

        # Await the async operation
        results = await client.create(self._obj.tolist())

        return pd.Series(
            results,
            index=self._obj.index,
            name=self._obj.name,
        )

    async def task_with_cache(
        self,
        task: PreparedTask[ResponseFormat],
        cache: AsyncBatchingMapProxy[str, ResponseFormat],
        **api_kwargs,
    ) -> pd.Series:
        """Execute a prepared task on every Series element using a provided cache (asynchronously).

        This method allows external control over caching behavior by accepting
        a pre-configured AsyncBatchingMapProxy instance, enabling cache sharing
        across multiple operations or custom batch size management. The concurrency
        is controlled by the cache instance itself.

        Args:
            task (PreparedTask): A pre-configured task containing instructions,
                response format, and other parameters for processing the inputs.
            cache (AsyncBatchingMapProxy[str, ResponseFormat]): Pre-configured cache
                instance for managing API call batching and deduplication.
                Set cache.batch_size=None to enable automatic batch size optimization.

        Additional Keyword Args:
            Arbitrary OpenAI Responses API parameters (e.g. ``frequency_penalty``, ``presence_penalty``,
            ``seed``, etc.) are forwarded verbatim to the underlying client. Core batching / routing
            keys (``model``, ``instructions`` / system message, user ``input``) are managed by the
            library and cannot be overridden.

        Returns:
            pandas.Series: Series whose values are instances of the task's
                response format, aligned with the original Series index.

        Example:
            ```python
            from openaivec._model import PreparedTask
            from openaivec._proxy import AsyncBatchingMapProxy

            # Create a shared cache with custom batch size and concurrency
            shared_cache = AsyncBatchingMapProxy(batch_size=64, max_concurrency=4)

            # Assume you have a prepared task for sentiment analysis
            sentiment_task = PreparedTask(...)

            reviews = pd.Series(["Great product!", "Not satisfied", "Amazing quality"])
            # Must be awaited
            results = await reviews.aio.task_with_cache(sentiment_task, cache=shared_cache)
            ```

        Note:
            This is an asynchronous method and must be awaited.
        """
        client = AsyncBatchResponses(
            client=CONTAINER.resolve(AsyncOpenAI),
            model_name=CONTAINER.resolve(ResponsesModelName).value,
            system_message=task.instructions,
            response_format=task.response_format,
            cache=cache,
            temperature=task.temperature,
            top_p=task.top_p,
        )
        # Await the async operation
        results = await client.parse(self._obj.tolist(), **api_kwargs)

        return pd.Series(results, index=self._obj.index, name=self._obj.name)

    async def responses(
        self,
        instructions: str,
        response_format: Type[ResponseFormat] = str,
        batch_size: int | None = None,
        temperature: float | None = 0.0,
        top_p: float = 1.0,
        max_concurrency: int = 8,
        show_progress: bool = False,
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
            response_format (Type[ResponseFormat], optional): Pydantic model or built‑in
                type the assistant should return. Defaults to ``str``.
            batch_size (int | None, optional): Number of prompts grouped into a single
                request. Defaults to ``None`` (automatic batch size optimization
                based on execution time). Set to a positive integer for fixed batch size.
            temperature (float, optional): Sampling temperature. Defaults to ``0.0``.
            top_p (float, optional): Nucleus sampling parameter. Defaults to ``1.0``.
            max_concurrency (int, optional): Maximum number of concurrent
                requests. Defaults to ``8``.
            show_progress (bool, optional): Show progress bar in Jupyter notebooks. Defaults to ``False``.

        Returns:
            pandas.Series: Series whose values are instances of ``response_format``.

        Note:
            This is an asynchronous method and must be awaited.
        """
        return await self.responses_with_cache(
            instructions=instructions,
            cache=AsyncBatchingMapProxy(
                batch_size=batch_size, max_concurrency=max_concurrency, show_progress=show_progress
            ),
            response_format=response_format,
            temperature=temperature,
            top_p=top_p,
            **api_kwargs,
        )

    async def embeddings(
        self, batch_size: int | None = None, max_concurrency: int = 8, show_progress: bool = False
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
            show_progress (bool, optional): Show progress bar in Jupyter notebooks. Defaults to ``False``.

        Returns:
            pandas.Series: Series whose values are ``np.ndarray`` objects
                (dtype ``float32``).

        Note:
            This is an asynchronous method and must be awaited.
        """
        return await self.embeddings_with_cache(
            cache=AsyncBatchingMapProxy(
                batch_size=batch_size, max_concurrency=max_concurrency, show_progress=show_progress
            ),
        )

    async def task(
        self,
        task: PreparedTask,
        batch_size: int | None = None,
        max_concurrency: int = 8,
        show_progress: bool = False,
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
                response format, and other parameters for processing the inputs.
            batch_size (int | None, optional): Number of prompts grouped into a single
                request to optimize API usage. Defaults to ``None`` (automatic batch size
                optimization based on execution time). Set to a positive integer for fixed batch size.
            max_concurrency (int, optional): Maximum number of concurrent
                requests. Defaults to 8.
            show_progress (bool, optional): Show progress bar in Jupyter notebooks. Defaults to ``False``.

        Additional Keyword Args:
            Arbitrary OpenAI Responses API parameters (e.g. ``frequency_penalty``, ``presence_penalty``,
            ``seed``, etc.) are forwarded verbatim to the underlying client. Core batching / routing
            keys (``model``, ``instructions`` / system message, user ``input``) are managed by the
            library and cannot be overridden.

        Returns:
            pandas.Series: Series whose values are instances of the task's
                response format, aligned with the original Series index.

        Note:
            This is an asynchronous method and must be awaited.
        """
        return await self.task_with_cache(
            task=task,
            cache=AsyncBatchingMapProxy(
                batch_size=batch_size, max_concurrency=max_concurrency, show_progress=show_progress
            ),
            **api_kwargs,
        )


@pd.api.extensions.register_dataframe_accessor("aio")
class AsyncOpenAIVecDataFrameAccessor:
    """pandas DataFrame accessor (``.aio``) that adds OpenAI helpers."""

    def __init__(self, df_obj: pd.DataFrame):
        self._obj = df_obj

    async def responses_with_cache(
        self,
        instructions: str,
        cache: AsyncBatchingMapProxy[str, ResponseFormat],
        response_format: Type[ResponseFormat] = str,
        temperature: float | None = 0.0,
        top_p: float = 1.0,
        **api_kwargs,
    ) -> pd.Series:
        """Generate a response for each row after serialising it to JSON using a provided cache (asynchronously).

        This method allows external control over caching behavior by accepting
        a pre-configured AsyncBatchingMapProxy instance, enabling cache sharing
        across multiple operations or custom batch size management. The concurrency
        is controlled by the cache instance itself.

        Args:
            instructions (str): System prompt for the assistant.
            cache (AsyncBatchingMapProxy[str, ResponseFormat]): Pre-configured cache
                instance for managing API call batching and deduplication.
                Set cache.batch_size=None to enable automatic batch size optimization.
            response_format (Type[ResponseFormat], optional): Desired Python type of the
                responses. Defaults to ``str``.
            temperature (float, optional): Sampling temperature. Defaults to ``0.0``.
            top_p (float, optional): Nucleus sampling parameter. Defaults to ``1.0``.

        Returns:
            pandas.Series: Responses aligned with the DataFrame's original index.

        Example:
            ```python
            from openaivec._proxy import AsyncBatchingMapProxy

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

        Note:
            This is an asynchronous method and must be awaited.
        """
        # Await the call to the async Series method using .aio
        return await _df_rows_to_json_series(self._obj).aio.responses_with_cache(
            instructions=instructions,
            cache=cache,
            response_format=response_format,
            temperature=temperature,
            top_p=top_p,
            **api_kwargs,
        )

    async def responses(
        self,
        instructions: str,
        response_format: Type[ResponseFormat] = str,
        batch_size: int | None = None,
        temperature: float | None = 0.0,
        top_p: float = 1.0,
        max_concurrency: int = 8,
        show_progress: bool = False,
        **api_kwargs,
    ) -> pd.Series:
        """Generate a response for each row after serialising it to JSON (asynchronously).

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
            response_format (Type[ResponseFormat], optional): Desired Python type of the
                responses. Defaults to ``str``.
            batch_size (int | None, optional): Number of requests sent in one batch.
                Defaults to ``None`` (automatic batch size optimization
                based on execution time). Set to a positive integer for fixed batch size.
            temperature (float, optional): Sampling temperature. Defaults to ``0.0``.
            top_p (float, optional): Nucleus sampling parameter. Defaults to ``1.0``.
            max_concurrency (int, optional): Maximum number of concurrent
                requests. Defaults to ``8``.
            show_progress (bool, optional): Show progress bar in Jupyter notebooks. Defaults to ``False``.

        Returns:
            pandas.Series: Responses aligned with the DataFrame's original index.

        Note:
            This is an asynchronous method and must be awaited.
        """
        return await self.responses_with_cache(
            instructions=instructions,
            cache=AsyncBatchingMapProxy(
                batch_size=batch_size, max_concurrency=max_concurrency, show_progress=show_progress
            ),
            response_format=response_format,
            temperature=temperature,
            top_p=top_p,
            **api_kwargs,
        )

    async def task(
        self,
        task: PreparedTask,
        batch_size: int | None = None,
        max_concurrency: int = 8,
        show_progress: bool = False,
        **api_kwargs,
    ) -> pd.Series:
        """Execute a prepared task on each DataFrame row after serialising it to JSON (asynchronously).

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
                response format, and other parameters for processing the inputs.
            batch_size (int | None, optional): Number of requests sent in one batch
                to optimize API usage. Defaults to ``None`` (automatic batch size
                optimization based on execution time). Set to a positive integer for fixed batch size.
            max_concurrency (int, optional): Maximum number of concurrent
                requests. Defaults to 8.
            show_progress (bool, optional): Show progress bar in Jupyter notebooks. Defaults to ``False``.

        Additional Keyword Args:
            Arbitrary OpenAI Responses API parameters (e.g. ``frequency_penalty``, ``presence_penalty``,
            ``seed``, etc.) are forwarded verbatim to the underlying client. Core batching / routing
            keys (``model``, ``instructions`` / system message, user ``input``) are managed by the
            library and cannot be overridden.

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

    async def task_with_cache(
        self,
        task: PreparedTask[ResponseFormat],
        cache: AsyncBatchingMapProxy[str, ResponseFormat],
        **api_kwargs,
    ) -> pd.Series:
        """Execute a prepared task on each DataFrame row after serializing it to JSON using a provided cache (async).

        Args:
            task (PreparedTask): Prepared task (instructions + response_format + sampling params).
            cache (AsyncBatchingMapProxy[str, ResponseFormat]): Pre‑configured async cache instance.

        Additional Keyword Args:
            Arbitrary OpenAI Responses API parameters forwarded verbatim. Core routing keys are protected.

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

    async def pipe(self, func: Callable[[pd.DataFrame], Awaitable[T] | T]) -> T:
        """
        Apply a function to the DataFrame, supporting both synchronous and asynchronous functions.

        This method allows chaining operations on the DataFrame, similar to pandas' `pipe` method,
        but with support for asynchronous functions.

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
        show_progress: bool = False,
        **api_kwargs,
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
            show_progress (bool, optional): Show progress bar in Jupyter notebooks. Defaults to ``False``.

        Additional Keyword Args:
            Arbitrary OpenAI Responses API parameters (e.g. ``frequency_penalty``, ``presence_penalty``,
            ``seed``, etc.) are forwarded verbatim to the underlying task execution.

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

        task: PreparedTask = fillna(self._obj, target_column_name, max_examples)
        missing_rows = self._obj[self._obj[target_column_name].isna()]
        if missing_rows.empty:
            return self._obj

        filled_values: List[FillNaResponse] = await missing_rows.aio.task(
            task=task, batch_size=batch_size, max_concurrency=max_concurrency, show_progress=show_progress, **api_kwargs
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
