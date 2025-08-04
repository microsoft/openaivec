"""Pandas Series / DataFrame extension for OpenAI.

## Setup
```python
from openai import OpenAI, AzureOpenAI, AsyncOpenAI, AsyncAzureOpenAI
from openaivec import pandas_ext

# Option 1: Use environment variables (automatic detection)
# Set OPENAI_API_KEY or Azure OpenAI environment variables
# (AZURE_OPENAI_API_KEY, AZURE_OPENAI_API_ENDPOINT, AZURE_OPENAI_API_VERSION)
# No explicit setup needed - clients are automatically created

# Option 2: Use an existing OpenAI client instance
client = OpenAI(api_key="your-api-key")
pandas_ext.use(client)

# Option 3: Use an existing Azure OpenAI client instance
azure_client = AzureOpenAI(
    api_key="your-azure-key",
    azure_endpoint="https://your-resource.openai.azure.com",
    api_version="2024-02-01"
)
pandas_ext.use(azure_client)

# Option 4: Use async clients
async_client = AsyncOpenAI(api_key="your-api-key")
pandas_ext.use_async(async_client)

# Set up model names (optional, defaults shown)
pandas_ext.responses_model("gpt-4o-mini")
pandas_ext.embeddings_model("text-embedding-3-small")
```

This module provides `.ai` and `.aio` accessors for pandas Series and DataFrames
to easily interact with OpenAI APIs for tasks like generating responses or embeddings.
"""

import inspect
import json
import logging
from typing import Any, Awaitable, Callable, List, Type, TypeVar

import numpy as np
import pandas as pd
import tiktoken
from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel

from .di import Container
from .embeddings import AsyncBatchEmbeddings, BatchEmbeddings
from .model import EmbeddingsModelName, PreparedTask, ResponsesModelName
from .provider import provide_async_openai_client, provide_openai_client
from .responses import AsyncBatchResponses, BatchResponses
from .task.table import FillNaResponse, fillna

__all__ = [
    "use",
    "use_async",
    "responses_model",
    "embeddings_model",
]

_LOGGER = logging.getLogger(__name__)


T = TypeVar("T")

_DI = Container()
_DI.register(OpenAI, provide_openai_client)
_DI.register(AsyncOpenAI, provide_async_openai_client)
_DI.register(ResponsesModelName, lambda: ResponsesModelName("gpt-4o-mini"))
_DI.register(EmbeddingsModelName, lambda: EmbeddingsModelName("text-embedding-3-small"))


def _provide_tiktoken_encoding() -> tiktoken.Encoding:
    model_name = _DI.resolve(ResponsesModelName).value
    try:
        return tiktoken.encoding_for_model(model_name)
    except KeyError:
        _LOGGER.info(
            "The model name '%s' is not supported by tiktoken. Using 'o200k_base' encoding instead.",
            model_name,
        )
        return tiktoken.get_encoding("o200k_base")


_DI.register(tiktoken.Encoding, _provide_tiktoken_encoding)


def use(client: OpenAI) -> None:
    """Register a custom OpenAI‑compatible client.

    Args:
        client (OpenAI): A pre‑configured `openai.OpenAI` or
            `openai.AzureOpenAI` instance.
            The same instance is reused by every helper in this module.
    """
    _DI.register(OpenAI, lambda: client)


def use_async(client: AsyncOpenAI) -> None:
    """Register a custom asynchronous OpenAI‑compatible client.

    Args:
        client (AsyncOpenAI): A pre‑configured `openai.AsyncOpenAI` or
            `openai.AsyncAzureOpenAI` instance.
            The same instance is reused by every helper in this module.
    """
    _DI.register(AsyncOpenAI, lambda: client)


def responses_model(name: str) -> None:
    """Override the model used for text responses.

    Args:
        name (str): Model name as listed in the OpenAI API
            (for example, ``gpt-4o-mini``).
    """
    _DI.register(ResponsesModelName, lambda: ResponsesModelName(name))
    _DI.register(tiktoken.Encoding, _provide_tiktoken_encoding)


def embeddings_model(name: str) -> None:
    """Override the model used for text embeddings.

    Args:
        name (str): Embedding model name, e.g. ``text-embedding-3-small``.
    """
    _DI.register(EmbeddingsModelName, lambda: EmbeddingsModelName(name))


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

    def responses(
        self,
        instructions: str,
        response_format: Type[T] = str,
        batch_size: int = 128,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> pd.Series:
        """Call an LLM once for every Series element.

        Example:
            ```python
            animals = pd.Series(["cat", "dog", "elephant"])
            animals.ai.responses("translate to French")
            ```
            This method returns a Series of strings, each containing the
            assistant's response to the corresponding input.
            The model used is set by the `responses_model` function.
            The default model is `gpt-4o-mini`.

        Args:
            instructions (str): System prompt prepended to every user message.
            response_format (Type[T], optional): Pydantic model or built‑in
                type the assistant should return. Defaults to ``str``.
            batch_size (int, optional): Number of prompts grouped into a single
                request. Defaults to ``128``.
            temperature (float, optional): Sampling temperature. Defaults to ``0.0``.
            top_p (float, optional): Nucleus sampling parameter. Defaults to ``1.0``.

        Returns:
            pandas.Series: Series whose values are instances of ``response_format``.
        """
        client: BatchResponses = BatchResponses(
            client=_DI.resolve(OpenAI),
            model_name=_DI.resolve(ResponsesModelName).value,
            system_message=instructions,
            response_format=response_format,
            temperature=temperature,
            top_p=top_p,
        )

        return pd.Series(
            client.parse(self._obj.tolist(), batch_size=batch_size),
            index=self._obj.index,
            name=self._obj.name,
        )

    def task(self, task: PreparedTask, batch_size: int = 128) -> pd.Series:
        """Execute a prepared task on every Series element.

        This method applies a pre-configured task to each element in the Series,
        using the task's instructions and response format to generate structured
        responses from the language model.

        Example:
            ```python
            from openaivec.model import PreparedTask

            # Assume you have a prepared task for sentiment analysis
            sentiment_task = PreparedTask(...)

            reviews = pd.Series(["Great product!", "Not satisfied", "Amazing quality"])
            results = reviews.ai.task(sentiment_task)
            ```
            This method returns a Series containing the task results for each
            corresponding input element, following the task's defined structure.

        Args:
            task (PreparedTask): A pre-configured task containing instructions,
                response format, and other parameters for processing the inputs.
            batch_size (int, optional): Number of prompts grouped into a single
                request to optimize API usage. Defaults to 128.

        Returns:
            pandas.Series: Series whose values are instances of the task's
                response format, aligned with the original Series index.
        """
        client = BatchResponses.of_task(
            client=_DI.resolve(OpenAI), model_name=_DI.resolve(ResponsesModelName).value, task=task
        )

        return pd.Series(
            client.parse(self._obj.tolist(), batch_size=batch_size),
            index=self._obj.index,
            name=self._obj.name,
        )

    def embeddings(self, batch_size: int = 128) -> pd.Series:
        """Compute OpenAI embeddings for every Series element.

        Example:
            ```python
            animals = pd.Series(["cat", "dog", "elephant"])
            animals.ai.embeddings()
            ```
            This method returns a Series of numpy arrays, each containing the
            embedding vector for the corresponding input.
            The embedding model is set by the `embeddings_model` function.
            The default embedding model is `text-embedding-3-small`.

        Args:
            batch_size (int, optional): Number of inputs grouped into a
                single request. Defaults to ``128``.

        Returns:
            pandas.Series: Series whose values are ``np.ndarray`` objects
                (dtype ``float32``).
        """
        client: BatchEmbeddings = BatchEmbeddings(
            client=_DI.resolve(OpenAI),
            model_name=_DI.resolve(EmbeddingsModelName).value,
        )

        return pd.Series(
            client.create(self._obj.tolist(), batch_size=batch_size),
            index=self._obj.index,
            name=self._obj.name,
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
        encoding: tiktoken.Encoding = _DI.resolve(tiktoken.Encoding)
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

    def responses(
        self,
        instructions: str,
        response_format: Type[T] = str,
        batch_size: int = 128,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> pd.Series:
        """Generate a response for each row after serialising it to JSON.

        Example:
            ```python
            df = pd.DataFrame([
                {"name": "cat", "legs": 4},
                {"name": "dog", "legs": 4},
                {"name": "elephant", "legs": 4},
            ])
            df.ai.responses("what is the animal's name?")
            ```
            This method returns a Series of strings, each containing the
            assistant's response to the corresponding input.
            Each row is serialised to JSON before being sent to the assistant.
            The model used is set by the `responses_model` function.
            The default model is `gpt-4o-mini`.

        Args:
            instructions (str): System prompt for the assistant.
            response_format (Type[T], optional): Desired Python type of the
                responses. Defaults to ``str``.
            batch_size (int, optional): Number of requests sent in one batch.
                Defaults to ``128``.
            temperature (float, optional): Sampling temperature. Defaults to ``0.0``.
            top_p (float, optional): Nucleus sampling parameter. Defaults to ``1.0``.

        Returns:
            pandas.Series: Responses aligned with the DataFrame’s original index.
        """
        return self._obj.pipe(
            lambda df: (
                df.pipe(lambda df: pd.Series(df.to_dict(orient="records"), index=df.index, name="record"))
                .map(lambda x: json.dumps(x, ensure_ascii=False))
                .ai.responses(
                    instructions=instructions,
                    response_format=response_format,
                    batch_size=batch_size,
                    temperature=temperature,
                    top_p=top_p,
                )
            )
        )

    def task(self, task: PreparedTask, batch_size: int = 128) -> pd.Series:
        """Execute a prepared task on each DataFrame row after serialising it to JSON.

        This method applies a pre-configured task to each row in the DataFrame,
        using the task's instructions and response format to generate structured
        responses from the language model. Each row is serialised to JSON before
        being processed by the task.

        Example:
            ```python
            from openaivec.model import PreparedTask

            # Assume you have a prepared task for data analysis
            analysis_task = PreparedTask(...)

            df = pd.DataFrame([
                {"name": "cat", "legs": 4},
                {"name": "dog", "legs": 4},
                {"name": "elephant", "legs": 4},
            ])
            results = df.ai.task(analysis_task)
            ```
            This method returns a Series containing the task results for each
            corresponding row, following the task's defined structure.

        Args:
            task (PreparedTask): A pre-configured task containing instructions,
                response format, and other parameters for processing the inputs.
            batch_size (int, optional): Number of requests sent in one batch
                to optimize API usage. Defaults to 128.

        Returns:
            pandas.Series: Series whose values are instances of the task's
                response format, aligned with the DataFrame's original index.
        """
        return self._obj.pipe(
            lambda df: (
                df.pipe(lambda df: pd.Series(df.to_dict(orient="records"), index=df.index, name="record"))
                .map(lambda x: json.dumps(x, ensure_ascii=False))
                .ai.task(task=task, batch_size=batch_size)
            )
        )

    def fillna(self, target_column_name: str, max_examples: int = 500, batch_size: int = 128) -> pd.DataFrame:
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
            batch_size (int, optional): Number of requests sent in one batch
                to optimize API usage. Defaults to 128.

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
            ```

        Note:
            If the target column has no missing values, the original DataFrame
            is returned unchanged.
        """

        task: PreparedTask = fillna(self._obj, target_column_name, max_examples)
        missing_rows = self._obj[self._obj[target_column_name].isna()]
        if missing_rows.empty:
            return self._obj

        filled_values: List[FillNaResponse] = missing_rows.ai.task(task=task, batch_size=batch_size)

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
        return self._obj.apply(
            lambda row: np.dot(row[col1], row[col2]) / (np.linalg.norm(row[col1]) * np.linalg.norm(row[col2])),
            axis=1,
        ).rename("similarity")


@pd.api.extensions.register_series_accessor("aio")
class AsyncOpenAIVecSeriesAccessor:
    """pandas Series accessor (``.aio``) that adds OpenAI helpers."""

    def __init__(self, series_obj: pd.Series):
        self._obj = series_obj

    async def responses(
        self,
        instructions: str,
        response_format: Type[T] = str,
        batch_size: int = 128,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_concurrency: int = 8,
    ) -> pd.Series:
        """Call an LLM once for every Series element (asynchronously).

        Example:
            ```python
            animals = pd.Series(["cat", "dog", "elephant"])
            # Must be awaited
            results = await animals.aio.responses("translate to French")
            ```
            This method returns a Series of strings, each containing the
            assistant's response to the corresponding input.
            The model used is set by the `responses_model` function.
            The default model is `gpt-4o-mini`.

        Args:
            instructions (str): System prompt prepended to every user message.
            response_format (Type[T], optional): Pydantic model or built‑in
                type the assistant should return. Defaults to ``str``.
            batch_size (int, optional): Number of prompts grouped into a single
                request. Defaults to ``128``.
            temperature (float, optional): Sampling temperature. Defaults to ``0.0``.
            top_p (float, optional): Nucleus sampling parameter. Defaults to ``1.0``.
            max_concurrency (int, optional): Maximum number of concurrent
                requests. Defaults to ``8``.

        Returns:
            pandas.Series: Series whose values are instances of ``response_format``.

        Note:
            This is an asynchronous method and must be awaited.
        """
        client: AsyncBatchResponses = AsyncBatchResponses(
            client=_DI.resolve(AsyncOpenAI),
            model_name=_DI.resolve(ResponsesModelName).value,
            system_message=instructions,
            response_format=response_format,
            temperature=temperature,
            top_p=top_p,
            max_concurrency=max_concurrency,
        )

        # Await the async operation
        results = await client.parse(self._obj.tolist(), batch_size=batch_size)

        return pd.Series(
            results,
            index=self._obj.index,
            name=self._obj.name,
        )

    async def embeddings(self, batch_size: int = 128, max_concurrency: int = 8) -> pd.Series:
        """Compute OpenAI embeddings for every Series element (asynchronously).

        Example:
            ```python
            animals = pd.Series(["cat", "dog", "elephant"])
            # Must be awaited
            embeddings = await animals.aio.embeddings()
            ```
            This method returns a Series of numpy arrays, each containing the
            embedding vector for the corresponding input.
            The embedding model is set by the `embeddings_model` function.
            The default embedding model is `text-embedding-3-small`.

        Args:
            batch_size (int, optional): Number of inputs grouped into a
                single request. Defaults to ``128``.
            max_concurrency (int, optional): Maximum number of concurrent
                requests. Defaults to ``8``.

        Returns:
            pandas.Series: Series whose values are ``np.ndarray`` objects
                (dtype ``float32``).

        Note:
            This is an asynchronous method and must be awaited.
        """
        client: AsyncBatchEmbeddings = AsyncBatchEmbeddings(
            client=_DI.resolve(AsyncOpenAI),
            model_name=_DI.resolve(EmbeddingsModelName).value,
            max_concurrency=max_concurrency,
        )

        # Await the async operation
        results = await client.create(self._obj.tolist(), batch_size=batch_size)

        return pd.Series(
            results,
            index=self._obj.index,
            name=self._obj.name,
        )

    async def task(self, task: PreparedTask, batch_size: int = 128, max_concurrency: int = 8) -> pd.Series:
        """Execute a prepared task on every Series element (asynchronously).

        This method applies a pre-configured task to each element in the Series,
        using the task's instructions and response format to generate structured
        responses from the language model.

        Example:
            ```python
            from openaivec.model import PreparedTask

            # Assume you have a prepared task for sentiment analysis
            sentiment_task = PreparedTask(...)

            reviews = pd.Series(["Great product!", "Not satisfied", "Amazing quality"])
            # Must be awaited
            results = await reviews.aio.task(sentiment_task)
            ```
            This method returns a Series containing the task results for each
            corresponding input element, following the task's defined structure.

        Args:
            task (PreparedTask): A pre-configured task containing instructions,
                response format, and other parameters for processing the inputs.
            batch_size (int, optional): Number of prompts grouped into a single
                request to optimize API usage. Defaults to 128.
            max_concurrency (int, optional): Maximum number of concurrent
                requests. Defaults to 8.

        Returns:
            pandas.Series: Series whose values are instances of the task's
                response format, aligned with the original Series index.

        Note:
            This is an asynchronous method and must be awaited.
        """
        client = AsyncBatchResponses.of_task(
            client=_DI.resolve(AsyncOpenAI),
            model_name=_DI.resolve(ResponsesModelName).value,
            task=task,
            max_concurrency=max_concurrency,
        )

        # Await the async operation
        results = await client.parse(self._obj.tolist(), batch_size=batch_size)

        return pd.Series(
            results,
            index=self._obj.index,
            name=self._obj.name,
        )


@pd.api.extensions.register_dataframe_accessor("aio")
class AsyncOpenAIVecDataFrameAccessor:
    """pandas DataFrame accessor (``.aio``) that adds OpenAI helpers."""

    def __init__(self, df_obj: pd.DataFrame):
        self._obj = df_obj

    async def responses(
        self,
        instructions: str,
        response_format: Type[T] = str,
        batch_size: int = 128,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_concurrency: int = 8,
    ) -> pd.Series:
        """Generate a response for each row after serialising it to JSON (asynchronously).

        Example:
            ```python
            df = pd.DataFrame([
                {\"name\": \"cat\", \"legs\": 4},
                {\"name\": \"dog\", \"legs\": 4},
                {\"name\": \"elephant\", \"legs\": 4},
            ])
            # Must be awaited
            results = await df.aio.responses(\"what is the animal\'s name?\")
            ```
            This method returns a Series of strings, each containing the
            assistant's response to the corresponding input.
            Each row is serialised to JSON before being sent to the assistant.
            The model used is set by the `responses_model` function.
            The default model is `gpt-4o-mini`.

        Args:
            instructions (str): System prompt for the assistant.
            response_format (Type[T], optional): Desired Python type of the
                responses. Defaults to ``str``.
            batch_size (int, optional): Number of requests sent in one batch.
                Defaults to ``128``.
            temperature (float, optional): Sampling temperature. Defaults to ``0.0``.
            top_p (float, optional): Nucleus sampling parameter. Defaults to ``1.0``.
            max_concurrency (int, optional): Maximum number of concurrent
                requests. Defaults to ``8``.

        Returns:
            pandas.Series: Responses aligned with the DataFrame’s original index.

        Note:
            This is an asynchronous method and must be awaited.
        """
        series_of_json = self._obj.pipe(
            lambda df: (
                pd.Series(df.to_dict(orient="records"), index=df.index, name="record").map(
                    lambda x: json.dumps(x, ensure_ascii=False)
                )
            )
        )
        # Await the call to the async Series method using .aio
        return await series_of_json.aio.responses(
            instructions=instructions,
            response_format=response_format,
            batch_size=batch_size,
            temperature=temperature,
            top_p=top_p,
            max_concurrency=max_concurrency,
        )

    async def task(self, task: PreparedTask, batch_size: int = 128, max_concurrency: int = 8) -> pd.Series:
        """Execute a prepared task on each DataFrame row after serialising it to JSON (asynchronously).

        This method applies a pre-configured task to each row in the DataFrame,
        using the task's instructions and response format to generate structured
        responses from the language model. Each row is serialised to JSON before
        being processed by the task.

        Example:
            ```python
            from openaivec.model import PreparedTask

            # Assume you have a prepared task for data analysis
            analysis_task = PreparedTask(...)

            df = pd.DataFrame([
                {"name": "cat", "legs": 4},
                {"name": "dog", "legs": 4},
                {"name": "elephant", "legs": 4},
            ])
            # Must be awaited
            results = await df.aio.task(analysis_task)
            ```
            This method returns a Series containing the task results for each
            corresponding row, following the task's defined structure.

        Args:
            task (PreparedTask): A pre-configured task containing instructions,
                response format, and other parameters for processing the inputs.
            batch_size (int, optional): Number of requests sent in one batch
                to optimize API usage. Defaults to 128.
            max_concurrency (int, optional): Maximum number of concurrent
                requests. Defaults to 8.

        Returns:
            pandas.Series: Series whose values are instances of the task's
                response format, aligned with the DataFrame's original index.

        Note:
            This is an asynchronous method and must be awaited.
        """
        series_of_json = self._obj.pipe(
            lambda df: (
                pd.Series(df.to_dict(orient="records"), index=df.index, name="record").map(
                    lambda x: json.dumps(x, ensure_ascii=False)
                )
            )
        )
        # Await the call to the async Series method using .aio
        return await series_of_json.aio.task(
            task=task,
            batch_size=batch_size,
            max_concurrency=max_concurrency,
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

    async def assign(self, **kwargs: Any) -> pd.DataFrame:
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
            **kwargs: Any. Column names as keys and either static values or callables
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
        self, target_column_name: str, max_examples: int = 500, batch_size: int = 128, max_concurrency: int = 8
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
            batch_size (int, optional): Number of requests sent in one batch
                to optimize API usage. Defaults to 128.
            max_concurrency (int, optional): Maximum number of concurrent
                requests. Defaults to 8.

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
            task=task, batch_size=batch_size, max_concurrency=max_concurrency
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
