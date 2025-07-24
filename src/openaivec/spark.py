"""Asynchronous Spark UDFs for the OpenAI and Azure OpenAI APIs.

This module provides builder classes (`ResponsesUDFBuilder`, `EmbeddingsUDFBuilder`)
for creating asynchronous Spark UDFs that communicate with either the public
OpenAI API or Azure OpenAI using the `openaivec.spark` subpackage.
It supports UDFs for generating responses and creating embeddings asynchronously.
The UDFs operate on Spark DataFrames and leverage asyncio for potentially
improved performance in I/O-bound operations.

## Setup

First, obtain a Spark session:

```python
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()
```

Next, instantiate UDF builders with your OpenAI API key (or Azure credentials)
and model/deployment names, then register the desired UDFs:

```python
import os
from openaivec.spark import ResponsesUDFBuilder, EmbeddingsUDFBuilder
from pydantic import BaseModel

# Option 1: Using OpenAI
resp_builder = ResponsesUDFBuilder.of_openai(
    api_key=os.getenv("OPENAI_API_KEY"),
    model_name="gpt-4o-mini", # Model for responses
)
emb_builder = EmbeddingsUDFBuilder.of_openai(
    api_key=os.getenv("OPENAI_API_KEY"),
    model_name="text-embedding-3-small", # Model for embeddings
)

# Option 2: Using Azure OpenAI
# resp_builder = ResponsesUDFBuilder.of_azure_openai(
#     api_key=os.getenv("AZURE_OPENAI_KEY"),
#     endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
#     api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
#     model_name="your-resp-deployment-name", # Deployment for responses
# )
# emb_builder = EmbeddingsUDFBuilder.of_azure_openai(
#     api_key=os.getenv("AZURE_OPENAI_KEY"),
#     endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
#     api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
#     model_name="your-emb-deployment-name", # Deployment for embeddings
# )

# Define a Pydantic model for structured responses (optional)
class Translation(BaseModel):
    en: str
    fr: str
    # ... other languages

# Register the asynchronous responses UDF
spark.udf.register(
    "translate_async",
    resp_builder.build(
        instructions="Translate the text to multiple languages.",
        response_format=Translation,
    ),
)

# Or use a predefined task with build_from_task method
from openaivec.task import nlp
spark.udf.register(
    "sentiment_async",
    resp_builder.build_from_task(nlp.SENTIMENT_ANALYSIS),
)

# Register the asynchronous embeddings UDF
spark.udf.register(
    "embed_async",
    emb_builder.build(),
)
```

You can now invoke the UDFs from Spark SQL:

```sql
SELECT
    text,
    translate_async(text) AS translation,
    sentiment_async(text) AS sentiment,
    embed_async(text) AS embedding
FROM your_table;
```

Note: This module provides asynchronous support through the pandas extensions.
"""

import asyncio
from dataclasses import dataclass
from typing import Dict, Iterator, List, Type, TypeVar, Union, get_args, get_origin, Optional
from typing_extensions import Literal
from enum import Enum
import logging
from pyspark.sql.pandas.functions import pandas_udf
from pyspark.sql.udf import UserDefinedFunction
from pyspark.sql.types import BooleanType, IntegerType, StringType, ArrayType, FloatType, StructField, StructType
from openai import AsyncOpenAI, AsyncAzureOpenAI
import tiktoken
from . import pandas_ext
import pandas as pd
from pydantic import BaseModel

from .serialize import deserialize_base_model, serialize_base_model
from .util import TextChunker
from .task.model import PreparedTask

__all__ = [
    "ResponsesUDFBuilder",
    "EmbeddingsUDFBuilder",
    "split_to_chunks_udf",
    "count_tokens_udf",
    "similarity_udf",
]

ResponseFormat = BaseModel | Type[str]
T = TypeVar("T", bound=BaseModel)

_INITIALIZED: bool = False
_LOGGER: logging.Logger = logging.getLogger(__name__)
_TIKTOKEN_ENC: tiktoken.Encoding | None = None


def _initialize(api_key: str, endpoint: str | None, api_version: str | None) -> None:
    """Initializes the OpenAI client for asynchronous operations.

    This function sets up the global asynchronous OpenAI client instance
    (either `AsyncOpenAI` or `AsyncAzureOpenAI`) used by the UDFs in this
    module. It ensures the client is initialized only once.

    Args:
        api_key: The OpenAI or Azure OpenAI API key.
        endpoint: The Azure OpenAI endpoint URL. Required for Azure.
        api_version: The Azure OpenAI API version. Required for Azure.
    """
    global _INITIALIZED
    if not _INITIALIZED:
        if endpoint and api_version:
            pandas_ext.use_async(AsyncAzureOpenAI(api_key=api_key, azure_endpoint=endpoint, api_version=api_version))
        else:
            pandas_ext.use_async(AsyncOpenAI(api_key=api_key))
        _INITIALIZED = True


def _python_type_to_spark(python_type):
    origin = get_origin(python_type)

    # For list types (e.g., List[int])
    if origin is list or origin is List:
        # Retrieve the inner type and recursively convert it
        inner_type = get_args(python_type)[0]
        return ArrayType(_python_type_to_spark(inner_type))

    # For Optional types (Union[..., None])
    elif origin is Union:
        non_none_args = [arg for arg in get_args(python_type) if arg is not type(None)]
        if len(non_none_args) == 1:
            return _python_type_to_spark(non_none_args[0])
        else:
            raise ValueError(f"Unsupported Union type with multiple non-None types: {python_type}")

    # For Literal types - treat as StringType since Spark doesn't have enum types
    elif origin is Literal:
        return StringType()

    # For Enum types - also treat as StringType since Spark doesn't have enum types
    elif hasattr(python_type, '__bases__') and Enum in python_type.__bases__:
        return StringType()

    # For nested Pydantic models (to be treated as Structs)
    elif isinstance(python_type, type) and issubclass(python_type, BaseModel):
        return _pydantic_to_spark_schema(python_type)

    # Basic type mapping
    elif python_type is int:
        return IntegerType()
    elif python_type is float:
        return FloatType()
    elif python_type is str:
        return StringType()
    elif python_type is bool:
        return BooleanType()
    else:
        raise ValueError(f"Unsupported type: {python_type}")


def _pydantic_to_spark_schema(model: Type[BaseModel]) -> StructType:
    fields = []
    for field_name, field in model.model_fields.items():
        field_type = field.annotation
        # Use outer_type_ to correctly handle types like Optional
        spark_type = _python_type_to_spark(field_type)
        # Set nullable to True (adjust logic as needed)
        fields.append(StructField(field_name, spark_type, nullable=True))
    return StructType(fields)


def _safe_cast_str(x: Optional[str]) -> Optional[str]:
    try:
        if x is None:
            return None

        return str(x)
    except Exception as e:
        _LOGGER.info(f"Error during casting to str: {e}")
        return None


def _safe_dump(x: Optional[BaseModel]) -> Dict:
    try:
        if x is None:
            return {}

        return x.model_dump()
    except Exception as e:
        _LOGGER.info(f"Error during model_dump: {e}")
        return {}


@dataclass(frozen=True)
class ResponsesUDFBuilder:
    """Builder for asynchronous Spark pandas UDFs for generating responses.

    Configures and builds UDFs that leverage `pandas_ext.aio.responses`
    to generate text or structured responses from OpenAI models asynchronously.
    An instance stores authentication parameters and the model name.

    This builder supports two main methods:
    - `build()`: Creates UDFs with custom instructions and response formats
    - `build_from_task()`: Creates UDFs from predefined tasks (e.g., sentiment analysis)

    Attributes:
        api_key (str): OpenAI or Azure API key.
        endpoint (Optional[str]): Azure endpoint base URL. None for public OpenAI.
        api_version (Optional[str]): Azure API version. Ignored for public OpenAI.
        model_name (str): Deployment name (Azure) or model name (OpenAI) for responses.
    """

    # Params for OpenAI SDK
    api_key: str
    endpoint: str | None
    api_version: str | None

    # Params for Responses API
    model_name: str

    @classmethod
    def of_openai(cls, api_key: str, model_name: str) -> "ResponsesUDFBuilder":
        """Creates a builder configured for the public OpenAI API.

        Args:
            api_key (str): The OpenAI API key.
            model_name (str): The OpenAI model name for responses (e.g., "gpt-4o-mini").

        Returns:
            ResponsesUDFBuilder: A builder instance configured for OpenAI responses.
        """
        return cls(api_key=api_key, endpoint=None, api_version=None, model_name=model_name)

    @classmethod
    def of_azure_openai(cls, api_key: str, endpoint: str, api_version: str, model_name: str) -> "ResponsesUDFBuilder":
        """Creates a builder configured for Azure OpenAI.

        Args:
            api_key (str): The Azure OpenAI API key.
            endpoint (str): The Azure OpenAI endpoint URL.
            api_version (str): The Azure OpenAI API version (e.g., "2024-02-01").
            model_name (str): The Azure OpenAI deployment name for responses.

        Returns:
            ResponsesUDFBuilder: A builder instance configured for Azure OpenAI responses.
        """
        return cls(api_key=api_key, endpoint=endpoint, api_version=api_version, model_name=model_name)

    def build(
        self,
        instructions: str,
        response_format: Type[T] = str,
        batch_size: int = 128,  # Default batch size for async might differ
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_concurrency: int = 8,
    ) -> UserDefinedFunction:
        """Builds the asynchronous pandas UDF for generating responses.

        Args:
            instructions (str): The system prompt or instructions for the model.
            response_format (Type[T]): The desired output format. Either `str` for plain text
                or a Pydantic `BaseModel` for structured JSON output. Defaults to `str`.
            batch_size (int): Number of rows per async batch request passed to the underlying
                `pandas_ext` function. Defaults to 128.
            temperature (float): Sampling temperature (0.0 to 2.0). Defaults to 0.0.
            top_p (float): Nucleus sampling parameter. Defaults to 1.0.

        Returns:
            UserDefinedFunction: A Spark pandas UDF configured to generate responses asynchronously.
                Output schema is `StringType` or a struct derived from `response_format`.

        Raises:
            ValueError: If `response_format` is not `str` or a Pydantic `BaseModel`.
        """
        if issubclass(response_format, BaseModel):
            spark_schema = _pydantic_to_spark_schema(response_format)
            json_schema_string = serialize_base_model(response_format)

            @pandas_udf(returnType=spark_schema)
            def structure_udf(col: Iterator[pd.Series]) -> Iterator[pd.DataFrame]:
                _initialize(self.api_key, self.endpoint, self.api_version)
                pandas_ext.responses_model(self.model_name)

                for part in col:
                    predictions: pd.Series = asyncio.run(
                        part.aio.responses(
                            instructions=instructions,
                            response_format=deserialize_base_model(json_schema_string),
                            batch_size=batch_size,
                            temperature=temperature,
                            top_p=top_p,
                            max_concurrency=max_concurrency,
                        )
                    )
                    yield pd.DataFrame(predictions.map(_safe_dump).tolist())

            return structure_udf

        elif issubclass(response_format, str):

            @pandas_udf(returnType=StringType())
            def string_udf(col: Iterator[pd.Series]) -> Iterator[pd.Series]:
                _initialize(self.api_key, self.endpoint, self.api_version)
                pandas_ext.responses_model(self.model_name)

                for part in col:
                    predictions: pd.Series = asyncio.run(
                        part.aio.responses(
                            instructions=instructions,
                            response_format=str,
                            batch_size=batch_size,
                            temperature=temperature,
                            top_p=top_p,
                            max_concurrency=max_concurrency,
                        )
                    )
                    yield predictions.map(_safe_cast_str)

            return string_udf

        else:
            raise ValueError(f"Unsupported response_format: {response_format}")

    def build_from_task(
        self,
        task: PreparedTask,
        batch_size: int = 128,
        max_concurrency: int = 8,
    ) -> UserDefinedFunction:
        """Builds the asynchronous pandas UDF from a predefined task.

        This method allows users to create UDFs from predefined tasks such as sentiment analysis,
        translation, or other common NLP operations defined in the openaivec.task module.

        Args:
            task (PreparedTask): A predefined task configuration containing instructions,
                response format, temperature, and top_p settings.
            batch_size (int): Number of rows per async batch request passed to the underlying
                `pandas_ext` function. Defaults to 128.
            max_concurrency (int): Maximum number of concurrent requests. Defaults to 8.

        Returns:
            UserDefinedFunction: A Spark pandas UDF configured to execute the specified task
                asynchronously, returning a struct derived from the task's response format.

        Example:
            ```python
            from openaivec.task import nlp
            
            builder = ResponsesUDFBuilder.of_openai(
                api_key="your-api-key",
                model_name="gpt-4o-mini"
            )
            
            sentiment_udf = builder.build_from_task(nlp.SENTIMENT_ANALYSIS)
            
            spark.udf.register("analyze_sentiment", sentiment_udf)
            ```
        """
        # Serialize task parameters for Spark serialization compatibility
        task_instructions = task.instructions
        task_response_format_json = serialize_base_model(task.response_format)
        task_temperature = task.temperature
        task_top_p = task.top_p

        # Deserialize the response format from JSON
        response_format = deserialize_base_model(task_response_format_json)
        spark_schema = _pydantic_to_spark_schema(response_format)

        @pandas_udf(returnType=spark_schema)
        def task_udf(col: Iterator[pd.Series]) -> Iterator[pd.DataFrame]:
            _initialize(self.api_key, self.endpoint, self.api_version)
            pandas_ext.responses_model(self.model_name)

            for part in col:
                predictions: pd.Series = asyncio.run(
                    part.aio.responses(
                        instructions=task_instructions,
                        response_format=response_format,
                        batch_size=batch_size,
                        temperature=task_temperature,
                        top_p=task_top_p,
                        max_concurrency=max_concurrency,
                    )
                )
                yield pd.DataFrame(predictions.map(_safe_dump).tolist())

        return task_udf


@dataclass(frozen=True)
class EmbeddingsUDFBuilder:
    """Builder for asynchronous Spark pandas UDFs for creating embeddings.

    Configures and builds UDFs that leverage `pandas_ext.aio.embeddings`
    to generate vector embeddings from OpenAI models asynchronously.
    An instance stores authentication parameters and the model name.

    Attributes:
        api_key (str): OpenAI or Azure API key.
        endpoint (Optional[str]): Azure endpoint base URL. None for public OpenAI.
        api_version (Optional[str]): Azure API version. Ignored for public OpenAI.
        model_name (str): Deployment name (Azure) or model name (OpenAI) for embeddings.
    """

    # Params for OpenAI SDK
    api_key: str
    endpoint: str | None
    api_version: str | None

    # Params for Embeddings API
    model_name: str

    @classmethod
    def of_openai(cls, api_key: str, model_name: str) -> "EmbeddingsUDFBuilder":
        """Creates a builder configured for the public OpenAI API.

        Args:
            api_key (str): The OpenAI API key.
            model_name (str): The OpenAI model name for embeddings (e.g., "text-embedding-3-small").

        Returns:
            EmbeddingsUDFBuilder: A builder instance configured for OpenAI embeddings.
        """
        return cls(api_key=api_key, endpoint=None, api_version=None, model_name=model_name)

    @classmethod
    def of_azure_openai(cls, api_key: str, endpoint: str, api_version: str, model_name: str) -> "EmbeddingsUDFBuilder":
        """Creates a builder configured for Azure OpenAI.

        Args:
            api_key (str): The Azure OpenAI API key.
            endpoint (str): The Azure OpenAI endpoint URL.
            api_version (str): The Azure OpenAI API version (e.g., "2024-02-01").
            model_name (str): The Azure OpenAI deployment name for embeddings.

        Returns:
            EmbeddingsUDFBuilder: A builder instance configured for Azure OpenAI embeddings.
        """
        return cls(api_key=api_key, endpoint=endpoint, api_version=api_version, model_name=model_name)

    def build(self, batch_size: int = 128, max_concurrency: int = 8) -> UserDefinedFunction:
        """Builds the asynchronous pandas UDF for generating embeddings.

        Args:
            batch_size (int): Number of rows per async batch request passed to the underlying
                `pandas_ext` function. Defaults to 128.

        Returns:
            UserDefinedFunction: A Spark pandas UDF configured to generate embeddings asynchronously,
                returning an `ArrayType(FloatType())` column.
        """

        @pandas_udf(returnType=ArrayType(FloatType()))
        def embeddings_udf(col: Iterator[pd.Series]) -> Iterator[pd.Series]:
            _initialize(self.api_key, self.endpoint, self.api_version)
            pandas_ext.embeddings_model(self.model_name)

            for part in col:
                embeddings: pd.Series = asyncio.run(
                    part.aio.embeddings(batch_size=batch_size, max_concurrency=max_concurrency)
                )
                yield embeddings.map(lambda x: x.tolist())

        return embeddings_udf



def split_to_chunks_udf(model_name: str, max_tokens: int, sep: List[str]) -> UserDefinedFunction:
    """Create a pandas‑UDF that splits text into token‑bounded chunks.

    Args:
        model_name: Model identifier passed to *tiktoken*.
        max_tokens: Maximum tokens allowed per chunk.
        sep: Ordered list of separator strings used by ``TextChunker``.

    Returns:
        A pandas UDF producing an ``ArrayType(StringType())`` column whose
            values are lists of chunks respecting the ``max_tokens`` limit.
    """

    @pandas_udf(ArrayType(StringType()))
    def fn(col: Iterator[pd.Series]) -> Iterator[pd.Series]:
        global _TIKTOKEN_ENC
        if _TIKTOKEN_ENC is None:
            _TIKTOKEN_ENC = tiktoken.encoding_for_model(model_name)

        chunker = TextChunker(_TIKTOKEN_ENC)

        for part in col:
            yield part.map(lambda x: chunker.split(x, max_tokens=max_tokens, sep=sep) if isinstance(x, str) else [])

    return fn


def count_tokens_udf(model_name: str = "gpt-4o") -> UserDefinedFunction:
    """Create a pandas‑UDF that counts tokens for every string cell.

    The UDF uses *tiktoken* to approximate tokenisation and caches the
    resulting ``Encoding`` object per executor.

    Args:
        model_name: Model identifier understood by ``tiktoken``.

    Returns:
        A pandas UDF producing an ``IntegerType`` column with token counts.
    """

    @pandas_udf(IntegerType())
    def fn(col: Iterator[pd.Series]) -> Iterator[pd.Series]:
        global _TIKTOKEN_ENC
        if _TIKTOKEN_ENC is None:
            _TIKTOKEN_ENC = tiktoken.encoding_for_model(model_name)

        for part in col:
            yield part.map(lambda x: len(_TIKTOKEN_ENC.encode(x)) if isinstance(x, str) else 0)

    return fn


def similarity_udf() -> UserDefinedFunction:
    @pandas_udf(FloatType())
    def fn(a: pd.Series, b: pd.Series) -> pd.Series:
        """Compute cosine similarity between two vectors.

        Args:
            a: First vector.
            b: Second vector.

        Returns:
            Cosine similarity between the two vectors.
        """
        pandas_ext._wakeup()
        return pd.DataFrame({"a": a, "b": b}).ai.similarity("a", "b")

    return fn
