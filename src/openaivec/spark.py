"""Asynchronous Spark UDFs for the OpenAI and Azure OpenAI APIs.

This module provides functions (`responses_udf`, `task_udf`, `embeddings_udf`,
`count_tokens_udf`, `split_to_chunks_udf`, `similarity_udf`, `parse_udf`)
for creating asynchronous Spark UDFs that communicate with either the public
OpenAI API or Azure OpenAI using the `openaivec.spark` subpackage.
It supports UDFs for generating responses, creating embeddings, parsing text,
and computing similarities asynchronously. The UDFs operate on Spark DataFrames
and leverage asyncio for improved performance in I/O-bound operations.

**Performance Optimization**: All AI-powered UDFs (`responses_udf`, `task_udf`, `embeddings_udf`, `parse_udf`)
automatically cache duplicate inputs within each partition, significantly reducing
API calls and costs when processing datasets with overlapping content.


## Setup

First, obtain a Spark session and configure authentication:

```python
from pyspark.sql import SparkSession
from openaivec.spark import setup, setup_azure

spark = SparkSession.builder.getOrCreate()

# Option 1: Using OpenAI
setup(
    spark,
    api_key="your-openai-api-key",
    responses_model_name="gpt-4.1-mini",  # Optional: set default model
    embeddings_model_name="text-embedding-3-small"  # Optional: set default model
)

# Option 2: Using Azure OpenAI
# setup_azure(
#     spark,
#     api_key="your-azure-openai-api-key",
#     base_url="https://YOUR-RESOURCE-NAME.services.ai.azure.com/openai/v1/",
#     api_version="preview",
#     responses_model_name="my-gpt4-deployment",  # Optional: set default deployment
#     embeddings_model_name="my-embedding-deployment"  # Optional: set default deployment
# )
```

Next, create UDFs and register them:

```python
from openaivec.spark import responses_udf, task_udf, embeddings_udf, count_tokens_udf, split_to_chunks_udf
from pydantic import BaseModel

# Define a Pydantic model for structured responses (optional)
class Translation(BaseModel):
    en: str
    fr: str
    # ... other languages

# Register the asynchronous responses UDF with performance tuning
spark.udf.register(
    "translate_async",
    responses_udf(
        instructions="Translate the text to multiple languages.",
        response_format=Translation,
        model_name="gpt-4.1-mini",  # For Azure: deployment name, for OpenAI: model name
        batch_size=64,              # Rows per API request within partition
        max_concurrency=8           # Concurrent requests PER EXECUTOR
    ),
)

# Or use a predefined task with task_udf
from openaivec.task import nlp
spark.udf.register(
    "sentiment_async",
    task_udf(nlp.SENTIMENT_ANALYSIS),
)

# Register the asynchronous embeddings UDF with performance tuning
spark.udf.register(
    "embed_async",
    embeddings_udf(
        model_name="text-embedding-3-small",  # For Azure: deployment name, for OpenAI: model name
        batch_size=128,                       # Larger batches for embeddings
        max_concurrency=8                     # Concurrent requests PER EXECUTOR
    ),
)

# Register token counting, text chunking, and similarity UDFs
spark.udf.register("count_tokens", count_tokens_udf())
spark.udf.register("split_chunks", split_to_chunks_udf(max_tokens=512, sep=[".", "!", "?"]))
spark.udf.register("compute_similarity", similarity_udf())
```

You can now invoke the UDFs from Spark SQL:

```sql
SELECT
    text,
    translate_async(text) AS translation,
    sentiment_async(text) AS sentiment,
    embed_async(text) AS embedding,
    count_tokens(text) AS token_count,
    split_chunks(text) AS chunks,
    compute_similarity(embed_async(text1), embed_async(text2)) AS similarity
FROM your_table;
```

## Performance Considerations

When using these UDFs in distributed Spark environments:

- **`batch_size`**: Controls rows processed per API request within each partition.
  Recommended: 32-128 for responses, 64-256 for embeddings.

- **`max_concurrency`**: Sets concurrent API requests **PER EXECUTOR**, not per cluster.
  Total cluster concurrency = max_concurrency × number_of_executors.
  Recommended: 4-12 per executor to avoid overwhelming OpenAI rate limits.

- **Rate Limit Management**: Monitor OpenAI API usage when scaling executors.
  Consider your OpenAI tier limits and adjust max_concurrency accordingly.

Example for a 5-executor cluster with max_concurrency=8:
Total concurrent requests = 8 × 5 = 40 simultaneous API calls.

Note: This module provides asynchronous support through the pandas extensions.
"""

import asyncio
import logging
import os
from collections.abc import Iterator
from enum import Enum
from typing import Union, get_args, get_origin

import numpy as np
import pandas as pd
import tiktoken
from pydantic import BaseModel
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.pandas.functions import pandas_udf
from pyspark.sql.types import ArrayType, BooleanType, FloatType, IntegerType, StringType, StructField, StructType
from pyspark.sql.udf import UserDefinedFunction
from typing_extensions import Literal

from openaivec import pandas_ext
from openaivec._model import EmbeddingsModelName, PreparedTask, ResponseFormat, ResponsesModelName
from openaivec._provider import CONTAINER
from openaivec._proxy import AsyncBatchingMapProxy
from openaivec._schema import InferredSchema, SchemaInferenceInput, SchemaInferer
from openaivec._serialize import deserialize_base_model, serialize_base_model
from openaivec._util import TextChunker

__all__ = [
    "responses_udf",
    "task_udf",
    "embeddings_udf",
    "infer_schema",
    "parse_udf",
    "split_to_chunks_udf",
    "count_tokens_udf",
    "similarity_udf",
]


_LOGGER: logging.Logger = logging.getLogger(__name__)


def setup(
    spark: SparkSession, api_key: str, responses_model_name: str | None = None, embeddings_model_name: str | None = None
):
    """Setup OpenAI authentication and default model names in Spark environment.
    1. Configures OpenAI API key in SparkContext environment.
    2. Configures OpenAI API key in local process environment.
    3. Optionally registers default model names for responses and embeddings in the DI container.

    Args:
        spark (SparkSession): The Spark session to configure.
        api_key (str): OpenAI API key for authentication.
        responses_model_name (str | None): Default model name for response generation.
            If provided, registers `ResponsesModelName` in the DI container.
        embeddings_model_name (str | None): Default model name for embeddings.
            If provided, registers `EmbeddingsModelName` in the DI container.
    """

    CONTAINER.register(SparkSession, lambda: spark)
    CONTAINER.register(SparkContext, lambda: CONTAINER.resolve(SparkSession).sparkContext)

    sc = CONTAINER.resolve(SparkContext)
    sc.environment["OPENAI_API_KEY"] = api_key

    os.environ["OPENAI_API_KEY"] = api_key

    if responses_model_name:
        CONTAINER.register(ResponsesModelName, lambda: ResponsesModelName(responses_model_name))

    if embeddings_model_name:
        CONTAINER.register(EmbeddingsModelName, lambda: EmbeddingsModelName(embeddings_model_name))

    CONTAINER.clear_singletons()


def setup_azure(
    spark: SparkSession,
    api_key: str,
    base_url: str,
    api_version: str = "preview",
    responses_model_name: str | None = None,
    embeddings_model_name: str | None = None,
):
    """Setup Azure OpenAI authentication and default model names in Spark environment.
    1. Configures Azure OpenAI API key, base URL, and API version in SparkContext environment.
    2. Configures Azure OpenAI API key, base URL, and API version in local process environment.
    3. Optionally registers default model names for responses and embeddings in the DI container.
    Args:
        spark (SparkSession): The Spark session to configure.
        api_key (str): Azure OpenAI API key for authentication.
        base_url (str): Base URL for the Azure OpenAI resource.
        api_version (str): API version to use. Defaults to "preview".
        responses_model_name (str | None): Default model name for response generation.
            If provided, registers `ResponsesModelName` in the DI container.
        embeddings_model_name (str | None): Default model name for embeddings.
            If provided, registers `EmbeddingsModelName` in the DI container.
    """

    CONTAINER.register(SparkSession, lambda: spark)
    CONTAINER.register(SparkContext, lambda: CONTAINER.resolve(SparkSession).sparkContext)

    sc = CONTAINER.resolve(SparkContext)
    sc.environment["AZURE_OPENAI_API_KEY"] = api_key
    sc.environment["AZURE_OPENAI_BASE_URL"] = base_url
    sc.environment["AZURE_OPENAI_API_VERSION"] = api_version

    os.environ["AZURE_OPENAI_API_KEY"] = api_key
    os.environ["AZURE_OPENAI_BASE_URL"] = base_url
    os.environ["AZURE_OPENAI_API_VERSION"] = api_version

    if responses_model_name:
        CONTAINER.register(ResponsesModelName, lambda: ResponsesModelName(responses_model_name))

    if embeddings_model_name:
        CONTAINER.register(EmbeddingsModelName, lambda: EmbeddingsModelName(embeddings_model_name))

    CONTAINER.clear_singletons()


def set_responses_model(model_name: str):
    """Set the default model name for response generation in the DI container.

    Args:
        model_name (str): The model name to set as default for responses.
    """
    CONTAINER.register(ResponsesModelName, lambda: ResponsesModelName(model_name))
    CONTAINER.clear_singletons()


def get_responses_model() -> str | None:
    """Get the default model name for response generation from the DI container.

    Returns:
        str | None: The default model name for responses, or None if not set.
    """
    try:
        return CONTAINER.resolve(ResponsesModelName).value
    except Exception:
        return None


def set_embeddings_model(model_name: str):
    """Set the default model name for embeddings in the DI container.

    Args:
        model_name (str): The model name to set as default for embeddings.
    """
    CONTAINER.register(EmbeddingsModelName, lambda: EmbeddingsModelName(model_name))
    CONTAINER.clear_singletons()


def get_embeddings_model() -> str | None:
    """Get the default model name for embeddings from the DI container.

    Returns:
        str | None: The default model name for embeddings, or None if not set.
    """
    try:
        return CONTAINER.resolve(EmbeddingsModelName).value
    except Exception:
        return None


def _python_type_to_spark(python_type):
    origin = get_origin(python_type)

    # For list types (e.g., list[int])
    if origin is list:
        # Retrieve the inner type and recursively convert it
        inner_type = get_args(python_type)[0]
        return ArrayType(_python_type_to_spark(inner_type))

    # For Optional types (T | None via Union internally)
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
    elif hasattr(python_type, "__bases__") and Enum in python_type.__bases__:
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


def _pydantic_to_spark_schema(model: type[BaseModel]) -> StructType:
    fields = []
    for field_name, field in model.model_fields.items():
        field_type = field.annotation
        # Use outer_type_ to correctly handle types like Optional
        spark_type = _python_type_to_spark(field_type)
        # Set nullable to True (adjust logic as needed)
        fields.append(StructField(field_name, spark_type, nullable=True))
    return StructType(fields)


def _safe_cast_str(x: str | None) -> str | None:
    try:
        if x is None:
            return None

        return str(x)
    except Exception as e:
        _LOGGER.info(f"Error during casting to str: {e}")
        return None


def _safe_dump(x: BaseModel | None) -> dict:
    try:
        if x is None:
            return {}

        return x.model_dump()
    except Exception as e:
        _LOGGER.info(f"Error during model_dump: {e}")
        return {}


def responses_udf(
    instructions: str,
    response_format: type[ResponseFormat] = str,
    model_name: str | None = None,
    batch_size: int | None = None,
    max_concurrency: int = 8,
    **api_kwargs,
) -> UserDefinedFunction:
    """Create an asynchronous Spark pandas UDF for generating responses.

    Configures and builds UDFs that leverage `pandas_ext.aio.responses_with_cache`
    to generate text or structured responses from OpenAI models asynchronously.
    Each partition maintains its own cache to eliminate duplicate API calls within
    the partition, significantly reducing API usage and costs when processing
    datasets with overlapping content.

    Note:
        Authentication must be configured via SparkContext environment variables.
        Set the appropriate environment variables on the SparkContext:

        For OpenAI:
            sc.environment["OPENAI_API_KEY"] = "your-openai-api-key"

        For Azure OpenAI:
            sc.environment["AZURE_OPENAI_API_KEY"] = "your-azure-openai-api-key"
            sc.environment["AZURE_OPENAI_BASE_URL"] = "https://YOUR-RESOURCE-NAME.services.ai.azure.com/openai/v1/"
            sc.environment["AZURE_OPENAI_API_VERSION"] = "preview"

    Args:
        instructions (str): The system prompt or instructions for the model.
        response_format (type[ResponseFormat]): The desired output format. Either `str` for plain text
            or a Pydantic `BaseModel` for structured JSON output. Defaults to `str`.
        model_name (str | None): For Azure OpenAI, use your deployment name (e.g., "my-gpt4-deployment").
            For OpenAI, use the model name (e.g., "gpt-4.1-mini"). Defaults to configured model in DI container
            via ResponsesModelName if not provided.
        batch_size (int | None): Number of rows per async batch request within each partition.
            Larger values reduce API call overhead but increase memory usage.
            Defaults to None (automatic batch size optimization that dynamically
            adjusts based on execution time, targeting 30-60 seconds per batch).
            Set to a positive integer (e.g., 32-128) for fixed batch size.
        max_concurrency (int): Maximum number of concurrent API requests **PER EXECUTOR**.
            Total cluster concurrency = max_concurrency × number_of_executors.
            Higher values increase throughput but may hit OpenAI rate limits.
            Recommended: 4-12 per executor. Defaults to 8.
        **api_kwargs: Additional OpenAI API parameters (e.g. ``temperature``, ``top_p``,
            ``frequency_penalty``, ``presence_penalty``, ``seed``, ``max_output_tokens``, etc.)
            forwarded verbatim to the underlying API calls. These parameters are applied to
            all API requests made by the UDF.

    Returns:
        UserDefinedFunction: A Spark pandas UDF configured to generate responses asynchronously.
            Output schema is `StringType` or a struct derived from `response_format`.

    Raises:
        ValueError: If `response_format` is not `str` or a Pydantic `BaseModel`.

    Note:
        For optimal performance in distributed environments:
        - **Automatic Caching**: Duplicate inputs within each partition are cached,
          reducing API calls and costs significantly on datasets with repeated content
        - Monitor OpenAI API rate limits when scaling executor count
        - Consider your OpenAI tier limits: total_requests = max_concurrency × executors
        - Use Spark UI to optimize partition sizes relative to batch_size
    """
    _model_name = model_name or CONTAINER.resolve(ResponsesModelName).value

    if issubclass(response_format, BaseModel):
        spark_schema = _pydantic_to_spark_schema(response_format)
        json_schema_string = serialize_base_model(response_format)

        @pandas_udf(returnType=spark_schema)  # type: ignore[call-overload]
        def structure_udf(col: Iterator[pd.Series]) -> Iterator[pd.DataFrame]:
            pandas_ext.responses_model(_model_name)
            response_format = deserialize_base_model(json_schema_string)
            cache = AsyncBatchingMapProxy[str, response_format](
                batch_size=batch_size,
                max_concurrency=max_concurrency,
            )

            try:
                for part in col:
                    predictions: pd.Series = asyncio.run(
                        part.aio.responses_with_cache(
                            instructions=instructions,
                            response_format=response_format,
                            cache=cache,
                            **api_kwargs,
                        )
                    )
                    yield pd.DataFrame(predictions.map(_safe_dump).tolist())
            finally:
                asyncio.run(cache.clear())

        return structure_udf  # type: ignore[return-value]

    elif issubclass(response_format, str):

        @pandas_udf(returnType=StringType())  # type: ignore[call-overload]
        def string_udf(col: Iterator[pd.Series]) -> Iterator[pd.Series]:
            pandas_ext.responses_model(_model_name)
            cache = AsyncBatchingMapProxy[str, str](
                batch_size=batch_size,
                max_concurrency=max_concurrency,
            )

            try:
                for part in col:
                    predictions: pd.Series = asyncio.run(
                        part.aio.responses_with_cache(
                            instructions=instructions,
                            response_format=str,
                            cache=cache,
                            **api_kwargs,
                        )
                    )
                    yield predictions.map(_safe_cast_str)
            finally:
                asyncio.run(cache.clear())

        return string_udf  # type: ignore[return-value]

    else:
        raise ValueError(f"Unsupported response_format: {response_format}")


def task_udf(
    task: PreparedTask[ResponseFormat],
    model_name: str | None = None,
    batch_size: int | None = None,
    max_concurrency: int = 8,
    **api_kwargs,
) -> UserDefinedFunction:
    """Create an asynchronous Spark pandas UDF from a predefined task.

    This function allows users to create UDFs from predefined tasks such as sentiment analysis,
    translation, or other common NLP operations defined in the openaivec.task module.
    Each partition maintains its own cache to eliminate duplicate API calls within
    the partition, significantly reducing API usage and costs when processing
    datasets with overlapping content.

    Args:
        task (PreparedTask): A predefined task configuration containing instructions,
            response format, and API parameters.
        model_name (str | None): For Azure OpenAI, use your deployment name (e.g., "my-gpt4-deployment").
            For OpenAI, use the model name (e.g., "gpt-4.1-mini"). Defaults to configured model in DI container
            via ResponsesModelName if not provided.
        batch_size (int | None): Number of rows per async batch request within each partition.
            Larger values reduce API call overhead but increase memory usage.
            Defaults to None (automatic batch size optimization that dynamically
            adjusts based on execution time, targeting 30-60 seconds per batch).
            Set to a positive integer (e.g., 32-128) for fixed batch size.
        max_concurrency (int): Maximum number of concurrent API requests **PER EXECUTOR**.
            Total cluster concurrency = max_concurrency × number_of_executors.
            Higher values increase throughput but may hit OpenAI rate limits.
            Recommended: 4-12 per executor. Defaults to 8.

    Additional Keyword Args:
        Arbitrary OpenAI Responses API parameters (e.g. ``temperature``, ``top_p``,
        ``frequency_penalty``, ``presence_penalty``, ``seed``, ``max_output_tokens``, etc.)
        are forwarded verbatim to the underlying API calls. These parameters are applied to
        all API requests made by the UDF and override any parameters set in the task configuration.

    Returns:
        UserDefinedFunction: A Spark pandas UDF configured to execute the specified task
            asynchronously with automatic caching for duplicate inputs within each partition.
            Output schema is StringType for str response format or a struct derived from
            the task's response format for BaseModel.

    Example:
        ```python
        from openaivec.task import nlp

        sentiment_udf = task_udf(nlp.SENTIMENT_ANALYSIS)

        spark.udf.register("analyze_sentiment", sentiment_udf)
        ```

    Note:
        **Automatic Caching**: Duplicate inputs within each partition are cached,
        reducing API calls and costs significantly on datasets with repeated content.
    """
    # Merge task's api_kwargs with caller's api_kwargs (caller takes precedence)
    merged_kwargs = {**task.api_kwargs, **api_kwargs}

    return responses_udf(
        instructions=task.instructions,
        response_format=task.response_format,
        model_name=model_name,
        batch_size=batch_size,
        max_concurrency=max_concurrency,
        **merged_kwargs,
    )


def infer_schema(
    instructions: str,
    example_table_name: str,
    example_field_name: str,
    max_examples: int = 100,
) -> InferredSchema:
    """Infer the schema for a response format based on example data.

    This function retrieves examples from a Spark table and infers the schema
    for the response format using the provided instructions. It is useful when
    you want to dynamically generate a schema based on existing data.

    Args:
        instructions (str): Instructions for the model to infer the schema.
        example_table_name (str | None): Name of the Spark table containing example data.
        example_field_name (str | None): Name of the field in the table to use as examples.
        max_examples (int): Maximum number of examples to retrieve for schema inference.

    Returns:
        InferredSchema: An object containing the inferred schema and response format.
    """

    spark = CONTAINER.resolve(SparkSession)
    examples: list[str] = (
        spark.table(example_table_name).rdd.map(lambda row: row[example_field_name]).takeSample(False, max_examples)
    )

    input = SchemaInferenceInput(
        instructions=instructions,
        examples=examples,
    )
    inferer = CONTAINER.resolve(SchemaInferer)
    return inferer.infer_schema(input)


def parse_udf(
    instructions: str,
    response_format: type[ResponseFormat] | None = None,
    example_table_name: str | None = None,
    example_field_name: str | None = None,
    max_examples: int = 100,
    model_name: str | None = None,
    batch_size: int | None = None,
    max_concurrency: int = 8,
    **api_kwargs,
) -> UserDefinedFunction:
    """Create an asynchronous Spark pandas UDF for parsing responses.
    This function allows users to create UDFs that parse responses based on
    provided instructions and either a predefined response format or example data.
    It supports both structured responses using Pydantic models and plain text responses.
    Each partition maintains its own cache to eliminate duplicate API calls within
    the partition, significantly reducing API usage and costs when processing
    datasets with overlapping content.

    Args:
        instructions (str): The system prompt or instructions for the model.
        response_format (type[ResponseFormat] | None): The desired output format.
            Either `str` for plain text or a Pydantic `BaseModel` for structured JSON output.
            If not provided, the schema will be inferred from example data.
        example_table_name (str | None): Name of the Spark table containing example data.
            If provided, `example_field_name` must also be specified.
        example_field_name (str | None): Name of the field in the table to use as examples.
            If provided, `example_table_name` must also be specified.
        max_examples (int): Maximum number of examples to retrieve for schema inference.
            Defaults to 100.
        model_name (str | None): For Azure OpenAI, use your deployment name (e.g., "my-gpt4-deployment").
            For OpenAI, use the model name (e.g., "gpt-4.1-mini"). Defaults to configured model in DI container
            via ResponsesModelName if not provided.
        batch_size (int | None): Number of rows per async batch request within each partition.
            Larger values reduce API call overhead but increase memory usage.
            Defaults to None (automatic batch size optimization that dynamically
            adjusts based on execution time, targeting 30-60 seconds per batch).
            Set to a positive integer (e.g., 32-128) for fixed batch size
        max_concurrency (int): Maximum number of concurrent API requests **PER EXECUTOR**.
            Total cluster concurrency = max_concurrency × number_of_executors.
            Higher values increase throughput but may hit OpenAI rate limits.
            Recommended: 4-12 per executor. Defaults to 8.
        **api_kwargs: Additional OpenAI API parameters (e.g. ``temperature``, ``top_p``,
            ``frequency_penalty``, ``presence_penalty``, ``seed``, ``max_output_tokens``, etc.)
            forwarded verbatim to the underlying API calls. These parameters are applied to
            all API requests made by the UDF and override any parameters set in the
            response_format or example data.
    Returns:
        UserDefinedFunction: A Spark pandas UDF configured to parse responses asynchronously.
            Output schema is `StringType` for str response format or a struct derived from
            the response_format for BaseModel.
    Raises:
        ValueError: If neither `response_format` nor `example_table_name` and `example_field_name` are provided.
    """

    if not response_format and not (example_field_name and example_table_name):
        raise ValueError("Either response_format or example_table_name and example_field_name must be provided.")

    schema: InferredSchema | None = None

    if not response_format:
        schema = infer_schema(
            instructions=instructions,
            example_table_name=example_table_name,
            example_field_name=example_field_name,
            max_examples=max_examples,
        )

    return responses_udf(
        instructions=schema.inference_prompt if schema else instructions,
        response_format=schema.model if schema else response_format,
        model_name=model_name,
        batch_size=batch_size,
        max_concurrency=max_concurrency,
        **api_kwargs,
    )


def embeddings_udf(
    model_name: str | None = None,
    batch_size: int | None = None,
    max_concurrency: int = 8,
    **api_kwargs,
) -> UserDefinedFunction:
    """Create an asynchronous Spark pandas UDF for generating embeddings.

    Configures and builds UDFs that leverage `pandas_ext.aio.embeddings_with_cache`
    to generate vector embeddings from OpenAI models asynchronously.
    Each partition maintains its own cache to eliminate duplicate API calls within
    the partition, significantly reducing API usage and costs when processing
    datasets with overlapping content.

    Note:
        Authentication must be configured via SparkContext environment variables.
        Set the appropriate environment variables on the SparkContext:

        For OpenAI:
            sc.environment["OPENAI_API_KEY"] = "your-openai-api-key"

        For Azure OpenAI:
            sc.environment["AZURE_OPENAI_API_KEY"] = "your-azure-openai-api-key"
            sc.environment["AZURE_OPENAI_BASE_URL"] = "https://YOUR-RESOURCE-NAME.services.ai.azure.com/openai/v1/"
            sc.environment["AZURE_OPENAI_API_VERSION"] = "preview"

    Args:
        model_name (str | None): For Azure OpenAI, use your deployment name (e.g., "my-embedding-deployment").
            For OpenAI, use the model name (e.g., "text-embedding-3-small").
            Defaults to configured model in DI container via EmbeddingsModelName if not provided.
        batch_size (int | None): Number of rows per async batch request within each partition.
            Larger values reduce API call overhead but increase memory usage.
            Defaults to None (automatic batch size optimization that dynamically
            adjusts based on execution time, targeting 30-60 seconds per batch).
            Set to a positive integer (e.g., 64-256) for fixed batch size.
            Embeddings typically handle larger batches efficiently.
        max_concurrency (int): Maximum number of concurrent API requests **PER EXECUTOR**.
            Total cluster concurrency = max_concurrency × number_of_executors.
            Higher values increase throughput but may hit OpenAI rate limits.
            Recommended: 4-12 per executor. Defaults to 8.
        **api_kwargs: Additional OpenAI API parameters (e.g., dimensions for text-embedding-3 models).

    Returns:
        UserDefinedFunction: A Spark pandas UDF configured to generate embeddings asynchronously
            with automatic caching for duplicate inputs within each partition,
            returning an `ArrayType(FloatType())` column.

    Note:
        For optimal performance in distributed environments:
        - **Automatic Caching**: Duplicate inputs within each partition are cached,
          reducing API calls and costs significantly on datasets with repeated content
        - Monitor OpenAI API rate limits when scaling executor count
        - Consider your OpenAI tier limits: total_requests = max_concurrency × executors
        - Embeddings API typically has higher throughput than chat completions
        - Use larger batch_size for embeddings compared to response generation
    """

    _model_name = model_name or CONTAINER.resolve(EmbeddingsModelName).value

    @pandas_udf(returnType=ArrayType(FloatType()))  # type: ignore[call-overload,misc]
    def _embeddings_udf(col: Iterator[pd.Series]) -> Iterator[pd.Series]:
        pandas_ext.embeddings_model(_model_name)
        cache = AsyncBatchingMapProxy[str, np.ndarray](
            batch_size=batch_size,
            max_concurrency=max_concurrency,
        )

        try:
            for part in col:
                embeddings: pd.Series = asyncio.run(part.aio.embeddings_with_cache(cache=cache, **api_kwargs))
                yield embeddings.map(lambda x: x.tolist())
        finally:
            asyncio.run(cache.clear())

    return _embeddings_udf  # type: ignore[return-value]


def split_to_chunks_udf(max_tokens: int, sep: list[str]) -> UserDefinedFunction:
    """Create a pandas‑UDF that splits text into token‑bounded chunks.

    Args:
        max_tokens (int): Maximum tokens allowed per chunk.
        sep (list[str]): Ordered list of separator strings used by ``TextChunker``.

    Returns:
        A pandas UDF producing an ``ArrayType(StringType())`` column whose
            values are lists of chunks respecting the ``max_tokens`` limit.
    """

    @pandas_udf(ArrayType(StringType()))  # type: ignore[call-overload,misc]
    def fn(col: Iterator[pd.Series]) -> Iterator[pd.Series]:
        encoding = tiktoken.get_encoding("o200k_base")
        chunker = TextChunker(encoding)

        for part in col:
            yield part.map(lambda x: chunker.split(x, max_tokens=max_tokens, sep=sep) if isinstance(x, str) else [])

    return fn  # type: ignore[return-value]


def count_tokens_udf() -> UserDefinedFunction:
    """Create a pandas‑UDF that counts tokens for every string cell.

    The UDF uses *tiktoken* to approximate tokenisation and caches the
    resulting ``Encoding`` object per executor.

    Returns:
        A pandas UDF producing an ``IntegerType`` column with token counts.
    """

    @pandas_udf(IntegerType())  # type: ignore[call-overload]
    def fn(col: Iterator[pd.Series]) -> Iterator[pd.Series]:
        encoding = tiktoken.get_encoding("o200k_base")

        for part in col:
            yield part.map(lambda x: len(encoding.encode(x)) if isinstance(x, str) else 0)

    return fn  # type: ignore[return-value]


def similarity_udf() -> UserDefinedFunction:
    """Create a pandas-UDF that computes cosine similarity between embedding vectors.

    Returns:
        UserDefinedFunction: A Spark pandas UDF that takes two embedding vector columns
            and returns their cosine similarity as a FloatType column.
    """

    @pandas_udf(FloatType())  # type: ignore[call-overload]
    def fn(a: pd.Series, b: pd.Series) -> pd.Series:
        # Import pandas_ext to ensure .ai accessor is available in Spark workers
        from openaivec import pandas_ext

        # Explicitly reference pandas_ext to satisfy linters
        assert pandas_ext is not None

        return pd.DataFrame({"a": a, "b": b}).ai.similarity("a", "b")

    return fn  # type: ignore[return-value]
