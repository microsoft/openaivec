"""Schema inference and parse UDF registration for DuckDB."""

from __future__ import annotations

from typing import Any

import duckdb
from pydantic import BaseModel

from openaivec._model import ResponseFormat
from openaivec._provider import CONTAINER
from openaivec._retry import RetryPolicy
from openaivec._schema import SchemaInferenceInput, SchemaInferenceOutput, SchemaInferer
from openaivec.duckdb_ext._udfs import responses_udf

__all__ = ["infer_schema", "parse_udf"]

def _quote_identifier(name: str) -> str:
    """Quote a single DuckDB column name, including embedded double quotes."""
    if not isinstance(name, str) or not name:
        raise ValueError("example_field_name must be a non-empty string")
    return '"' + name.replace('"', '""') + '"'


def infer_schema(
    conn: duckdb.DuckDBPyConnection,
    instructions: str,
    example_table_name: str,
    example_field_name: str,
    max_examples: int = 100,
    *,
    max_retries: int = 8,
    retry_policy: RetryPolicy | None = None,
    **api_kwargs: Any,
) -> SchemaInferenceOutput:
    """Infer a response schema from a bounded sample of a DuckDB column.

    The table is resolved through DuckDB's relation API, which accepts regular,
    schema-qualified, and quoted table names without interpolating the name in
    SQL. NULL values are skipped before the sample limit is applied.

    Args:
        conn (duckdb.DuckDBPyConnection): Connection containing the example table.
        instructions (str): Description of the information to extract.
        example_table_name (str): Table or view containing examples.
        example_field_name (str): Column containing example text.
        max_examples (int): Maximum number of non-NULL examples. Must be positive.
        max_retries (int): Maximum schema inference attempts. Must be positive.
        retry_policy (RetryPolicy | None): Transport retry policy.
        **api_kwargs: Extra parameters for schema inference API calls.

    Returns:
        SchemaInferenceOutput: Inferred prompt and Pydantic model.

    Raises:
        ValueError: If limits are invalid or the sample has no non-NULL values.

    Example:
        >>> import duckdb
        >>> conn = duckdb.connect()
        >>> conn.sql("CREATE TABLE reviews(text VARCHAR)")
        >>> # infer_schema(conn, "Extract sentiment", "reviews", "text")
    """
    if max_examples < 1:
        raise ValueError("max_examples must be >= 1")
    if max_retries < 1:
        raise ValueError("max_retries must be >= 1")
    if not isinstance(example_table_name, str) or not example_table_name:
        raise ValueError("example_table_name must be a non-empty string")

    column = _quote_identifier(example_field_name)
    examples = [
        row[0]
        for row in (
            conn.table(example_table_name)
            .filter(f"{column} IS NOT NULL")
            .select(f"CAST({column} AS VARCHAR)")
            .limit(max_examples)
            .fetchall()
        )
    ]
    if not examples:
        raise ValueError("No non-NULL examples found in the selected column")

    inference_input = SchemaInferenceInput(examples=examples, instructions=instructions)
    inferer = CONTAINER.resolve(SchemaInferer)
    return inferer.infer_schema(
        inference_input,
        max_retries=max_retries,
        retry_policy=retry_policy,
        **api_kwargs,
    )


def parse_udf(
    conn: duckdb.DuckDBPyConnection,
    name: str,
    *,
    instructions: str,
    response_format: type[ResponseFormat] | None = None,
    example_table_name: str | None = None,
    example_field_name: str | None = None,
    max_examples: int = 100,
    model_name: str | None = None,
    batch_size: int | None = 64,
    max_concurrency: int = 8,
    multimodal: bool = False,
    max_retries: int = 8,
    max_validation_retries: int = 3,
    retry_policy: RetryPolicy | None = None,
    **api_kwargs: Any,
) -> None:
    """Register a DuckDB parse UDF with an explicit or inferred output schema.

    DuckDB fixes UDF return types at registration. If ``response_format`` is
    omitted, the schema is inferred once from ``example_table_name`` and
    ``example_field_name`` before registering the UDF. The generated UDF then
    returns a typed DuckDB ``STRUCT`` for each non-NULL input.

    Args:
        conn (duckdb.DuckDBPyConnection): Connection on which to register the UDF.
        name (str): UDF name visible in SQL.
        instructions (str): Description of the information to parse.
        response_format (type[ResponseFormat] | None): ``str`` or a Pydantic
            ``BaseModel`` subclass. If None, infer the schema from examples.
        example_table_name (str | None): Source table for schema inference.
        example_field_name (str | None): Text column for schema inference.
        max_examples (int): Maximum non-NULL examples to inspect.
        model_name (str | None): Model or deployment name.
        batch_size (int | None): Inputs per API batch; None enables auto-tuning.
        max_concurrency (int): Maximum concurrent API requests.
        multimodal (bool): Whether file paths and URLs are multimodal inputs.
        max_retries (int): Maximum schema inference attempts.
        max_validation_retries (int): Additional extraction corrections.
        retry_policy (RetryPolicy | None): Transport retry policy.
        **api_kwargs: Extra parameters for inference and parsing API calls.

    Raises:
        ValueError: If limits are invalid or inference inputs are missing.
        TypeError: If an explicit response format is unsupported.

    Example:
        >>> import duckdb
        >>> from pydantic import BaseModel
        >>> class Parsed(BaseModel):
        ...     label: str
        >>> conn = duckdb.connect()
        >>> parse_udf(conn, "parse_label", instructions="Extract label", response_format=Parsed)
    """
    if max_examples < 1:
        raise ValueError("max_examples must be >= 1")
    if max_retries < 1:
        raise ValueError("max_retries must be >= 1")
    if max_validation_retries < 0:
        raise ValueError("max_validation_retries must be >= 0")

    if response_format is None:
        if not example_table_name or not example_field_name:
            raise ValueError("Both example_table_name and example_field_name are required for schema inference")
        schema = infer_schema(
            conn,
            instructions,
            example_table_name,
            example_field_name,
            max_examples=max_examples,
            max_retries=max_retries,
            retry_policy=retry_policy,
            **api_kwargs,
        )
        resolved_instructions = schema.inference_prompt
        resolved_response_format = schema.model
    elif response_format is str or (isinstance(response_format, type) and issubclass(response_format, BaseModel)):
        resolved_instructions = instructions
        resolved_response_format = response_format
    else:
        raise TypeError("response_format must be str or a Pydantic BaseModel subclass")

    responses_udf(
        conn,
        name,
        instructions=resolved_instructions,
        response_format=resolved_response_format,
        model_name=model_name,
        batch_size=batch_size,
        max_concurrency=max_concurrency,
        multimodal=multimodal,
        max_validation_retries=max_validation_retries,
        retry_policy=retry_policy,
        **api_kwargs,
    )
