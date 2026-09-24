"""Missing value imputation task for DataFrame columns.

This module provides functionality to intelligently fill missing values in DataFrame
columns using AI-powered analysis. The task analyzes existing data patterns to
generate contextually appropriate values for missing entries.

Example:
    Basic usage with pandas DataFrame:

    ```python
    import pandas as pd
    from openaivec import pandas_ext  # Required for .ai accessor
    from openaivec.task.table import fillna

    # Create DataFrame with missing values
    df = pd.DataFrame({
        "name": ["Alice", "Bob", None, "David"],
        "age": [25, 30, 35, None],
        "city": ["New York", "London", "Tokyo", "Paris"],
        "salary": [50000, 60000, 70000, None]
    })

    # Fill missing values in the 'salary' column
    task = fillna(df, "salary")
    filled_salaries = df[df["salary"].isna()].ai.task(task)

    missing_positions = df["salary"].isna().to_numpy().nonzero()[0]
    for position, result in zip(missing_positions, filled_salaries):
        df.iat[position, df.columns.get_loc("salary")] = result.output
    ```

    Opt in to LLM prompt refinement before asynchronous execution:

    ```python
    from openaivec.task.table import fillna

    task = fillna(df, "target_column", improve_prompt=True)
    missing_rows = df[df["target_column"].isna()]
    filled_values = await missing_rows.aio.task(task)
    ```
"""

import json
from xml.etree import ElementTree

import numpy as np
import pandas as pd
import tiktoken
from pydantic import BaseModel, ConfigDict, Field

from openaivec._model import PreparedTask
from openaivec._prompt import FewShotPromptBuilder
from openaivec._provider import CONTAINER
from openaivec.task._prompt_templates import same_language_policy
from openaivec.task._registry import TaskSpec

__all__ = ["FillNaResponse", "fillna"]

_DEFAULT_EXAMPLES = 8
_EXAMPLE_CHAR_BUDGET = 6000
_EXAMPLE_TOKEN_BUDGET = 1500
_EXAMPLE_XML_OVERHEAD_TOKENS = 16


def _get_examples(df: pd.DataFrame, target_column_name: str, max_examples: int) -> list[tuple[str, str]]:
    from openaivec.pandas_ext._common import _df_rows_to_json_series

    positions = np.flatnonzero(df[target_column_name].notna().to_numpy())
    candidate_count = min(len(positions), max_examples * 4, _EXAMPLE_CHAR_BUDGET // 30 * 4)
    selected = np.random.default_rng(0).choice(positions, size=candidate_count, replace=False)
    samples = df.iloc[selected].copy()
    outputs = samples[target_column_name].tolist()
    samples[target_column_name] = None
    inputs = _df_rows_to_json_series(samples)

    examples: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    used_chars = 0
    used_tokens = 0
    encoding = CONTAINER.resolve(tiktoken.Encoding)
    for input_value, output in zip(inputs, outputs):
        output_value = json.dumps({"output": output}, ensure_ascii=False, default=str)
        pair = (input_value, output_value)
        pair_chars = len(input_value) + len(output_value) + 100
        if pair in seen or pair_chars + used_chars > _EXAMPLE_CHAR_BUDGET:
            continue
        example = ElementTree.Element("Example")
        ElementTree.SubElement(example, "Input").text = input_value
        ElementTree.SubElement(example, "Output").text = output_value
        pair_tokens = (
            len(encoding.encode_ordinary(ElementTree.tostring(example, encoding="unicode")))
            + _EXAMPLE_XML_OVERHEAD_TOKENS
        )
        if pair_tokens + used_tokens > _EXAMPLE_TOKEN_BUDGET:
            continue
        seen.add(pair)
        examples.append(pair)
        used_chars += pair_chars
        used_tokens += pair_tokens
        if len(examples) == max_examples:
            break
    return examples


def _build_instructions(df: pd.DataFrame, target_column_name: str, max_examples: int, improve_prompt: bool) -> str:
    examples = _get_examples(df, target_column_name, max_examples)

    builder = (
        FewShotPromptBuilder()
        .purpose(
            f"Fill the missing value in column {target_column_name!r} of the JSON row. "
            "Return an object with only the output field."
        )
        .caution("Ensure that the filled values are consistent with the data in other columns.")
        .caution(same_language_policy())
    )

    if not examples:
        if improve_prompt:
            raise ValueError("No examples fit the prompt budget; prompt improvement requires at least one example.")
        return (
            f"Fill the missing value in column {target_column_name!r} of the input JSON row. "
            "Return an object with only the output field. "
            "Ensure the value is consistent with the other columns. "
            f"{same_language_policy()}"
        )
    for input_value, output_value in examples:
        builder.example(input_value=input_value, output_value=output_value)

    if improve_prompt:
        builder.improve()
    return builder.build()


class FillNaResponse(BaseModel):
    """Response model for missing value imputation results.

    Contains the imputed value for a specific missing entry in the target column.
    """

    model_config = ConfigDict(extra="forbid")

    output: int | float | str | bool | None = Field(
        description="Filled value for the target column. This value should be JSON-compatible "
        "and match the target column type in the original DataFrame."
    )


def fillna(
    df: pd.DataFrame, target_column_name: str, max_examples: int = _DEFAULT_EXAMPLES, *, improve_prompt: bool = False
) -> PreparedTask[FillNaResponse]:
    """Create a prepared task for filling missing values in a DataFrame column.

    Analyzes the provided DataFrame to understand data patterns and creates
    a configured task that can intelligently fill missing values in the
    specified target column. The task uses few-shot learning with examples
    extracted from non-null rows in the DataFrame.

    Args:
        df (pd.DataFrame): Source DataFrame containing the data with missing values.
        target_column_name (str): Name of the column to fill missing values for.
            This column should exist in the DataFrame and contain some
            non-null values to serve as training examples.
        max_examples (int): Maximum number of example rows to use for few-shot
            learning. Defaults to 8. Example text is limited to 6000 characters
            and an estimated 1500 tokens. Sampling is deterministic.
        improve_prompt (bool): Request optional LLM prompt refinement. Defaults
            to ``False``, so construction is local and requires no API access.
            For async execution, prepare explicitly before awaiting ``df.aio.task(task)``.

    Returns:
        PreparedTask configured for missing value imputation with:
        - Instructions based on DataFrame patterns
        - FillNaResponse format for structured output
        - No embedded API parameter defaults

    Raises:
        ValueError: If target_column_name doesn't exist in DataFrame,
            contains no non-null values for training examples, DataFrame is empty,
            or max_examples is not a positive integer.

    Example:
        ```python
        import pandas as pd
        from openaivec.task.table import fillna

        df = pd.DataFrame({
            "product": ["laptop", "phone", "tablet", "laptop"],
            "brand": ["Apple", "Samsung", None, "Dell"],
            "price": [1200, 800, 600, 1000]
        })

        # Create task to fill missing brand values
        task = fillna(df, "brand")

        # Use with pandas AI accessor
        missing_brands = df[df["brand"].isna()].ai.task(task)
        ```
    """
    if df.empty:
        raise ValueError("DataFrame is empty.")
    if not isinstance(max_examples, int) or max_examples <= 0:
        raise ValueError("max_examples must be a positive integer.")
    if target_column_name not in df.columns:
        raise ValueError(f"Column '{target_column_name}' does not exist in the DataFrame.")
    if df[target_column_name].notna().sum() == 0:
        raise ValueError(f"Column '{target_column_name}' contains no non-null values for training examples.")
    instructions = _build_instructions(df, target_column_name, max_examples, improve_prompt)
    return PreparedTask(instructions=instructions, response_format=FillNaResponse)


TASK_SPEC = TaskSpec(
    key="table.fillna",
    domain="table",
    summary="Fill missing DataFrame column values using few-shot context rows.",
    factory=fillna,
    response_format=FillNaResponse,
)
