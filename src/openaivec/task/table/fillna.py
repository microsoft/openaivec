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

    # Apply filled values back to DataFrame
    for result in filled_salaries:
        df.loc[result.index, "salary"] = result.output
    ```

    With BatchResponses for more control:

    ```python
    from openai import OpenAI
    from openaivec._responses import BatchResponses
    from openaivec.task.table import fillna

    client = OpenAI()
    df = pd.DataFrame({...})  # Your DataFrame with missing values

    # Create fillna task for target column
    task = fillna(df, "target_column")

    # Get rows with missing values in target column
    missing_rows = df[df["target_column"].isna()]

    # Process with BatchResponses
    filler = BatchResponses.of_task(
        client=client,
        model_name="gpt-4.1-mini",
        task=task
    )

    # Generate inputs for missing rows
    inputs = []
    for idx, row in missing_rows.iterrows():
        inputs.append({
            "index": idx,
            "input": {k: v for k, v in row.items() if k != "target_column"}
        })

    filled_values = filler.parse(inputs)
    ```
"""

import json

import pandas as pd
from pydantic import BaseModel, Field

from openaivec._model import PreparedTask
from openaivec._prompt import FewShotPromptBuilder

__all__ = ["fillna", "FillNaResponse"]


def get_examples(df: pd.DataFrame, target_column_name: str, max_examples: int) -> list[dict]:
    examples: list[dict] = []

    samples: pd.DataFrame = df.sample(frac=1).reset_index(drop=True).drop_duplicates()
    samples = samples.dropna(subset=[target_column_name])

    for i, row in samples.head(max_examples).iterrows():
        examples.append(
            {
                "index": i,
                "input": {k: v for k, v in row.items() if k != target_column_name},
                "output": row[target_column_name],
            }
        )

    return examples


def get_instructions(df: pd.DataFrame, target_column_name: str, max_examples: int) -> str:
    examples = get_examples(df, target_column_name, max_examples)

    builder = (
        FewShotPromptBuilder()
        .purpose("Fill missing values in the target column based on the context provided by other columns.")
        .caution("Ensure that the filled values are consistent with the data in other columns.")
    )

    for row in examples:
        builder.example(
            input_value=json.dumps({"index": row["index"], "input": row["input"]}, ensure_ascii=False),
            output_value=json.dumps({"index": row["index"], "output": row["output"]}, ensure_ascii=False),
        )

    return builder.improve().build()


class FillNaResponse(BaseModel):
    """Response model for missing value imputation results.

    Contains the row index and the imputed value for a specific missing
    entry in the target column.
    """

    index: int = Field(description="Index of the row in the original DataFrame")
    output: int | float | str | bool | None = Field(
        description="Filled value for the target column. This value should be JSON-compatible "
        "and match the target column type in the original DataFrame."
    )


def fillna(df: pd.DataFrame, target_column_name: str, max_examples: int = 500, **api_kwargs) -> PreparedTask:
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
            learning. Defaults to 500. Higher values provide more context
            but increase token usage and processing time.
        **api_kwargs: Additional keyword arguments to pass to the OpenAI API,
            such as temperature, top_p, etc.

    Returns:
        PreparedTask configured for missing value imputation with:
        - Instructions based on DataFrame patterns
        - FillNaResponse format for structured output
        - Default deterministic settings (temperature=0.0, top_p=1.0)

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
    instructions = get_instructions(df, target_column_name, max_examples)
    # Set default values for deterministic results if not provided
    if not api_kwargs:
        api_kwargs = {"temperature": 0.0, "top_p": 1.0}
    return PreparedTask(instructions=instructions, response_format=FillNaResponse, api_kwargs=api_kwargs)
