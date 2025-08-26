"""Dependency parsing task for OpenAI API.

This module provides a predefined task for dependency parsing that analyzes
syntactic dependencies between words in sentences using OpenAI's language models.

Example:
    Basic usage with BatchResponses:

    ```python
    from openai import OpenAI
    from openaivec._responses import BatchResponses
    from openaivec.task import nlp

    client = OpenAI()
    analyzer = BatchResponses.of_task(
        client=client,
        model_name="gpt-4.1-mini",
        task=nlp.DEPENDENCY_PARSING
    )

    texts = ["The cat sat on the mat.", "She quickly ran to the store."]
    analyses = analyzer.parse(texts)

    for analysis in analyses:
        print(f"Tokens: {analysis.tokens}")
        print(f"Dependencies: {analysis.dependencies}")
        print(f"Root: {analysis.root_word}")
    ```

    With pandas integration:

    ```python
    import pandas as pd
    from openaivec import pandas_ext  # Required for .ai accessor
    from openaivec.task import nlp

    df = pd.DataFrame({"text": ["The cat sat on the mat.", "She quickly ran to the store."]})
    df["parsing"] = df["text"].ai.task(nlp.DEPENDENCY_PARSING)

    # Extract parsing components
    extracted_df = df.ai.extract("parsing")
    print(extracted_df[["text", "parsing_tokens", "parsing_root_word", "parsing_syntactic_structure"]])
    ```

Attributes:
    DEPENDENCY_PARSING (PreparedTask): A prepared task instance
        configured for dependency parsing with temperature=0.0 and
        top_p=1.0 for deterministic output.
"""

from pydantic import BaseModel, Field

from openaivec._model import PreparedTask

__all__ = ["DEPENDENCY_PARSING"]


class DependencyRelation(BaseModel):
    head: str = Field(description="Head word in the dependency relation")
    dependent: str = Field(description="Dependent word in the dependency relation")
    relation: str = Field(description="Type of dependency relation")
    head_pos: int = Field(description="Position of head word in the sentence")
    dependent_pos: int = Field(description="Position of dependent word in the sentence")


class DependencyParsing(BaseModel):
    tokens: list[str] = Field(description="List of tokens in the sentence")
    dependencies: list[DependencyRelation] = Field(description="Dependency relations between tokens")
    root_word: str = Field(description="Root word of the sentence")
    syntactic_structure: str = Field(description="Tree representation of the syntactic structure")


DEPENDENCY_PARSING = PreparedTask(
    instructions="Parse the syntactic dependencies in the following text. Identify dependency "
    "relations between words, determine the root word, and provide a tree representation of the "
    "syntactic structure.",
    response_format=DependencyParsing,
    api_kwargs={"temperature": 0.0, "top_p": 1.0},
)
