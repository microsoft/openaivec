"""Morphological analysis task for OpenAI API.

This module provides a predefined task for morphological analysis including
tokenization, part-of-speech tagging, and lemmatization using OpenAI's
language models.

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
        task=nlp.MORPHOLOGICAL_ANALYSIS
    )

    texts = ["Running quickly", "The cats are sleeping"]
    analyses = analyzer.parse(texts)

    for analysis in analyses:
        print(f"Tokens: {analysis.tokens}")
        print(f"POS Tags: {analysis.pos_tags}")
        print(f"Lemmas: {analysis.lemmas}")
    ```

    With pandas integration:

    ```python
    import pandas as pd
    from openaivec import pandas_ext  # Required for .ai accessor
    from openaivec.task import nlp

    df = pd.DataFrame({"text": ["Running quickly", "The cats are sleeping"]})
    df["analysis"] = df["text"].ai.task(nlp.MORPHOLOGICAL_ANALYSIS)

    # Extract analysis components
    extracted_df = df.ai.extract("analysis")
    print(extracted_df[["text", "analysis_tokens", "analysis_pos_tags", "analysis_lemmas"]])
    ```

Attributes:
    MORPHOLOGICAL_ANALYSIS (PreparedTask): A prepared task instance
        configured for morphological analysis with temperature=0.0 and
        top_p=1.0 for deterministic output.
"""

from pydantic import BaseModel, Field

from openaivec._model import PreparedTask

__all__ = ["MORPHOLOGICAL_ANALYSIS"]


class MorphologicalAnalysis(BaseModel):
    tokens: list[str] = Field(description="List of tokens in the text")
    pos_tags: list[str] = Field(description="Part-of-speech tags for each token")
    lemmas: list[str] = Field(description="Lemmatized form of each token")
    morphological_features: list[str] = Field(
        description="Morphological features for each token (e.g., tense, number, case)"
    )


MORPHOLOGICAL_ANALYSIS = PreparedTask(
    instructions="Perform morphological analysis on the following text. Break it down into tokens, "
    "identify part-of-speech tags, provide lemmatized forms, and extract morphological features "
    "for each token.",
    response_format=MorphologicalAnalysis,
    api_kwargs={"temperature": 0.0, "top_p": 1.0},
)
