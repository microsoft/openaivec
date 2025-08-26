"""Keyword extraction task for OpenAI API.

This module provides a predefined task for keyword extraction that identifies
important keywords and phrases from text using OpenAI's language models.

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
        task=nlp.KEYWORD_EXTRACTION
    )

    texts = ["Machine learning is transforming the technology industry.",
             "Climate change affects global weather patterns."]
    analyses = analyzer.parse(texts)

    for analysis in analyses:
        print(f"Keywords: {analysis.keywords}")
        print(f"Key phrases: {analysis.keyphrases}")
        print(f"Topics: {analysis.topics}")
    ```

    With pandas integration:

    ```python
    import pandas as pd
    from openaivec import pandas_ext  # Required for .ai accessor
    from openaivec.task import nlp

    df = pd.DataFrame({"text": ["Machine learning is transforming the technology industry.",
                               "Climate change affects global weather patterns."]})
    df["keywords"] = df["text"].ai.task(nlp.KEYWORD_EXTRACTION)

    # Extract keyword components
    extracted_df = df.ai.extract("keywords")
    print(extracted_df[["text", "keywords_keywords", "keywords_topics", "keywords_summary"]])
    ```

Attributes:
    KEYWORD_EXTRACTION (PreparedTask): A prepared task instance
        configured for keyword extraction with temperature=0.0 and
        top_p=1.0 for deterministic output.
"""

from pydantic import BaseModel, Field

from openaivec._model import PreparedTask

__all__ = ["KEYWORD_EXTRACTION"]


class Keyword(BaseModel):
    text: str = Field(description="The keyword or phrase")
    score: float = Field(description="Importance score (0.0-1.0)")
    frequency: int = Field(description="Frequency of occurrence in the text")
    context: str | None = Field(description="Context where the keyword appears")


class KeywordExtraction(BaseModel):
    keywords: list[Keyword] = Field(description="Extracted keywords ranked by importance")
    keyphrases: list[Keyword] = Field(description="Extracted multi-word phrases ranked by importance")
    topics: list[str] = Field(description="Identified main topics in the text")
    summary: str = Field(description="Brief summary of the text content")


KEYWORD_EXTRACTION = PreparedTask(
    instructions="Extract important keywords and phrases from the following text. Rank them "
    "by importance, provide frequency counts, identify main topics, and generate a brief summary.",
    response_format=KeywordExtraction,
    api_kwargs={"temperature": 0.0, "top_p": 1.0},
)
