"""Named entity recognition task for OpenAI API.

This module provides a predefined task for named entity recognition that
identifies and classifies named entities in text using OpenAI's language models.

Example:
    Basic usage with BatchResponses:

    ```python
    from openai import OpenAI
    from openaivec.responses import BatchResponses
    from openaivec.task import nlp

    client = OpenAI()
    analyzer = BatchResponses.of_task(
        client=client,
        model_name="gpt-4o-mini",
        task=nlp.NAMED_ENTITY_RECOGNITION
    )

    texts = ["John works at Microsoft in Seattle", "The meeting is on March 15th"]
    analyses = analyzer.parse(texts)

    for analysis in analyses:
        print(f"Persons: {analysis.persons}")
        print(f"Organizations: {analysis.organizations}")
        print(f"Locations: {analysis.locations}")
    ```

    With pandas integration:

    ```python
    import pandas as pd
    from openaivec import pandas_ext  # Required for .ai accessor
    from openaivec.task import nlp

    df = pd.DataFrame({"text": ["John works at Microsoft in Seattle", "The meeting is on March 15th"]})
    df["entities"] = df["text"].ai.task(nlp.NAMED_ENTITY_RECOGNITION)

    # Extract entity components
    extracted_df = df.ai.extract("entities")
    print(extracted_df[["text", "entities_persons", "entities_organizations", "entities_locations"]])
    ```

Attributes:
    NAMED_ENTITY_RECOGNITION (PreparedTask): A prepared task instance
        configured for named entity recognition with temperature=0.0 and
        top_p=1.0 for deterministic output.
"""

from typing import List, Optional

from pydantic import BaseModel, Field

from ...model import PreparedTask

__all__ = ["NAMED_ENTITY_RECOGNITION"]


class NamedEntity(BaseModel):
    text: str = Field(description="The entity text")
    label: str = Field(description="Entity type label")
    start: int = Field(description="Start position in the original text")
    end: int = Field(description="End position in the original text")
    confidence: Optional[float] = Field(description="Confidence score (0.0-1.0)")


class NamedEntityRecognition(BaseModel):
    persons: List[NamedEntity] = Field(description="Person entities")
    organizations: List[NamedEntity] = Field(description="Organization entities")
    locations: List[NamedEntity] = Field(description="Location entities")
    dates: List[NamedEntity] = Field(description="Date and time entities")
    money: List[NamedEntity] = Field(description="Money and currency entities")
    percentages: List[NamedEntity] = Field(description="Percentage entities")
    miscellaneous: List[NamedEntity] = Field(description="Other named entities")


NAMED_ENTITY_RECOGNITION = PreparedTask(
    instructions="Identify and classify named entities in the following text. Extract persons, organizations, locations, dates, money, percentages, and other miscellaneous entities with their positions and confidence scores.",
    response_format=NamedEntityRecognition,
    temperature=0.0,
    top_p=1.0,
)
