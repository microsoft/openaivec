"""Sentiment analysis task for OpenAI API.

This module provides a predefined task for sentiment analysis that analyzes
sentiment and emotions in text using OpenAI's language models.

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
        task=nlp.SENTIMENT_ANALYSIS
    )

    texts = ["I love this product!", "This is terrible and disappointing."]
    analyses = analyzer.parse(texts)

    for analysis in analyses:
        print(f"Sentiment: {analysis.sentiment}")
        print(f"Confidence: {analysis.confidence}")
        print(f"Emotions: {analysis.emotions}")
    ```

    With pandas integration:

    ```python
    import pandas as pd
    from openaivec import pandas_ext  # Required for .ai accessor
    from openaivec.task import nlp

    df = pd.DataFrame({"text": ["I love this product!", "This is terrible and disappointing."]})
    df["sentiment"] = df["text"].ai.task(nlp.SENTIMENT_ANALYSIS)

    # Extract sentiment components
    extracted_df = df.ai.extract("sentiment")
    print(extracted_df[["text", "sentiment_sentiment", "sentiment_confidence", "sentiment_polarity"]])
    ```

Attributes:
    SENTIMENT_ANALYSIS (PreparedTask): A prepared task instance
        configured for sentiment analysis with temperature=0.0 and
        top_p=1.0 for deterministic output.
"""

from typing import Literal

from pydantic import BaseModel, Field

from openaivec._model import PreparedTask

__all__ = ["SENTIMENT_ANALYSIS"]


class SentimentAnalysis(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"] = Field(
        description="Overall sentiment (positive, negative, neutral)"
    )
    confidence: float = Field(description="Confidence score for sentiment (0.0-1.0)")
    emotions: list[Literal["joy", "sadness", "anger", "fear", "surprise", "disgust"]] = Field(
        description="Detected emotions (joy, sadness, anger, fear, surprise, disgust)"
    )
    emotion_scores: list[float] = Field(description="Confidence scores for each emotion (0.0-1.0)")
    polarity: float = Field(description="Polarity score from -1.0 (negative) to 1.0 (positive)")
    subjectivity: float = Field(description="Subjectivity score from 0.0 (objective) to 1.0 (subjective)")


SENTIMENT_ANALYSIS = PreparedTask(
    instructions="Analyze the sentiment and emotions in the following text. Provide overall "
    "sentiment classification, confidence scores, detected emotions, polarity, and subjectivity "
    "measures.\n\nIMPORTANT: Provide all analysis in the same language as the input text, except "
    "for the predefined categorical fields (sentiment, emotions) which must use the exact "
    "English values specified (positive/negative/neutral for sentiment, and "
    "joy/sadness/anger/fear/surprise/disgust for emotions).",
    response_format=SentimentAnalysis,
    api_kwargs={"temperature": 0.0, "top_p": 1.0},
)
