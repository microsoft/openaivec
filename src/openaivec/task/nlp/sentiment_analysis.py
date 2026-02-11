"""Sentiment analysis task definition."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from openaivec._model import PreparedTask
from openaivec.task._prompt_templates import english_categorical_policy, join_sections, same_language_policy
from openaivec.task._registry import TaskSpec

__all__ = ["sentiment_analysis"]


class SentimentAnalysis(BaseModel):
    """Sentiment analysis output."""

    model_config = ConfigDict(extra="forbid")

    sentiment: Literal["positive", "negative", "neutral"] = Field(
        description="Overall sentiment (positive, negative, neutral)"
    )
    confidence: float = Field(ge=0, le=1, description="Confidence score for sentiment (0.0-1.0)")
    emotions: list[Literal["joy", "sadness", "anger", "fear", "surprise", "disgust"]] = Field(
        description="Detected emotions (joy, sadness, anger, fear, surprise, disgust)"
    )
    emotion_scores: list[Annotated[float, Field(ge=0, le=1)]] = Field(
        description="Confidence scores for each emotion (0.0-1.0)"
    )
    polarity: float = Field(ge=-1, le=1, description="Polarity score from -1.0 (negative) to 1.0 (positive)")
    subjectivity: float = Field(ge=0, le=1, description="Subjectivity score from 0.0 (objective) to 1.0 (subjective)")


def _build_instructions() -> str:
    return join_sections(
        "Analyze sentiment and emotions in the input text.",
        "Return sentiment class, sentiment confidence, emotions, emotion scores, polarity, and subjectivity.",
        same_language_policy(),
        english_categorical_policy(["sentiment", "emotions"]),
    )


def sentiment_analysis() -> PreparedTask[SentimentAnalysis]:
    """Create a sentiment analysis task."""
    return PreparedTask(
        instructions=_build_instructions(),
        response_format=SentimentAnalysis,
    )


TASK_SPEC = TaskSpec(
    key="nlp.sentiment_analysis",
    domain="nlp",
    summary="Analyze sentiment, emotions, polarity, and subjectivity.",
    factory=sentiment_analysis,
    response_format=SentimentAnalysis,
)

