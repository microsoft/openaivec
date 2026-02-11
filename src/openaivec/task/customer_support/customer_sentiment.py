"""Customer support sentiment task definition."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from openaivec._model import PreparedTask
from openaivec.task._prompt_templates import english_categorical_policy, join_sections, same_language_policy
from openaivec.task._registry import TaskSpec

__all__ = ["customer_sentiment"]


class CustomerSentiment(BaseModel):
    """Customer sentiment analysis output."""

    model_config = ConfigDict(extra="forbid")

    sentiment: Literal["positive", "negative", "neutral", "mixed"] = Field(
        description="Overall sentiment (positive, negative, neutral, mixed)"
    )
    satisfaction_level: Literal["very_satisfied", "satisfied", "neutral", "dissatisfied", "very_dissatisfied"] = Field(
        description="Customer satisfaction level"
    )
    emotional_state: Literal["happy", "frustrated", "angry", "disappointed", "confused", "grateful", "worried"] = (
        Field(description="Primary emotional state")
    )
    confidence: float = Field(ge=0, le=1, description="Confidence score for sentiment analysis (0.0-1.0)")
    churn_risk: Literal["low", "medium", "high", "critical"] = Field(description="Risk of customer churn")
    sentiment_intensity: float = Field(ge=0, le=1, description="Intensity of sentiment from 0.0 to 1.0")
    polarity_score: float = Field(ge=-1, le=1, description="Polarity score from -1.0 to 1.0")
    tone_indicators: list[str] = Field(description="Specific words or phrases indicating tone")
    relationship_status: Literal["new", "loyal", "at_risk", "detractor", "advocate"] = Field(
        description="Customer relationship status"
    )
    response_approach: Literal["empathetic", "professional", "solution_focused", "escalation_required"] = Field(
        description="Recommended response approach"
    )


def _build_instructions(business_context: str) -> str:
    return join_sections(
        "Analyze customer sentiment in support interactions.",
        f"Business context: {business_context}",
        (
            "Assess sentiment, satisfaction level, emotional state, churn risk, relationship status, "
            "and recommended response approach."
        ),
        same_language_policy(),
        english_categorical_policy(
            [
                "sentiment",
                "satisfaction_level",
                "emotional_state",
                "churn_risk",
                "relationship_status",
                "response_approach",
            ]
        ),
    )


def customer_sentiment(business_context: str = "general customer support") -> PreparedTask[CustomerSentiment]:
    """Create a customer support sentiment analysis task."""
    return PreparedTask(
        instructions=_build_instructions(business_context=business_context),
        response_format=CustomerSentiment,
    )


TASK_SPEC = TaskSpec(
    key="customer_support.customer_sentiment",
    domain="customer_support",
    summary="Analyze sentiment and churn risk for customer inquiries.",
    factory=customer_sentiment,
    response_format=CustomerSentiment,
)

