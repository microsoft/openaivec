"""Customer sentiment analysis task for support interactions.

This module provides a predefined task for analyzing customer sentiment specifically
in support contexts, including satisfaction levels and emotional states that affect
customer experience and support strategy.

Example:
    Basic usage with BatchResponses:

    ```python
    from openai import OpenAI
    from openaivec._responses import BatchResponses
    from openaivec.task import customer_support

    client = OpenAI()
    analyzer = BatchResponses.of_task(
        client=client,
        model_name="gpt-4.1-mini",
        task=customer_support.CUSTOMER_SENTIMENT
    )

    inquiries = [
        "I'm really disappointed with your service. This is the third time I've had this issue.",
        "Thank you so much for your help! You've been incredibly patient.",
        "I need to cancel my subscription. It's not working for me."
    ]
    sentiments = analyzer.parse(inquiries)

    for sentiment in sentiments:
        print(f"Sentiment: {sentiment.sentiment}")
        print(f"Satisfaction: {sentiment.satisfaction_level}")
        print(f"Churn Risk: {sentiment.churn_risk}")
        print(f"Emotional State: {sentiment.emotional_state}")
    ```

    With pandas integration:

    ```python
    import pandas as pd
    from openaivec import pandas_ext  # Required for .ai accessor
    from openaivec.task import customer_support

    df = pd.DataFrame({"inquiry": [
        "I'm really disappointed with your service. This is the third time I've had this issue.",
        "Thank you so much for your help! You've been incredibly patient.",
        "I need to cancel my subscription. It's not working for me."
    ]})
    df["sentiment"] = df["inquiry"].ai.task(customer_support.CUSTOMER_SENTIMENT)

    # Extract sentiment components
    extracted_df = df.ai.extract("sentiment")
    print(extracted_df[[
        "inquiry", "sentiment_satisfaction_level",
        "sentiment_churn_risk", "sentiment_emotional_state"
    ]])
    ```

Attributes:
    CUSTOMER_SENTIMENT (PreparedTask): A prepared task instance
        configured for customer sentiment analysis with temperature=0.0 and
        top_p=1.0 for deterministic output.
"""

from typing import Literal

from pydantic import BaseModel, Field

from openaivec._model import PreparedTask

__all__ = ["customer_sentiment"]


class CustomerSentiment(BaseModel):
    sentiment: Literal["positive", "negative", "neutral", "mixed"] = Field(
        description="Overall sentiment (positive, negative, neutral, mixed)"
    )
    satisfaction_level: Literal["very_satisfied", "satisfied", "neutral", "dissatisfied", "very_dissatisfied"] = Field(
        description="Customer satisfaction (very_satisfied, satisfied, neutral, dissatisfied, very_dissatisfied)"
    )
    emotional_state: Literal["happy", "frustrated", "angry", "disappointed", "confused", "grateful", "worried"] = Field(
        description="Primary emotional state (happy, frustrated, angry, disappointed, confused, grateful, worried)"
    )
    confidence: float = Field(description="Confidence score for sentiment analysis (0.0-1.0)")
    churn_risk: Literal["low", "medium", "high", "critical"] = Field(
        description="Risk of customer churn (low, medium, high, critical)"
    )
    sentiment_intensity: float = Field(description="Intensity of sentiment from 0.0 (mild) to 1.0 (extreme)")
    polarity_score: float = Field(description="Polarity score from -1.0 (very negative) to 1.0 (very positive)")
    tone_indicators: list[str] = Field(description="Specific words or phrases indicating tone")
    relationship_status: Literal["new", "loyal", "at_risk", "detractor", "advocate"] = Field(
        description="Customer relationship status (new, loyal, at_risk, detractor, advocate)"
    )
    response_approach: Literal["empathetic", "professional", "solution_focused", "escalation_required"] = Field(
        description="Recommended response approach (empathetic, professional, solution_focused, escalation_required)"
    )


def customer_sentiment(business_context: str = "general customer support", **api_kwargs) -> PreparedTask:
    """Create a configurable customer sentiment analysis task.

    Args:
        business_context (str): Business context for sentiment analysis.
        **api_kwargs: Additional OpenAI API parameters (temperature, top_p, etc.).

    Returns:
        PreparedTask configured for customer sentiment analysis.
    """

    instructions = f"""Analyze customer sentiment in the context of support interactions, focusing on
satisfaction, emotional state, and business implications.

Business Context: {business_context}

Sentiment Categories:
- positive: Customer is happy, satisfied, or grateful
- negative: Customer is unhappy, frustrated, or disappointed
- neutral: Customer is matter-of-fact, without strong emotions
- mixed: Customer expresses both positive and negative sentiments

Satisfaction Levels:
- very_satisfied: Extremely happy, praising service, expressing gratitude
- satisfied: Content, appreciative, positive feedback
- neutral: Neither satisfied nor dissatisfied, factual communication
- dissatisfied: Unhappy, expressing concerns, mild complaints
- very_dissatisfied: Extremely unhappy, angry, threatening to leave

Emotional States:
- happy: Cheerful, pleased, content
- frustrated: Annoyed, impatient, struggling with issues
- angry: Hostile, aggressive, demanding immediate action
- disappointed: Let down, expectations not met
- confused: Lost, needing clarification, overwhelmed
- grateful: Thankful, appreciative of help received
- worried: Anxious, concerned about outcomes

Churn Risk Assessment:
- low: Happy customers, positive experience
- medium: Neutral customers, some concerns but manageable
- high: Dissatisfied customers, multiple issues, expressing frustration
- critical: Extremely unhappy, threatening to cancel, demanding escalation

Relationship Status:
- new: First-time contact, tentative, learning
- loyal: Long-term customer, familiar with service
- at_risk: Showing signs of dissatisfaction, needs attention
- detractor: Actively unhappy, may spread negative feedback
- advocate: Extremely satisfied, promotes service to others

Response Approach:
- empathetic: Use compassionate language, acknowledge feelings
- professional: Maintain formal, solution-oriented communication
- solution_focused: Directly address problems, provide clear next steps
- escalation_required: Immediately involve management or specialists

Analyze tone indicators like:
- Positive: "thank you", "great", "helpful", "love", "excellent"
- Negative: "terrible", "disappointed", "frustrated", "awful", "horrible"
- Urgency: "urgent", "immediately", "ASAP", "critical"
- Threat: "cancel", "switch", "competitor", "lawyer", "report"

IMPORTANT: Provide analysis responses in the same language as the input text, except for the
predefined categorical fields (sentiment, satisfaction_level, emotional_state, churn_risk,
relationship_status, response_approach) which must use the exact English values specified above.
For example, if the input is in Spanish, provide tone_indicators in Spanish, but use English
values like "positive" for sentiment.

Provide comprehensive sentiment analysis with business context and recommended response strategy."""

    return PreparedTask(instructions=instructions, response_format=CustomerSentiment, api_kwargs=api_kwargs)


# Backward compatibility - default configuration
CUSTOMER_SENTIMENT = customer_sentiment(temperature=0.0, top_p=1.0)
