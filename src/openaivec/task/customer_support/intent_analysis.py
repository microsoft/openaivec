"""Customer inquiry intent-analysis task definition."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from openaivec._model import PreparedTask
from openaivec.task._prompt_templates import english_categorical_policy, join_sections, same_language_policy
from openaivec.task._registry import TaskSpec

__all__ = ["intent_analysis"]


class IntentAnalysis(BaseModel):
    """Intent analysis output."""

    model_config = ConfigDict(extra="forbid")

    primary_intent: Literal[
        "get_help",
        "make_purchase",
        "cancel_service",
        "get_refund",
        "report_issue",
        "seek_information",
        "request_feature",
        "provide_feedback",
    ] = Field(description="Primary customer intent")
    secondary_intents: list[str] = Field(description="Additional intents if multiple goals are present")
    action_required: Literal[
        "provide_information",
        "troubleshoot",
        "process_request",
        "escalate",
        "redirect",
        "schedule_callback",
    ] = Field(description="Required action")
    intent_confidence: float = Field(ge=0, le=1, description="Confidence in intent detection (0.0-1.0)")
    success_likelihood: Literal["very_high", "high", "medium", "low", "very_low"] = Field(
        description="Likelihood of successful resolution"
    )
    customer_goal: str = Field(description="What the customer ultimately wants to achieve")
    implicit_needs: list[str] = Field(description="Unstated needs or concerns")
    blocking_factors: list[str] = Field(description="Potential obstacles to achieving customer goal")
    next_steps: list[str] = Field(description="Recommended next steps")
    resolution_complexity: Literal["simple", "moderate", "complex", "very_complex"] = Field(
        description="Complexity of resolution"
    )


def _build_instructions(business_context: str) -> str:
    return join_sections(
        "Analyze customer intent in the inquiry and identify how support should respond.",
        f"Business context: {business_context}",
        (
            "Classify primary intent, action required, success likelihood, and resolution complexity. "
            "Also provide concise customer goal, implicit needs, blockers, and next steps."
        ),
        same_language_policy(),
        english_categorical_policy(
            ["primary_intent", "action_required", "success_likelihood", "resolution_complexity"]
        ),
    )


def intent_analysis(business_context: str = "general customer support") -> PreparedTask[IntentAnalysis]:
    """Create a customer intent analysis task."""
    return PreparedTask(
        instructions=_build_instructions(business_context=business_context),
        response_format=IntentAnalysis,
    )


TASK_SPEC = TaskSpec(
    key="customer_support.intent_analysis",
    domain="customer_support",
    summary="Classify customer intent and recommended support actions.",
    factory=intent_analysis,
    response_format=IntentAnalysis,
)

