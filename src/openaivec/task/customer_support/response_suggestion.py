"""Customer response-suggestion task definition."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from openaivec._model import PreparedTask
from openaivec.task._prompt_templates import english_categorical_policy, join_sections, same_language_policy
from openaivec.task._registry import TaskSpec

__all__ = ["response_suggestion"]


class ResponseSuggestion(BaseModel):
    """Response suggestion output."""

    model_config = ConfigDict(extra="forbid")

    suggested_response: str = Field(description="Professional response draft for the customer inquiry")
    tone: Literal["empathetic", "professional", "friendly", "apologetic", "solution_focused"] = Field(
        description="Recommended tone"
    )
    priority: Literal["immediate", "high", "medium", "low"] = Field(description="Response priority")
    response_type: Literal["acknowledgment", "solution", "escalation", "information_request", "closure"] = Field(
        description="Type of response"
    )
    key_points: list[str] = Field(description="Main points that must be addressed")
    follow_up_required: bool = Field(description="Whether follow-up communication is needed")
    escalation_suggested: bool = Field(description="Whether escalation to management is recommended")
    resources_needed: list[str] = Field(description="Additional resources or information required")
    estimated_resolution_time: Literal["immediate", "hours", "days", "weeks"] = Field(
        description="Estimated time to resolution"
    )
    alternative_responses: list[str] = Field(description="Alternative response options")
    personalization_notes: str = Field(description="Suggestions for personalizing the response")


def _build_instructions(response_style: str, company_name: str, business_context: str) -> str:
    style_guidance = {
        "professional": "Maintain professional tone with clear, direct communication.",
        "friendly": "Use warm, approachable language while staying professional.",
        "empathetic": "Show clear understanding and compassion for customer concerns.",
        "formal": "Use formal business language appropriate for official communications.",
    }
    return join_sections(
        "Generate a helpful response suggestion for the customer inquiry.",
        f"Business context: {business_context}",
        f"Company name for context: {company_name}",
        style_guidance.get(response_style, style_guidance["professional"]),
        (
            "Provide suggested response, tone, priority, response type, key points, follow-up/escalation flags, "
            "resources needed, estimated resolution time, alternatives, and personalization notes."
        ),
        same_language_policy(),
        english_categorical_policy(["tone", "priority", "response_type", "estimated_resolution_time"]),
    )


def response_suggestion(
    response_style: str = "professional",
    company_name: str = "our company",
    business_context: str = "general customer support",
) -> PreparedTask[ResponseSuggestion]:
    """Create a response suggestion task."""
    return PreparedTask(
        instructions=_build_instructions(
            response_style=response_style,
            company_name=company_name,
            business_context=business_context,
        ),
        response_format=ResponseSuggestion,
    )


TASK_SPEC = TaskSpec(
    key="customer_support.response_suggestion",
    domain="customer_support",
    summary="Generate structured customer-support response drafts.",
    factory=response_suggestion,
    response_format=ResponseSuggestion,
)

