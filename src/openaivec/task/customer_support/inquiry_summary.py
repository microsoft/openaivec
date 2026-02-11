"""Customer inquiry-summary task definition."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from openaivec._model import PreparedTask
from openaivec.task._prompt_templates import english_categorical_policy, join_sections, same_language_policy
from openaivec.task._registry import TaskSpec

__all__ = ["inquiry_summary"]


class InquirySummary(BaseModel):
    """Inquiry summary output."""

    model_config = ConfigDict(extra="forbid")

    summary: str = Field(description="Concise summary of the customer inquiry")
    main_issue: str = Field(description="Primary problem or request being addressed")
    secondary_issues: list[str] = Field(description="Additional issues mentioned in the inquiry")
    customer_background: str = Field(description="Relevant customer context or history")
    actions_taken: list[str] = Field(description="Steps the customer has already attempted")
    timeline: str = Field(description="Timeline of events or when the issue started")
    impact_description: str = Field(description="How the issue affects the customer")
    resolution_status: Literal["not_started", "in_progress", "needs_escalation", "resolved"] = Field(
        description="Current status"
    )
    key_details: list[str] = Field(description="Important technical details and specifics")
    follow_up_needed: bool = Field(description="Whether follow-up communication is required")
    summary_confidence: float = Field(ge=0, le=1, description="Confidence in summary accuracy (0.0-1.0)")


def _build_instructions(summary_length: str, business_context: str) -> str:
    length_instruction_map = {
        "concise": "Write a concise 2-3 sentence summary.",
        "detailed": "Write a detailed 4-6 sentence summary.",
        "bullet_points": "Write summary in bullet-point style.",
    }
    style_instruction = length_instruction_map.get(summary_length, length_instruction_map["concise"])
    return join_sections(
        "Summarize the customer inquiry for support handoff and reporting.",
        f"Business context: {business_context}",
        style_instruction,
        (
            "Capture main issue, secondary issues, customer context, attempted actions, timeline, impact, "
            "resolution status, key details, and follow-up requirement."
        ),
        same_language_policy(),
        english_categorical_policy(["resolution_status"]),
    )


def inquiry_summary(
    summary_length: str = "concise",
    business_context: str = "general customer support",
) -> PreparedTask[InquirySummary]:
    """Create an inquiry summary task."""
    return PreparedTask(
        instructions=_build_instructions(summary_length=summary_length, business_context=business_context),
        response_format=InquirySummary,
    )


TASK_SPEC = TaskSpec(
    key="customer_support.inquiry_summary",
    domain="customer_support",
    summary="Summarize inquiries into structured support handoff fields.",
    factory=inquiry_summary,
    response_format=InquirySummary,
)

