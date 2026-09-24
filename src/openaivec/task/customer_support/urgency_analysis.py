"""Customer inquiry urgency-analysis task definition."""

from collections.abc import Mapping
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, create_model

from openaivec._model import PreparedTask
from openaivec.task._prompt_templates import (
    english_categorical_policy,
    grouped_mapping_section,
    join_sections,
    mapping_section,
    same_language_policy,
)
from openaivec.task._registry import TaskSpec

__all__ = ["urgency_analysis"]


class UrgencyAnalysis(BaseModel):
    """Urgency analysis output."""

    model_config = ConfigDict(extra="forbid")

    urgency_level: Literal["critical", "high", "medium", "low"] = Field(description="Urgency level")
    urgency_score: float = Field(ge=0, le=1, description="Urgency score from 0.0 to 1.0")
    response_time: Literal["immediate", "within_1_hour", "within_4_hours", "within_24_hours"] = Field(
        description="Recommended response time"
    )
    escalation_required: bool = Field(description="Whether this inquiry requires escalation")
    urgency_indicators: list[str] = Field(description="Words or phrases indicating urgency")
    business_impact: Literal["none", "low", "medium", "high", "critical"] = Field(
        description="Potential business impact"
    )
    customer_tier: Literal["enterprise", "premium", "standard", "basic"] = Field(description="Inferred customer tier")
    reasoning: str = Field(description="Brief explanation of urgency assessment")
    sla_compliance: bool = Field(description="Whether recommended response time aligns with SLA")


def _default_urgency_levels() -> dict[str, str]:
    return {
        "critical": "Service outages, security breaches, data loss, system failures",
        "high": "Account lock, payment failures, urgent deadlines, strong frustration",
        "medium": "Feature issue, delivery delay, billing question, moderate frustration",
        "low": "General question, feature request, feedback, minor issue",
    }


def _default_response_times() -> dict[str, str]:
    return {
        "critical": "immediate",
        "high": "within_1_hour",
        "medium": "within_4_hours",
        "low": "within_24_hours",
    }


def _default_customer_tiers() -> dict[str, str]:
    return {
        "enterprise": "Large contracts and business-critical usage",
        "premium": "Paid plans with higher expectations",
        "standard": "Regular paid users",
        "basic": "Free or casual users",
    }


def _default_escalation_rules() -> dict[str, str]:
    return {
        "immediate": "Critical issues, security breaches, outages",
        "within_1_hour": "High urgency with enterprise or premium tier",
        "manager_review": "Cancellation threats, legal or compliance language",
        "no_escalation": "Standard support can handle",
    }


def _default_urgency_keywords() -> dict[str, list[str]]:
    return {
        "critical": ["urgent", "emergency", "critical", "down", "outage", "security", "breach", "immediate"],
        "high": ["ASAP", "urgent", "problem", "issue", "error", "bug", "frustrated", "angry"],
        "medium": ["question", "help", "support", "feedback", "concern", "delayed"],
        "low": ["information", "thank", "compliment", "suggestion", "general", "when convenient"],
    }


def _default_sla_rules() -> dict[str, str]:
    return {
        "enterprise": "Critical: 15min, High: 1hr, Medium: 4hr, Low: 24hr",
        "premium": "Critical: 30min, High: 2hr, Medium: 8hr, Low: 48hr",
        "standard": "Critical: 1hr, High: 4hr, Medium: 24hr, Low: 72hr",
        "basic": "Critical: 4hr, High: 24hr, Medium: 72hr, Low: 1week",
    }


def _choices(name: str, options: Mapping[str, str]) -> tuple[str, ...]:
    choices = tuple(options)
    if not choices or any(not isinstance(choice, str) or not choice.strip() for choice in choices):
        raise ValueError(f"{name} must contain nonempty choices")
    return choices


def _build_instructions(
    urgency_levels: Mapping[str, str],
    response_times: Mapping[str, str],
    customer_tiers: Mapping[str, str],
    escalation_rules: Mapping[str, str],
    urgency_keywords: Mapping[str, list[str]],
    business_context: str,
    business_hours: str,
    sla_rules: Mapping[str, str],
) -> str:
    return join_sections(
        "Analyze urgency of the customer inquiry for support prioritization.",
        f"Business context: {business_context}",
        f"Business hours: {business_hours}",
        mapping_section("Urgency levels", urgency_levels),
        mapping_section("Response times", response_times),
        mapping_section("Customer tiers", customer_tiers),
        mapping_section("Escalation rules", escalation_rules),
        grouped_mapping_section("Urgency keywords", urgency_keywords),
        mapping_section("SLA rules", sla_rules),
        (
            "Return urgency level/score, response time, escalation flag, urgency indicators, business impact, "
            "customer tier, reasoning, and SLA compliance."
        ),
        same_language_policy(),
        english_categorical_policy(["urgency_level", "response_time", "business_impact", "customer_tier"]),
    )


def urgency_analysis(
    urgency_levels: Mapping[str, str] | None = None,
    response_times: Mapping[str, str] | None = None,
    customer_tiers: Mapping[str, str] | None = None,
    escalation_rules: Mapping[str, str] | None = None,
    urgency_keywords: Mapping[str, list[str]] | None = None,
    business_context: str = "general customer support",
    business_hours: str = "24/7 support",
    sla_rules: Mapping[str, str] | None = None,
) -> PreparedTask[UrgencyAnalysis]:
    """Create an urgency task with schema choices matching configured prompt options.

    Args:
        urgency_levels (Mapping[str, str] | None): Level names and descriptions.
        response_times (Mapping[str, str] | None): Level-to-response-time choices.
        customer_tiers (Mapping[str, str] | None): Tier names and descriptions.
        escalation_rules (Mapping[str, str] | None): Escalation guidance.
        urgency_keywords (Mapping[str, list[str]] | None): Keywords per level.
        business_context (str): Customer-support context.
        business_hours (str): Operating hours.
        sla_rules (Mapping[str, str] | None): SLA guidance by tier.

    Returns:
        PreparedTask[UrgencyAnalysis]: Configured task and response schema.

    Raises:
        ValueError: If a choice set is empty or response-time levels do not match urgency levels.
    """
    resolved_urgency_levels = dict(_default_urgency_levels() if urgency_levels is None else urgency_levels)
    resolved_response_times = dict(_default_response_times() if response_times is None else response_times)
    resolved_customer_tiers = dict(_default_customer_tiers() if customer_tiers is None else customer_tiers)
    levels = _choices("urgency_levels", resolved_urgency_levels)
    _choices("response_times", resolved_response_times)
    tiers = _choices("customer_tiers", resolved_customer_tiers)
    if set(resolved_response_times) != set(levels):
        raise ValueError("response_times must define exactly the configured urgency_levels")
    times = tuple(dict.fromkeys(resolved_response_times.values()))
    if any(not isinstance(value, str) or not value.strip() for value in times):
        raise ValueError("response_times must contain nonempty response-time values")
    default_escalation_rules = {
        key: value
        for key, value in _default_escalation_rules().items()
        if key not in _default_response_times().values() or key in times
    }
    resolved_escalation_rules = dict(default_escalation_rules if escalation_rules is None else escalation_rules)
    default_keywords = {key: value for key, value in _default_urgency_keywords().items() if key in levels}
    resolved_urgency_keywords = dict(default_keywords if urgency_keywords is None else urgency_keywords)
    default_sla_rules = {key: value for key, value in _default_sla_rules().items() if key in tiers}
    resolved_sla_rules = dict(default_sla_rules if sla_rules is None else sla_rules)
    response_format = UrgencyAnalysis
    if urgency_levels is not None or response_times is not None or customer_tiers is not None:
        response_format = create_model(
            "ConfiguredUrgencyAnalysis",
            __base__=UrgencyAnalysis,
            __module__=__name__,
            urgency_level=(getattr(Literal, "__getitem__")(levels), Field(description="Urgency level")),
            response_time=(getattr(Literal, "__getitem__")(times), Field(description="Recommended response time")),
            customer_tier=(getattr(Literal, "__getitem__")(tiers), Field(description="Inferred customer tier")),
        )
    return PreparedTask(
        instructions=_build_instructions(
            urgency_levels=resolved_urgency_levels,
            response_times=resolved_response_times,
            customer_tiers=resolved_customer_tiers,
            escalation_rules=resolved_escalation_rules,
            urgency_keywords=resolved_urgency_keywords,
            business_context=business_context,
            business_hours=business_hours,
            sla_rules=resolved_sla_rules,
        ),
        response_format=response_format,
    )


TASK_SPEC = TaskSpec(
    key="customer_support.urgency_analysis",
    domain="customer_support",
    summary="Assess inquiry urgency, escalation need, and SLA alignment.",
    factory=urgency_analysis,
    response_format=UrgencyAnalysis,
)
