"""Customer inquiry-classification task definition."""

from collections.abc import Mapping
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from openaivec._model import PreparedTask
from openaivec.task._prompt_templates import (
    english_categorical_policy,
    grouped_mapping_section,
    join_sections,
    mapping_section,
    same_language_policy,
)
from openaivec.task._registry import TaskSpec

__all__ = ["inquiry_classification"]


class InquiryClassification(BaseModel):
    """Inquiry classification output."""

    model_config = ConfigDict(extra="forbid")

    category: str = Field(description="Primary category from configured categories")
    subcategory: str = Field(description="Specific subcategory within the primary category")
    confidence: float = Field(ge=0, le=1, description="Confidence score for classification (0.0-1.0)")
    routing: str = Field(description="Recommended routing destination")
    keywords: list[str] = Field(description="Key terms that influenced the classification")
    priority: Literal["low", "medium", "high", "urgent"] = Field(description="Suggested priority level")
    business_context_match: bool = Field(description="Whether the inquiry matches the business context")


def _default_categories() -> dict[str, list[str]]:
    return {
        "technical": [
            "login_issues",
            "password_reset",
            "app_crashes",
            "connectivity_problems",
            "feature_not_working",
        ],
        "billing": [
            "payment_failed",
            "invoice_questions",
            "refund_request",
            "pricing_inquiry",
            "subscription_changes",
        ],
        "product": [
            "feature_request",
            "product_information",
            "compatibility_questions",
            "how_to_use",
            "bug_reports",
        ],
        "shipping": [
            "delivery_status",
            "shipping_address",
            "delivery_issues",
            "tracking_number",
            "expedited_shipping",
        ],
        "account": ["account_creation", "profile_updates", "account_deletion", "data_export", "privacy_settings"],
        "general": ["compliments", "complaints", "feedback", "partnership_inquiry", "other"],
    }


def _default_routing_rules() -> dict[str, str]:
    return {
        "technical": "tech_support",
        "billing": "billing_team",
        "product": "product_team",
        "shipping": "shipping_team",
        "account": "account_management",
        "general": "general_support",
    }


def _default_priority_rules() -> dict[str, str]:
    return {
        "urgent": "urgent, emergency, critical, down, outage, security, breach, immediate",
        "high": "login, password, payment, billing, delivery, problem, issue, error, bug",
        "medium": "feature, request, question, how, help, support, feedback",
        "low": "information, compliment, thank, suggestion, general, other",
    }


def _build_instructions(
    categories: Mapping[str, list[str]],
    routing_rules: Mapping[str, str],
    priority_rules: Mapping[str, str],
    business_context: str,
    custom_keywords: Mapping[str, list[str]] | None,
) -> str:
    keywords_section = (
        grouped_mapping_section("Custom keywords for classification", custom_keywords) if custom_keywords else ""
    )
    return join_sections(
        "Classify the customer inquiry into category and subcategory for support routing.",
        f"Business context: {business_context}",
        grouped_mapping_section("Categories and subcategories", categories),
        mapping_section("Routing options", routing_rules),
        mapping_section("Priority keyword hints", priority_rules),
        keywords_section,
        (
            "Return category, subcategory, routing, priority, confidence, keywords, and "
            "whether the inquiry matches business context."
        ),
        same_language_policy(),
        english_categorical_policy(["priority"]),
    )


def inquiry_classification(
    categories: Mapping[str, list[str]] | None = None,
    routing_rules: Mapping[str, str] | None = None,
    priority_rules: Mapping[str, str] | None = None,
    business_context: str = "general customer support",
    custom_keywords: Mapping[str, list[str]] | None = None,
) -> PreparedTask[InquiryClassification]:
    """Create an inquiry classification task."""
    resolved_categories = dict(categories or _default_categories())
    resolved_routing_rules = dict(routing_rules or _default_routing_rules())
    resolved_priority_rules = dict(priority_rules or _default_priority_rules())
    return PreparedTask(
        instructions=_build_instructions(
            categories=resolved_categories,
            routing_rules=resolved_routing_rules,
            priority_rules=resolved_priority_rules,
            business_context=business_context,
            custom_keywords=custom_keywords,
        ),
        response_format=InquiryClassification,
    )


TASK_SPEC = TaskSpec(
    key="customer_support.inquiry_classification",
    domain="customer_support",
    summary="Classify support inquiries into categories and routing destinations.",
    factory=inquiry_classification,
    response_format=InquiryClassification,
)

