"""Inquiry classification task for customer support.

This module provides a configurable task for classifying customer inquiries into
different categories to help route them to the appropriate support team.

Example:
    Basic usage with default settings:

    ```python
    from openai import OpenAI
    from openaivec._responses import BatchResponses
    from openaivec.task import customer_support

    client = OpenAI()
    classifier = BatchResponses.of_task(
        client=client,
        model_name="gpt-4.1-mini",
        task=customer_support.inquiry_classification()
    )

    inquiries = [
        "I can't log into my account",
        "When will my order arrive?",
        "I want to cancel my subscription"
    ]
    classifications = classifier.parse(inquiries)

    for classification in classifications:
        print(f"Category: {classification.category}")
        print(f"Subcategory: {classification.subcategory}")
        print(f"Confidence: {classification.confidence}")
        print(f"Routing: {classification.routing}")
    ```

    Customized for e-commerce:

    ```python
    from openaivec.task import customer_support

    # E-commerce specific categories
    ecommerce_categories = {
        "order_management": ["order_status", "order_cancellation", "order_modification", "returns"],
        "payment": ["payment_failed", "refund_request", "payment_methods", "billing_inquiry"],
        "product": ["product_info", "size_guide", "availability", "recommendations"],
        "shipping": ["delivery_status", "shipping_cost", "delivery_options", "tracking"],
        "account": ["login_issues", "account_settings", "profile_updates", "password_reset"],
        "general": ["complaints", "compliments", "feedback", "other"]
    }

    ecommerce_routing = {
        "order_management": "order_team",
        "payment": "billing_team",
        "product": "product_team",
        "shipping": "logistics_team",
        "account": "account_support",
        "general": "general_support"
    }

    task = customer_support.inquiry_classification(
        categories=ecommerce_categories,
        routing_rules=ecommerce_routing,
        business_context="e-commerce platform"
    )

    classifier = BatchResponses.of_task(
        client=client,
        model_name="gpt-4.1-mini",
        task=task
    )
    ```

    With pandas integration:

    ```python
    import pandas as pd
    from openaivec import pandas_ext  # Required for .ai accessor
    from openaivec.task import customer_support

    df = pd.DataFrame({"inquiry": [
        "I can't log into my account",
        "When will my order arrive?",
        "I want to cancel my subscription"
    ]})
    df["classification"] = df["inquiry"].ai.task(customer_support.inquiry_classification())

    # Extract classification components
    extracted_df = df.ai.extract("classification")
    print(extracted_df[[
        "inquiry", "classification_category",
        "classification_subcategory", "classification_confidence"
    ]])
    ```
"""

from typing import Dict, Literal

from pydantic import BaseModel, Field

from openaivec._model import PreparedTask

__all__ = ["inquiry_classification"]


class InquiryClassification(BaseModel):
    category: str = Field(description="Primary category from the configured categories")
    subcategory: str = Field(description="Specific subcategory within the primary category")
    confidence: float = Field(description="Confidence score for classification (0.0-1.0)")
    routing: str = Field(description="Recommended routing destination")
    keywords: list[str] = Field(description="Key terms that influenced the classification")
    priority: Literal["low", "medium", "high", "urgent"] = Field(
        description="Suggested priority level (low, medium, high, urgent)"
    )
    business_context_match: bool = Field(description="Whether the inquiry matches the business context")


def inquiry_classification(
    categories: Dict[str, list[str]] | None = None,
    routing_rules: Dict[str, str] | None = None,
    priority_rules: Dict[str, str] | None = None,
    business_context: str = "general customer support",
    custom_keywords: Dict[str, list[str]] | None = None,
    **api_kwargs,
) -> PreparedTask:
    """Create a configurable inquiry classification task.

    Args:
        categories (dict[str, list[str]] | None): Dictionary mapping category names to lists of subcategories.
            Default provides standard support categories.
        routing_rules (dict[str, str] | None): Dictionary mapping categories to routing destinations.
            Default provides standard routing options.
        priority_rules (dict[str, str] | None): Dictionary mapping keywords/patterns to priority levels.
            Default uses standard priority indicators.
        business_context (str): Description of the business context to help with classification.
        custom_keywords (dict[str, list[str]] | None): Dictionary mapping categories to relevant keywords.
        **api_kwargs: Additional keyword arguments to pass to the OpenAI API,
            such as temperature, top_p, etc.

    Returns:
        PreparedTask configured for inquiry classification.
    """

    # Default categories
    if categories is None:
        categories = {
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

    # Default routing rules
    if routing_rules is None:
        routing_rules = {
            "technical": "tech_support",
            "billing": "billing_team",
            "product": "product_team",
            "shipping": "shipping_team",
            "account": "account_management",
            "general": "general_support",
        }

    # Default priority rules
    if priority_rules is None:
        priority_rules = {
            "urgent": "urgent, emergency, critical, down, outage, security, breach, immediate",
            "high": "login, password, payment, billing, delivery, problem, issue, error, bug",
            "medium": "feature, request, question, how, help, support, feedback",
            "low": "information, compliment, thank, suggestion, general, other",
        }

    # Build categories section
    categories_text = "Categories and subcategories:\n"
    for category, subcategories in categories.items():
        categories_text += f"- {category}: {', '.join(subcategories)}\n"

    # Build routing section
    routing_text = "Routing options:\n"
    for category, routing in routing_rules.items():
        routing_text += f"- {routing}: {category.replace('_', ' ').title()} issues\n"

    # Build priority section
    priority_text = "Priority levels:\n"
    for priority, keywords in priority_rules.items():
        priority_text += f"- {priority}: {keywords}\n"

    # Build custom keywords section
    keywords_text = ""
    if custom_keywords:
        keywords_text = "\nCustom keywords for classification:\n"
        for category, keywords in custom_keywords.items():
            keywords_text += f"- {category}: {', '.join(keywords)}\n"

    instructions = f"""Classify the customer inquiry into the appropriate category and subcategory
based on the configured categories and business context.

Business Context: {business_context}

{categories_text}

{routing_text}

{priority_text}

{keywords_text}

Instructions:
1. Analyze the inquiry in the context of: {business_context}
2. Classify into the most appropriate category and subcategory
3. Provide confidence score based on clarity of the inquiry
4. Suggest routing based on the configured rules
5. Extract relevant keywords that influenced the classification
6. Assign priority level based on content and urgency indicators
7. Indicate whether the inquiry matches the business context

Consider:
- Explicit keywords and phrases
- Implied intent and context
- Emotional tone and urgency
- Technical complexity
- Business impact
- Customer type indicators

IMPORTANT: Provide analysis responses in the same language as the input text, except for the
predefined categorical fields (priority) which must use the exact English values specified above.
Category, subcategory, routing, and keywords should reflect the content and can be in the input
language where appropriate, but priority must use English values like "high".

Provide accurate classification with detailed reasoning."""

    return PreparedTask(instructions=instructions, response_format=InquiryClassification, api_kwargs=api_kwargs)


# Backward compatibility - default configuration
INQUIRY_CLASSIFICATION = inquiry_classification(temperature=0.0, top_p=1.0)
