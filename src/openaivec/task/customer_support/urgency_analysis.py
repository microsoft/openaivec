"""Urgency analysis task for customer support.

This module provides a configurable task for analyzing the urgency level of customer
inquiries to help prioritize support queue and response times.

Example:
    Basic usage with default settings:

    ```python
    from openai import OpenAI
    from openaivec._responses import BatchResponses
    from openaivec.task import customer_support

    client = OpenAI()
    analyzer = BatchResponses.of_task(
        client=client,
        model_name="gpt-4.1-mini",
        task=customer_support.urgency_analysis()
    )

    inquiries = [
        "URGENT: My website is down and I'm losing customers!",
        "Can you help me understand how to use the new feature?",
        "I haven't received my order from last week"
    ]
    analyses = analyzer.parse(inquiries)

    for analysis in analyses:
        print(f"Urgency Level: {analysis.urgency_level}")
        print(f"Score: {analysis.urgency_score}")
        print(f"Response Time: {analysis.response_time}")
        print(f"Escalation: {analysis.escalation_required}")
    ```

    Customized for SaaS platform with business hours:

    ```python
    from openaivec.task import customer_support

    # SaaS-specific urgency levels
    saas_urgency_levels = {
        "critical": "Service outages, security breaches, data loss",
        "high": "Login issues, payment failures, API errors",
        "medium": "Feature bugs, performance issues, billing questions",
        "low": "Feature requests, documentation questions, general feedback"
    }

    # Custom response times based on SLA
    saas_response_times = {
        "critical": "immediate",
        "high": "within_1_hour",
        "medium": "within_4_hours",
        "low": "within_24_hours"
    }

    # Enterprise customer tier gets priority
    enterprise_customer_tiers = {
        "enterprise": "Priority support, dedicated account manager",
        "business": "Standard business support",
        "professional": "Professional plan support",
        "starter": "Basic support"
    }

    task = customer_support.urgency_analysis(
        urgency_levels=saas_urgency_levels,
        response_times=saas_response_times,
        customer_tiers=enterprise_customer_tiers,
        business_context="SaaS platform",
        business_hours="9 AM - 5 PM EST, Monday-Friday"
    )

    analyzer = BatchResponses.of_task(
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
        "URGENT: My website is down and I'm losing customers!",
        "Can you help me understand how to use the new feature?",
        "I haven't received my order from last week"
    ]})
    df["urgency"] = df["inquiry"].ai.task(customer_support.urgency_analysis())

    # Extract urgency components
    extracted_df = df.ai.extract("urgency")
    print(extracted_df[["inquiry", "urgency_urgency_level", "urgency_urgency_score", "urgency_response_time"]])
    ```
"""

from typing import Dict, Literal

from pydantic import BaseModel, Field

from openaivec._model import PreparedTask

__all__ = ["urgency_analysis"]


class UrgencyAnalysis(BaseModel):
    urgency_level: Literal["critical", "high", "medium", "low"] = Field(
        description="Urgency level from configured levels (critical, high, medium, low)"
    )
    urgency_score: float = Field(description="Urgency score from 0.0 (not urgent) to 1.0 (extremely urgent)")
    response_time: Literal["immediate", "within_1_hour", "within_4_hours", "within_24_hours"] = Field(
        description="Recommended response time from configured times "
        "(immediate, within_1_hour, within_4_hours, within_24_hours)"
    )
    escalation_required: bool = Field(description="Whether this inquiry requires escalation to management")
    urgency_indicators: list[str] = Field(description="Specific words or phrases that indicate urgency")
    business_impact: Literal["none", "low", "medium", "high", "critical"] = Field(
        description="Potential business impact (none, low, medium, high, critical)"
    )
    customer_tier: Literal["enterprise", "premium", "standard", "basic"] = Field(
        description="Inferred customer tier from configured tiers (enterprise, premium, standard, basic)"
    )
    reasoning: str = Field(description="Brief explanation of urgency assessment")
    sla_compliance: bool = Field(description="Whether response time aligns with SLA requirements")


def urgency_analysis(
    urgency_levels: Dict[str, str] | None = None,
    response_times: Dict[str, str] | None = None,
    customer_tiers: Dict[str, str] | None = None,
    escalation_rules: Dict[str, str] | None = None,
    urgency_keywords: Dict[str, list[str]] | None = None,
    business_context: str = "general customer support",
    business_hours: str = "24/7 support",
    sla_rules: Dict[str, str] | None = None,
    **api_kwargs,
) -> PreparedTask:
    """Create a configurable urgency analysis task.

    Args:
        urgency_levels (dict[str, str] | None): Dictionary mapping urgency levels to descriptions.
        response_times (dict[str, str] | None): Dictionary mapping urgency levels to response times.
        customer_tiers (dict[str, str] | None): Dictionary mapping tier names to descriptions.
        escalation_rules (dict[str, str] | None): Dictionary mapping conditions to escalation actions.
        urgency_keywords (dict[str, list[str]] | None): Dictionary mapping urgency levels to indicator keywords.
        business_context (str): Description of the business context.
        business_hours (str): Description of business hours for response time calculation.
        sla_rules (dict[str, str] | None): Dictionary mapping customer tiers to SLA requirements.
        **api_kwargs: Additional keyword arguments to pass to the OpenAI API,
            such as temperature, top_p, etc.

    Returns:
        PreparedTask configured for urgency analysis.
    """

    # Default urgency levels
    if urgency_levels is None:
        urgency_levels = {
            "critical": "Service outages, security breaches, data loss, system failures affecting business operations",
            "high": "Account locked, payment failures, urgent deadlines, angry customers, revenue-impacting issues",
            "medium": "Feature not working, delivery delays, billing questions, moderate customer frustration",
            "low": "General questions, feature requests, feedback, compliments, minor issues",
        }

    # Default response times
    if response_times is None:
        response_times = {
            "critical": "immediate",
            "high": "within_1_hour",
            "medium": "within_4_hours",
            "low": "within_24_hours",
        }

    # Default customer tiers
    if customer_tiers is None:
        customer_tiers = {
            "enterprise": "Large contracts, multiple users, business-critical usage",
            "premium": "Paid plans, professional use, higher expectations",
            "standard": "Regular paid users, normal expectations",
            "basic": "Free users, casual usage, lower priority",
        }

    # Default escalation rules
    if escalation_rules is None:
        escalation_rules = {
            "immediate": "Critical issues, security breaches, service outages",
            "within_1_hour": "High urgency with customer tier enterprise or premium",
            "manager_review": "Threats to cancel, legal language, compliance issues",
            "no_escalation": "Standard support can handle",
        }

    # Default urgency keywords
    if urgency_keywords is None:
        urgency_keywords = {
            "critical": ["urgent", "emergency", "critical", "down", "outage", "security", "breach", "immediate"],
            "high": ["ASAP", "urgent", "problem", "issue", "error", "bug", "frustrated", "angry"],
            "medium": ["question", "help", "support", "feedback", "concern", "delayed"],
            "low": ["information", "thank", "compliment", "suggestion", "general", "when convenient"],
        }

    # Default SLA rules
    if sla_rules is None:
        sla_rules = {
            "enterprise": "Critical: 15min, High: 1hr, Medium: 4hr, Low: 24hr",
            "premium": "Critical: 30min, High: 2hr, Medium: 8hr, Low: 48hr",
            "standard": "Critical: 1hr, High: 4hr, Medium: 24hr, Low: 72hr",
            "basic": "Critical: 4hr, High: 24hr, Medium: 72hr, Low: 1week",
        }

    # Build urgency levels section
    urgency_text = "Urgency Levels:\n"
    for level, description in urgency_levels.items():
        urgency_text += f"- {level}: {description}\n"

    # Build response times section
    response_text = "Response Times:\n"
    for level, time in response_times.items():
        response_text += f"- {level}: {time}\n"

    # Build customer tiers section
    tiers_text = "Customer Tiers:\n"
    for tier, description in customer_tiers.items():
        tiers_text += f"- {tier}: {description}\n"

    # Build escalation rules section
    escalation_text = "Escalation Rules:\n"
    for condition, action in escalation_rules.items():
        escalation_text += f"- {condition}: {action}\n"

    # Build urgency keywords section
    keywords_text = "Urgency Keywords:\n"
    for level, keywords in urgency_keywords.items():
        keywords_text += f"- {level}: {', '.join(keywords)}\n"

    # Build SLA rules section
    sla_text = "SLA Rules:\n"
    for tier, sla in sla_rules.items():
        sla_text += f"- {tier}: {sla}\n"

    instructions = f"""Analyze the urgency level of the customer inquiry based on language, content, and context.

Business Context: {business_context}
Business Hours: {business_hours}

{urgency_text}

{response_text}

{tiers_text}

{escalation_text}

{keywords_text}

{sla_text}

Instructions:
1. Analyze the inquiry in the context of: {business_context}
2. Identify urgency indicators in the language and content
3. Classify into the appropriate urgency level
4. Calculate urgency score (0.0-1.0) based on multiple factors
5. Recommend response time based on urgency level and customer tier
6. Determine if escalation is required based on configured rules
7. Assess potential business impact
8. Infer customer tier from language and context
9. Provide reasoning for the urgency assessment
10. Check SLA compliance with recommended response time

Consider:
- Explicit urgency language and keywords
- Emotional tone and intensity
- Business impact indicators
- Time pressure and deadlines
- Customer tier indicators
- Previous escalation language
- Revenue or operational impact
- Compliance or legal implications

IMPORTANT: Provide analysis responses in the same language as the input text, except for the
predefined categorical fields (urgency_level, response_time, business_impact, customer_tier)
which must use the exact English values specified above. For example, if the input is in French,
provide urgency_indicators and reasoning in French, but use English values like "critical" for
urgency_level.

Provide detailed analysis with clear reasoning for urgency level and response time recommendations."""

    return PreparedTask(instructions=instructions, response_format=UrgencyAnalysis, api_kwargs=api_kwargs)


# Backward compatibility - default configuration
URGENCY_ANALYSIS = urgency_analysis(temperature=0.0, top_p=1.0)
