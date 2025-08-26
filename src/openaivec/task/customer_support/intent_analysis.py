"""Intent analysis task for customer support interactions.

This module provides a predefined task for analyzing customer intent to understand
what the customer is trying to achieve and how to best assist them.

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
        task=customer_support.INTENT_ANALYSIS
    )

    inquiries = [
        "I want to upgrade my plan to get more storage",
        "How do I delete my account? I'm not satisfied with the service",
        "Can you walk me through setting up the mobile app?"
    ]
    intents = analyzer.parse(inquiries)

    for intent in intents:
        print(f"Primary Intent: {intent.primary_intent}")
        print(f"Action Required: {intent.action_required}")
        print(f"Success Likelihood: {intent.success_likelihood}")
        print(f"Next Steps: {intent.next_steps}")
    ```

    With pandas integration:

    ```python
    import pandas as pd
    from openaivec import pandas_ext  # Required for .ai accessor
    from openaivec.task import customer_support

    df = pd.DataFrame({"inquiry": [
        "I want to upgrade my plan to get more storage",
        "How do I delete my account? I'm not satisfied with the service",
        "Can you walk me through setting up the mobile app?"
    ]})
    df["intent"] = df["inquiry"].ai.task(customer_support.INTENT_ANALYSIS)

    # Extract intent components
    extracted_df = df.ai.extract("intent")
    print(extracted_df[["inquiry", "intent_primary_intent", "intent_action_required", "intent_success_likelihood"]])
    ```

Attributes:
    INTENT_ANALYSIS (PreparedTask): A prepared task instance
        configured for intent analysis with temperature=0.0 and
        top_p=1.0 for deterministic output.
"""

from typing import Literal

from pydantic import BaseModel, Field

from openaivec._model import PreparedTask

__all__ = ["intent_analysis"]


class IntentAnalysis(BaseModel):
    primary_intent: Literal[
        "get_help",
        "make_purchase",
        "cancel_service",
        "get_refund",
        "report_issue",
        "seek_information",
        "request_feature",
        "provide_feedback",
    ] = Field(
        description="Primary customer intent (get_help, make_purchase, cancel_service, "
        "get_refund, report_issue, seek_information, request_feature, provide_feedback)"
    )
    secondary_intents: list[str] = Field(description="Additional intents if multiple goals are present")
    action_required: Literal[
        "provide_information", "troubleshoot", "process_request", "escalate", "redirect", "schedule_callback"
    ] = Field(
        description="Required action (provide_information, troubleshoot, process_request, "
        "escalate, redirect, schedule_callback)"
    )
    intent_confidence: float = Field(description="Confidence in intent detection (0.0-1.0)")
    success_likelihood: Literal["very_high", "high", "medium", "low", "very_low"] = Field(
        description="Likelihood of successful resolution (very_high, high, medium, low, very_low)"
    )
    customer_goal: str = Field(description="What the customer ultimately wants to achieve")
    implicit_needs: list[str] = Field(description="Unstated needs or concerns that may need addressing")
    blocking_factors: list[str] = Field(description="Potential obstacles to achieving customer goal")
    next_steps: list[str] = Field(description="Recommended next steps to address customer intent")
    resolution_complexity: Literal["simple", "moderate", "complex", "very_complex"] = Field(
        description="Complexity of resolution (simple, moderate, complex, very_complex)"
    )


def intent_analysis(business_context: str = "general customer support", **api_kwargs) -> PreparedTask:
    """Create a configurable intent analysis task.

    Args:
        business_context (str): Business context for intent analysis.
        **api_kwargs: Additional keyword arguments to pass to the OpenAI API,
            such as temperature, top_p, etc.

    Returns:
        PreparedTask configured for intent analysis.
    """

    instructions = f"""Analyze customer intent to understand their goals, needs, and how to best assist them.

Business Context: {business_context}

Primary Intent Categories:
- get_help: Seeking assistance with existing product or service
- make_purchase: Interested in buying or upgrading service
- cancel_service: Wants to terminate subscription or service
- get_refund: Seeking monetary reimbursement
- report_issue: Reporting problems or bugs
- seek_information: Looking for details about products, policies, or procedures
- request_feature: Asking for new functionality or improvements
- provide_feedback: Sharing opinions, suggestions, or experiences

Action Required:
- provide_information: Share knowledge, documentation, or explanations
- troubleshoot: Diagnose and resolve technical issues
- process_request: Handle account changes, orders, or service requests
- escalate: Transfer to specialized team or management
- redirect: Point to appropriate resources or departments
- schedule_callback: Arrange follow-up communication

Success Likelihood Factors:
- very_high: Simple request, clear solution available
- high: Standard procedure, likely to be resolved quickly
- medium: Requires some investigation or coordination
- low: Complex issue, may need multiple touchpoints
- very_low: Unclear requirements, potential policy conflicts

Resolution Complexity:
- simple: Can be resolved in single interaction with standard procedures
- moderate: May require 2-3 interactions or coordination with another team
- complex: Requires significant investigation, multiple teams, or policy exceptions
- very_complex: Involves technical issues, legal considerations, or major system changes

Analysis Guidelines:
1. Look for explicit statements of what customer wants
2. Identify implicit needs based on context and emotional state
3. Consider potential blocking factors (technical, policy, or procedural)
4. Assess realistic success likelihood based on typical resolution patterns
5. Recommend specific next steps that advance toward customer goal

Pay attention to:
- Direct requests: "I want to...", "I need to...", "Can you help me..."
- Problem statements: "I'm having trouble with...", "It's not working..."
- Emotional context: Frustration may indicate deeper issues beyond stated problem
- Urgency indicators: Time pressure affects resolution approach
- Previous interactions: References to prior support contacts

IMPORTANT: Provide analysis responses in the same language as the input text, except for the
predefined categorical fields (primary_intent, action_required, success_likelihood,
resolution_complexity) which must use the exact English values specified above. For example,
if the input is in Japanese, provide customer_goal, implicit_needs, blocking_factors,
next_steps, and reasoning in Japanese, but use English values like "get_help" for primary_intent.

Provide comprehensive intent analysis with actionable recommendations."""

    return PreparedTask(instructions=instructions, response_format=IntentAnalysis, api_kwargs=api_kwargs)


# Backward compatibility - default configuration
INTENT_ANALYSIS = intent_analysis(temperature=0.0, top_p=1.0)
