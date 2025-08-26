"""Inquiry summary task for customer support interactions.

This module provides a predefined task for summarizing customer inquiries,
extracting key information, and creating concise summaries for support agents
and management reporting.

Example:
    Basic usage with BatchResponses:

    ```python
    from openai import OpenAI
    from openaivec._responses import BatchResponses
    from openaivec.task import customer_support

    client = OpenAI()
    summarizer = BatchResponses.of_task(
        client=client,
        model_name="gpt-4.1-mini",
        task=customer_support.INQUIRY_SUMMARY
    )

    inquiries = [
        '''Hi there, I've been having trouble with my account for the past week.
        Every time I try to log in, it says my password is incorrect, but I'm sure
        it's right. I tried resetting it twice but the email never arrives.
        I'm getting really frustrated because I need to access my files for work tomorrow.''',

        '''I love your product! It's been incredibly helpful for my team.
        However, I was wondering if there's any way to get more storage space?
        We're running out and would like to upgrade our plan.'''
    ]
    summaries = summarizer.parse(inquiries)

    for summary in summaries:
        print(f"Summary: {summary.summary}")
        print(f"Issue: {summary.main_issue}")
        print(f"Actions Taken: {summary.actions_taken}")
        print(f"Resolution Status: {summary.resolution_status}")
    ```

    With pandas integration:

    ```python
    import pandas as pd
    from openaivec import pandas_ext  # Required for .ai accessor
    from openaivec.task import customer_support

    df = pd.DataFrame({"inquiry": [long_inquiry_text]})
    df["summary"] = df["inquiry"].ai.task(customer_support.INQUIRY_SUMMARY)

    # Extract summary components
    extracted_df = df.ai.extract("summary")
    print(extracted_df[["inquiry", "summary_main_issue", "summary_resolution_status"]])
    ```

Attributes:
    INQUIRY_SUMMARY (PreparedTask): A prepared task instance
        configured for inquiry summarization with temperature=0.0 and
        top_p=1.0 for deterministic output.
"""

from typing import Literal

from pydantic import BaseModel, Field

from openaivec._model import PreparedTask

__all__ = ["inquiry_summary"]


class InquirySummary(BaseModel):
    summary: str = Field(description="Concise summary of the customer inquiry (2-3 sentences)")
    main_issue: str = Field(description="Primary problem or request being addressed")
    secondary_issues: list[str] = Field(description="Additional issues mentioned in the inquiry")
    customer_background: str = Field(description="Relevant customer context or history mentioned")
    actions_taken: list[str] = Field(description="Steps the customer has already attempted")
    timeline: str = Field(description="Timeline of events or when the issue started")
    impact_description: str = Field(description="How the issue affects the customer")
    resolution_status: Literal["not_started", "in_progress", "needs_escalation", "resolved"] = Field(
        description="Current status (not_started, in_progress, needs_escalation, resolved)"
    )
    key_details: list[str] = Field(description="Important technical details, error messages, or specifics")
    follow_up_needed: bool = Field(description="Whether follow-up communication is required")
    summary_confidence: float = Field(description="Confidence in summary accuracy (0.0-1.0)")


def inquiry_summary(
    summary_length: str = "concise",
    business_context: str = "general customer support",
    **api_kwargs,
) -> PreparedTask:
    """Create a configurable inquiry summary task.

    Args:
        summary_length (str): Length of summary (concise, detailed, bullet_points).
        business_context (str): Business context for summary.
        **api_kwargs: Additional keyword arguments to pass to the OpenAI API,
            such as temperature, top_p, etc.

    Returns:
        PreparedTask configured for inquiry summarization.
    """

    length_instructions = {
        "concise": "Write a concise 2-3 sentence summary that captures the essence of the inquiry",
        "detailed": "Write a detailed 4-6 sentence summary that includes comprehensive context",
        "bullet_points": "Create a bullet-point summary with key facts and actions",
    }

    instructions = f"""Create a comprehensive summary of the customer inquiry that captures all
essential information for support agents and management.

Business Context: {business_context}
Summary Style: {length_instructions.get(summary_length, length_instructions["concise"])}

Summary Guidelines:
1. {length_instructions.get(summary_length, length_instructions["concise"])}
2. Identify the primary issue or request clearly
3. Note any secondary issues that may need attention
4. Extract relevant customer background or context
5. List any troubleshooting steps the customer has already tried
6. Include timeline information about when issues started
7. Describe the business or personal impact on the customer
8. Assess current resolution status based on the inquiry
9. Extract key technical details, error messages, or specific information
10. Determine if follow-up communication will be needed

Resolution Status Categories:
- not_started: New inquiry, no resolution attempts yet
- in_progress: Customer has tried some solutions, but issue persists
- needs_escalation: Complex issue requiring specialized attention
- resolved: Issue appears to be resolved based on customer feedback

Key Details to Extract:
- Error messages or codes
- Product versions or configurations
- Account information (without sensitive data)
- Technical specifications
- Business impact details
- Deadline or time constraints
- Previous ticket references

Impact Assessment:
- Business operations affected
- Revenue implications
- User experience degradation
- Time-sensitive requirements
- Reputation concerns

Focus on:
- Factual information over emotional content
- Actionable details that help resolution
- Context that aids in prioritization
- Clear distinction between symptoms and root causes
- Relevant background without unnecessary details

IMPORTANT: Provide summary responses in the same language as the input text, except for the
predefined categorical field (resolution_status) which must use the exact English values
specified above (not_started, in_progress, needs_escalation, resolved). For example, if the
input is in German, provide all summary content in German, but use English values like
"in_progress" for resolution_status.

Provide accurate, actionable summary that enables efficient support resolution."""

    return PreparedTask(instructions=instructions, response_format=InquirySummary, api_kwargs=api_kwargs)


# Backward compatibility - default configuration
INQUIRY_SUMMARY = inquiry_summary(temperature=0.0, top_p=1.0)
