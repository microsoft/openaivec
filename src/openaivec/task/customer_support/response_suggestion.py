"""Response suggestion task for customer support interactions.

This module provides a predefined task for generating suggested responses to
customer inquiries, helping support agents provide consistent, helpful,
and professional communication.

Example:
    Basic usage with BatchResponses:

    ```python
    from openai import OpenAI
    from openaivec._responses import BatchResponses
    from openaivec.task import customer_support

    client = OpenAI()
    responder = BatchResponses.of_task(
        client=client,
        model_name="gpt-4.1-mini",
        task=customer_support.RESPONSE_SUGGESTION
    )

    inquiries = [
        "I can't access my account. I've tried resetting my password but the email never arrives.",
        "I'm really disappointed with your service. This is the third time I've had issues.",
        "Thank you for your help yesterday! The problem is now resolved."
    ]
    responses = responder.parse(inquiries)

    for response in responses:
        print(f"Suggested Response: {response.suggested_response}")
        print(f"Tone: {response.tone}")
        print(f"Priority: {response.priority}")
        print(f"Follow-up: {response.follow_up_required}")
    ```

    With pandas integration:

    ```python
    import pandas as pd
    from openaivec import pandas_ext  # Required for .ai accessor
    from openaivec.task import customer_support

    df = pd.DataFrame({"inquiry": [
        "I can't access my account. I've tried resetting my password but the email never arrives.",
        "I'm really disappointed with your service. This is the third time I've had issues."
    ]})
    df["response"] = df["inquiry"].ai.task(customer_support.RESPONSE_SUGGESTION)

    # Extract response components
    extracted_df = df.ai.extract("response")
    print(extracted_df[["inquiry", "response_suggested_response", "response_tone", "response_priority"]])
    ```

Attributes:
    RESPONSE_SUGGESTION (PreparedTask): A prepared task instance
        configured for response suggestion with temperature=0.0 and
        top_p=1.0 for deterministic output.
"""

from typing import Literal

from pydantic import BaseModel, Field

from openaivec._model import PreparedTask

__all__ = ["response_suggestion"]


class ResponseSuggestion(BaseModel):
    suggested_response: str = Field(description="Professional response draft for the customer inquiry")
    tone: Literal["empathetic", "professional", "friendly", "apologetic", "solution_focused"] = Field(
        description="Recommended tone (empathetic, professional, friendly, apologetic, solution_focused)"
    )
    priority: Literal["immediate", "high", "medium", "low"] = Field(
        description="Response priority (immediate, high, medium, low)"
    )
    response_type: Literal["acknowledgment", "solution", "escalation", "information_request", "closure"] = Field(
        description="Type of response (acknowledgment, solution, escalation, information_request, closure)"
    )
    key_points: list[str] = Field(description="Main points that must be addressed in the response")
    follow_up_required: bool = Field(description="Whether follow-up communication is needed")
    escalation_suggested: bool = Field(description="Whether escalation to management is recommended")
    resources_needed: list[str] = Field(description="Additional resources or information required")
    estimated_resolution_time: Literal["immediate", "hours", "days", "weeks"] = Field(
        description="Estimated time to resolution (immediate, hours, days, weeks)"
    )
    alternative_responses: list[str] = Field(description="Alternative response options for different scenarios")
    personalization_notes: str = Field(description="Suggestions for personalizing the response")


def response_suggestion(
    response_style: str = "professional",
    company_name: str = "our company",
    business_context: str = "general customer support",
    **api_kwargs,
) -> PreparedTask:
    """Create a configurable response suggestion task.

    Args:
        response_style (str): Style of response (professional, friendly, empathetic, formal).
        company_name (str): Name of the company for personalization.
        business_context (str): Business context for responses.
        **api_kwargs: Additional keyword arguments to pass to the OpenAI API,
            such as temperature, top_p, etc.

    Returns:
        PreparedTask configured for response suggestions.
    """

    style_instructions = {
        "professional": "Maintain professional tone with clear, direct communication",
        "friendly": "Use warm, approachable language while remaining professional",
        "empathetic": "Show understanding and compassion for customer concerns",
        "formal": "Use formal business language appropriate for official communications",
    }

    instructions = f"""Generate a professional, helpful response suggestion for the customer
inquiry that addresses their needs effectively.

Business Context: {business_context}
Company Name: {company_name}
Response Style: {style_instructions.get(response_style, style_instructions["professional"])}

Response Guidelines:
1. Address the customer's main concern directly
2. Use appropriate tone based on customer sentiment
3. Provide clear next steps or solutions
4. Include empathy when dealing with frustrated customers
5. Maintain professional standards while being human
6. Offer specific help rather than generic responses
7. Set appropriate expectations for resolution time
8. Include any necessary disclaimers or policy information

Tone Selection:
- empathetic: For frustrated, disappointed, or upset customers
- professional: For business inquiries, formal requests, or complex issues
- friendly: For positive interactions, thank you messages, or simple questions
- apologetic: For service failures, bugs, or company mistakes
- solution_focused: For technical issues requiring specific steps

Response Types:
- acknowledgment: Confirming receipt and understanding of the inquiry
- solution: Providing direct answers or resolution steps
- escalation: Transferring to appropriate team or management
- information_request: Asking for additional details to help resolve
- closure: Confirming resolution and checking customer satisfaction

Priority Levels:
- immediate: Critical issues requiring instant response
- high: Urgent problems needing quick attention
- medium: Standard inquiries with normal response time
- low: General questions or feedback with flexible timing

Key Elements to Include:
- Acknowledge the customer's specific issue
- Show understanding of their frustration or needs
- Provide clear, actionable next steps
- Set realistic expectations for resolution
- Offer additional assistance if needed
- Include relevant contact information or resources

Response Structure:
1. Opening: Acknowledge and thank the customer
2. Empathy: Show understanding of their situation
3. Solution: Provide specific help or next steps
4. Follow-up: Offer continued assistance
5. Closing: Professional sign-off

Personalization Considerations:
- Use customer's name if provided
- Reference specific details from their inquiry
- Acknowledge their loyalty or relationship length
- Tailor language to their communication style
- Consider their apparent technical expertise level

Avoid:
- Generic, templated responses
- Overly technical language for non-technical customers
- Making promises that can't be kept
- Dismissing customer concerns
- Lengthy responses that don't address the main issue

IMPORTANT: Generate responses in the same language as the input text, except for the predefined
categorical fields (tone, priority, response_type, estimated_resolution_time) which must use
the exact English values specified above. For example, if the input is in Italian, provide
suggested_response, key_points, alternative_responses, and personalization_notes in Italian,
but use English values like "empathetic" for tone.

Generate helpful, professional response that moves toward resolution while maintaining
positive customer relationship."""

    return PreparedTask(instructions=instructions, response_format=ResponseSuggestion, api_kwargs=api_kwargs)


# Backward compatibility - default configuration
RESPONSE_SUGGESTION = response_suggestion(temperature=0.0, top_p=1.0)
