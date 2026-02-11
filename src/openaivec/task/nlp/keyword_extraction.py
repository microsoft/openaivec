"""Keyword extraction task definition."""

from pydantic import BaseModel, ConfigDict, Field

from openaivec._model import PreparedTask
from openaivec.task._prompt_templates import join_sections, same_language_policy
from openaivec.task._registry import TaskSpec

__all__ = ["keyword_extraction"]


class Keyword(BaseModel):
    """Single keyword or keyphrase entry."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(description="The keyword or phrase")
    score: float = Field(ge=0, le=1, description="Importance score (0.0-1.0)")
    frequency: int = Field(ge=0, description="Frequency of occurrence in the text")
    context: str | None = Field(default=None, description="Context where the keyword appears")


class KeywordExtraction(BaseModel):
    """Keyword extraction output."""

    model_config = ConfigDict(extra="forbid")

    keywords: list[Keyword] = Field(description="Extracted keywords ranked by importance")
    keyphrases: list[Keyword] = Field(description="Extracted multi-word phrases ranked by importance")
    topics: list[str] = Field(description="Identified main topics in the text")
    summary: str = Field(description="Brief summary of the text content")


def _build_instructions() -> str:
    return join_sections(
        "Extract important keywords and keyphrases from the input text.",
        "Rank keywords by importance, include frequency and context, identify topics, and provide a short summary.",
        same_language_policy(),
    )


def keyword_extraction() -> PreparedTask[KeywordExtraction]:
    """Create a keyword extraction task."""
    return PreparedTask(
        instructions=_build_instructions(),
        response_format=KeywordExtraction,
    )


TASK_SPEC = TaskSpec(
    key="nlp.keyword_extraction",
    domain="nlp",
    summary="Extract ranked keywords, keyphrases, and topics.",
    factory=keyword_extraction,
    response_format=KeywordExtraction,
)

