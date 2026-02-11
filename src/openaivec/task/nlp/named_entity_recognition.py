"""Named entity recognition task definition."""

from pydantic import BaseModel, ConfigDict, Field

from openaivec._model import PreparedTask
from openaivec.task._prompt_templates import join_sections, same_language_policy
from openaivec.task._registry import TaskSpec

__all__ = ["named_entity_recognition"]


class NamedEntity(BaseModel):
    """Single named-entity span."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(description="The entity text")
    label: str = Field(description="Entity type label")
    start: int = Field(description="Start position in the original text")
    end: int = Field(description="End position in the original text")
    confidence: float | None = Field(default=None, ge=0, le=1, description="Confidence score (0.0-1.0)")


class NamedEntityRecognition(BaseModel):
    """Named entity recognition output."""

    model_config = ConfigDict(extra="forbid")

    persons: list[NamedEntity] = Field(description="Person entities")
    organizations: list[NamedEntity] = Field(description="Organization entities")
    locations: list[NamedEntity] = Field(description="Location entities")
    dates: list[NamedEntity] = Field(description="Date and time entities")
    money: list[NamedEntity] = Field(description="Money and currency entities")
    percentages: list[NamedEntity] = Field(description="Percentage entities")
    miscellaneous: list[NamedEntity] = Field(description="Other named entities")


def _build_instructions() -> str:
    return join_sections(
        "Identify named entities in the input text.",
        "Extract persons, organizations, locations, dates, money, percentages, and miscellaneous entities.",
        "Each entity must include text, label, span offsets, and optional confidence.",
        same_language_policy(),
    )


def named_entity_recognition() -> PreparedTask[NamedEntityRecognition]:
    """Create a named entity recognition task."""
    return PreparedTask(
        instructions=_build_instructions(),
        response_format=NamedEntityRecognition,
    )


TASK_SPEC = TaskSpec(
    key="nlp.named_entity_recognition",
    domain="nlp",
    summary="Extract named entities with span offsets and confidence.",
    factory=named_entity_recognition,
    response_format=NamedEntityRecognition,
)

