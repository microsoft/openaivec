"""Morphological analysis task definition."""

from pydantic import BaseModel, ConfigDict, Field

from openaivec._model import PreparedTask
from openaivec.task._prompt_templates import join_sections, same_language_policy
from openaivec.task._registry import TaskSpec

__all__ = ["morphological_analysis"]


class MorphologicalAnalysis(BaseModel):
    """Morphological analysis output."""

    model_config = ConfigDict(extra="forbid")

    tokens: list[str] = Field(description="List of tokens in the text")
    pos_tags: list[str] = Field(description="Part-of-speech tags for each token")
    lemmas: list[str] = Field(description="Lemmatized form of each token")
    morphological_features: list[str] = Field(
        description="Morphological features for each token (for example tense, number, case)"
    )


def _build_instructions() -> str:
    return join_sections(
        "Perform morphological analysis for the input text.",
        "Return tokenization, part-of-speech tags, lemmas, and morphological features.",
        same_language_policy(),
    )


def morphological_analysis() -> PreparedTask[MorphologicalAnalysis]:
    """Create a morphological analysis task."""
    return PreparedTask(
        instructions=_build_instructions(),
        response_format=MorphologicalAnalysis,
    )


TASK_SPEC = TaskSpec(
    key="nlp.morphological_analysis",
    domain="nlp",
    summary="Tokenize text and return POS, lemmas, and morphological features.",
    factory=morphological_analysis,
    response_format=MorphologicalAnalysis,
)

