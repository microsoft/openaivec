"""Dependency parsing task definition."""

from pydantic import BaseModel, ConfigDict, Field

from openaivec._model import PreparedTask
from openaivec.task._prompt_templates import join_sections, same_language_policy
from openaivec.task._registry import TaskSpec

__all__ = ["dependency_parsing"]


class DependencyRelation(BaseModel):
    """Single dependency edge."""

    model_config = ConfigDict(extra="forbid")

    head: str = Field(description="Head word in the dependency relation")
    dependent: str = Field(description="Dependent word in the dependency relation")
    relation: str = Field(description="Type of dependency relation")
    head_pos: int = Field(description="Position of head word in the sentence")
    dependent_pos: int = Field(description="Position of dependent word in the sentence")


class DependencyParsing(BaseModel):
    """Dependency parsing output."""

    model_config = ConfigDict(extra="forbid")

    tokens: list[str] = Field(description="List of tokens in the sentence")
    dependencies: list[DependencyRelation] = Field(description="Dependency relations between tokens")
    root_word: str = Field(description="Root word of the sentence")
    syntactic_structure: str = Field(description="Tree representation of the syntactic structure")


def _build_instructions() -> str:
    return join_sections(
        "Parse syntactic dependencies in the input text.",
        "Return tokens, dependency relations, root word, and a concise syntactic structure string.",
        same_language_policy(),
    )


def dependency_parsing() -> PreparedTask[DependencyParsing]:
    """Create a dependency parsing task."""
    return PreparedTask(
        instructions=_build_instructions(),
        response_format=DependencyParsing,
    )


TASK_SPEC = TaskSpec(
    key="nlp.dependency_parsing",
    domain="nlp",
    summary="Parse dependency relations and syntactic structure.",
    factory=dependency_parsing,
    response_format=DependencyParsing,
)

