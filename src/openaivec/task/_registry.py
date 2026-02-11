"""Internal task registry for curated task factories."""

from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from openaivec._model import PreparedTask

__all__ = []

TaskFactory = Callable[..., PreparedTask[Any]]


@dataclass(frozen=True)
class TaskSpec:
    """Specification for a registered task factory."""

    key: str
    domain: str
    summary: str
    factory: TaskFactory
    response_format: type[Any]


@lru_cache(maxsize=1)
def _task_specs_by_key() -> dict[str, TaskSpec]:
    from openaivec.task.customer_support.customer_sentiment import TASK_SPEC as CUSTOMER_SENTIMENT_TASK_SPEC
    from openaivec.task.customer_support.inquiry_classification import TASK_SPEC as INQUIRY_CLASSIFICATION_TASK_SPEC
    from openaivec.task.customer_support.inquiry_summary import TASK_SPEC as INQUIRY_SUMMARY_TASK_SPEC
    from openaivec.task.customer_support.intent_analysis import TASK_SPEC as INTENT_ANALYSIS_TASK_SPEC
    from openaivec.task.customer_support.response_suggestion import TASK_SPEC as RESPONSE_SUGGESTION_TASK_SPEC
    from openaivec.task.customer_support.urgency_analysis import TASK_SPEC as URGENCY_ANALYSIS_TASK_SPEC
    from openaivec.task.nlp.dependency_parsing import TASK_SPEC as DEPENDENCY_PARSING_TASK_SPEC
    from openaivec.task.nlp.keyword_extraction import TASK_SPEC as KEYWORD_EXTRACTION_TASK_SPEC
    from openaivec.task.nlp.morphological_analysis import TASK_SPEC as MORPHOLOGICAL_ANALYSIS_TASK_SPEC
    from openaivec.task.nlp.named_entity_recognition import TASK_SPEC as NAMED_ENTITY_RECOGNITION_TASK_SPEC
    from openaivec.task.nlp.sentiment_analysis import TASK_SPEC as SENTIMENT_ANALYSIS_TASK_SPEC
    from openaivec.task.nlp.translation import TASK_SPEC as MULTILINGUAL_TRANSLATION_TASK_SPEC
    from openaivec.task.table.fillna import TASK_SPEC as FILLNA_TASK_SPEC

    specs = [
        MULTILINGUAL_TRANSLATION_TASK_SPEC,
        MORPHOLOGICAL_ANALYSIS_TASK_SPEC,
        NAMED_ENTITY_RECOGNITION_TASK_SPEC,
        SENTIMENT_ANALYSIS_TASK_SPEC,
        DEPENDENCY_PARSING_TASK_SPEC,
        KEYWORD_EXTRACTION_TASK_SPEC,
        CUSTOMER_SENTIMENT_TASK_SPEC,
        INQUIRY_CLASSIFICATION_TASK_SPEC,
        INQUIRY_SUMMARY_TASK_SPEC,
        INTENT_ANALYSIS_TASK_SPEC,
        RESPONSE_SUGGESTION_TASK_SPEC,
        URGENCY_ANALYSIS_TASK_SPEC,
        FILLNA_TASK_SPEC,
    ]

    mapping: dict[str, TaskSpec] = {}
    for spec in specs:
        if spec.key in mapping:
            raise ValueError(f"Duplicate task key detected: {spec.key}")
        mapping[spec.key] = spec
    return mapping


def list_task_specs(domain: str | None = None) -> list[TaskSpec]:
    """Return registered task specs sorted by key."""
    specs = list(_task_specs_by_key().values())
    if domain is not None:
        prefix = domain if domain.endswith(".") else f"{domain}."
        specs = [spec for spec in specs if spec.key.startswith(prefix)]
    return sorted(specs, key=lambda spec: spec.key)


def get_task_spec_or_raise(key: str) -> TaskSpec:
    """Return a registered task spec or raise ``KeyError``."""
    specs = _task_specs_by_key()
    if key not in specs:
        raise KeyError(key)
    return specs[key]
