import importlib
import inspect

import pandas as pd
import pytest
from pydantic import BaseModel, ValidationError

from openaivec._model import PreparedTask
from openaivec.task import customer_support, get_task, get_task_spec, list_tasks, nlp

EXPECTED_TASK_KEYS = [
    "customer_support.customer_sentiment",
    "customer_support.inquiry_classification",
    "customer_support.inquiry_summary",
    "customer_support.intent_analysis",
    "customer_support.response_suggestion",
    "customer_support.urgency_analysis",
    "nlp.dependency_parsing",
    "nlp.keyword_extraction",
    "nlp.morphological_analysis",
    "nlp.multilingual_translation",
    "nlp.named_entity_recognition",
    "nlp.sentiment_analysis",
    "table.fillna",
]


def _assert_prepared_task(task: PreparedTask) -> None:
    assert isinstance(task, PreparedTask)
    assert isinstance(task.instructions, str)
    assert task.instructions.strip()
    assert isinstance(task.response_format, type)
    assert issubclass(task.response_format, BaseModel)


def test_nlp_exports_are_functions():
    for name in nlp.__all__:
        symbol = getattr(nlp, name)
        assert inspect.isfunction(symbol)
        _assert_prepared_task(symbol())


def test_customer_support_exports_are_functions():
    for name in customer_support.__all__:
        symbol = getattr(customer_support, name)
        assert inspect.isfunction(symbol)
        _assert_prepared_task(symbol())


def test_registry_lists_all_tasks():
    assert list_tasks() == sorted(EXPECTED_TASK_KEYS)


def test_registry_list_tasks_domain_filter():
    assert list_tasks("nlp") == sorted([key for key in EXPECTED_TASK_KEYS if key.startswith("nlp.")])
    assert list_tasks("customer_support") == sorted(
        [key for key in EXPECTED_TASK_KEYS if key.startswith("customer_support.")]
    )


def test_registry_get_task_spec_and_get_task(monkeypatch):
    for key in EXPECTED_TASK_KEYS:
        spec = get_task_spec(key)
        assert spec.key == key
        assert isinstance(spec.summary, str)
        assert spec.summary.strip()
        assert issubclass(spec.response_format, BaseModel)

        if key == "table.fillna":
            fillna_module = importlib.import_module("openaivec.task.table.fillna")
            monkeypatch.setattr(fillna_module, "_build_instructions", lambda *_args, **_kwargs: "stub prompt")
            df = pd.DataFrame({"name": ["Alice", None], "city": ["Tokyo", "Osaka"]})
            task = get_task(key, df=df, target_column_name="name")
        else:
            task = get_task(key)
        _assert_prepared_task(task)
        assert task.response_format is spec.response_format


def test_registry_raises_on_unknown_key():
    with pytest.raises(KeyError):
        get_task_spec("unknown.task")
    with pytest.raises(KeyError):
        get_task("unknown.task")


def test_removed_constant_exports_are_absent():
    assert not hasattr(nlp, "SENTIMENT_ANALYSIS")
    assert not hasattr(nlp, "MULTILINGUAL_TRANSLATION")
    assert not hasattr(customer_support, "INTENT_ANALYSIS")
    assert not hasattr(customer_support, "URGENCY_ANALYSIS")

    with pytest.raises(ImportError):
        exec("from openaivec.task.nlp import SENTIMENT_ANALYSIS")
    with pytest.raises(ImportError):
        exec("from openaivec.task.customer_support import INTENT_ANALYSIS")


def test_response_models_forbid_extra_fields():
    for key in EXPECTED_TASK_KEYS:
        spec = get_task_spec(key)
        assert spec.response_format.model_config.get("extra") == "forbid"


def test_numeric_range_validation_targets():
    with pytest.raises(ValidationError):
        get_task_spec("nlp.sentiment_analysis").response_format.model_validate(
            {
                "sentiment": "positive",
                "confidence": 1.5,
                "emotions": ["joy"],
                "emotion_scores": [0.8],
                "polarity": 0.5,
                "subjectivity": 0.4,
            }
        )

    with pytest.raises(ValidationError):
        get_task_spec("nlp.keyword_extraction").response_format.model_validate(
            {
                "keywords": [{"text": "a", "score": 2.0, "frequency": 1, "context": None}],
                "keyphrases": [],
                "topics": [],
                "summary": "x",
            }
        )

    with pytest.raises(ValidationError):
        get_task_spec("customer_support.customer_sentiment").response_format.model_validate(
            {
                "sentiment": "positive",
                "satisfaction_level": "satisfied",
                "emotional_state": "happy",
                "confidence": -0.1,
                "churn_risk": "low",
                "sentiment_intensity": 0.5,
                "polarity_score": 0.1,
                "tone_indicators": [],
                "relationship_status": "loyal",
                "response_approach": "professional",
            }
        )

    with pytest.raises(ValidationError):
        get_task_spec("customer_support.urgency_analysis").response_format.model_validate(
            {
                "urgency_level": "high",
                "urgency_score": 2.0,
                "response_time": "within_1_hour",
                "escalation_required": False,
                "urgency_indicators": [],
                "business_impact": "medium",
                "customer_tier": "standard",
                "reasoning": "x",
                "sla_compliance": True,
            }
        )


def test_customer_support_customization_reflected_in_prompt():
    task = customer_support.inquiry_classification(
        categories={"billing": ["refund_request", "invoice_question"]},
        routing_rules={"billing": "billing_team"},
        business_context="subscription SaaS",
        custom_keywords={"billing": ["chargeback", "invoice"]},
    )
    assert "subscription SaaS" in task.instructions
    assert "refund_request" in task.instructions
    assert "billing_team" in task.instructions
    assert "chargeback" in task.instructions


@pytest.mark.parametrize(
    ("key", "aligned", "field", "error_field"),
    [
        (
            "nlp.morphological_analysis",
            {
                "tokens": ["dogs"],
                "pos_tags": ["NOUN"],
                "lemmas": ["dog"],
                "morphological_features": ["plural"],
            },
            "tokens",
            "pos_tags",
        ),
        (
            "nlp.morphological_analysis",
            {
                "tokens": ["dogs"],
                "pos_tags": ["NOUN"],
                "lemmas": ["dog"],
                "morphological_features": ["plural"],
            },
            "pos_tags",
            "pos_tags",
        ),
        (
            "nlp.morphological_analysis",
            {
                "tokens": ["dogs"],
                "pos_tags": ["NOUN"],
                "lemmas": ["dog"],
                "morphological_features": ["plural"],
            },
            "lemmas",
            "lemmas",
        ),
        (
            "nlp.morphological_analysis",
            {
                "tokens": ["dogs"],
                "pos_tags": ["NOUN"],
                "lemmas": ["dog"],
                "morphological_features": ["plural"],
            },
            "morphological_features",
            "morphological_features",
        ),
        (
            "nlp.sentiment_analysis",
            {
                "sentiment": "positive",
                "confidence": 0.9,
                "emotions": ["joy"],
                "emotion_scores": [0.8],
                "polarity": 0.5,
                "subjectivity": 0.4,
            },
            "emotions",
            "emotion_scores",
        ),
        (
            "nlp.sentiment_analysis",
            {
                "sentiment": "positive",
                "confidence": 0.9,
                "emotions": ["joy"],
                "emotion_scores": [0.8],
                "polarity": 0.5,
                "subjectivity": 0.4,
            },
            "emotion_scores",
            "emotion_scores",
        ),
    ],
)
def test_curated_parallel_arrays_require_matching_lengths(key, aligned, field, error_field):
    model = get_task_spec(key).response_format
    assert model.model_validate(aligned)
    for replacement in ([], [*aligned[field], *aligned[field]]):
        with pytest.raises(ValidationError, match=error_field):
            model.model_validate({**aligned, field: replacement})
    with pytest.raises(ValidationError, match="Field required"):
        model.model_validate({name: value for name, value in aligned.items() if name != field})
    assert model.model_validate({key: [] if isinstance(value, list) else value for key, value in aligned.items()})


def test_urgency_custom_choices_match_schema():
    task = customer_support.urgency_analysis(
        urgency_levels={"urgent": "Outage", "routine": "General question"},
        response_times={"urgent": "within_15_minutes", "routine": "within_2_days"},
        customer_tiers={"vip": "Priority customer", "free": "Free customer"},
    )
    schema = task.response_format.model_json_schema()["properties"]
    assert schema["urgency_level"]["enum"] == ["urgent", "routine"]
    assert schema["response_time"]["enum"] == ["within_15_minutes", "within_2_days"]
    assert schema["customer_tier"]["enum"] == ["vip", "free"]
    for choice in ("urgent", "within_15_minutes", "vip"):
        assert choice in task.instructions
    assert "within_1_hour:" not in task.instructions
    assert "- critical:" not in task.instructions
    default = {
        "urgency_level": "urgent",
        "urgency_score": 0.9,
        "response_time": "within_15_minutes",
        "escalation_required": True,
        "urgency_indicators": [],
        "business_impact": "high",
        "customer_tier": "vip",
        "reasoning": "outage",
        "sla_compliance": True,
    }
    assert task.response_format.model_validate(default)
    with pytest.raises(ValidationError):
        task.response_format.model_validate({**default, "response_time": "immediate"})
    with pytest.raises(ValidationError):
        task.response_format.model_validate({**default, "unrecognized": True})


@pytest.mark.parametrize("field", ["urgency_levels", "response_times", "customer_tiers"])
def test_urgency_rejects_empty_choices(field):
    with pytest.raises(ValueError, match=field):
        customer_support.urgency_analysis(**{field: {}})


def test_urgency_response_times_can_be_customized_alone():
    task = customer_support.urgency_analysis(
        response_times={
            "critical": "within_15_minutes",
            "high": "within_2_hours",
            "medium": "within_8_hours",
            "low": "within_48_hours",
        }
    )
    assert task.response_format.model_json_schema()["properties"]["response_time"]["enum"] == [
        "within_15_minutes",
        "within_2_hours",
        "within_8_hours",
        "within_48_hours",
    ]
    assert "within_1_hour:" not in task.instructions


def test_urgency_rejects_unmapped_levels():
    with pytest.raises(ValueError, match="response_times"):
        customer_support.urgency_analysis(urgency_levels={"urgent": "Outage"})


def test_response_suggestion_formal_tone_is_valid():
    task = customer_support.response_suggestion(response_style="formal")
    assert "formal" in task.instructions
    assert "formal" in task.response_format.model_json_schema()["properties"]["tone"]["enum"]


def test_response_suggestion_rejects_unknown_style():
    with pytest.raises(ValueError, match="response_style"):
        customer_support.response_suggestion(response_style="unknown")
