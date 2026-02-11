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
