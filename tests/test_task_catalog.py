import inspect

from pydantic import BaseModel

from openaivec._model import PreparedTask
from openaivec.task import customer_support, nlp


def _assert_prepared_task(task: PreparedTask) -> None:
    assert isinstance(task, PreparedTask)
    assert isinstance(task.instructions, str)
    assert task.instructions.strip()
    assert isinstance(task.response_format, type)
    assert issubclass(task.response_format, BaseModel)


def test_nlp_exported_tasks_are_prepared_tasks():
    for name in nlp.__all__:
        symbol = getattr(nlp, name)
        _assert_prepared_task(symbol)


def test_customer_support_exports_return_or_are_prepared_tasks():
    for name in customer_support.__all__:
        symbol = getattr(customer_support, name)
        if inspect.isfunction(symbol):
            task = symbol()
        else:
            task = symbol
        _assert_prepared_task(task)


def test_customer_support_inquiry_classification_customization_is_reflected_in_prompt():
    task = customer_support.inquiry_classification(
        categories={"billing": ["refund_request", "invoice_question"]},
        routing_rules={"billing": "billing_team"},
        business_context="subscription SaaS",
        custom_keywords={"billing": ["chargeback", "invoice"]},
    )

    _assert_prepared_task(task)
    assert "subscription SaaS" in task.instructions
    assert "refund_request" in task.instructions
    assert "billing_team" in task.instructions
    assert "chargeback" in task.instructions


def test_customer_sentiment_custom_context_is_reflected_in_prompt():
    task = customer_support.customer_sentiment(business_context="healthcare appointment platform")

    _assert_prepared_task(task)
    assert "healthcare appointment platform" in task.instructions
