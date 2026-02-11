"""Prebuilt task factories and registry helpers.

Use function-based task factories from domain modules:

- ``openaivec.task.nlp``
- ``openaivec.task.customer_support``
- ``openaivec.task.table``

Registry helpers provide discoverable task lookup by key.

Migration note (breaking change):
- Constant-style task names were removed.
- Example: ``nlp.SENTIMENT_ANALYSIS`` -> ``nlp.sentiment_analysis()``
- Example: ``customer_support.INTENT_ANALYSIS`` -> ``customer_support.intent_analysis()``
"""

from typing import Any

from openaivec._model import PreparedTask
from openaivec.task._registry import TaskSpec, get_task_spec_or_raise, list_task_specs

from . import customer_support, nlp, table

__all__ = [
    "TaskSpec",
    "customer_support",
    "get_task",
    "get_task_spec",
    "list_tasks",
    "nlp",
    "table",
]


def list_tasks(domain: str | None = None) -> list[str]:
    """Return registered task keys, optionally filtered by domain prefix."""
    return [spec.key for spec in list_task_specs(domain=domain)]


def get_task_spec(key: str) -> TaskSpec:
    """Return the task spec for a registry key."""
    return get_task_spec_or_raise(key)


def get_task(key: str, **kwargs: Any) -> PreparedTask[Any]:
    """Create a prepared task instance from a registry key."""
    spec = get_task_spec_or_raise(key)
    return spec.factory(**kwargs)
