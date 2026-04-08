"""Deprecated configuration helpers — use ``openaivec.set_*`` / ``openaivec.get_*`` instead.

These wrappers emit a ``DeprecationWarning`` and delegate to
``openaivec._provider``.
"""

import warnings

from openaivec._provider import (
    get_async_client as _get_async_client,
)
from openaivec._provider import (
    get_client as _get_client,
)
from openaivec._provider import (
    get_embeddings_model as _get_embeddings_model,
)
from openaivec._provider import (
    get_responses_model as _get_responses_model,
)
from openaivec._provider import (
    set_async_client as _set_async_client,
)
from openaivec._provider import (
    set_client as _set_client,
)
from openaivec._provider import (
    set_embeddings_model as _set_embeddings_model,
)
from openaivec._provider import (
    set_responses_model as _set_responses_model,
)

__all__ = [
    "get_async_client",
    "get_client",
    "get_embeddings_model",
    "get_responses_model",
    "set_async_client",
    "set_client",
    "set_embeddings_model",
    "set_responses_model",
]

_MSG = "pandas_ext.{name}() is deprecated. Use openaivec.{name}() instead."


def set_client(client):
    """Deprecated: use ``openaivec.set_client()``."""
    warnings.warn(_MSG.format(name="set_client"), DeprecationWarning, stacklevel=2)
    _set_client(client)


def get_client():
    """Deprecated: use ``openaivec.get_client()``."""
    warnings.warn(_MSG.format(name="get_client"), DeprecationWarning, stacklevel=2)
    return _get_client()


def set_async_client(client):
    """Deprecated: use ``openaivec.set_async_client()``."""
    warnings.warn(_MSG.format(name="set_async_client"), DeprecationWarning, stacklevel=2)
    _set_async_client(client)


def get_async_client():
    """Deprecated: use ``openaivec.get_async_client()``."""
    warnings.warn(_MSG.format(name="get_async_client"), DeprecationWarning, stacklevel=2)
    return _get_async_client()


def set_responses_model(name):
    """Deprecated: use ``openaivec.set_responses_model()``."""
    warnings.warn(_MSG.format(name="set_responses_model"), DeprecationWarning, stacklevel=2)
    _set_responses_model(name)


def get_responses_model():
    """Deprecated: use ``openaivec.get_responses_model()``."""
    warnings.warn(_MSG.format(name="get_responses_model"), DeprecationWarning, stacklevel=2)
    return _get_responses_model()


def set_embeddings_model(name):
    """Deprecated: use ``openaivec.set_embeddings_model()``."""
    warnings.warn(_MSG.format(name="set_embeddings_model"), DeprecationWarning, stacklevel=2)
    _set_embeddings_model(name)


def get_embeddings_model():
    """Deprecated: use ``openaivec.get_embeddings_model()``."""
    warnings.warn(_MSG.format(name="get_embeddings_model"), DeprecationWarning, stacklevel=2)
    return _get_embeddings_model()
