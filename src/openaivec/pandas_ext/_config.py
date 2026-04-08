"""Client and model configuration helpers for the pandas_ext package."""

from openai import AsyncOpenAI, OpenAI

from openaivec._model import EmbeddingsModelName, ResponsesModelName
from openaivec._provider import CONTAINER, _check_azure_v1_api_url


def set_client(client: OpenAI) -> None:
    """Register a custom OpenAI-compatible client for pandas helpers.

    Args:
        client (OpenAI): A pre-configured `openai.OpenAI` or
            `openai.AzureOpenAI` instance reused by every helper in this module.
    """
    if client.__class__.__name__ == "AzureOpenAI" and hasattr(client, "base_url"):
        _check_azure_v1_api_url(str(client.base_url))

    CONTAINER.register(OpenAI, lambda: client)


def get_client() -> OpenAI:
    """Get the currently registered OpenAI-compatible client.

    Returns:
        OpenAI: The registered `openai.OpenAI` or `openai.AzureOpenAI` instance.
    """
    return CONTAINER.resolve(OpenAI)


def set_async_client(client: AsyncOpenAI) -> None:
    """Register a custom asynchronous OpenAI-compatible client.

    Args:
        client (AsyncOpenAI): A pre-configured `openai.AsyncOpenAI` or
            `openai.AsyncAzureOpenAI` instance reused by every helper in this module.
    """
    if client.__class__.__name__ == "AsyncAzureOpenAI" and hasattr(client, "base_url"):
        _check_azure_v1_api_url(str(client.base_url))

    CONTAINER.register(AsyncOpenAI, lambda: client)


def get_async_client() -> AsyncOpenAI:
    """Get the currently registered asynchronous OpenAI-compatible client.

    Returns:
        AsyncOpenAI: The registered `openai.AsyncOpenAI` or `openai.AsyncAzureOpenAI` instance.
    """
    return CONTAINER.resolve(AsyncOpenAI)


def set_responses_model(name: str) -> None:
    """Override the model used for text responses.

    Args:
        name (str): For Azure OpenAI, use your deployment name. For OpenAI, use the model name
            (for example, ``gpt-4.1-mini``).
    """
    CONTAINER.register(ResponsesModelName, lambda: ResponsesModelName(name))


def get_responses_model() -> str:
    """Get the currently registered model name for text responses.

    Returns:
        str: The model name (for example, ``gpt-4.1-mini``).
    """
    return CONTAINER.resolve(ResponsesModelName).value


def set_embeddings_model(name: str) -> None:
    """Override the model used for text embeddings.

    Args:
        name (str): For Azure OpenAI, use your deployment name. For OpenAI, use the model name,
            e.g. ``text-embedding-3-small``.
    """
    CONTAINER.register(EmbeddingsModelName, lambda: EmbeddingsModelName(name))


def get_embeddings_model() -> str:
    """Get the currently registered model name for text embeddings.

    Returns:
        str: The model name (for example, ``text-embedding-3-small``).
    """
    return CONTAINER.resolve(EmbeddingsModelName).value
