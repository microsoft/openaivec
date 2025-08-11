import os

import tiktoken
from openai import AsyncAzureOpenAI, AsyncOpenAI, AzureOpenAI, OpenAI

from . import di
from .model import (
    AzureOpenAIAPIKey,
    AzureOpenAIAPIVersion,
    AzureOpenAIBaseURL,
    EmbeddingsModelName,
    OpenAIAPIKey,
    ResponsesModelName,
)
from .util import TextChunker

CONTAINER = di.Container()


def provide_openai_client() -> OpenAI:
    """Provide OpenAI client based on environment variables. Prioritizes OpenAI over Azure."""
    openai_api_key = CONTAINER.resolve(OpenAIAPIKey)
    if openai_api_key.value:
        return OpenAI()

    azure_api_key = CONTAINER.resolve(AzureOpenAIAPIKey)
    azure_base_url = CONTAINER.resolve(AzureOpenAIBaseURL)
    azure_api_version = CONTAINER.resolve(AzureOpenAIAPIVersion)

    if all(param.value for param in [azure_api_key, azure_base_url, azure_api_version]):
        return AzureOpenAI(
            api_key=azure_api_key.value,
            base_url=azure_base_url.value,
            api_version=azure_api_version.value,
        )

    raise ValueError(
        "No valid OpenAI or Azure OpenAI environment variables found. "
        "Please set either OPENAI_API_KEY or AZURE_OPENAI_API_KEY, "
        "AZURE_OPENAI_BASE_URL, and AZURE_OPENAI_API_VERSION."
    )


def provide_async_openai_client() -> AsyncOpenAI:
    """Provide async OpenAI client based on environment variables. Prioritizes OpenAI over Azure."""
    openai_api_key = CONTAINER.resolve(OpenAIAPIKey)
    if openai_api_key.value:
        return AsyncOpenAI()

    azure_api_key = CONTAINER.resolve(AzureOpenAIAPIKey)
    azure_base_url = CONTAINER.resolve(AzureOpenAIBaseURL)
    azure_api_version = CONTAINER.resolve(AzureOpenAIAPIVersion)

    if all(param.value for param in [azure_api_key, azure_base_url, azure_api_version]):
        return AsyncAzureOpenAI(
            api_key=azure_api_key.value,
            base_url=azure_base_url.value,
            api_version=azure_api_version.value,
        )

    raise ValueError(
        "No valid OpenAI or Azure OpenAI environment variables found. "
        "Please set either OPENAI_API_KEY or AZURE_OPENAI_API_KEY, "
        "AZURE_OPENAI_BASE_URL, and AZURE_OPENAI_API_VERSION."
    )


CONTAINER.register(ResponsesModelName, lambda: ResponsesModelName("gpt-4.1-mini"))
CONTAINER.register(EmbeddingsModelName, lambda: EmbeddingsModelName("text-embedding-3-small"))
CONTAINER.register(OpenAIAPIKey, lambda: OpenAIAPIKey(os.getenv("OPENAI_API_KEY")))
CONTAINER.register(AzureOpenAIAPIKey, lambda: AzureOpenAIAPIKey(os.getenv("AZURE_OPENAI_API_KEY")))
CONTAINER.register(AzureOpenAIBaseURL, lambda: AzureOpenAIBaseURL(os.getenv("AZURE_OPENAI_BASE_URL")))
CONTAINER.register(
    cls=AzureOpenAIAPIVersion,
    provider=lambda: AzureOpenAIAPIVersion(os.getenv("AZURE_OPENAI_API_VERSION", "preview")),
)
CONTAINER.register(OpenAI, provide_openai_client)
CONTAINER.register(AsyncOpenAI, provide_async_openai_client)
CONTAINER.register(tiktoken.Encoding, lambda: tiktoken.get_encoding("o200k_base"))
CONTAINER.register(TextChunker, lambda: TextChunker(CONTAINER.resolve(tiktoken.Encoding)))


def reset_environment_registrations():
    """Reset environment variable related registrations in the container.

    This function re-registers environment variable dependent services to pick up
    current environment variable values. Useful for testing when environment
    variables are changed after initial container setup.
    """
    CONTAINER.register(OpenAIAPIKey, lambda: OpenAIAPIKey(os.getenv("OPENAI_API_KEY")))
    CONTAINER.register(AzureOpenAIAPIKey, lambda: AzureOpenAIAPIKey(os.getenv("AZURE_OPENAI_API_KEY")))
    CONTAINER.register(AzureOpenAIBaseURL, lambda: AzureOpenAIBaseURL(os.getenv("AZURE_OPENAI_BASE_URL")))
    CONTAINER.register(
        cls=AzureOpenAIAPIVersion,
        provider=lambda: AzureOpenAIAPIVersion(os.getenv("AZURE_OPENAI_API_VERSION", "preview")),
    )
    CONTAINER.register(OpenAI, provide_openai_client)
    CONTAINER.register(AsyncOpenAI, provide_async_openai_client)
