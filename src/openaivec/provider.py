import os

import tiktoken
from openai import AsyncAzureOpenAI, AsyncOpenAI, AzureOpenAI, OpenAI

from openaivec import di
from openaivec.model import (
    AzureOpenAIAPIKey,
    AzureOpenAIAPIVersion,
    AzureOpenAIEndpoint,
    EmbeddingsModelName,
    OpenAIAPIKey,
    ResponsesModelName,
)

CONTAINER = di.Container()


def provide_openai_client() -> OpenAI:
    """Provide OpenAI client based on environment variables. Prioritizes OpenAI over Azure."""
    openai_api_key = CONTAINER.resolve(OpenAIAPIKey)
    if openai_api_key.value:
        return OpenAI()

    azure_api_key = CONTAINER.resolve(AzureOpenAIAPIKey)
    azure_endpoint = CONTAINER.resolve(AzureOpenAIEndpoint)
    azure_api_version = CONTAINER.resolve(AzureOpenAIAPIVersion)

    if all(param.value for param in [azure_api_key, azure_endpoint, azure_api_version]):
        return AzureOpenAI(
            api_key=azure_api_key.value,
            azure_endpoint=azure_endpoint.value,
            api_version=azure_api_version.value,
        )

    raise ValueError(
        "No valid OpenAI or Azure OpenAI environment variables found. "
        "Please set either OPENAI_API_KEY or AZURE_OPENAI_API_KEY, "
        "AZURE_OPENAI_API_ENDPOINT, and AZURE_OPENAI_API_VERSION."
    )


def provide_async_openai_client() -> AsyncOpenAI:
    """Provide async OpenAI client based on environment variables. Prioritizes OpenAI over Azure."""
    openai_api_key = CONTAINER.resolve(OpenAIAPIKey)
    if openai_api_key.value:
        return AsyncOpenAI()

    azure_api_key = CONTAINER.resolve(AzureOpenAIAPIKey)
    azure_endpoint = CONTAINER.resolve(AzureOpenAIEndpoint)
    azure_api_version = CONTAINER.resolve(AzureOpenAIAPIVersion)

    if all(param.value for param in [azure_api_key, azure_endpoint, azure_api_version]):
        return AsyncAzureOpenAI(
            api_key=azure_api_key.value,
            azure_endpoint=azure_endpoint.value,
            api_version=azure_api_version.value,
        )

    raise ValueError(
        "No valid OpenAI or Azure OpenAI environment variables found. "
        "Please set either OPENAI_API_KEY or AZURE_OPENAI_API_KEY, "
        "AZURE_OPENAI_API_ENDPOINT, and AZURE_OPENAI_API_VERSION."
    )


CONTAINER.register(ResponsesModelName, lambda: ResponsesModelName("gpt-4.1-mini"))
CONTAINER.register(EmbeddingsModelName, lambda: EmbeddingsModelName("text-embedding-3-small"))
CONTAINER.register(OpenAIAPIKey, lambda: OpenAIAPIKey(os.getenv("OPENAI_API_KEY")))
CONTAINER.register(AzureOpenAIAPIKey, lambda: AzureOpenAIAPIKey(os.getenv("AZURE_OPENAI_API_KEY")))
CONTAINER.register(AzureOpenAIEndpoint, lambda: AzureOpenAIEndpoint(os.getenv("AZURE_OPENAI_API_ENDPOINT")))
CONTAINER.register(
    cls=AzureOpenAIAPIVersion,
    provider=lambda: AzureOpenAIAPIVersion(os.getenv("AZURE_OPENAI_API_VERSION", "2025-04-01-preview")),
)
CONTAINER.register(OpenAI, provide_openai_client)
CONTAINER.register(AsyncOpenAI, provide_async_openai_client)
CONTAINER.register(tiktoken.Encoding, lambda: tiktoken.get_encoding("o200k_base"))
