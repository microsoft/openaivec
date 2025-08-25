import os
import warnings

import tiktoken
from openai import AsyncAzureOpenAI, AsyncOpenAI, AzureOpenAI, OpenAI

from openaivec import _di as di
from openaivec._model import (
    AzureOpenAIAPIKey,
    AzureOpenAIAPIVersion,
    AzureOpenAIBaseURL,
    EmbeddingsModelName,
    OpenAIAPIKey,
    ResponsesModelName,
)
from openaivec._schema import SchemaInferer
from openaivec._util import TextChunker

__all__ = []

CONTAINER = di.Container()


def _check_azure_v1_api_url(base_url: str) -> None:
    """Check if Azure OpenAI base URL uses the recommended v1 API format.

    Issues a warning if the URL doesn't end with '/openai/v1/' to encourage
    migration to the v1 API format as recommended by Microsoft.

    Reference: https://learn.microsoft.com/en-us/azure/ai-foundry/openai/api-version-lifecycle

    Args:
        base_url (str): The Azure OpenAI base URL to check.
    """
    if base_url and not base_url.rstrip("/").endswith("/openai/v1"):
        warnings.warn(
            "⚠️  Azure OpenAI v1 API is recommended. Your base URL should end with '/openai/v1/'. "
            f"Current URL: '{base_url}'. "
            "Consider updating to: 'https://YOUR-RESOURCE-NAME.services.ai.azure.com/openai/v1/' "
            "for better performance and future compatibility. "
            "See: https://learn.microsoft.com/en-us/azure/ai-foundry/openai/api-version-lifecycle",
            UserWarning,
            stacklevel=3,
        )


def provide_openai_client() -> OpenAI:
    """Provide OpenAI client based on environment variables.

    Automatically detects and prioritizes OpenAI over Azure OpenAI configuration.
    Checks the following environment variables in order:
    1. OPENAI_API_KEY - if set, creates standard OpenAI client
    2. Azure OpenAI variables (AZURE_OPENAI_API_KEY, AZURE_OPENAI_BASE_URL,
       AZURE_OPENAI_API_VERSION) - if all set, creates Azure OpenAI client

    Returns:
        OpenAI: Configured OpenAI or AzureOpenAI client instance.

    Raises:
        ValueError: If no valid environment variables are found for either service.
    """
    openai_api_key = CONTAINER.resolve(OpenAIAPIKey)
    if openai_api_key.value:
        return OpenAI()

    azure_api_key = CONTAINER.resolve(AzureOpenAIAPIKey)
    azure_base_url = CONTAINER.resolve(AzureOpenAIBaseURL)
    azure_api_version = CONTAINER.resolve(AzureOpenAIAPIVersion)

    if all(param.value for param in [azure_api_key, azure_base_url, azure_api_version]):
        # Type checker support: values are guaranteed non-None by the all() check above
        assert azure_api_key.value is not None
        assert azure_base_url.value is not None
        assert azure_api_version.value is not None

        _check_azure_v1_api_url(azure_base_url.value)
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
    """Provide asynchronous OpenAI client based on environment variables.

    Automatically detects and prioritizes OpenAI over Azure OpenAI configuration.
    Checks the following environment variables in order:
    1. OPENAI_API_KEY - if set, creates standard AsyncOpenAI client
    2. Azure OpenAI variables (AZURE_OPENAI_API_KEY, AZURE_OPENAI_BASE_URL,
       AZURE_OPENAI_API_VERSION) - if all set, creates AsyncAzureOpenAI client

    Returns:
        AsyncOpenAI: Configured AsyncOpenAI or AsyncAzureOpenAI client instance.

    Raises:
        ValueError: If no valid environment variables are found for either service.
    """
    openai_api_key = CONTAINER.resolve(OpenAIAPIKey)
    if openai_api_key.value:
        return AsyncOpenAI()

    azure_api_key = CONTAINER.resolve(AzureOpenAIAPIKey)
    azure_base_url = CONTAINER.resolve(AzureOpenAIBaseURL)
    azure_api_version = CONTAINER.resolve(AzureOpenAIAPIVersion)

    if all(param.value for param in [azure_api_key, azure_base_url, azure_api_version]):
        # Type checker support: values are guaranteed non-None by the all() check above
        assert azure_api_key.value is not None
        assert azure_base_url.value is not None
        assert azure_api_version.value is not None

        _check_azure_v1_api_url(azure_base_url.value)
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


def set_default_registrations():
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
    CONTAINER.register(
        SchemaInferer,
        lambda: SchemaInferer(
            client=CONTAINER.resolve(OpenAI),
            model_name=CONTAINER.resolve(ResponsesModelName).value,
        ),
    )


set_default_registrations()
