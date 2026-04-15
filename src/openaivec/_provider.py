import os
import threading
import warnings

import duckdb
import tiktoken
from azure.identity import ClientSecretCredential, DefaultAzureCredential, get_bearer_token_provider
from openai import AsyncAzureOpenAI, AsyncOpenAI, AzureOpenAI, OpenAI

from openaivec import _di as di
from openaivec import _fabric as fabric
from openaivec._model import (
    AzureClientID,
    AzureClientSecret,
    AzureOpenAIAPIKey,
    AzureOpenAIAPIVersion,
    AzureOpenAIBaseURL,
    AzureTenantID,
    BearerTokenProvider,
    EmbeddingsModelName,
    KeyVaultSecretName,
    KeyVaultURL,
    OpenAIAPIKey,
    ResponsesModelName,
)
from openaivec._schema import SchemaInferer
from openaivec._util import TextChunker

__all__ = []

CONTAINER = di.Container()
_DEFAULT_REGISTRATIONS_LOCK = threading.RLock()
_DEFAULT_REGISTRATIONS_READY = False
_IGNORE_API_KEYS = frozenset(["place_holder_for_fabric_internal"])


def _build_missing_credentials_error(
    openai_api_key: str | None,
    azure_api_key: str | None,
    azure_base_url: str | None,
    azure_api_version: str | None,
) -> str:
    """Build a detailed error message for missing credentials.

    Args:
        openai_api_key (str | None): The OpenAI API key value.
        azure_api_key (str | None): The Azure OpenAI API key value.
        azure_base_url (str | None): The Azure OpenAI base URL value.
        azure_api_version (str | None): The Azure OpenAI API version value.

    Returns:
        str: A detailed error message with missing variables and setup instructions.
    """
    lines = ["No valid OpenAI or Azure OpenAI credentials found.", ""]

    # Check OpenAI
    lines.append("Option 1: Set OPENAI_API_KEY for OpenAI")
    if openai_api_key:
        lines.append("  ✓ OPENAI_API_KEY is set")
    else:
        lines.append("  ✗ OPENAI_API_KEY is not set")
        lines.append('    Example: export OPENAI_API_KEY="sk-..."')
    lines.append("")

    # Check Azure OpenAI
    lines.append("Option 2: Configure Azure OpenAI endpoint (API key or Entra ID)")
    lines.append("  Azure requires: AZURE_OPENAI_BASE_URL and AZURE_OPENAI_API_VERSION")
    lines.append("  Authentication: AZURE_OPENAI_API_KEY or Entra ID (DefaultAzureCredential)")
    azure_vars = [
        ("AZURE_OPENAI_API_KEY (optional)", azure_api_key, '"your-azure-api-key"'),
        ("AZURE_OPENAI_BASE_URL", azure_base_url, '"https://YOUR-RESOURCE-NAME.services.ai.azure.com/openai/v1/"'),
        ("AZURE_OPENAI_API_VERSION", azure_api_version, '"v1"'),
    ]
    for var_name, var_value, example in azure_vars:
        if var_value:
            lines.append(f"  ✓ {var_name} is set")
        else:
            lines.append(f"  ✗ {var_name} is not set")
            lines.append(f"    Example: export {var_name}={example}")

    if fabric.is_fabric_environment():
        lines.extend(fabric.build_credentials_error_section())

    return "\n".join(lines)


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
            "Azure OpenAI v1 API is recommended. Your base URL should end with '/openai/v1/'. "
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
    1. OPENAI_API_KEY - if set, creates standard OpenAI client.
    2. AZURE_OPENAI_BASE_URL and AZURE_OPENAI_API_VERSION - if set, creates AzureOpenAI.
       Authentication uses AZURE_OPENAI_API_KEY when present, otherwise Entra ID
       via bearer token provider.

    Returns:
        OpenAI: Configured OpenAI or AzureOpenAI client instance.

    Raises:
        ValueError: If no valid environment variables are found for either service.
    """
    ensure_default_registrations()
    openai_api_key = CONTAINER.resolve(OpenAIAPIKey)
    if openai_api_key.value and openai_api_key.value not in _IGNORE_API_KEYS:
        return OpenAI()

    azure_api_key = CONTAINER.resolve(AzureOpenAIAPIKey)
    azure_base_url = CONTAINER.resolve(AzureOpenAIBaseURL)
    azure_api_version = CONTAINER.resolve(AzureOpenAIAPIVersion)

    if azure_base_url.value and azure_api_version.value:
        _check_azure_v1_api_url(azure_base_url.value)

        if azure_api_key.value and azure_api_key.value not in _IGNORE_API_KEYS:
            return AzureOpenAI(
                api_key=azure_api_key.value,
                base_url=azure_base_url.value,
                api_version=azure_api_version.value,
            )

        return AzureOpenAI(
            azure_ad_token_provider=CONTAINER.resolve(BearerTokenProvider).value,
            base_url=azure_base_url.value,
            api_version=azure_api_version.value,
        )

    raise ValueError(
        _build_missing_credentials_error(
            openai_api_key=openai_api_key.value,
            azure_api_key=azure_api_key.value,
            azure_base_url=azure_base_url.value,
            azure_api_version=azure_api_version.value,
        )
    )


def provide_async_openai_client() -> AsyncOpenAI:
    """Provide asynchronous OpenAI client based on environment variables.

    Automatically detects and prioritizes OpenAI over Azure OpenAI configuration.
    Checks the following environment variables in order:
    1. OPENAI_API_KEY - if set, creates standard AsyncOpenAI client.
    2. AZURE_OPENAI_BASE_URL and AZURE_OPENAI_API_VERSION - if set, creates AsyncAzureOpenAI.
       Authentication uses AZURE_OPENAI_API_KEY when present, otherwise Entra ID
       via bearer token provider.

    Returns:
        AsyncOpenAI: Configured AsyncOpenAI or AsyncAzureOpenAI client instance.

    Raises:
        ValueError: If no valid environment variables are found for either service.
    """
    ensure_default_registrations()
    openai_api_key = CONTAINER.resolve(OpenAIAPIKey)
    if openai_api_key.value and openai_api_key.value not in _IGNORE_API_KEYS:
        return AsyncOpenAI()

    azure_api_key = CONTAINER.resolve(AzureOpenAIAPIKey)
    azure_base_url = CONTAINER.resolve(AzureOpenAIBaseURL)
    azure_api_version = CONTAINER.resolve(AzureOpenAIAPIVersion)

    if azure_base_url.value and azure_api_version.value:
        _check_azure_v1_api_url(azure_base_url.value)

        if azure_api_key.value and azure_api_key.value not in _IGNORE_API_KEYS:
            return AsyncAzureOpenAI(
                api_key=azure_api_key.value,
                base_url=azure_base_url.value,
                api_version=azure_api_version.value,
            )

        return AsyncAzureOpenAI(
            azure_ad_token_provider=CONTAINER.resolve(BearerTokenProvider).value,
            base_url=azure_base_url.value,
            api_version=azure_api_version.value,
        )

    raise ValueError(
        _build_missing_credentials_error(
            openai_api_key=openai_api_key.value,
            azure_api_key=azure_api_key.value,
            azure_base_url=azure_base_url.value,
            azure_api_version=azure_api_version.value,
        )
    )


def _provide_azure_client_secret() -> AzureClientSecret:
    """Provide ``AzureClientSecret``, auto-retrieving from Key Vault on Fabric."""
    secret = os.getenv("AZURE_CLIENT_SECRET")
    if not secret and fabric.is_fabric_environment():
        secret = fabric.retrieve_client_secret(
            kv_url=CONTAINER.resolve(KeyVaultURL).value,
            secret_name=CONTAINER.resolve(KeyVaultSecretName).value,
        )
    return AzureClientSecret(secret)


def _provide_bearer_token_provider() -> BearerTokenProvider:
    """Provide ``BearerTokenProvider`` using the best available credential.

    When ``AzureTenantID``, ``AzureClientID``, and ``AzureClientSecret`` are all
    present in the DI container, builds a ``ClientSecretCredential`` directly.
    Otherwise falls back to ``DefaultAzureCredential``.
    """
    tenant_id = CONTAINER.resolve(AzureTenantID).value
    client_id = CONTAINER.resolve(AzureClientID).value
    client_secret = CONTAINER.resolve(AzureClientSecret).value

    if tenant_id and client_id and client_secret:
        credential = ClientSecretCredential(tenant_id, client_id, client_secret)
    else:
        credential = DefaultAzureCredential()

    return BearerTokenProvider(
        value=get_bearer_token_provider(credential, "https://cognitiveservices.azure.com/.default")
    )


def _register_default_providers() -> None:
    """Install the library's default provider graph into the shared container."""
    CONTAINER.register(ResponsesModelName, lambda: ResponsesModelName("gpt-4.1-mini"))
    CONTAINER.register(EmbeddingsModelName, lambda: EmbeddingsModelName("text-embedding-3-small"))

    CONTAINER.register(OpenAIAPIKey, lambda: OpenAIAPIKey(os.getenv("OPENAI_API_KEY")))
    CONTAINER.register(AzureOpenAIAPIKey, lambda: AzureOpenAIAPIKey(os.getenv("AZURE_OPENAI_API_KEY")))
    CONTAINER.register(AzureOpenAIBaseURL, lambda: AzureOpenAIBaseURL(os.getenv("AZURE_OPENAI_BASE_URL")))
    CONTAINER.register(
        cls=AzureOpenAIAPIVersion,
        provider=lambda: AzureOpenAIAPIVersion(os.getenv("AZURE_OPENAI_API_VERSION", "v1")),
    )
    CONTAINER.register(AzureTenantID, lambda: AzureTenantID(os.getenv("AZURE_TENANT_ID")))
    CONTAINER.register(AzureClientID, lambda: AzureClientID(os.getenv("AZURE_CLIENT_ID")))
    CONTAINER.register(KeyVaultURL, lambda: KeyVaultURL(os.getenv("KEY_VAULT_URL")))
    CONTAINER.register(KeyVaultSecretName, lambda: KeyVaultSecretName(os.getenv("KEY_VAULT_SECRET_NAME")))
    CONTAINER.register(AzureClientSecret, _provide_azure_client_secret)

    if fabric.is_fabric_environment():
        CONTAINER.resolve(AzureClientSecret)
        fabric.log_environment_info()
        if fabric.is_partially_configured():
            fabric.warn_incomplete_configuration()

    CONTAINER.register(BearerTokenProvider, _provide_bearer_token_provider)

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
    CONTAINER.register(duckdb.DuckDBPyConnection, lambda: duckdb.connect(":memory:"))


def _defaults_installed() -> bool:
    """Return whether the minimum default provider graph is currently installed."""
    required = [
        ResponsesModelName,
        EmbeddingsModelName,
        OpenAIAPIKey,
        AzureOpenAIBaseURL,
        AzureOpenAIAPIVersion,
        OpenAI,
        AsyncOpenAI,
        tiktoken.Encoding,
    ]
    return all(CONTAINER.is_registered(cls) for cls in required)


def ensure_default_registrations() -> None:
    """Install default registrations lazily and re-install after container clears."""
    global _DEFAULT_REGISTRATIONS_READY
    if _DEFAULT_REGISTRATIONS_READY and _defaults_installed():
        return
    with _DEFAULT_REGISTRATIONS_LOCK:
        if _DEFAULT_REGISTRATIONS_READY and _defaults_installed():
            return
        _register_default_providers()
        _DEFAULT_REGISTRATIONS_READY = True


def set_default_registrations() -> None:
    """Reset the shared container to the library's default registrations.

    This is primarily useful in tests or when callers intentionally want to
    discard custom registrations and rebuild the default provider graph from a
    clean container.
    """
    global _DEFAULT_REGISTRATIONS_READY
    with _DEFAULT_REGISTRATIONS_LOCK:
        CONTAINER.clear()
        _register_default_providers()
        _DEFAULT_REGISTRATIONS_READY = True


ensure_default_registrations()


# ---------------------------------------------------------------------------
# Public configuration helpers
# ---------------------------------------------------------------------------


def set_client(client: OpenAI) -> None:
    """Register a custom OpenAI-compatible client.

    Args:
        client (OpenAI): A pre-configured ``openai.OpenAI`` or
            ``openai.AzureOpenAI`` instance.
    """
    if client.__class__.__name__ == "AzureOpenAI" and hasattr(client, "base_url"):
        _check_azure_v1_api_url(str(client.base_url))
    CONTAINER.register(OpenAI, lambda: client)


def get_client() -> OpenAI:
    """Get the currently registered OpenAI-compatible client.

    Returns:
        OpenAI: The registered client instance.
    """
    return CONTAINER.resolve(OpenAI)


def set_async_client(client: AsyncOpenAI) -> None:
    """Register a custom asynchronous OpenAI-compatible client.

    Args:
        client (AsyncOpenAI): A pre-configured ``openai.AsyncOpenAI`` or
            ``openai.AsyncAzureOpenAI`` instance.
    """
    if client.__class__.__name__ == "AsyncAzureOpenAI" and hasattr(client, "base_url"):
        _check_azure_v1_api_url(str(client.base_url))
    CONTAINER.register(AsyncOpenAI, lambda: client)


def get_async_client() -> AsyncOpenAI:
    """Get the currently registered asynchronous OpenAI-compatible client.

    Returns:
        AsyncOpenAI: The registered async client instance.
    """
    return CONTAINER.resolve(AsyncOpenAI)


def set_responses_model(name: str) -> None:
    """Override the model used for text responses.

    Args:
        name (str): Model or deployment name (e.g. ``"gpt-4.1-mini"``).
    """
    CONTAINER.register(ResponsesModelName, lambda: ResponsesModelName(name))


def get_responses_model() -> str:
    """Get the currently registered model name for text responses.

    Returns:
        str: The model name.
    """
    return CONTAINER.resolve(ResponsesModelName).value


def set_embeddings_model(name: str) -> None:
    """Override the model used for text embeddings.

    Args:
        name (str): Model or deployment name (e.g. ``"text-embedding-3-small"``).
    """
    CONTAINER.register(EmbeddingsModelName, lambda: EmbeddingsModelName(name))


def get_embeddings_model() -> str:
    """Get the currently registered model name for text embeddings.

    Returns:
        str: The model name.
    """
    return CONTAINER.resolve(EmbeddingsModelName).value
