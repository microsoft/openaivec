import os
import threading

import duckdb
import tiktoken
from azure.identity import ClientSecretCredential, DefaultAzureCredential, get_bearer_token_provider
from openai import AsyncOpenAI, OpenAI

from openaivec import _di as di
from openaivec import _fabric as fabric
from openaivec._model import (
    AzureClientID,
    AzureClientSecret,
    AzureOpenAIAPIKey,
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

_FOUNDRY_SCOPE = "https://ai.azure.com/.default"

CONTAINER = di.Container()
_DEFAULT_REGISTRATIONS_LOCK = threading.RLock()
_DEFAULT_REGISTRATIONS_READY = False
_IGNORE_API_KEYS = frozenset(["place_holder_for_fabric_internal"])


# ---------------------------------------------------------------------------
# Credential / URL helpers
# ---------------------------------------------------------------------------


def _ensure_v1(base_url: str) -> str:
    """Normalize an Azure OpenAI base URL to end with ``/openai/v1/``.

    The v1 API is the only supported surface; legacy URLs are silently rewritten.

    Args:
        base_url (str): A base URL such as ``https://X.services.ai.azure.com``,
            ``https://X.services.ai.azure.com/openai``, or
            ``https://X.services.ai.azure.com/openai/v1/``.

    Returns:
        str: A URL guaranteed to end with ``/openai/v1/``.
    """
    trimmed = base_url.rstrip("/")
    if trimmed.endswith("/openai/v1"):
        return trimmed + "/"
    if trimmed.endswith("/openai"):
        return trimmed + "/v1/"
    return trimmed + "/openai/v1/"


def _is_usable(value: str | None) -> bool:
    """Return True when ``value`` is a non-empty, non-placeholder credential."""
    return bool(value) and value not in _IGNORE_API_KEYS


def _build_client_kwargs() -> dict:
    """Resolve credentials from the DI container and build ``OpenAI(**kw)`` kwargs.

    Selection order:

    1. ``OPENAI_API_KEY`` set → OpenAI public API (``{"api_key": key}``).
    2. ``AZURE_OPENAI_BASE_URL`` set:

       a. ``AZURE_OPENAI_API_KEY`` set → Azure API-key auth
          (``{"api_key": key, "base_url": url}``).
       b. Otherwise → Azure Entra ID auth
          (``{"api_key": token_provider, "base_url": url}``).
          The bearer-token provider is built from a ``ClientSecretCredential``
          when Tenant/Client/Secret are present (Fabric retrieves the secret
          from Key Vault automatically), otherwise from
          ``DefaultAzureCredential``.

    Returns:
        dict: Keyword arguments to pass to ``OpenAI`` / ``AsyncOpenAI``.

    Raises:
        ValueError: When no usable credentials are found.
    """
    openai_key = CONTAINER.resolve(OpenAIAPIKey).value
    if _is_usable(openai_key):
        return {"api_key": openai_key}

    base_url = CONTAINER.resolve(AzureOpenAIBaseURL).value
    if base_url:
        base_url = _ensure_v1(base_url)
        azure_key = CONTAINER.resolve(AzureOpenAIAPIKey).value
        if _is_usable(azure_key):
            return {"api_key": azure_key, "base_url": base_url}
        token_provider = CONTAINER.resolve(BearerTokenProvider).value
        return {"api_key": token_provider, "base_url": base_url}

    raise ValueError(
        _build_missing_credentials_error(
            openai_api_key=openai_key,
            azure_api_key=CONTAINER.resolve(AzureOpenAIAPIKey).value,
            azure_base_url=base_url,
        )
    )


def _build_missing_credentials_error(
    openai_api_key: str | None,
    azure_api_key: str | None,
    azure_base_url: str | None,
) -> str:
    """Build a detailed error message for missing credentials.

    Args:
        openai_api_key (str | None): The OpenAI API key value.
        azure_api_key (str | None): The Azure OpenAI API key value.
        azure_base_url (str | None): The Azure OpenAI base URL value.

    Returns:
        str: A detailed error message with missing variables and setup instructions.
    """
    lines = ["No valid OpenAI or Azure OpenAI credentials found.", ""]

    lines.append("Option 1: Set OPENAI_API_KEY for OpenAI")
    if openai_api_key:
        lines.append("  ✓ OPENAI_API_KEY is set")
    else:
        lines.append("  ✗ OPENAI_API_KEY is not set")
        lines.append('    Example: export OPENAI_API_KEY="sk-..."')
    lines.append("")

    lines.append("Option 2: Configure Azure OpenAI endpoint (API key or Entra ID)")
    lines.append("  Azure requires: AZURE_OPENAI_BASE_URL")
    lines.append("  Authentication: AZURE_OPENAI_API_KEY or Entra ID (DefaultAzureCredential)")
    azure_vars = [
        ("AZURE_OPENAI_API_KEY (optional)", azure_api_key, '"your-azure-api-key"'),
        ("AZURE_OPENAI_BASE_URL", azure_base_url, '"https://YOUR-RESOURCE-NAME.services.ai.azure.com/openai/v1/"'),
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


# ---------------------------------------------------------------------------
# Provider functions registered in the DI container
# ---------------------------------------------------------------------------


def provide_openai_client() -> OpenAI:
    """Provide an ``OpenAI`` client configured from environment / DI state.

    With the v1 API a single ``openai.OpenAI`` class covers all three auth
    paths (OpenAI direct, Azure API key, Azure Entra ID) by varying
    ``api_key`` (string or callable) and ``base_url``.

    Returns:
        OpenAI: A configured client.

    Raises:
        ValueError: When no usable credentials are found.
    """
    ensure_default_registrations()
    return OpenAI(**_build_client_kwargs())


def provide_async_openai_client() -> AsyncOpenAI:
    """Provide an ``AsyncOpenAI`` client configured from environment / DI state.

    Mirrors :func:`provide_openai_client` for the async client. The same
    bearer-token provider (a synchronous callable) is accepted by the SDK as
    ``api_key`` for both sync and async clients.

    Returns:
        AsyncOpenAI: A configured async client.

    Raises:
        ValueError: When no usable credentials are found.
    """
    ensure_default_registrations()
    return AsyncOpenAI(**_build_client_kwargs())


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

    Uses ``ClientSecretCredential`` when Tenant/Client/Secret are all present
    in the DI container, otherwise falls back to ``DefaultAzureCredential``.
    """
    tenant_id = CONTAINER.resolve(AzureTenantID).value
    client_id = CONTAINER.resolve(AzureClientID).value
    client_secret = CONTAINER.resolve(AzureClientSecret).value

    if tenant_id and client_id and client_secret:
        credential = ClientSecretCredential(tenant_id, client_id, client_secret)
    else:
        credential = DefaultAzureCredential()

    return BearerTokenProvider(value=get_bearer_token_provider(credential, _FOUNDRY_SCOPE))


# ---------------------------------------------------------------------------
# Default registrations
# ---------------------------------------------------------------------------


def _register_default_providers() -> None:
    """Install the library's default provider graph into the shared container."""
    CONTAINER.register(ResponsesModelName, lambda: ResponsesModelName("gpt-4.1-mini"))
    CONTAINER.register(EmbeddingsModelName, lambda: EmbeddingsModelName("text-embedding-3-small"))

    CONTAINER.register(OpenAIAPIKey, lambda: OpenAIAPIKey(os.getenv("OPENAI_API_KEY")))
    CONTAINER.register(AzureOpenAIAPIKey, lambda: AzureOpenAIAPIKey(os.getenv("AZURE_OPENAI_API_KEY")))
    CONTAINER.register(AzureOpenAIBaseURL, lambda: AzureOpenAIBaseURL(os.getenv("AZURE_OPENAI_BASE_URL")))
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

    Primarily useful in tests or when callers intentionally want to discard
    custom registrations and rebuild the default provider graph from a clean
    container.
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
    """Register a custom ``OpenAI`` client.

    Args:
        client (OpenAI): A pre-configured ``openai.OpenAI`` instance. To target
            Azure OpenAI, construct it with ``base_url`` ending in ``/openai/v1/``
            and either an API key or a bearer-token provider callable as
            ``api_key``.
    """
    CONTAINER.register(OpenAI, lambda: client)


def get_client() -> OpenAI:
    """Get the currently registered ``OpenAI`` client.

    Returns:
        OpenAI: The registered client instance.
    """
    return CONTAINER.resolve(OpenAI)


def set_async_client(client: AsyncOpenAI) -> None:
    """Register a custom ``AsyncOpenAI`` client.

    Args:
        client (AsyncOpenAI): A pre-configured ``openai.AsyncOpenAI`` instance.
            See :func:`set_client` for the Azure configuration pattern.
    """
    CONTAINER.register(AsyncOpenAI, lambda: client)


def get_async_client() -> AsyncOpenAI:
    """Get the currently registered ``AsyncOpenAI`` client.

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
