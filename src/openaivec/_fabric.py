"""Microsoft Fabric environment detection and Entra ID authentication.

When running inside a Fabric notebook, ``notebookutils`` is available as a
site-package and provides ``credentials.getSecret`` for retrieving secrets from
Azure Key Vault.  This module exposes helpers that:

1. Detect the Fabric runtime.
2. Retrieve the Service Principal client secret from Key Vault via
   ``notebookutils.credentials.getSecret``.
3. Check whether the required environment variables are configured.
4. Log / warn / format environment-variable status for notebook UX.

On the Fabric driver, set ``AZURE_TENANT_ID``, ``AZURE_CLIENT_ID``,
``KEY_VAULT_URL``, and ``KEY_VAULT_SECRET_NAME``.  The library retrieves the
client secret from Key Vault automatically and builds a
``ClientSecretCredential`` via the DI container.
"""

import logging
import os
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from importlib import import_module
from typing import Any, cast
from urllib.parse import urlsplit

from openai import AsyncAzureOpenAI, AzureOpenAI
from openai._models import FinalRequestOptions
from openai.lib.azure import API_KEY_SENTINEL

__all__ = []

_LOGGER = logging.getLogger(__name__)
_FABRIC_PLACEHOLDER = "place_holder_for_fabric_internal"


@dataclass(init=False, eq=False)
class _FabricOpenAI(AzureOpenAI):
    """Azure SDK client using Fabric's stateless built-in model endpoint."""

    def _prepare_options(self, options: FinalRequestOptions) -> FinalRequestOptions:
        return super()._prepare_options(_stateless_response_options(options))


@dataclass(init=False, eq=False)
class _AsyncFabricOpenAI(AsyncAzureOpenAI):
    """Async Azure SDK client using Fabric's stateless built-in model endpoint."""

    async def _prepare_options(self, options: FinalRequestOptions) -> FinalRequestOptions:
        return await super()._prepare_options(_stateless_response_options(options))


def _stateless_response_options(options: FinalRequestOptions) -> FinalRequestOptions:
    if options.method.lower() != "post" or not urlsplit(options.url).path.rstrip("/").endswith("/responses"):
        return options
    json_data: object = options.json_data
    if not isinstance(json_data, Mapping):
        raise TypeError("Fabric Responses requests require a JSON object.")
    body = dict(cast(Mapping[str, object], json_data))
    extra: Mapping[str, object] = options.extra_json if isinstance(options.extra_json, Mapping) else {}
    store = extra.get("store", body.get("store"))
    if store is not None and store is not False:
        raise ValueError("Fabric built-in models require store=False.")
    if extra.get("previous_response_id", body.get("previous_response_id")) is not None:
        raise ValueError("Fabric built-in models do not support previous_response_id.")
    prepared = options.model_copy()
    body["store"] = False
    body.pop("previous_response_id", None)
    prepared.json_data = body
    if "store" in extra or "previous_response_id" in extra:
        extra = dict(extra)
        if "store" in extra:
            extra["store"] = False
        extra.pop("previous_response_id", None)
        prepared.extra_json = extra
    return prepared


def require_fabric_runtime() -> None:
    """Require the notebook runtime before replacing any configured clients."""
    if not is_fabric_environment():
        raise RuntimeError("setup_fabric() requires a Microsoft Fabric notebook runtime.")
    _require_fabric_helpers()


def _require_fabric_helpers() -> None:
    try:
        credentials = import_module("synapse.ml.fabric.credentials")
    except ImportError as exc:
        raise RuntimeError("The Fabric runtime is missing the SynapseML authentication helpers.") from exc
    if not callable(getattr(credentials, "get_openai_httpx_sync_client", None)):
        raise RuntimeError("The Fabric runtime is missing get_openai_httpx_sync_client().")


def _fabric_client_kwargs(*, api_version: str, async_client: bool = False) -> dict[str, Any]:
    _require_fabric_helpers()
    credentials = import_module("synapse.ml.fabric.credentials")
    helper_name = "get_openai_httpx_async_client" if async_client else "get_openai_httpx_sync_client"
    http_client_factory = getattr(credentials, helper_name, None)
    if not callable(http_client_factory):
        raise RuntimeError(
            "The async OpenAI HTTP client helper is unavailable in this Fabric runtime. "
            "Use the synchronous API or a Fabric runtime that provides get_openai_httpx_async_client()."
        )
    discovery = import_module("synapse.ml.fabric.service_discovery")
    endpoint = discovery.get_fabric_env_config().fabric_env_config.ml_workload_endpoint
    if not isinstance(endpoint, str) or urlsplit(endpoint).scheme != "https" or not urlsplit(endpoint).hostname:
        raise ValueError("Fabric service discovery must return an HTTPS workload endpoint.")
    return {
        "api_version": api_version,
        "azure_endpoint": endpoint.rstrip("/") + "/cognitive/openai",
        "api_key": API_KEY_SENTINEL,
        "azure_ad_token": _FABRIC_PLACEHOLDER,
        "default_headers": {
            "Authorization": f"Bearer {_FABRIC_PLACEHOLDER}",
            "api-key": _FABRIC_PLACEHOLDER,
        },
        "http_client": http_client_factory(),
    }


def provide_fabric_client(*, api_version: str) -> AzureOpenAI:
    """Build a sync client with runtime-managed Fabric authentication."""
    return _FabricOpenAI(**_fabric_client_kwargs(api_version=api_version))


def provide_async_fabric_client(*, api_version: str) -> AsyncAzureOpenAI:
    """Build an async client only when the Fabric runtime supports it."""
    return _AsyncFabricOpenAI(**_fabric_client_kwargs(api_version=api_version, async_client=True))


REQUIRED_VARS: list[str] = [
    "AZURE_TENANT_ID",
    "AZURE_CLIENT_ID",
    "KEY_VAULT_URL",
    "KEY_VAULT_SECRET_NAME",
]

_ALL_AUTH_VARS: list[str] = [*REQUIRED_VARS, "AZURE_CLIENT_SECRET"]

_ENV_DESCRIPTIONS: dict[str, str] = {
    "AZURE_TENANT_ID": "Entra ID tenant ID (directory containing the Service Principal)",
    "AZURE_CLIENT_ID": "Service Principal (App Registration) client ID",
    "KEY_VAULT_URL": "Key Vault URL (the secret reader needs 'Key Vault Secrets User' or equivalent access)",
    "KEY_VAULT_SECRET_NAME": "Secret name in Key Vault (stores the SP client secret)",
}

_ENV_EXAMPLES: dict[str, str] = {
    "AZURE_TENANT_ID": '"your-tenant-id"',
    "AZURE_CLIENT_ID": '"your-client-id"',
    "KEY_VAULT_URL": '"https://YOUR-KEYVAULT.vault.azure.net/"',
    "KEY_VAULT_SECRET_NAME": '"your-secret-name"',
}

_AUTH_FLOW_GUIDE = (
    "Authentication to your own Azure OpenAI resource from Microsoft Fabric:\n"
    "  1. notebookutils.credentials.getSecret() reads Key Vault using the current user credentials\n"
    "     (verify the execution identity for scheduled notebooks; Workspace Identity is not automatic)\n"
    "  2. Key Vault stores the client secret of a Service Principal (App Registration)\n"
    '  3. Assign "Cognitive Services OpenAI User" to the Service Principal on the Azure OpenAI resource\n'
    "  4. openaivec retrieves the secret only when the Entra authentication route is selected\n"
    "     and builds a ClientSecretCredential; retrieval failures stop authentication\n"
    "  5. The credential authenticates as the Service Principal with a refreshable token provider"
)

_SETUP_GUIDE = (
    "Setup steps:\n"
    "  1. Create a Service Principal (App Registration) in Entra ID\n"
    '  2. Assign "Cognitive Services OpenAI User" on your Azure OpenAI resource\n'
    "  3. Store the Service Principal's client secret in Azure Key Vault\n"
    '  4. Grant the identity reading the secret "Key Vault Secrets User" or equivalent access\n'
    "  5. Set these environment variables before importing openaivec in your Fabric notebook:\n"
    '     os.environ["AZURE_TENANT_ID"] = "<your-tenant-id>"\n'
    '     os.environ["AZURE_CLIENT_ID"] = "<your-client-id>"\n'
    '     os.environ["KEY_VAULT_URL"] = "<your-keyvault-url>"\n'
    '     os.environ["KEY_VAULT_SECRET_NAME"] = "<your-secret-name>"\n'
    "  6. Set AZURE_OPENAI_BASE_URL to https://YOUR-RESOURCE.openai.azure.com/openai/v1/\n"
    "  7. Configure Spark executor authentication separately; driver identity is not propagated automatically"
)


def is_fabric_environment() -> bool:
    """Detect whether the current runtime is a Microsoft Fabric notebook.

    Checks for the ``notebookutils`` package (installed at
    ``site-packages/notebookutils`` on the Fabric runtime) and verifies
    that the ``credentials.getSecret`` capability is available.

    Returns:
        bool: ``True`` if running inside a Fabric notebook with Key Vault
            support, ``False`` otherwise.
    """
    try:
        nbu = import_module("notebookutils")

        return hasattr(nbu, "credentials") and callable(getattr(nbu.credentials, "getSecret", None))
    except ImportError:
        return False


def is_auth_configured() -> bool:
    """Check whether service-principal authentication is fully configured.

    Returns ``True`` when either path is ready:

    * **Key Vault path** — all ``REQUIRED_VARS`` are set (the library
      retrieves ``AZURE_CLIENT_SECRET`` from Key Vault automatically).
    * **Direct path** — ``AZURE_TENANT_ID``, ``AZURE_CLIENT_ID``, and
      ``AZURE_CLIENT_SECRET`` are all set explicitly.

    Returns:
        bool: ``True`` if authentication can proceed.
    """
    kv_path = all(os.getenv(name) for name in REQUIRED_VARS)
    direct_path = all(os.getenv(name) for name in ("AZURE_TENANT_ID", "AZURE_CLIENT_ID", "AZURE_CLIENT_SECRET"))
    return kv_path or direct_path


def is_partially_configured() -> bool:
    """Check whether some but not all Fabric authentication variables are set.

    Considers both the Key Vault path variables and ``AZURE_CLIENT_SECRET``
    (direct path).  Returns ``False`` when :func:`is_auth_configured`
    returns ``True`` or when no relevant variables are set at all.

    Returns:
        bool: ``True`` if at least one relevant variable is set
            but authentication is not yet fully configured.
    """
    if is_auth_configured():
        return False
    return any(os.getenv(name) for name in _ALL_AUTH_VARS)


def retrieve_client_secret(*, kv_url: str | None = None, secret_name: str | None = None) -> str | None:
    """Retrieve the client secret from Key Vault on a Fabric driver.

    Calls ``notebookutils.credentials.getSecret`` when both ``kv_url`` and
    ``secret_name`` are provided.

    Args:
        kv_url (str | None): Key Vault URL.
        secret_name (str | None): Secret name in Key Vault.

    Returns:
        str | None: The secret value, or ``None`` when retrieval is skipped.

    Raises:
        ValueError: Key Vault returns an empty secret. Retrieval errors propagate
            to the caller without falling back to another identity.
    """
    if not kv_url or not secret_name:
        return None

    nbu = import_module("notebookutils")

    client_secret: str = nbu.credentials.getSecret(kv_url, secret_name)
    if not client_secret:
        raise ValueError("Key Vault returned an empty client secret.")
    return client_secret


def log_environment_info() -> None:
    """Log Fabric detection and the current authentication configuration status.

    Emits an INFO-level log confirming Fabric detection and listing which
    credential variables are set.
    """
    if is_auth_configured():
        lines = ["Microsoft Fabric environment detected. Authentication is fully configured."]
        for var_name in _ALL_AUTH_VARS:
            if os.getenv(var_name):
                lines.append(f"  ✓ {var_name}")
        _LOGGER.info("\n".join(lines))
    else:
        lines = [
            "Microsoft Fabric environment detected.",
            "Authentication variable status:",
        ]
        for var_name, description in _ENV_DESCRIPTIONS.items():
            status = "✓" if os.getenv(var_name) else "✗"
            lines.append(f"  {status} {var_name} — {description}")
        _LOGGER.info("\n".join(lines))


def warn_incomplete_configuration() -> None:
    """Emit a ``UserWarning`` with detailed setup guidance for Fabric auth.

    Called when the Fabric runtime is detected and some — but not all — of
    the required environment variables are set, which strongly suggests the
    user intends to use Service Principal authentication but has an
    incomplete setup.

    The warning includes the full authentication flow description, per-variable
    status with examples, and step-by-step setup instructions.
    """
    lines = [
        "Microsoft Fabric environment detected but authentication is not fully configured.",
        "DefaultAzureCredential does not automatically use the Fabric notebook or workspace identity.",
        "For Fabric built-in models, call openaivec.setup_fabric() instead of configuring a service principal.",
        "",
        _AUTH_FLOW_GUIDE,
        "",
        "Required environment variables:",
    ]
    for var_name in REQUIRED_VARS:
        value = os.getenv(var_name)
        desc = _ENV_DESCRIPTIONS[var_name]
        if value:
            lines.append(f"  ✓ {var_name} — {desc}")
        else:
            example = _ENV_EXAMPLES[var_name]
            lines.append(f"  ✗ {var_name} — {desc}")
            lines.append(f"    → export {var_name}={example}")
    lines.append("")
    lines.append(_SETUP_GUIDE)
    lines.append("")
    lines.append("Note: AZURE_OPENAI_BASE_URL is also required for Azure OpenAI endpoint configuration.")
    lines.append(
        "If you change these variables after importing openaivec, configure explicit clients with "
        "openaivec.set_client()/set_async_client(), or restart the session."
    )
    warnings.warn("\n".join(lines), UserWarning, stacklevel=3)


def build_credentials_error_section() -> list[str]:
    """Build error-message lines describing Fabric auth setup for credentials errors.

    Includes the authentication flow explanation, per-variable status with
    examples, and step-by-step setup instructions.

    Returns:
        list[str]: Lines to append to a credentials error message.
    """
    lines: list[str] = [
        "",
        "Option 3: Use Fabric built-in models (Fabric environment detected)",
        "  import openaivec",
        "  openaivec.setup_fabric()",
        "  No API key or Azure OpenAI resource required; usage is billed to Fabric capacity.",
        "",
        "Option 4: Configure a service principal for your own Azure OpenAI resource using Key Vault",
        "",
        _AUTH_FLOW_GUIDE,
        "",
        "Required environment variables:",
    ]
    for var_name in REQUIRED_VARS:
        var_value = os.getenv(var_name)
        desc = _ENV_DESCRIPTIONS[var_name]
        example = _ENV_EXAMPLES[var_name]
        if var_value:
            lines.append(f"  ✓ {var_name} — {desc}")
        else:
            lines.append(f"  ✗ {var_name} — {desc}")
            lines.append(f"    → export {var_name}={example}")
    lines.append("")
    lines.append(_SETUP_GUIDE)
    return lines
