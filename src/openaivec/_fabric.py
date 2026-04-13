"""Microsoft Fabric environment detection and Entra ID authentication.

When running inside a Fabric notebook, ``notebookutils`` is available as a
built-in and provides ``credentials.getSecret`` for retrieving secrets from
Azure Key Vault.  This module exposes helpers that:

1. Detect the Fabric runtime.
2. Automatically retrieve the Service Principal client secret from Key Vault
   via ``notebookutils.credentials.getSecret`` and set it as
   ``AZURE_CLIENT_SECRET`` for ``DefaultAzureCredential``.
3. Check whether the required environment variables are configured.
4. Log / warn / format environment-variable status for notebook UX.

On the Fabric driver, set ``AZURE_TENANT_ID``, ``AZURE_CLIENT_ID``,
``KEY_VAULT_URL``, and ``KEY_VAULT_SECRET_NAME``.  The library retrieves the
client secret from Key Vault automatically and sets ``AZURE_CLIENT_SECRET``
so that ``DefaultAzureCredential`` authenticates as the Service Principal.
"""

import logging
import os
import warnings

__all__ = []

_LOGGER = logging.getLogger(__name__)

REQUIRED_VARS: list[str] = [
    "AZURE_TENANT_ID",
    "AZURE_CLIENT_ID",
    "KEY_VAULT_URL",
    "KEY_VAULT_SECRET_NAME",
]

_ALL_AUTH_VARS: list[str] = [*REQUIRED_VARS, "AZURE_CLIENT_SECRET"]

_ENV_DESCRIPTIONS: dict[str, str] = {
    "AZURE_TENANT_ID": "Azure AD tenant ID (directory containing the Service Principal)",
    "AZURE_CLIENT_ID": "Service Principal (App Registration) client ID",
    "KEY_VAULT_URL": "Key Vault URL (Workspace must have 'Key Vault Secrets User' role)",
    "KEY_VAULT_SECRET_NAME": "Secret name in Key Vault (stores the SP client secret)",
}

_ENV_EXAMPLES: dict[str, str] = {
    "AZURE_TENANT_ID": '"your-tenant-id"',
    "AZURE_CLIENT_ID": '"your-client-id"',
    "KEY_VAULT_URL": '"https://YOUR-KEYVAULT.vault.azure.net/"',
    "KEY_VAULT_SECRET_NAME": '"your-secret-name"',
}

_AUTH_FLOW_GUIDE = (
    "Authentication flow in Microsoft Fabric:\n"
    "  1. Fabric Workspace identity accesses Azure Key Vault\n"
    '     (Workspace must have "Key Vault Secrets User" role on the Key Vault)\n'
    "  2. Key Vault stores the client secret of a Service Principal (App Registration)\n"
    '  3. The Service Principal must have the "AI User" role on the Azure AI Foundry resource\n'
    "  4. openaivec retrieves the secret via notebookutils.credentials.getSecret()\n"
    "     and sets AZURE_CLIENT_SECRET automatically\n"
    "  5. DefaultAzureCredential detects the environment variables\n"
    "     and authenticates as the Service Principal"
)

_SETUP_GUIDE = (
    "Setup steps:\n"
    "  1. Create a Service Principal (App Registration) in Azure AD\n"
    '  2. Assign "AI User" role to the Service Principal on your Azure AI Foundry resource\n'
    "  3. Store the Service Principal's client secret in Azure Key Vault\n"
    '  4. Grant Fabric Workspace identity "Key Vault Secrets User" role on the Key Vault\n'
    "  5. Set the following environment variables in your Fabric notebook:\n"
    '     os.environ["AZURE_TENANT_ID"] = "<your-tenant-id>"\n'
    '     os.environ["AZURE_CLIENT_ID"] = "<your-client-id>"\n'
    '     os.environ["KEY_VAULT_URL"] = "<your-keyvault-url>"\n'
    '     os.environ["KEY_VAULT_SECRET_NAME"] = "<your-secret-name>"\n'
    "  6. openaivec automatically retrieves the secret from Key Vault on import\n"
    "  7. For Spark executors, propagate credentials via setup_entra_id() or sc.environment"
)


def is_fabric_environment() -> bool:
    """Detect whether the current runtime is a Microsoft Fabric notebook.

    Checks for the ``notebookutils`` built-in injected by Fabric and verifies
    that the ``credentials.getSecret`` capability is available.

    Returns:
        bool: ``True`` if running inside a Fabric notebook with Key Vault
            support, ``False`` otherwise.
    """
    try:
        import builtins

        nbu = getattr(builtins, "notebookutils", None)
        if nbu is None:
            return False
        return hasattr(nbu, "credentials") and callable(getattr(nbu.credentials, "getSecret", None))
    except Exception:
        return False


def is_auth_configured() -> bool:
    """Check whether Fabric authentication is fully configured.

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


def retrieve_client_secret() -> bool:
    """Retrieve the client secret from Key Vault on a Fabric driver.

    When ``KEY_VAULT_URL`` and ``KEY_VAULT_SECRET_NAME`` are set and
    ``AZURE_CLIENT_SECRET`` is not yet present, calls
    ``notebookutils.credentials.getSecret`` and stores the result in
    ``os.environ["AZURE_CLIENT_SECRET"]``.

    Returns:
        bool: ``True`` if the secret was successfully retrieved and set.
    """
    if os.getenv("AZURE_CLIENT_SECRET"):
        return False

    kv_url = os.getenv("KEY_VAULT_URL")
    secret_name = os.getenv("KEY_VAULT_SECRET_NAME")
    if not kv_url or not secret_name:
        return False

    try:
        import builtins

        nbu = getattr(builtins, "notebookutils")
        client_secret: str = nbu.credentials.getSecret(kv_url, secret_name)
        os.environ["AZURE_CLIENT_SECRET"] = client_secret
        _LOGGER.info("Retrieved client secret from Key Vault (%s) and set AZURE_CLIENT_SECRET.", kv_url)
        return True
    except Exception as exc:
        _LOGGER.warning("Failed to retrieve client secret from Key Vault: %s", exc)
        return False


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
        "DefaultAzureCredential will attempt other identity sources (managed identity, Azure CLI, etc.).",
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
    lines.append(
        "Note: AZURE_OPENAI_BASE_URL and AZURE_OPENAI_API_VERSION are also required "
        "for Azure OpenAI endpoint configuration."
    )
    lines.append(
        "If you set these variables after importing openaivec, call "
        "openaivec.set_default_registrations() to re-initialize."
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
        "Option 3: Configure Fabric Key Vault authentication (Fabric environment detected)",
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
