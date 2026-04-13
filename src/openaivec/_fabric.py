"""Microsoft Fabric environment detection and Key Vault authentication.

When running inside a Fabric notebook, ``notebookutils`` is available as a
built-in and provides ``credentials.getSecret`` for retrieving secrets from
Azure Key Vault.  This module exposes helpers that:

1. Detect the Fabric runtime.
2. Check whether the required environment variables are configured.
3. Build a ``Callable[[], str]`` bearer-token provider backed by
   ``ClientSecretCredential`` whose client secret comes from Key Vault.
4. Log / warn / format environment-variable status for notebook UX.

All heavy Azure SDK imports are at the top level (they are already
transitive dependencies of the library), while ``builtins.notebookutils``
is accessed only at call time.
"""

import logging
import os
import warnings
from collections.abc import Callable

from azure.identity import ClientSecretCredential

__all__ = []

_LOGGER = logging.getLogger(__name__)

REQUIRED_VARS: list[str] = [
    "AZURE_TENANT_ID",
    "AZURE_APP_CLIENT_ID",
    "KEY_VAULT_URL",
    "KEY_VAULT_SECRET_NAME",
]

_ENV_DESCRIPTIONS: dict[str, str] = {
    "AZURE_TENANT_ID": "Azure AD tenant ID (directory containing the Service Principal)",
    "AZURE_APP_CLIENT_ID": "Service Principal (App Registration) client ID",
    "KEY_VAULT_URL": "Key Vault URL (Workspace must have access via 'Key Vault Secrets User' role)",
    "KEY_VAULT_SECRET_NAME": "Secret name in Key Vault (stores the Service Principal's client secret)",
}

_ENV_EXAMPLES: dict[str, str] = {
    "AZURE_TENANT_ID": '"your-tenant-id"',
    "AZURE_APP_CLIENT_ID": '"your-client-id"',
    "KEY_VAULT_URL": '"https://YOUR-KEYVAULT.vault.azure.net/"',
    "KEY_VAULT_SECRET_NAME": '"your-secret-name"',
}

_AUTH_FLOW_GUIDE = (
    "Authentication flow in Microsoft Fabric:\n"
    "  1. Fabric Workspace identity accesses Azure Key Vault\n"
    '     (Workspace must have "Key Vault Secrets User" role on the Key Vault)\n'
    "  2. Key Vault stores the client secret of a Service Principal (App Registration)\n"
    '  3. The Service Principal must have the "AI User" role on the Azure AI Foundry resource\n'
    "  4. openaivec retrieves the secret via notebookutils, creates ClientSecretCredential,\n"
    "     and obtains bearer tokens for Azure OpenAI"
)

_SETUP_GUIDE = (
    "Setup steps:\n"
    "  1. Create a Service Principal (App Registration) in Azure AD\n"
    '  2. Assign "AI User" role to the Service Principal on your Azure AI Foundry resource\n'
    "  3. Store the Service Principal's client secret in Azure Key Vault\n"
    '  4. Grant Fabric Workspace identity "Key Vault Secrets User" role on the Key Vault\n'
    "  5. Set the environment variables listed above in your Fabric notebook"
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
    """Check whether all Fabric Key Vault authentication variables are set.

    Returns:
        bool: ``True`` if every variable in ``REQUIRED_VARS`` is
            present and non-empty in the environment.
    """
    return all(os.getenv(name) for name in REQUIRED_VARS)


def is_partially_configured() -> bool:
    """Check whether some but not all Fabric Key Vault variables are set.

    Returns:
        bool: ``True`` if at least one variable in ``REQUIRED_VARS`` is set
            but not all of them are present.
    """
    set_count = sum(1 for name in REQUIRED_VARS if os.getenv(name))
    return 0 < set_count < len(REQUIRED_VARS)


def log_environment_info() -> None:
    """Log Fabric detection and the current authentication configuration status.

    Emits an INFO-level log confirming Fabric detection and listing which
    Fabric-specific variables are set.
    """
    if is_auth_configured():
        lines = ["Microsoft Fabric environment detected. Key Vault authentication is fully configured."]
        for var_name in REQUIRED_VARS:
            lines.append(f"  ✓ {var_name}")
        _LOGGER.info("\n".join(lines))
    else:
        lines = [
            "Microsoft Fabric environment detected.",
            "Key Vault authentication variable status:",
        ]
        for var_name, description in _ENV_DESCRIPTIONS.items():
            status = "✓" if os.getenv(var_name) else "✗"
            lines.append(f"  {status} {var_name} — {description}")
        _LOGGER.info("\n".join(lines))


def warn_incomplete_configuration() -> None:
    """Emit a ``UserWarning`` with detailed setup guidance for Fabric auth.

    Called when the Fabric runtime is detected and some — but not all — of
    the required Key Vault environment variables are set, which strongly
    suggests the user intends to use Fabric authentication but has an
    incomplete setup.

    The warning includes the full authentication flow description, per-variable
    status with examples, and step-by-step setup instructions.
    """
    lines = [
        "Microsoft Fabric environment detected but Key Vault authentication is not fully configured.",
        "Falling back to DefaultAzureCredential (this may fail if Entra ID is not configured).",
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


def build_token_provider() -> Callable[[], str]:
    """Build a bearer-token provider for Fabric using Key Vault and ClientSecretCredential.

    Retrieves the app client secret from Azure Key Vault via
    ``notebookutils.credentials.getSecret``, creates a
    ``ClientSecretCredential``, and returns a callable that produces
    fresh bearer tokens for Azure OpenAI.

    Returns:
        Callable[[], str]: A callable that returns a bearer token string.

    Raises:
        ValueError: If any required Fabric environment variable is missing.
    """
    env_values = {name: os.getenv(name, "") for name in REQUIRED_VARS}
    missing = [name for name, val in env_values.items() if not val]
    if missing:
        raise ValueError(
            "Microsoft Fabric environment detected but required environment variables are missing:\n"
            + "\n".join(f"  ✗ {name} — {_ENV_DESCRIPTIONS[name]}" for name in missing)
            + "\n\n"
            + _AUTH_FLOW_GUIDE
            + "\n\n"
            + _SETUP_GUIDE
        )

    import builtins

    nbu = getattr(builtins, "notebookutils")
    client_secret: str = nbu.credentials.getSecret(env_values["KEY_VAULT_URL"], env_values["KEY_VAULT_SECRET_NAME"])

    credential = ClientSecretCredential(
        tenant_id=env_values["AZURE_TENANT_ID"],
        client_id=env_values["AZURE_APP_CLIENT_ID"],
        client_secret=client_secret,
    )

    def token_provider() -> str:
        return credential.get_token("https://cognitiveservices.azure.com/.default").token

    return token_provider


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
