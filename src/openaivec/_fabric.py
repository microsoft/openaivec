"""Microsoft Fabric environment detection and Key Vault authentication.

When running inside a Fabric notebook, ``notebookutils`` is available as a
built-in and provides ``credentials.getSecret`` for retrieving secrets from
Azure Key Vault.  This module exposes helpers that:

1. Detect the Fabric runtime.
2. Check whether the required environment variables are configured.
3. Build a ``Callable[[], str]`` bearer-token provider backed by
   ``ClientSecretCredential`` whose client secret comes from Key Vault.
4. Log / format environment-variable status for notebook UX.

All heavy Azure SDK imports are at the top level (they are already
transitive dependencies of the library), while ``builtins.notebookutils``
is accessed only at call time.
"""

import logging
import os
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
    "AZURE_TENANT_ID": "Azure AD tenant ID",
    "AZURE_APP_CLIENT_ID": "Azure app registration client ID",
    "KEY_VAULT_URL": 'Key Vault URL (e.g. "https://YOUR-KEYVAULT.vault.azure.net/")',
    "KEY_VAULT_SECRET_NAME": "Key Vault secret name storing the app client secret",
}

_ENV_EXAMPLES: dict[str, str] = {
    "AZURE_TENANT_ID": '"your-tenant-id"',
    "AZURE_APP_CLIENT_ID": '"your-client-id"',
    "KEY_VAULT_URL": '"https://YOUR-KEYVAULT.vault.azure.net/"',
    "KEY_VAULT_SECRET_NAME": '"your-secret-name"',
}


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


def log_environment_info() -> None:
    """Log Fabric detection and required environment variable status.

    Emits an INFO-level log showing which Fabric-specific variables are
    set and which are missing so that notebook users get immediate feedback.
    """
    lines = [
        "Microsoft Fabric environment detected. "
        "Set the following variables to enable Key Vault + ClientSecretCredential authentication:",
    ]
    for var_name, description in _ENV_DESCRIPTIONS.items():
        value = os.getenv(var_name)
        status = "✓" if value else "✗"
        lines.append(f"  {status} {var_name} — {description}")
    _LOGGER.info("\n".join(lines))


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
            + "\n".join(f"  ✗ {name} is not set" for name in missing)
            + "\n\nSet these variables to enable Key Vault + ClientSecretCredential authentication."
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
    """Build error-message lines describing missing Fabric env vars.

    Returns:
        list[str]: Lines to append to a credentials error message, including
            a header and per-variable status with examples.
    """
    lines: list[str] = [
        "",
        "Fabric environment detected — Key Vault authentication also requires:",
    ]
    for var_name in REQUIRED_VARS:
        var_value = os.getenv(var_name)
        example = _ENV_EXAMPLES[var_name]
        if var_value:
            lines.append(f"  ✓ {var_name} is set")
        else:
            lines.append(f"  ✗ {var_name} is not set")
            lines.append(f"    Example: export {var_name}={example}")
    return lines
