"""Microsoft Fabric environment detection and Entra ID authentication guidance.

When running inside a Fabric notebook, ``notebookutils`` is available as a
built-in and provides ``credentials.getSecret`` for retrieving secrets from
Azure Key Vault.  This module exposes helpers that:

1. Detect the Fabric runtime.
2. Check whether the required Service Principal environment variables are set.
3. Log / warn / format environment-variable status for notebook UX.

Authentication uses ``DefaultAzureCredential`` which automatically detects
``AZURE_TENANT_ID``, ``AZURE_CLIENT_ID``, and ``AZURE_CLIENT_SECRET`` via
its ``EnvironmentCredential`` source.  In Fabric, the client secret is
typically retrieved from Key Vault via ``notebookutils.credentials.getSecret``
and then set as ``AZURE_CLIENT_SECRET``.
"""

import logging
import os
import warnings

__all__ = []

_LOGGER = logging.getLogger(__name__)

REQUIRED_VARS: list[str] = [
    "AZURE_TENANT_ID",
    "AZURE_CLIENT_ID",
    "AZURE_CLIENT_SECRET",
]

_ENV_DESCRIPTIONS: dict[str, str] = {
    "AZURE_TENANT_ID": "Azure AD tenant ID (directory containing the Service Principal)",
    "AZURE_CLIENT_ID": "Service Principal (App Registration) client ID",
    "AZURE_CLIENT_SECRET": "Service Principal client secret (retrieve from Key Vault in Fabric)",
}

_ENV_EXAMPLES: dict[str, str] = {
    "AZURE_TENANT_ID": '"your-tenant-id"',
    "AZURE_CLIENT_ID": '"your-client-id"',
    "AZURE_CLIENT_SECRET": '"<retrieved from Key Vault>"',
}

_AUTH_FLOW_GUIDE = (
    "Authentication flow in Microsoft Fabric:\n"
    "  1. Fabric Workspace identity accesses Azure Key Vault\n"
    '     (Workspace must have "Key Vault Secrets User" role on the Key Vault)\n'
    "  2. Key Vault stores the client secret of a Service Principal (App Registration)\n"
    '  3. The Service Principal must have the "AI User" role on the Azure AI Foundry resource\n'
    "  4. Retrieve the secret via notebookutils and set as AZURE_CLIENT_SECRET\n"
    "  5. DefaultAzureCredential automatically detects the environment variables\n"
    "     and authenticates as the Service Principal"
)

_SETUP_GUIDE = (
    "Setup steps:\n"
    "  1. Create a Service Principal (App Registration) in Azure AD\n"
    '  2. Assign "AI User" role to the Service Principal on your Azure AI Foundry resource\n'
    "  3. Store the Service Principal's client secret in Azure Key Vault\n"
    '  4. Grant Fabric Workspace identity "Key Vault Secrets User" role on the Key Vault\n'
    "  5. In your notebook, retrieve the secret and set environment variables:\n"
    '     secret = notebookutils.credentials.getSecret("<kv-url>", "<secret-name>")\n'
    '     os.environ["AZURE_TENANT_ID"] = "<your-tenant-id>"\n'
    '     os.environ["AZURE_CLIENT_ID"] = "<your-client-id>"\n'
    '     os.environ["AZURE_CLIENT_SECRET"] = secret\n'
    "  6. For Spark executors, propagate credentials via setup_entra_id() or sc.environment"
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
    """Log Fabric detection and the current Service Principal credential status.

    Emits an INFO-level log confirming Fabric detection and listing which
    credential variables are set.
    """
    if is_auth_configured():
        lines = ["Microsoft Fabric environment detected. Service Principal credentials are fully configured."]
        for var_name in REQUIRED_VARS:
            lines.append(f"  ✓ {var_name}")
        _LOGGER.info("\n".join(lines))
    else:
        lines = [
            "Microsoft Fabric environment detected.",
            "Service Principal credential status:",
        ]
        for var_name, description in _ENV_DESCRIPTIONS.items():
            status = "✓" if os.getenv(var_name) else "✗"
            lines.append(f"  {status} {var_name} — {description}")
        _LOGGER.info("\n".join(lines))


def warn_incomplete_configuration() -> None:
    """Emit a ``UserWarning`` with detailed setup guidance for Fabric auth.

    Called when the Fabric runtime is detected and some — but not all — of
    the required Service Principal environment variables are set, which
    strongly suggests the user intends to use Service Principal authentication
    but has an incomplete setup.

    The warning includes the full authentication flow description, per-variable
    status with examples, and step-by-step setup instructions.
    """
    lines = [
        "Microsoft Fabric environment detected but Service Principal credentials are not fully configured.",
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
        "Option 3: Configure Service Principal authentication (Fabric environment detected)",
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
