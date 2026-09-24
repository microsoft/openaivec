#!/usr/bin/env python3
"""Inspect openaivec dependencies and authentication configuration without network access."""

from __future__ import annotations

import argparse
import os
import sys
from importlib.metadata import PackageNotFoundError, version
from urllib.parse import urlsplit

_PLACEHOLDERS = frozenset(
    {
        "place_holder_for_fabric_internal",
        "your-openai-api-key",
        "your-azure-api-key",
    }
)


def _distribution_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _is_usable_secret(value: str | None) -> bool:
    if not value or not value.strip():
        return False
    normalized = value.strip().lower()
    return normalized not in _PLACEHOLDERS and "<" not in normalized and ">" not in normalized


def _azure_endpoint_warning(base_url: str) -> str | None:
    parsed = urlsplit(base_url)
    if parsed.scheme != "https" or not parsed.netloc:
        return "AZURE_OPENAI_BASE_URL must be an HTTPS URL."
    if not parsed.path.rstrip("/").endswith("/openai/v1"):
        return "AZURE_OPENAI_BASE_URL should end with /openai/v1/."
    return None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check openaivec, DuckDB, and selected authentication configuration without making a network request."
        )
    )
    parser.add_argument(
        "--fabric",
        action="store_true",
        help="Validate the package environment for explicit openaivec.setup_fabric() use.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    errors: list[str] = []
    warnings: list[str] = []

    if sys.version_info < (3, 10):
        errors.append("Python 3.10 or newer is required.")

    for distribution in ("openaivec", "duckdb"):
        installed_version = _distribution_version(distribution)
        if installed_version is None:
            errors.append(f"{distribution} is not installed.")
        else:
            print(f"{distribution}: {installed_version}")

    openai_key = os.getenv("OPENAI_API_KEY")
    azure_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_base_url = os.getenv("AZURE_OPENAI_BASE_URL")
    tenant_id = os.getenv("AZURE_TENANT_ID")
    client_id = os.getenv("AZURE_CLIENT_ID")
    client_secret = os.getenv("AZURE_CLIENT_SECRET")

    if args.fabric:
        print("authentication route: Fabric built-in models")
        print("required action: call openaivec.setup_fabric() before registering DuckDB UDFs")
        if any(_is_usable_secret(value) for value in (openai_key, azure_key, client_secret)):
            warnings.append("API credentials are set but setup_fabric() explicitly replaces the default clients.")
    elif _is_usable_secret(openai_key):
        print("authentication route: OpenAI API key")
        if azure_base_url or _is_usable_secret(azure_key):
            warnings.append("OPENAI_API_KEY takes precedence; Azure OpenAI settings will not be selected.")
    elif azure_base_url:
        endpoint_warning = _azure_endpoint_warning(azure_base_url)
        if endpoint_warning:
            warnings.append(endpoint_warning)
        if _is_usable_secret(azure_key):
            print("authentication route: Azure OpenAI API key")
        elif _is_usable_secret(client_secret):
            print("authentication route: Azure OpenAI Entra ID service principal")
            if not tenant_id or not client_id:
                errors.append("AZURE_CLIENT_SECRET requires both AZURE_TENANT_ID and AZURE_CLIENT_ID for this route.")
        else:
            print("authentication route: Azure OpenAI Entra ID via DefaultAzureCredential")
            if bool(tenant_id) != bool(client_id):
                warnings.append(
                    "Only one of AZURE_TENANT_ID and AZURE_CLIENT_ID is set; "
                    "verify the intended DefaultAzureCredential."
                )
    else:
        if _is_usable_secret(azure_key) or any((tenant_id, client_id, _is_usable_secret(client_secret))):
            errors.append(
                "Azure authentication values are set but AZURE_OPENAI_BASE_URL is missing. "
                "Set the Azure v1 endpoint ending in /openai/v1/."
            )
        else:
            errors.append(
                "No route is configured. Set OPENAI_API_KEY, or set AZURE_OPENAI_BASE_URL with an Azure API key "
                "or Entra ID, or rerun with --fabric in a supported Fabric notebook."
            )

    for warning in warnings:
        print(f"warning: {warning}", file=sys.stderr)
    for error in errors:
        print(f"error: {error}", file=sys.stderr)

    print("network check: not performed; use a one-row synthetic smoke test before sending user data")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
