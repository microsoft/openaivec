import os
import sys
import warnings
from unittest.mock import MagicMock, patch

import pytest
from openai import AsyncOpenAI, OpenAI

from openaivec._model import AzureClientSecret
from openaivec._provider import (
    CONTAINER,
    _build_client_kwargs,
    _build_missing_credentials_error,
    _ensure_v1,
    provide_async_openai_client,
    provide_openai_client,
    set_default_registrations,
)

_ENV_KEYS = [
    "OPENAI_API_KEY",
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_BASE_URL",
    "AZURE_TENANT_ID",
    "AZURE_CLIENT_ID",
    "AZURE_CLIENT_SECRET",
    "KEY_VAULT_URL",
    "KEY_VAULT_SECRET_NAME",
]


def _clear_env() -> None:
    for key in _ENV_KEYS:
        os.environ.pop(key, None)


def _set_env(**env_vars: str) -> None:
    _clear_env()
    for key, value in env_vars.items():
        os.environ[key] = value
    set_default_registrations()


@pytest.fixture(autouse=True)
def _env(reset_environment):
    _clear_env()
    set_default_registrations()
    yield
    _clear_env()
    set_default_registrations()


# ---------------------------------------------------------------------------
# _ensure_v1
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "base_url",
    [
        "https://x.services.ai.azure.com/openai/v1/",
        "https://x.services.ai.azure.com/openai/v1",
        "https://x.services.ai.azure.com/openai",
        "https://x.services.ai.azure.com/openai/",
        "https://x.services.ai.azure.com",
        "https://x.services.ai.azure.com/",
    ],
)
def test_ensure_v1_normalizes_url(base_url):
    assert _ensure_v1(base_url) == "https://x.services.ai.azure.com/openai/v1/"


# ---------------------------------------------------------------------------
# OpenAI direct
# ---------------------------------------------------------------------------


class TestOpenAIDirect:
    def test_sync_with_openai_key(self):
        _set_env(OPENAI_API_KEY="sk-test")
        client = provide_openai_client()
        assert isinstance(client, OpenAI)
        assert client.api_key == "sk-test"

    def test_async_with_openai_key(self):
        _set_env(OPENAI_API_KEY="sk-test")
        client = provide_async_openai_client()
        assert isinstance(client, AsyncOpenAI)
        assert client.api_key == "sk-test"

    def test_openai_key_wins_over_azure(self):
        _set_env(
            OPENAI_API_KEY="sk-test",
            AZURE_OPENAI_API_KEY="azure-key",
            AZURE_OPENAI_BASE_URL="https://x.services.ai.azure.com/openai/v1/",
        )
        client = provide_openai_client()
        assert client.api_key == "sk-test"

    def test_empty_openai_key_falls_through(self):
        _set_env(
            OPENAI_API_KEY="",
            AZURE_OPENAI_API_KEY="azure-key",
            AZURE_OPENAI_BASE_URL="https://x.services.ai.azure.com/openai/v1/",
        )
        client = provide_openai_client()
        assert client.api_key == "azure-key"

    def test_fabric_placeholder_key_falls_through(self):
        _set_env(
            OPENAI_API_KEY="place_holder_for_fabric_internal",
            AZURE_OPENAI_API_KEY="azure-key",
            AZURE_OPENAI_BASE_URL="https://x.services.ai.azure.com/openai/v1/",
        )
        client = provide_openai_client()
        assert client.api_key == "azure-key"

    def test_reinstalls_defaults_after_container_clear(self):
        _set_env(OPENAI_API_KEY="sk-test")
        CONTAINER.clear()
        client = provide_openai_client()
        assert isinstance(client, OpenAI)
        assert client.api_key == "sk-test"


# ---------------------------------------------------------------------------
# Azure API key
# ---------------------------------------------------------------------------


class TestAzureAPIKey:
    def test_sync(self):
        _set_env(
            AZURE_OPENAI_API_KEY="azure-key",
            AZURE_OPENAI_BASE_URL="https://x.services.ai.azure.com/openai/v1/",
        )
        client = provide_openai_client()
        assert isinstance(client, OpenAI)
        assert client.api_key == "azure-key"
        assert str(client.base_url).endswith("/openai/v1/")

    def test_async(self):
        _set_env(
            AZURE_OPENAI_API_KEY="azure-key",
            AZURE_OPENAI_BASE_URL="https://x.services.ai.azure.com/openai/v1/",
        )
        client = provide_async_openai_client()
        assert isinstance(client, AsyncOpenAI)
        assert client.api_key == "azure-key"
        assert str(client.base_url).endswith("/openai/v1/")

    def test_legacy_url_is_normalized_to_v1(self):
        _set_env(
            AZURE_OPENAI_API_KEY="azure-key",
            AZURE_OPENAI_BASE_URL="https://x.services.ai.azure.com",
        )
        client = provide_openai_client()
        assert str(client.base_url).endswith("/openai/v1/")


# ---------------------------------------------------------------------------
# Azure Entra ID
# ---------------------------------------------------------------------------


class TestAzureEntraID:
    def test_sync_falls_back_to_token_provider(self):
        _set_env(AZURE_OPENAI_BASE_URL="https://x.services.ai.azure.com/openai/v1/")
        kwargs = _build_client_kwargs()
        assert callable(kwargs["api_key"])
        assert kwargs["base_url"].endswith("/openai/v1/")
        client = provide_openai_client()
        assert isinstance(client, OpenAI)
        assert str(client.base_url).endswith("/openai/v1/")

    def test_async_uses_callable_token_provider(self):
        _set_env(AZURE_OPENAI_BASE_URL="https://x.services.ai.azure.com/openai/v1/")
        kwargs = _build_client_kwargs()
        assert callable(kwargs["api_key"])
        client = provide_async_openai_client()
        assert isinstance(client, AsyncOpenAI)

    def test_empty_azure_key_falls_back_to_entra(self):
        _set_env(
            AZURE_OPENAI_API_KEY="",
            AZURE_OPENAI_BASE_URL="https://x.services.ai.azure.com/openai/v1/",
        )
        kwargs = _build_client_kwargs()
        assert callable(kwargs["api_key"])


# ---------------------------------------------------------------------------
# Missing credentials
# ---------------------------------------------------------------------------


class TestMissingCredentials:
    def test_sync_raises(self):
        _clear_env()
        set_default_registrations()
        with pytest.raises(ValueError, match="No valid OpenAI or Azure OpenAI credentials found"):
            provide_openai_client()

    def test_async_raises(self):
        _clear_env()
        set_default_registrations()
        with pytest.raises(ValueError, match="No valid OpenAI or Azure OpenAI credentials found"):
            provide_async_openai_client()

    def test_error_message_includes_setup_examples(self):
        message = _build_missing_credentials_error(
            openai_api_key=None,
            azure_api_key=None,
            azure_base_url=None,
        )
        assert "OPENAI_API_KEY" in message
        assert "AZURE_OPENAI_API_KEY" in message
        assert "AZURE_OPENAI_BASE_URL" in message
        assert "export OPENAI_API_KEY" in message
        assert "Option 1" in message
        assert "Option 2" in message

    def test_error_marks_set_values(self):
        message = _build_missing_credentials_error(
            openai_api_key="sk-test",
            azure_api_key="azure-key",
            azure_base_url=None,
        )
        assert "✓ OPENAI_API_KEY is set" in message
        assert "✓ AZURE_OPENAI_API_KEY (optional) is set" in message
        assert "✗ AZURE_OPENAI_BASE_URL is not set" in message


# ---------------------------------------------------------------------------
# set_client / set_async_client
# ---------------------------------------------------------------------------


class TestSetClient:
    def test_set_client_registers_instance(self):
        custom = OpenAI(api_key="custom")
        from openaivec._provider import get_client, set_client

        set_client(custom)
        assert get_client() is custom

    def test_set_async_client_registers_instance(self):
        custom = AsyncOpenAI(api_key="custom")
        from openaivec._provider import get_async_client, set_async_client

        set_async_client(custom)
        assert get_async_client() is custom


# ---------------------------------------------------------------------------
# Microsoft Fabric integration
# ---------------------------------------------------------------------------


class TestFabricEnvironment:
    def test_is_fabric_environment_returns_false_by_default(self):
        from openaivec._fabric import is_fabric_environment

        assert is_fabric_environment() is False

    def test_is_fabric_environment_returns_true_with_notebookutils(self):
        from openaivec._fabric import is_fabric_environment

        sys.modules["notebookutils"] = MagicMock()
        try:
            assert is_fabric_environment() is True
        finally:
            sys.modules.pop("notebookutils", None)

    def test_is_fabric_environment_false_without_credentials(self):
        from openaivec._fabric import is_fabric_environment

        sys.modules["notebookutils"] = MagicMock(spec=[])
        try:
            assert is_fabric_environment() is False
        finally:
            sys.modules.pop("notebookutils", None)

    def test_retrieve_client_secret_from_kv(self):
        from openaivec._fabric import retrieve_client_secret

        mock_nbu = MagicMock()
        mock_nbu.credentials.getSecret.return_value = "fake-client-secret"
        sys.modules["notebookutils"] = mock_nbu
        try:
            result = retrieve_client_secret(
                kv_url="https://kv.vault.azure.net/",
                secret_name="my-secret",
            )
            assert result == "fake-client-secret"
            mock_nbu.credentials.getSecret.assert_called_once_with("https://kv.vault.azure.net/", "my-secret")
        finally:
            sys.modules.pop("notebookutils", None)

    def test_retrieve_client_secret_returns_none_when_args_missing(self):
        from openaivec._fabric import retrieve_client_secret

        assert retrieve_client_secret() is None
        assert retrieve_client_secret(kv_url="https://kv.vault.azure.net/") is None
        assert retrieve_client_secret(secret_name="my-secret") is None

    @patch("openaivec._fabric.is_fabric_environment", return_value=True)
    def test_provide_openai_client_uses_dac_in_fabric(self, _mock_fabric):
        sys.modules["notebookutils"] = MagicMock()
        try:
            _set_env(
                AZURE_TENANT_ID="t",
                AZURE_CLIENT_ID="c",
                AZURE_CLIENT_SECRET="s",
                AZURE_OPENAI_BASE_URL="https://x.services.ai.azure.com/openai/v1/",
            )
            kwargs = _build_client_kwargs()
            assert callable(kwargs["api_key"])
            client = provide_openai_client()
            assert isinstance(client, OpenAI)
        finally:
            sys.modules.pop("notebookutils", None)

    @patch("openaivec._fabric.is_fabric_environment", return_value=True)
    def test_provide_async_openai_client_uses_dac_in_fabric(self, _mock_fabric):
        sys.modules["notebookutils"] = MagicMock()
        try:
            _set_env(
                AZURE_TENANT_ID="t",
                AZURE_CLIENT_ID="c",
                AZURE_CLIENT_SECRET="s",
                AZURE_OPENAI_BASE_URL="https://x.services.ai.azure.com/openai/v1/",
            )
            kwargs = _build_client_kwargs()
            assert callable(kwargs["api_key"])
            client = provide_async_openai_client()
            assert isinstance(client, AsyncOpenAI)
        finally:
            sys.modules.pop("notebookutils", None)

    @patch("openaivec._fabric.is_fabric_environment", return_value=True)
    def test_openai_key_wins_over_fabric(self, _mock_fabric):
        sys.modules["notebookutils"] = MagicMock()
        try:
            _set_env(
                OPENAI_API_KEY="sk-test",
                AZURE_TENANT_ID="t",
                AZURE_CLIENT_ID="c",
                AZURE_CLIENT_SECRET="s",
                AZURE_OPENAI_BASE_URL="https://x.services.ai.azure.com/openai/v1/",
            )
            client = provide_openai_client()
            assert client.api_key == "sk-test"
        finally:
            sys.modules.pop("notebookutils", None)

    @patch("openaivec._fabric.is_fabric_environment", return_value=True)
    def test_azure_api_key_wins_over_dac(self, _mock_fabric):
        sys.modules["notebookutils"] = MagicMock()
        try:
            _set_env(
                AZURE_OPENAI_API_KEY="azure-key",
                AZURE_TENANT_ID="t",
                AZURE_CLIENT_ID="c",
                AZURE_CLIENT_SECRET="s",
                AZURE_OPENAI_BASE_URL="https://x.services.ai.azure.com/openai/v1/",
            )
            client = provide_openai_client()
            assert client.api_key == "azure-key"
        finally:
            sys.modules.pop("notebookutils", None)

    @patch("openaivec._fabric.is_fabric_environment", return_value=True)
    def test_error_message_includes_fabric_section(self, _mock_fabric):
        message = _build_missing_credentials_error(
            openai_api_key=None,
            azure_api_key=None,
            azure_base_url=None,
        )
        assert "Fabric environment detected" in message
        assert "AZURE_TENANT_ID" in message
        assert "AZURE_CLIENT_ID" in message
        assert "KEY_VAULT_URL" in message
        assert "KEY_VAULT_SECRET_NAME" in message

    def test_error_message_excludes_fabric_section_outside_fabric(self):
        message = _build_missing_credentials_error(
            openai_api_key=None,
            azure_api_key=None,
            azure_base_url=None,
        )
        assert "Fabric" not in message
        assert "KEY_VAULT_URL" not in message

    @patch("openaivec._fabric.is_fabric_environment", return_value=True)
    def test_partial_config_emits_warning(self, _mock_fabric):
        sys.modules["notebookutils"] = MagicMock()
        try:
            os.environ["AZURE_TENANT_ID"] = "t"
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                set_default_registrations()
            fabric_warnings = [x for x in caught if "Fabric" in str(x.message)]
            assert len(fabric_warnings) == 1
        finally:
            sys.modules.pop("notebookutils", None)

    @patch("openaivec._fabric.is_fabric_environment", return_value=True)
    def test_kv_retrieval_populates_di(self, _mock_fabric):
        mock_nbu = MagicMock()
        mock_nbu.credentials.getSecret.return_value = "kv-secret"
        sys.modules["notebookutils"] = mock_nbu
        try:
            _set_env(
                AZURE_TENANT_ID="t",
                AZURE_CLIENT_ID="c",
                KEY_VAULT_URL="https://kv.vault.azure.net/",
                KEY_VAULT_SECRET_NAME="my-secret",
                AZURE_OPENAI_BASE_URL="https://x.services.ai.azure.com/openai/v1/",
            )
            assert CONTAINER.resolve(AzureClientSecret).value == "kv-secret"
            assert os.environ.get("AZURE_CLIENT_SECRET") is None
        finally:
            sys.modules.pop("notebookutils", None)
