import builtins
import os
import warnings
from unittest.mock import MagicMock, patch

import pytest
from openai import AsyncAzureOpenAI, AsyncOpenAI, AzureOpenAI, OpenAI

from openaivec._provider import (
    CONTAINER,
    _build_missing_credentials_error,
    provide_async_openai_client,
    provide_openai_client,
    set_default_registrations,
)


class TestProvideOpenAIClient:
    @pytest.fixture(autouse=True)
    def setup_and_teardown(self, reset_environment):
        """Use shared environment reset fixture."""
        # Clear all environment variables at start
        env_keys = ["OPENAI_API_KEY", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_BASE_URL", "AZURE_OPENAI_API_VERSION"]
        for key in env_keys:
            if key in os.environ:
                del os.environ[key]

        # Reset environment registrations to ensure fresh state for each test
        set_default_registrations()
        yield
        # Reset environment registrations after test
        set_default_registrations()

    def set_env_and_reset(self, **env_vars):
        """Helper method to set environment variables and reset registrations."""
        # First clear all relevant environment variables
        env_keys = ["OPENAI_API_KEY", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_BASE_URL", "AZURE_OPENAI_API_VERSION"]
        for key in env_keys:
            if key in os.environ:
                del os.environ[key]

        # Then set the new environment variables
        for key, value in env_vars.items():
            os.environ[key] = value

        set_default_registrations()

    def test_provide_openai_client_with_openai_key(self):
        """Test creating OpenAI client when OPENAI_API_KEY is set."""
        self.set_env_and_reset(OPENAI_API_KEY="test-key")

        client = provide_openai_client()

        assert isinstance(client, OpenAI)

    def test_provide_openai_client_with_azure_keys(self):
        """Test creating Azure OpenAI client when Azure environment variables are set."""
        self.set_env_and_reset(
            AZURE_OPENAI_API_KEY="test-azure-key",
            AZURE_OPENAI_BASE_URL="https://test.services.ai.azure.com/openai/v1/",
            AZURE_OPENAI_API_VERSION="v1",
        )

        client = provide_openai_client()

        assert isinstance(client, AzureOpenAI)

    def test_provide_openai_client_prioritizes_openai_over_azure(self):
        """Test that OpenAI client is preferred when both sets of keys are available."""
        self.set_env_and_reset(
            OPENAI_API_KEY="test-key",
            AZURE_OPENAI_API_KEY="test-azure-key",
            AZURE_OPENAI_BASE_URL="https://test.services.ai.azure.com/openai/v1/",
            AZURE_OPENAI_API_VERSION="v1",
        )

        client = provide_openai_client()

        assert isinstance(client, OpenAI)

    def test_provide_openai_client_with_incomplete_azure_config(self):
        """Test creating Azure OpenAI client via Entra ID when API key is missing."""
        self.set_env_and_reset(
            AZURE_OPENAI_BASE_URL="https://test.services.ai.azure.com/openai/v1/", AZURE_OPENAI_API_VERSION="v1"
        )
        # Missing AZURE_OPENAI_API_KEY should fall back to Entra ID token provider.
        client = provide_openai_client()
        assert isinstance(client, AzureOpenAI)

    def test_provide_openai_client_with_azure_keys_default_version(self):
        """Test creating Azure OpenAI client with default API version when not specified."""
        self.set_env_and_reset(
            AZURE_OPENAI_API_KEY="test-azure-key", AZURE_OPENAI_BASE_URL="https://test.services.ai.azure.com/openai/v1/"
        )
        # AZURE_OPENAI_API_VERSION not set, should use default

        client = provide_openai_client()

        assert isinstance(client, AzureOpenAI)

    def test_provide_openai_client_with_no_environment_variables(self):
        """Test error when no environment variables are set."""
        with pytest.raises(ValueError) as context:
            provide_openai_client()

        error_message = str(context.value)
        # Check that the error message contains helpful information
        assert "No valid OpenAI or Azure OpenAI credentials found" in error_message
        assert "OPENAI_API_KEY" in error_message
        assert "AZURE_OPENAI_API_KEY" in error_message
        assert "AZURE_OPENAI_BASE_URL" in error_message
        assert "AZURE_OPENAI_API_VERSION" in error_message
        # Check that setup examples are provided
        assert "export OPENAI_API_KEY" in error_message
        assert "export AZURE_OPENAI_API_KEY" in error_message

    def test_provide_openai_client_with_empty_openai_key(self):
        """Test that empty OPENAI_API_KEY is treated as not set."""
        self.set_env_and_reset(
            OPENAI_API_KEY="",
            AZURE_OPENAI_API_KEY="test-azure-key",
            AZURE_OPENAI_BASE_URL="https://test.services.ai.azure.com/openai/v1/",
            AZURE_OPENAI_API_VERSION="v1",
        )

        client = provide_openai_client()

        assert isinstance(client, AzureOpenAI)

    def test_provide_openai_client_with_empty_azure_keys(self):
        """Test that empty Azure keys fall back to Entra ID token provider."""
        os.environ["AZURE_OPENAI_API_KEY"] = ""
        os.environ["AZURE_OPENAI_BASE_URL"] = "https://test.services.ai.azure.com/openai/v1/"
        os.environ["AZURE_OPENAI_API_VERSION"] = "v1"
        set_default_registrations()

        client = provide_openai_client()
        assert isinstance(client, AzureOpenAI)

    def test_provide_openai_client_reinstalls_defaults_after_container_clear(self):
        """Test lazy default reinstallation after the shared container is cleared."""
        self.set_env_and_reset(OPENAI_API_KEY="test-key")
        CONTAINER.clear()

        client = provide_openai_client()

        assert isinstance(client, OpenAI)


class TestProvideAsyncOpenAIClient:
    @pytest.fixture(autouse=True)
    def setup_and_teardown(self, reset_environment):
        """Use shared environment reset fixture."""
        # Clear all environment variables at start
        env_keys = ["OPENAI_API_KEY", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_BASE_URL", "AZURE_OPENAI_API_VERSION"]
        for key in env_keys:
            if key in os.environ:
                del os.environ[key]

        # Reset environment registrations to ensure fresh state for each test
        set_default_registrations()
        yield
        # Reset environment registrations after test
        set_default_registrations()

    def set_env_and_reset(self, **env_vars):
        """Helper method to set environment variables and reset registrations."""
        # First clear all relevant environment variables
        env_keys = ["OPENAI_API_KEY", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_BASE_URL", "AZURE_OPENAI_API_VERSION"]
        for key in env_keys:
            if key in os.environ:
                del os.environ[key]

        # Then set the new environment variables
        for key, value in env_vars.items():
            os.environ[key] = value

        set_default_registrations()

    def test_provide_async_openai_client_with_openai_key(self):
        """Test creating async OpenAI client when OPENAI_API_KEY is set."""
        self.set_env_and_reset(OPENAI_API_KEY="test-key")

        client = provide_async_openai_client()

        assert isinstance(client, AsyncOpenAI)

    def test_provide_async_openai_client_with_azure_keys(self):
        """Test creating async Azure OpenAI client when Azure environment variables are set."""
        self.set_env_and_reset(
            AZURE_OPENAI_API_KEY="test-azure-key",
            AZURE_OPENAI_BASE_URL="https://test.services.ai.azure.com/openai/v1/",
            AZURE_OPENAI_API_VERSION="v1",
        )

        client = provide_async_openai_client()

        assert isinstance(client, AsyncAzureOpenAI)

    def test_provide_async_openai_client_prioritizes_openai_over_azure(self):
        """Test that async OpenAI client is preferred when both sets of keys are available."""
        self.set_env_and_reset(
            OPENAI_API_KEY="test-key",
            AZURE_OPENAI_API_KEY="test-azure-key",
            AZURE_OPENAI_BASE_URL="https://test.services.ai.azure.com/openai/v1/",
            AZURE_OPENAI_API_VERSION="v1",
        )

        client = provide_async_openai_client()

        assert isinstance(client, AsyncOpenAI)

    def test_provide_async_openai_client_with_incomplete_azure_config(self):
        """Test error when Azure config is incomplete - missing endpoint."""
        self.set_env_and_reset(AZURE_OPENAI_API_KEY="test-azure-key", AZURE_OPENAI_API_VERSION="v1")
        # Missing AZURE_OPENAI_BASE_URL

        with pytest.raises(ValueError) as context:
            provide_async_openai_client()

        assert "No valid OpenAI or Azure OpenAI credentials found" in str(context.value)

    def test_provide_async_openai_client_with_azure_keys_default_version(self):
        """Test creating async Azure OpenAI client with default API version when not specified."""
        self.set_env_and_reset(
            AZURE_OPENAI_API_KEY="test-azure-key", AZURE_OPENAI_BASE_URL="https://test.services.ai.azure.com/openai/v1/"
        )
        # AZURE_OPENAI_API_VERSION not set, should use default

        client = provide_async_openai_client()

        assert isinstance(client, AsyncAzureOpenAI)

    def test_provide_async_openai_client_with_no_environment_variables(self):
        """Test error when no environment variables are set."""
        with pytest.raises(ValueError) as context:
            provide_async_openai_client()

        error_message = str(context.value)
        # Check that the error message contains helpful information
        assert "No valid OpenAI or Azure OpenAI credentials found" in error_message
        assert "OPENAI_API_KEY" in error_message
        assert "AZURE_OPENAI_API_KEY" in error_message
        assert "AZURE_OPENAI_BASE_URL" in error_message
        assert "AZURE_OPENAI_API_VERSION" in error_message
        # Check that setup examples are provided
        assert "export OPENAI_API_KEY" in error_message
        assert "export AZURE_OPENAI_API_KEY" in error_message

    def test_provide_async_openai_client_with_empty_openai_key(self):
        """Test that empty OPENAI_API_KEY is treated as not set."""
        self.set_env_and_reset(
            OPENAI_API_KEY="",
            AZURE_OPENAI_API_KEY="test-azure-key",
            AZURE_OPENAI_BASE_URL="https://test.services.ai.azure.com/openai/v1/",
            AZURE_OPENAI_API_VERSION="v1",
        )

        client = provide_async_openai_client()

        assert isinstance(client, AsyncAzureOpenAI)

    def test_provide_async_openai_client_with_empty_azure_keys(self):
        """Test that empty Azure keys fall back to Entra ID token provider."""
        self.set_env_and_reset(
            AZURE_OPENAI_API_KEY="",
            AZURE_OPENAI_BASE_URL="https://test.services.ai.azure.com/openai/v1/",
            AZURE_OPENAI_API_VERSION="v1",
        )

        client = provide_async_openai_client()
        assert isinstance(client, AsyncAzureOpenAI)

    def test_provide_async_openai_client_reinstalls_defaults_after_container_clear(self):
        """Test lazy async default reinstallation after the shared container is cleared."""
        self.set_env_and_reset(OPENAI_API_KEY="test-key")
        CONTAINER.clear()

        client = provide_async_openai_client()

        assert isinstance(client, AsyncOpenAI)


@pytest.mark.integration
class TestProviderIntegration:
    """Integration tests for both provider functions."""

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self, reset_environment):
        """Use shared environment reset fixture."""
        # Clear all environment variables at start
        env_keys = ["OPENAI_API_KEY", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_BASE_URL", "AZURE_OPENAI_API_VERSION"]
        for key in env_keys:
            if key in os.environ:
                del os.environ[key]

        # Reset environment registrations to ensure fresh state for each test
        set_default_registrations()
        yield
        # Reset environment registrations after test
        set_default_registrations()

    def set_env_and_reset(self, **env_vars):
        """Helper method to set environment variables and reset registrations."""
        # First clear all relevant environment variables
        env_keys = ["OPENAI_API_KEY", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_BASE_URL", "AZURE_OPENAI_API_VERSION"]
        for key in env_keys:
            if key in os.environ:
                del os.environ[key]

        # Then set the new environment variables
        for key, value in env_vars.items():
            os.environ[key] = value

        set_default_registrations()

    def test_both_functions_return_consistent_client_types(self):
        """Test that both functions return consistent client types for the same environment."""
        # Test with OpenAI environment
        self.set_env_and_reset(OPENAI_API_KEY="test-key")

        sync_client = provide_openai_client()
        async_client = provide_async_openai_client()

        assert isinstance(sync_client, OpenAI)
        assert isinstance(async_client, AsyncOpenAI)

        # Clear and test with Azure environment
        self.set_env_and_reset(
            AZURE_OPENAI_API_KEY="test-azure-key",
            AZURE_OPENAI_BASE_URL="https://test.services.ai.azure.com/openai/v1/",
            AZURE_OPENAI_API_VERSION="v1",
        )

        sync_client = provide_openai_client()
        async_client = provide_async_openai_client()

        assert isinstance(sync_client, AzureOpenAI)
        assert isinstance(async_client, AsyncAzureOpenAI)

    def test_azure_client_configuration(self):
        """Test that Azure clients are configured with correct parameters."""
        self.set_env_and_reset(
            AZURE_OPENAI_API_KEY="test-azure-key",
            AZURE_OPENAI_BASE_URL="https://test.services.ai.azure.com/openai/v1/",
            AZURE_OPENAI_API_VERSION="v1",
        )

        sync_client = provide_openai_client()
        async_client = provide_async_openai_client()

        # Check that Azure clients are created with correct configuration
        assert isinstance(sync_client, AzureOpenAI)
        assert isinstance(async_client, AsyncAzureOpenAI)


class TestAzureV1ApiWarning:
    """Test Azure v1 API URL warning functionality."""

    def test_check_azure_v1_api_url_no_warning_for_v1_url(self):
        """Test that v1 API URLs don't trigger warnings."""
        from openaivec._provider import _check_azure_v1_api_url

        v1_urls = [
            "https://myresource.services.ai.azure.com/openai/v1/",
            "https://myresource.services.ai.azure.com/openai/v1",
            "https://test.services.ai.azure.com/openai/v1/",
        ]

        for url in v1_urls:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                _check_azure_v1_api_url(url)
                assert len(w) == 0, f"Unexpected warning for URL: {url}"

    def test_check_azure_v1_api_url_warning_for_legacy_url(self):
        """Test that legacy API URLs trigger warnings."""
        from openaivec._provider import _check_azure_v1_api_url

        legacy_urls = [
            "https://myresource.services.ai.azure.com/",
            "https://myresource.openai.azure.com/",
            "https://test.services.ai.azure.com/openai/",
        ]

        for url in legacy_urls:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                _check_azure_v1_api_url(url)
                assert len(w) > 0, f"Expected warning for URL: {url}"
                assert "v1 API is recommended" in str(w[0].message)
                assert "learn.microsoft.com" in str(w[0].message)

    @pytest.mark.parametrize(
        "legacy_url,should_warn",
        [
            ("https://test.openai.azure.com/", True),
            ("https://test.services.ai.azure.com/", True),
            ("https://test.services.ai.azure.com/openai/v1/", False),
        ],
    )
    def test_azure_v1_warning_parametrized(self, legacy_url, should_warn):
        """Test Azure v1 API URL warning with different URL patterns."""
        from openaivec._provider import _check_azure_v1_api_url

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _check_azure_v1_api_url(legacy_url)

            if should_warn:
                assert len(w) > 0, f"Expected warning for URL: {legacy_url}"
                assert "v1 API is recommended" in str(w[0].message)
            else:
                assert len(w) == 0, f"Unexpected warning for URL: {legacy_url}"

    def test_pandas_ext_set_client_azure_warning(self):
        """Test that openaivec.set_client() shows warning for legacy Azure URLs."""
        from openai import AzureOpenAI

        import openaivec

        # Test with legacy URL (non-v1)
        legacy_client = AzureOpenAI(api_key="test-key", base_url="https://test.openai.azure.com/", api_version="v1")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            openaivec.set_client(legacy_client)
            assert len(w) > 0, "Expected warning for legacy Azure URL"
            assert "v1 API is recommended" in str(w[0].message)

        set_default_registrations()


class TestBuildMissingCredentialsError:
    """Test the _build_missing_credentials_error helper function."""

    def test_all_variables_missing(self):
        """Test error message when all variables are missing."""
        message = _build_missing_credentials_error(
            openai_api_key=None,
            azure_api_key=None,
            azure_base_url=None,
            azure_api_version=None,
        )

        assert "No valid OpenAI or Azure OpenAI credentials found" in message
        assert "✗ OPENAI_API_KEY is not set" in message
        assert "✗ AZURE_OPENAI_API_KEY (optional) is not set" in message
        assert "✗ AZURE_OPENAI_BASE_URL is not set" in message
        assert "✗ AZURE_OPENAI_API_VERSION is not set" in message
        assert 'export OPENAI_API_KEY="sk-..."' in message

    def test_only_openai_key_set(self):
        """Test error message when only OpenAI key is set (but this shouldn't trigger error)."""
        message = _build_missing_credentials_error(
            openai_api_key="sk-test",
            azure_api_key=None,
            azure_base_url=None,
            azure_api_version=None,
        )

        assert "✓ OPENAI_API_KEY is set" in message
        assert "✗ AZURE_OPENAI_API_KEY (optional) is not set" in message

    def test_partial_azure_config(self):
        """Test error message when Azure config is partially set."""
        message = _build_missing_credentials_error(
            openai_api_key=None,
            azure_api_key="test-key",
            azure_base_url=None,
            azure_api_version="v1",
        )

        assert "✗ OPENAI_API_KEY is not set" in message
        assert "✓ AZURE_OPENAI_API_KEY (optional) is set" in message
        assert "✗ AZURE_OPENAI_BASE_URL is not set" in message
        assert "✓ AZURE_OPENAI_API_VERSION is set" in message
        # Should include example for missing URL
        assert "export AZURE_OPENAI_BASE_URL=" in message

    def test_all_azure_variables_set(self):
        """Test error message when all Azure variables are set."""
        message = _build_missing_credentials_error(
            openai_api_key=None,
            azure_api_key="test-key",
            azure_base_url="https://test.openai.azure.com/openai/v1/",
            azure_api_version="v1",
        )

        assert "✗ OPENAI_API_KEY is not set" in message
        assert "✓ AZURE_OPENAI_API_KEY (optional) is set" in message
        assert "✓ AZURE_OPENAI_BASE_URL is set" in message
        assert "✓ AZURE_OPENAI_API_VERSION is set" in message

    def test_error_message_includes_examples(self):
        """Test that error message includes setup examples."""
        message = _build_missing_credentials_error(
            openai_api_key=None,
            azure_api_key=None,
            azure_base_url=None,
            azure_api_version=None,
        )

        assert "Option 1: Set OPENAI_API_KEY for OpenAI" in message
        assert "Option 2: Configure Azure OpenAI endpoint (API key or Entra ID)" in message
        assert "Example:" in message


class TestFabricEnvironment:
    """Tests for Microsoft Fabric environment detection and authentication."""

    _ALL_ENV_KEYS = [
        "OPENAI_API_KEY",
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_BASE_URL",
        "AZURE_OPENAI_API_VERSION",
        "AZURE_TENANT_ID",
        "AZURE_APP_CLIENT_ID",
        "KEY_VAULT_URL",
        "KEY_VAULT_SECRET_NAME",
    ]

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self, reset_environment):
        """Use shared environment reset fixture."""
        for key in self._ALL_ENV_KEYS:
            if key in os.environ:
                del os.environ[key]
        set_default_registrations()
        yield
        set_default_registrations()

    def set_env_and_reset(self, **env_vars):
        """Helper method to set environment variables and reset registrations."""
        for key in self._ALL_ENV_KEYS:
            if key in os.environ:
                del os.environ[key]
        for key, value in env_vars.items():
            os.environ[key] = value
        set_default_registrations()

    # -- Detection --

    def test_is_fabric_environment_returns_false_by_default(self):
        """Test that is_fabric_environment returns False when notebookutils is absent."""
        from openaivec._fabric import is_fabric_environment

        assert is_fabric_environment() is False

    def test_is_fabric_environment_returns_true_with_notebookutils(self):
        """Test that is_fabric_environment returns True when notebookutils is in builtins."""
        from openaivec._fabric import is_fabric_environment

        mock_nbu = MagicMock()
        builtins.notebookutils = mock_nbu
        try:
            assert is_fabric_environment() is True
        finally:
            del builtins.notebookutils

    def test_is_fabric_environment_false_without_credentials(self):
        """Test that detection fails when notebookutils lacks credentials.getSecret."""
        from openaivec._fabric import is_fabric_environment

        mock_nbu = MagicMock(spec=[])  # no attributes
        builtins.notebookutils = mock_nbu
        try:
            assert is_fabric_environment() is False
        finally:
            del builtins.notebookutils

    # -- Configuration check --

    def test_fabric_auth_configured_true_when_all_vars_set(self):
        """Test that is_auth_configured returns True when all vars present."""
        from openaivec._fabric import is_auth_configured

        os.environ["AZURE_TENANT_ID"] = "t"
        os.environ["AZURE_APP_CLIENT_ID"] = "c"
        os.environ["KEY_VAULT_URL"] = "https://kv.vault.azure.net/"
        os.environ["KEY_VAULT_SECRET_NAME"] = "s"

        assert is_auth_configured() is True

    def test_fabric_auth_configured_false_when_vars_missing(self):
        """Test that is_auth_configured returns False when a var is absent."""
        from openaivec._fabric import is_auth_configured

        os.environ["AZURE_TENANT_ID"] = "t"
        # others not set
        assert is_auth_configured() is False

    def test_partially_configured_true_when_some_vars_set(self):
        """Test that is_partially_configured returns True with partial vars."""
        from openaivec._fabric import is_partially_configured

        os.environ["AZURE_TENANT_ID"] = "t"
        os.environ["AZURE_APP_CLIENT_ID"] = "c"
        # KEY_VAULT_URL and KEY_VAULT_SECRET_NAME not set

        assert is_partially_configured() is True

    def test_partially_configured_false_when_all_set(self):
        """Test that is_partially_configured returns False when all vars set."""
        from openaivec._fabric import is_partially_configured

        os.environ["AZURE_TENANT_ID"] = "t"
        os.environ["AZURE_APP_CLIENT_ID"] = "c"
        os.environ["KEY_VAULT_URL"] = "https://kv.vault.azure.net/"
        os.environ["KEY_VAULT_SECRET_NAME"] = "s"

        assert is_partially_configured() is False

    def test_partially_configured_false_when_none_set(self):
        """Test that is_partially_configured returns False when no vars set."""
        from openaivec._fabric import is_partially_configured

        assert is_partially_configured() is False

    # -- Token provider --

    def test_build_fabric_token_provider_raises_when_vars_missing(self):
        """Test that build_token_provider raises ValueError with auth flow guidance."""
        from openaivec._fabric import build_token_provider

        with pytest.raises(ValueError, match="required environment variables are missing") as exc_info:
            build_token_provider()

        message = str(exc_info.value)
        assert "Authentication flow" in message
        assert "Setup steps" in message

    def test_build_fabric_token_provider_calls_key_vault(self):
        """Test that build_token_provider retrieves secret from Key Vault."""
        from openaivec._fabric import build_token_provider

        os.environ["AZURE_TENANT_ID"] = "test-tenant"
        os.environ["AZURE_APP_CLIENT_ID"] = "test-client"
        os.environ["KEY_VAULT_URL"] = "https://kv.vault.azure.net/"
        os.environ["KEY_VAULT_SECRET_NAME"] = "my-secret"

        mock_nbu = MagicMock()
        mock_nbu.credentials.getSecret.return_value = "fake-client-secret"
        builtins.notebookutils = mock_nbu

        try:
            with patch("openaivec._fabric.ClientSecretCredential") as mock_cred_cls:
                mock_cred = MagicMock()
                mock_cred.get_token.return_value = MagicMock(token="fake-bearer-token")
                mock_cred_cls.return_value = mock_cred

                provider = build_token_provider()
                token = provider()

                assert token == "fake-bearer-token"
                mock_nbu.credentials.getSecret.assert_called_once_with("https://kv.vault.azure.net/", "my-secret")
                mock_cred_cls.assert_called_once_with(
                    tenant_id="test-tenant",
                    client_id="test-client",
                    client_secret="fake-client-secret",
                )
                mock_cred.get_token.assert_called_once_with("https://cognitiveservices.azure.com/.default")
        finally:
            del builtins.notebookutils

    # -- Client creation (sync) --

    @patch("openaivec._fabric.is_fabric_environment", return_value=True)
    def test_provide_openai_client_uses_fabric_auth(self, _mock_fabric):
        """Test that provide_openai_client uses Fabric token provider when configured."""
        mock_nbu = MagicMock()
        mock_nbu.credentials.getSecret.return_value = "fake-secret"
        builtins.notebookutils = mock_nbu

        try:
            with patch("openaivec._fabric.ClientSecretCredential") as mock_cred_cls:
                mock_cred = MagicMock()
                mock_cred.get_token.return_value = MagicMock(token="tok")
                mock_cred_cls.return_value = mock_cred

                self.set_env_and_reset(
                    AZURE_TENANT_ID="t",
                    AZURE_APP_CLIENT_ID="c",
                    KEY_VAULT_URL="https://kv.vault.azure.net/",
                    KEY_VAULT_SECRET_NAME="s",
                    AZURE_OPENAI_BASE_URL="https://test.services.ai.azure.com/openai/v1/",
                    AZURE_OPENAI_API_VERSION="v1",
                )

                client = provide_openai_client()
                assert isinstance(client, AzureOpenAI)
        finally:
            del builtins.notebookutils

    # -- Client creation (async) --

    @patch("openaivec._fabric.is_fabric_environment", return_value=True)
    def test_provide_async_openai_client_uses_fabric_auth(self, _mock_fabric):
        """Test that provide_async_openai_client uses Fabric token provider when configured."""
        mock_nbu = MagicMock()
        mock_nbu.credentials.getSecret.return_value = "fake-secret"
        builtins.notebookutils = mock_nbu

        try:
            with patch("openaivec._fabric.ClientSecretCredential") as mock_cred_cls:
                mock_cred = MagicMock()
                mock_cred.get_token.return_value = MagicMock(token="tok")
                mock_cred_cls.return_value = mock_cred

                self.set_env_and_reset(
                    AZURE_TENANT_ID="t",
                    AZURE_APP_CLIENT_ID="c",
                    KEY_VAULT_URL="https://kv.vault.azure.net/",
                    KEY_VAULT_SECRET_NAME="s",
                    AZURE_OPENAI_BASE_URL="https://test.services.ai.azure.com/openai/v1/",
                    AZURE_OPENAI_API_VERSION="v1",
                )

                client = provide_async_openai_client()
                assert isinstance(client, AsyncAzureOpenAI)
        finally:
            del builtins.notebookutils

    # -- Precedence --

    @patch("openaivec._fabric.is_fabric_environment", return_value=True)
    def test_openai_key_wins_over_fabric(self, _mock_fabric):
        """Test that OPENAI_API_KEY takes priority even in Fabric environment."""
        mock_nbu = MagicMock()
        mock_nbu.credentials.getSecret.return_value = "fake-secret"
        builtins.notebookutils = mock_nbu

        try:
            self.set_env_and_reset(
                OPENAI_API_KEY="test-key",
                AZURE_TENANT_ID="t",
                AZURE_APP_CLIENT_ID="c",
                KEY_VAULT_URL="https://kv.vault.azure.net/",
                KEY_VAULT_SECRET_NAME="s",
                AZURE_OPENAI_BASE_URL="https://test.services.ai.azure.com/openai/v1/",
                AZURE_OPENAI_API_VERSION="v1",
            )

            client = provide_openai_client()
            assert isinstance(client, OpenAI)
        finally:
            del builtins.notebookutils

    @patch("openaivec._fabric.is_fabric_environment", return_value=True)
    def test_azure_api_key_wins_over_fabric_token(self, _mock_fabric):
        """Test that AZURE_OPENAI_API_KEY takes priority over Fabric token provider."""
        mock_nbu = MagicMock()
        mock_nbu.credentials.getSecret.return_value = "fake-secret"
        builtins.notebookutils = mock_nbu

        try:
            self.set_env_and_reset(
                AZURE_OPENAI_API_KEY="azure-key",
                AZURE_TENANT_ID="t",
                AZURE_APP_CLIENT_ID="c",
                KEY_VAULT_URL="https://kv.vault.azure.net/",
                KEY_VAULT_SECRET_NAME="s",
                AZURE_OPENAI_BASE_URL="https://test.services.ai.azure.com/openai/v1/",
                AZURE_OPENAI_API_VERSION="v1",
            )

            client = provide_openai_client()
            assert isinstance(client, AzureOpenAI)
        finally:
            del builtins.notebookutils

    # -- Fallback to DefaultAzureCredential --

    @patch("openaivec._fabric.is_fabric_environment", return_value=True)
    def test_fabric_falls_back_to_dac_when_vars_missing(self, _mock_fabric):
        """Test that Fabric without required vars falls back to DefaultAzureCredential."""
        mock_nbu = MagicMock()
        mock_nbu.credentials.getSecret.return_value = "fake-secret"
        builtins.notebookutils = mock_nbu

        try:
            self.set_env_and_reset(
                AZURE_OPENAI_BASE_URL="https://test.services.ai.azure.com/openai/v1/",
                AZURE_OPENAI_API_VERSION="v1",
                # Fabric vars NOT set → should fall back to DAC
            )

            client = provide_openai_client()
            assert isinstance(client, AzureOpenAI)
        finally:
            del builtins.notebookutils

    # -- Error message --

    @patch("openaivec._fabric.is_fabric_environment", return_value=True)
    def test_error_message_includes_fabric_auth_flow(self, _mock_fabric):
        """Test that error message includes auth flow and setup guidance when Fabric detected."""
        message = _build_missing_credentials_error(
            openai_api_key=None,
            azure_api_key=None,
            azure_base_url=None,
            azure_api_version=None,
        )

        assert "Fabric environment detected" in message
        assert "AZURE_TENANT_ID" in message
        assert "AZURE_APP_CLIENT_ID" in message
        assert "KEY_VAULT_URL" in message
        assert "KEY_VAULT_SECRET_NAME" in message
        assert "Authentication flow" in message
        assert "Key Vault Secrets User" in message
        assert "AI User" in message
        assert "Setup steps" in message
        assert "Service Principal" in message

    def test_error_message_excludes_fabric_vars_outside_fabric(self):
        """Test that error message omits Fabric section when not in Fabric."""
        message = _build_missing_credentials_error(
            openai_api_key=None,
            azure_api_key=None,
            azure_base_url=None,
            azure_api_version=None,
        )

        assert "Fabric" not in message
        assert "KEY_VAULT_URL" not in message

    # -- Warning on partial configuration --

    @patch("openaivec._fabric.is_fabric_environment", return_value=True)
    def test_partial_config_emits_warning_with_guidance(self, _mock_fabric):
        """Test that partial Fabric config emits a UserWarning with detailed setup guidance."""
        mock_nbu = MagicMock()
        mock_nbu.credentials.getSecret.return_value = "fake-secret"
        builtins.notebookutils = mock_nbu

        try:
            os.environ["AZURE_TENANT_ID"] = "t"
            # Only 1 of 4 vars set → partial config
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                set_default_registrations()

            fabric_warnings = [x for x in w if "Fabric" in str(x.message)]
            assert len(fabric_warnings) == 1
            msg = str(fabric_warnings[0].message)
            assert "not fully configured" in msg
            assert "Authentication flow" in msg
            assert "Key Vault Secrets User" in msg
            assert "AI User" in msg
            assert "Setup steps" in msg
            assert "Service Principal" in msg
            assert "✓ AZURE_TENANT_ID" in msg
            assert "✗ AZURE_APP_CLIENT_ID" in msg
            assert "AZURE_OPENAI_BASE_URL" in msg
            assert "set_default_registrations()" in msg
        finally:
            del builtins.notebookutils

    @patch("openaivec._fabric.is_fabric_environment", return_value=True)
    def test_no_warning_when_fully_configured(self, _mock_fabric):
        """Test that no warning is emitted when all Fabric vars are set."""
        mock_nbu = MagicMock()
        mock_nbu.credentials.getSecret.return_value = "fake-secret"
        builtins.notebookutils = mock_nbu

        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                with patch("openaivec._fabric.ClientSecretCredential"):
                    self.set_env_and_reset(
                        AZURE_TENANT_ID="t",
                        AZURE_APP_CLIENT_ID="c",
                        KEY_VAULT_URL="https://kv.vault.azure.net/",
                        KEY_VAULT_SECRET_NAME="s",
                        AZURE_OPENAI_BASE_URL="https://test.services.ai.azure.com/openai/v1/",
                        AZURE_OPENAI_API_VERSION="v1",
                    )

            fabric_warnings = [x for x in w if "Fabric" in str(x.message)]
            assert len(fabric_warnings) == 0
        finally:
            del builtins.notebookutils

    @patch("openaivec._fabric.is_fabric_environment", return_value=True)
    def test_no_warning_when_no_fabric_vars_set(self, _mock_fabric):
        """Test that no warning is emitted when zero Fabric vars are set (intentional DAC/API key)."""
        mock_nbu = MagicMock()
        builtins.notebookutils = mock_nbu

        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                self.set_env_and_reset(
                    AZURE_OPENAI_BASE_URL="https://test.services.ai.azure.com/openai/v1/",
                    AZURE_OPENAI_API_VERSION="v1",
                )

            fabric_warnings = [x for x in w if "Fabric" in str(x.message)]
            assert len(fabric_warnings) == 0
        finally:
            del builtins.notebookutils

    # -- Logging --

    @patch("openaivec._fabric.is_fabric_environment", return_value=True)
    def test_fully_configured_logs_info(self, _mock_fabric, caplog):
        """Test that fully configured Fabric logs INFO confirmation."""
        import logging

        mock_nbu = MagicMock()
        mock_nbu.credentials.getSecret.return_value = "fake-secret"
        builtins.notebookutils = mock_nbu

        try:
            with caplog.at_level(logging.INFO, logger="openaivec._fabric"):
                with patch("openaivec._fabric.ClientSecretCredential"):
                    self.set_env_and_reset(
                        AZURE_TENANT_ID="t",
                        AZURE_APP_CLIENT_ID="c",
                        KEY_VAULT_URL="https://kv.vault.azure.net/",
                        KEY_VAULT_SECRET_NAME="s",
                        AZURE_OPENAI_BASE_URL="https://test.services.ai.azure.com/openai/v1/",
                        AZURE_OPENAI_API_VERSION="v1",
                    )

            assert "fully configured" in caplog.text
            assert "✓ AZURE_TENANT_ID" in caplog.text
        finally:
            del builtins.notebookutils

    @patch("openaivec._fabric.is_fabric_environment", return_value=True)
    def test_partial_config_logs_var_status(self, _mock_fabric, caplog):
        """Test that partial config logs variable status at INFO level."""
        import logging

        mock_nbu = MagicMock()
        builtins.notebookutils = mock_nbu

        try:
            os.environ["AZURE_TENANT_ID"] = "t"
            with caplog.at_level(logging.INFO, logger="openaivec._fabric"):
                with warnings.catch_warnings():
                    warnings.simplefilter("always")
                    set_default_registrations()

            assert "Microsoft Fabric environment detected" in caplog.text
            assert "✓ AZURE_TENANT_ID" in caplog.text
            assert "✗ AZURE_APP_CLIENT_ID" in caplog.text
        finally:
            del builtins.notebookutils
