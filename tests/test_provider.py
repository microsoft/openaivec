import os
import warnings

import pytest
from openai import AsyncAzureOpenAI, AsyncOpenAI, AzureOpenAI, OpenAI

from openaivec._provider import provide_async_openai_client, provide_openai_client, set_default_registrations


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
            AZURE_OPENAI_API_VERSION="preview",
        )

        client = provide_openai_client()

        assert isinstance(client, AzureOpenAI)

    def test_provide_openai_client_prioritizes_openai_over_azure(self):
        """Test that OpenAI client is preferred when both sets of keys are available."""
        self.set_env_and_reset(
            OPENAI_API_KEY="test-key",
            AZURE_OPENAI_API_KEY="test-azure-key",
            AZURE_OPENAI_BASE_URL="https://test.services.ai.azure.com/openai/v1/",
            AZURE_OPENAI_API_VERSION="preview",
        )

        client = provide_openai_client()

        assert isinstance(client, OpenAI)

    def test_provide_openai_client_with_incomplete_azure_config(self):
        """Test error when Azure config is incomplete - missing API key."""
        self.set_env_and_reset(
            AZURE_OPENAI_BASE_URL="https://test.services.ai.azure.com/openai/v1/", AZURE_OPENAI_API_VERSION="preview"
        )
        # Missing AZURE_OPENAI_API_KEY

        with pytest.raises(ValueError) as context:
            provide_openai_client()

        assert "No valid OpenAI or Azure OpenAI environment variables found" in str(context.value)

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

        expected_message = (
            "No valid OpenAI or Azure OpenAI environment variables found. "
            "Please set either OPENAI_API_KEY or AZURE_OPENAI_API_KEY, "
            "AZURE_OPENAI_BASE_URL, and AZURE_OPENAI_API_VERSION."
        )
        assert str(context.value) == expected_message

    def test_provide_openai_client_with_empty_openai_key(self):
        """Test that empty OPENAI_API_KEY is treated as not set."""
        self.set_env_and_reset(
            OPENAI_API_KEY="",
            AZURE_OPENAI_API_KEY="test-azure-key",
            AZURE_OPENAI_BASE_URL="https://test.services.ai.azure.com/openai/v1/",
            AZURE_OPENAI_API_VERSION="preview",
        )

        client = provide_openai_client()

        assert isinstance(client, AzureOpenAI)

    def test_provide_openai_client_with_empty_azure_keys(self):
        """Test that empty Azure keys are treated as not set."""
        os.environ["AZURE_OPENAI_API_KEY"] = ""
        os.environ["AZURE_OPENAI_BASE_URL"] = "https://test.services.ai.azure.com/openai/v1/"
        os.environ["AZURE_OPENAI_API_VERSION"] = "preview"

        with pytest.raises(ValueError):
            provide_openai_client()


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
            AZURE_OPENAI_API_VERSION="preview",
        )

        client = provide_async_openai_client()

        assert isinstance(client, AsyncAzureOpenAI)

    def test_provide_async_openai_client_prioritizes_openai_over_azure(self):
        """Test that async OpenAI client is preferred when both sets of keys are available."""
        self.set_env_and_reset(
            OPENAI_API_KEY="test-key",
            AZURE_OPENAI_API_KEY="test-azure-key",
            AZURE_OPENAI_BASE_URL="https://test.services.ai.azure.com/openai/v1/",
            AZURE_OPENAI_API_VERSION="preview",
        )

        client = provide_async_openai_client()

        assert isinstance(client, AsyncOpenAI)

    def test_provide_async_openai_client_with_incomplete_azure_config(self):
        """Test error when Azure config is incomplete - missing endpoint."""
        self.set_env_and_reset(AZURE_OPENAI_API_KEY="test-azure-key", AZURE_OPENAI_API_VERSION="preview")
        # Missing AZURE_OPENAI_BASE_URL

        with pytest.raises(ValueError) as context:
            provide_async_openai_client()

        assert "No valid OpenAI or Azure OpenAI environment variables found" in str(context.value)

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

        expected_message = (
            "No valid OpenAI or Azure OpenAI environment variables found. "
            "Please set either OPENAI_API_KEY or AZURE_OPENAI_API_KEY, "
            "AZURE_OPENAI_BASE_URL, and AZURE_OPENAI_API_VERSION."
        )
        assert str(context.value) == expected_message

    def test_provide_async_openai_client_with_empty_openai_key(self):
        """Test that empty OPENAI_API_KEY is treated as not set."""
        self.set_env_and_reset(
            OPENAI_API_KEY="",
            AZURE_OPENAI_API_KEY="test-azure-key",
            AZURE_OPENAI_BASE_URL="https://test.services.ai.azure.com/openai/v1/",
            AZURE_OPENAI_API_VERSION="preview",
        )

        client = provide_async_openai_client()

        assert isinstance(client, AsyncAzureOpenAI)

    def test_provide_async_openai_client_with_empty_azure_keys(self):
        """Test that empty Azure keys are treated as not set."""
        self.set_env_and_reset(
            AZURE_OPENAI_API_KEY="",
            AZURE_OPENAI_BASE_URL="https://test.services.ai.azure.com/openai/v1/",
            AZURE_OPENAI_API_VERSION="preview",
        )

        with pytest.raises(ValueError):
            provide_async_openai_client()


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
            AZURE_OPENAI_API_VERSION="preview",
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
            AZURE_OPENAI_API_VERSION="preview",
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

    def test_pandas_ext_use_azure_warning(self):
        """Test that pandas_ext.use() shows warning for legacy Azure URLs."""
        from openai import AzureOpenAI

        from openaivec import pandas_ext

        # Test with legacy URL (non-v1)
        legacy_client = AzureOpenAI(
            api_key="test-key", base_url="https://test.openai.azure.com/", api_version="preview"
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            pandas_ext.use(legacy_client)
            assert len(w) > 0, "Expected warning for legacy Azure URL"
            assert "v1 API is recommended" in str(w[0].message)

        set_default_registrations()
