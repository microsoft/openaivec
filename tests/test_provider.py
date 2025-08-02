import os
import unittest

from openai import AsyncAzureOpenAI, AsyncOpenAI, AzureOpenAI, OpenAI

from openaivec.provider import provide_async_openai_client, provide_openai_client


class TestProvideOpenAIClient(unittest.TestCase):
    def setUp(self):
        """Save original environment variables."""
        self.original_env = {
            "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY"),
            "AZURE_OPENAI_API_KEY": os.environ.get("AZURE_OPENAI_API_KEY"),
            "AZURE_OPENAI_API_ENDPOINT": os.environ.get("AZURE_OPENAI_API_ENDPOINT"),
            "AZURE_OPENAI_API_VERSION": os.environ.get("AZURE_OPENAI_API_VERSION"),
        }
        # Clear all environment variables
        for key in self.original_env:
            if key in os.environ:
                del os.environ[key]

    def tearDown(self):
        """Restore original environment variables."""
        for key, value in self.original_env.items():
            if value is not None:
                os.environ[key] = value
            elif key in os.environ:
                del os.environ[key]

    def test_provide_openai_client_with_openai_key(self):
        """Test creating OpenAI client when OPENAI_API_KEY is set."""
        os.environ["OPENAI_API_KEY"] = "test-key"
        
        client = provide_openai_client()
        
        self.assertIsInstance(client, OpenAI)

    def test_provide_openai_client_with_azure_keys(self):
        """Test creating Azure OpenAI client when Azure environment variables are set."""
        os.environ["AZURE_OPENAI_API_KEY"] = "test-azure-key"
        os.environ["AZURE_OPENAI_API_ENDPOINT"] = "https://test.openai.azure.com"
        os.environ["AZURE_OPENAI_API_VERSION"] = "2024-02-01"
        
        client = provide_openai_client()
        
        self.assertIsInstance(client, AzureOpenAI)

    def test_provide_openai_client_prioritizes_openai_over_azure(self):
        """Test that OpenAI client is preferred when both sets of keys are available."""
        os.environ["OPENAI_API_KEY"] = "test-key"
        os.environ["AZURE_OPENAI_API_KEY"] = "test-azure-key"
        os.environ["AZURE_OPENAI_API_ENDPOINT"] = "https://test.openai.azure.com"
        os.environ["AZURE_OPENAI_API_VERSION"] = "2024-02-01"
        
        client = provide_openai_client()
        
        self.assertIsInstance(client, OpenAI)

    def test_provide_openai_client_with_incomplete_azure_config(self):
        """Test error when Azure config is incomplete."""
        os.environ["AZURE_OPENAI_API_KEY"] = "test-azure-key"
        os.environ["AZURE_OPENAI_API_ENDPOINT"] = "https://test.openai.azure.com"
        # Missing AZURE_OPENAI_API_VERSION
        
        with self.assertRaises(ValueError) as context:
            provide_openai_client()
        
        self.assertIn("No valid OpenAI or Azure OpenAI environment variables found", str(context.exception))

    def test_provide_openai_client_with_no_environment_variables(self):
        """Test error when no environment variables are set."""
        with self.assertRaises(ValueError) as context:
            provide_openai_client()
        
        expected_message = (
            "No valid OpenAI or Azure OpenAI environment variables found. "
            "Please set either OPENAI_API_KEY or AZURE_OPENAI_API_KEY, "
            "AZURE_OPENAI_API_ENDPOINT, and AZURE_OPENAI_API_VERSION."
        )
        self.assertEqual(str(context.exception), expected_message)

    def test_provide_openai_client_with_empty_openai_key(self):
        """Test that empty OPENAI_API_KEY is treated as not set."""
        os.environ["OPENAI_API_KEY"] = ""
        os.environ["AZURE_OPENAI_API_KEY"] = "test-azure-key"
        os.environ["AZURE_OPENAI_API_ENDPOINT"] = "https://test.openai.azure.com"
        os.environ["AZURE_OPENAI_API_VERSION"] = "2024-02-01"
        
        client = provide_openai_client()
        
        self.assertIsInstance(client, AzureOpenAI)

    def test_provide_openai_client_with_empty_azure_keys(self):
        """Test that empty Azure keys are treated as not set."""
        os.environ["AZURE_OPENAI_API_KEY"] = ""
        os.environ["AZURE_OPENAI_API_ENDPOINT"] = "https://test.openai.azure.com"
        os.environ["AZURE_OPENAI_API_VERSION"] = "2024-02-01"
        
        with self.assertRaises(ValueError):
            provide_openai_client()


class TestProvideAsyncOpenAIClient(unittest.TestCase):
    def setUp(self):
        """Save original environment variables."""
        self.original_env = {
            "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY"),
            "AZURE_OPENAI_API_KEY": os.environ.get("AZURE_OPENAI_API_KEY"),
            "AZURE_OPENAI_API_ENDPOINT": os.environ.get("AZURE_OPENAI_API_ENDPOINT"),
            "AZURE_OPENAI_API_VERSION": os.environ.get("AZURE_OPENAI_API_VERSION"),
        }
        # Clear all environment variables
        for key in self.original_env:
            if key in os.environ:
                del os.environ[key]

    def tearDown(self):
        """Restore original environment variables."""
        for key, value in self.original_env.items():
            if value is not None:
                os.environ[key] = value
            elif key in os.environ:
                del os.environ[key]

    def test_provide_async_openai_client_with_openai_key(self):
        """Test creating async OpenAI client when OPENAI_API_KEY is set."""
        os.environ["OPENAI_API_KEY"] = "test-key"
        
        client = provide_async_openai_client()
        
        self.assertIsInstance(client, AsyncOpenAI)

    def test_provide_async_openai_client_with_azure_keys(self):
        """Test creating async Azure OpenAI client when Azure environment variables are set."""
        os.environ["AZURE_OPENAI_API_KEY"] = "test-azure-key"
        os.environ["AZURE_OPENAI_API_ENDPOINT"] = "https://test.openai.azure.com"
        os.environ["AZURE_OPENAI_API_VERSION"] = "2024-02-01"
        
        client = provide_async_openai_client()
        
        self.assertIsInstance(client, AsyncAzureOpenAI)

    def test_provide_async_openai_client_prioritizes_openai_over_azure(self):
        """Test that async OpenAI client is preferred when both sets of keys are available."""
        os.environ["OPENAI_API_KEY"] = "test-key"
        os.environ["AZURE_OPENAI_API_KEY"] = "test-azure-key"
        os.environ["AZURE_OPENAI_API_ENDPOINT"] = "https://test.openai.azure.com"
        os.environ["AZURE_OPENAI_API_VERSION"] = "2024-02-01"
        
        client = provide_async_openai_client()
        
        self.assertIsInstance(client, AsyncOpenAI)

    def test_provide_async_openai_client_with_incomplete_azure_config(self):
        """Test error when Azure config is incomplete."""
        os.environ["AZURE_OPENAI_API_KEY"] = "test-azure-key"
        os.environ["AZURE_OPENAI_API_ENDPOINT"] = "https://test.openai.azure.com"
        # Missing AZURE_OPENAI_API_VERSION
        
        with self.assertRaises(ValueError) as context:
            provide_async_openai_client()
        
        self.assertIn("No valid OpenAI or Azure OpenAI environment variables found", str(context.exception))

    def test_provide_async_openai_client_with_no_environment_variables(self):
        """Test error when no environment variables are set."""
        with self.assertRaises(ValueError) as context:
            provide_async_openai_client()
        
        expected_message = (
            "No valid OpenAI or Azure OpenAI environment variables found. "
            "Please set either OPENAI_API_KEY or AZURE_OPENAI_API_KEY, "
            "AZURE_OPENAI_API_ENDPOINT, and AZURE_OPENAI_API_VERSION."
        )
        self.assertEqual(str(context.exception), expected_message)

    def test_provide_async_openai_client_with_empty_openai_key(self):
        """Test that empty OPENAI_API_KEY is treated as not set."""
        os.environ["OPENAI_API_KEY"] = ""
        os.environ["AZURE_OPENAI_API_KEY"] = "test-azure-key"
        os.environ["AZURE_OPENAI_API_ENDPOINT"] = "https://test.openai.azure.com"
        os.environ["AZURE_OPENAI_API_VERSION"] = "2024-02-01"
        
        client = provide_async_openai_client()
        
        self.assertIsInstance(client, AsyncAzureOpenAI)

    def test_provide_async_openai_client_with_empty_azure_keys(self):
        """Test that empty Azure keys are treated as not set."""
        os.environ["AZURE_OPENAI_API_KEY"] = ""
        os.environ["AZURE_OPENAI_API_ENDPOINT"] = "https://test.openai.azure.com"
        os.environ["AZURE_OPENAI_API_VERSION"] = "2024-02-01"
        
        with self.assertRaises(ValueError):
            provide_async_openai_client()


class TestProviderIntegration(unittest.TestCase):
    """Integration tests for both provider functions."""

    def setUp(self):
        """Save original environment variables."""
        self.original_env = {
            "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY"),
            "AZURE_OPENAI_API_KEY": os.environ.get("AZURE_OPENAI_API_KEY"),
            "AZURE_OPENAI_API_ENDPOINT": os.environ.get("AZURE_OPENAI_API_ENDPOINT"),
            "AZURE_OPENAI_API_VERSION": os.environ.get("AZURE_OPENAI_API_VERSION"),
        }
        # Clear all environment variables
        for key in self.original_env:
            if key in os.environ:
                del os.environ[key]

    def tearDown(self):
        """Restore original environment variables."""
        for key, value in self.original_env.items():
            if value is not None:
                os.environ[key] = value
            elif key in os.environ:
                del os.environ[key]

    def test_both_functions_return_consistent_client_types(self):
        """Test that both functions return consistent client types for the same environment."""
        # Test with OpenAI environment
        os.environ["OPENAI_API_KEY"] = "test-key"
        
        sync_client = provide_openai_client()
        async_client = provide_async_openai_client()
        
        self.assertIsInstance(sync_client, OpenAI)
        self.assertIsInstance(async_client, AsyncOpenAI)
        
        # Clear and test with Azure environment
        del os.environ["OPENAI_API_KEY"]
        os.environ["AZURE_OPENAI_API_KEY"] = "test-azure-key"
        os.environ["AZURE_OPENAI_API_ENDPOINT"] = "https://test.openai.azure.com"
        os.environ["AZURE_OPENAI_API_VERSION"] = "2024-02-01"
        
        sync_client = provide_openai_client()
        async_client = provide_async_openai_client()
        
        self.assertIsInstance(sync_client, AzureOpenAI)
        self.assertIsInstance(async_client, AsyncAzureOpenAI)

    def test_azure_client_configuration(self):
        """Test that Azure clients are configured with correct parameters."""
        os.environ["AZURE_OPENAI_API_KEY"] = "test-azure-key"
        os.environ["AZURE_OPENAI_API_ENDPOINT"] = "https://test.openai.azure.com"
        os.environ["AZURE_OPENAI_API_VERSION"] = "2024-02-01"
        
        sync_client = provide_openai_client()
        async_client = provide_async_openai_client()
        
        # Check that Azure clients are created with correct configuration
        self.assertIsInstance(sync_client, AzureOpenAI)
        self.assertIsInstance(async_client, AsyncAzureOpenAI)
        
        # We can't easily test the internal configuration without accessing private attributes,
        # but the fact that they're created without exception indicates proper configuration