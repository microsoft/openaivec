import unittest
import asyncio
import numpy as np
import os

from openai import AsyncOpenAI

from openaivec.embeddings import AsyncBatchEmbeddings


@unittest.skipIf(not os.getenv("OPENAI_API_KEY"), "OPENAI_API_KEY not set in environment")
class TestAsyncBatchEmbeddings(unittest.TestCase):
    def setUp(self):
        self.openai_client = AsyncOpenAI()
        self.model_name = "text-embedding-3-small"
        self.embedding_dim = 1536

    def test_create_basic(self):
        """Test basic embedding creation with a small batch size."""
        client = AsyncBatchEmbeddings(
            client=self.openai_client,
            model_name=self.model_name,
        )
        inputs = ["apple", "banana", "orange", "pineapple"]
        batch_size = 2

        response = asyncio.run(client.create(inputs, batch_size=batch_size))

        self.assertEqual(len(response), len(inputs))
        for embedding in response:
            self.assertIsInstance(embedding, np.ndarray)
            self.assertEqual(embedding.shape, (self.embedding_dim,))
            self.assertEqual(embedding.dtype, np.float32)
            self.assertTrue(np.all(np.isfinite(embedding)))

    def test_create_empty_input(self):
        """Test embedding creation with an empty input list."""
        client = AsyncBatchEmbeddings(
            client=self.openai_client,
            model_name=self.model_name,
        )
        inputs = []
        response = asyncio.run(client.create(inputs, batch_size=1))

        self.assertEqual(len(response), 0)

    def test_create_with_duplicates(self):
        """Test embedding creation with duplicate inputs. Should return correct embeddings in order."""
        client = AsyncBatchEmbeddings(
            client=self.openai_client,
            model_name=self.model_name,
        )
        inputs = ["apple", "banana", "apple", "orange", "banana"]
        batch_size = 2

        response = asyncio.run(client.create(inputs, batch_size=batch_size))

        self.assertEqual(len(response), len(inputs))
        for embedding in response:
            self.assertIsInstance(embedding, np.ndarray)
            self.assertEqual(embedding.shape, (self.embedding_dim,))
            self.assertEqual(embedding.dtype, np.float32)

        unique_inputs_first_occurrence_indices = {text: inputs.index(text) for text in set(inputs)}
        expected_map = {text: response[index] for text, index in unique_inputs_first_occurrence_indices.items()}

        for i, text in enumerate(inputs):
            self.assertTrue(np.allclose(response[i], expected_map[text]))

    def test_create_batch_size_larger_than_unique(self):
        """Test when batch_size is larger than the number of unique inputs."""
        client = AsyncBatchEmbeddings(
            client=self.openai_client,
            model_name=self.model_name,
        )
        inputs = ["apple", "banana", "orange", "apple"]
        batch_size = 5

        response = asyncio.run(client.create(inputs, batch_size=batch_size))

        self.assertEqual(len(response), len(inputs))
        unique_inputs_first_occurrence_indices = {text: inputs.index(text) for text in set(inputs)}
        expected_map = {text: response[index] for text, index in unique_inputs_first_occurrence_indices.items()}
        for i, text in enumerate(inputs):
            self.assertTrue(np.allclose(response[i], expected_map[text]))
            self.assertEqual(response[i].shape, (self.embedding_dim,))
            self.assertEqual(response[i].dtype, np.float32)

    def test_create_batch_size_one(self):
        """Test embedding creation with batch_size = 1."""
        client = AsyncBatchEmbeddings(
            client=self.openai_client,
            model_name=self.model_name,
        )
        inputs = ["apple", "banana", "orange"]
        batch_size = 1

        response = asyncio.run(client.create(inputs, batch_size=batch_size))

        self.assertEqual(len(response), len(inputs))
        for embedding in response:
            self.assertIsInstance(embedding, np.ndarray)
            self.assertEqual(embedding.shape, (self.embedding_dim,))
            self.assertEqual(embedding.dtype, np.float32)

    def test_initialization_default_concurrency(self):
        """Test initialization uses default max_concurrency."""
        client = AsyncBatchEmbeddings(
            client=self.openai_client,
            model_name=self.model_name,
        )
        self.assertEqual(client.max_concurrency, 8)

    def test_initialization_custom_concurrency(self):
        """Test initialization with custom max_concurrency."""
        custom_concurrency = 4
        client = AsyncBatchEmbeddings(
            client=self.openai_client, model_name=self.model_name, max_concurrency=custom_concurrency
        )
        self.assertEqual(client.max_concurrency, custom_concurrency)
