import asyncio
import os
import unittest

import numpy as np
from openai import AsyncOpenAI

from openaivec._embeddings import AsyncBatchEmbeddings


@unittest.skipIf(not os.getenv("OPENAI_API_KEY"), "OPENAI_API_KEY not set in environment")
class TestAsyncBatchEmbeddings(unittest.TestCase):
    def setUp(self):
        self.openai_client = AsyncOpenAI()
        self.model_name = "text-embedding-3-small"
        self.embedding_dim = 1536

    def test_create_basic(self):
        """Test basic embedding creation with a small batch size."""
        client = AsyncBatchEmbeddings.of(
            client=self.openai_client,
            model_name=self.model_name,
            batch_size=2,
        )
        inputs = ["apple", "banana", "orange", "pineapple"]

        response = asyncio.run(client.create(inputs))

        self.assertEqual(len(response), len(inputs))
        for embedding in response:
            self.assertIsInstance(embedding, np.ndarray)
            self.assertEqual(embedding.shape, (self.embedding_dim,))
            self.assertEqual(embedding.dtype, np.float32)
            self.assertTrue(np.all(np.isfinite(embedding)))

    def test_create_empty_input(self):
        """Test embedding creation with an empty input list."""
        client = AsyncBatchEmbeddings.of(
            client=self.openai_client,
            model_name=self.model_name,
            batch_size=1,
        )
        inputs = []
        response = asyncio.run(client.create(inputs))

        self.assertEqual(len(response), 0)

    def test_create_with_duplicates(self):
        """Test embedding creation with duplicate inputs. Should return correct embeddings in order."""
        client = AsyncBatchEmbeddings.of(
            client=self.openai_client,
            model_name=self.model_name,
            batch_size=2,
        )
        inputs = ["apple", "banana", "apple", "orange", "banana"]

        response = asyncio.run(client.create(inputs))

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
        client = AsyncBatchEmbeddings.of(
            client=self.openai_client,
            model_name=self.model_name,
            batch_size=5,
        )
        inputs = ["apple", "banana", "orange", "apple"]

        response = asyncio.run(client.create(inputs))

        self.assertEqual(len(response), len(inputs))
        unique_inputs_first_occurrence_indices = {text: inputs.index(text) for text in set(inputs)}
        expected_map = {text: response[index] for text, index in unique_inputs_first_occurrence_indices.items()}
        for i, text in enumerate(inputs):
            self.assertTrue(np.allclose(response[i], expected_map[text]))
            self.assertEqual(response[i].shape, (self.embedding_dim,))
            self.assertEqual(response[i].dtype, np.float32)

    def test_create_batch_size_one(self):
        """Test embedding creation with batch_size = 1."""
        client = AsyncBatchEmbeddings.of(
            client=self.openai_client,
            model_name=self.model_name,
            batch_size=1,
        )
        inputs = ["apple", "banana", "orange"]

        response = asyncio.run(client.create(inputs))

        self.assertEqual(len(response), len(inputs))
        for embedding in response:
            self.assertIsInstance(embedding, np.ndarray)
            self.assertEqual(embedding.shape, (self.embedding_dim,))
            self.assertEqual(embedding.dtype, np.float32)

    def test_initialization_default_concurrency(self):
        """Test initialization uses default max_concurrency."""
        client = AsyncBatchEmbeddings.of(
            client=self.openai_client,
            model_name=self.model_name,
        )
        self.assertEqual(client.cache.max_concurrency, 8)

    def test_initialization_custom_concurrency(self):
        """Test initialization with custom max_concurrency."""
        custom_concurrency = 4
        client = AsyncBatchEmbeddings.of(
            client=self.openai_client, model_name=self.model_name, max_concurrency=custom_concurrency
        )
        self.assertEqual(client.cache.max_concurrency, custom_concurrency)
