import numpy as np
import pytest

from openaivec._embeddings import AsyncBatchEmbeddings


@pytest.mark.requires_api
class TestAsyncBatchEmbeddings:
    @pytest.fixture(autouse=True)
    def setup_client(self, async_openai_client, embeddings_model_name, embedding_dim):
        self.openai_client = async_openai_client
        self.model_name = embeddings_model_name
        self.embedding_dim = embedding_dim
        yield

    @pytest.mark.asyncio
    async def test_create_basic(self):
        """Test basic embedding creation with a small batch size."""
        client = AsyncBatchEmbeddings.of(
            client=self.openai_client,
            model_name=self.model_name,
            batch_size=2,
        )
        inputs = ["apple", "banana", "orange", "pineapple"]

        response = await client.create(inputs)

        assert len(response) == len(inputs)
        for embedding in response:
            assert isinstance(embedding, np.ndarray)
            assert embedding.shape == (self.embedding_dim,)
            assert embedding.dtype == np.float32
            assert np.all(np.isfinite(embedding))

    @pytest.mark.asyncio
    async def test_create_empty_input(self):
        """Test embedding creation with an empty input list."""
        client = AsyncBatchEmbeddings.of(
            client=self.openai_client,
            model_name=self.model_name,
            batch_size=1,
        )
        inputs = []
        response = await client.create(inputs)

        assert len(response) == 0

    @pytest.mark.asyncio
    async def test_create_with_duplicates(self):
        """Test embedding creation with duplicate inputs. Should return correct embeddings in order."""
        client = AsyncBatchEmbeddings.of(
            client=self.openai_client,
            model_name=self.model_name,
            batch_size=2,
        )
        inputs = ["apple", "banana", "apple", "orange", "banana"]

        response = await client.create(inputs)

        assert len(response) == len(inputs)
        for embedding in response:
            assert isinstance(embedding, np.ndarray)
            assert embedding.shape == (self.embedding_dim,)
            assert embedding.dtype == np.float32

        unique_inputs_first_occurrence_indices = {text: inputs.index(text) for text in set(inputs)}
        expected_map = {text: response[index] for text, index in unique_inputs_first_occurrence_indices.items()}

        for i, text in enumerate(inputs):
            assert np.allclose(response[i], expected_map[text])

    @pytest.mark.asyncio
    async def test_create_batch_size_larger_than_unique(self):
        """Test when batch_size is larger than the number of unique inputs."""
        client = AsyncBatchEmbeddings.of(
            client=self.openai_client,
            model_name=self.model_name,
            batch_size=5,
        )
        inputs = ["apple", "banana", "orange", "apple"]

        response = await client.create(inputs)

        assert len(response) == len(inputs)
        unique_inputs_first_occurrence_indices = {text: inputs.index(text) for text in set(inputs)}
        expected_map = {text: response[index] for text, index in unique_inputs_first_occurrence_indices.items()}
        for i, text in enumerate(inputs):
            assert np.allclose(response[i], expected_map[text])
            assert response[i].shape == (self.embedding_dim,)
            assert response[i].dtype == np.float32

    @pytest.mark.asyncio
    async def test_create_batch_size_one(self):
        """Test embedding creation with batch_size = 1."""
        client = AsyncBatchEmbeddings.of(
            client=self.openai_client,
            model_name=self.model_name,
            batch_size=1,
        )
        inputs = ["apple", "banana", "orange"]

        response = await client.create(inputs)

        assert len(response) == len(inputs)
        for embedding in response:
            assert isinstance(embedding, np.ndarray)
            assert embedding.shape == (self.embedding_dim,)
            assert embedding.dtype == np.float32

    def test_initialization_default_concurrency(self):
        """Test initialization uses default max_concurrency."""
        client = AsyncBatchEmbeddings.of(
            client=self.openai_client,
            model_name=self.model_name,
        )
        assert client.cache.max_concurrency == 8

    def test_initialization_custom_concurrency(self):
        """Test initialization with custom max_concurrency."""
        custom_concurrency = 4
        client = AsyncBatchEmbeddings.of(
            client=self.openai_client, model_name=self.model_name, max_concurrency=custom_concurrency
        )
        assert client.cache.max_concurrency == custom_concurrency

    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    @pytest.mark.asyncio
    async def test_create_with_different_batch_sizes(self, batch_size):
        """Test embedding creation with various batch sizes."""
        client = AsyncBatchEmbeddings.of(
            client=self.openai_client,
            model_name=self.model_name,
            batch_size=batch_size,
        )
        inputs = ["test1", "test2", "test3", "test4"]

        response = await client.create(inputs)

        assert len(response) == len(inputs)
        for embedding in response:
            assert isinstance(embedding, np.ndarray)
            assert embedding.shape == (self.embedding_dim,)
            assert embedding.dtype == np.float32

    @pytest.mark.parametrize("concurrency", [2, 4, 8])
    @pytest.mark.asyncio
    async def test_create_with_different_concurrency(self, concurrency):
        """Test embedding creation with various concurrency settings."""
        client = AsyncBatchEmbeddings.of(
            client=self.openai_client,
            model_name=self.model_name,
            max_concurrency=concurrency,
        )
        inputs = ["concurrent1", "concurrent2", "concurrent3"]

        response = await client.create(inputs)

        assert len(response) == len(inputs)
        assert client.cache.max_concurrency == concurrency
