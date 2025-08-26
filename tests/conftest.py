"""
Shared pytest fixtures and configuration for openaivec tests.

This module contains:
- Common test data fixtures
- Client setup fixtures
- Custom pytest marks
- Shared test utilities
"""

import os
import warnings
from typing import Any, Generator

import numpy as np
import pandas as pd
import pytest
from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel, Field

# Suppress OpenAI API warnings during tests
warnings.filterwarnings("ignore", category=UserWarning, module="openai")


def pytest_configure(config):
    """Configure custom pytest marks."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line(
        "markers", "requires_api: marks tests as requiring OPENAI_API_KEY (tests will fail if not set)"
    )
    config.addinivalue_line("markers", "asyncio: marks tests as async (handled by pytest-asyncio)")
    config.addinivalue_line("markers", "spark: marks tests as requiring Spark session")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")


def pytest_collection_modifyitems(config, items):
    """Automatically mark async tests."""
    for item in items:
        # Automatically mark async tests (check if the function is actually async)
        import asyncio

        if hasattr(item.function, "__code__") and asyncio.iscoroutinefunction(item.function):
            item.add_marker(pytest.mark.asyncio)


# ===== CLIENT FIXTURES =====


@pytest.fixture(scope="session")
def openai_client() -> Generator[OpenAI, None, None]:
    """Provide OpenAI client for tests. Will fail if OPENAI_API_KEY not set."""
    # Let OpenAI client construction fail naturally if no API key
    client = OpenAI()
    yield client


@pytest.fixture(scope="session")
def async_openai_client() -> Generator[AsyncOpenAI, None, None]:
    """Provide async OpenAI client for tests. Will fail if OPENAI_API_KEY not set."""
    # Let AsyncOpenAI client construction fail naturally if no API key
    client = AsyncOpenAI()
    yield client


# ===== MODEL FIXTURES =====


@pytest.fixture(scope="session")
def responses_model_name() -> str:
    """Default model name for response generation."""
    return "gpt-4.1-mini"


@pytest.fixture(scope="session")
def embeddings_model_name() -> str:
    """Default model name for embeddings."""
    return "text-embedding-3-small"


@pytest.fixture(scope="session")
def embedding_dim() -> int:
    """Default embedding dimension for text-embedding-3-small."""
    return 1536


# ===== DATA FIXTURES =====


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """Basic DataFrame with fruit names for testing."""
    return pd.DataFrame(
        {
            "name": ["apple", "banana", "cherry"],
        }
    )


@pytest.fixture
def sample_series() -> pd.Series:
    """Basic Series with fruit names for testing."""
    return pd.Series(["apple", "banana", "cherry"], name="fruit")


@pytest.fixture
def sentiment_series() -> pd.Series:
    """Series with sentiment data for testing."""
    return pd.Series(
        ["Great product! Love it.", "Terrible quality, broke immediately.", "Average item, nothing special."],
        name="reviews",
    )


@pytest.fixture
def multilingual_series() -> pd.Series:
    """Series with multilingual text for testing."""
    return pd.Series(["Hello world", "Bonjour le monde", "Hola mundo", "こんにちは世界"], name="text")


@pytest.fixture
def large_series(request) -> pd.Series:
    """Large series for performance testing. Size configurable via indirect parametrization."""
    size = getattr(request, "param", 100)  # Default size 100
    return pd.Series([f"item_{i}" for i in range(size)], name="items")


@pytest.fixture
def vector_dataframe() -> pd.DataFrame:
    """DataFrame with vector columns for similarity testing."""
    return pd.DataFrame(
        {
            "vector1": [np.array([1, 0]), np.array([0, 1]), np.array([1, 1])],
            "vector2": [np.array([1, 0]), np.array([0, 1]), np.array([1, -1])],
        }
    )


@pytest.fixture
def review_dataframe() -> pd.DataFrame:
    """DataFrame with product reviews for testing."""
    return pd.DataFrame(
        [
            {"review": "Great product!", "user": "Alice", "rating": 5},
            {"review": "Terrible quality", "user": "Bob", "rating": 1},
            {"review": "Average item", "user": "Charlie", "rating": 3},
        ]
    )


@pytest.fixture
def missing_data_dataframe() -> pd.DataFrame:
    """DataFrame with missing values for fillna testing."""
    return pd.DataFrame(
        {
            "name": ["Alice", "Bob", None, "David", None],
            "age": [25, 30, 35, 40, 45],
            "city": ["Tokyo", "Osaka", "Kyoto", "Tokyo", "Nagoya"],
        }
    )


# ===== PYDANTIC MODEL FIXTURES =====


@pytest.fixture(scope="session")
def fruit_model():
    """Pydantic model for fruit data."""

    class Fruit(BaseModel):
        name: str
        color: str
        taste: str

    return Fruit


@pytest.fixture(scope="session")
def sentiment_model():
    """Pydantic model for sentiment analysis."""

    class SentimentResult(BaseModel):
        sentiment: str = Field(description="Sentiment: positive, negative, or neutral")
        confidence: float = Field(description="Confidence score between 0.0 and 1.0")
        polarity: str = Field(description="Polarity: positive, negative, neutral")

    return SentimentResult


@pytest.fixture(scope="session")
def person_model():
    """Pydantic model for person data."""

    class Person(BaseModel):
        name: str
        age: int
        email: str = Field(description="Email address")

    return Person


# ===== UTILITY FIXTURES =====


@pytest.fixture
def reset_environment():
    """Reset environment variables after test to avoid test pollution."""
    original_env = {
        "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY"),
        "AZURE_OPENAI_API_KEY": os.environ.get("AZURE_OPENAI_API_KEY"),
        "AZURE_OPENAI_BASE_URL": os.environ.get("AZURE_OPENAI_BASE_URL"),
        "AZURE_OPENAI_API_VERSION": os.environ.get("AZURE_OPENAI_API_VERSION"),
    }
    yield
    # Restore original environment
    for key, value in original_env.items():
        if value is not None:
            os.environ[key] = value
        elif key in os.environ:
            del os.environ[key]


@pytest.fixture
def mock_embedding_vector(embedding_dim):
    """Factory fixture for creating mock embedding vectors."""

    def _create_vector(seed: int = 42) -> np.ndarray:
        rng = np.random.default_rng(seed)
        vector = rng.random(embedding_dim).astype(np.float32)
        # Normalize to unit vector
        return vector / np.linalg.norm(vector)

    return _create_vector


# ===== PERFORMANCE FIXTURES =====


@pytest.fixture(scope="session")
def performance_timer():
    """Utility for timing test operations."""
    import time

    class Timer:
        def __init__(self):
            self.times = {}

        def time_operation(self, name: str):
            """Context manager for timing operations."""
            return self._TimeContext(self, name)

        class _TimeContext:
            def __init__(self, timer, name):
                self.timer = timer
                self.name = name

            def __enter__(self):
                self.start = time.perf_counter()
                return self

            def __exit__(self, *args):
                self.timer.times[self.name] = time.perf_counter() - self.start

    return Timer()


# ===== CACHE FIXTURES =====


@pytest.fixture
def batch_cache():
    """BatchingMapProxy cache for testing."""
    from openaivec._proxy import BatchingMapProxy

    return BatchingMapProxy(batch_size=32)


@pytest.fixture
def async_batch_cache():
    """AsyncBatchingMapProxy cache for testing."""
    from openaivec._proxy import AsyncBatchingMapProxy

    return AsyncBatchingMapProxy(batch_size=32, max_concurrency=4)


# ===== PARAMETRIZED FIXTURES =====


@pytest.fixture(params=[1, 2, 4, 8])
def batch_sizes(request) -> int:
    """Parametrized batch sizes for testing."""
    return request.param


@pytest.fixture(params=[0.0, 0.3, 0.7, 1.0])
def temperature_values(request) -> float:
    """Parametrized temperature values for testing."""
    return request.param


@pytest.fixture(params=[str, dict])
def response_formats(request) -> type:
    """Parametrized response formats for testing."""
    return request.param


# ===== TEST DATA FACTORIES =====


@pytest.fixture
def create_test_series():
    """Factory for creating test series with various configurations."""

    def _create(data: list[Any], name: str = None, dtype=None) -> pd.Series:
        return pd.Series(data, name=name, dtype=dtype)

    return _create


@pytest.fixture
def create_test_dataframe():
    """Factory for creating test DataFrames with various configurations."""

    def _create(data: dict[str, list], index=None) -> pd.DataFrame:
        return pd.DataFrame(data, index=index)

    return _create


# ===== SPARK FIXTURES =====


@pytest.fixture(scope="session")
def spark_session():
    """SparkSession for Spark-related tests."""
    try:
        from pyspark.sql import SparkSession

        spark = (
            SparkSession.builder.appName("TestSparkSession")
            .master("local[*]")
            .config("spark.driver.memory", "1g")
            .config("spark.executor.memory", "1g")
            .config("spark.sql.adaptive.enabled", "false")
            .getOrCreate()
        )

        spark.sparkContext.setLogLevel("WARN")  # Reduce log noise
        yield spark

        spark.stop()
    except ImportError:
        pytest.skip("PySpark not available")
