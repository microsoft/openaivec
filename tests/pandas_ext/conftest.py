"""Shared fixtures for pandas_ext tests."""

import pytest

from openaivec import pandas_ext


@pytest.fixture(autouse=True)
def configure_clients(openai_client, async_openai_client, responses_model_name, embeddings_model_name):
    """Wire test clients and model names into pandas_ext before each test."""
    pandas_ext.set_client(openai_client)
    pandas_ext.set_async_client(async_openai_client)
    pandas_ext.set_responses_model(responses_model_name)
    pandas_ext.set_embeddings_model(embeddings_model_name)
    yield
