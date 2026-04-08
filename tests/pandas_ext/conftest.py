"""Shared fixtures for pandas_ext tests."""

import pytest

import openaivec
from openaivec import pandas_ext  # noqa: F401 — registers .ai/.aio accessors


@pytest.fixture(autouse=True)
def configure_clients(openai_client, async_openai_client, responses_model_name, embeddings_model_name):
    """Wire test clients and model names before each test."""
    openaivec.set_client(openai_client)
    openaivec.set_async_client(async_openai_client)
    openaivec.set_responses_model(responses_model_name)
    openaivec.set_embeddings_model(embeddings_model_name)
    yield
