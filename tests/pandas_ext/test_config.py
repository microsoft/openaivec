"""Tests for configuration and parameter consistency."""

import inspect

import pandas as pd
import pytest

import openaivec
from openaivec import pandas_ext  # noqa: F401 — registers .ai/.aio accessors


@pytest.mark.requires_api
class TestConfig:
    def test_configuration_methods(self, openai_client, async_openai_client):
        assert callable(openaivec.set_client)
        assert callable(openaivec.get_client)
        assert callable(openaivec.set_async_client)
        assert callable(openaivec.get_async_client)
        assert callable(openaivec.set_responses_model)
        assert callable(openaivec.get_responses_model)
        assert callable(openaivec.set_embeddings_model)
        assert callable(openaivec.get_embeddings_model)

        try:
            openaivec.set_client(openai_client)
            assert openaivec.get_client() is openai_client
            openaivec.set_async_client(async_openai_client)
            assert openaivec.get_async_client() is async_openai_client
            openaivec.set_responses_model("gpt-4.1-mini")
            assert openaivec.get_responses_model() == "gpt-4.1-mini"
            openaivec.set_embeddings_model("text-embedding-3-small")
            assert openaivec.get_embeddings_model() == "text-embedding-3-small"
        except Exception as e:
            pytest.fail(f"Model configuration failed unexpectedly: {e}")

    def test_show_progress_parameter_consistency(self):
        series = pd.Series(["test"])
        df = pd.DataFrame({"col": ["test"]})

        assert "show_progress" in inspect.signature(series.ai.responses).parameters
        assert "show_progress" in inspect.signature(series.ai.embeddings).parameters
        assert "show_progress" in inspect.signature(series.ai.task).parameters
        assert "show_progress" in inspect.signature(df.ai.responses).parameters
        assert "show_progress" in inspect.signature(df.ai.task).parameters
        assert "show_progress" in inspect.signature(df.ai.fillna).parameters

        assert "show_progress" in inspect.signature(series.aio.responses).parameters
        assert "show_progress" in inspect.signature(series.aio.embeddings).parameters
        assert "show_progress" in inspect.signature(series.aio.task).parameters
        assert "show_progress" in inspect.signature(df.aio.responses).parameters
        assert "show_progress" in inspect.signature(df.aio.task).parameters
        assert "show_progress" in inspect.signature(df.aio.fillna).parameters

    def test_max_concurrency_parameter_consistency(self):
        series = pd.Series(["test"])
        df = pd.DataFrame({"col": ["test"]})

        assert "max_concurrency" not in inspect.signature(series.ai.responses).parameters
        assert "max_concurrency" not in inspect.signature(series.ai.embeddings).parameters
        assert "max_concurrency" not in inspect.signature(series.ai.task).parameters
        assert "max_concurrency" not in inspect.signature(df.ai.responses).parameters
        assert "max_concurrency" not in inspect.signature(df.ai.task).parameters
        assert "max_concurrency" not in inspect.signature(df.ai.fillna).parameters

        assert "max_concurrency" in inspect.signature(series.aio.responses).parameters
        assert "max_concurrency" in inspect.signature(series.aio.embeddings).parameters
        assert "max_concurrency" in inspect.signature(series.aio.task).parameters
        assert "max_concurrency" in inspect.signature(df.aio.responses).parameters
        assert "max_concurrency" in inspect.signature(df.aio.task).parameters
        assert "max_concurrency" in inspect.signature(df.aio.fillna).parameters

    def test_method_parameter_ordering(self):
        series = pd.Series(["test"])
        responses_params = list(inspect.signature(series.ai.responses).parameters.keys())
        aio_responses_params = list(inspect.signature(series.aio.responses).parameters.keys())
        common_params = ["instructions", "response_format", "batch_size", "show_progress"]
        sync_filtered = [p for p in responses_params if p in common_params]
        assert sync_filtered == common_params
        async_filtered = [p for p in aio_responses_params if p in common_params or p == "max_concurrency"]
        expected_async = common_params[:-1] + ["max_concurrency"] + [common_params[-1]]
        assert async_filtered == expected_async
