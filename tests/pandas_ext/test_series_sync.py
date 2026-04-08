"""Tests for the synchronous pandas Series accessor (``.ai``)."""

import numpy as np
import pandas as pd
import pytest
from pydantic import BaseModel

from openaivec import pandas_ext  # noqa: F401 – registers accessors


@pytest.mark.requires_api
class TestSeriesSync:
    def test_series_embeddings(self, sample_dataframe):
        embeddings = sample_dataframe["name"].ai.embeddings()
        assert len(embeddings) == 3
        assert all(len(emb) > 0 for emb in embeddings)
        assert embeddings.index.equals(sample_dataframe.index)

    def test_series_responses(self, sample_dataframe):
        names_fr = sample_dataframe["name"].ai.responses("translate to French")
        assert all(isinstance(x, str) for x in names_fr)
        assert names_fr.shape == (3,)
        assert names_fr.index.equals(sample_dataframe.index)

    def test_series_count_tokens(self, sample_dataframe):
        num_tokens = sample_dataframe.name.ai.count_tokens()
        assert all(isinstance(num_token, int) for num_token in num_tokens)
        assert num_tokens.shape == (3,)

    def test_series_parse(self, sentiment_series):
        results = sentiment_series.ai.parse(
            instructions="Extract sentiment (positive/negative/neutral) and a confidence score (0-1)",
            batch_size=2,
            show_progress=False,
        )
        assert len(results) == 3
        assert results.index.equals(sentiment_series.index)
        assert all(isinstance(result, (dict, BaseModel)) for result in results)

    def test_series_infer_schema(self):
        reviews = pd.Series(
            [
                "Great product! 5 stars. Fast shipping.",
                "Poor quality. 1 star. Broke after one day.",
                "Average item. 3 stars. Decent value.",
                "Excellent service! 5 stars. Highly recommend.",
                "Terrible experience. 2 stars. Slow delivery.",
            ]
        )
        schema = reviews.ai.infer_schema(instructions="Extract product review analysis data", max_examples=3)
        assert schema is not None
        assert schema.model is not None
        assert schema.task is not None
        assert schema.object_spec is not None
        assert schema.object_spec.fields is not None
        assert isinstance(schema.object_spec.fields, list)
        assert len(schema.object_spec.fields) > 0
        assert hasattr(schema.model, "__name__")

    def test_series_task(self):
        from openaivec._model import PreparedTask

        task = PreparedTask(instructions="Translate to French", response_format=str)
        series = pd.Series(["cat", "dog"])
        results = series.ai.task(task=task, batch_size=2, show_progress=False, temperature=0.0, top_p=1.0)
        assert len(results) == 2
        assert results.index.equals(series.index)
        assert all(isinstance(result, str) for result in results)

    def test_series_extract_pydantic(self, fruit_model):
        sample_series = pd.Series(
            [
                fruit_model(name="apple", color="red", taste="crunchy"),
                fruit_model(name="banana", color="yellow", taste="soft"),
                fruit_model(name="cherry", color="red", taste="tart"),
            ],
            name="fruit",
        )
        extracted_df = sample_series.ai.extract()
        expected_columns = ["fruit_name", "fruit_color", "fruit_taste"]
        assert list(extracted_df.columns) == expected_columns

    def test_series_extract_dict(self):
        sample_series = pd.Series(
            [
                {"name": "apple", "color": "red", "taste": "crunchy"},
                {"name": "banana", "color": "yellow", "taste": "soft"},
                {"name": "cherry", "color": "red", "taste": "tart"},
            ],
            name="fruit",
        )
        extracted_df = sample_series.ai.extract()
        expected_columns = ["fruit_name", "fruit_color", "fruit_taste"]
        assert list(extracted_df.columns) == expected_columns

    def test_series_extract_without_name(self, fruit_model):
        sample_series = pd.Series(
            [
                fruit_model(name="apple", color="red", taste="crunchy"),
                fruit_model(name="banana", color="yellow", taste="soft"),
                fruit_model(name="cherry", color="red", taste="tart"),
            ]
        )
        extracted_df = sample_series.ai.extract()
        expected_columns = ["name", "color", "taste"]
        assert list(extracted_df.columns) == expected_columns

    def test_series_extract_with_none(self, fruit_model):
        sample_series = pd.Series(
            [
                fruit_model(name="apple", color="red", taste="crunchy"),
                None,
                fruit_model(name="banana", color="yellow", taste="soft"),
            ],
            name="fruit",
        )
        extracted_df = sample_series.ai.extract()
        expected_columns = ["fruit_name", "fruit_color", "fruit_taste"]
        assert list(extracted_df.columns) == expected_columns
        assert extracted_df.iloc[1].isna().all()

    def test_series_extract_with_invalid_row(self, fruit_model):
        sample_series = pd.Series(
            [
                fruit_model(name="apple", color="red", taste="crunchy"),
                123,
                fruit_model(name="banana", color="yellow", taste="soft"),
            ],
            name="fruit",
        )
        extracted_df = sample_series.ai.extract()
        expected_columns = ["fruit_name", "fruit_color", "fruit_taste"]
        assert list(extracted_df.columns) == expected_columns
        assert extracted_df.iloc[1].isna().all()

    def test_structured_output_with_pydantic(self, sentiment_model):
        series = pd.Series(["I love this product!", "This is terrible"])
        results = series.ai.responses(
            instructions="Analyze sentiment and provide confidence score",
            response_format=sentiment_model,
            batch_size=2,
            show_progress=False,
        )
        assert len(results) == 2
        for result in results:
            assert isinstance(result, sentiment_model)
            assert result.sentiment.lower() in ["positive", "negative", "neutral"]
            assert isinstance(result.confidence, float)

    def test_shared_cache_responses_sync(self):
        from openaivec._cache import BatchCache

        shared_cache = BatchCache(batch_size=32)
        series1 = pd.Series(["cat", "dog", "elephant"])
        series2 = pd.Series(["dog", "elephant", "lion"])

        result1 = series1.ai.responses_with_cache(instructions="translate to French", cache=shared_cache)
        result2 = series2.ai.responses_with_cache(instructions="translate to French", cache=shared_cache)

        assert all(isinstance(x, str) for x in result1)
        assert all(isinstance(x, str) for x in result2)
        assert len(result1) == 3
        assert len(result2) == 3
        dog_idx1 = series1.loc[series1 == "dog"].index[0]
        dog_idx2 = series2.loc[series2 == "dog"].index[0]
        elephant_idx1 = series1.loc[series1 == "elephant"].index[0]
        elephant_idx2 = series2.loc[series2 == "elephant"].index[0]
        assert result1[dog_idx1] == result2[dog_idx2]
        assert result1[elephant_idx1] == result2[elephant_idx2]

    def test_shared_cache_embeddings_sync(self):
        from openaivec._cache import BatchCache

        shared_cache = BatchCache(batch_size=32)
        series1 = pd.Series(["apple", "banana", "cherry"])
        series2 = pd.Series(["banana", "cherry", "date"])

        embeddings1 = series1.ai.embeddings_with_cache(cache=shared_cache)
        embeddings2 = series2.ai.embeddings_with_cache(cache=shared_cache)

        assert all(len(emb) > 0 for emb in embeddings1)
        assert all(len(emb) > 0 for emb in embeddings2)
        assert len(embeddings1) == 3
        assert len(embeddings2) == 3
        banana_idx1 = series1.loc[series1 == "banana"].index[0]
        banana_idx2 = series2.loc[series2 == "banana"].index[0]
        cherry_idx1 = series1.loc[series1 == "cherry"].index[0]
        cherry_idx2 = series2.loc[series2 == "cherry"].index[0]
        np.testing.assert_array_equal(
            np.asarray(embeddings1[banana_idx1], dtype=np.float32),
            np.asarray(embeddings2[banana_idx2], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            np.asarray(embeddings1[cherry_idx1], dtype=np.float32),
            np.asarray(embeddings2[cherry_idx2], dtype=np.float32),
        )

    def test_parse_with_cache_methods(self):
        from openaivec._cache import BatchCache

        series = pd.Series(["Good product", "Bad experience"])
        cache = BatchCache(batch_size=2)
        results = series.ai.parse_with_cache(instructions="Extract sentiment", cache=cache)
        assert len(results) == 2
        assert all(isinstance(result, (dict, BaseModel)) for result in results)

    def test_empty_series_handling(self):
        empty_series = pd.Series([], dtype=str)
        embeddings = empty_series.ai.embeddings()
        assert len(embeddings) == 0
        assert embeddings.index.equals(empty_series.index)
        responses = empty_series.ai.responses("translate to French")
        assert len(responses) == 0
        assert responses.index.equals(empty_series.index)
        tokens = empty_series.ai.count_tokens()
        assert len(tokens) == 0

    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    def test_batch_size_consistency(self, sample_series, batch_size):
        result1 = sample_series.ai.responses("translate to French", batch_size=batch_size, show_progress=False)
        result2 = sample_series.ai.responses("translate to French", batch_size=batch_size, show_progress=False)
        assert len(result1) == len(result2) == len(sample_series)
        assert result1.index.equals(result2.index)
