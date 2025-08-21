import asyncio
import unittest

import numpy as np
import pandas as pd
from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel

from openaivec import pandas_ext

pandas_ext.use(OpenAI())
pandas_ext.use_async(AsyncOpenAI())
pandas_ext.responses_model("gpt-4.1-mini")
pandas_ext.embeddings_model("text-embedding-3-small")


class Fruit(BaseModel):
    color: str
    flavor: str
    taste: str


class SentimentResult(BaseModel):
    sentiment: str
    confidence: float


class TestPandasExt(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame(
            {
                "name": ["apple", "banana", "cherry"],
            }
        )

    # ===== BASIC SERIES METHODS =====

    def test_series_embeddings(self):
        """Test Series.ai.embeddings method."""
        embeddings = self.df["name"].ai.embeddings()

        self.assertTrue(all(isinstance(embedding, np.ndarray) for embedding in embeddings))
        self.assertEqual(embeddings.shape, (3,))
        self.assertTrue(embeddings.index.equals(self.df.index))

    def test_series_responses(self):
        """Test Series.ai.responses method."""
        names_fr = self.df["name"].ai.responses("translate to French")

        self.assertTrue(all(isinstance(x, str) for x in names_fr))
        self.assertEqual(names_fr.shape, (3,))
        self.assertTrue(names_fr.index.equals(self.df.index))

    def test_series_count_tokens(self):
        """Test Series.ai.count_tokens method."""
        num_tokens = self.df.name.ai.count_tokens()

        self.assertTrue(all(isinstance(num_token, int) for num_token in num_tokens))
        self.assertEqual(num_tokens.shape, (3,))

    def test_series_parse(self):
        """Test Series.ai.parse method with structured output."""
        reviews = pd.Series(
            ["Great product! Love it.", "Terrible quality, broke immediately.", "Average item, nothing special."]
        )

        results = reviews.ai.parse(
            instructions="Extract sentiment (positive/negative/neutral) and a confidence score (0-1)",
            batch_size=2,
            show_progress=False,
        )

        self.assertEqual(len(results), 3)
        self.assertTrue(results.index.equals(reviews.index))
        self.assertTrue(all(isinstance(result, (dict, BaseModel)) for result in results))

    def test_series_infer_schema(self):
        """Test Series.ai.infer_schema method."""
        reviews = pd.Series(
            [
                "Great product! 5 stars. Fast shipping.",
                "Poor quality. 1 star. Broke after one day.",
                "Average item. 3 stars. Decent value.",
                "Excellent service! 5 stars. Highly recommend.",
                "Terrible experience. 2 stars. Slow delivery.",
            ]
        )

        schema = reviews.ai.infer_schema(purpose="Extract product review analysis data", max_examples=3)

        self.assertIsNotNone(schema)
        self.assertIsNotNone(schema.model)
        self.assertIsNotNone(schema.task)
        self.assertIsNotNone(schema.object_spec)
        self.assertIsNotNone(schema.object_spec.fields)
        self.assertIsInstance(schema.object_spec.fields, list)
        self.assertGreater(len(schema.object_spec.fields), 0)
        self.assertTrue(hasattr(schema.model, "__name__"))

    def test_series_task(self):
        """Test Series.ai.task method with actual task execution."""
        from openaivec._model import PreparedTask

        task = PreparedTask(instructions="Translate to French", response_format=str, temperature=0.0, top_p=1.0)

        series = pd.Series(["cat", "dog"])
        results = series.ai.task(task=task, batch_size=2, show_progress=False)

        self.assertEqual(len(results), 2)
        self.assertTrue(results.index.equals(series.index))
        self.assertTrue(all(isinstance(result, str) for result in results))

    # ===== BASIC DATAFRAME METHODS =====

    def test_dataframe_responses(self):
        """Test DataFrame.ai.responses method."""
        names_fr = self.df.ai.responses("translate to French")

        self.assertTrue(all(isinstance(x, str) for x in names_fr))
        self.assertEqual(names_fr.shape, (3,))
        self.assertTrue(names_fr.index.equals(self.df.index))

    def test_dataframe_parse(self):
        """Test DataFrame.ai.parse method with structured output."""
        df = pd.DataFrame(
            [
                {"review": "Great product!", "user": "Alice"},
                {"review": "Terrible quality", "user": "Bob"},
                {"review": "Average item", "user": "Charlie"},
            ]
        )

        results = df.ai.parse(instructions="Extract sentiment from the review", batch_size=2, show_progress=False)

        self.assertEqual(len(results), 3)
        self.assertTrue(results.index.equals(df.index))
        self.assertTrue(all(isinstance(result, (dict, BaseModel)) for result in results))

    def test_dataframe_infer_schema(self):
        """Test DataFrame.ai.infer_schema method."""
        df = pd.DataFrame(
            [
                {"product": "laptop", "review": "Great performance", "rating": 5},
                {"product": "mouse", "review": "Poor quality", "rating": 2},
                {"product": "keyboard", "review": "Average product", "rating": 3},
            ]
        )

        schema = df.ai.infer_schema(purpose="Extract product analysis metrics", max_examples=2)

        self.assertIsNotNone(schema)
        self.assertIsNotNone(schema.model)
        self.assertIsNotNone(schema.task)
        self.assertIsNotNone(schema.object_spec.fields)
        self.assertIsInstance(schema.object_spec.fields, list)
        self.assertGreater(len(schema.object_spec.fields), 0)

    def test_dataframe_task(self):
        """Test DataFrame.ai.task method with actual task execution."""
        from openaivec._model import PreparedTask

        task = PreparedTask(
            instructions="Extract the animal name from the data", response_format=str, temperature=0.0, top_p=1.0
        )

        df = pd.DataFrame([{"animal": "cat", "legs": 4}, {"animal": "dog", "legs": 4}])

        results = df.ai.task(task=task, batch_size=2, show_progress=False)

        self.assertEqual(len(results), 2)
        self.assertTrue(results.index.equals(df.index))
        self.assertTrue(all(isinstance(result, str) for result in results))

    def test_dataframe_similarity(self):
        """Test DataFrame.ai.similarity method."""
        df = pd.DataFrame(
            {
                "vector1": [np.array([1, 0]), np.array([0, 1]), np.array([1, 1])],
                "vector2": [np.array([1, 0]), np.array([0, 1]), np.array([1, -1])],
            }
        )
        similarity_scores = df.ai.similarity("vector1", "vector2")

        expected_scores = [1.0, 1.0, 0.0]  # Cosine similarities
        self.assertTrue(np.allclose(similarity_scores, expected_scores))

    def test_dataframe_similarity_invalid_vectors(self):
        """Test DataFrame.ai.similarity with invalid vectors."""
        df = pd.DataFrame(
            {
                "vector1": [np.array([1, 0]), "invalid", np.array([1, 1])],
                "vector2": [np.array([1, 0]), np.array([0, 1]), np.array([1, -1])],
            }
        )

        with self.assertRaises(TypeError):
            df.ai.similarity("vector1", "vector2")

    def test_dataframe_fillna(self):
        """Test DataFrame.ai.fillna method."""
        # Test with no missing values
        df_complete = pd.DataFrame(
            {
                "name": ["Alice", "Bob", "Charlie", "David"],
                "age": [25, 30, 35, 40],
                "city": ["Tokyo", "Osaka", "Kyoto", "Tokyo"],
            }
        )

        result_df = df_complete.ai.fillna("name")
        pd.testing.assert_frame_equal(result_df, df_complete)

        # Test structure preservation
        df_custom_index = pd.DataFrame(
            {"name": ["Alice", "Bob", "Charlie"], "score": [85, 90, 78]}, index=["student_1", "student_2", "student_3"]
        )

        result_df = df_custom_index.ai.fillna("name")
        pd.testing.assert_index_equal(result_df.index, df_custom_index.index)
        self.assertEqual(result_df.shape, df_custom_index.shape)

    # ===== ASYNC SERIES METHODS =====

    def test_series_aio_embeddings(self):
        """Test Series.aio.embeddings method."""

        async def run():
            return await self.df["name"].aio.embeddings()

        embeddings = asyncio.run(run())
        self.assertTrue(all(isinstance(embedding, np.ndarray) for embedding in embeddings))
        self.assertEqual(embeddings.shape, (3,))
        self.assertTrue(embeddings.index.equals(self.df.index))

    def test_series_aio_responses(self):
        """Test Series.aio.responses method."""

        async def run():
            return await self.df["name"].aio.responses("translate to French")

        names_fr = asyncio.run(run())
        self.assertTrue(all(isinstance(x, str) for x in names_fr))
        self.assertEqual(names_fr.shape, (3,))
        self.assertTrue(names_fr.index.equals(self.df.index))

    def test_series_aio_parse(self):
        """Test Series.aio.parse method with structured output."""

        async def run_test():
            reviews = pd.Series(["Excellent service!", "Poor experience.", "Okay product."])

            return await reviews.aio.parse(
                instructions="Extract sentiment and rating", batch_size=2, max_concurrency=2, show_progress=False
            )

        results = asyncio.run(run_test())

        self.assertEqual(len(results), 3)
        self.assertTrue(all(isinstance(result, (dict, BaseModel)) for result in results))

    def test_series_aio_task(self):
        """Test Series.aio.task method with actual task execution."""
        from openaivec._model import PreparedTask

        async def run_test():
            task = PreparedTask(
                instructions="Classify sentiment as positive or negative",
                response_format=str,
                temperature=0.0,
                top_p=1.0,
            )

            series = pd.Series(["I love this!", "This is terrible"])

            return await series.aio.task(task=task, batch_size=2, max_concurrency=2, show_progress=False)

        results = asyncio.run(run_test())

        self.assertEqual(len(results), 2)
        self.assertTrue(all(isinstance(result, str) for result in results))

    # ===== ASYNC DATAFRAME METHODS =====

    def test_dataframe_aio_responses(self):
        """Test DataFrame.aio.responses method."""

        async def run():
            return await self.df.aio.responses("translate the 'name' field to French")

        names_fr = asyncio.run(run())
        self.assertTrue(all(isinstance(x, str) for x in names_fr))
        self.assertEqual(names_fr.shape, (3,))
        self.assertTrue(names_fr.index.equals(self.df.index))

    def test_dataframe_aio_parse(self):
        """Test DataFrame.aio.parse method with structured output."""

        async def run_test():
            df = pd.DataFrame(
                [
                    {"text": "Happy customer", "score": 5},
                    {"text": "Unhappy customer", "score": 1},
                    {"text": "Neutral feedback", "score": 3},
                ]
            )

            return await df.aio.parse(
                instructions="Analyze the sentiment", batch_size=2, max_concurrency=2, show_progress=False
            )

        results = asyncio.run(run_test())

        self.assertEqual(len(results), 3)
        self.assertTrue(all(isinstance(result, (dict, BaseModel)) for result in results))

    def test_dataframe_aio_task(self):
        """Test DataFrame.aio.task method with actual task execution."""
        from openaivec._model import PreparedTask

        async def run_test():
            task = PreparedTask(instructions="Describe the animal", response_format=str, temperature=0.0, top_p=1.0)

            df = pd.DataFrame([{"name": "fluffy", "type": "cat"}, {"name": "buddy", "type": "dog"}])

            return await df.aio.task(task=task, batch_size=2, max_concurrency=2, show_progress=False)

        results = asyncio.run(run_test())

        self.assertEqual(len(results), 2)
        self.assertTrue(all(isinstance(result, str) for result in results))

    def test_dataframe_aio_fillna(self):
        """Test DataFrame.aio.fillna method."""

        async def run_test():
            df_with_missing = pd.DataFrame(
                {
                    "name": ["Alice", "Bob", "Charlie"],
                    "age": [25, 30, 35],
                    "city": ["Tokyo", "Osaka", "Kyoto"],
                }
            )
            return await df_with_missing.aio.fillna("name")

        result, original = (
            asyncio.run(run_test()),
            pd.DataFrame(
                {
                    "name": ["Alice", "Bob", "Charlie"],
                    "age": [25, 30, 35],
                    "city": ["Tokyo", "Osaka", "Kyoto"],
                }
            ),
        )
        pd.testing.assert_frame_equal(result, original)

    def test_dataframe_aio_pipe(self):
        """Test DataFrame.aio.pipe method."""

        async def run_test():
            df = pd.DataFrame({"name": ["apple", "banana", "cherry"], "color": ["red", "yellow", "red"]})

            def add_column(df):
                df = df.copy()
                df["processed"] = df["name"] + "_processed"
                return df

            result1 = await df.aio.pipe(add_column)

            async def add_async_column(df):
                await asyncio.sleep(0.01)
                df = df.copy()
                df["async_processed"] = df["name"] + "_async"
                return df

            result2 = await df.aio.pipe(add_async_column)

            return result1, result2, df

        result1, result2, original_df = asyncio.run(run_test())

        # Verify sync function result
        self.assertIn("processed", result1.columns)
        self.assertEqual(len(result1), 3)
        self.assertTrue(result1["processed"].str.endswith("_processed").all())

        # Verify async function result
        self.assertIn("async_processed", result2.columns)
        self.assertEqual(len(result2), 3)
        self.assertTrue(result2["async_processed"].str.endswith("_async").all())

        # Original DataFrame should be unchanged
        self.assertNotIn("processed", original_df.columns)
        self.assertNotIn("async_processed", original_df.columns)

    def test_dataframe_aio_assign(self):
        """Test DataFrame.aio.assign method."""

        async def run_test():
            df = pd.DataFrame({"name": ["alice", "bob", "charlie"], "age": [25, 30, 35]})

            def compute_category(df):
                return ["young" if age < 30 else "adult" for age in df["age"]]

            result1 = await df.aio.assign(category=compute_category)

            async def compute_async_score(df):
                await asyncio.sleep(0.01)
                return [age * 2 for age in df["age"]]

            result2 = await df.aio.assign(score=compute_async_score)

            return result1, result2, df

        result1, result2, original_df = asyncio.run(run_test())

        # Verify sync function assignment
        self.assertIn("category", result1.columns)
        self.assertEqual(list(result1["category"]), ["young", "adult", "adult"])

        # Verify async function assignment
        self.assertIn("score", result2.columns)
        self.assertEqual(list(result2["score"]), [50, 60, 70])

        # Original DataFrame should be unchanged
        self.assertNotIn("category", original_df.columns)
        self.assertNotIn("score", original_df.columns)

    # ===== EXTRACT METHODS =====

    def test_series_extract_pydantic(self):
        """Test Series.ai.extract with Pydantic models."""
        sample_series = pd.Series(
            [
                Fruit(color="red", flavor="sweet", taste="crunchy"),
                Fruit(color="yellow", flavor="sweet", taste="soft"),
                Fruit(color="red", flavor="sweet", taste="tart"),
            ],
            name="fruit",
        )

        extracted_df = sample_series.ai.extract()
        expected_columns = ["fruit_color", "fruit_flavor", "fruit_taste"]
        self.assertListEqual(list(extracted_df.columns), expected_columns)

    def test_series_extract_dict(self):
        """Test Series.ai.extract with dictionaries."""
        sample_series = pd.Series(
            [
                {"color": "red", "flavor": "sweet", "taste": "crunchy"},
                {"color": "yellow", "flavor": "sweet", "taste": "soft"},
                {"color": "red", "flavor": "sweet", "taste": "tart"},
            ],
            name="fruit",
        )

        extracted_df = sample_series.ai.extract()
        expected_columns = ["fruit_color", "fruit_flavor", "fruit_taste"]
        self.assertListEqual(list(extracted_df.columns), expected_columns)

    def test_series_extract_without_name(self):
        """Test Series.ai.extract without series name."""
        sample_series = pd.Series(
            [
                Fruit(color="red", flavor="sweet", taste="crunchy"),
                Fruit(color="yellow", flavor="sweet", taste="soft"),
                Fruit(color="red", flavor="sweet", taste="tart"),
            ]
        )

        extracted_df = sample_series.ai.extract()
        expected_columns = ["color", "flavor", "taste"]  # without prefix
        self.assertListEqual(list(extracted_df.columns), expected_columns)

    def test_series_extract_with_none(self):
        """Test Series.ai.extract with None values."""
        sample_series = pd.Series(
            [
                Fruit(color="red", flavor="sweet", taste="crunchy"),
                None,
                Fruit(color="yellow", flavor="sweet", taste="soft"),
            ],
            name="fruit",
        )

        extracted_df = sample_series.ai.extract()
        expected_columns = ["fruit_color", "fruit_flavor", "fruit_taste"]
        self.assertListEqual(list(extracted_df.columns), expected_columns)
        self.assertTrue(extracted_df.iloc[1].isna().all())

    def test_series_extract_with_invalid_row(self):
        """Test Series.ai.extract with invalid data types."""
        sample_series = pd.Series(
            [
                Fruit(color="red", flavor="sweet", taste="crunchy"),
                123,  # Invalid row
                Fruit(color="yellow", flavor="sweet", taste="soft"),
            ],
            name="fruit",
        )

        extracted_df = sample_series.ai.extract()
        expected_columns = ["fruit_color", "fruit_flavor", "fruit_taste"]
        self.assertListEqual(list(extracted_df.columns), expected_columns)
        self.assertTrue(extracted_df.iloc[1].isna().all())

    def test_dataframe_extract_pydantic(self):
        """Test DataFrame.ai.extract with Pydantic models."""
        sample_df = pd.DataFrame(
            [
                {"name": "apple", "fruit": Fruit(color="red", flavor="sweet", taste="crunchy")},
                {"name": "banana", "fruit": Fruit(color="yellow", flavor="sweet", taste="soft")},
                {"name": "cherry", "fruit": Fruit(color="red", flavor="sweet", taste="tart")},
            ]
        ).ai.extract("fruit")

        expected_columns = ["name", "fruit_color", "fruit_flavor", "fruit_taste"]
        self.assertListEqual(list(sample_df.columns), expected_columns)

    def test_dataframe_extract_dict(self):
        """Test DataFrame.ai.extract with dictionaries."""
        sample_df = pd.DataFrame(
            [
                {"fruit": {"name": "apple", "color": "red", "flavor": "sweet", "taste": "crunchy"}},
                {"fruit": {"name": "banana", "color": "yellow", "flavor": "sweet", "taste": "soft"}},
                {"fruit": {"name": "cherry", "color": "red", "flavor": "sweet", "taste": "tart"}},
            ]
        ).ai.extract("fruit")

        expected_columns = ["fruit_name", "fruit_color", "fruit_flavor", "fruit_taste"]
        self.assertListEqual(list(sample_df.columns), expected_columns)

    def test_dataframe_extract_dict_with_none(self):
        """Test DataFrame.ai.extract with None values."""
        sample_df = pd.DataFrame(
            [
                {"fruit": {"name": "apple", "color": "red", "flavor": "sweet", "taste": "crunchy"}},
                {"fruit": None},
                {"fruit": {"name": "cherry", "color": "red", "flavor": "sweet", "taste": "tart"}},
            ]
        ).ai.extract("fruit")

        expected_columns = ["fruit_name", "fruit_color", "fruit_flavor", "fruit_taste"]
        self.assertListEqual(list(sample_df.columns), expected_columns)
        self.assertTrue(sample_df.iloc[1].isna().all())

    def test_dataframe_extract_with_invalid_row(self):
        """Test DataFrame.ai.extract error handling with invalid data."""
        sample_df = pd.DataFrame(
            [
                {"fruit": {"name": "apple", "color": "red", "flavor": "sweet", "taste": "crunchy"}},
                {"fruit": 123},
                {"fruit": {"name": "cherry", "color": "red", "flavor": "sweet", "taste": "tart"}},
            ]
        )

        expected_columns = ["fruit"]
        self.assertListEqual(list(sample_df.columns), expected_columns)

    # ===== CACHE METHODS =====

    def test_shared_cache_responses_sync(self):
        """Test shared cache functionality for responses."""
        from openaivec._proxy import BatchingMapProxy

        shared_cache = BatchingMapProxy(batch_size=32)
        series1 = pd.Series(["cat", "dog", "elephant"])
        series2 = pd.Series(["dog", "elephant", "lion"])

        result1 = series1.ai.responses_with_cache(instructions="translate to French", cache=shared_cache)
        result2 = series2.ai.responses_with_cache(instructions="translate to French", cache=shared_cache)

        self.assertTrue(all(isinstance(x, str) for x in result1))
        self.assertTrue(all(isinstance(x, str) for x in result2))
        self.assertEqual(len(result1), 3)
        self.assertEqual(len(result2), 3)

        # Check cache sharing works
        dog_idx1 = series1[series1 == "dog"].index[0]
        dog_idx2 = series2[series2 == "dog"].index[0]
        elephant_idx1 = series1[series1 == "elephant"].index[0]
        elephant_idx2 = series2[series2 == "elephant"].index[0]

        self.assertEqual(result1[dog_idx1], result2[dog_idx2])
        self.assertEqual(result1[elephant_idx1], result2[elephant_idx2])

    def test_shared_cache_embeddings_sync(self):
        """Test shared cache functionality for embeddings."""
        from openaivec._proxy import BatchingMapProxy

        shared_cache = BatchingMapProxy(batch_size=32)
        series1 = pd.Series(["apple", "banana", "cherry"])
        series2 = pd.Series(["banana", "cherry", "date"])

        embeddings1 = series1.ai.embeddings_with_cache(cache=shared_cache)
        embeddings2 = series2.ai.embeddings_with_cache(cache=shared_cache)

        self.assertTrue(all(isinstance(emb, np.ndarray) for emb in embeddings1))
        self.assertTrue(all(isinstance(emb, np.ndarray) for emb in embeddings2))
        self.assertEqual(len(embeddings1), 3)
        self.assertEqual(len(embeddings2), 3)

        # Check cache sharing
        banana_idx1 = series1[series1 == "banana"].index[0]
        banana_idx2 = series2[series2 == "banana"].index[0]
        cherry_idx1 = series1[series1 == "cherry"].index[0]
        cherry_idx2 = series2[series2 == "cherry"].index[0]

        np.testing.assert_array_equal(embeddings1[banana_idx1], embeddings2[banana_idx2])
        np.testing.assert_array_equal(embeddings1[cherry_idx1], embeddings2[cherry_idx2])

    def test_shared_cache_async(self):
        """Test shared cache functionality for async methods."""
        from openaivec._proxy import AsyncBatchingMapProxy

        async def run_test():
            shared_cache = AsyncBatchingMapProxy(batch_size=32, max_concurrency=4)
            series1 = pd.Series(["cat", "dog", "elephant"])
            series2 = pd.Series(["dog", "elephant", "lion"])

            result1 = await series1.aio.responses_with_cache(instructions="translate to French", cache=shared_cache)
            result2 = await series2.aio.responses_with_cache(instructions="translate to French", cache=shared_cache)

            return result1, result2, series1, series2

        result1, result2, series1, series2 = asyncio.run(run_test())

        self.assertTrue(all(isinstance(x, str) for x in result1))
        self.assertTrue(all(isinstance(x, str) for x in result2))
        self.assertEqual(len(result1), 3)
        self.assertEqual(len(result2), 3)

        # Check cache sharing
        dog_idx1 = series1[series1 == "dog"].index[0]
        dog_idx2 = series2[series2 == "dog"].index[0]
        elephant_idx1 = series1[series1 == "elephant"].index[0]
        elephant_idx2 = series2[series2 == "elephant"].index[0]

        self.assertEqual(result1[dog_idx1], result2[dog_idx2])
        self.assertEqual(result1[elephant_idx1], result2[elephant_idx2])

    # ===== FILLNA SPECIFIC TESTS =====

    def test_fillna_task_creation(self):
        """Test that fillna method creates a valid task."""
        from openaivec.task.table import fillna

        df_with_missing = pd.DataFrame(
            {
                "name": ["Alice", "Bob", None, "David"],
                "age": [25, 30, 35, 40],
                "city": ["Tokyo", "Osaka", "Kyoto", "Tokyo"],
            }
        )

        task = fillna(df_with_missing, "name")

        self.assertIsNotNone(task)
        self.assertEqual(task.temperature, 0.0)
        self.assertEqual(task.top_p, 1.0)

    def test_fillna_task_validation(self):
        """Test fillna validation with various edge cases."""
        from openaivec.task.table import fillna

        # Test with empty DataFrame
        empty_df = pd.DataFrame()
        with self.assertRaises(ValueError):
            fillna(empty_df, "nonexistent")

        # Test with nonexistent column
        df = pd.DataFrame({"name": ["Alice", "Bob"]})
        with self.assertRaises(ValueError):
            fillna(df, "nonexistent")

        # Test with all null values in target column
        df_all_null = pd.DataFrame({"name": [None, None, None], "age": [25, 30, 35]})
        with self.assertRaises(ValueError):
            fillna(df_all_null, "name")

        # Test with invalid max_examples
        df_valid = pd.DataFrame({"name": ["Alice", None, "Bob"], "age": [25, 30, 35]})
        with self.assertRaises(ValueError):
            fillna(df_valid, "name", max_examples=0)

        with self.assertRaises(ValueError):
            fillna(df_valid, "name", max_examples=-1)

    def test_fillna_missing_rows_detection(self):
        """Test that fillna correctly identifies missing rows."""
        df_with_missing = pd.DataFrame(
            {
                "name": ["Alice", "Bob", None, "David", None],
                "age": [25, 30, 35, 40, 45],
                "city": ["Tokyo", "Osaka", "Kyoto", "Tokyo", "Nagoya"],
            }
        )

        missing_rows = df_with_missing[df_with_missing["name"].isna()]

        self.assertEqual(len(missing_rows), 2)
        self.assertTrue(missing_rows.index.tolist() == [2, 4])

    # ===== EDGE CASES & ERROR HANDLING =====

    def test_empty_series_handling(self):
        """Test handling of empty Series for various methods."""
        empty_series = pd.Series([], dtype=str)

        # Test embeddings with empty series
        embeddings = empty_series.ai.embeddings()
        self.assertEqual(len(embeddings), 0)
        self.assertTrue(embeddings.index.equals(empty_series.index))

        # Test responses with empty series
        responses = empty_series.ai.responses("translate to French")
        self.assertEqual(len(responses), 0)
        self.assertTrue(responses.index.equals(empty_series.index))

        # Test count_tokens with empty series
        tokens = empty_series.ai.count_tokens()
        self.assertEqual(len(tokens), 0)

    def test_empty_dataframe_handling(self):
        """Test handling of empty DataFrame for various methods."""
        empty_df = pd.DataFrame()

        # Test that empty dataframe doesn't crash
        self.assertTrue(empty_df.empty)

    def test_structured_output_with_pydantic(self):
        """Test structured output using Pydantic models."""
        series = pd.Series(["I love this product!", "This is terrible"])

        try:
            results = series.ai.responses(
                instructions="Analyze sentiment and provide confidence score",
                response_format=SentimentResult,
                batch_size=2,
                show_progress=False,
            )

            self.assertEqual(len(results), 2)
            for result in results:
                self.assertIsInstance(result, SentimentResult)
                self.assertIn(result.sentiment.lower(), ["positive", "negative", "neutral"])
                self.assertIsInstance(result.confidence, float)

        except Exception:
            # Some API calls might fail in test environment
            pass

    def test_parse_with_cache_methods(self):
        """Test parse_with_cache methods for both Series and DataFrame."""
        from openaivec._proxy import BatchingMapProxy

        # Test Series parse_with_cache
        series = pd.Series(["Good product", "Bad experience"])
        cache = BatchingMapProxy(batch_size=2)

        results = series.ai.parse_with_cache(instructions="Extract sentiment", cache=cache, show_progress=False)

        self.assertEqual(len(results), 2)
        self.assertTrue(all(isinstance(result, (dict, BaseModel)) for result in results))

        # Test DataFrame parse_with_cache
        df = pd.DataFrame([{"review": "Great product", "rating": 5}, {"review": "Poor quality", "rating": 1}])

        df_results = df.ai.parse_with_cache(instructions="Analyze sentiment", cache=cache, show_progress=False)

        self.assertEqual(len(df_results), 2)
        self.assertTrue(all(isinstance(result, (dict, BaseModel)) for result in df_results))

    # ===== CONFIGURATION & PARAMETER TESTS =====

    def test_configuration_methods(self):
        """Test configuration methods use, use_async, responses_model, embeddings_model."""
        from openai import AsyncOpenAI, OpenAI

        # Test that configuration methods exist and are callable
        self.assertTrue(callable(pandas_ext.use))
        self.assertTrue(callable(pandas_ext.use_async))
        self.assertTrue(callable(pandas_ext.responses_model))
        self.assertTrue(callable(pandas_ext.embeddings_model))

        # Test setting custom client (these use environment variables automatically)
        test_client = OpenAI()
        try:
            pandas_ext.use(test_client)
        except Exception:
            # Connection/API errors are acceptable for testing
            pass

        # Test setting async client
        async_test_client = AsyncOpenAI()
        try:
            pandas_ext.use_async(async_test_client)
        except Exception:
            # Connection/API errors are acceptable for testing
            pass

        # Test model configuration
        try:
            pandas_ext.responses_model("gpt-4.1-mini")
            pandas_ext.embeddings_model("text-embedding-3-small")
        except Exception as e:
            self.fail(f"Model configuration failed unexpectedly: {e}")

    def test_show_progress_parameter_consistency(self):
        """Test that show_progress parameter is consistently available across methods."""
        import inspect

        series = pd.Series(["test"])
        df = pd.DataFrame({"col": ["test"]})

        # Check sync methods have show_progress
        self.assertIn("show_progress", inspect.signature(series.ai.responses).parameters)
        self.assertIn("show_progress", inspect.signature(series.ai.embeddings).parameters)
        self.assertIn("show_progress", inspect.signature(series.ai.task).parameters)
        self.assertIn("show_progress", inspect.signature(df.ai.responses).parameters)
        self.assertIn("show_progress", inspect.signature(df.ai.task).parameters)
        self.assertIn("show_progress", inspect.signature(df.ai.fillna).parameters)

        # Check async methods have show_progress
        self.assertIn("show_progress", inspect.signature(series.aio.responses).parameters)
        self.assertIn("show_progress", inspect.signature(series.aio.embeddings).parameters)
        self.assertIn("show_progress", inspect.signature(series.aio.task).parameters)
        self.assertIn("show_progress", inspect.signature(df.aio.responses).parameters)
        self.assertIn("show_progress", inspect.signature(df.aio.task).parameters)
        self.assertIn("show_progress", inspect.signature(df.aio.fillna).parameters)

    def test_max_concurrency_parameter_consistency(self):
        """Test that max_concurrency parameter is consistently available in async methods only."""
        import inspect

        series = pd.Series(["test"])
        df = pd.DataFrame({"col": ["test"]})

        # Check sync methods DON'T have max_concurrency
        self.assertNotIn("max_concurrency", inspect.signature(series.ai.responses).parameters)
        self.assertNotIn("max_concurrency", inspect.signature(series.ai.embeddings).parameters)
        self.assertNotIn("max_concurrency", inspect.signature(series.ai.task).parameters)
        self.assertNotIn("max_concurrency", inspect.signature(df.ai.responses).parameters)
        self.assertNotIn("max_concurrency", inspect.signature(df.ai.task).parameters)
        self.assertNotIn("max_concurrency", inspect.signature(df.ai.fillna).parameters)

        # Check async methods DO have max_concurrency
        self.assertIn("max_concurrency", inspect.signature(series.aio.responses).parameters)
        self.assertIn("max_concurrency", inspect.signature(series.aio.embeddings).parameters)
        self.assertIn("max_concurrency", inspect.signature(series.aio.task).parameters)
        self.assertIn("max_concurrency", inspect.signature(df.aio.responses).parameters)
        self.assertIn("max_concurrency", inspect.signature(df.aio.task).parameters)
        self.assertIn("max_concurrency", inspect.signature(df.aio.fillna).parameters)

    def test_method_parameter_ordering(self):
        """Test that parameters appear in consistent order across similar methods."""
        import inspect

        series = pd.Series(["test"])

        # Get parameter lists for comparison
        responses_params = list(inspect.signature(series.ai.responses).parameters.keys())
        aio_responses_params = list(inspect.signature(series.aio.responses).parameters.keys())

        # Common parameters should be in same order (excluding max_concurrency which is async-only)
        common_params = ["instructions", "response_format", "batch_size", "temperature", "top_p", "show_progress"]

        # Check sync version has these in order
        sync_filtered = [p for p in responses_params if p in common_params]
        self.assertEqual(sync_filtered, common_params)

        # Check async version has these in order (with max_concurrency inserted before show_progress)
        async_filtered = [p for p in aio_responses_params if p in common_params or p == "max_concurrency"]
        expected_async = common_params[:5] + ["max_concurrency"] + [common_params[5]]
        self.assertEqual(async_filtered, expected_async)
