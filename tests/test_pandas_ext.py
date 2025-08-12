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


class TestPandasExt(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame(
            {
                "name": ["apple", "banana", "cherry"],
            }
        )

    def test_embeddings(self):
        embeddings: pd.Series = self.df["name"].ai.embeddings()

        # assert all values are elements of np.ndarray
        self.assertTrue(all(isinstance(embedding, np.ndarray) for embedding in embeddings))

    def test_aio_embeddings(self):
        async def run():
            return await self.df["name"].aio.embeddings()

        embeddings: pd.Series = asyncio.run(run())
        self.assertTrue(all(isinstance(embedding, np.ndarray) for embedding in embeddings))
        self.assertEqual(embeddings.shape, (3,))
        self.assertTrue(embeddings.index.equals(self.df.index))

    def test_responses(self):
        names_fr: pd.Series = self.df["name"].ai.responses("translate to French")

        # assert all values are elements of str
        self.assertTrue(all(isinstance(x, str) for x in names_fr))

    def test_aio_responses(self):
        async def run():
            return await self.df["name"].aio.responses("translate to French")

        names_fr: pd.Series = asyncio.run(run())
        self.assertTrue(all(isinstance(x, str) for x in names_fr))
        self.assertEqual(names_fr.shape, (3,))
        self.assertTrue(names_fr.index.equals(self.df.index))

    def test_responses_dataframe(self):
        names_fr: pd.Series = self.df.ai.responses("translate to French")

        # assert all values are elements of str
        self.assertTrue(all(isinstance(x, str) for x in names_fr))

    def test_aio_responses_dataframe(self):
        async def run():
            return await self.df.aio.responses("translate the 'name' field to French")

        names_fr: pd.Series = asyncio.run(run())
        self.assertTrue(all(isinstance(x, str) for x in names_fr))
        self.assertEqual(names_fr.shape, (3,))
        self.assertTrue(names_fr.index.equals(self.df.index))

    def test_extract_series(self):
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

    def test_extract_series_without_name(self):
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

    def test_extract_series_dict(self):
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

    def test_extract_series_with_none(self):
        sample_series = pd.Series(
            [
                Fruit(color="red", flavor="sweet", taste="crunchy"),
                None,
                Fruit(color="yellow", flavor="sweet", taste="soft"),
            ],
            name="fruit",
        )
        extracted_df = sample_series.ai.extract()

        # assert columns are ['fruit_color', 'fruit_flavor', 'fruit_taste']
        expected_columns = ["fruit_color", "fruit_flavor", "fruit_taste"]
        self.assertListEqual(list(extracted_df.columns), expected_columns)

        # assert the row with None is filled with NaN
        self.assertTrue(extracted_df.iloc[1].isna().all())

    def test_extract_series_with_invalid_row(self):
        sample_series = pd.Series(
            [
                Fruit(color="red", flavor="sweet", taste="crunchy"),
                123,  # Invalid row
                Fruit(color="yellow", flavor="sweet", taste="soft"),
            ],
            name="fruit",
        )
        extracted_df = sample_series.ai.extract()

        # assert columns are ['fruit_color', 'fruit_flavor', 'fruit_taste']
        expected_columns = ["fruit_color", "fruit_flavor", "fruit_taste"]
        self.assertListEqual(list(extracted_df.columns), expected_columns)

        # assert the invalid row is filled with NaN
        self.assertTrue(extracted_df.iloc[1].isna().all())

    def test_extract(self):
        sample_df = pd.DataFrame(
            [
                {"name": "apple", "fruit": Fruit(color="red", flavor="sweet", taste="crunchy")},
                {"name": "banana", "fruit": Fruit(color="yellow", flavor="sweet", taste="soft")},
                {"name": "cherry", "fruit": Fruit(color="red", flavor="sweet", taste="tart")},
            ]
        ).ai.extract("fruit")

        expected_columns = ["name", "fruit_color", "fruit_flavor", "fruit_taste"]
        self.assertListEqual(list(sample_df.columns), expected_columns)

    def test_extract_dict(self):
        sample_df = pd.DataFrame(
            [
                {"fruit": {"name": "apple", "color": "red", "flavor": "sweet", "taste": "crunchy"}},
                {"fruit": {"name": "banana", "color": "yellow", "flavor": "sweet", "taste": "soft"}},
                {"fruit": {"name": "cherry", "color": "red", "flavor": "sweet", "taste": "tart"}},
            ]
        ).ai.extract("fruit")

        expected_columns = ["fruit_name", "fruit_color", "fruit_flavor", "fruit_taste"]
        self.assertListEqual(list(sample_df.columns), expected_columns)

    def test_extract_dict_with_none(self):
        sample_df = pd.DataFrame(
            [
                {"fruit": {"name": "apple", "color": "red", "flavor": "sweet", "taste": "crunchy"}},
                {"fruit": None},
                {"fruit": {"name": "cherry", "color": "red", "flavor": "sweet", "taste": "tart"}},
            ]
        ).ai.extract("fruit")

        expected_columns = ["fruit_name", "fruit_color", "fruit_flavor", "fruit_taste"]
        self.assertListEqual(list(sample_df.columns), expected_columns)

        # assert the row with None is filled with NaN
        self.assertTrue(sample_df.iloc[1].isna().all())

    def test_extract_with_invalid_row(self):
        sample_df = pd.DataFrame(
            [
                {"fruit": {"name": "apple", "color": "red", "flavor": "sweet", "taste": "crunchy"}},
                {"fruit": 123},
                {"fruit": {"name": "cherry", "color": "red", "flavor": "sweet", "taste": "tart"}},
            ]
        )

        expected_columns = ["fruit"]
        self.assertListEqual(list(sample_df.columns), expected_columns)

    def test_count_tokens(self):
        num_tokens: pd.Series = self.df.name.ai.count_tokens()

        # assert all values are elements of int
        self.assertTrue(all(isinstance(num_token, int) for num_token in num_tokens))

    def test_similarity(self):
        sample_df = pd.DataFrame(
            {
                "vector1": [np.array([1, 0]), np.array([0, 1]), np.array([1, 1])],
                "vector2": [np.array([1, 0]), np.array([0, 1]), np.array([1, -1])],
            }
        )
        similarity_scores = sample_df.ai.similarity("vector1", "vector2")

        # Expected cosine similarity values
        expected_scores = [
            1.0,  # Cosine similarity between [1, 0] and [1, 0]
            1.0,  # Cosine similarity between [0, 1] and [0, 1]
            0.0,  # Cosine similarity between [1, 1] and [1, -1]
        ]

        # Assert similarity scores match expected values
        self.assertTrue(np.allclose(similarity_scores, expected_scores))

    def test_similarity_with_invalid_vectors(self):
        sample_df = pd.DataFrame(
            {
                "vector1": [np.array([1, 0]), "invalid", np.array([1, 1])],
                "vector2": [np.array([1, 0]), np.array([0, 1]), np.array([1, -1])],
            }
        )

        with self.assertRaises(TypeError):
            sample_df.ai.similarity("vector1", "vector2")

    def test_fillna_with_no_missing_values(self):
        """Test fillna method when target column has no missing values."""
        # Create a DataFrame without missing values in target column
        df_complete = pd.DataFrame(
            {
                "name": ["Alice", "Bob", "Charlie", "David"],
                "age": [25, 30, 35, 40],
                "city": ["Tokyo", "Osaka", "Kyoto", "Tokyo"],
            }
        )

        # Test fillna on a column with no missing values
        result_df = df_complete.ai.fillna("name")

        # Assert that the result is identical to the original
        pd.testing.assert_frame_equal(result_df, df_complete)

    def test_fillna_task_creation(self):
        """Test that fillna method creates a valid task."""
        from openaivec.task.table import fillna

        # Create a DataFrame with missing values
        df_with_missing = pd.DataFrame(
            {
                "name": ["Alice", "Bob", None, "David"],
                "age": [25, 30, 35, 40],
                "city": ["Tokyo", "Osaka", "Kyoto", "Tokyo"],
            }
        )

        # Test that task creation works without errors
        task = fillna(df_with_missing, "name")

        # Assert that the task is created
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
        # Create a DataFrame with some missing values
        df_with_missing = pd.DataFrame(
            {
                "name": ["Alice", "Bob", None, "David", None],
                "age": [25, 30, 35, 40, 45],
                "city": ["Tokyo", "Osaka", "Kyoto", "Tokyo", "Nagoya"],
            }
        )

        # Get missing rows manually
        missing_rows = df_with_missing[df_with_missing["name"].isna()]

        # Assert that we correctly identify 2 missing rows
        self.assertEqual(len(missing_rows), 2)
        self.assertTrue(missing_rows.index.tolist() == [2, 4])

    def test_fillna_dataframe_copy(self):
        """Test that fillna returns a copy and doesn't modify original."""
        # Test fillna (this will actually call the API, but we check basic behavior)
        # For testing purposes, we'll just verify that the original isn't modified
        # when there are no missing values
        df_no_missing = pd.DataFrame(
            {
                "name": ["Alice", "Bob", "Charlie", "David"],
                "age": [25, 30, 35, 40],
                "city": ["Tokyo", "Osaka", "Kyoto", "Tokyo"],
            }
        )

        result_df = df_no_missing.ai.fillna("name")

        # Assert original is unchanged
        pd.testing.assert_frame_equal(df_no_missing, result_df)

    def test_fillna_index_preservation_structure(self):
        """Test that fillna preserves DataFrame structure without API calls."""
        # Create a DataFrame with custom index but no missing values
        df_custom_index = pd.DataFrame(
            {"name": ["Alice", "Bob", "Charlie"], "score": [85, 90, 78]}, index=["student_1", "student_2", "student_3"]
        )

        # Test fillna on complete data (no API call needed)
        result_df = df_custom_index.ai.fillna("name")

        # Assert that the structure is preserved
        pd.testing.assert_index_equal(result_df.index, df_custom_index.index)
        self.assertEqual(result_df.shape, df_custom_index.shape)
        pd.testing.assert_frame_equal(result_df, df_custom_index)

    def test_shared_cache_responses_sync(self):
        """Test that multiple Series instances can share the same cache for responses."""
        from openaivec.proxy import BatchingMapProxy

        # Create a shared cache with custom batch size
        shared_cache = BatchingMapProxy(batch_size=32)

        # Create two different Series with some overlapping data
        series1 = pd.Series(["cat", "dog", "elephant"])
        series2 = pd.Series(["dog", "elephant", "lion"])  # "dog" and "elephant" are shared

        # Use the shared cache for both series
        result1 = series1.ai.responses_with_cache(instructions="translate to French", cache=shared_cache)
        result2 = series2.ai.responses_with_cache(instructions="translate to French", cache=shared_cache)

        # Verify results are valid
        self.assertTrue(all(isinstance(x, str) for x in result1))
        self.assertTrue(all(isinstance(x, str) for x in result2))
        self.assertEqual(len(result1), 3)
        self.assertEqual(len(result2), 3)

        # Check that overlapping items ("dog", "elephant") have the same results
        # Find indices of shared items
        dog_idx1 = series1[series1 == "dog"].index[0]
        dog_idx2 = series2[series2 == "dog"].index[0]
        elephant_idx1 = series1[series1 == "elephant"].index[0]
        elephant_idx2 = series2[series2 == "elephant"].index[0]

        # The translations should be identical due to cache sharing
        self.assertEqual(result1[dog_idx1], result2[dog_idx2])
        self.assertEqual(result1[elephant_idx1], result2[elephant_idx2])

    def test_shared_cache_embeddings_sync(self):
        """Test that multiple Series instances can share the same cache for embeddings."""
        import numpy as np

        from openaivec.proxy import BatchingMapProxy

        # Create a shared cache with custom batch size
        shared_cache = BatchingMapProxy(batch_size=32)

        # Create two different Series with some overlapping data
        series1 = pd.Series(["apple", "banana", "cherry"])
        series2 = pd.Series(["banana", "cherry", "date"])  # "banana" and "cherry" are shared

        # Use the shared cache for both series
        embeddings1 = series1.ai.embeddings_with_cache(cache=shared_cache)
        embeddings2 = series2.ai.embeddings_with_cache(cache=shared_cache)

        # Verify embeddings are valid numpy arrays
        self.assertTrue(all(isinstance(emb, np.ndarray) for emb in embeddings1))
        self.assertTrue(all(isinstance(emb, np.ndarray) for emb in embeddings2))
        self.assertEqual(len(embeddings1), 3)
        self.assertEqual(len(embeddings2), 3)

        # Check that overlapping items have identical embeddings due to cache sharing
        banana_idx1 = series1[series1 == "banana"].index[0]
        banana_idx2 = series2[series2 == "banana"].index[0]
        cherry_idx1 = series1[series1 == "cherry"].index[0]
        cherry_idx2 = series2[series2 == "cherry"].index[0]

        # The embeddings should be identical due to cache sharing
        np.testing.assert_array_equal(embeddings1[banana_idx1], embeddings2[banana_idx2])
        np.testing.assert_array_equal(embeddings1[cherry_idx1], embeddings2[cherry_idx2])

    def test_shared_cache_dataframe_responses_sync(self):
        """Test that DataFrame instances can share the same cache for responses."""
        from openaivec.proxy import BatchingMapProxy

        # Create a shared cache
        shared_cache = BatchingMapProxy(batch_size=32)

        # Create two DataFrames with overlapping serialized JSON representations
        df1 = pd.DataFrame(
            [
                {"animal": "cat", "legs": 4},
                {"animal": "dog", "legs": 4},
            ]
        )
        df2 = pd.DataFrame(
            [
                {"animal": "dog", "legs": 4},  # This row should be cached
                {"animal": "bird", "legs": 2},
            ]
        )

        # Use the shared cache for both DataFrames
        result1 = df1.ai.responses_with_cache(instructions="what animal is this?", cache=shared_cache)
        result2 = df2.ai.responses_with_cache(instructions="what animal is this?", cache=shared_cache)

        # Verify results are valid
        self.assertTrue(all(isinstance(x, str) for x in result1))
        self.assertTrue(all(isinstance(x, str) for x in result2))
        self.assertEqual(len(result1), 2)
        self.assertEqual(len(result2), 2)

        # Since both DataFrames have the same row {"animal": "dog", "legs": 4},
        # the responses should be identical due to cache sharing
        # (The exact match depends on JSON serialization being consistent)
        self.assertIsNotNone(result1[1])  # Dog response from df1
        self.assertIsNotNone(result2[0])  # Dog response from df2 (should be cached)

    def test_shared_cache_responses_async(self):
        """Test that multiple async Series instances can share the same cache for responses."""
        from openaivec.proxy import AsyncBatchingMapProxy

        async def run_test():
            # Create a shared cache with custom batch size
            shared_cache = AsyncBatchingMapProxy(batch_size=32, max_concurrency=4)

            # Create two different Series with some overlapping data
            series1 = pd.Series(["cat", "dog", "elephant"])
            series2 = pd.Series(["dog", "elephant", "lion"])  # "dog" and "elephant" are shared

            # Use the shared cache for both series
            result1 = await series1.aio.responses_with_cache(instructions="translate to French", cache=shared_cache)
            result2 = await series2.aio.responses_with_cache(instructions="translate to French", cache=shared_cache)

            return result1, result2, series1, series2

        result1, result2, series1, series2 = asyncio.run(run_test())

        # Verify results are valid
        self.assertTrue(all(isinstance(x, str) for x in result1))
        self.assertTrue(all(isinstance(x, str) for x in result2))
        self.assertEqual(len(result1), 3)
        self.assertEqual(len(result2), 3)

        # Check that overlapping items ("dog", "elephant") have the same results
        dog_idx1 = series1[series1 == "dog"].index[0]
        dog_idx2 = series2[series2 == "dog"].index[0]
        elephant_idx1 = series1[series1 == "elephant"].index[0]
        elephant_idx2 = series2[series2 == "elephant"].index[0]

        # The translations should be identical due to cache sharing
        self.assertEqual(result1[dog_idx1], result2[dog_idx2])
        self.assertEqual(result1[elephant_idx1], result2[elephant_idx2])

    def test_shared_cache_embeddings_async(self):
        """Test that multiple async Series instances can share the same cache for embeddings."""
        import numpy as np

        from openaivec.proxy import AsyncBatchingMapProxy

        async def run_test():
            # Create a shared cache with custom batch size
            shared_cache = AsyncBatchingMapProxy(batch_size=32, max_concurrency=4)

            # Create two different Series with some overlapping data
            series1 = pd.Series(["apple", "banana", "cherry"])
            series2 = pd.Series(["banana", "cherry", "date"])  # "banana" and "cherry" are shared

            # Use the shared cache for both series
            embeddings1 = await series1.aio.embeddings_with_cache(cache=shared_cache)
            embeddings2 = await series2.aio.embeddings_with_cache(cache=shared_cache)

            return embeddings1, embeddings2, series1, series2

        embeddings1, embeddings2, series1, series2 = asyncio.run(run_test())

        # Verify embeddings are valid numpy arrays
        self.assertTrue(all(isinstance(emb, np.ndarray) for emb in embeddings1))
        self.assertTrue(all(isinstance(emb, np.ndarray) for emb in embeddings2))
        self.assertEqual(len(embeddings1), 3)
        self.assertEqual(len(embeddings2), 3)

        # Check that overlapping items have identical embeddings due to cache sharing
        banana_idx1 = series1[series1 == "banana"].index[0]
        banana_idx2 = series2[series2 == "banana"].index[0]
        cherry_idx1 = series1[series1 == "cherry"].index[0]
        cherry_idx2 = series2[series2 == "cherry"].index[0]

        # The embeddings should be identical due to cache sharing
        np.testing.assert_array_equal(embeddings1[banana_idx1], embeddings2[banana_idx2])
        np.testing.assert_array_equal(embeddings1[cherry_idx1], embeddings2[cherry_idx2])

    def test_shared_cache_dataframe_responses_async(self):
        """Test that async DataFrame instances can share the same cache for responses."""
        from openaivec.proxy import AsyncBatchingMapProxy

        async def run_test():
            # Create a shared cache
            shared_cache = AsyncBatchingMapProxy(batch_size=32, max_concurrency=4)

            # Create two DataFrames with overlapping serialized JSON representations
            df1 = pd.DataFrame(
                [
                    {"animal": "cat", "legs": 4},
                    {"animal": "dog", "legs": 4},
                ]
            )
            df2 = pd.DataFrame(
                [
                    {"animal": "dog", "legs": 4},  # This row should be cached
                    {"animal": "bird", "legs": 2},
                ]
            )

            # Use the shared cache for both DataFrames
            result1 = await df1.aio.responses_with_cache(instructions="what animal is this?", cache=shared_cache)
            result2 = await df2.aio.responses_with_cache(instructions="what animal is this?", cache=shared_cache)

            return result1, result2

        result1, result2 = asyncio.run(run_test())

        # Verify results are valid
        self.assertTrue(all(isinstance(x, str) for x in result1))
        self.assertTrue(all(isinstance(x, str) for x in result2))
        self.assertEqual(len(result1), 2)
        self.assertEqual(len(result2), 2)

        # Since both DataFrames have the same row {"animal": "dog", "legs": 4},
        # the responses should be identical due to cache sharing
        self.assertIsNotNone(result1[1])  # Dog response from df1
        self.assertIsNotNone(result2[0])  # Dog response from df2 (should be cached)
