from typing import List
from unittest import TestCase

from pydantic import BaseModel
from pyspark.sql.session import SparkSession
from pyspark.sql.types import ArrayType, FloatType, IntegerType, StringType, StructField, StructType

from openaivec._model import PreparedTask
from openaivec.spark import (
    _pydantic_to_spark_schema,
    count_tokens_udf,
    embeddings_udf,
    responses_udf,
    similarity_udf,
    task_udf,
)
from openaivec.task import nlp


class TestSparkUDFs(TestCase):
    """Test all Spark UDF functions."""

    def setUp(self):
        self.spark: SparkSession = (
            SparkSession.builder.appName("TestSparkUDF")
            .master("local[*]")
            .config("spark.driver.memory", "1g")
            .config("spark.executor.memory", "1g")
            .config("spark.sql.adaptive.enabled", "false")
            .getOrCreate()
        )
        self.spark.sparkContext.setLogLevel("INFO")

    def tearDown(self):
        if self.spark:
            self.spark.stop()

    def test_responses_udf_string_format(self):
        """Test responses_udf with string response format."""
        self.spark.udf.register(
            "repeat",
            responses_udf("Repeat twice input string.", model_name="gpt-4.1-nano"),
        )
        dummy_df = self.spark.range(31)
        dummy_df.createOrReplaceTempView("dummy")

        df = self.spark.sql(
            """
            SELECT id, repeat(cast(id as STRING)) as v from dummy
            """
        )

        df_pandas = df.toPandas()
        assert df_pandas.shape == (31, 2)

    def test_responses_udf_structured_format(self):
        """Test responses_udf with Pydantic BaseModel response format."""

        class Fruit(BaseModel):
            name: str
            color: str
            taste: str

        self.spark.udf.register(
            "fruit",
            responses_udf(
                instructions="return the color and taste of given fruit",
                response_format=Fruit,
                model_name="gpt-4.1-nano",
            ),
        )

        fruit_data = [("apple",), ("banana",), ("cherry",)]
        dummy_df = self.spark.createDataFrame(fruit_data, ["name"])
        dummy_df.createOrReplaceTempView("dummy")

        df = self.spark.sql(
            """
            with t as (SELECT fruit(name) as info from dummy)
            select info.name, info.color, info.taste from t
            """
        )
        df_pandas = df.toPandas()
        assert df_pandas.shape == (3, 3)

    def test_task_udf_basemodel(self):
        """Test task_udf with predefined BaseModel task."""
        self.spark.udf.register(
            "analyze_sentiment",
            task_udf(task=nlp.SENTIMENT_ANALYSIS, model_name="gpt-4.1-nano"),
        )

        text_data = [
            ("I love this product!",),
            ("This is terrible and disappointing.",),
            ("It's okay, nothing special.",),
        ]
        dummy_df = self.spark.createDataFrame(text_data, ["text"])
        dummy_df.createOrReplaceTempView("reviews")

        df = self.spark.sql(
            """
            with t as (SELECT analyze_sentiment(text) as sentiment from reviews)
            select sentiment.sentiment, sentiment.confidence, sentiment.polarity from t
            """
        )
        df_pandas = df.toPandas()
        assert df_pandas.shape == (3, 3)

    def test_task_udf_string_format(self):
        """Test task_udf with string response format."""
        simple_task = PreparedTask(
            instructions="Repeat the input text twice, separated by a space.",
            response_format=str,
            temperature=0.0,
            top_p=1.0,
        )

        self.spark.udf.register(
            "repeat_text",
            task_udf(task=simple_task, model_name="gpt-4.1-nano"),
        )

        text_data = [("hello",), ("world",), ("test",)]
        dummy_df = self.spark.createDataFrame(text_data, ["text"])
        dummy_df.createOrReplaceTempView("simple_text")

        df = self.spark.sql(
            """
            SELECT text, repeat_text(text) as repeated from simple_text
            """
        )
        df_pandas = df.toPandas()
        assert df_pandas.shape == (3, 2)
        # Verify string column type
        assert df.dtypes[1][1] == "string"

    def test_task_udf_custom_basemodel(self):
        """Test task_udf with custom BaseModel response format."""

        class SimpleResponse(BaseModel):
            original: str
            length: int

        structured_task = PreparedTask(
            instructions="Analyze the text and return the original text and its length.",
            response_format=SimpleResponse,
            temperature=0.0,
            top_p=1.0,
        )

        self.spark.udf.register(
            "analyze_text",
            task_udf(task=structured_task, model_name="gpt-4.1-nano"),
        )

        text_data = [("hello",), ("world",), ("testing",)]
        dummy_df = self.spark.createDataFrame(text_data, ["text"])
        dummy_df.createOrReplaceTempView("struct_text")

        df = self.spark.sql(
            """
            with t as (SELECT analyze_text(text) as result from struct_text)
            select result.original, result.length from t
            """
        )
        df_pandas = df.toPandas()
        assert df_pandas.shape == (3, 2)

    def test_embeddings_udf(self):
        """Test embeddings_udf functionality."""
        self.spark.udf.register(
            "embed",
            embeddings_udf(model_name="text-embedding-3-small", batch_size=8),
        )
        dummy_df = self.spark.range(31)
        dummy_df.createOrReplaceTempView("dummy")

        df = self.spark.sql(
            """
            SELECT id, embed(cast(id as STRING)) as v from dummy
            """
        )

        df_pandas = df.toPandas()
        assert df_pandas.shape == (31, 2)

    def test_count_tokens_udf(self):
        """Test count_tokens_udf functionality."""
        self.spark.udf.register(
            "count_tokens",
            count_tokens_udf(),
        )

        sentences = [
            ("How many tokens in this sentence?",),
            ("Understanding token counts helps optimize language model inputs",),
            ("Tokenization is a crucial step in natural language processing tasks",),
        ]
        dummy_df = self.spark.createDataFrame(sentences, ["sentence"])
        dummy_df.createOrReplaceTempView("sentences")

        result_df = self.spark.sql(
            """
            SELECT sentence, count_tokens(sentence) as token_count from sentences
            """
        )
        df_pandas = result_df.toPandas()
        assert df_pandas.shape == (3, 2)

    def test_similarity_udf(self):
        """Test similarity_udf functionality."""
        self.spark.udf.register("similarity", similarity_udf())

        df = self.spark.createDataFrame(
            [
                (1, [0.1, 0.2, 0.3]),
                (2, [0.4, 0.5, 0.6]),
                (3, [0.7, 0.8, 0.9]),
            ],
            ["id", "vector"],
        )
        df.createOrReplaceTempView("vectors")
        result_df = self.spark.sql(
            """
            SELECT id, similarity(vector, vector) as similarity_score
            FROM vectors
            """
        )
        df_pandas = result_df.toPandas()
        assert df_pandas.shape == (3, 2)


class TestSchemaMapping(TestCase):
    """Test Pydantic to Spark schema mapping functionality."""

    def test_pydantic_to_spark_schema(self):
        """Test _pydantic_to_spark_schema function with nested models."""

        class InnerModel(BaseModel):
            inner_id: int
            description: str

        class OuterModel(BaseModel):
            id: int
            name: str
            values: List[float]
            inner: InnerModel

        schema = _pydantic_to_spark_schema(OuterModel)

        expected = StructType(
            [
                StructField("id", IntegerType(), True),
                StructField("name", StringType(), True),
                StructField("values", ArrayType(FloatType(), True), True),
                StructField(
                    "inner",
                    StructType(
                        [StructField("inner_id", IntegerType(), True), StructField("description", StringType(), True)]
                    ),
                    True,
                ),
            ]
        )

        self.assertEqual(schema, expected)
