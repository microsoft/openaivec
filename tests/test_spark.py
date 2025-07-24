import os
from typing import List
from unittest import TestCase

from pydantic import BaseModel
from pyspark.sql.session import SparkSession
from pyspark.sql.types import ArrayType, FloatType, IntegerType, StringType, StructField, StructType

from openaivec.spark import (
    EmbeddingsUDFBuilder,
    ResponsesUDFBuilder,
    _pydantic_to_spark_schema,
    count_tokens_udf,
    similarity_udf,
)
from openaivec.task import nlp


class TestUDFBuilder(TestCase):
    def setUp(self):
        self.responses = ResponsesUDFBuilder.of_openai(
            api_key=os.environ.get("OPENAI_API_KEY"),
            model_name="gpt-4.1-nano",
        )
        self.embeddings = EmbeddingsUDFBuilder.of_openai(
            api_key=os.environ.get("OPENAI_API_KEY"),
            model_name="text-embedding-3-small",
        )
        self.spark: SparkSession = SparkSession.builder \
            .appName("TestSparkUDF") \
            .master("local[*]") \
            .config("spark.driver.memory", "1g") \
            .config("spark.executor.memory", "1g") \
            .config("spark.sql.adaptive.enabled", "false") \
            .getOrCreate()
        self.spark.sparkContext.setLogLevel("INFO")

    def tearDown(self):
        if self.spark:
            self.spark.stop()

    def test_responses(self):
        self.spark.udf.register(
            "repeat",
            self.responses.build("Repeat twice input string."),
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

    def test_responses_structured(self):
        class Fruit(BaseModel):
            name: str
            color: str
            taste: str

        self.spark.udf.register(
            "fruit",
            self.responses.build(
                instructions="return the color and taste of given fruit",
                response_format=Fruit,
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

    def test_embeddings(self):
        self.spark.udf.register(
            "embed",
            self.embeddings.build(batch_size=8),
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

    def test_task_sentiment_analysis(self):
        # Test using the build_from_task method with predefined tasks
        self.spark.udf.register(
            "analyze_sentiment",
            self.responses.build_from_task(task=nlp.SENTIMENT_ANALYSIS),
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

    def test_responses_build_from_task_method(self):
        """Test the build_from_task method in ResponsesUDFBuilder."""
        # Test that build_from_task works with predefined tasks
        self.spark.udf.register(
            "analyze_sentiment_new",
            self.responses.build_from_task(task=nlp.SENTIMENT_ANALYSIS),
        )
        
        text_data = [("This is a great product!",)]
        dummy_df = self.spark.createDataFrame(text_data, ["text"])
        dummy_df.createOrReplaceTempView("test_reviews")

        df = self.spark.sql(
            """
            with t as (SELECT analyze_sentiment_new(text) as sentiment from test_reviews)
            select sentiment.sentiment, sentiment.confidence from t
            """
        )
        df_pandas = df.toPandas()
        assert df_pandas.shape == (1, 2)



class TestMappingFunctions(TestCase):
    def test_pydantic_to_spark_schema(self):
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


class TestCountTokensUDF(TestCase):
    def setUp(self):
        self.spark: SparkSession = SparkSession.builder.getOrCreate()
        self.spark.sparkContext.setLogLevel("INFO")
        self.spark.udf.register(
            "count_tokens",
            count_tokens_udf("gpt-4o"),
        )

    def test_count_token(self):
        sentences = [
            ("How many tokens in this sentence?",),
            ("Understanding token counts helps optimize language model inputs",),
            ("Tokenization is a crucial step in natural language processing tasks",),
        ]
        dummy_df = self.spark.createDataFrame(sentences, ["sentence"])
        dummy_df.createOrReplaceTempView("sentences")

        self.spark.sql(
            """
            SELECT sentence, count_tokens(sentence) as token_count from sentences
            """
        ).show(truncate=False)


class TestSimilarityUDF(TestCase):
    def setUp(self):
        self.spark: SparkSession = SparkSession.builder.getOrCreate()
        self.spark.sparkContext.setLogLevel("INFO")
        self.spark.udf.register("similarity", similarity_udf())

    def test_similarity(self):
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
        result_df.show(truncate=False)
        df_pandas = result_df.toPandas()
        assert df_pandas.shape == (3, 2)
