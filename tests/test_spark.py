import os

import pytest
from pydantic import BaseModel
from pyspark.sql.types import ArrayType, FloatType, IntegerType, StringType, StructField, StructType

from openaivec._model import PreparedTask
from openaivec._provider import set_default_registrations
from openaivec.spark import (
    _pydantic_to_spark_schema,
    count_tokens_udf,
    embeddings_udf,
    infer_schema,
    parse_udf,
    responses_udf,
    setup,
    setup_azure,
    similarity_udf,
    split_to_chunks_udf,
    task_udf,
)
from openaivec.task import nlp


@pytest.mark.spark
@pytest.mark.requires_api
class TestSparkUDFs:
    """Test all Spark UDF functions."""

    @pytest.fixture(autouse=True)
    def setup_spark_openaivec(self, spark_session, responses_model_name, embeddings_model_name):
        """Setup Spark session with openaivec configuration."""
        self.spark = spark_session
        set_default_registrations()
        api_key = os.environ.get("OPENAI_API_KEY")
        assert api_key is not None
        setup(
            spark=self.spark,
            api_key=api_key,
            responses_model_name=responses_model_name,
            embeddings_model_name=embeddings_model_name,
        )
        yield

    @pytest.mark.parametrize("test_size", [5, 10])
    def test_responses_udf_string_format(self, test_size):
        """Test responses_udf with string response format."""
        self.spark.udf.register(
            "repeat",
            responses_udf("Repeat twice input string."),
        )
        dummy_df = self.spark.range(test_size)
        dummy_df.createOrReplaceTempView("dummy")

        df = self.spark.sql(
            """
            SELECT id, repeat(cast(id as STRING)) as v from dummy
            """
        )

        df_pandas = df.toPandas()
        assert df_pandas.shape == (test_size, 2)

    def test_responses_udf_structured_format(self, fruit_model):
        """Test responses_udf with Pydantic BaseModel response format."""
        self.spark.udf.register(
            "fruit",
            responses_udf(
                instructions="return the color and taste of given fruit",
                response_format=fruit_model,
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
            task_udf(task=nlp.sentiment_analysis()),
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
        )

        self.spark.udf.register(
            "repeat_text",
            task_udf(task=simple_task, temperature=0.0, top_p=1.0),
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
        )

        self.spark.udf.register(
            "analyze_text",
            task_udf(task=structured_task, temperature=0.0, top_p=1.0),
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

    @pytest.mark.parametrize("batch_size", [4, 8])
    def test_embeddings_udf(self, embeddings_model_name, batch_size):
        """Test embeddings_udf functionality."""
        self.spark.udf.register(
            "embed",
            embeddings_udf(model_name=embeddings_model_name, batch_size=batch_size),
        )
        test_size = 10  # Reduced from 31 for faster tests
        dummy_df = self.spark.range(test_size)
        dummy_df.createOrReplaceTempView("dummy")

        df = self.spark.sql(
            """
            SELECT id, embed(cast(id as STRING)) as v from dummy
            """
        )

        df_pandas = df.toPandas()
        assert df_pandas.shape == (test_size, 2)

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

    def test_infer_schema(self):
        """Test infer_schema functionality."""
        # Create a sample table with example data
        sample_data = [
            ("apple is red and sweet",),
            ("banana is yellow and tropical",),
            ("cherry is small and tart",),
        ]
        dummy_df = self.spark.createDataFrame(sample_data, ["description"])
        dummy_df.createOrReplaceTempView("fruits")

        # Infer schema from the example data
        inferred = infer_schema(
            instructions="Extract fruit name, color, and taste from the description",
            example_table_name="fruits",
            example_field_name="description",
            max_examples=3,
        )

        # Verify the inferred schema has the expected structure
        assert inferred.model is not None
        assert inferred.inference_prompt is not None
        assert len(inferred.inference_prompt) > 0

    def test_parse_udf_with_response_format(self):
        """Test parse_udf with explicit response format."""

        class ParsedData(BaseModel):
            product: str
            price: float
            quantity: int

        self.spark.udf.register(
            "parse_product",
            parse_udf(
                instructions="Extract product information from the text",
                response_format=ParsedData,
            ),
        )

        text_data = [
            ("Buy 5 apples for $10.50",),
            ("Purchase 3 bananas at $6.75",),
            ("Get 10 oranges for $15.00",),
        ]
        dummy_df = self.spark.createDataFrame(text_data, ["text"])
        dummy_df.createOrReplaceTempView("products")

        df = self.spark.sql(
            """
            with t as (SELECT parse_product(text) as info from products)
            select info.product, info.price, info.quantity from t
            """
        )
        df_pandas = df.toPandas()
        assert df_pandas.shape == (3, 3)
        # Verify column types
        assert "product" in df_pandas.columns
        assert "price" in df_pandas.columns
        assert "quantity" in df_pandas.columns

    def test_parse_udf_with_example_data(self):
        """Test parse_udf with schema inference from example data."""
        # Create example data for schema inference
        sample_data = [
            ("Meeting scheduled for 2024-01-15 at 10:00 AM with John",),
            ("Conference call on 2024-01-16 at 2:30 PM with Sarah",),
            ("Presentation on 2024-01-17 at 9:00 AM with the team",),
        ]
        dummy_df = self.spark.createDataFrame(sample_data, ["event_text"])
        dummy_df.createOrReplaceTempView("events")

        # Create UDF with schema inference
        self.spark.udf.register(
            "parse_event",
            parse_udf(
                instructions="Extract date, time, and participants from the event description",
                example_table_name="events",
                example_field_name="event_text",
                max_examples=3,
            ),
        )

        # Test with new data
        test_data = [
            ("Workshop on 2024-01-20 at 11:00 AM with Alice",),
            ("Training session on 2024-01-21 at 3:00 PM with Bob",),
        ]
        test_df = self.spark.createDataFrame(test_data, ["text"])
        test_df.createOrReplaceTempView("test_events")

        df = self.spark.sql(
            """
            SELECT parse_event(text) as parsed from test_events
            """
        )
        df_pandas = df.toPandas()
        assert df_pandas.shape[0] == 2
        # Parsed column should contain structured data
        assert "parsed" in df_pandas.columns

    def test_parse_udf_string_response(self):
        """Test parse_udf with string response format."""
        self.spark.udf.register(
            "summarize",
            parse_udf(
                instructions="Summarize the text in one sentence",
                response_format=str,
            ),
        )

        text_data = [
            ("The quick brown fox jumps over the lazy dog multiple times throughout the day",),
            ("Scientists discovered a new species of butterfly in the Amazon rainforest",),
        ]
        dummy_df = self.spark.createDataFrame(text_data, ["text"])
        dummy_df.createOrReplaceTempView("texts")

        df = self.spark.sql(
            """
            SELECT text, summarize(text) as summary from texts
            """
        )
        df_pandas = df.toPandas()
        assert df_pandas.shape == (2, 2)
        # Verify string column type
        assert df.dtypes[1][1] == "string"


class TestSchemaMapping:
    """Test Pydantic to Spark schema mapping functionality."""

    @pytest.fixture
    def nested_models(self):
        """Fixture providing nested Pydantic models for testing."""

        class InnerModel(BaseModel):
            inner_id: int
            description: str

        class OuterModel(BaseModel):
            id: int
            name: str
            values: list[float]
            inner: InnerModel

        return InnerModel, OuterModel

    def test_pydantic_to_spark_schema(self, nested_models):
        """Test _pydantic_to_spark_schema function with nested models."""
        InnerModel, OuterModel = nested_models
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

        assert schema == expected

    def test_basic_type_mapping(self):
        """Test basic Pydantic type to Spark type mapping."""

        # Test str type
        class StrModel(BaseModel):
            test_field: str

        schema = _pydantic_to_spark_schema(StrModel)
        assert len(schema.fields) == 1
        assert schema.fields[0].dataType == StringType()

        # Test int type
        class IntModel(BaseModel):
            test_field: int

        schema = _pydantic_to_spark_schema(IntModel)
        assert len(schema.fields) == 1
        assert schema.fields[0].dataType == IntegerType()

        # Test float type
        class FloatModel(BaseModel):
            test_field: float

        schema = _pydantic_to_spark_schema(FloatModel)
        assert len(schema.fields) == 1
        assert schema.fields[0].dataType == FloatType()


class TestSparkConfigAndValidation:
    def test_parse_udf_requires_response_format_or_example_source(self):
        with pytest.raises(ValueError, match="Either response_format or example_table_name"):
            parse_udf(instructions="Extract fields")

    @pytest.mark.parametrize(
        "example_table_name,example_field_name",
        [
            ("events", None),
            (None, "body"),
        ],
    )
    def test_parse_udf_requires_both_example_inputs(self, example_table_name, example_field_name):
        with pytest.raises(ValueError, match="Either response_format or example_table_name"):
            parse_udf(
                instructions="Extract fields",
                response_format=None,
                example_table_name=example_table_name,
                example_field_name=example_field_name,
            )

    def test_responses_udf_rejects_unsupported_response_format(self):
        with pytest.raises(ValueError, match="Unsupported response_format"):
            responses_udf(instructions="echo", response_format=dict)  # type: ignore[arg-type]


@pytest.mark.spark
class TestSparkNonApiUdfs:
    def test_setup_azure_sets_spark_and_local_environment(self, spark_session, reset_environment):
        set_default_registrations()

        setup_azure(
            spark=spark_session,
            api_key="azure-key",
            base_url="https://example.services.ai.azure.com/openai/v1/",
            api_version="v1",
            responses_model_name="responses-deployment",
            embeddings_model_name="embeddings-deployment",
        )

        sc_env = spark_session.sparkContext.environment
        assert sc_env["AZURE_OPENAI_API_KEY"] == "azure-key"
        assert sc_env["AZURE_OPENAI_BASE_URL"] == "https://example.services.ai.azure.com/openai/v1/"
        assert sc_env["AZURE_OPENAI_API_VERSION"] == "v1"

        assert os.environ["AZURE_OPENAI_API_KEY"] == "azure-key"
        assert os.environ["AZURE_OPENAI_BASE_URL"] == "https://example.services.ai.azure.com/openai/v1/"
        assert os.environ["AZURE_OPENAI_API_VERSION"] == "v1"

    def test_setup_azure_without_api_key_clears_azure_key(self, spark_session, reset_environment):
        set_default_registrations()

        spark_session.sparkContext.environment["AZURE_OPENAI_API_KEY"] = "stale-key"
        os.environ["AZURE_OPENAI_API_KEY"] = "stale-key"

        setup_azure(
            spark=spark_session,
            base_url="https://example.services.ai.azure.com/openai/v1/",
            api_version="v1",
            responses_model_name="responses-deployment",
            embeddings_model_name="embeddings-deployment",
        )

        sc_env = spark_session.sparkContext.environment
        assert "AZURE_OPENAI_API_KEY" not in sc_env
        assert sc_env["AZURE_OPENAI_BASE_URL"] == "https://example.services.ai.azure.com/openai/v1/"
        assert sc_env["AZURE_OPENAI_API_VERSION"] == "v1"

        assert "AZURE_OPENAI_API_KEY" not in os.environ
        assert os.environ["AZURE_OPENAI_BASE_URL"] == "https://example.services.ai.azure.com/openai/v1/"
        assert os.environ["AZURE_OPENAI_API_VERSION"] == "v1"

    def test_setup_azure_requires_base_url(self, spark_session):
        with pytest.raises(ValueError, match="base_url is required"):
            setup_azure(spark=spark_session, api_key="azure-key")

    def test_split_to_chunks_udf(self, spark_session):
        spark_session.udf.register("split_chunks", split_to_chunks_udf(max_tokens=8, sep=[".", " "]))

        sample = spark_session.createDataFrame(
            [("alpha beta gamma. delta epsilon zeta",), (None,)],
            ["body"],
        )
        sample.createOrReplaceTempView("chunk_input")

        out = spark_session.sql("SELECT split_chunks(body) AS chunks FROM chunk_input").toPandas()

        assert isinstance(out.iloc[0]["chunks"], list)
        assert len(out.iloc[0]["chunks"]) >= 1
        assert out.iloc[1]["chunks"] == []
