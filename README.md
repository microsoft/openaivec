# openaivec

**Transform your data analysis with AI-powered text processing at scale.**

**openaivec** enables data analysts to seamlessly integrate OpenAI's language models into their pandas and Spark workflows. Process thousands of text records with natural language instructions, turning unstructured data into actionable insights with just a few lines of code.

## 🚀 Quick Start: From Text to Insights in Seconds

Imagine analyzing 10,000 customer reviews. Instead of manual work, just write:

```python
import pandas as pd
from openaivec import pandas_ext

# Your data
reviews = pd.DataFrame({
    "review": ["Great product, fast delivery!", "Terrible quality, very disappointed", ...]
})

# AI-powered analysis in one line
results = reviews.assign(
    sentiment=lambda df: df.review.ai.responses("Classify sentiment: positive/negative/neutral"),
    issues=lambda df: df.review.ai.responses("Extract main issues or compliments"),
    priority=lambda df: df.review.ai.responses("Priority for follow-up: low/medium/high")
)
```

**Result**: Thousands of reviews classified and analyzed in minutes, not days.

📓 **[Try it yourself →](https://microsoft.github.io/openaivec/examples/pandas/)**

## 💡 Real-World Impact

### Customer Feedback Analysis

```python
# Process 50,000 support tickets automatically
tickets.assign(
    category=lambda df: df.description.ai.responses("Categorize: billing/technical/feature_request"),
    urgency=lambda df: df.description.ai.responses("Urgency level: low/medium/high/critical"),
    solution_type=lambda df: df.description.ai.responses("Best resolution approach")
)
```

### Market Research at Scale

```python
# Analyze multilingual social media data
social_data.assign(
    english_text=lambda df: df.post.ai.responses("Translate to English"),
    brand_mention=lambda df: df.english_text.ai.responses("Extract brand mentions and sentiment"),
    market_trend=lambda df: df.english_text.ai.responses("Identify emerging trends or concerns")
)
```

### Survey Data Transformation

```python
# Convert free-text responses to structured data
from pydantic import BaseModel

class Demographics(BaseModel):
    age_group: str
    location: str
    interests: list[str]

survey_responses.assign(
    structured=lambda df: df.response.ai.responses(
        "Extract demographics as structured data",
        response_format=Demographics
    )
).ai.extract("structured")  # Auto-expands to columns
```

📓 **[See more examples →](https://microsoft.github.io/openaivec/examples/pandas/)**

# Overview

This package provides a vectorized interface for the OpenAI API, enabling you to process multiple inputs with a single
API call instead of sending requests one by one.
This approach helps reduce latency and simplifies your code.

Additionally, it integrates effortlessly with Pandas DataFrames and Apache Spark UDFs, making it easy to incorporate
into your data processing pipelines.

## Features

- Vectorized API requests for processing multiple inputs at once.
- Seamless integration with Pandas DataFrames.
- A UDF builder for Apache Spark.
- Compatibility with multiple OpenAI clients, including Azure OpenAI.

## Key Benefits

- **🚀 Performance**: Vectorized processing handles thousands of records in minutes, not hours
- **💰 Cost Efficiency**: Automatic deduplication significantly reduces API costs on typical datasets
- **🔗 Integration**: Works within existing pandas/Spark workflows without architectural changes
- **📈 Scalability**: Same API scales from exploratory analysis (100s of records) to production systems (millions of records)
- **🎯 Pre-configured Tasks**: Ready-to-use task library with optimized prompts for common use cases
- **🏢 Enterprise Ready**: Microsoft Fabric integration, Apache Spark UDFs, Azure OpenAI compatibility

## Requirements

- Python 3.10 or higher

## Installation

Install the package with:

```bash
pip install openaivec
```

If you want to uninstall the package, you can do so with:

```bash
pip uninstall openaivec
```

## Basic Usage

### Direct API Usage

For maximum control over batch processing:

```python
import os
from openai import OpenAI
from openaivec import BatchResponses

# Initialize the batch client
client = BatchResponses.of(
    client=OpenAI(),
    model_name="gpt-4.1-mini",
    system_message="Please answer only with 'xx family' and do not output anything else.",
    # batch_size defaults to None (automatic optimization)
)

result = client.parse(["panda", "rabbit", "koala"])
print(result)  # Expected output: ['bear family', 'rabbit family', 'koala family']
```

📓 **[Complete tutorial →](https://microsoft.github.io/openaivec/examples/pandas/)**

### Pandas Integration (Recommended)

The easiest way to get started with your DataFrames:

```python
import os
import pandas as pd
from openaivec import pandas_ext

# Authentication Option 1: Environment variables (automatic detection)
# For OpenAI:
os.environ["OPENAI_API_KEY"] = "your-api-key-here"
# Or for Azure OpenAI:
# os.environ["AZURE_OPENAI_API_KEY"] = "your-azure-key"
# os.environ["AZURE_OPENAI_BASE_URL"] = "https://YOUR-RESOURCE-NAME.services.ai.azure.com/openai/v1/"
# os.environ["AZURE_OPENAI_API_VERSION"] = "preview"

# Authentication Option 2: Custom client (optional)
# from openai import OpenAI, AsyncOpenAI
# pandas_ext.use(OpenAI())
# For async operations:
# pandas_ext.use_async(AsyncOpenAI())

# Configure model (optional - defaults to gpt-4.1-mini)
# For Azure OpenAI: use your deployment name, for OpenAI: use model name
pandas_ext.responses_model("gpt-4.1-mini")

# Create your data
df = pd.DataFrame({"name": ["panda", "rabbit", "koala"]})

# Add AI-powered columns
result = df.assign(
    family=lambda df: df.name.ai.responses("What animal family? Answer with 'X family'"),
    habitat=lambda df: df.name.ai.responses("Primary habitat in one word"),
    fun_fact=lambda df: df.name.ai.responses("One interesting fact in 10 words or less")
)
```

| name   | family           | habitat | fun_fact                   |
| ------ | ---------------- | ------- | -------------------------- |
| panda  | bear family      | forest  | Eats bamboo 14 hours daily |
| rabbit | rabbit family    | meadow  | Can see nearly 360 degrees |
| koala  | marsupial family | tree    | Sleeps 22 hours per day    |

📓 **[Interactive pandas examples →](https://microsoft.github.io/openaivec/examples/pandas/)**

### Using with Reasoning Models

When using reasoning models (o1-preview, o1-mini, o3-mini, etc.), you must set `temperature=None` to avoid API errors:

```python
# For reasoning models like o1-preview, o1-mini, o3-mini
pandas_ext.responses_model("o1-mini")  # Set your reasoning model

# MUST use temperature=None with reasoning models
result = df.assign(
    analysis=lambda df: df.text.ai.responses(
        "Analyze this text step by step",
        temperature=None  # Required for reasoning models
    )
)
```

**Why this is needed**: Reasoning models don't support temperature parameters and will return an error if temperature is specified. The library automatically detects these errors and provides guidance on how to fix them.

**Reference**: [Azure OpenAI Reasoning Models](https://learn.microsoft.com/en-us/azure/ai-foundry/openai/how-to/reasoning)

### Using Pre-configured Tasks

For common text processing operations, openaivec provides ready-to-use tasks that eliminate the need to write custom prompts:

```python
from openaivec.task import nlp, customer_support

# Text analysis with pre-configured tasks
text_df = pd.DataFrame({
    "text": [
        "Great product, fast delivery!",
        "Need help with billing issue",
        "How do I reset my password?"
    ]
})

# Use pre-configured tasks for consistent, optimized results
results = text_df.assign(
    sentiment=lambda df: df.text.ai.task(nlp.SENTIMENT_ANALYSIS),
    entities=lambda df: df.text.ai.task(nlp.NAMED_ENTITY_RECOGNITION),
    intent=lambda df: df.text.ai.task(customer_support.INTENT_ANALYSIS),
    urgency=lambda df: df.text.ai.task(customer_support.URGENCY_ANALYSIS)
)

# Extract structured results into separate columns (one at a time)
extracted_results = (results
    .ai.extract("sentiment")
    .ai.extract("entities")
    .ai.extract("intent")
    .ai.extract("urgency")
)
```

**Available Task Categories:**

- **Text Analysis**: `nlp.SENTIMENT_ANALYSIS`, `nlp.MULTILINGUAL_TRANSLATION`, `nlp.NAMED_ENTITY_RECOGNITION`, `nlp.KEYWORD_EXTRACTION`
- **Content Classification**: `customer_support.INTENT_ANALYSIS`, `customer_support.URGENCY_ANALYSIS`, `customer_support.INQUIRY_CLASSIFICATION`

**Benefits of Pre-configured Tasks:**

- Optimized prompts tested across diverse datasets
- Consistent structured outputs with Pydantic validation
- Multilingual support with standardized categorical fields
- Extensible framework for adding domain-specific tasks
- Direct compatibility with Spark UDFs

### Asynchronous Processing with `.aio`

For high-performance concurrent processing, use the `.aio` accessor which provides asynchronous versions of all AI operations:

```python
import asyncio
import pandas as pd
from openaivec import pandas_ext

# Setup (same as synchronous version)
pandas_ext.responses_model("gpt-4.1-mini")

df = pd.DataFrame({"text": [
    "This product is amazing!",
    "Terrible customer service",
    "Good value for money",
    "Not what I expected"
] * 250})  # 1000 rows for demonstration

async def process_data():
    # Asynchronous processing with fine-tuned concurrency control
    results = await df["text"].aio.responses(
        "Analyze sentiment and classify as positive/negative/neutral",
        # batch_size defaults to None (automatic optimization)
        max_concurrency=12    # Allow up to 12 concurrent requests
    )
    return results

# Run the async operation
sentiments = asyncio.run(process_data())
```

**Key Parameters for Performance Tuning:**

- **`batch_size`** (default: None): Controls how many inputs are grouped into a single API request. When None (default), automatic batch size optimization adjusts based on execution time. Set to a positive integer for fixed batch size. Higher values reduce API call overhead but increase memory usage and request processing time.
- **`max_concurrency`** (default: 8): Limits the number of concurrent API requests. Higher values increase throughput but may hit rate limits or overwhelm the API.

**Performance Benefits:**

- Process thousands of records in parallel
- Automatic request batching and deduplication
- Built-in rate limiting and error handling
- Memory-efficient streaming for large datasets

## Using with Apache Spark UDFs

Scale to enterprise datasets with distributed processing:

📓 **[Complete Spark tutorial →](https://microsoft.github.io/openaivec/examples/spark/)**

First, obtain a Spark session and configure authentication:

```python
from pyspark.sql import SparkSession
from openaivec.spark import setup, setup_azure

spark = SparkSession.builder.getOrCreate()

# Option 1: Using OpenAI
setup(
    spark,
    api_key="your-openai-api-key",
    responses_model_name="gpt-4.1-mini",  # Optional: set default model
    embeddings_model_name="text-embedding-3-small"  # Optional: set default model
)

# Option 2: Using Azure OpenAI
# setup_azure(
#     spark,
#     api_key="your-azure-openai-api-key",
#     base_url="https://YOUR-RESOURCE-NAME.services.ai.azure.com/openai/v1/",
#     api_version="preview",
#     responses_model_name="my-gpt4-deployment",  # Optional: set default deployment
#     embeddings_model_name="my-embedding-deployment"  # Optional: set default deployment
# )
```

Next, create and register UDFs using the provided functions:

```python
from openaivec.spark import responses_udf, task_udf, embeddings_udf, count_tokens_udf, similarity_udf, parse_udf
from pydantic import BaseModel

# --- Register Responses UDF (String Output) ---
spark.udf.register(
    "extract_brand",
    responses_udf(
        instructions="Extract the brand name from the product. Return only the brand name."
    )
)

# --- Register Responses UDF (Structured Output with Pydantic) ---
class Translation(BaseModel):
    en: str
    fr: str
    ja: str

spark.udf.register(
    "translate_struct",
    responses_udf(
        instructions="Translate the text to English, French, and Japanese.",
        response_format=Translation
    )
)

# --- Register Embeddings UDF ---
spark.udf.register(
    "embed_text",
    embeddings_udf()
)

# --- Register Token Counting UDF ---
spark.udf.register("count_tokens", count_tokens_udf())

# --- Register Similarity UDF ---
spark.udf.register("compute_similarity", similarity_udf())

# --- Register UDFs with Pre-configured Tasks ---
from openaivec.task import nlp, customer_support

spark.udf.register(
    "analyze_sentiment",
    task_udf(
        task=nlp.SENTIMENT_ANALYSIS
    )
)

spark.udf.register(
    "classify_intent",
    task_udf(
        task=customer_support.INTENT_ANALYSIS
    )
)

# --- Register UDF for Reasoning Models ---
# For reasoning models (o1-preview, o1-mini, o3, etc.), set temperature=None
spark.udf.register(
    "reasoning_analysis",
    responses_udf(
        instructions="Analyze this step by step with detailed reasoning",
        temperature=None  # Required for reasoning models
    )
)

# --- Register Parse UDF (Dynamic Schema Inference) ---
spark.udf.register(
    "parse_dynamic",
    parse_udf(
        instructions="Extract key entities and attributes from the text",
        example_table_name="sample_texts",  # Infer schema from examples
        example_field_name="text",
        max_examples=50
    )
)

```

You can now use these UDFs in Spark SQL:

```sql
-- Create a sample table (replace with your actual table)
CREATE OR REPLACE TEMP VIEW product_reviews AS SELECT * FROM VALUES
  ('1001', 'The new TechPhone X camera quality is amazing, Nexus Corp really outdid themselves this time!'),
  ('1002', 'Quantum Galaxy has great battery life but the price is too high for what you get'),
  ('1003', 'Zephyr mobile phone crashed twice today, very disappointed with this purchase')
AS product_reviews(id, review_text);

-- Use the registered UDFs (including pre-configured tasks)
SELECT
    id,
    review_text,
    extract_brand(review_text) AS brand,
    translate_struct(review_text) AS translation,
    analyze_sentiment(review_text).sentiment AS sentiment,
    analyze_sentiment(review_text).confidence AS sentiment_confidence,
    classify_intent(review_text).primary_intent AS intent,
    classify_intent(review_text).action_required AS action_required,
    embed_text(review_text) AS embedding,
    count_tokens(review_text) AS token_count
FROM product_reviews;
```

Example Output (structure might vary slightly):

| id   | review_text                                                                   | brand      | translation                 | sentiment | sentiment_confidence | intent           | action_required    | embedding              | token_count |
| ---- | ----------------------------------------------------------------------------- | ---------- | --------------------------- | --------- | -------------------- | ---------------- | ------------------ | ---------------------- | ----------- |
| 1001 | The new TechPhone X camera quality is amazing, Nexus Corp really outdid...    | Nexus Corp | {en: ..., fr: ..., ja: ...} | positive  | 0.95                 | provide_feedback | acknowledge_review | [0.1, -0.2, ..., 0.5]  | 19          |
| 1002 | Quantum Galaxy has great battery life but the price is too high for what...   | Quantum    | {en: ..., fr: ..., ja: ...} | mixed     | 0.78                 | provide_feedback | follow_up_pricing  | [-0.3, 0.1, ..., -0.1] | 16          |
| 1003 | Zephyr mobile phone crashed twice today, very disappointed with this purchase | Zephyr     | {en: ..., fr: ..., ja: ...} | negative  | 0.88                 | complaint        | investigate_issue  | [0.0, 0.4, ..., 0.2]   | 12          |

### Spark Performance Tuning

When using openaivec with Spark, proper configuration of `batch_size` and `max_concurrency` is crucial for optimal performance:

**Automatic Caching** (New):

- **Duplicate Detection**: All AI-powered UDFs (`responses_udf`, `task_udf`, `embeddings_udf`) automatically cache duplicate inputs within each partition
- **Cost Reduction**: Significantly reduces API calls and costs on datasets with repeated content
- **Transparent**: Works automatically without code changes - your existing UDFs become more efficient
- **Partition-Level**: Each partition maintains its own cache, optimal for distributed processing patterns

**`batch_size`** (default: None):

- Controls how many rows are processed together in each API request within a partition
- **Default (None)**: Automatic batch size optimization adjusts based on execution time
- **Positive integer**: Fixed batch size - larger values reduce API calls but increase memory usage
- **Recommendation**: Use default automatic optimization, or set 32-128 for fixed batch size

**`max_concurrency`** (default: 8):

- **Important**: This is the number of concurrent API requests **PER EXECUTOR**
- Total cluster concurrency = `max_concurrency × number_of_executors`
- **Higher values**: Faster processing but may overwhelm API rate limits
- **Lower values**: More conservative, better for shared API quotas
- **Recommendation**: 4-12 per executor, considering your OpenAI tier limits

**Example for a 10-executor cluster:**

```python
# With max_concurrency=8, total cluster concurrency = 8 × 10 = 80 concurrent requests
spark.udf.register(
    "analyze_sentiment",
    responses_udf(
        instructions="Analyze sentiment as positive/negative/neutral",
        # batch_size defaults to None (automatic optimization)
        max_concurrency=8     # 80 total concurrent requests across cluster
    )
)
```

**Monitoring and Scaling:**

- Monitor OpenAI API rate limits and adjust `max_concurrency` accordingly
- Use Spark UI to optimize partition sizes and executor configurations
- Consider your OpenAI tier limits when scaling clusters

## Building Prompts

Building prompt is a crucial step in using LLMs.
In particular, providing a few examples in a prompt can significantly improve an LLM’s performance,
a technique known as "few-shot learning." Typically, a few-shot prompt consists of a purpose, cautions,
and examples.

📓 **[Advanced prompting techniques →](https://microsoft.github.io/openaivec/examples/prompt/)**

The `FewShotPromptBuilder` helps you create structured, high-quality prompts with examples, cautions, and automatic improvement.

### Basic Usage

`FewShotPromptBuilder` requires simply a purpose, cautions, and examples, and `build` method will
return rendered prompt with XML format.

Here is an example:

```python
from openaivec import FewShotPromptBuilder

prompt: str = (
    FewShotPromptBuilder()
    .purpose("Return the smallest category that includes the given word")
    .caution("Never use proper nouns as categories")
    .example("Apple", "Fruit")
    .example("Car", "Vehicle")
    .example("Tokyo", "City")
    .example("Keiichi Sogabe", "Musician")
    .example("America", "Country")
    .build()
)
print(prompt)
```

The output will be:

```xml

<Prompt>
    <Purpose>Return the smallest category that includes the given word</Purpose>
    <Cautions>
        <Caution>Never use proper nouns as categories</Caution>
    </Cautions>
    <Examples>
        <Example>
            <Input>Apple</Input>
            <Output>Fruit</Output>
        </Example>
        <Example>
            <Input>Car</Input>
            <Output>Vehicle</Output>
        </Example>
        <Example>
            <Input>Tokyo</Input>
            <Output>City</Output>
        </Example>
        <Example>
            <Input>Keiichi Sogabe</Input>
            <Output>Musician</Output>
        </Example>
        <Example>
            <Input>America</Input>
            <Output>Country</Output>
        </Example>
    </Examples>
</Prompt>
```

### Improve with OpenAI

For most users, it can be challenging to write a prompt entirely free of contradictions, ambiguities, or
redundancies.
`FewShotPromptBuilder` provides an `improve` method to refine your prompt using OpenAI's API.

`improve` method will try to eliminate contradictions, ambiguities, and redundancies in the prompt with OpenAI's API,
and iterate the process up to `max_iter` times.

Here is an example:

```python
from openai import OpenAI
from openaivec import FewShotPromptBuilder

client = OpenAI(...)
model_name = "<your-model-name>"
improved_prompt: str = (
    FewShotPromptBuilder()
    .purpose("Return the smallest category that includes the given word")
    .caution("Never use proper nouns as categories")
    # Examples which has contradictions, ambiguities, or redundancies
    .example("Apple", "Fruit")
    .example("Apple", "Technology")
    .example("Apple", "Company")
    .example("Apple", "Color")
    .example("Apple", "Animal")
    # improve the prompt with OpenAI's API
    .improve()
    .build()
)
print(improved_prompt)
```

Then we will get the improved prompt with extra examples, improved purpose, and cautions:

```xml
<Prompt>
    <Purpose>Classify a given word into its most relevant category by considering its context and potential meanings.
        The input is a word accompanied by context, and the output is the appropriate category based on that context.
        This is useful for disambiguating words with multiple meanings, ensuring accurate understanding and
        categorization.
    </Purpose>
    <Cautions>
        <Caution>Ensure the context of the word is clear to avoid incorrect categorization.</Caution>
        <Caution>Be aware of words with multiple meanings and provide the most relevant category.</Caution>
        <Caution>Consider the possibility of new or uncommon contexts that may not fit traditional categories.</Caution>
    </Cautions>
    <Examples>
        <Example>
            <Input>Apple (as a fruit)</Input>
            <Output>Fruit</Output>
        </Example>
        <Example>
            <Input>Apple (as a tech company)</Input>
            <Output>Technology</Output>
        </Example>
        <Example>
            <Input>Java (as a programming language)</Input>
            <Output>Technology</Output>
        </Example>
        <Example>
            <Input>Java (as an island)</Input>
            <Output>Geography</Output>
        </Example>
        <Example>
            <Input>Mercury (as a planet)</Input>
            <Output>Astronomy</Output>
        </Example>
        <Example>
            <Input>Mercury (as an element)</Input>
            <Output>Chemistry</Output>
        </Example>
        <Example>
            <Input>Bark (as a sound made by a dog)</Input>
            <Output>Animal Behavior</Output>
        </Example>
        <Example>
            <Input>Bark (as the outer covering of a tree)</Input>
            <Output>Botany</Output>
        </Example>
        <Example>
            <Input>Bass (as a type of fish)</Input>
            <Output>Aquatic Life</Output>
        </Example>
        <Example>
            <Input>Bass (as a low-frequency sound)</Input>
            <Output>Music</Output>
        </Example>
    </Examples>
</Prompt>
```

## Using with Microsoft Fabric

[Microsoft Fabric](https://www.microsoft.com/en-us/microsoft-fabric/) is a unified, cloud-based analytics platform that
seamlessly integrates data engineering, warehousing, and business intelligence to simplify the journey from raw data to
actionable insights.

This section provides instructions on how to integrate and use `openaivec` within Microsoft Fabric. Follow these
steps:

1. **Create an Environment in Microsoft Fabric:**

   - In Microsoft Fabric, click on **New item** in your workspace.
   - Select **Environment** to create a new environment for Apache Spark.
   - Determine the environment name, eg. `openai-environment`.
   - ![image](https://github.com/user-attachments/assets/bd1754ef-2f58-46b4-83ed-b335b64aaa1c)
     _Figure: Creating a new Environment in Microsoft Fabric._

2. **Add `openaivec` to the Environment from Public Library**

   - Once your environment is set up, go to the **Custom Library** section within that environment.
   - Click on **Add from PyPI** and search for latest version of `openaivec`.
   - Save and publish to reflect the changes.
   - ![image](https://github.com/user-attachments/assets/7b6320db-d9d6-4b89-a49d-e55b1489d1ae)
     _Figure: Add `openaivec` from PyPI to Public Library_

3. **Use the Environment from a Notebook:**

   - Open a notebook within Microsoft Fabric.
   - Select the environment you created in the previous steps.
   - ![image](https://github.com/user-attachments/assets/2457c078-1691-461b-b66e-accc3989e419)
     _Figure: Using custom environment from a notebook._
   - In the notebook, import and use `openaivec.spark` functions as you normally would. For example:

     ```python
     from openaivec.spark import setup_azure, responses_udf, embeddings_udf

     # In Microsoft Fabric, spark session is automatically available
     # spark = SparkSession.builder.getOrCreate()
     
     # Configure Azure OpenAI authentication
     setup_azure(
         spark,
         api_key="<your-api-key>",
         base_url="https://YOUR-RESOURCE-NAME.services.ai.azure.com/openai/v1/",
         api_version="preview",
         responses_model_name="my-gpt4-deployment"  # Your Azure deployment name
     )

     # Register UDFs
     spark.udf.register(
         "analyze_text",
         responses_udf(
             instructions="Analyze the sentiment of the text",
             model_name="gpt-4.1-mini"  # Use your Azure deployment name here
         )
     )
     ```

Following these steps allows you to successfully integrate and use `openaivec` within Microsoft Fabric.

## Contributing

We welcome contributions to this project! If you would like to contribute, please follow these guidelines:

1. Fork the repository and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. Ensure the test suite passes.
4. Make sure your code lints.

### Installing Dependencies

To install the necessary dependencies for development, run:

```bash
uv sync --all-extras --dev
```

### Code Formatting

To reformat the code, use the following command:

```bash
uv run ruff check . --fix
```

## Additional Resources

📓 **[Customer feedback analysis →](https://microsoft.github.io/openaivec/examples/customer_analysis/)** - Sentiment analysis & prioritization  
📓 **[Survey data transformation →](https://microsoft.github.io/openaivec/examples/survey_transformation/)** - Unstructured to structured data  
📓 **[Asynchronous processing examples →](https://microsoft.github.io/openaivec/examples/aio/)** - High-performance async workflows  
📓 **[Auto-generate FAQs from documents →](https://microsoft.github.io/openaivec/examples/generate_faq/)** - Create FAQs using AI  
📓 **[All examples →](https://microsoft.github.io/openaivec/examples/pandas/)** - Complete collection of tutorials and use cases

## Community

Join our Discord community for developers: https://discord.gg/vbb83Pgn
