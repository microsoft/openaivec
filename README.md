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

📓 **[Try it yourself →](https://openaivec.anareg.design/examples/pandas/)**

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

📓 **[See more examples →](https://openaivec.anareg.design/examples/)**

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
- **💰 Cost Efficiency**: Automatic deduplication reduces API costs by 50-90% on typical datasets  
- **🔗 Integration**: Works within existing pandas/Spark workflows without architectural changes
- **📈 Scalability**: Same API scales from exploratory analysis (100s of records) to production systems (millions of records)
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
client = BatchResponses(
    client=OpenAI(),
    model_name="gpt-4o-mini",
    system_message="Please answer only with 'xx family' and do not output anything else."
)

result = client.parse(["panda", "rabbit", "koala"], batch_size=32)
print(result)  # Expected output: ['bear family', 'rabbit family', 'koala family']
```

📓 **[Complete tutorial →](https://openaivec.anareg.design/examples/pandas/)**

### Pandas Integration (Recommended)

The easiest way to get started with your DataFrames:

```python
import pandas as pd
from openaivec import pandas_ext

# Setup (optional - uses OPENAI_API_KEY environment variable by default)
pandas_ext.responses_model("gpt-4o-mini")

# Create your data
df = pd.DataFrame({"name": ["panda", "rabbit", "koala"]})

# Add AI-powered columns
result = df.assign(
    family=lambda df: df.name.ai.responses("What animal family? Answer with 'X family'"),
    habitat=lambda df: df.name.ai.responses("Primary habitat in one word"),
    fun_fact=lambda df: df.name.ai.responses("One interesting fact in 10 words or less")
)
```

| name   | family        | habitat | fun_fact                    |
|--------|---------------|---------|-----------------------------|
| panda  | bear family   | forest  | Eats bamboo 14 hours daily  |
| rabbit | rabbit family | meadow  | Can see nearly 360 degrees  |
| koala  | marsupial family | tree   | Sleeps 22 hours per day    |

📓 **[Interactive pandas examples →](https://openaivec.anareg.design/examples/pandas/)**

## Using with Apache Spark UDFs

Scale to enterprise datasets with distributed processing:

📓 **[Complete Spark tutorial →](https://openaivec.anareg.design/examples/spark/)**

First, obtain a Spark session:

```python
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()
```

Next, instantiate UDF builders using either OpenAI or Azure OpenAI credentials and register the UDFs.

```python
import os
from openaivec.spark import ResponsesUDFBuilder, EmbeddingsUDFBuilder, count_tokens_udf
from pydantic import BaseModel

# --- Option 1: Using OpenAI ---
resp_builder_openai = ResponsesUDFBuilder.of_openai(
    api_key=os.getenv("OPENAI_API_KEY"),
    model_name="gpt-4o-mini", # Model for responses
)
emb_builder_openai = EmbeddingsUDFBuilder.of_openai(
    api_key=os.getenv("OPENAI_API_KEY"),
    model_name="text-embedding-3-small", # Model for embeddings
)

# --- Option 2: Using Azure OpenAI ---
# resp_builder_azure = ResponsesUDFBuilder.of_azure_openai(
#     api_key=os.getenv("AZURE_OPENAI_KEY"),
#     endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
#     api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
#     model_name="<your-resp-deployment-name>", # Deployment for responses
# )
# emb_builder_azure = EmbeddingsUDFBuilder.of_azure_openai(
#     api_key=os.getenv("AZURE_OPENAI_KEY"),
#     endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
#     api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
#     model_name="<your-emb-deployment-name>", # Deployment for embeddings
# )

# --- Register Responses UDF (String Output) ---
# Use the builder corresponding to your setup (OpenAI or Azure)
spark.udf.register(
    "parse_flavor",
    resp_builder_openai.build( # or resp_builder_azure.build(...)
        instructions="Extract flavor-related information. Return only the concise flavor name.",
        response_format=str, # Specify string output
    )
)

# --- Register Responses UDF (Structured Output with Pydantic) ---
class Translation(BaseModel):
    en: str
    fr: str
    ja: str

spark.udf.register(
    "translate_struct",
    resp_builder_openai.build( # or resp_builder_azure.build(...)
        instructions="Translate the text to English, French, and Japanese.",
        response_format=Translation, # Specify Pydantic model for structured output
    )
)

# --- Register Embeddings UDF ---
spark.udf.register(
    "embed_text",
    emb_builder_openai.build() # or emb_builder_azure.build()
)

# --- Register Token Counting UDF ---
spark.udf.register("count_tokens", count_tokens_udf("gpt-4o"))

```

You can now use these UDFs in Spark SQL:

```sql
-- Create a sample table (replace with your actual table)
CREATE OR REPLACE TEMP VIEW product_names AS SELECT * FROM VALUES
  ('4414732714624', 'Cafe Mocha Smoothie (Trial Size)'),
  ('4200162318339', 'Dark Chocolate Tea (New Product)'),
  ('4920122084098', 'Uji Matcha Tea (New Product)')
AS product_names(id, product_name);

-- Use the registered UDFs
SELECT
    id,
    product_name,
    parse_flavor(product_name) AS flavor,
    translate_struct(product_name) AS translation,
    embed_text(product_name) AS embedding,
    count_tokens(product_name) AS token_count
FROM product_names;
```

Example Output (structure might vary slightly):

| id            | product_name                      | flavor    | translation                      | embedding                      | token_count |
|---------------|-----------------------------------|-----------|----------------------------------|--------------------------------|-------------|
| 4414732714624 | Cafe Mocha Smoothie (Trial Size)  | Mocha     | {en: ..., fr: ..., ja: ...}    | [0.1, -0.2, ..., 0.5]          | 8           |
| 4200162318339 | Dark Chocolate Tea (New Product)  | Chocolate | {en: ..., fr: ..., ja: ...}    | [-0.3, 0.1, ..., -0.1]         | 7           |
| 4920122084098 | Uji Matcha Tea (New Product)      | Matcha    | {en: ..., fr: ..., ja: ...}    | [0.0, 0.4, ..., 0.2]           | 8           |

## Building Prompts

Building prompt is a crucial step in using LLMs.
In particular, providing a few examples in a prompt can significantly improve an LLM’s performance,
a technique known as "few-shot learning." Typically, a few-shot prompt consists of a purpose, cautions,
and examples.

📓 **[Advanced prompting techniques →](https://openaivec.anareg.design/examples/prompt/)**

The `FewShotPromptBuilder` helps you create structured, high-quality prompts with examples, cautions, and automatic improvement.

### Basic Usage

`FewShotPromptBuilder` requires simply a purpose, cautions, and examples, and `build` method will
return rendered prompt with XML format.

Here is an example:

```python
from openaivec.prompt import FewShotPromptBuilder

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
from openaivec.prompt import FewShotPromptBuilder

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
    .improve(client, model_name)
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
   - In the notebook, import and use `openaivec.spark.ResponsesUDFBuilder` as you normally would. For example:

     ```python
     from openaivec.spark import ResponsesUDFBuilder

     resp_builder = ResponsesUDFBuilder.of_azure_openai(
         api_key="<your-api-key>",
         endpoint="https://<your-resource-name>.openai.azure.com",
         api_version="2024-10-21",
         model_name="<your-deployment-name>"
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

📓 **[Customer feedback analysis →](https://openaivec.anareg.design/examples/customer_analysis/)** - Sentiment analysis & prioritization  
📓 **[Survey data transformation →](https://openaivec.anareg.design/examples/survey_transformation/)** - Unstructured to structured data  
📓 **[Asynchronous processing examples →](https://openaivec.anareg.design/examples/aio/)** - High-performance async workflows  
📓 **[Auto-generate FAQs from documents →](https://openaivec.anareg.design/examples/generate_faq/)** - Create FAQs using AI  
📓 **[All examples →](https://openaivec.anareg.design/examples/)** - Complete collection of tutorials and use cases

## Community

Join our Discord community for developers: https://discord.gg/vbb83Pgn
