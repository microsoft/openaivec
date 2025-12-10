# openaivec

Transform pandas and Spark workflows with AI-powered text processing—batching, caching, and guardrails included. Built for OpenAI batch pipelines so you can group prompts, cut API overhead, and keep outputs aligned with your data.

[Contributor guidelines](AGENTS.md)

## Quick start

```bash
pip install openaivec
```

```python
import os
import pandas as pd
from openaivec import pandas_ext

# Auth: choose OpenAI or Azure OpenAI
os.environ["OPENAI_API_KEY"] = "your-api-key"
# Azure alternative:
# os.environ["AZURE_OPENAI_API_KEY"] = "your-azure-key"
# os.environ["AZURE_OPENAI_BASE_URL"] = "https://YOUR-RESOURCE-NAME.services.ai.azure.com/openai/v1/"
# os.environ["AZURE_OPENAI_API_VERSION"] = "preview"

pandas_ext.set_responses_model("gpt-5.1")  # Optional override (use deployment name for Azure)

reviews = pd.Series([
    "Great coffee and friendly staff.",
    "Delivery was late and the package was damaged.",
])

sentiment = reviews.ai.responses(
    "Summarize sentiment in one short sentence.",
    reasoning={"effort": "none"},  # Mirrors OpenAI SDK for reasoning models
)
print(sentiment.tolist())
```

**Try it live:** https://microsoft.github.io/openaivec/examples/pandas/

## Benchmarks

Simple task benchmark from [benchmark.ipynb](https://github.com/microsoft/openaivec/blob/main/docs/examples/benchmark.ipynb) (100 numeric strings → integer literals, `Series.aio.responses`, model `gpt-5.1`):

| Mode                | Settings                                        | Time (s) |
| ------------------- | ----------------------------------------------- | -------- |
| Serial              | `batch_size=1`, `max_concurrency=1`             | ~141     |
| Batching            | default `batch_size`, `max_concurrency=1`       | ~15      |
| Concurrent batching | default `batch_size`, default `max_concurrency` | ~6       |

Batching alone removes most HTTP overhead, and letting batching overlap with concurrency cuts total runtime to a few seconds while still yielding one output per input.

<img alt="image" src="https://github.com/user-attachments/assets/8ace9bcd-bcae-4023-a37e-13082cd645e5" />

## Contents

- [Why openaivec?](#why-openaivec)
- [Core Workflows](#core-workflows)
- [Using with Apache Spark UDFs](#using-with-apache-spark-udfs)
- [Building Prompts](#building-prompts)
- [Using with Microsoft Fabric](#using-with-microsoft-fabric)
- [Contributing](#contributing)
- [Additional Resources](#additional-resources)
- [Community](#community)

## Why openaivec?

- Drop-in `.ai` and `.aio` accessors keep pandas analysts in familiar tooling.
- OpenAI batch-optimized: `BatchingMapProxy`/`AsyncBatchingMapProxy` coalesce requests, dedupe prompts, and keep column order stable.
- Smart batching (`BatchingMapProxy`/`AsyncBatchingMapProxy`) dedupes prompts, preserves order, and releases waiters on failure.
- Reasoning support mirrors the OpenAI SDK; structured outputs accept Pydantic `response_format`.
- Built-in caches and retries remove boilerplate; helpers reuse caches across pandas, Spark, and async flows.
- Spark UDFs and Microsoft Fabric guides move notebooks into production-scale ETL.
- Prompt tooling (`FewShotPromptBuilder`, `improve`) and the task library ship curated prompts with validated outputs.

# Overview

Vectorized OpenAI batch processing so you handle many inputs per call instead of one-by-one. Batching proxies dedupe inputs, enforce ordered outputs, and unblock waiters even on upstream errors. Cache helpers (`responses_with_cache`, Spark UDF builders) plug into the same layer so expensive prompts are reused across pandas, Spark, and async flows. Reasoning models honor SDK semantics. Requires Python 3.10+.

## Core Workflows

### Direct API usage

For maximum control over batch processing:

```python
import os
from openai import OpenAI
from openaivec import BatchResponses

# Initialize the batch client
client = BatchResponses.of(
    client=OpenAI(),
    model_name="gpt-5.1",
    system_message="Please answer only with 'xx family' and do not output anything else.",
    # batch_size defaults to None (automatic optimization)
)

result = client.parse(
    ["panda", "rabbit", "koala"],
    reasoning={"effort": "none"},
)
print(result)  # Expected output: ['bear family', 'rabbit family', 'koala family']
```

📓 **[Complete tutorial →](https://microsoft.github.io/openaivec/examples/pandas/)**

### pandas integration (recommended)

The easiest way to get started with your DataFrames:

```python
import os
import pandas as pd
from openaivec import pandas_ext

# Authentication Option 1: Environment variables (automatic detection)
os.environ["OPENAI_API_KEY"] = "your-api-key-here"
# Or for Azure OpenAI:
# os.environ["AZURE_OPENAI_API_KEY"] = "your-azure-key"
# os.environ["AZURE_OPENAI_BASE_URL"] = "https://YOUR-RESOURCE-NAME.services.ai.azure.com/openai/v1/"
# os.environ["AZURE_OPENAI_API_VERSION"] = "preview"

# Authentication Option 2: Custom client (optional)
# from openai import OpenAI, AsyncOpenAI
# pandas_ext.set_client(OpenAI())
# pandas_ext.set_async_client(AsyncOpenAI())

# Configure model (optional - defaults to gpt-5.1; use deployment name for Azure)
pandas_ext.set_responses_model("gpt-5.1")

# Create your data
df = pd.DataFrame({"name": ["panda", "rabbit", "koala"]})

# Add AI-powered columns
result = df.assign(
    family=lambda df: df.name.ai.responses(
        "What animal family? Answer with 'X family'",
        reasoning={"effort": "none"},
    ),
    habitat=lambda df: df.name.ai.responses(
        "Primary habitat in one word",
        reasoning={"effort": "none"},
    ),
    fun_fact=lambda df: df.name.ai.responses(
        "One interesting fact in 10 words or less",
        reasoning={"effort": "none"},
    ),
)
```

| name   | family           | habitat | fun_fact                   |
| ------ | ---------------- | ------- | -------------------------- |
| panda  | bear family      | forest  | Eats bamboo 14 hours daily |
| rabbit | rabbit family    | meadow  | Can see nearly 360 degrees |
| koala  | marsupial family | tree    | Sleeps 22 hours per day    |

📓 **[Interactive pandas examples →](https://microsoft.github.io/openaivec/examples/pandas/)**

### Using with reasoning models

Reasoning models (o1-preview, o1-mini, o3-mini, etc.) work without special flags. `reasoning` mirrors the OpenAI SDK.

```python
pandas_ext.set_responses_model("o1-mini")  # Set your reasoning model

result = df.assign(
    analysis=lambda df: df.text.ai.responses(
        "Analyze this text step by step",
        reasoning={"effort": "none"}  # Optional: mirrors the OpenAI SDK argument
    )
)
```

You can omit `reasoning` to use the model defaults or tune it per request with the same shape (`dict` with effort) as the OpenAI SDK.

### Using pre-configured tasks

For common text processing operations, openaivec provides ready-to-use tasks that eliminate the need to write custom prompts:

```python
from openaivec.task import nlp, customer_support

text_df = pd.DataFrame({
    "text": [
        "Great product, fast delivery!",
        "Need help with billing issue",
        "How do I reset my password?"
    ]
})

results = text_df.assign(
    sentiment=lambda df: df.text.ai.task(
        nlp.SENTIMENT_ANALYSIS,
        reasoning={"effort": "none"},
    ),
    intent=lambda df: df.text.ai.task(
        customer_support.INTENT_ANALYSIS,
        reasoning={"effort": "none"},
    ),
)

# Extract structured results into separate columns
extracted_results = results.ai.extract("sentiment")
```

**Task categories:** Text analysis (`nlp.SENTIMENT_ANALYSIS`, `nlp.MULTILINGUAL_TRANSLATION`, `nlp.NAMED_ENTITY_RECOGNITION`, `nlp.KEYWORD_EXTRACTION`); Content classification (`customer_support.INTENT_ANALYSIS`, `customer_support.URGENCY_ANALYSIS`, `customer_support.INQUIRY_CLASSIFICATION`).

### Asynchronous processing with `.aio`

High-throughput workloads use the `.aio` accessor for async versions of all operations:

```python
import asyncio
import pandas as pd
from openaivec import pandas_ext

pandas_ext.set_responses_model("gpt-5.1")

df = pd.DataFrame({"text": [
    "This product is amazing!",
    "Terrible customer service",
    "Good value for money",
    "Not what I expected"
] * 250})  # 1000 rows for demonstration

async def process_data():
    return await df["text"].aio.responses(
        "Analyze sentiment and classify as positive/negative/neutral",
        reasoning={"effort": "none"},  # Required for gpt-5.1
        max_concurrency=12    # Allow up to 12 concurrent requests
    )

sentiments = asyncio.run(process_data())
```

**Performance benefits:** Parallel processing with automatic batching/deduplication, built-in rate limiting and error handling, and memory-efficient streaming for large datasets.

## Using with Apache Spark UDFs

Scale to enterprise datasets with distributed processing.

📓 **[Spark tutorial →](https://microsoft.github.io/openaivec/examples/spark/)**

First, obtain a Spark session and configure authentication:

```python
from pyspark.sql import SparkSession
from openaivec.spark import setup, setup_azure

spark = SparkSession.builder.getOrCreate()

# Option 1: Using OpenAI
setup(
    spark,
    api_key="your-openai-api-key",
    responses_model_name="gpt-5.1",  # Optional: set default model
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

Create and register UDFs using the provided helpers:

```python
from openaivec.spark import responses_udf, task_udf, embeddings_udf, count_tokens_udf, similarity_udf, parse_udf
from pydantic import BaseModel

spark.udf.register(
    "extract_brand",
    responses_udf(
        instructions="Extract the brand name from the product. Return only the brand name.",
        reasoning={"effort": "none"},  # Recommended with gpt-5.1
    )
)

class Translation(BaseModel):
    en: str
    fr: str
    ja: str

spark.udf.register(
    "translate_struct",
    responses_udf(
        instructions="Translate the text to English, French, and Japanese.",
        response_format=Translation,
        reasoning={"effort": "none"},  # Recommended with gpt-5.1
    )
)

spark.udf.register("embed_text", embeddings_udf())
spark.udf.register("count_tokens", count_tokens_udf())
spark.udf.register("compute_similarity", similarity_udf())
```

### Spark performance tips

- Duplicate detection automatically caches repeated inputs per partition for UDFs.
- `batch_size=None` auto-optimizes; set 32–128 for fixed sizes if needed.
- `max_concurrency` is per executor; total concurrency = executors × max_concurrency. Start with 4–12.
- Monitor rate limits and adjust concurrency to your OpenAI tier.

## Building Prompts

Few-shot prompts improve LLM quality. `FewShotPromptBuilder` structures purpose, cautions, and examples; `improve()` iterates with OpenAI to remove contradictions.

```python
from openaivec import FewShotPromptBuilder

prompt = (
    FewShotPromptBuilder()
    .purpose("Return the smallest category that includes the given word")
    .caution("Never use proper nouns as categories")
    .example("Apple", "Fruit")
    .example("Car", "Vehicle")
    .improve(max_iter=1)  # optional
    .build()
)
```

📓 **[Advanced prompting techniques →](https://microsoft.github.io/openaivec/examples/prompt/)**

## Using with Microsoft Fabric

[Microsoft Fabric](https://www.microsoft.com/en-us/microsoft-fabric/) is a unified, cloud-based analytics platform. Add `openaivec` from PyPI in your Fabric environment, select it in your notebook, and use `openaivec.spark` like standard Spark.

## Contributing

We welcome contributions! Please:

1. Fork and branch from `main`.
2. Add or update tests when you change code.
3. Run formatting and tests before opening a PR.

Install dev deps:

```bash
uv sync --all-extras --dev
```

Lint and format:

```bash
uv run ruff check . --fix
```

Quick test pass:

```bash
uv run pytest -m "not slow and not requires_api"
```

## Additional Resources

📓 **[Customer feedback analysis →](https://microsoft.github.io/openaivec/examples/customer_analysis/)** - Sentiment analysis & prioritization  
📓 **[Survey data transformation →](https://microsoft.github.io/openaivec/examples/survey_transformation/)** - Unstructured to structured data  
📓 **[Asynchronous processing examples →](https://microsoft.github.io/openaivec/examples/aio/)** - High-performance async workflows  
📓 **[Auto-generate FAQs from documents →](https://microsoft.github.io/openaivec/examples/generate_faq/)** - Create FAQs using AI  
📓 **[All examples →](https://microsoft.github.io/openaivec/examples/pandas/)** - Complete collection of tutorials and use cases

## Community

Join our Discord community for support and announcements: https://discord.gg/hXCS9J6Qek
