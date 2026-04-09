---
title: OpenAI Batch Processing for Pandas & Spark
---

# OpenAI Batch Processing for Pandas & Spark

Welcome to **openaivec** - Transform your data analysis with OpenAI's language models and batch-first pipelines! This library enables seamless integration of AI text processing, sentiment analysis, NLP tasks, and embeddings into your [**Pandas**](https://pandas.pydata.org/) DataFrames and [**Apache Spark**](https://spark.apache.org/) workflows for scalable data insights, while automatically handling OpenAI batch orchestration.

## 🚀 Quick Start Example

Transform your data with AI in just one line:

```python
import pandas as pd
from openaivec import pandas_ext

# AI-powered data processing
fruits = pd.Series(["apple", "banana", "orange", "grape", "kiwi"])
fruits.ai.responses("Translate this fruit name into French.")
# Result: ['pomme', 'banane', 'orange', 'raisin', 'kiwi']
```

Perfect for **data scientists**, **analysts**, and **ML engineers** who want to leverage AI for text processing at scale.

## 📦 Installation

=== "pip"
    ```bash
    pip install openaivec
    ```

=== "uv"
    ```bash
    uv add openaivec
    ```

=== "With Spark Support"
    ```bash
    pip install "openaivec[spark]"
    # or
    uv add "openaivec[spark]"
    ```

## 🎯 Key Features

- **🚀 Vectorized Processing**: Handle thousands of records in minutes, not hours
- **⚡ Asynchronous Interface**: `.aio` accessor with `batch_size` and `max_concurrency` control
- **📦 OpenAI Batch Friendly**: `BatchCache` groups prompts, dedupes inputs, and keeps outputs aligned for pandas and Spark
- **💰 Cost Efficient**: Automatic deduplication significantly reduces API costs
- **🔗 Seamless Integration**: Works within existing pandas/Spark workflows
- **📈 Enterprise Scale**: From 100s to millions of records
- **🤖 Advanced NLP**: Pre-built tasks for sentiment analysis, translation, NER, and more

## Links
- [GitHub Repository](https://github.com/microsoft/openaivec/)
- [PyPI Package](https://pypi.org/project/openaivec/)
- [Complete Documentation](https://microsoft.github.io/openaivec/)

## 📚 Examples & Tutorials

Get started with these comprehensive examples:

📓 **[Getting Started](examples/pandas.ipynb)** - Basic pandas integration and usage  
📓 **[PreparedTask Guide](examples/prepared_task.ipynb)** - Define and reuse custom task schemas  
📓 **[Customer Feedback Analysis](examples/customer_analysis.ipynb)** - Sentiment analysis & prioritization  
📓 **[Survey Data Transformation](examples/survey_transformation.ipynb)** - Unstructured to structured data  
📓 **[Spark Processing](examples/spark.ipynb)** - Enterprise-scale distributed processing  
📓 **[Async Workflows](examples/aio.ipynb)** - High-performance async processing  
📓 **[Prompt Engineering](examples/prompt.ipynb)** - Advanced prompting techniques  
📓 **[FAQ Generation](examples/generate_faq.ipynb)** - Auto-generate FAQs from documents

## 📖 API Reference

Detailed documentation for all components:

🔗 **[Main Package](api/main.md)** - Core classes (BatchResponses, BatchEmbeddings, FewShotPromptBuilder)  
🔗 **[pandas_ext](api/pandas_ext.md)** - Pandas Series and DataFrame extensions  
🔗 **[spark_ext](api/spark_ext.md)** - Apache Spark UDF builders  
🔗 **[task](api/task.md)** - Pre-built task modules for NLP and customer support

## Quick Start

Here is a simple example of how to use `openaivec` with `pandas`:

```python
import openaivec
import pandas as pd
from openai import OpenAI
from openaivec import pandas_ext

from typing import List

# Set OpenAI/Azure client (optional; auto-detected from environment variables)
openaivec.set_client(OpenAI())

# Set models for responses and embeddings (optional; defaults shown)
openaivec.set_responses_model("gpt-4.1-mini")
openaivec.set_embeddings_model("text-embedding-3-small")


fruits: List[str] = ["apple", "banana", "orange", "grape", "kiwi", "mango", "peach", "pear", "pineapple", "strawberry"]
fruits_df = pd.DataFrame({"name": fruits})
```

`fruits_df` is a `pandas` DataFrame with a single column `name` containing the names of fruits. We can mutate the field `name` with the accessor `ai` to add a new column `color` with the color of each fruit:

```python
fruits_df.assign(
    color=lambda df: df["name"].ai.responses("What is the color of this fruit?")
)
```

The result is a new DataFrame with the same number of rows as `fruits_df`, but with an additional column `color` containing the color of each fruit. The `ai` accessor uses the OpenAI API to generate the responses for each fruit name in the `name` column.


| name       | color   |
|------------|---------|
| apple      | red     |
| banana     | yellow  |
| orange     | orange  |
| grape      | purple  |
| kiwi       | brown   |
| mango      | orange  |
| peach      | orange  |
| pear       | green   |
| pineapple  | brown   |
| strawberry | red     |


Structured Output is also supported. For example, we will translate the name of each fruit into multiple languages. We can use the `ai` accessor to generate a new column `translation` with the translation of each fruit name into English, French, Japanese, Spanish, German, Italian, Portuguese and Russian:

```python
from pydantic import BaseModel

class Translation(BaseModel):
    en: str  # English
    fr: str  # French
    ja: str  # Japanese
    es: str  # Spanish
    de: str  # German
    it: str  # Italian
    pt: str  # Portuguese
    ru: str  # Russian

fruits_df.assign(
    translation=lambda df: df["name"].ai.responses(
        instructions="Translate this fruit name into English, French, Japanese, Spanish, German, Italian, Portuguese and Russian.",
        response_format=Translation,
    )
)
```

| name       | translation                                                               |
|------------|----------------------------------------------------------------------------|
| apple      | en='Apple' fr='Pomme' ja='リンゴ' es='Manzana' de...                       |
| banana     | en='Banana' fr='Banane' ja='バナナ' es='Banana' de...                      |
| orange     | en='Orange' fr='Orange' ja='オレンジ' es='Naranja' de...                   |
| grape      | en='Grape' fr='Raisin' ja='ブドウ' es='Uva' de='T...                       |
| kiwi       | en='Kiwi' fr='Kiwi' ja='キウイ' es='Kiwi' de='Kiw...                       |
| mango      | en='Mango' fr='Mangue' ja='マンゴー' es='Mango' de...                      |
| peach      | en='Peach' fr='Pêche' ja='モモ' es='Durazno' de...                         |
| pear       | en='Pear' fr='Poire' ja='梨' es='Pera' de='Birn...                         |
| pineapple  | en='Pineapple' fr='Ananas' ja='パイナップル' es='Piñ...                    |
| strawberry | en='Strawberry' fr='Fraise' ja='イチゴ' es='Fresa...                       |

Structured output can be extracted into separate columns using the `extract` method. For example, we can extract the translations into separate columns for each language:

```python
fruits_df.assign(
    translation=lambda df: df["name"].ai.responses(
        instructions="Translate this fruit name into English, French, Japanese, Spanish, German, Italian, Portuguese and Russian.",
        response_format=Translation,
    )
).ai.extract("translation")
```

| name       | translation_en | translation_fr | translation_ja | translation_es | translation_de | translation_it | translation_pt | translation_ru |
|------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|
| apple      | Apple          | Pomme          | リンゴ         | Manzana        | Apfel          | Mela           | Maçã           | Яблоко         |
| banana     | Banana         | Banane         | バナナ         | Banana         | Banane         | Banana         | Banana         | Банан          |
| orange     | Orange         | Orange         | オレンジ       | Naranja        | Orange         | Arancia        | Laranja        | Апельсин       |
| grape      | Grape          | Raisin         | ブドウ         | Uva            | Traube         | Uva            | Uva            | Виноград       |
| kiwi       | Kiwi           | Kiwi           | キウイ         | Kiwi           | Kiwi           | Kiwi           | Kiwi           | Киви           |
| mango      | Mango          | Mangue         | マンゴー       | Mango          | Mango          | Mango          | Manga          | Манго          |
| peach      | Peach          | Pêche          | モモ           | Durazno        | Pfirsich       | Pesca          | Pêssego        | Персик         |
| pear       | Pear           | Poire          | 梨             | Pera           | Birne          | Pera           | Pêra           | Груша          |
| pineapple  | Pineapple      | Ananas         | パイナップル   | Piña           | Ananas         | Ananas         | Abacaxi        | Ананас         |
| strawberry | Strawberry     | Fraise         | イチゴ         | Fresa          | Erdbeere       | Fragola        | Morango        | Клубника       |

## Asynchronous Processing for High Performance

For processing large datasets efficiently, openaivec provides the `.aio` accessor that enables asynchronous, concurrent processing:

```python
import asyncio
import pandas as pd
from openaivec import pandas_ext

# Large dataset processing
df = pd.DataFrame({
    "customer_feedback": [
        "Love the new features!",
        "App crashes frequently",
        "Great customer support",
        # ... thousands more rows
    ]
})

async def analyze_feedback():
    # Process with optimized parameters
    sentiments = await df["customer_feedback"].aio.responses(
        "Classify sentiment as positive, negative, or neutral",
        batch_size=64,         # Group 64 requests per API call
        max_concurrency=16     # Allow 16 concurrent requests
    )
    
    # Also works with embeddings
    embeddings = await df["customer_feedback"].aio.embeddings(
        batch_size=128,        # Larger batches for embeddings
        max_concurrency=8      # Conservative concurrency for embeddings
    )
    
    return sentiments, embeddings

# Execute async processing
results = asyncio.run(analyze_feedback())
```

### Performance Tuning Parameters

**`batch_size`** (default: adaptive auto-tuning):
- Leave unset (`None`) to let `BatchCache` pick an efficient size (targets 30–60 seconds per batch)
- Set a positive integer for deterministic batch sizes when coordinating with rate limits
- Use `0` or a negative value only when everything fits in a single request
- Typical ranges: 32–128 for responses, 64–256 for embeddings when you need fixed sizes

**`max_concurrency`** (default: 8):
- Limits the number of simultaneous API requests
- **Higher values**: Faster processing but may hit rate limits
- **Lower values**: More conservative, better for shared API quotas
- **Recommended**: 4-16 depending on your OpenAI tier and usage patterns

### When to Use Async vs Sync

- **Use `.aio`** for: Large datasets (1000+ rows), time-sensitive processing, concurrent workflows
- **Use `.ai`** for: Small datasets, interactive analysis, simple one-off operations
