# Main Package API

The main `openaivec` package provides the core classes for AI-powered data processing.

## Core Classes

All core functionality is accessible through the main package imports:

::: openaivec.BatchResponses
    options:
      members:
        - of
        - of_task
        - parse

::: openaivec.AsyncBatchResponses
    options:
      members:
        - of
        - of_task
        - parse

::: openaivec.BatchEmbeddings
    options:
      members:
        - of
        - create

::: openaivec.AsyncBatchEmbeddings
    options:
      members:
        - of
        - create

## Task Configuration

::: openaivec.PreparedTask

## Prompt Building

::: openaivec.FewShotPromptBuilder
    options:
      members:
        - purpose
        - caution
        - example
        - improve
        - build
        - build_json
        - get_object

## Usage Examples

### Basic Batch Processing

```python
from openaivec import BatchResponses
from openai import OpenAI

# Create batch client
client = BatchResponses.of(
    client=OpenAI(),
    model_name="gpt-4.1-mini"
)

# Process multiple inputs
results = client.parse([
    "Translate 'hello' to French",
    "What is 2+2?",
    "Name three colors"
])
```

### Structured Outputs with Tasks

```python
from openaivec import BatchResponses, PreparedTask
from openai import OpenAI
from pydantic import BaseModel

class Sentiment(BaseModel):
    sentiment: str
    confidence: float

task = PreparedTask(
    instructions="Analyze sentiment",
    response_format=Sentiment,
    temperature=0.0
)

client = BatchResponses.of_task(
    client=OpenAI(),
    model_name="gpt-4.1-mini", 
    task=task
)

results = client.parse([
    "I love this product!",
    "This is terrible quality"
])
```

### Advanced Prompt Building

```python
from openaivec import FewShotPromptBuilder

prompt = (
    FewShotPromptBuilder()
    .purpose("Classify animals by their habitat")
    .caution("Consider both land and water animals")
    .example("dolphin", "aquatic")
    .example("eagle", "aerial") 
    .example("bear", "terrestrial")
    .improve()  # AI-powered improvement
    .build()
)
```