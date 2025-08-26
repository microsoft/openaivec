"""Pre-configured task library for OpenAI API structured outputs.

This module provides a comprehensive collection of pre-configured tasks designed for
various business and academic use cases. Tasks are organized into domain-specific
submodules, each containing ready-to-use `PreparedTask` instances that work seamlessly
with openaivec's batch processing capabilities.

## Available Task Domains

### Natural Language Processing (`nlp`)
Core NLP tasks for text analysis and processing:

- **Translation**: Multi-language translation with 40+ language support
- **Sentiment Analysis**: Emotion detection and sentiment scoring
- **Named Entity Recognition**: Extract people, organizations, locations
- **Morphological Analysis**: Part-of-speech tagging and lemmatization
- **Dependency Parsing**: Syntactic structure analysis
- **Keyword Extraction**: Important term identification

### Customer Support (`customer_support`)
Specialized tasks for customer service operations:

- **Intent Analysis**: Understand customer goals and requirements
- **Sentiment Analysis**: Customer satisfaction and emotional state
- **Urgency Analysis**: Priority assessment and response time recommendations
- **Inquiry Classification**: Automatic categorization and routing
- **Inquiry Summary**: Comprehensive issue summarization
- **Response Suggestion**: AI-powered response drafting

## Usage Patterns

### Quick Start with Default Tasks
```python
from openai import OpenAI
from openaivec._responses import BatchResponses
from openaivec.task import nlp, customer_support

client = OpenAI()

# Use pre-configured tasks
sentiment_analyzer = BatchResponses.of_task(
    client=client,
    model_name="gpt-4.1-mini",
    task=nlp.SENTIMENT_ANALYSIS
)

intent_analyzer = BatchResponses.of_task(
    client=client,
    model_name="gpt-4.1-mini",
    task=customer_support.INTENT_ANALYSIS
)
```

### Customized Task Configuration
```python
from openaivec.task.customer_support import urgency_analysis

# Create customized urgency analysis
custom_urgency = urgency_analysis(
    business_context="SaaS platform support",
    urgency_levels={
        "critical": "Service outages, security breaches",
        "high": "Login issues, payment failures",
        "medium": "Feature bugs, billing questions",
        "low": "Feature requests, general feedback"
    }
)

analyzer = BatchResponses.of_task(
    client=client,
    model_name="gpt-4.1-mini",
    task=custom_urgency
)
```

### Pandas Integration
```python
import pandas as pd
from openaivec import pandas_ext

df = pd.DataFrame({"text": ["I love this!", "This is terrible."]})

# Apply tasks directly to DataFrame columns
df["sentiment"] = df["text"].ai.task(nlp.SENTIMENT_ANALYSIS)
df["intent"] = df["text"].ai.task(customer_support.INTENT_ANALYSIS)

# Extract structured results
results_df = df.ai.extract("sentiment")
```

### Spark Integration
```python
from openaivec.spark import ResponsesUDFBuilder

# Register UDF for large-scale processing
spark.udf.register(
    "analyze_sentiment",
    ResponsesUDFBuilder.of_openai(
        api_key=api_key,
        model_name="gpt-4.1-mini"
    ).build_from_task(task=nlp.SENTIMENT_ANALYSIS)
)

# Use in Spark SQL
df = spark.sql(\"\"\"
    SELECT text, analyze_sentiment(text) as sentiment
    FROM customer_feedback
\"\"\")
```

## Task Architecture

### PreparedTask Structure
All tasks are built using the `PreparedTask` dataclass:

```python
@dataclass(frozen=True)
class PreparedTask:
    instructions: str           # Detailed prompt for the LLM
    response_format: type[ResponseFormat]    # Pydantic model or str for structured/plain output
    temperature: float = 0.0    # Sampling temperature
    top_p: float = 1.0         # Nucleus sampling parameter
```

### Response Format Standards
- **Literal Types**: Categorical fields use `typing.Literal` for type safety
- **Multilingual**: Non-categorical fields respond in input language
- **Validation**: Pydantic models ensure data integrity
- **Spark Compatible**: All types map correctly to Spark schemas

### Design Principles
1. **Consistency**: Uniform API across all task domains
2. **Configurability**: Customizable parameters for different use cases
3. **Type Safety**: Strong typing with Pydantic validation
4. **Scalability**: Optimized for batch processing and large datasets
5. **Extensibility**: Easy to add new domains and tasks

## Adding New Task Domains

To add a new domain (e.g., `finance`, `healthcare`, `legal`):

1. **Create Domain Module**: `src/openaivec/task/new_domain/`
2. **Implement Tasks**: Following existing patterns with Pydantic models
3. **Add Multilingual Support**: Include language-aware instructions
4. **Export Functions**: Both configurable functions and constants
5. **Update Documentation**: Add to this module docstring

### Example New Domain Structure
```
src/openaivec/task/finance/
├── __init__.py              # Export all functions and constants
├── risk_assessment.py       # Credit risk, market risk analysis
├── document_analysis.py     # Financial document processing
└── compliance_check.py      # Regulatory compliance verification
```

## Performance Considerations

- **Batch Processing**: Use `BatchResponses` for multiple inputs
- **Deduplication**: Automatic duplicate removal reduces API costs
- **Caching**: Results are cached based on input content
- **Async Support**: `AsyncBatchResponses` for concurrent processing
- **Token Optimization**: Vectorized system messages for efficiency

## Best Practices

1. **Choose Appropriate Models**:
   - `gpt-4.1-mini`: Fast, cost-effective for most tasks
   - `gpt-4o`: Higher accuracy for complex analysis

2. **Customize When Needed**:
   - Use default tasks for quick prototyping
   - Configure custom tasks for production use

3. **Handle Multilingual Input**:
   - Tasks automatically detect and respond in input language
   - Categorical fields remain in English for system compatibility

4. **Monitor Performance**:
   - Use batch sizes appropriate for your use case
   - Monitor token usage for cost optimization

See individual task modules for detailed documentation and examples.
"""

__all__ = []
