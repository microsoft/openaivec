# Main Package API

The main `openaivec` package provides the core classes for AI-powered data processing.

## Configuration

Client and model configuration helpers. These are the canonical entry point
for setting up OpenAI / Azure OpenAI credentials and model names. The same
configuration is shared across `pandas_ext`, `duckdb_ext`, and `spark`.

::: openaivec.set_client

::: openaivec.get_client

::: openaivec.set_async_client

::: openaivec.get_async_client

::: openaivec.set_responses_model

::: openaivec.get_responses_model

::: openaivec.set_embeddings_model

::: openaivec.get_embeddings_model

## Core Classes

All core functionality is accessible through the main package imports:

::: openaivec.BatchResponses

::: openaivec.AsyncBatchResponses

::: openaivec.BatchEmbeddings

::: openaivec.AsyncBatchEmbeddings

## Prompt Building

::: openaivec.FewShotPromptBuilder
