# Main Package API

The main `openaivec` package provides the core classes for AI-powered data processing.

## Configuration

Client and model configuration helpers. These are the canonical entry point
for setting up OpenAI / Azure OpenAI credentials, Fabric built-in models, and model
names. Configuration is shared within the current Python process; Spark executors
require their own setup. See the [authentication guide](../authentication.md).

::: openaivec.setup_fabric

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
