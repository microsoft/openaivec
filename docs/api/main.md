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

### Responses request budgets

Text Responses requests are planned against a conservative **128,000-token**
context estimate and **128 items** per request, independently of cache
`batch_size`. Each request includes the serialized input envelope, vectorized
instructions, generated structured-output schema, 256 expected output tokens
per item, and a 256-token validation-feedback allowance when retries are on.
`batch_size=None` remains adaptive; fixed sizes and nonpositive sizes also
respect the request budget. An oversized single text input raises `ValueError`
before any requests for that cache chunk are sent. Results preserve IDs, order,
and cache deduplication across request splits.

Check the actual **context window and output requirements** of your model or
Azure deployment. The default is not model metadata and token counts are
estimates, not a guarantee against server-side request limits. Override the
limits and tokenizer when deployment metadata is unavailable:

```python
from openaivec import BatchResponses, ResponseLimits

responses = BatchResponses.of(
    client,
    "my-deployment",
    "Summarize each item.",
    batch_size=64,
    limits=ResponseLimits(
        max_request_tokens=32000,
        max_inputs=32,
        expected_output_tokens_per_item=512,
        encoding_name="o200k_base",
    ),
)
```

The same `limits` option works on `AsyncBatchResponses.of`, both `of_task`
factories, and direct constructors. `max_output_tokens` (if supplied to the
API) is also reserved when larger than the per-item allowance. The planner
applies only to batched text inputs, not individual multimodal requests.

### Multimodal files and URLs

Set `multimodal=True` to classify local images and binary documents, or image
and document URLs, by their file extension. Signed URL query parameters are
preserved in the request but are not used to classify the media type. Audio
(`.mp3` and `.wav`) is rejected by the Responses API wrapper.

Local text files are read into the text batch; images are inlined. Local
binary documents are uploaded to the Files API as **temporary,
request-owned files** and deleted when the request finishes, including failed
requests. Validation correction attempts reuse the same upload before it is
deleted. The caller remains responsible for any files uploaded directly
through the OpenAI client or through the internal builder's standalone
`build()` method. The cache identifies local files by their contents as well
as path, so a changed file does not reuse an earlier result.

::: openaivec.ResponseLimits

::: openaivec.BatchEmbeddings

::: openaivec.AsyncBatchEmbeddings

## Prompt Building

::: openaivec.FewShotPromptBuilder
