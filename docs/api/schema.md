# Schema Inference

`SchemaInferer` and `AsyncSchemaInferer` use the supplied client and model for
every inference attempt. The asynchronous variant does not create or resolve a
synchronous client and propagates cancellation to the active request.

`max_retries=8` means at most eight total schema inference attempts, not eight
additional retries. It must be at least one. Missing parsed output, SDK schema
validation errors, and invalid generated models receive bounded correction
feedback. Unrelated API or configuration errors propagate immediately.

The pandas `series.ai.infer_schema()` and `await series.aio.infer_schema()`
methods forward API parameters such as `store`, `timeout`, `temperature`, and
`max_output_tokens` to the configured client. These options are not schema input
fields. Schema-less `.aio.parse()` also uses the configured asynchronous client
for both inference and extraction. Provide an explicit schema to skip inference.

The pandas `parse()` / `parse_with_cache()` methods and Spark `parse_udf()`
separate `max_retries=8` (total schema inference attempts) from
`max_validation_retries=3` (additional extraction corrections). Set the latter
to zero to disable extraction corrections. Neither option is forwarded as an
OpenAI API parameter. `store`, `timeout`, and other API options apply to both
inference and extraction; neither validation control changes transport retries.
Use `retry_policy` to apply [transport limits](retries.md) to both stages.
Spark `infer_schema()` also accepts `max_retries`, `retry_policy`, and API options.

```python
import openaivec
from openai import AsyncOpenAI

async def infer():
    async with AsyncOpenAI() as client:
        inferer = openaivec.AsyncSchemaInferer(client=client, model_name="gpt-4.1-mini")
        return await inferer.infer_schema(
            openaivec.SchemaInferenceInput(
                examples=["Order 42 has shipped"],
                instructions="Extract order status",
            ),
            max_retries=2,
            store=False,
        )
```

::: openaivec.SchemaInferer

::: openaivec.AsyncSchemaInferer

::: openaivec.SchemaInferenceInput

::: openaivec.SchemaInferenceOutput