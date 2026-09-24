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

The pandas `parse()` / `parse_with_cache()` methods and Spark and DuckDB `parse_udf()`
separate `max_retries=8` (total schema inference attempts) from
`max_validation_retries=3` (additional extraction corrections). Set the latter
to zero to disable extraction corrections. Neither option is forwarded as an
OpenAI API parameter. `store`, `timeout`, and other API options apply to both
inference and extraction; neither validation control changes transport retries.
Use `retry_policy` to apply [transport limits](retries.md) to both stages.
Spark and DuckDB `infer_schema()` also accept `max_retries`, `retry_policy`,
and API options. DuckDB infers from a bounded, non-NULL table column sample
before registering a UDF because the SQL return type must be known in advance.

Generated schemas require at least one field in every object, use exact
case-sensitive string enum values (1–24 raw labels), and reject unknown fields
at the root and in nested objects. Exact duplicate enum values are removed in
first-seen order; labels that cannot be used as Python member names retain their
original JSON values under generated member names. A field marked `nullable`
still must appear in the output, but may contain `null`.

`FieldSpec.minimum` and `maximum` enforce inclusive, finite numeric bounds;
integer bounds must be whole numbers. `FieldSpec.boolean_value` enforces a
fixed boolean. Unsupported type/constraint combinations and reversed bounds
fail model construction. Descriptions provide **semantic guidance only**; they
do not impose additional runtime constraints. The extraction prompt's field
contract is generated from the validated specification rather than trusting
the model's independently generated `inference_prompt`. Extraction validation
errors, including unexpected fields, use the existing bounded correction path
controlled by `max_validation_retries`. User-supplied explicit Pydantic models
keep their own configuration and are not modified.

```python
import openaivec
from openai import AsyncOpenAI

async def infer():
    async with AsyncOpenAI() as client:
        inferer = openaivec.AsyncSchemaInferer(client=client, model_name="gpt-6-luna")
        return await inferer.infer_schema(
            openaivec.SchemaInferenceInput(
                examples=["Order 42 has shipped"],
                instructions="Extract order status",
            ),
            max_retries=2,
            reasoning={"effort": "none"},
            store=False,
        )
```

::: openaivec.SchemaInferer

::: openaivec.AsyncSchemaInferer

::: openaivec.SchemaInferenceInput

::: openaivec.SchemaInferenceOutput
