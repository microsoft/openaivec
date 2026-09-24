# Migrating to GPT-6

The default OpenAI Responses model is now `gpt-6-luna`, replacing
`gpt-4.1-mini`. This affects calls that resolve the package default, including
pandas accessors, schema inference, prompt improvement, and newly created
Spark or DuckDB UDFs. Explicit model names are preserved. The Responses API and
Pydantic structured-output interfaces stay the same; embeddings are unchanged.

Fabric built-in defaults remain `gpt-5.1` for Responses and
`text-embedding-ada-002` for embeddings. Their availability follows Fabric's
release cycle. Azure OpenAI calls still require the deployment name configured
in your own resource; an OpenAI API model name does not establish Azure or Fabric
availability. See [authentication](authentication.md) for those routes.

## Choose a model and reasoning setting

Start with Luna and `reasoning={"effort": "none"}` for classification,
extraction, translation, and routine table processing. Evaluate Luna with `low`
or Sol with `low` when your task needs more complex judgment. These are starting
settings for evaluation, not a guarantee of equivalent output quality.

| Workload | Model | Initial reasoning setting |
| --- | --- | --- |
| Routine processing of many rows | `gpt-6-luna` | `none` |
| More demanding tasks | `gpt-6-luna` | `low` |
| Complex analysis or schema inference | `gpt-6-sol` | `low` |

Luna and Sol support the Responses API and Structured Outputs. Both default to
`medium` reasoning if you omit the setting. Changing only the model name can
therefore change latency and token usage. `openaivec` forwards your API options
unchanged and does not inject a reasoning setting or remove incompatible options.
See the official [Luna model page](https://developers.openai.com/api/docs/models/gpt-6-luna)
and [Sol model page](https://developers.openai.com/api/docs/models/gpt-6-sol).

For reasoning other than `none`, omit `temperature`, `top_p`, and
logprobs-related options. Review both the options supplied to a wrapper factory
and those supplied to pandas methods or UDF factories. See the official
[migration guidance](https://developers.openai.com/api/docs/guides/latest-model/gpt-6-astra.md#migration-quickstart).

## Update calls

Configure authentication and the model before making calls or creating reusable
wrappers and UDFs. Existing applications can pin their previous model explicitly
while evaluating the new default.

```python
import openaivec
import pandas as pd
from openaivec import pandas_ext

# Set OPENAI_API_KEY in the environment before running this example.
openaivec.set_responses_model("gpt-6-luna")

texts = pd.Series(["Great product!", "The delivery was late."])
sentiments = texts.ai.responses(
    "Classify sentiment as positive, negative, or neutral.",
    reasoning={"effort": "none"},
    batch_size=2,
)
```

Async accessors accept the same options:

```python
async def classify(texts):
    return await texts.aio.responses(
        "Classify sentiment as positive, negative, or neutral.",
        reasoning={"effort": "none"},
        batch_size=2,
        max_concurrency=1,
    )
```

Pass API options to `BatchResponses.of()` or `of_task()` when constructing a
wrapper. Its `parse()` method receives only the input list. `PreparedTask`
stores instructions and a schema, so supply the reasoning setting each time you
construct a wrapper or call `.ai.task()` / `.aio.task()`.

```python
from openai import OpenAI
from openaivec import BatchResponses

with OpenAI() as client:
    responses = BatchResponses.of(
        client=client,
        model_name="gpt-6-luna",
        system_message="Translate each input to French.",
        reasoning={"effort": "none"},
        batch_size=2,
    )
    translations = responses.parse(["apple", "banana"])
```

For schema-less parsing, the API options apply to schema inference and extraction.
Provide an explicit Pydantic `response_format` if the output shape must stay
fixed. Update any manually constructed `SchemaInferer` or `AsyncSchemaInferer`
with the selected model. `FewShotPromptBuilder.improve()` also accepts
`reasoning={"effort": "none"}`.

## Recreate UDFs after configuration changes

For Spark, call `setup()` or `setup_azure()` before creating UDFs so the driver and
executor configuration agree. Spark and DuckDB UDF factories capture their model
and request settings when created. Calling `set_responses_model()` later does
not update an existing UDF.

```python
from openaivec.spark_ext import responses_udf, setup

setup(spark, api_key=api_key, responses_model_name="gpt-6-luna")
spark.udf.register(
    "classify_sentiment",
    responses_udf(
        "Classify sentiment as positive, negative, or neutral.",
        reasoning={"effort": "none"},
        batch_size=2,
        max_concurrency=1,
    ),
)
```

Recreate and register Spark UDFs after changing the model or request options.
For an existing DuckDB function, remove it with `conn.remove_function(name)`
before registering its replacement. Rebuild any wrappers that were created with
the old model. Keep Fabric UDFs on the models supplied by the Fabric runtime;
configure them with `setup_fabric()` before registration.

## Isolate caches across model changes

`BatchCache` and `AsyncBatchCache` cache by input. Their keys do not automatically
include the model, instructions, response schema, reasoning, or other API options.
Reuse a cache only for the same operation and configuration. Create a new cache
when any of those change, including during model comparisons or rollback.

For persistent `DuckDBCacheBackend` data, use a separate table or database for
each model, task/schema version, and set of request options. For example:

```python
from openaivec._cache import BatchCache, DuckDBCacheBackend

backend = DuckDBCacheBackend.of(
    "responses.duckdb",
    table="sentiment_v1_gpt6_luna_none",
)
cache = BatchCache[str, str](cache=backend, batch_size=2)
try:
    sentiments = texts.ai.responses_with_cache(
        "Classify sentiment as positive, negative, or neutral.",
        cache=cache,
        reasoning={"effort": "none"},
    )
finally:
    backend.close()
```

Use a new table name if you change the prompt, schema, or any output-affecting
setting. This explicit naming is application-managed isolation, not automatic
versioning by the backend. Clear temporary caches when finished, and call
`aclose()` for async cache workers. Do not clear a persistent table that you
intend to retain for rollback.

## Validate and roll out

Evaluate a fixed sample against the previous model before processing a full
dataset. Include the languages and difficult cases used by your application.
Measure task quality, schema validation success, missing or duplicate rows,
validation corrections, latency, and total cost per successful row. Count
reasoning tokens and retries when comparing costs.

A bounded live regression check is available for bilingual sentiment, ordered
outputs, and explicit or omitted reasoning settings:

```bash
OPENAIVEC_TEST_RESPONSES_MODEL=gpt-6-luna uv run pytest tests/test_model_migration.py
```

The test requires API credentials and access to the selected model. It checks
request and output contracts; use application-specific labeled data to assess
quality and record latency and costs separately. Run the baseline with
`OPENAIVEC_TEST_RESPONSES_MODEL` set to the previous model; the test applies
reasoning settings only to compatible GPT-6 models.

Start with a small fixed batch and low concurrency, then tune batching against
your provider limits. Verify sync and async paths and the dataframe or SQL
integration you deploy. The retained benchmark notebooks contain historical
measurements and have not been rerun for GPT-6; they do not establish GPT-6
performance or quality. Example outputs from earlier runs are labeled as such.

To roll back, explicitly select your previous model before recreating wrappers
and UDFs. For the former OpenAI default:

```python
openaivec.set_responses_model("gpt-4.1-mini")

# Restore options supported by the previous model; omit GPT-6 reasoning settings.
sentiments = texts.ai.responses(
    "Classify sentiment as positive, negative, or neutral.",
    batch_size=2,
)
```

Restore the corresponding cache namespace and provider configuration as well.
For Azure, use your previous deployment name; for Fabric built-in models, retain
the runtime-supported configuration. Avoid combining cached outputs from the
new and previous model in the same comparison.
