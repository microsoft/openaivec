# Authentication

Choose a route before creating batch wrappers, caches, or UDFs. Client and model
configuration is local to the current Python process.

| Route | Configuration | Usage billing |
|---|---|---|
| Public OpenAI | `OPENAI_API_KEY` or an explicit `OpenAI` client | OpenAI |
| Your Azure OpenAI resource | Azure API key or an authorized Entra identity with `/openai/v1/` | Azure OpenAI |
| Fabric built-in models | `openaivec.setup_fabric()` for driver-local calls; `openaivec.spark_ext.setup_fabric(spark)` for Spark UDFs | Fabric capacity |

Without explicit client registration, a usable `OPENAI_API_KEY` takes precedence.
Otherwise, `AZURE_OPENAI_BASE_URL` selects Azure: an Azure API key takes precedence
over a service-principal secret or `DefaultAzureCredential`. Set environment variables
before importing `openaivec`. Existing clients are cached; changing environment
variables does not reconfigure them.

## Fabric built-in models

Install `openaivec` and runtime-compatible dependencies in a Fabric Environment,
publish it, attach it to the notebook, and start a new session. The same Environment
can be used for driver-local calls and [Spark UDFs](#spark-udfs).

Then configure the notebook driver:

```python
import pandas as pd
import openaivec
from openaivec import pandas_ext

openaivec.setup_fabric()

answers = pd.Series(["apple", "banana", "apple"]).ai.responses("Translate to French.")
vectors = pd.Series(["apple", "banana"]).ai.embeddings()
```

This is an explicit opt-in. Importing the package alone does not switch an existing
OpenAI or Azure configuration to Fabric. `setup_fabric()` replaces the default clients
and models, without modifying environment variables. Client construction and token
acquisition remain lazy. Authentication and routing are delegated to Fabric's SynapseML
HTTP client helpers and service discovery, not to `DefaultAzureCredential`.

Defaults, based on the Fabric documentation checked on September 10, 2026:

| Setting | Default |
|---|---|
| Responses model | `gpt-5.1` |
| Embeddings model | `text-embedding-ada-002` |
| Fabric API version | `2025-04-01-preview` |

For example, select another supported Responses model:

```python
openaivec.setup_fabric(responses_model="gpt-5-mini")
```

`embeddings_model` and `api_version` can also be overridden. Use only values supported
by your Fabric runtime and region. A model name here is a Fabric built-in model, not
the name of a deployment in your own Azure resource.

### Spark UDFs

Install only `openaivec==2.5.1` in a **Fabric Environment** and let the package's runtime
dependencies resolve during Full publication. Version 2.5.1 includes the dependency
fixes for this setup; 2.5.0 does not. Use the [Environment setup below](#validated-environment),
publish, attach the Environment to the notebook, and start a new
session. Fabric's `%pip` installs on the driver and executors, but is session-scoped
and disabled in pipeline runs by default; `!pip` installs only on the driver.
Do not install `openaivec[spark]` in Fabric: the platform supplies PySpark, SynapseML,
and NotebookUtils.

Use the Spark-specific setup before constructing UDFs, then register them for SQL:

```python
from openaivec.spark_ext import embeddings_udf, responses_udf, setup_fabric

setup_fabric(spark)

spark.udf.register("ai_translate", responses_udf("Translate to French.", batch_size=2, max_concurrency=1))
spark.udf.register("ai_embeddings", embeddings_udf(batch_size=2, max_concurrency=1))
texts = spark.createDataFrame([(0, "apple"), (1, "banana"), (2, "apple")], ["id", "text"])
texts.createOrReplaceTempView("example_texts")
spark.sql("SELECT id, ai_translate(text) AS translation FROM example_texts").show()
spark.sql("SELECT id, ai_embeddings(text) AS embedding FROM example_texts").show()
```

Register functions once per Spark session. They are also available to subsequent
`%%sql` cells in that session, not to the Lakehouse T-SQL analytics endpoint.
Importing `openaivec` does not choose authentication or register SQL names automatically.

This also configures driver-local operations. Spark UDFs capture only the API
version, model names, and their own options. Each partition creates a runtime-managed
async HTTP client inside its event loop, reuses it across Arrow batches, and closes
it at partition completion or failure. Driver tokens and live HTTP clients are not
serialized to workers. The runtime transport supplies authentication and service
routing; no custom token broadcast is needed.

`responses_udf`, `embeddings_udf`, `task_udf`, and `parse_udf` share this route.
Structured output schemas, input/output correspondence, and partition-local
deduplication are preserved. Recreate UDFs after changing configuration; existing
UDFs retain their captured settings. The Spark setup uses `responses_model_name`
and `embeddings_model_name`, matching the other Spark setup functions.

The [Fabric Spark example](examples/fabric_spark.ipynb) exercises all five UDF paths
across two partitions and writes its validation report to the attached Lakehouse.

### Validated Environment

Use a dedicated Runtime 1.3 Environment. Import
[fabric_environment.yml](examples/fabric_environment.yml) to set `openaivec==2.5.1`
as its sole External library entry. Remove any older openaivec wheel from Custom
libraries when switching to the PyPI release. Do not replace a shared Environment's
library list. Publish in **Full** mode, attach the Environment, and start a **new
session**. The package supplies its own dependency declarations; no individual
dependency pins or custom wheel are required for the release.

Before release, the dependency-complete candidate `2.5.1.dev1` was validated on
September 10, 2026 with **one custom wheel and zero external library entries**,
replacing the previous 15-entry definition. Version 2.5.1 has the same package code
and runtime dependency declarations as that candidate. The verified publication
resolved the wheel's dependencies for both the driver and workers without individual
package pins. Model availability and resolved dependencies can differ between tenants
and publication dates; the results below describe that candidate run.

| Component | Tested value |
|---|---|
| Fabric runtime | 1.3 |
| Spark | 3.5.5.5.4.20260807.1 |
| Python | 3.11.8 |
| Capacity and region | F64, Japan East |
| Loaded OpenAI SDK / aiohttp / httpx | 3.11.0 / 3.14.3 / 0.28.1 |
| Loaded NumPy / pandas / PyArrow | 1.26.4 / 3.0.5 / 25.0.1 |
| Loaded Pydantic | 2.13.5 |

The package now declares its direct `httpx`, `numpy`, `pydantic`, and
`typing-extensions` imports as runtime dependencies. The other runtime dependencies,
including pandas and PyArrow for vectorized UDFs, were already declared. The
`aiohttp>=3.10.0` requirement excludes the older transport missing
`aiohttp.SocketTimeoutError`: SDK 3.11.0 failed with the runtime's aiohttp 3.9.3,
but passed this validation with the resolved aiohttp 3.14.3. A global SDK 2.0.0 pin
is no longer needed for the tested workflow. One library entry still installs
transitive dependencies; it does not mean that only one Python package is present.

Driver and worker imports matched for the seven libraries listed above. Fabric
still retained older distribution metadata: pandas reported 2.1.4 through
`importlib.metadata` while importing 3.0.5, and aiohttp reported 3.9.3 while importing
3.14.3 with the timeout exception available. PyArrow showed the same kind of
discrepancy. A successful run does not certify a clean platform-wide `pip check`.
Compare loaded modules and UDF results, and do not delete Fabric-managed files.

The live test registered all five paths and invoked them through `spark.sql`:
string Responses, structured Responses, Embeddings, `task_udf`, and `parse_udf`.
Each returned six rows in two partitions with two-row Arrow batches. Row-ID
correspondence, partition-local duplicate results, nonzero 1,536-dimensional
embeddings, and a repeated SQL action passed. The fresh report is written to
`Files/openaivec-example/spark-sql-udfs.json`. No driver token or client was broadcast.

### Requirements and limitations

- Built-in models are in preview. Verify supported capacity, region, tenant settings,
  and any cross-region processing requirements with your Fabric administrator.
- No user-owned Azure OpenAI resource, API key, or service-principal secret is needed.
  Inference is charged to Fabric capacity; notebook Spark compute is billed separately.
- Responses use `store=False`. `store=True` and non-null `previous_response_id` are
  rejected before sending, including values passed through `extra_body`. Supply any
  conversation context explicitly in the input. These restrictions apply to direct
  client requests, batched calls, and automatic schema inference.
- Async methods require the runtime's `get_openai_httpx_async_client()` helper. If
  unavailable, resolving the async client raises a clear error; no synchronous
  network fallback is used. The sync helper is documented by Microsoft, while async
  helper availability must be checked in the actual runtime.
- `openaivec.setup_fabric()` alone covers driver-local batch, pandas, and DuckDB
  operations. Spark UDFs require `openaivec.spark_ext.setup_fabric(spark)` and the
  async runtime helper on workers. Do not send driver bearer tokens or live clients
  to executors.
- Only the documented Responses and Embeddings flows are covered here. Do not assume
  that other OpenAI features, such as stored responses or file uploads, are supported.
- Live validation covers the runtime and manual notebook execution described above.
  Long-running token refresh, scheduled execution identities, and continued reuse of
  the same worker process were not tested. Verify permissions and capacity in your
  own tenant. Local regression tests additionally cover request routing, credential
  isolation, storage restrictions, client cleanup, ordering, and deduplication with
  simulated runtime helpers.

Close any already-resolved clients before calling setup again or replacing them:
`openaivec.get_client().close()` and, if an async client was used,
`await openaivec.get_async_client().close()`. Existing batch wrappers and caches retain
their original client and model; recreate them after reconfiguration. Do not resolve
an unused async client solely to close it on runtimes without the async helper.

## Your own Azure OpenAI resource

Use `OpenAI` / `AsyncOpenAI` with a base URL ending in `/openai/v1/`. Both
`https://RESOURCE.openai.azure.com/openai/v1/` and
`https://RESOURCE.services.ai.azure.com/openai/v1/` are supported. This v1 route does
not require a dated API version. Model parameters refer to your deployment names.

For Entra authentication, grant the actual inference identity
`Cognitive Services OpenAI User` at the resource scope. Allow time for RBAC propagation.
Use a refreshable bearer-token provider with `https://ai.azure.com/.default`, rather
than storing a one-time access token in an environment variable. Scope and roles for
other Foundry APIs are separate contracts; do not assume a project endpoint and an
Azure OpenAI resource endpoint are interchangeable.

Outside Fabric, `DefaultAzureCredential` can use a supported managed/workload identity
or local developer login. In production, prefer an explicit identity appropriate to
your host rather than an ambiguous credential chain. Supply custom clients with
`openaivec.set_client()` and `openaivec.set_async_client()`. Use the async Azure Identity
credential and token-provider variants for async clients. Close custom credentials
as well as clients when their work is finished.

### Service principal with Key Vault in Fabric

Fabric does not automatically expose its current user or workspace identity through
`DefaultAzureCredential`. The documented `notebookutils.credentials.getToken` audiences
do not establish general Azure OpenAI token support. A Fabric workspace identity is
not automatically an Azure-host managed identity.

When a service-principal secret is required, keep it in Key Vault and never put it
in notebook source, output, parameters, or committed files. Grant the principal
`Cognitive Services OpenAI User` on the inference resource. Separately, grant the
identity reading the secret `Key Vault Secrets User` on an RBAC-enabled vault, or
equivalent permitted secret-read access.

`notebookutils.credentials.getSecret()` is documented to use current-user credentials.
For scheduled notebook activities, verify the configured execution identity and its
permissions. Workspace identity is not selected automatically merely because the
workspace has one.

Set the following before importing `openaivec`:

```python
import os

os.environ.pop("OPENAI_API_KEY", None)
os.environ.pop("AZURE_OPENAI_API_KEY", None)
os.environ["AZURE_OPENAI_BASE_URL"] = "https://YOUR-RESOURCE.openai.azure.com/openai/v1/"
os.environ["AZURE_TENANT_ID"] = "YOUR-TENANT-ID"
os.environ["AZURE_CLIENT_ID"] = "YOUR-CLIENT-ID"
os.environ["KEY_VAULT_URL"] = "https://YOUR-KEYVAULT.vault.azure.net/"
os.environ["KEY_VAULT_SECRET_NAME"] = "YOUR-SECRET-NAME"

import openaivec

openaivec.set_responses_model("YOUR-RESPONSES-DEPLOYMENT")
openaivec.set_embeddings_model("YOUR-EMBEDDINGS-DEPLOYMENT")
```

Do not call `setup_fabric()` for this route. The secret is read only when an Entra
client is resolved, not at import time and not when an API key is selected. An explicit
`AZURE_CLIENT_SECRET` takes precedence over Key Vault retrieval. Both the tenant ID
and client ID are required when a secret is supplied. Incomplete Key Vault settings,
an empty retrieved secret, or a retrieval failure stop authentication instead of
silently selecting another identity. Dependency-injection failures are reported as
`ProviderError` with the underlying error retained as the cause.

## Official references

- [Fabric OpenAI Python SDK and Responses support](https://learn.microsoft.com/en-us/fabric/data-science/ai-services/how-to-use-openai-python-sdk)
- [Fabric model availability and prerequisites](https://learn.microsoft.com/en-us/fabric/data-science/ai-services/ai-services-overview)
- [Fabric service discovery and REST API](https://learn.microsoft.com/en-us/fabric/data-science/ai-services/how-to-use-openai-via-rest-api)
- [Azure OpenAI v1 and refreshable Entra tokens](https://learn.microsoft.com/en-us/azure/foundry/openai/api-version-lifecycle)
- [Azure OpenAI Entra authentication and RBAC](https://learn.microsoft.com/en-us/azure/foundry-classic/openai/how-to/managed-identity)
- [NotebookUtils credentials](https://learn.microsoft.com/en-us/fabric/data-engineering/notebookutils/notebookutils-credentials)
- [Fabric notebook activity execution identity](https://learn.microsoft.com/en-us/fabric/data-factory/notebook-activity)
- [Fabric notebook library management](https://learn.microsoft.com/en-us/fabric/data-engineering/library-management)
- [Fabric Environment libraries and publish modes](https://learn.microsoft.com/en-us/fabric/data-engineering/environment-manage-library)