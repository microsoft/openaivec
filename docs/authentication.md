# Authentication

Choose a route before creating batch wrappers, caches, or UDFs. Client and model
configuration is local to the current Python process.

| Route | Configuration | Usage billing |
|---|---|---|
| Public OpenAI | `OPENAI_API_KEY` or an explicit `OpenAI` client | OpenAI |
| Your Azure OpenAI resource | Azure API key or an authorized Entra identity with `/openai/v1/` | Azure OpenAI |
| Fabric built-in models | `openaivec.setup_fabric()` inside a supported Fabric notebook | Fabric capacity |

Without explicit client registration, a usable `OPENAI_API_KEY` takes precedence.
Otherwise, `AZURE_OPENAI_BASE_URL` selects Azure: an Azure API key takes precedence
over a service-principal secret or `DefaultAzureCredential`. Set environment variables
before importing `openaivec`. Existing clients are cached; changing environment
variables does not reconfigure them.

## Fabric built-in models

Install or update the package in a Fabric notebook:

```python
%pip install -U openaivec
```

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
- The setup covers driver-local batch, pandas, and DuckDB operations. It does not
  propagate authentication to Spark executors. Do not send driver bearer tokens or
  live clients to executors.
- Only the documented Responses and Embeddings flows are covered here. Do not assume
  that other OpenAI features, such as stored responses or file uploads, are supported.
- Local tests verify request routing, headers, storage restrictions, ordering, and
  deduplication with simulated runtime helpers. They do not verify live Fabric token
  acquisition, tenant permissions, capacity availability, or inference.

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