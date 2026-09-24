# Authentication recovery

Read this reference when configuration is missing or a remote smoke test fails
with an authentication or authorization error.

## Conversation contract

Stop before processing user data and say which route is missing or failed.
For a nontechnical user, do not begin by asking them to choose an
authentication protocol. Ask one question about where the work is running,
using a single-select UI with a free-text alternative:

1. A company Microsoft Fabric notebook.
2. A company-managed Azure AI environment.
3. An OpenAI account or team subscription.
4. Not sure; I need help from an administrator.

If the harness automatically provides an **Other** text box, do not duplicate
it. Explain the corresponding route only after the user answers:

1. Fabric built-in models when already in a supported Fabric notebook.
2. Azure OpenAI with company sign-in (Entra ID), or an Azure API key when the
   administrator specifically provides one.
3. An OpenAI API key stored in the user's approved secret store.
4. No remote run until an administrator confirms an approved route.

Ask the user to configure credentials in their local environment or secret
store and tell you when it is ready. Never ask them to paste a credential into
chat, place it in source code, commit it, print it, or include it in a SQL
statement. Do not silently fall back to another route.

When a supported Fabric notebook is already detected, recommend its built-in
route because no separate API key is needed. When an approved Azure endpoint
and company identity are already configured, recommend Entra ID. Otherwise do
not imply that a free or anonymous fallback exists.

Environment variables must be configured before importing `openaivec`.
Resolved clients are cached in the current Python process. After changing
variables, start a new process or explicitly register new public clients.

## OpenAI API key

Configure the key in the process environment:

```bash
export OPENAI_API_KEY="<set this in your secret store>"
unset AZURE_OPENAI_API_KEY
unset AZURE_OPENAI_BASE_URL
```

`OPENAI_API_KEY` has precedence over every Azure route. If it is accidentally
set while Azure was intended, remove it and restart the process.

The default Responses model is `gpt-6-luna`; the default embedding model is
`text-embedding-3-small`. Override them before UDF registration when needed:

```python
import openaivec

openaivec.set_responses_model("gpt-6-luna")
openaivec.set_embeddings_model("text-embedding-3-small")
```

## Azure OpenAI with an API key

Remove `OPENAI_API_KEY`, then configure the v1 endpoint and Azure key:

```bash
unset OPENAI_API_KEY
export AZURE_OPENAI_BASE_URL="https://YOUR-RESOURCE.services.ai.azure.com/openai/v1/"
export AZURE_OPENAI_API_KEY="<set this in your secret store>"
```

The base URL should end with `/openai/v1/`. Use the Azure deployment name as
`model_name` on the UDF or register it before creating the UDF:

```python
import openaivec

openaivec.set_responses_model("YOUR-RESPONSES-DEPLOYMENT")
openaivec.set_embeddings_model("YOUR-EMBEDDINGS-DEPLOYMENT")
```

## Azure OpenAI with Entra ID

Remove both API keys and retain the Azure v1 endpoint:

```bash
unset OPENAI_API_KEY
unset AZURE_OPENAI_API_KEY
export AZURE_OPENAI_BASE_URL="https://YOUR-RESOURCE.services.ai.azure.com/openai/v1/"
```

For local development, authenticate an identity available to
`DefaultAzureCredential`, for example with Azure CLI:

```bash
az login
```

For a service principal, set all three values through the environment's secret
management:

```bash
export AZURE_TENANT_ID="<tenant id>"
export AZURE_CLIENT_ID="<client id>"
export AZURE_CLIENT_SECRET="<set this in your secret store>"
```

The selected identity must already have permission to invoke the Azure OpenAI
deployment. Role assignment and resource provisioning are outside this skill;
ask the user's Azure administrator to grant the appropriate data-plane access.

## Fabric built-in models

Only use this route in a supported Fabric notebook runtime:

```python
import openaivec

openaivec.setup_fabric()
```

Call it before registering DuckDB UDFs. No OpenAI or Azure OpenAI API key is
required; usage is billed to Fabric capacity. This is driver-local DuckDB
processing, not Spark executor setup.

## Custom clients

Environment configuration is preferred. If the application owns explicit
clients, register the async client used by DuckDB UDFs. Register the sync client
too when schema inference may run:

```python
import openaivec
from openai import AsyncOpenAI, OpenAI

openaivec.set_client(OpenAI())
openaivec.set_async_client(AsyncOpenAI())
```

Do not close caller-owned clients inside the skill workflow.

## Safe smoke test

After configuration, use synthetic text and one request before any user data:

```python
import duckdb
from openaivec.duckdb_ext import responses_udf

with duckdb.connect() as conn:
    responses_udf(
        conn,
        "auth_smoke",
        instructions="Return only OK.",
        batch_size=1,
        max_concurrency=1,
        reasoning={"effort": "none"},
    )
    print(conn.sql("SELECT auth_smoke('ping')").fetchone())
```

This verifies the credential, endpoint, model/deployment, and network path. It
is a billable API request.

## Error routing

| Symptom | Guidance |
| --- | --- |
| `No valid OpenAI or Azure OpenAI credentials found` | Configure one route above before import, then restart. |
| HTTP 401 / authentication error | Replace or refresh the selected route's credential; do not try another provider automatically. |
| HTTP 403 / permission denied | The identity is known but lacks data-plane permission. Ask the resource administrator. |
| HTTP 404 / model not found | Check the v1 base URL and model or Azure deployment name. |
| HTTP 429 / rate limit | Authentication succeeded. Reduce `max_concurrency`, wait for quota, or use a bounded `RetryPolicy`. |
| Timeout / connection error | Verify network access and endpoint; do not describe it as a credential failure without evidence. |
