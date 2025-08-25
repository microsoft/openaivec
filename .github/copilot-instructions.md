# Copilot Instructions – openaivec

Concise guide for generating code that fits this project’s architecture, performance model, style, and public API. Favor these rules over generic heuristics.

---

## 1. Purpose & Scope

Provide high‑throughput, batched access to OpenAI / Azure OpenAI Responses + Embeddings for pandas & Spark with strict ordering, deduplication, and structured outputs.

---

## 2. Public Surface (primary exports)

From `openaivec.__init__`:

- `BatchResponses`, `AsyncBatchResponses`
- `BatchEmbeddings`, `AsyncBatchEmbeddings`
- `PreparedTask`, `FewShotPromptBuilder`

Entry points:

- Pandas accessors: `Series.ai` / `Series.aio`
- Spark UDF builders in `spark.py`
- Structured tasks under `task/`

Azure note: Use deployment name as `model`. Standard Azure OpenAI configuration uses:
- Base URL: `https://YOUR-RESOURCE-NAME.services.ai.azure.com/openai/v1/`
- API Version: `"preview"`
Warn if base URL not v1. Behavior otherwise mirrors OpenAI.

---

## 3. Architecture Map (roles)

Underscore modules are internal (not exported). Public surface = `__init__`, `pandas_ext.py`, `spark.py`, and `task/`.

Core batching & optimization:

- `_proxy.py`: Order‑preserving dedup, caching, progressive mini‑batch execution, progress bars (only notebooks), dynamic batch sizing when `batch_size=None` via `_optimize.BatchSizeSuggester`; sync + async variants.
- `_optimize.py`: `BatchSizeSuggester` adaptive control loop (targets 30–60s batches) + metrics capture.

Model / task abstractions:

- `_model.py`: Typed wrappers (model names, task configs, response/embedding model name value objects).
- `_prompt.py`: Few‑shot / structured prompt assembly (`FewShotPromptBuilder`).
- `task/`: Pre‑packaged `PreparedTask` definitions for common workflows (re-exported publicly).

LLM interaction layers:

- `_responses.py`: Vectorized JSON‑mode wrapper (`BatchResponses` / `AsyncBatchResponses`); enforces same‑length contract; structured parse via `responses.parse`; reasoning model temperature guard & enhanced guidance warnings; retries with `backoff`.
- `_embeddings.py`: Embedding batching (`BatchEmbeddings` / `AsyncBatchEmbeddings`) returning `np.float32` arrays, de‑dup aware.
- `_schema.py`: Dynamic schema inference (`SchemaInferer`) producing Pydantic models at runtime; internal, not exported.

I/O & provider setup:

- `_provider.py`: Environment-driven auto detection (OpenAI vs Azure). Registers defaults, validates Azure v1 base URL, DI container root (`CONTAINER`).
- `_di.py`: Lightweight dependency injection container; registration & resolution helpers.

Utilities & cross‑cutting concerns:

- `_util.py`: `backoff` / `backoff_async`, `TextChunker` token-based splitter.
- `_serialize.py`: Pydantic (de)serialization and Spark schema bridging support.
- `_log.py`: Observation decorator used for tracing (`@observe`).

DataFrame / Spark integration:

- `pandas_ext.py`: `.ai` / `.aio` accessors (sync + async), shared cache variants, model configuration helpers (`responses_model`, `embeddings_model`, `use`, `use_async`). Maintains Series length/index; optional auto batch size; exposes reasoning temperature control.
- `spark.py`: Async UDF builders (`responses_udf`, `task_udf`, `embeddings_udf`, `count_tokens_udf`, `split_to_chunks_udf`, `similarity_udf`). Per-partition duplicate caching; Pydantic → Spark `StructType` conversion; concurrency per executor with `max_concurrency`.
- `spark.py`: Async UDF builders (`responses_udf`, `task_udf`, `embeddings_udf`, `count_tokens_udf`, `split_to_chunks_udf`, `similarity_udf` – cosine similarity on embedding vectors). Per-partition duplicate caching; Pydantic → Spark `StructType` conversion; concurrency per executor with `max_concurrency`.

Observability & progress:

- Progress bars only when `show_progress=True` AND notebook environment heuristics in `_proxy.py` pass.
- Adaptive batch suggestions recorded automatically around each unit API call.

Public exports (`__init__.py`): `BatchResponses`, `AsyncBatchResponses`, `BatchEmbeddings`, `AsyncBatchEmbeddings`, `PreparedTask`, `FewShotPromptBuilder`.

---

## 4. Core Principles & Contracts

1. Always batch via the Proxy; never per-item API loops.
2. map_func must return a list of identical length & order; mismatch => raise `ValueError` after releasing events (deadlock prevention).
3. Deduplicate inputs; restore original ordering in outputs.
4. Preserve pandas index & Spark schema deterministically.
5. Show progress only in notebooks and only if `show_progress=True`.
6. Reasoning models (o1/o3 families and similar) must use `temperature=None`.
7. Attach exponential backoff for transient RateLimit / 5xx errors.
8. Structured outputs (Pydantic) preferred over free-form JSON/text.

---

## 5. Batching Proxy Rules

- Same-length return invariant is critical (break = bug).
- Async variant enforces `max_concurrency` via semaphore.
- Shared caches (`*_with_cache`) enable cross-operation reuse; do not bypass them.
- Release all waiting events if an exception occurs (avoid deadlocks).
- Progress bars use `tqdm.auto`; only displayed if notebook heuristics pass AND `show_progress=True`.

---

## 6. Responses API Guidelines

- Use Responses JSON mode (`responses.parse`).
- Reasoning model safety: force `temperature=None`; provide clear error guidance.
- Favor small, reusable prompts enabling dedup benefits.
- Encourage Pydantic `response_format` for schema validation & Spark schema inference.

---

## 7. Embeddings Guidelines

- Return `np.ndarray` of dtype `float32`.
- Batch sizes typically larger than for Responses; keep order stable.
- Avoid per-item postprocessing—vector ops should stay batched.

---

## 8. pandas Extension Rules

- `.ai.responses` / `.ai.embeddings` preserve Series length & index.
- Async via `.aio.*` with configurable `batch_size` & `max_concurrency`.
- `*_with_cache` shares a passed proxy (promote reuse, minimal API calls).
- No hidden reindexing or sorting; user order is authoritative.

---

## 9. Spark UDF Rules

- Cache duplicates per partition (dict lookup) before remote calls.
- Convert Pydantic -> Spark StructType; treat `Enum`/`Literal` as `StringType`.
- Respect reasoning `temperature=None` rule.
- Provide chunking & token counting via helper UDFs.
- Avoid excessive nested structs—keep schemas shallow & ergonomic.

---

## 10. Provider / Azure Rules

- Auto-detect provider from env variables; deployment name = model for Azure.
- Standard Azure OpenAI configuration:
  - Base URL: `https://YOUR-RESOURCE-NAME.services.ai.azure.com/openai/v1/`
  - API Version: `"preview"`
  - Environment variables:
    ```bash
    export AZURE_OPENAI_API_KEY="your-azure-key"
    export AZURE_OPENAI_BASE_URL="https://YOUR-RESOURCE-NAME.services.ai.azure.com/openai/v1/"
    export AZURE_OPENAI_API_VERSION="preview"
    ```
- Warn (don't fail) if Azure base URL not v1 format; still proceed.
- Keep code paths unified; avoid forking logic unless behavior diverges.

---

## 11. Coding Standards

- Python ≥ 3.10; Ruff for lint/format (`line-length=120`).
- Absolute imports (except re-export patterns in `__init__.py`) – enforced by Ruff rule TID252.
- Modern typing syntax (Python 3.9+):
  - **Built-in generic types**: Use `list[T]`, `dict[K, V]`, `set[T]`, `tuple[T, ...]`, `type[T]` instead of `typing` equivalents
  - **Union types**: Use `|` syntax (`int | str | None`) instead of `Union[...]`
  - **Optional types**: Use `S | None` instead of `Optional[S]`
  - **Collections.abc**: Use `collections.abc.Callable`, `collections.abc.Awaitable`, `collections.abc.Iterator` instead of `typing` equivalents
- Prefer `@dataclass` for simple immutable-ish contracts; use Pydantic only for validation-boundaries.
- Raise narrow exceptions (`ValueError`, `TypeError`) on contract violations—avoid broad except.
- Public APIs: Google-style docstrings with return/raises sections.

---

## 12. Testing Strategy

Live-first philosophy: call real OpenAI / Azure endpoints when tests validate core contracts and remain fast. Use mocks only for: (a) forced transient errors, (b) rare fault paths, (c) deterministic pure utilities.

Key rules:

1. Skip (not fail) when credentials (`OPENAI_API_KEY` or Azure env) absent.
2. Keep prompts minimal; batch size 1–4 for speed & cost.
3. Assertions allow natural-language variance—focus on structure, ordering, lengths, types.
4. Test dedup, ordering, cache reuse, concurrency limits, reasoning temperature enforcement.
5. Inject retries by patching the smallest internal callable (not the whole client) for fault tests.
6. Mark heavier suites separately if needed (e.g., `@pytest.mark.heavy_live`).
7. Flake mitigation: broaden assertions (containment / regex / type+length) instead of pinning brittle verbatim strings.

---

## 13. Performance Guidance

- Responses batch size: 32–128 (default 128). Embeddings: 64–256.
- Async `max_concurrency`: typical 4–12 (tune per rate limits).
- Exploit dedup to collapse repeated prompts/inputs.
- Reuse caches across Series operations & Spark partitions.
- Avoid synchronous hotspots inside async loops (keep map_func lean).
- Automatic batch size mode targets ~30–60s per batch (`BatchSizeSuggester`).

---

## 14. Public / Internal Module Policy (`__all__`)

Public: `pandas_ext.py`, `spark.py`, everything under `task/`.
Internal: all underscore-prefixed modules; set `__all__ = []` explicitly.
Package exports: maintain alphabetical `__all__` in `__init__.py` for core classes (`BatchResponses`, etc.).
When adding public symbols: update `__all__`, docs (`docs/api/`), and examples if helpful.

Best practices:

1. Internal-only code never leaks via wildcard import.
2. Task modules export their primary callable/class.
3. Keep `__all__` diff minimal & alphabetized.

---

## 15. Documentation

- New APIs: add or update `docs/api/*.md`; brief runnable snippet preferred over prose.
- Add concise example notebooks only if they illustrate distinct usage (avoid overlap).
- Update `mkdocs.yml` nav for new pages.

---

## 16. PR Checklist

- [ ] Ruff check & format pass.
- [ ] Public API contracts (length/order/types) preserved.
- [ ] All remote calls batched (no per-item loops).
- [ ] Reasoning models enforce `temperature=None`.
- [ ] Tests updated/added: live where feasible; skip gracefully without credentials.
- [ ] Mock usage (if any) narrowly scoped & justified.
- [ ] Docs + `__all__` updated for new public symbols.
- [ ] Performance considerations (batch sizes, concurrency) sensible.

---

## 17. Common Snippets

New batched API wrapper (sync):

```python
@observe(_LOGGER)
@backoff(exceptions=[RateLimitError, InternalServerError], scale=1, max_retries=12)
def _unit_of_work(self, xs: list[str]) -> list[TOut]:
  resp = self.client.api(xs)
  return convert(resp)  # Same length/order

def create(self, inputs: list[str]) -> list[TOut]:
  return self.cache.map(inputs, self._unit_of_work)
```

Reasoning model temperature:

```python
# o1/o3 & similar reasoning models must set temperature None
temperature=None
```

pandas `.ai` with shared cache:

```python
from openaivec._proxy import BatchingMapProxy
shared = BatchingMapProxy[str, str](batch_size=64)
df["text"].ai.responses_with_cache("instructions", cache=shared)
```

Spark structured Responses UDF:

```python
from pydantic import BaseModel
from openaivec.spark import responses_udf

class R(BaseModel):
  value: str

udf = responses_udf(
  instructions="Do something",
  response_format=R,
  batch_size=64,
  max_concurrency=8,
)
```

Register custom OpenAI / Azure clients for pandas extension:

```python
from openai import OpenAI, AzureOpenAI, AsyncAzureOpenAI
from openaivec import pandas_ext

# OpenAI client
client = OpenAI(api_key="sk-...")
pandas_ext.use(client)

# Azure OpenAI sync
azure = AzureOpenAI(
  api_key="...",
  base_url="https://YOUR-RESOURCE-NAME.services.ai.azure.com/openai/v1/",
  api_version="preview",
)
pandas_ext.use(azure)

# Azure OpenAI async
azure_async = AsyncAzureOpenAI(
  api_key="...",
  base_url="https://YOUR-RESOURCE-NAME.services.ai.azure.com/openai/v1/",
  api_version="preview",
)
pandas_ext.use_async(azure_async)

// Override model names (optional)
pandas_ext.responses_model("gpt-4.1-mini")
pandas_ext.embeddings_model("text-embedding-3-small")
```

---

When unsure, inspect implementations (`_proxy.py`, `_responses.py`, `_embeddings.py`, `pandas_ext.py`, `spark.py`) and related tests. Keep suggestions minimal, batched, and structurally safe.

---

## 18. Dev Workflow Commands

Canonical local commands (uv-based). Prefer these in automation & docs.

Install (all extras + dev):

```bash
uv sync --all-extras --dev
```

Editable install (if needed by external tooling):

```bash
uv pip install -e .
```

Lint & format (Ruff):

```bash
uv run ruff check . --fix
uv run ruff format .
```

Run full test suite (quiet):

```bash
uv run pytest -q
```

Run a focused test:

```bash
uv run pytest tests/test_responses.py::test_reasoning_temperature_guard -q
```

Serve docs (MkDocs live reload):

```bash
uv run mkdocs serve
```

Environment setup notes:

- Set `OPENAI_API_KEY` or Azure trio (`AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_BASE_URL`, `AZURE_OPENAI_API_VERSION`).
- Standard Azure OpenAI configuration:
  - `AZURE_OPENAI_BASE_URL="https://YOUR-RESOURCE-NAME.services.ai.azure.com/openai/v1/"`
  - `AZURE_OPENAI_API_VERSION="preview"`
- Tests auto-skip live paths when credentials absent.
- Use separate shell profiles per provider if switching frequently.
- Azure canonical base URL must end with `/openai/v1/` (e.g. `https://YOUR-RESOURCE-NAME.services.ai.azure.com/openai/v1/`); non‑v1 forms emit a warning.
