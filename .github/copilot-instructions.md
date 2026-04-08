# Copilot Instructions – openaivec

High-throughput, batched OpenAI / Azure OpenAI Responses + Embeddings for pandas & Spark with strict ordering, deduplication, and structured outputs.

---

## Dev Commands

All tooling runs via **uv**. Python ≥ 3.10.

```bash
uv sync --all-extras --dev          # install deps + dev tools
uv run ruff check . --fix           # lint (auto-fix)
uv run ruff format .                # format
uv run pyright src/openaivec        # type check
uv run pytest -q                    # full test suite
uv run pytest tests/test_responses.py::test_reasoning_temperature_guard -q  # single test
uv run pytest -m "not slow and not requires_api"  # fast iteration without API keys
uv run mkdocs serve                 # local docs
uv build                            # validate distribution
```

Environment: set `OPENAI_API_KEY`, or for Azure set `AZURE_OPENAI_API_KEY` + `AZURE_OPENAI_BASE_URL` (must end with `/openai/v1/`) + `AZURE_OPENAI_API_VERSION` (`"preview"`). Tests auto-skip when credentials are absent.

---

## Architecture

### Public surface

| Layer | Entry point | Notes |
|-------|------------|-------|
| Package exports | `openaivec.__init__` | `BatchResponses`, `AsyncBatchResponses`, `BatchEmbeddings`, `AsyncBatchEmbeddings`, `PreparedTask`, `FewShotPromptBuilder`, `FewShotPrompt`, `SchemaInferer`, `SchemaInferenceInput`, `SchemaInferenceOutput` |
| pandas accessors | `Series.ai` / `Series.aio` | Sync + async; registered by importing `openaivec.pandas_ext` |
| Spark UDFs | `openaivec.spark` | `responses_udf`, `task_udf`, `embeddings_udf`, `count_tokens_udf`, `split_to_chunks_udf`, `similarity_udf`, `parse_udf`, `infer_schema` |
| Task factories | `openaivec.task.nlp`, `.customer_support`, `.table` | Call as functions: `nlp.sentiment_analysis()`, not constants |

### Data flow

```
User input (list / Series / Spark column)
  → BatchCache (_cache/proxy.py)    # dedup, order-preserve, mini-batch
    → _responses.py / _embeddings.py      # OpenAI API with backoff
  → results reassembled in original order
```

`BatchCache` / `AsyncBatchCache` in `_cache/` are the core execution engine. They deduplicate inputs, chunk into batches, call a `map_func`, and restore original ordering. The `map_func` **must** return a list of identical length and order — a mismatch raises `ValueError` after releasing in-flight waiters (deadlock prevention).

`batch_size=None` enables `BatchSizeSuggester` (`_cache/optimize.py`) auto-tuning that targets ~30–60s per batch. Positive values force fixed chunks; `<= 0` processes everything in one call.

### Internal vs public boundary

Underscore-prefixed modules (`_responses.py`, `_cache/`, `_schema/`, `_di.py`, etc.) are internal — set `__all__ = []`. Public modules: `pandas_ext/`, `spark.py`, `task/`, and `__init__.py`.

---

## Key Contracts

1. **Batch everything** — all remote calls go through the proxy; never per-item API loops.
2. **Same-length invariant** — `map_func` output must match input length and order exactly.
3. **Dedup + restore** — duplicate inputs are collapsed; outputs are expanded back to original positions.
4. **Preserve pandas index / Spark schema** — no hidden reindexing or sorting.
5. **Reasoning models** (o1/o3 families) — must set `temperature=None`.
6. **Exponential backoff** — `@backoff` / `@backoff_async` for `RateLimitError` / `InternalServerError`, max 12 retries.
7. **Structured outputs preferred** — use Pydantic `response_format` over free-form JSON/text.
8. **Progress bars** — only in notebooks and only when `show_progress=True`.

---

## Coding Conventions

- **Ruff** lint + format, `line-length=120`, target `py310`.
- **Absolute imports only** — enforced by Ruff TID252. Exception: `__init__.py` may use relative imports for re-exports.
- **Modern typing** — `list[T]`, `dict[K, V]`, `X | None` (not `Optional`), `collections.abc.Callable` (not `typing.Callable`).
- **`@dataclass`** for all classes — including wrappers, backends, and caches. Every field must have an explicit type annotation. Use **Pydantic** only at validation boundaries (API responses, task models).
- **Dependency injection** — never create collaborators inside `__init__` / `__post_init__`. Accept them as typed dataclass fields so they can be swapped or mocked. Provide a `@classmethod of(...)` factory for convenient construction with sensible defaults.
- **Narrow exceptions** — `ValueError`, `TypeError` on contract violations; no broad `except`.
- **Google-style docstrings** — Args with `(type)` annotations, Returns/Raises sections.
- **`__all__`** — alphabetized in every module. Internal modules: `__all__ = []`. New public symbols → update `__all__`, `docs/api/`, and `mkdocs.yml`.
- **No comments** unless essential for clarification.

---

## Testing

Pytest with markers defined in `pytest.ini`: `requires_api`, `slow`, `spark`, `integration`, `asyncio`.

- **Live-first** — call real OpenAI endpoints for core contract tests. Use mocks only for forced transient errors, rare fault paths, or deterministic utilities.
- **Skip gracefully** — `pytest.skip()` when credentials absent; never fail.
- **Keep prompts minimal** — batch sizes 1–4 for speed and cost.
- **Assert structure, not text** — check types, lengths, ordering, containment; avoid pinning verbatim LLM output.
- **Patch narrowly** — mock the smallest internal callable, not the whole client.
- Task response Pydantic models use `ConfigDict(extra="forbid")`.

---

## Commits & PRs

- Commit format: `type(scope): summary` (e.g., `fix(pandas): guard empty batch`).
- PRs: explain motivation, link issues, include `uv run pytest` + `uv run ruff check . --fix` output.

---

## Snippets

### pandas client setup

```python
import openaivec
from openai import OpenAI, AzureOpenAI, AsyncAzureOpenAI
from openaivec import pandas_ext

openaivec.set_client(OpenAI(api_key="sk-..."))

# Azure
openaivec.set_client(AzureOpenAI(
    api_key="...",
    base_url="https://YOUR-RESOURCE.services.ai.azure.com/openai/v1/",
    api_version="preview",
))
openaivec.set_async_client(AsyncAzureOpenAI(...))

# Override default models
openaivec.set_responses_model("gpt-4.1-mini")
openaivec.set_embeddings_model("text-embedding-3-small")
```

### Shared cache across operations

```python
from openaivec._cache import BatchCache
shared = BatchCache[str, str](batch_size=64)
df["text"].ai.responses_with_cache("instructions", cache=shared)
```

### Spark UDF with structured output

```python
from pydantic import BaseModel
from openaivec.spark import responses_udf

class Result(BaseModel):
    value: str

udf = responses_udf(
    instructions="Do something",
    response_format=Result,
    batch_size=64,
    max_concurrency=8,
)
```

### Adding a batched API wrapper

```python
@observe(_LOGGER)
@backoff(exceptions=[RateLimitError, InternalServerError], scale=1, max_retries=12)
def _unit_of_work(self, xs: list[str]) -> list[TOut]:
    resp = self.client.api(xs)
    return convert(resp)  # must be same length/order as xs
```
