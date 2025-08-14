# Copilot instructions for openaivec

This repository-wide guide tells GitHub Copilot how to propose code that fits our architecture, APIs, style, tests, and docs. Prefer these rules when completing or generating code.

## Project overview

- Goal: Provide a vectorized (batched) interface to OpenAI/Azure OpenAI so pandas/Spark can process large text corpora with high throughput.
- Public API exports (`src/openaivec/__init__.py`):
  - Responses: `BatchResponses`, `AsyncBatchResponses`
  - Embeddings: `BatchEmbeddings`, `AsyncBatchEmbeddings`
- First-class pandas extensions (`.ai` / `.aio`) and Spark UDF builders
- Azure OpenAI is supported with the same APIs (use deployment name as the “model” for Azure)

## Architecture and roles

- `src/openaivec/_proxy.py` (internal)
  - Core batching, deduplication, order preservation, and caching
  - `BatchingMapProxy[S, T]` (sync) / `AsyncBatchingMapProxy[S, T]` (async)
  - The map_func contract is strict: return a list of the same length and order as the inputs
  - Progress bars only in notebook environments via `tqdm.auto`, gated by `show_progress=True`
- `src/openaivec/_responses.py` (internal)
  - Batched wrapper over OpenAI Responses JSON-mode API
  - `BatchResponses` / `AsyncBatchResponses` use the proxy internally
  - Retries via `backoff`/`backoff_async` for transient errors (RateLimit, 5xx)
  - Reasoning models (o1/o3 family) must use `temperature=None`; helpful guidance on errors
- `src/openaivec/_embeddings.py` (internal)
  - Batched embeddings (sync/async)
- `src/openaivec/pandas_ext.py` (public)
  - `Series.ai` / `Series.aio` entry points for responses/embeddings
  - Uses DI container (`_provider.CONTAINER`) to get client and model names
  - Supports batch size, progress, and cache sharing (`*_with_cache`)
- `src/openaivec/spark.py` (public)
  - UDF builders: `responses_udf` / `task_udf` / `embeddings_udf` / `count_tokens_udf` / `split_to_chunks_udf`
  - Per-partition duplicate caching to reduce API calls
  - Pydantic → Spark StructType schema conversion
- `src/openaivec/_provider.py` (internal)
  - DI container and automatic OpenAI/Azure OpenAI client provisioning
  - Warns if Azure base URL isn't v1 format
- `src/openaivec/_util.py` (internal)
  - `backoff` / `backoff_async` and `TextChunker`
- Additional internal modules
  - `src/openaivec/_di.py`: lightweight DI container
  - `src/openaivec/_log.py`: logging/observe helpers
  - `src/openaivec/_prompt.py`: few-shot prompt building
  - `src/openaivec/_serialize.py`: Pydantic schema (de)serialization
  - `src/openaivec/_model.py`: task configuration models
  - `src/openaivec/_optimize.py`: performance optimization
- `src/openaivec/task/` (public)
  - Pre-built, structured task library

## Dev commands (uv)

```bash
# Install all dependencies (dev + extras)
uv sync --all-extras --dev

# Install in editable mode
uv pip install -e .

# Lint and format
uv run ruff check . --fix && uv run ruff format .

# Run tests
uv run pytest

# Build/serve docs
uv run mkdocs serve
```

## Coding standards (Ruff/types/style)

- Python ≥ 3.10
- Lint/format via Ruff (line-length=120, target=py310)
- Imports: absolute only (enforced by TID252), except `__init__.py` may re-export relatively
- Type hints required using modern syntax (`str | None` over `Optional[str]`)
- Public APIs should document return values and exceptions (Google-style docstrings preferred)
- Favor `@dataclass` for simple data contracts; separate mutable state cleanly
- Don’t swallow errors broadly; raise `ValueError` etc. on contract violations

## API contracts and critical rules

- Proxy (`_proxy.py` - BatchingMapProxy / AsyncBatchingMapProxy)
  - map_func must return a list with the same length and order as inputs; on mismatch, release events and raise ValueError
  - Inputs are de-duplicated while preserving first-occurrence order; outputs are restored to the original order
  - Progress is only shown in notebooks when `show_progress=True`
  - Async version enforces `max_concurrency` via `asyncio.Semaphore`
- Responses (`_responses.py`)
  - Use OpenAI Responses JSON mode (`responses.parse`)
  - For reasoning models (o1/o3 families), you MUST set `temperature=None`; helpful error messaging is built-in
  - Strongly prefer structured outputs with Pydantic models
  - Retries with exponential backoff for RateLimit/5xx
- Embeddings (`_embeddings.py`)
  - Return NumPy float32 arrays
- pandas extensions
  - `.ai.responses` / `.ai.embeddings` strictly preserve Series index and length
  - `.aio` provides async variants; tune with `max_concurrency` and `batch_size`
  - `*_with_cache` variants let callers share external caches across ops
- Spark UDFs
  - Cache duplicates within each partition to minimize API cost
  - Convert Pydantic models to Spark schemas; treat Enum/Literal as strings
  - Reasoning models require `temperature=None`
  - Provide token counting and text chunking helpers
- Provider/DI and Azure (`_provider.py` / `_di.py`)
  - Auto-detect OpenAI vs Azure OpenAI from env vars
  - Azure requires v1 base URL (warn otherwise) and uses deployment name as the "model"

## Preferred patterns (Do) and Avoid (Don’t)

Do

- Batch through the Proxy rather than per-item loops
- Attach `backoff`/`backoff_async` to external API calls (RateLimit, 5xx)
- Preserve index/order/schema for pandas/Spark APIs
- Clarify Azure specifics (“deployment name” vs “model name”); respect `_check_azure_v1_api_url`
- When changing public APIs, update `__all__` and docs in `docs/`

Don’t

- Break the Proxy contract (same-length, ordered result)
- Fire one API request per item—always batch via the Proxy
- Show progress outside notebook contexts or ignore `show_progress`
- Use relative imports (except `__init__.py` re-exports)
- Hit real external APIs in unit tests (prefer mocks/stubs)

## Performance guidance

- Typical batch size ranges: Responses 32–128, Embeddings 64–256 (defaults are 128 in code)
- Async `max_concurrency` commonly 4–12 per process/partition; scale with rate limits in mind
- Partition-level caching (Spark) and cross-op cache sharing (pandas `*_with_cache`) greatly reduce costs

## Testing strategy (pytest)

- Tests live in `tests/`; cover both sync and async where applicable
- Prefer mocks/stubs for external API calls; keep data small and deterministic
- Focus areas:
  - Order/length preservation
  - Deduplication and cache reuse
  - Event release on exceptions (deadlock prevention)
  - `max_concurrency` is not exceeded
  - Reasoning model guidance (`temperature=None`)
- Use `asyncio.run` in async tests (mirrors existing tests)
- Optional integration tests can run with valid API keys; keep unit tests independent of network

## Package Visibility Guidelines (`__all__`)

### Public API Modules
These modules are part of the public API and have appropriate `__all__` declarations:

- `pandas_ext.py` - Pandas DataFrame/Series extensions with `.ai/.aio` accessors
- `spark.py` - Apache Spark UDF builders for distributed processing
- `task/*` - All task modules (NLP, customer support, table operations)

### Internal Modules (underscore-prefixed)
These modules are for internal use only and have `__all__ = []`:

- `_embeddings.py` - Batch embedding processing (internal implementation)
- `_model.py` - Task configuration models (internal types)
- `_prompt.py` - Few-shot prompt building (internal implementation)  
- `_responses.py` - Batch response processing (internal implementation)
- `_util.py`, `_serialize.py`, `_log.py`, `_provider.py`, `_proxy.py`, `_di.py`, `_optimize.py` - Internal utilities

### Main Package API
Users access core functionality through `__init__.py` exports:
- `BatchResponses`, `AsyncBatchResponses`
- `BatchEmbeddings`, `AsyncBatchEmbeddings` 
- `PreparedTask`, `FewShotPromptBuilder`

### `__all__` Best Practices

1. **Public modules**: Include all classes, functions, and constants intended for external use
2. **Internal modules**: Use `__all__ = []` to explicitly mark as internal-only
3. **Task modules**: Each task module should export its main classes/functions
4. **Package `__init__.py`**: Re-export public API from all public modules
5. **Consistency**: Maintain alphabetical ordering within `__all__` lists

## Documentation (MkDocs)

- For new developer-facing APIs, update `docs/api/` and consider a short example under `docs/examples/`
- Keep pandas/Spark examples concise to minimize learning curve
- Update `mkdocs.yml` navigation when adding modules or examples

## PR checklist (pre-merge)

- [ ] Ruff check/format passes (line-length 120, absolute imports)
- [ ] Public API contracts (order/length/types) are satisfied
- [ ] Large-scale processing is batched via the Proxy
- [ ] Reasoning models use `temperature=None` where applicable
- [ ] Tests added/updated without calling live external APIs
- [ ] Docs updated if needed (`docs/` and/or examples)

## Common snippets (what to suggest)

- New batched API wrapper (sync)

  ```python
  @observe(_LOGGER)
  @backoff(exceptions=[RateLimitError, InternalServerError], scale=1, max_retries=12)
  def _unit_of_work(self, xs: list[str]) -> list[TOut]:
      resp = self.client.api(xs)  # real API call
      return convert(resp)        # same length/order as xs

  def create(self, inputs: list[str]) -> list[TOut]:
      return self.cache.map(inputs, self._unit_of_work)
  ```

- Reasoning model temperature
  ```python
  # o1/o3 and similar reasoning models must use None
  temperature=None
  ```
- pandas `.ai` with shared cache

  ```python
  from openaivec._proxy import BatchingMapProxy

  shared = BatchingMapProxy[str, str](batch_size=64)
  df["text"].ai.responses_with_cache("instructions", cache=shared)
  ```

- Spark UDF (structured output)

  ```python
  from pydantic import BaseModel
  from openaivec.spark import responses_udf

  class R(BaseModel):
      value: str

  udf = responses_udf("do something", response_format=R, batch_size=64, max_concurrency=8)
  ```

---

By following this guide, Copilot suggestions will match the repository’s design, performance goals, and testing standards. When in doubt, read the implementations in `proxy.py`, `responses.py`, `pandas_ext.py`, and `spark.py`, and the tests under `tests/`.
