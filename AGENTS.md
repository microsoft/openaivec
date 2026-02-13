# Repository Guidelines

## Project Layout
- `src/openaivec/`: core batched wrappers (`_responses.py`, `_embeddings.py`), batching/caching internals (`_cache/proxy.py`, `_cache/optimize.py`), provider/DI setup (`_provider.py`, `_di.py`), schema inference (`_schema/`), and integrations (`pandas_ext.py`, `spark.py`).
- `src/openaivec/task/`: function-style task factories by domain (`nlp/`, `customer_support/`, `table/`) plus registry plumbing in `_registry.py`.
- `tests/`: mirrors the source layout, including focused suites in `tests/_cache/` and `tests/_schema/`.
- `docs/` holds MkDocs sources, `site/` generated pages, and `artifacts/` scratch assets kept out of releases.

## Core Components & Contracts
- Remote batched execution goes through `BatchingMapProxy` / `AsyncBatchingMapProxy` in `openaivec._cache`; proxies dedupe inputs in-order, require same-length outputs, and release in-flight waiters on failure.
- `batch_size` behavior is shared across sync/async proxies: `None` enables `BatchSizeSuggester` auto-tuning (target ~30-60s per batch), positive values force fixed chunks, and `<= 0` processes all items in one call.
- Progress bars appear only when `show_progress=True` and the runtime is notebook-like.
- `_responses.py` / `_embeddings.py` are batched OpenAI wrappers with retry/backoff; structured outputs use Pydantic `response_format`, and `_responses.py` retries schema failures with validation feedback (`max_validation_retries`).
- `parse` helpers infer schema when `response_format=None`; pass explicit models when deterministic output shape is required.
- Reuse caches from `*_with_cache` helpers (or Spark UDF-local caches) per operation and clear them (`clear`/`aclose`) when finished to avoid unbounded cache growth.

## Task & Schema Conventions
- Export tasks as factory functions (for example `nlp.sentiment_analysis()`), not constant task instances.
- Each task module should define a `TASK_SPEC` entry for `task._registry`, and task response models should reject unknown fields (`ConfigDict(extra="forbid")`).
- Use `PreparedTask` for reusable instruction/schema pairs; it is immutable and intentionally does not store default API kwargs.

## Development Workflow
- `uv sync --all-extras --dev` prepares extras and tooling; iterate with `uv run pytest -m "not slow and not requires_api"` before a full `uv run pytest`.
- Run focused suites for touched subsystems when possible (for example `uv run pytest tests/_cache tests/_schema`).
- `uv run ruff check . --fix` enforces style, `uv run pyright` guards API changes, and `uv build` validates the distribution.
- Use `uv pip install -e .` only when external tooling requires an editable install.

## Coding Standards
- Target Python 3.10+, rely on absolute imports, and keep helpers private with leading underscores; expose symbols via explicit `__all__` (internal modules can keep `__all__ = []` unless specific exports are required).
- Apply Google-style docstrings with `(type)` Args, Returns/Raises sections, double-backtick literals, and doctest-style `Example:` blocks (`>>>`) when useful.
- Keep sync/async APIs behaviorally aligned (`.ai.*` vs `.aio.*`, `Batch*` vs `AsyncBatch*`), dataframe accessors descriptive (`responses`, `extract`, `fillna`), and raise narrow exceptions (`ValueError`, `TypeError`).

## Testing Guidelines
- Pytest discovers `tests/test_*.py`; parametrize to cover pandas vectorization, Spark UDFs, and async pathways.
- Use markers consistently: `@pytest.mark.requires_api`, `@pytest.mark.slow`, `@pytest.mark.spark`, `@pytest.mark.integration`, `@pytest.mark.asyncio`; skip gracefully when credentials or optional deps are missing.
- Add regression tests before fixes, assert on structure/length/order rather than verbatim text, and prefer shared fixtures over heavy mocking.

## Collaboration
- Commits follow `type(scope): summary` (e.g., `fix(pandas): guard empty batch`) and avoid merge commits within feature branches.
- Pull requests explain motivation, outline the solution, link issues, list doc updates, and include the latest `uv run pytest` and `uv run ruff check . --fix` output; attach screenshots for doc or tutorial changes.

## Environment & Secrets
- Auth precedence is `OPENAI_API_KEY` first, then Azure (`AZURE_OPENAI_BASE_URL` + `AZURE_OPENAI_API_VERSION`, with optional `AZURE_OPENAI_API_KEY` for API-key auth).
- Azure endpoints should end with `/openai/v1/` (legacy paths work but emit warnings).
- For Spark, call `setup` / `setup_azure` before registering UDFs so local and executor environments stay in sync.
- Keep local secrets under `artifacts/`, never commit credentials, and rely on CI-managed secrets when extending automation.
