# Repository Guidelines

## Project Layout
- `src/openaivec/`: batching core (`_proxy.py`, `_responses.py`, `_embeddings.py`), integrations (`pandas_ext.py`, `spark.py`), and tasks (`task/`); keep additions beside the APIs they extend.
- `tests/`: mirrors the source layout; use common pandas, Spark, and async fixtures.
- `docs/` holds MkDocs sources, `site/` generated pages, and `artifacts/` scratch assets kept out of releases.

## Core Components & Contracts
- Remote work goes through `BatchingMapProxy`/`AsyncBatchingMapProxy`; they dedupe inputs, require same-length outputs, release waiters on failure, and show progress only when `show_progress=True` in notebooks.
- `_responses.py` enforces reasoning rules: o1/o3-family models must use `temperature=None`, and structured scenarios pass a Pydantic `response_format`.
- Reuse caches from `*_with_cache` or Spark UDF builders per operation and clear them afterward to avoid large payloads.

## Development Workflow
- `uv sync --all-extras --dev` prepares extras and tooling; iterate with `uv run pytest -m "not slow and not requires_api"` before a full `uv run pytest`.
- `uv run ruff check . --fix` enforces style, `uv run pyright` guards API changes, and `uv build` validates the distribution.
- Use `uv pip install -e .` only when external tooling requires an editable install.

## Coding Standards
- Target Python 3.10+, rely on absolute imports, and keep helpers private with leading underscores; public modules publish alphabetical `__all__`, internal ones set `__all__ = []`.
- Apply Google-style docstrings with `(type)` Args, Returns/Raises sections, double-backtick literals, and doctest-style `Example:` blocks (`>>>`) when useful.
- Async helpers end with `_async`; dataframe accessors use descriptive nouns (`responses`, `extract`); raise narrow exceptions (`ValueError`, `TypeError`).

## Testing Guidelines
- Pytest discovers `tests/test_*.py`; parametrize to cover pandas vectorization, Spark UDFs, and async pathways.
- Mark network tests `@pytest.mark.requires_api`, long jobs `@pytest.mark.slow`, Spark flows `@pytest.mark.spark`; skip gracefully when credentials are missing.
- Add regression tests before fixes, assert on structure/length/order rather than verbatim text, and prefer shared fixtures over heavy mocking.

## Collaboration
- Commits follow `type(scope): summary` (e.g., `fix(pandas): guard empty batch`) and avoid merge commits within feature branches.
- Pull requests explain motivation, outline the solution, link issues, list doc updates, and include the latest `uv run pytest` and `uv run ruff check . --fix` output; attach screenshots for doc or tutorial changes.

## Environment & Secrets
- Export `OPENAI_API_KEY` or the Azure trio (`AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_BASE_URL`, `AZURE_OPENAI_API_VERSION`) before running `requires_api` tests; Azure endpoints must end with `/openai/v1/`.
- Keep local secrets under `artifacts/`, never commit credentials, and rely on CI-managed secrets when extending automation.
