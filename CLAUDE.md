# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is **openaivec**, a Python library that enables AI-powered text processing at scale for pandas DataFrames and Apache Spark. The project provides vectorized OpenAI API access, reducing latency and API costs through batch processing and automatic deduplication.

## Development Commands

### Dependencies and Environment
```bash
# Install all dependencies (development and optional extras)
uv sync --all-extras --dev

# Install package in development mode
uv pip install -e .
```

### Code Quality and Testing
```bash
# Run linting and formatting (check, auto-fix, and format)
uv run ruff check . --fix && uv run ruff format .

# Run tests
uv run pytest

# Run tests for a specific module
uv run pytest tests/test_[module_name].py
```

### Documentation
```bash
# Build and serve documentation locally (uses MkDocs)
uv run mkdocs serve

# Build documentation for deployment
uv run mkdocs build
```

## Architecture and Structure

### Core Components

- **BatchResponses/AsyncBatchResponses** (`src/openaivec/responses.py`): Vectorized API interface for OpenAI completions
- **BatchEmbeddings/AsyncBatchEmbeddings** (`src/openaivec/embeddings.py`): Vectorized embedding generation
- **pandas_ext** (`src/openaivec/pandas_ext/`): Pandas DataFrame/Series extensions with `.ai`/`.aio` accessors
- **spark** (`src/openaivec/spark.py`): Apache Spark UDF builders for distributed processing
- **task** (`src/openaivec/task/`): Pre-built task modules for NLP and customer support domains
- **di** (`src/openaivec/di.py`): Simple dependency injection container with singleton lifecycle management

### Key Architecture Patterns

1. **Vectorization**: All API calls are batched to reduce latency and costs
2. **Deduplication**: Automatic removal of duplicate inputs to minimize API usage
3. **Structured Output**: Pydantic model support for type-safe responses
4. **Extensibility**: Modular task system for domain-specific operations

### Package Structure
```
src/openaivec/
├── __init__.py              # Main exports (public API)
├── pandas_ext/              # Pandas integration (public, package)
│   ├── __init__.py          #   Re-exports public API, registers accessors
│   ├── _common.py           #   Shared helpers
│   ├── _config.py           #   Client/model configuration
│   ├── _series_sync.py      #   .ai Series accessor
│   ├── _dataframe_sync.py   #   .ai DataFrame accessor
│   ├── _series_async.py     #   .aio Series accessor
│   └── _dataframe_async.py  #   .aio DataFrame accessor
├── spark.py                 # Spark UDF builders (public)
├── _di.py                   # Dependency injection container (internal)
├── _embeddings.py           # Batch embedding processing (internal)
├── _log.py                  # Logging configuration (internal)
├── _model.py                # Task configuration models (internal)
├── _optimize.py             # Performance optimization (internal)
├── _prompt.py               # Few-shot prompt building (internal)
├── _provider.py             # Client provider and configuration (internal)
├── _proxy.py                # Batch processing proxies (internal)
├── _responses.py            # Batch response processing (internal)
├── _serialize.py            # Serialization utilities (internal)
├── _util.py                 # General utilities (internal)
└── task/                    # Pre-built task modules (public)
    ├── nlp/                 # NLP tasks (translation, sentiment, etc.)
    ├── customer_support/    # Customer support tasks
    └── table/               # Table-specific operations (fillna, etc.)
```

## Development Workflow

### Adding New Features

1. **Core API Changes**: Modify `_responses.py` or `_embeddings.py` for API-level features
2. **Pandas Integration**: Extend `pandas_ext/` sub-modules for DataFrame/Series functionality  
3. **Spark Integration**: Update `spark.py` for distributed processing features
4. **New Task Domains**: Add modules under `task/` following existing patterns
5. **Dependency Injection**: Modify `_di.py` for service lifecycle management

### Testing Strategy

- Unit tests in `tests/` mirror the `src/` structure
- Integration tests require `OPENAI_API_KEY` environment variable
- Test both sync and async variants where applicable
- Mock API responses for deterministic testing when needed
- Run specific module tests: `uv run pytest tests/test_[module_name].py -v`
- **Type Checking**: `uv run pyright src/openaivec` - enforced in CI with gradual improvement targets

### Documentation Updates

- API documentation is auto-generated from docstrings using `mkdocstrings`
- Examples in `docs/examples/` are Jupyter notebooks
- Update `mkdocs.yml` navigation when adding new modules or examples

### Code Style Guidelines

- **Docstrings**: Use Google-style docstrings for all public functions, classes, and methods
  - All Args sections must include type information in parentheses (e.g., `parameter_name (str): Description`)
  - Maintain consistency across all modules for better developer experience
- **Type Annotations**: Use modern Python type syntax throughout the codebase
  - **Built-in Generic Types**: Use built-in generic types instead of `typing` module equivalents:
    - `list[T]` instead of `List[T]`
    - `dict[K, V]` instead of `Dict[K, V]`
    - `set[T]` instead of `Set[T]`
    - `tuple[T, ...]` instead of `Tuple[T, ...]`
    - `type[T]` instead of `Type[T]`
  - **Union Types**: Use `|` union syntax instead of `Union[...]` (e.g., `int | str | None`)
  - **Optional Types**: **MUST use `S | None` instead of `Optional[S]`** - Always use the modern union syntax for optional types
  - **Collections.abc**: Use `collections.abc` for abstract types:
    - `collections.abc.Callable` instead of `typing.Callable`
    - `collections.abc.Awaitable` instead of `typing.Awaitable`
    - `collections.abc.Iterator` instead of `typing.Iterator`
  - Apply comprehensive type annotations to function signatures and variables
- **Comments**: Avoid adding comments unless explicitly requested
- **Classes**: Use `@dataclass` for all classes — including wrappers, backends, and caches. Every field must have an explicit type annotation. Use Pydantic only at validation boundaries (API responses, task models). Never create collaborators inside `__init__` / `__post_init__`; accept them as typed dataclass fields (dependency injection) so they can be swapped or mocked. Provide a `@classmethod of(...)` factory for convenient construction with sensible defaults.
- **Imports**: 
  - **MUST use absolute imports** - All imports within the project must use absolute paths (e.g., `from openaivec.responses import BatchResponses`)
  - Never use relative imports (e.g., `from .responses import ...` or `from ..task import ...`)
  - **Exception**: `__init__.py` files may use relative imports for re-exporting modules within the same package
  - This rule applies to all other modules including test files

## Important Implementation Notes

### OpenAI Client Configuration
- Supports both OpenAI and Azure OpenAI clients
- Default model: `gpt-4.1-mini` for responses, `text-embedding-3-small` for embeddings
- API key managed via `OPENAI_API_KEY` environment variable

### Azure OpenAI Standard Configuration
When using Azure OpenAI, always use the v1 API format:
- Base URL: `https://YOUR-RESOURCE-NAME.services.ai.azure.com/openai/v1/`
- API Version: `"preview"`
- Example environment variables:
  ```python
  os.environ["AZURE_OPENAI_API_KEY"] = "your-azure-key"
  os.environ["AZURE_OPENAI_BASE_URL"] = "https://YOUR-RESOURCE-NAME.services.ai.azure.com/openai/v1/"
  os.environ["AZURE_OPENAI_API_VERSION"] = "preview"
  ```

### Performance Considerations
- Batch processing reduces API calls by ~90% on typical datasets
- Automatic deduplication can save 50-90% of API costs
- Batch sizes are automatically optimized by default for optimal throughput

### Error Handling
- Graceful degradation when API limits are hit
- Proper serialization handling for complex data types
- Comprehensive logging for debugging batch operations

## Package Visibility Guidelines (`__all__`)

### Public API Modules
These modules are part of the public API and have appropriate `__all__` declarations:

- `pandas_ext/` - Pandas DataFrame/Series extensions with `.ai/.aio` accessors (package)
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