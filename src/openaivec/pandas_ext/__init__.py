"""Pandas Series / DataFrame extension for OpenAI.

## Setup
```python
import openaivec
from openaivec import pandas_ext  # registers .ai / .aio accessors

# Option 1: Use environment variables (automatic detection)
# Set OPENAI_API_KEY or Azure OpenAI environment variables.
# No explicit setup needed - clients are automatically created

# Option 2: Register an existing OpenAI client instance
from openai import OpenAI
openaivec.set_client(OpenAI(api_key="your-api-key"))

# Option 3: Register an Azure OpenAI client instance
from openai import AzureOpenAI
openaivec.set_client(AzureOpenAI(
    api_key="your-azure-key",
    base_url="https://YOUR-RESOURCE-NAME.services.ai.azure.com/openai/v1/",
    api_version="v1"
))

# Set up model names (optional, defaults shown)
openaivec.set_responses_model("gpt-4.1-mini")
openaivec.set_embeddings_model("text-embedding-3-small")
```

This module provides `.ai` and `.aio` accessors for pandas Series and DataFrames
to easily interact with OpenAI APIs for tasks like generating responses or embeddings.
"""

# Re-export cache types used in tests via ``pandas_ext.AsyncBatchCache``
from openaivec._cache import AsyncBatchCache  # noqa: F401

# Re-export accessor classes and internal helpers so that existing code
# (tests, monkeypatching, spark.py) that references e.g.
# ``pandas_ext.OpenAIVecSeriesAccessor`` continues to work.
from openaivec.pandas_ext._common import _df_rows_to_json_series  # noqa: F401

# Re-export deprecated configuration helpers for backward compatibility.
# These emit DeprecationWarning; use openaivec.set_*/get_* instead.
from openaivec.pandas_ext._config import (  # noqa: F401
    get_async_client,
    get_client,
    get_embeddings_model,
    get_responses_model,
    set_async_client,
    set_client,
    set_embeddings_model,
    set_responses_model,
)
from openaivec.pandas_ext._dataframe_async import AsyncOpenAIVecDataFrameAccessor  # noqa: F401
from openaivec.pandas_ext._dataframe_sync import OpenAIVecDataFrameAccessor  # noqa: F401
from openaivec.pandas_ext._series_async import AsyncOpenAIVecSeriesAccessor  # noqa: F401
from openaivec.pandas_ext._series_sync import OpenAIVecSeriesAccessor  # noqa: F401

# Re-export fillna for monkeypatching in tests (``monkeypatch.setattr(pandas_ext, "fillna", ...)``)
from openaivec.task.table import fillna  # noqa: F401

__all__: list[str] = []
