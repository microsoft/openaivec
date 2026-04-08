"""Pandas Series / DataFrame extension for OpenAI.

## Setup
```python
from openai import OpenAI, AzureOpenAI, AsyncOpenAI, AsyncAzureOpenAI
from openaivec import pandas_ext

# Option 1: Use environment variables (automatic detection)
# Set OPENAI_API_KEY or Azure OpenAI environment variables.
# For Azure OpenAI, set AZURE_OPENAI_BASE_URL and AZURE_OPENAI_API_VERSION.
# Authentication uses AZURE_OPENAI_API_KEY when present, otherwise Entra ID
# via DefaultAzureCredential.
# No explicit setup needed - clients are automatically created

# Option 2: Register an existing OpenAI client instance
client = OpenAI(api_key="your-api-key")
pandas_ext.set_client(client)

# Option 3: Register an Azure OpenAI client instance
azure_client = AzureOpenAI(
    api_key="your-azure-key",
    base_url="https://YOUR-RESOURCE-NAME.services.ai.azure.com/openai/v1/",
    api_version="v1"
)
pandas_ext.set_client(azure_client)

# Option 4: Register an async Azure OpenAI client instance
async_azure_client = AsyncAzureOpenAI(
    api_key="your-azure-key",
    base_url="https://YOUR-RESOURCE-NAME.services.ai.azure.com/openai/v1/",
    api_version="v1"
)
pandas_ext.set_async_client(async_azure_client)

# Set up model names (optional, defaults shown)
pandas_ext.set_responses_model("gpt-4.1-mini")
pandas_ext.set_embeddings_model("text-embedding-3-small")

# Inspect current configuration
configured_model = pandas_ext.get_responses_model()
```

This module provides `.ai` and `.aio` accessors for pandas Series and DataFrames
to easily interact with OpenAI APIs for tasks like generating responses or embeddings.
"""

# Re-export public configuration helpers
# Re-export cache types used in tests via ``pandas_ext.AsyncBatchingMapProxy``
from openaivec._cache import AsyncBatchingMapProxy  # noqa: F401

# Re-export accessor classes and internal helpers so that existing code
# (tests, monkeypatching, spark.py) that references e.g.
# ``pandas_ext.OpenAIVecSeriesAccessor`` continues to work.
from openaivec.pandas_ext._common import _df_rows_to_json_series  # noqa: F401
from openaivec.pandas_ext._config import (
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

__all__ = [
    "get_async_client",
    "get_client",
    "get_embeddings_model",
    "get_responses_model",
    "set_async_client",
    "set_client",
    "set_embeddings_model",
    "set_responses_model",
]
