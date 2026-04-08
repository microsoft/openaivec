"""Shared helpers for the pandas_ext package."""

import logging
from typing import TypeVar, cast  # noqa: F401 – cast re-exported for sub-modules

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
from numpy.typing import NDArray
from pydantic import BaseModel

from openaivec._cache.proxy import DEFAULT_MANAGED_CACHE_SIZE  # noqa: F401

_LOGGER = logging.getLogger("openaivec.pandas_ext")

T = TypeVar("T")


def _df_rows_to_json_series(df: pd.DataFrame) -> pd.Series:
    """Return a Series of JSON strings representing DataFrame rows.

    Uses DuckDB's ``to_json`` for high-throughput C++ serialisation.
    Index and name are preserved so downstream operations retain alignment.
    """
    conn = duckdb.connect(":memory:")
    json_col = conn.sql("SELECT to_json(df) AS j FROM df").fetchdf()["j"]
    conn.close()
    return pd.Series(json_col.values, index=df.index, name="record")


def _embeddings_to_series(
    embeddings: list[NDArray[np.float32]],
    index: pd.Index,
    name: str | None = None,
) -> pd.Series:
    """Build an Arrow-backed Series from a list of embedding vectors.

    Each element is stored as a ``pyarrow.FixedSizeListArray<float32>``
    so the Series can be passed to DuckDB / Parquet without conversion.

    Args:
        embeddings (list[NDArray[np.float32]]): Embedding vectors.
        index (pd.Index): Index to align with.
        name (str | None): Series name.

    Returns:
        pandas.Series: Arrow-backed Series of fixed-size float32 lists.
    """
    if not embeddings:
        return pd.Series([], index=index, name=name, dtype=object)
    dim = len(embeddings[0])
    flat = np.concatenate(embeddings)
    arrow_arr = pa.FixedSizeListArray.from_arrays(pa.array(flat, type=pa.float32()), list_size=dim)
    return pd.Series(arrow_arr, index=index, name=name, dtype=pd.ArrowDtype(pa.list_(pa.float32(), dim)))


def _embedding_series_to_matrix(series: pd.Series) -> NDArray[np.float32]:
    """Extract a 2D float32 numpy matrix from an embedding Series.

    When the Series is Arrow-backed (``FixedSizeListArray``), the flat
    buffer is extracted with zero copy.  For plain object columns the
    fallback uses ``np.vstack``.

    Args:
        series (pd.Series): Series of embedding vectors.

    Returns:
        NDArray[np.float32]: 2D matrix of shape ``(n_rows, dim)``.
    """
    if hasattr(series, "array") and hasattr(series.array, "_pa_array"):
        pa_chunked = series.array._pa_array
        chunk = pa_chunked.combine_chunks()
        flat = chunk.values.to_numpy(zero_copy_only=False)
        return flat.reshape(len(chunk), -1).astype(np.float32, copy=False)
    return np.vstack(series.tolist())


def _extract_value(x, series_name):
    """Return a homogeneous ``dict`` representation of any Series value.

    Args:
        x (Any): Single element taken from the Series.
        series_name (str): Name of the Series (used for logging).

    Returns:
        dict: A dictionary representation or an empty ``dict`` if ``x`` cannot
            be coerced.
    """
    if x is None:
        return {}
    elif isinstance(x, BaseModel):
        return x.model_dump()
    elif isinstance(x, dict):
        return x

    _LOGGER.warning(
        f"The value '{x}' in the series '{series_name}' is not a dict or BaseModel. Returning an empty dict."
    )
    return {}
