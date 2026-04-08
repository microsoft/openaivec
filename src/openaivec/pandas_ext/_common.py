"""Shared helpers for the pandas_ext package."""

import json
import logging
from datetime import date, datetime, time
from typing import TypeVar, cast  # noqa: F401 – cast re-exported for sub-modules

import numpy as np
import pandas as pd
from pydantic import BaseModel

from openaivec._cache.proxy import DEFAULT_MANAGED_CACHE_SIZE  # noqa: F401

_LOGGER = logging.getLogger("openaivec.pandas_ext")

T = TypeVar("T")


def _df_rows_to_json_series(df: pd.DataFrame) -> pd.Series:
    """Return a Series of JSON strings (UTF-8, no ASCII escaping) representing DataFrame rows.

    Each element is the JSON serialisation of the corresponding row as a dict. Index and
    name are preserved so downstream operations retain alignment. This consolidates the
    previously duplicated inline pipeline used by responses*/task* DataFrame helpers.
    """

    def _to_json_default(value: object) -> object:
        if isinstance(value, (pd.Timestamp, datetime, date, time)):
            return value.isoformat()
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")

    return pd.Series(df.to_dict(orient="records"), index=df.index, name="record").map(
        lambda x: json.dumps(x, ensure_ascii=False, default=_to_json_default)
    )


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
