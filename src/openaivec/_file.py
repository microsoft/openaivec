"""File path detection utilities."""

from __future__ import annotations

import os
from pathlib import Path

__all__: list[str] = []

_FILE_EXTENSIONS = frozenset(
    {
        ".csv",
        ".tsv",
        ".json",
        ".jsonl",
        ".ndjson",
        ".parquet",
        ".pq",
        ".xlsx",
        ".xls",
        ".feather",
        ".arrow",
        ".ipc",
        ".orc",
        ".txt",
        ".log",
    }
)


def is_file_path(value: str) -> bool:
    """Check whether *value* looks like a file path.

    Returns ``True`` when any of the following hold:

    * The string points to an existing file on disk.
    * It contains a path separator (``/`` or ``os.sep``).
    * It ends with a recognised data-file extension.

    Args:
        value (str): The string to inspect.

    Returns:
        bool: ``True`` if *value* appears to be a file path.
    """
    if not value or not isinstance(value, str):
        return False

    if os.path.isfile(value):
        return True

    if "/" in value or (os.sep != "/" and os.sep in value):
        return True

    suffix = Path(value).suffix.lower()
    if suffix in _FILE_EXTENSIONS:
        return True

    return False
