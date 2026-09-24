"""DuckDB identifier validation and quoting."""

from __future__ import annotations

import re

__all__: list[str] = []

_TABLE_NAME = re.compile(r'(?:"(?:[^"]|"")*"|[\w ]+)(?:\.(?:"(?:[^"]|"")*"|[\w ]+)){0,2}', re.UNICODE)
_TABLE_PART = re.compile(r'"(?:[^"]|"")*"|[\w ]+', re.UNICODE)


def _quote_identifier(name: str) -> str:
    """Quote one literal identifier (including an embedded double quote)."""
    if not isinstance(name, str) or not name.strip() or not re.fullmatch(r'[\w ."]+', name, re.UNICODE):
        raise ValueError("identifier must be a non-empty name without SQL syntax")
    return '"' + name.replace('"', '""') + '"'


def _quote_table_name(name: str) -> str:
    """Quote a one- to three-part table name; quoted parts may contain dots."""
    if not isinstance(name, str) or not _TABLE_NAME.fullmatch(name):
        raise ValueError("table identifier must be a valid table or qualified name")
    parts = _TABLE_PART.findall(name)
    decoded = [part[1:-1].replace('""', '"') if part.startswith('"') else part.strip() for part in parts]
    return ".".join(_quote_identifier(part) for part in decoded)
