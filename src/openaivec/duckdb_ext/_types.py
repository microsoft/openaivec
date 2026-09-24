"""DuckDB type mapping and Pydantic DDL helpers."""

from __future__ import annotations

import types
import typing
from datetime import date, datetime, time
from decimal import Decimal
from enum import Enum
from typing import Any
from uuid import UUID

import duckdb
from duckdb.sqltypes import DuckDBPyType
from pydantic import BaseModel

from openaivec.duckdb_ext._identifiers import _quote_identifier, _quote_table_name

__all__ = ["pydantic_to_duckdb_ddl"]


def _pydantic_to_struct_type(model: type[BaseModel]) -> DuckDBPyType:
    """Convert a Pydantic model to a DuckDB STRUCT type for UDF return values."""
    fields: dict[str, str] = {}
    for field_name, field_info in model.model_fields.items():
        fields[field_name] = _python_type_to_duckdb(field_info.annotation) if field_info.annotation else "VARCHAR"
    return duckdb.struct_type(fields)


_PRIMITIVE_TYPE_MAP: dict[type, str] = {
    str: "VARCHAR",
    int: "INTEGER",
    float: "DOUBLE",
    bool: "BOOLEAN",
    bytes: "BLOB",
    datetime: "TIMESTAMP",
    date: "DATE",
    time: "TIME",
    Decimal: "DECIMAL",
    UUID: "UUID",
}


def _python_type_to_duckdb(py_type: Any) -> str:
    """Map a Python/Pydantic type to its DuckDB column type string."""
    if py_type in _PRIMITIVE_TYPE_MAP:
        return _PRIMITIVE_TYPE_MAP[py_type]

    origin = typing.get_origin(py_type)

    if isinstance(py_type, type) and issubclass(py_type, Enum):
        if issubclass(py_type, int):
            return "INTEGER"
        if issubclass(py_type, float):
            return "DOUBLE"
        return "VARCHAR"

    if origin is list:
        args = typing.get_args(py_type)
        inner = args[0] if args else Any
        return f"{_python_type_to_duckdb(inner)}[]"

    if origin is dict or py_type is dict:
        return "JSON"

    if origin in (typing.Union, types.UnionType):
        args = typing.get_args(py_type)
        non_none = [arg for arg in args if arg is not type(None)]
        if len(args) == 2 and len(non_none) == 1:
            return _python_type_to_duckdb(non_none[0])
        raise ValueError(f"Unsupported Union type: {py_type}")

    if isinstance(py_type, type) and issubclass(py_type, BaseModel):
        fields = [
            f"{_quote_identifier(name)} {_python_type_to_duckdb(info.annotation) if info.annotation else 'VARCHAR'}"
            for name, info in py_type.model_fields.items()
        ]
        return f"STRUCT({', '.join(fields)})"

    if origin is typing.Literal:
        return "VARCHAR"

    return "VARCHAR"


def _serialize_for_duckdb(value: Any) -> Any:
    """Recursively convert Enum values to their primitives for DuckDB."""
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {k: _serialize_for_duckdb(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_serialize_for_duckdb(v) for v in value]
    return value


def pydantic_to_duckdb_ddl(model: type[BaseModel], table_name: str) -> str:
    """Generate a ``CREATE TABLE`` DDL statement from a Pydantic model.

    Args:
        model (type[BaseModel]): The Pydantic model class.
        table_name (str): Name for the DuckDB table.

    Returns:
        str: A ``CREATE TABLE IF NOT EXISTS`` statement.

    Example:
        >>> from pydantic import BaseModel
        >>> from openaivec.duckdb_ext import pydantic_to_duckdb_ddl
        >>> class Review(BaseModel):
        ...     sentiment: str
        ...     rating: int
        ...     tags: list[str]
        >>> print(pydantic_to_duckdb_ddl(Review, "reviews"))
        CREATE TABLE IF NOT EXISTS "reviews" (
            "sentiment" VARCHAR,
            "rating" INTEGER,
            "tags" VARCHAR[]
        )
    """
    table = _quote_table_name(table_name)
    columns: list[str] = []
    for field_name, field_info in model.model_fields.items():
        col_type = _python_type_to_duckdb(field_info.annotation) if field_info.annotation else "VARCHAR"
        columns.append(f"    {_quote_identifier(field_name)} {col_type}")
    body = ",\n".join(columns)
    return f"CREATE TABLE IF NOT EXISTS {table} (\n{body}\n)"
