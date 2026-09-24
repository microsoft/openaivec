"""DuckDB DDL, STRUCT, and serialization type tests."""

from __future__ import annotations

import typing
from enum import Enum
from types import SimpleNamespace
from typing import Optional

import duckdb
import pytest
from pydantic import BaseModel, create_model

from openaivec.duckdb_ext import (
    _pydantic_to_struct_type,
    _python_type_to_duckdb,
    _serialize_for_duckdb,
    pydantic_to_duckdb_ddl,
)

# ---------------------------------------------------------------------------
# pydantic_to_duckdb_ddl
# ---------------------------------------------------------------------------


class SimpleModel(BaseModel):
    name: str
    age: int
    score: float
    active: bool


class NestedModel(BaseModel):
    label: str
    tags: list[str]


class ModelWithOptional(BaseModel):
    required_field: str
    optional_field: str | None = None


class ModelWithNested(BaseModel):
    info: NestedModel
    count: int


class TestPydanticToDuckDBDDL:
    def test_simple_model(self):
        ddl = pydantic_to_duckdb_ddl(SimpleModel, "simple")
        assert 'CREATE TABLE IF NOT EXISTS "simple"' in ddl
        assert '"name" VARCHAR' in ddl
        assert '"age" INTEGER' in ddl
        assert '"score" DOUBLE' in ddl
        assert '"active" BOOLEAN' in ddl

    def test_list_field(self):
        ddl = pydantic_to_duckdb_ddl(NestedModel, "nested")
        assert '"label" VARCHAR' in ddl
        assert '"tags" VARCHAR[]' in ddl

    def test_optional_field(self):
        ddl = pydantic_to_duckdb_ddl(ModelWithOptional, "opt")
        assert '"required_field" VARCHAR' in ddl
        assert '"optional_field" VARCHAR' in ddl

    def test_nested_struct(self):
        ddl = pydantic_to_duckdb_ddl(ModelWithNested, "with_nested")
        assert '"info" STRUCT' in ddl
        assert '"count" INTEGER' in ddl

    def test_ddl_is_executable(self):
        """Verify the generated DDL runs without error in DuckDB."""
        conn = duckdb.connect(":memory:")
        ddl = pydantic_to_duckdb_ddl(SimpleModel, "test_table")
        conn.execute(ddl)
        result = conn.execute("SELECT * FROM test_table").fetchall()
        assert result == []
        conn.close()

    def test_list_field_ddl_is_executable(self):
        conn = duckdb.connect(":memory:")
        ddl = pydantic_to_duckdb_ddl(NestedModel, "tag_table")
        conn.execute(ddl)
        conn.execute("INSERT INTO tag_table VALUES ('test', ['a', 'b'])")
        result = conn.execute("SELECT * FROM tag_table").fetchone()
        assert result is not None
        assert result[0] == "test"
        assert result[1] == ["a", "b"]
        conn.close()

    def test_datetime_fields(self):
        from datetime import date, datetime

        class Event(BaseModel):
            name: str
            occurred_at: datetime
            event_date: date

        ddl = pydantic_to_duckdb_ddl(Event, "events")
        assert '"occurred_at" TIMESTAMP' in ddl
        assert '"event_date" DATE' in ddl

        conn = duckdb.connect(":memory:")
        conn.execute(ddl)
        conn.execute("INSERT INTO events VALUES ('test', '2024-01-01 12:00:00', '2024-01-01')")
        row = conn.execute("SELECT * FROM events").fetchone()
        assert row is not None
        assert row[0] == "test"
        conn.close()

    def test_datetime_struct_udf(self):
        from datetime import datetime

        class Event(BaseModel):
            name: str
            ts: datetime

        st = _pydantic_to_struct_type(Event)
        conn = duckdb.connect(":memory:")

        def mock(x: str) -> dict:
            return {"name": x, "ts": datetime(2024, 6, 1, 12, 30)}

        conn.create_function("event_fn", mock, [duckdb.sqltype("VARCHAR")], st)
        row = conn.sql("SELECT event_fn('test').name, event_fn('test').ts").fetchone()
        assert row is not None
        assert row[0] == "test"
        assert row[1] == datetime(2024, 6, 1, 12, 30)
        conn.close()

    def test_decimal_and_uuid_fields(self):
        from decimal import Decimal
        from uuid import UUID

        class Payment(BaseModel):
            transaction_id: UUID
            amount: Decimal

        ddl = pydantic_to_duckdb_ddl(Payment, "payments")
        assert '"transaction_id" UUID' in ddl
        assert '"amount" DECIMAL' in ddl

        st = _pydantic_to_struct_type(Payment)
        conn = duckdb.connect(":memory:")

        test_uuid = UUID("12345678-1234-5678-1234-567812345678")

        def mock(x: str) -> dict:
            return {"transaction_id": test_uuid, "amount": Decimal("99.99")}

        conn.create_function("pay_fn", mock, [duckdb.sqltype("VARCHAR")], st)
        row = conn.sql("SELECT pay_fn('x').transaction_id, pay_fn('x').amount").fetchone()
        assert row is not None
        assert row[0] == test_uuid
        assert row[1] == Decimal("99.99")
        conn.close()

    def test_optional_annotation_spellings(self):
        class NullableModel(BaseModel):
            old: Optional[int]
            new: int | None
            nested: list[Optional[int]]

        ddl = pydantic_to_duckdb_ddl(NullableModel, "nullable")
        assert '"old" INTEGER' in ddl
        assert '"new" INTEGER' in ddl
        assert '"nested" INTEGER[]' in ddl
        assert _python_type_to_duckdb(Optional[int]) == _python_type_to_duckdb(int | None) == "INTEGER"

    def test_pep604_union_origin_on_older_python(self, monkeypatch):
        import openaivec.duckdb_ext._types as duckdb_types

        monkeypatch.setattr(
            duckdb_types,
            "typing",
            SimpleNamespace(
                get_origin=typing.get_origin,
                get_args=typing.get_args,
                Union=object(),
                Literal=typing.Literal,
            ),
        )
        assert duckdb_types._python_type_to_duckdb(int | None) == "INTEGER"

    def test_nonoptional_union_fails(self):
        with pytest.raises(ValueError, match="Unsupported Union"):
            _python_type_to_duckdb(int | str)
        with pytest.raises(ValueError, match="Unsupported Union"):
            _python_type_to_duckdb(Optional[int | str])

    def test_quoted_ddl_and_nested_struct(self):
        inner = create_model("Inner", **{'a"b': (int, ...)})
        outer = create_model("Outer", **{"select": (inner, ...), "field name": (str, ...)})

        conn = duckdb.connect(":memory:")
        conn.execute('CREATE SCHEMA "my schema"')
        ddl = pydantic_to_duckdb_ddl(outer, '"my schema"."table name"')
        assert '"select" STRUCT("a""b" INTEGER)' in ddl
        conn.execute(ddl)
        conn.execute('INSERT INTO "my schema"."table name" VALUES (?, ?)', ({'a"b': 3}, "ok"))
        assert conn.execute('SELECT "field name" FROM "my schema"."table name"').fetchone() == ("ok",)
        conn.close()

    @pytest.mark.parametrize("name", ["", "table; DROP TABLE x", "schema..table", "table -- comment"])
    def test_invalid_ddl_table_name(self, name):
        with pytest.raises(ValueError, match="identifier"):
            pydantic_to_duckdb_ddl(SimpleModel, name)

    def test_invalid_ddl_field_name(self):
        model = create_model("BadField", **{"field; DROP TABLE x": (int, ...)})
        with pytest.raises(ValueError, match="identifier"):
            pydantic_to_duckdb_ddl(model, "safe")


# ---------------------------------------------------------------------------
# Pydantic → DuckDB STRUCT type mapping
# ---------------------------------------------------------------------------


class SentimentResult(BaseModel):
    label: str
    score: float


class ReviewResult(BaseModel):
    sentiment: str
    rating: int
    tags: list[str]


class TestPydanticToStructType:
    def test_simple_struct(self):
        st = _pydantic_to_struct_type(SentimentResult)
        st_str = str(st)
        assert "VARCHAR" in st_str
        assert "DOUBLE" in st_str

    def test_struct_with_list(self):
        st = _pydantic_to_struct_type(ReviewResult)
        st_str = str(st)
        assert "INTEGER" in st_str
        assert "VARCHAR[]" in st_str

    def test_struct_udf_field_access(self):
        conn = duckdb.connect(":memory:")
        st = _pydantic_to_struct_type(SentimentResult)

        def mock_fn(x: str) -> dict:
            return {"label": "positive", "score": 0.95}

        conn.create_function("get_sentiment", mock_fn, [duckdb.sqltype("VARCHAR")], st)

        row = conn.sql("SELECT get_sentiment('hello').label AS label, get_sentiment('hello').score AS score").fetchone()
        assert row is not None
        assert row[0] == "positive"
        assert row[1] == pytest.approx(0.95)
        conn.close()

    def test_struct_udf_in_table(self):
        conn = duckdb.connect(":memory:")
        st = _pydantic_to_struct_type(SentimentResult)

        def mock_fn(x: str) -> dict:
            return {"label": "pos" if "good" in x else "neg", "score": 0.9}

        conn.create_function("classify", mock_fn, [duckdb.sqltype("VARCHAR")], st)
        conn.execute("CREATE TABLE texts (t VARCHAR)")
        conn.execute("INSERT INTO texts VALUES ('good day'), ('bad day')")

        df = conn.sql("SELECT t, classify(t).label AS label FROM texts").df()
        assert len(df) == 2
        assert df[df["t"] == "good day"].iloc[0]["label"] == "pos"
        assert df[df["t"] == "bad day"].iloc[0]["label"] == "neg"
        conn.close()


# ---------------------------------------------------------------------------
# Enum / Literal type mapping
# ---------------------------------------------------------------------------


class StrColor(str, Enum):
    RED = "red"
    GREEN = "green"


class IntPriority(int, Enum):
    LOW = 1
    HIGH = 2


class WithEnums(BaseModel):
    label: str
    color: StrColor
    priority: IntPriority


class TestEnumTypeMapping:
    def test_str_enum_maps_to_varchar(self):
        assert _python_type_to_duckdb(StrColor) == "VARCHAR"

    def test_int_enum_maps_to_integer(self):
        assert _python_type_to_duckdb(IntPriority) == "INTEGER"

    def test_struct_with_enums(self):
        st = _pydantic_to_struct_type(WithEnums)
        st_str = str(st)
        assert "INTEGER" in st_str
        assert "VARCHAR" in st_str

    def test_serialize_converts_enum_values(self):
        raw = {"label": "test", "color": StrColor.RED, "priority": IntPriority.HIGH}
        result = _serialize_for_duckdb(raw)
        assert result == {"label": "test", "color": "red", "priority": 2}

    def test_serialize_handles_nested_enums(self):
        raw = {"items": [StrColor.RED, StrColor.GREEN], "nested": {"p": IntPriority.LOW}}
        result = _serialize_for_duckdb(raw)
        assert result == {"items": ["red", "green"], "nested": {"p": 1}}

    def test_enum_struct_udf(self):
        conn = duckdb.connect(":memory:")
        st = _pydantic_to_struct_type(WithEnums)

        def mock(x: str) -> dict:
            return _serialize_for_duckdb({"label": x, "color": StrColor.GREEN, "priority": IntPriority.HIGH})

        conn.create_function("with_enums", mock, [duckdb.sqltype("VARCHAR")], st)
        row = conn.sql("SELECT with_enums('test').priority, with_enums('test').color").fetchone()
        assert row == (2, "green")
        conn.close()
