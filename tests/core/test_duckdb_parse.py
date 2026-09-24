"""Offline regression tests for DuckDB schema inference and parsing."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import duckdb
import pytest
from openai import AsyncOpenAI
from pydantic import BaseModel, ConfigDict

from openaivec import SchemaInferer, duckdb_ext
from openaivec._provider import CONTAINER
from openaivec.duckdb_ext import _schema as duckdb_schema


class ParsedLabel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str


def _install_async_responses(monkeypatch):
    seen: list[str] = []

    def respond(**kwargs):
        messages = json.loads(kwargs["input"])["user_messages"]
        seen.extend(message["body"] for message in messages)
        parsed = kwargs["text_format"].model_validate(
            {
                "assistant_messages": [
                    {"id": message["id"], "body": {"label": message["body"].upper()}} for message in messages
                ]
            }
        )
        return SimpleNamespace(output_parsed=parsed)

    parse = AsyncMock(side_effect=respond)
    client = Mock(spec=AsyncOpenAI)
    client.responses = Mock(parse=parse)
    resolve = CONTAINER.resolve
    monkeypatch.setattr(CONTAINER, "resolve", lambda kind: client if kind is AsyncOpenAI else resolve(kind))
    return seen, parse


def test_infer_schema_quotes_identifiers_and_excludes_null_examples(monkeypatch):
    inferred = SimpleNamespace(model=ParsedLabel, inference_prompt="Extract label")
    inferer = Mock(spec=SchemaInferer)
    inferer.infer_schema.return_value = inferred
    resolve = CONTAINER.resolve
    monkeypatch.setattr(CONTAINER, "resolve", lambda kind: inferer if kind is SchemaInferer else resolve(kind))

    with duckdb.connect() as connection:
        connection.execute('CREATE TABLE "select" ("from" VARCHAR)')
        connection.executemany('INSERT INTO "select" VALUES (?)', [("first",), (None,), ("second",)])
        result = duckdb_ext.infer_schema(
            connection,
            instructions="Extract label",
            example_table_name="select",
            example_field_name="from",
            max_examples=2,
        )

    assert result is inferred
    inferer.infer_schema.assert_called_once()
    data = inferer.infer_schema.call_args.args[0]
    assert data.instructions == "Extract label"
    assert len(data.examples) == 2
    assert set(data.examples) == {"first", "second"}


@pytest.mark.parametrize("max_examples", [0, -1])
def test_infer_schema_rejects_invalid_max_examples(max_examples):
    with duckdb.connect() as connection:
        connection.execute("CREATE TABLE examples (text VARCHAR)")
        with pytest.raises(ValueError, match="max_examples"):
            duckdb_ext.infer_schema(
                connection,
                instructions="Extract label",
                example_table_name="examples",
                example_field_name="text",
                max_examples=max_examples,
            )


@pytest.mark.parametrize("values", [[], [None, None]])
def test_infer_schema_rejects_empty_or_null_only_samples(values):
    with duckdb.connect() as connection:
        connection.execute("CREATE TABLE examples (text VARCHAR)")
        if values:
            connection.executemany("INSERT INTO examples VALUES (?)", [(value,) for value in values])
        with pytest.raises(ValueError):
            duckdb_ext.infer_schema(
                connection,
                instructions="Extract label",
                example_table_name="examples",
                example_field_name="text",
            )


@pytest.mark.parametrize(
    ("table_name", "field_name"), [("missing_table", "text"), ("examples", "missing_column")]
)
def test_infer_schema_reports_missing_source(table_name, field_name):
    with duckdb.connect() as connection:
        connection.execute("CREATE TABLE examples (text VARCHAR)")
        expected_error = duckdb.CatalogException if table_name == "missing_table" else duckdb.BinderException
        with pytest.raises(expected_error):
            duckdb_ext.infer_schema(
                connection,
                instructions="Extract label",
                example_table_name=table_name,
                example_field_name=field_name,
            )


def test_parse_udf_infers_once_at_registration_and_returns_typed_struct(monkeypatch):
    seen, parse = _install_async_responses(monkeypatch)
    inferred = SimpleNamespace(model=ParsedLabel, inference_prompt="Resolved instructions")
    inference = Mock(return_value=inferred)
    monkeypatch.setattr(duckdb_schema, "infer_schema", inference)

    with duckdb.connect() as connection:
        connection.execute("CREATE TABLE examples (id INTEGER, text VARCHAR)")
        connection.executemany(
            "INSERT INTO examples VALUES (?, ?)", [(0, "first"), (1, None), (2, "second"), (3, "first")]
        )
        duckdb_ext.parse_udf(
            connection,
            "parsed",
            instructions="Extract label",
            example_table_name="examples",
            example_field_name="text",
            batch_size=1,
        )
        inference.assert_called_once()
        assert parse.call_count == 0
        query = "SELECT id, parsed(text).label FROM examples ORDER BY id"
        expected = [(0, "FIRST"), (1, None), (2, "SECOND"), (3, "FIRST")]
        assert connection.sql(query).fetchall() == expected
        assert connection.sql(query).fetchall() == expected
        result_type = str(connection.sql("SELECT parsed(text) FROM examples LIMIT 0").types[0])

    assert result_type == str(duckdb_ext._pydantic_to_struct_type(ParsedLabel))
    inference.assert_called_once()
    assert sorted(seen) == ["first", "second"]
    assert all("Resolved instructions" in call.kwargs["instructions"] for call in parse.call_args_list)


def test_parse_udf_explicit_model_skips_inference(monkeypatch):
    seen, _ = _install_async_responses(monkeypatch)
    monkeypatch.setattr(duckdb_schema, "infer_schema", Mock(side_effect=AssertionError("unexpected inference")))

    with duckdb.connect() as connection:
        duckdb_ext.parse_udf(
            connection,
            "parsed",
            instructions="Extract label",
            response_format=ParsedLabel,
        )
        assert connection.sql("SELECT parsed('value').label").fetchone() == ("VALUE",)

    assert seen == ["value"]


def test_parse_udf_explicit_str_uses_varchar_response(monkeypatch):
    inference = Mock(side_effect=AssertionError("unexpected inference"))
    register = Mock()
    monkeypatch.setattr(duckdb_schema, "infer_schema", inference)
    monkeypatch.setattr(duckdb_schema, "responses_udf", register)

    with duckdb.connect() as connection:
        duckdb_ext.parse_udf(connection, "parsed", instructions="Translate", response_format=str)

    inference.assert_not_called()
    register.assert_called_once()
    assert register.call_args.kwargs["instructions"] == "Translate"
    assert register.call_args.kwargs["response_format"] is str


def test_parse_udf_rejects_unsupported_response_format():
    with duckdb.connect() as connection:
        with pytest.raises(TypeError, match="response_format"):
            duckdb_ext.parse_udf(connection, "parsed", instructions="Extract count", response_format=int)


@pytest.mark.parametrize(
    ("table_name", "field_name"), [(None, None), ("examples", None), (None, "text")]
)
def test_parse_udf_requires_complete_example_source(table_name, field_name):
    with duckdb.connect() as connection:
        with pytest.raises(ValueError, match="response_format|example"):
            duckdb_ext.parse_udf(
                connection,
                "parsed",
                instructions="Extract label",
                example_table_name=table_name,
                example_field_name=field_name,
            )
