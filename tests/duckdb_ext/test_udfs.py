import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import duckdb
import pyarrow as pa
import pytest
from openai import AsyncOpenAI
from pydantic import BaseModel, ConfigDict

from openaivec import PreparedTask, duckdb_ext
from openaivec._provider import CONTAINER


class NullableRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")
    label: str | None


class NestedRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")
    details: NullableRecord
    history: list[NullableRecord]
    tags: list[str]


@pytest.mark.parametrize("method_name", ["responses_udf", "task_udf"])
@pytest.mark.parametrize(
    ("response_format", "body"),
    [
        (str, "ok"),
        (NullableRecord, {"label": None}),
        (NestedRecord, {"details": {"label": None}, "history": [{"label": None}], "tags": []}),
    ],
)
@pytest.mark.parametrize("texts", [[], [None, None], ["missing", "missing"], ["value", None, "missing", "value"]])
def test_duckdb_response_nulls_preserve_type_and_order(monkeypatch, method_name, response_format, body, texts):
    seen = []

    def respond(**kwargs):
        messages = json.loads(kwargs["input"])["user_messages"]
        seen.extend(message["body"] for message in messages)
        if messages[0]["body"] == "missing":
            return SimpleNamespace(output_parsed=None)
        output = kwargs["text_format"].model_validate(
            {"assistant_messages": [{"id": message["id"], "body": body} for message in messages]}
        )
        return SimpleNamespace(output_parsed=output)

    parse = AsyncMock(side_effect=respond)
    client = Mock(spec=AsyncOpenAI)
    client.responses = Mock(parse=parse)
    resolve = CONTAINER.resolve
    monkeypatch.setattr(CONTAINER, "resolve", lambda kind: client if kind is AsyncOpenAI else resolve(kind))
    with duckdb.connect() as connection:
        if method_name == "task_udf":
            duckdb_ext.task_udf(
                connection,
                "reply",
                task=PreparedTask(instructions="echo", response_format=response_format),
                batch_size=1,
            )
        else:
            duckdb_ext.responses_udf(
                connection, "reply", instructions="echo", response_format=response_format, batch_size=1
            )
        connection.execute("CREATE TABLE inputs(id INTEGER, value VARCHAR)")
        if texts:
            connection.executemany("INSERT INTO inputs VALUES (?, ?)", list(enumerate(texts)))
        expected = [(index, body if text == "value" else None) for index, text in enumerate(texts)]
        assert connection.sql("SELECT id, reply(value) FROM inputs ORDER BY id").fetchall() == expected
        assert connection.sql("SELECT id, reply(value) FROM inputs ORDER BY id").fetchall() == expected
        assert None not in seen
        assert sorted(seen) == sorted(set(text for text in texts if text is not None))
        result_type = str(connection.sql("SELECT reply(value) FROM inputs").types[0])
        assert result_type == (
            "VARCHAR" if response_format is str else str(duckdb_ext._pydantic_to_struct_type(response_format))
        )


@pytest.mark.parametrize("response_format", [str, NullableRecord, NestedRecord])
@pytest.mark.parametrize("texts", [[], [None, None], ["missing"]])
def test_duckdb_all_null_arrow_output_uses_declared_type(monkeypatch, response_format, texts):
    parse = AsyncMock(return_value=SimpleNamespace(output_parsed=None))
    client = Mock(spec=AsyncOpenAI)
    client.responses = Mock(parse=parse)
    resolve = CONTAINER.resolve
    monkeypatch.setattr(CONTAINER, "resolve", lambda kind: client if kind is AsyncOpenAI else resolve(kind))
    with duckdb.connect() as connection:
        registration = Mock(wraps=connection)
        duckdb_ext.responses_udf(registration, "reply", instructions="echo", response_format=response_format)
        callback = registration.create_function.call_args.args[1]
        declared_type = connection.sql("SELECT reply(NULL::VARCHAR) AS value LIMIT 0").to_arrow_table().schema[0].type
        output = callback(pa.array(texts, type=pa.string()))
        assert output.type == declared_type
        assert output.to_pylist() == [None] * len(texts)
        assert parse.call_count == int("missing" in texts)
