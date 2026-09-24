"""Tests for the DuckDB token-count UDF."""

from unittest.mock import Mock

import duckdb
import pyarrow as pa
import tiktoken

from openaivec._provider import CONTAINER
from openaivec.duckdb_ext import count_tokens_udf


def test_count_tokens_udf_preserves_values_nulls_and_sql_type():
    encoding = CONTAINER.resolve(tiktoken.Encoding)
    texts = ["hello", None, "", "hello world", "こんにちは"]
    expected = [len(encoding.encode(text)) if text is not None else None for text in texts]

    with duckdb.connect() as conn:
        count_tokens_udf(conn, "token_count")
        conn.execute("CREATE TABLE inputs (id INTEGER, body VARCHAR)")
        conn.executemany("INSERT INTO inputs VALUES (?, ?)", list(enumerate(texts)))

        result = conn.sql("SELECT id, token_count(body) FROM inputs ORDER BY id")
        assert result.fetchall() == list(enumerate(expected))
        assert str(result.types[1]) == "BIGINT"
        assert str(conn.sql("SELECT token_count(body) FROM inputs WHERE FALSE").types[0]) == "BIGINT"


def test_count_tokens_udf_arrow_batches_keep_bigint_for_empty_and_all_null(monkeypatch):
    class RecordingEncoding:
        def __init__(self):
            self.inputs = []

        def encode_batch(self, texts):
            self.inputs.append(texts)
            return [list(range(len(text))) for text in texts]

    encoding = RecordingEncoding()
    resolve = CONTAINER.resolve
    monkeypatch.setattr(CONTAINER, "resolve", lambda kind: encoding if kind is tiktoken.Encoding else resolve(kind))

    with duckdb.connect() as conn:
        registration = Mock(wraps=conn)
        count_tokens_udf(registration, "token_count")
        callback = registration.create_function.call_args.args[1]

        for texts, expected in [([], []), ([None, None], [None, None]), (["a", None, "xyz"], [1, None, 3])]:
            result = callback(pa.array(texts, type=pa.string()))
            assert result.type == pa.int64()
            assert result.to_pylist() == expected

        assert encoding.inputs == [["a", "xyz"]]
