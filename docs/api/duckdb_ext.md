# DuckDB Extension

The DuckDB integration provides SQL UDFs for responses, embeddings, prepared
tasks, parsing, and token counting. The table below shows how to use the main
pandas features from DuckDB.

| pandas feature | DuckDB equivalent |
| --- | --- |
| `Series.ai.responses()` | `responses_udf()` |
| `Series.ai.embeddings()` | `embeddings_udf()` |
| `Series.ai.task()` | `task_udf()` |
| `Series.ai.parse()` with an explicit model | `parse_udf(..., response_format=Model)` |
| `Series.ai.parse()` with schema inference | `parse_udf(..., example_table_name=..., example_field_name=...)` |
| `Series.ai.infer_schema()` / `DataFrame.ai.infer_schema()` | `infer_schema()` on a text column or a view of JSON rows |
| `Series.ai.count_tokens()` | `count_tokens_udf()` |
| `Series.ai.extract()` | SQL `STRUCT` field access |
| `DataFrame.ai.responses()` / `.task()` / `.parse()` | Pass `to_json(t)` from a table alias to the UDF; use a view of those JSON rows for schema inference |
| `DataFrame.ai.similarity()` | SQL `list_cosine_similarity(a, b)` |

The AI-based `DataFrame.ai.fillna()` builds examples from the whole DataFrame
and predicts only missing rows. There is no dedicated DuckDB wrapper for that
table-level operation; `COALESCE` is useful for ordinary SQL defaults but does
not perform AI imputation.

The DuckDB UDF, schema, token, similarity, and type implementations live in
the `openaivec.duckdb_ext` package. Persistent caching is a core `BatchCache`
feature; its DuckDB backend remains in `openaivec._cache`:

```python
from openaivec._cache import DuckDBCacheBackend

backend = DuckDBCacheBackend.of("results.duckdb")
# Pass backend as the cache argument when constructing BatchCache.
```

Cache table names are single unqualified SQL identifiers matching
`[A-Za-z_][A-Za-z_0-9]*`; reserved words are supported. Keys can be strings,
integers, booleans, floats, bytes, or tuples of those types. Keys are encoded
with type tags, so `1` and `"1"` persist independently; other key types raise
`TypeError`. The current typed-key table schema is incompatible with tables
created by older releases. Opening an old cache table raises `ValueError`
instead of silently misreading its entries; choose a new table name or
explicitly migrate/rebuild the old data. The backend also provides `get_many`,
`put_many`, and `touch_many` for bounded bulk SQL operations. `get_many` does
not update LRU order; call `touch_many` after consuming the fetched keys.
Because `get_many` returns a Python dictionary, requesting distinct keys that
compare equal in Python (for example, `1` and `True`) together raises
`ValueError` rather than silently combining them; individual lookups remain
distinct.

## Schema inference and parsing

```python
import duckdb
from openaivec.duckdb_ext import infer_schema, parse_udf

conn = duckdb.connect()
conn.execute("CREATE TABLE reviews (review VARCHAR)")
conn.executemany(
    "INSERT INTO reviews VALUES (?)",
    [("The camera is excellent. 5 stars.",), ("The battery failed. 1 star.",)],
)

# Infer once and reuse the resulting model and prompt for multiple UDFs.
schema = infer_schema(
    conn,
    instructions="Extract product feedback and rating.",
    example_table_name="reviews",
    example_field_name="review",
)
parse_udf(
    conn,
    "parse_review",
    instructions=schema.inference_prompt,
    response_format=schema.model,
)
conn.sql("SELECT parse_review(review) AS parsed FROM reviews")

# Or infer and register in one call.
parse_udf(
    conn,
    "parse_review_auto",
    instructions="Extract product feedback and rating.",
    example_table_name="reviews",
    example_field_name="review",
)
conn.sql("SELECT parse_review_auto(review) AS parsed FROM reviews")
```

`parse_udf` infers the schema when it is registered. The sample excludes SQL
`NULL` values and is bounded by `max_examples` (default 100). An explicit
`response_format` skips inference. `max_retries` controls total inference
attempts; `max_validation_retries` controls additional extraction corrections.

## Token counts and complete rows

```python
from openaivec.duckdb_ext import count_tokens_udf, parse_udf, responses_udf

count_tokens_udf(conn, "count_tokens")
conn.sql("SELECT review, count_tokens(review) AS num_tokens FROM reviews")

responses_udf(conn, "describe_row", instructions="Summarize the complete row.")
conn.sql("SELECT describe_row(to_json(t)) FROM reviews AS t")

# For DataFrame-style automatic parsing, infer from JSON rows and pass the
# same expression to the registered UDF.
conn.execute("CREATE VIEW review_rows AS SELECT to_json(t) AS input FROM reviews AS t")
parse_udf(
    conn,
    "parse_review_row",
    instructions="Extract product feedback and rating from each row.",
    example_table_name="review_rows",
    example_field_name="input",
)
conn.sql("SELECT parse_review_row(to_json(t)) FROM reviews AS t")
```

`count_tokens_udf` uses the same configured `tiktoken.Encoding` as the pandas
accessor and returns SQL `NULL` for SQL `NULL` input. Structured outputs from
`responses_udf`, `task_udf`, and `parse_udf` are native DuckDB `STRUCT` values;
read fields with `my_udf(text).field_name` or expand a stored STRUCT column
with `parsed.*`.

## Similarity search and generated DDL

`similarity_search(conn, "documents", "queries", top_k=3)` returns
`query_id`, `query_text`, `target_text`, and `score`, ordered by query row and
descending score. `query_id` is a 1-based position assigned during the query
table scan (not a persistent key or a guaranteed order across scans); duplicate
query text still receives an independent top-k ranking. `top_k` must be a
positive integer and is passed as a SQL parameter.

`pydantic_to_duckdb_ddl(Model, "schema.table")` quotes each table/column
identifier, including nested STRUCT fields. Table names may be unqualified or
qualified as `schema.table` or `catalog.schema.table`; use DuckDB double quotes
around a part to include a literal dot (for example, `"my.schema".items`).
Spaces and reserved words are supported. Embedded double quotes in field names
are escaped; SQL expressions and statement separators in identifiers are
rejected. Both `typing.Optional[T]` and `T | None` map to the same DuckDB and
Spark type; unions of distinct non-null types are unsupported.

::: openaivec.duckdb_ext
