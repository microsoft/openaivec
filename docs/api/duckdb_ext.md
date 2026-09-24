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

All DuckDB-specific implementations live in the `openaivec.duckdb_ext`
package. Its public imports remain available from `openaivec.duckdb_ext`.
For persistent cache storage, use `DuckDBCacheBackend` from that package:

```python
from openaivec.duckdb_ext import DuckDBCacheBackend

backend = DuckDBCacheBackend.of("results.duckdb")
# Pass backend as the cache argument when constructing BatchCache.
```

The earlier `openaivec._cache` import path for `DuckDBCacheBackend` remains
available for existing callers.

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

::: openaivec.duckdb_ext
