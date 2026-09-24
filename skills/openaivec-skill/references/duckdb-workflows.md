# Internal DuckDB workflows

Use these internal patterns after the source, privacy boundary, provider,
model, destination, and write authorization are confirmed. Do not expose the
internal engine in ordinary user-facing progress messages.

Follow [data shaping and cross-tabs](data-shaping-and-crosstabs.md) before any
cleaning, join, deduplication, aggregation, or pivot. It defines when business
meaning must be clarified and how to reconcile local summaries after the AI
result is materialized.

## Input and output boundary

| User source | Internal pattern |
| --- | --- |
| DuckDB table or view | `conn.table("name")` |
| Parquet files or globs | `conn.read_parquet(..., filename=True, file_row_number=True)` |
| CSV/TSV files or globs | `conn.read_csv(..., filename=True)` |
| JSON/NDJSON files or globs | `conn.read_json(..., filename=True)` |
| Excel `.xlsx` | Follow `excel-setup.md`, then use `read_xlsx(...)` after support is available |
| SQLite database | Attach with the available SQLite extension, read-only by default |
| PostgreSQL or MySQL | Attach with the available official connector, read-only by default |
| Plain text files | DuckDB `read_text(...)` |
| File paths for openaivec multimodal handling | DuckDB `glob(...)`, then pass the path column with `multimodal=True` |

For other formats, use them only when the installed DuckDB version already
supports the format or its required extension. Do not add a non-DuckDB parser
to work around an unsupported source. Disable automatic extension installation
while probing and obtain explicit approval before installing an extension.

Use an in-memory connection and temporary tables unless the user explicitly
requests a persistent destination. Follow [safe data I/O](safe-data-io.md)
before any file or relational write. For an explicitly authorized new output,
verify that the destination does not exist, then export with parameterized
`COPY`:

```python
import os
from pathlib import Path

output_path = Path("new-results.parquet")
descriptor = os.open(output_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
os.close(descriptor)

conn.execute("COPY processed TO ? (FORMAT PARQUET, COMPRESSION ZSTD)", [str(output_path)])
```

Do not call `COPY` unless exclusive reservation succeeds; the writer can
replace an existing file. An explicit request to create a new output does not
authorize overwrite. On failure, do not delete the newly reserved incomplete
path unless failure cleanup was also authorized.

## Core structured text workflow

This example globally deduplicates inputs, invokes the remote UDF once per
distinct value, then restores every source row.

```python
import os
from pathlib import Path

import duckdb
import openaivec
from pydantic import BaseModel, ConfigDict

from openaivec.duckdb_ext import count_tokens_udf, responses_udf

input_glob = "data/reviews/*.parquet"
output_parquet = Path("review-results-20260924T183656.parquet")


class ReviewResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str
    summary: str


openaivec.set_responses_model("gpt-6-luna")
conn = duckdb.connect()
conn.read_parquet(
    input_glob,
    filename=True,
    file_row_number=True,
    union_by_name=True,
).create_view("source_raw")

conn.execute(
    """
    CREATE TEMP TABLE staged AS
    SELECT
        filename AS source_file,
        file_row_number AS source_row,
        CAST(review_text AS VARCHAR) AS input_text
    FROM source_raw
    """
)

count_tokens_udf(conn, "ai_token_count")
stats = conn.sql(
    """
    SELECT
        count(*) AS total_rows,
        count(input_text) AS non_null_rows,
        count(DISTINCT input_text) FILTER (WHERE input_text IS NOT NULL) AS distinct_inputs,
        sum(ai_token_count(input_text)) AS input_tokens
    FROM staged
    """
).fetchone()
print(stats)

responses_udf(
    conn,
    "ai_process",
    instructions="Classify the review and provide a concise factual summary.",
    response_format=ReviewResult,
    batch_size=None,
    max_concurrency=8,
    reasoning={"effort": "none"},
)

pilot = conn.sql(
    """
    SELECT input_text, ai_process(input_text) AS ai_result
    FROM (
        SELECT DISTINCT input_text
        FROM staged
        WHERE input_text IS NOT NULL
        LIMIT 5
    )
    """
).fetchall()
print(pilot)

conn.execute(
    """
    CREATE TEMP TABLE unique_results AS
    SELECT input_text, ai_process(input_text) AS ai_result
    FROM (
        SELECT DISTINCT input_text
        FROM staged
        WHERE input_text IS NOT NULL
    )
    """
)
conn.execute(
    """
    CREATE TEMP TABLE processed AS
    SELECT s.source_file, s.source_row, s.input_text, u.ai_result
    FROM staged AS s
    LEFT JOIN unique_results AS u USING (input_text)
    """
)

validation = conn.sql(
    """
    SELECT
        count(*) AS output_rows,
        count(ai_result) AS succeeded_rows,
        count(*) FILTER (WHERE input_text IS NOT NULL AND ai_result IS NULL) AS null_results
    FROM processed
    """
).fetchone()
print(validation)

descriptor = os.open(output_parquet, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
os.close(descriptor)
conn.execute(
    "COPY (SELECT * FROM processed ORDER BY source_file, source_row) TO ? (FORMAT PARQUET, COMPRESSION ZSTD)",
    [str(output_parquet)],
)
conn.close()
```

Do not run the pilot until the user approves a remote call. Do not run the
full `CREATE ... unique_results` until the pilot is accepted. If the full
materialization fails, report the error and do not export a success-shaped
partial result.

For CSV or JSON input, replace `read_parquet` with `read_csv` or `read_json`.
Materialize a `row_number() OVER ()` in `staged` when the reader does not
expose a file row number. That generated number is stable only after the
staging table has been materialized.

These examples use temporary tables in an in-memory connection. They do not
create or modify a persistent user database. A requested persistent destination
is a separate, explicitly authorized final step.

## Long-running work and progress checkpoints

Do not execute a large remote expression as one silent statement. Use this
pattern when a run is expected to exceed about two minutes, 1,000 distinct
text inputs, or 25 slower media files.

Choose a checkpoint that is large enough to preserve vectorization but small
enough to finish in roughly one or two minutes. Typical starting points are
1,000-5,000 distinct text inputs or 5-20 media files. Adjust from the pilot;
time takes precedence over row count. If even a few dozen inputs are expected
to be slow, use checkpoints of 5-10 inputs. Do not create one API call or one
progress message per row.

Replace the one-shot `unique_results` materialization with deterministic,
non-overlapping work ranges:

```python
import time

checkpoint_size = 5_000

conn.execute(
    """
    CREATE TEMP TABLE unique_inputs AS
    SELECT
        row_number() OVER (ORDER BY input_text) AS work_id,
        input_text
    FROM (
        SELECT DISTINCT input_text
        FROM staged
        WHERE input_text IS NOT NULL
    )
    """
)
total = conn.sql("SELECT count(*) FROM unique_inputs").fetchone()[0]

conn.execute(
    """
    CREATE TEMP TABLE unique_results AS
    SELECT work_id, input_text, ai_process(input_text) AS ai_result
    FROM unique_inputs
    WHERE false
    """
)

started = time.monotonic()
for first_work_id in range(1, total + 1, checkpoint_size):
    last_work_id_exclusive = min(first_work_id + checkpoint_size, total + 1)
    conn.execute(
        """
        INSERT INTO unique_results
        SELECT work_id, input_text, ai_process(input_text) AS ai_result
        FROM unique_inputs
        WHERE work_id >= ? AND work_id < ?
        ORDER BY work_id
        """,
        [first_work_id, last_work_id_exclusive],
    )
    completed, succeeded, unresolved = conn.sql(
        """
        SELECT count(*), count(ai_result), count(*) FILTER (WHERE ai_result IS NULL)
        FROM unique_results
        """
    ).fetchone()
    elapsed_seconds = time.monotonic() - started
    percent = 100.0 if total == 0 else 100.0 * completed / total
    print(
        f"processed={completed}/{total} ({percent:.1f}%), "
        f"succeeded={succeeded}, unresolved={unresolved}, "
        f"elapsed_seconds={elapsed_seconds:.0f}"
    )
```

Translate each checkpoint into the user's language and send it through the
harness progress channel. Include the current phase, completed/total unique
inputs, percentage, elapsed time, succeeded/NULL/failed counts, and whether
processing continues. Estimate remaining time only after at least two
representative chunks and label it as an estimate. Do not expose internal work
IDs unless troubleshooting requires them.

Also announce phase changes for source reading, local profiling, remote
processing, result restoration, validation, and authorized output. If a local
query cannot be partitioned without changing semantics, say that the phase is
running before it starts and report immediately after it finishes; never
invent a percentage.

Each `INSERT` remains a batched statement; it is not a per-row API loop. If a
chunk fails, stop, state its work-ID range and the last completed checkpoint,
and do not export a final result. Previous chunks remain only in the temporary
in-memory table. Persisting checkpoints for cross-process resume requires
separate approval for the exact path/table and later cleanup.

The registered openaivec UDF also has a bounded in-process cache, which avoids
some repeated work inside the run. Treat it as a performance safety net, not
as the correctness mechanism or a cross-session cache. The explicit distinct
input table and one materialized result per input provide the auditable
deduplication contract.

## Complete-row input

When the task needs multiple columns, create one deterministic JSON value:

```sql
CREATE TEMP TABLE staged AS
SELECT
    order_id,
    to_json(struct_pack(
        customer := customer,
        product := product,
        issue := issue
    )) AS input_text
FROM orders;
```

Keep the primary key outside the JSON so it can be validated and joined
without asking the model to reproduce it.

## Intelligent missing-value fill

DuckDB has no dedicated `fillna` UDF. For contextual imputation, use DuckDB to
select a bounded, representative exemplar DataFrame, build a few-shot task with
`openaivec.task.table.fillna`, give that task a target-specific output type,
and register it with `task_udf`. Apply it only to distinct missing-row JSON,
materialize once, and join results back by the preserved key.

Do not choose `max_examples` by intuition. Follow the held-out masking,
candidate-count comparison, stopping rules, and production pattern in
[intelligent fill](intelligent-fill.md).

## Schema inference

Prefer an explicit Pydantic model. If the shape is genuinely unknown, infer
once from a bounded, non-NULL sample and reuse the resulting model:

```python
from openaivec.duckdb_ext import infer_schema, parse_udf

schema = infer_schema(
    conn,
    instructions="Extract product, issue category, and requested action.",
    example_table_name="staged",
    example_field_name="input_text",
    max_examples=50,
)
parse_udf(
    conn,
    "ai_parse",
    instructions=schema.inference_prompt,
    response_format=schema.model,
    batch_size=None,
    max_concurrency=8,
    reasoning={"effort": "none"},
)
```

Schema inference is a separate billable call. Show the inferred fields before
the full extraction.

## Many PDFs, images, or documents

Let DuckDB discover paths and let openaivec handle supported media:

```python
import duckdb
from pydantic import BaseModel, ConfigDict

from openaivec.duckdb_ext import responses_udf


class DocumentResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_type: str
    summary: str


conn = duckdb.connect()
conn.sql(
    "SELECT file AS input_path FROM glob(?)",
    params=["documents/**/*.pdf"],
).create_view("source_paths")

responses_udf(
    conn,
    "ai_document",
    instructions="Identify the document type and summarize only facts present in the document.",
    response_format=DocumentResult,
    multimodal=True,
    batch_size=1,
    max_concurrency=4,
    reasoning={"effort": "none"},
)

conn.execute(
    """
    CREATE TEMP TABLE document_results AS
    SELECT input_path, ai_document(input_path) AS ai_result
    FROM (SELECT DISTINCT input_path FROM source_paths)
    """
)
```

Local text-readable files are read and sent through the batched text path.
Images are inlined. Binary documents are uploaded as request-owned temporary
files and deleted after the request. Audio `.mp3` and `.wav` inputs are not
supported by the Responses wrapper. Local files must be no larger than 20 MiB.
Confirm that the chosen model/deployment accepts the selected media type.

## Embeddings

Deduplicate text before creating vectors:

```python
from openaivec.duckdb_ext import embeddings_udf

embeddings_udf(
    conn,
    "ai_embed",
    batch_size=128,
    max_concurrency=8,
)
conn.execute(
    """
    CREATE TEMP TABLE unique_embeddings AS
    SELECT input_text, ai_embed(input_text) AS embedding
    FROM (
        SELECT DISTINCT input_text
        FROM staged
        WHERE input_text IS NOT NULL
    )
    """
)
```

Use `similarity_search` only after vectors are persisted. It performs local
DuckDB cosine similarity and does not call OpenAI.

## Validation queries

Run checks against materialized tables, never against a remote UDF expression:

```sql
SELECT count(*) FROM staged;
SELECT count(*) FROM processed;
SELECT count(*) FROM processed WHERE input_text IS NOT NULL AND ai_result IS NULL;
SELECT source_file, source_row, count(*)
FROM processed
GROUP BY ALL
HAVING count(*) > 1;
```

For structured results, inspect `DESCRIBE processed` and project fields from
the stored `STRUCT`, such as `ai_result.label`.
