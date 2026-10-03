# Assistant-only performance and execution

Keep this implementation out of the business conversation. Users choose the
outcome, review quality, and approve scope; the assistant selects and measures
the execution strategy. Do not promise a fixed speedup or price.

## Fast path selection

| Input / outcome | Public integration | Starting settings |
| --- | --- | --- |
| Text classification, extraction, summaries, or translation | `responses_udf`, `task_udf`, or typed `parse_udf` | `batch_size=None`, `max_concurrency=8`, `reasoning={"effort": "none"}` |
| Supported binary documents or images | Structured `responses_udf` with `multimodal=True` | `batch_size=1`, `max_concurrency=4`, `reasoning={"effort": "none"}` |
| Plain-text document files | Read text locally, then the text path | Adaptive text batches, not individual media uploads |
| Semantic matching | `embeddings_udf`, then local `similarity_search` | `batch_size=128`, `max_concurrency=8`; obey provider limits |
| Contextual missing-value fill | Evaluated `table.fillna` task and `task_udf` | Holdout quality gate first; adaptive production batches |

These are starting settings, not universal optima. Preserve approved model and
Azure deployment names and Fabric built-in defaults. Keep API options inside
the UDF registration, never on `BulkRunner`.

## Throughput rules

1. Push authorized source filters and column selection before the remote step.
   Avoid copying an entire database or adding unused text to the prompt.
2. Extract related output fields together in one structured call. Use short,
   stable instructions and an explicit model with `ConfigDict(extra="forbid")`.
3. Globally deduplicate the exact semantic input before remote evaluation.
   Text from different rows shares results only when all context needed for
   interpretation matches. Keep row keys and irrelevant lineage outside the
   input; include a branch, locale, or product field only when meaning needs it.
4. Reuse accepted pilot mappings explicitly. A bounded in-process UDF cache
   may evict entries; it is not the audit or resume mechanism.
5. Register once and materialize once per distinct input. Restore source rows
   with a local join. Preview, project multiple structured fields, group, and
   export only the stored result.
6. Keep concurrency bounded. Do not spawn one agent, process, or API loop per
   record. Do not submit several concurrent queries on one connection or
   multiply independent UDF workers that defeat the intended provider budget.
7. Let adaptive text batching measure request performance. Checkpoints are
   progress/commit boundaries, not API batch sizes. Preserve enough work in
   each checkpoint for Arrow vectorization and concurrency.
8. On 429/timeouts, stop the failed chunk and discuss a slower recovery or
   an explicit bounded public `RetryPolicy`. Do not wrap the UDF in a second
   transport retry loop. Keep completed mappings and separate validation
   corrections from transport attempts.

NULL inputs never leave the process. Treat blank text as NULL only under an
agreed blank-input rule; preserve raw source values. Media deduplication is
by exact input path/URL, not a guarantee that equal file contents at different
paths are processed once. Do not fetch signed URLs to manufacture a cache key.

## Canonical staged execution

Use `scripts/bulk_runner.py` from the installed Skill. It is an importable
assistant helper, not a CLI for the business user. Load it through the
harness's Python module mechanism with the installed `scripts` directory on
the import path; do not assume the user's current directory is the Skill.

Keep a single Python kernel or supported stateful execution session for
registration, pilot, approval, full run, and output. Execute the following
stages separately at their consent gates, not as one unattended script.
The helper imports DuckDB only, accepts the connection and registered public
openaivec UDF, and never opens a source, installs software, or writes a file.

If the harness cannot preserve Python state across an approval pause, explain
"A temporary saved preview is needed to continue without repeating paid
work." Ask separately for an exact dedicated checkpoint location and its
cleanup, or offer an already-managed notebook environment. Never silently
persist results, restart an unapproved run, or promise cross-process resume
from the in-memory helper. Approved checkpoint workflows must verify the
source and identical provider/model/prompt/schema/options before restoring
accepted mappings; a changed configuration requires a new pilot.

### 1. Local setup and staging

Follow `safe-data-io.md`. Start in memory with automatic extension installation
and disk spill disabled. If memory is insufficient, stop for a separately
approved temporary-workspace plan; do not quietly let DuckDB spill to disk.

```python
import duckdb

conn = duckdb.connect(
    config={"autoinstall_known_extensions": False, "temp_directory": ""}
)
```

Materialize an immutable `staged` temporary table with:

- unique non-NULL `row_id`, assigned once in the desired source order;
- VARCHAR `input_text`, containing only the approved semantic input;
- source identifiers, file lineage, raw fields, and any user-required columns.

Do not use `ai_result` for a source column; it is the helper's output column.
Generated row IDs preserve staged order within this run, not a durable
business identity. Preserve the original key alongside them.

Register the chosen public openaivec UDF once as `ai_process`, with the
structured schema and settings in the profile table.

```python
from bulk_runner import BulkRunner

runner = BulkRunner.of(
    conn,
    source_table="staged",
    udf_name="ai_process",
    prefix="bulk",
    checkpoint_size=2048,
    on_progress=report_progress,
)
plan = runner.plan
```

`report_progress` is the assistant's harness progress callback. Translate
events into business language; do not print raw dictionaries or source values.
The plan is local: source rows, non-NULL rows, distinct inputs, and repeated
evaluations avoided. The helper prepares an empty typed result table without
calling the registered UDF.

### 2. Approved pilot

Only after the user approves the selected service, data boundary, and small
billable preview:

```python
pilot = runner.pilot(approved=True, size=5)
```

The default pilot spans input-length buckets; it is not a claim of statistical
representativeness. Prefer three to ten approved work IDs covering business
segments, long/short input, languages, and unusual records:

```python
pilot = runner.pilot(approved=True, work_ids=approved_work_ids)
```

Read the stored pilot locally. Inspect structure, allowed categories,
evidence, NULL/review handling, and the agreed acceptance checks. Do not expose
raw source text in chat without its separate sample-disclosure approval.
If the pilot is rejected, change the configuration and start a new UDF/run
namespace. The helper's approval flags record the assistant's consent
decision; they are not a replacement for actually asking the user.

### 3. Accepted pilot and approved full scope

Only after pilot acceptance and full-run approval:

```python
processed = runner.run(pilot_approved=True)
```

The helper skips already materialized work IDs, including accepted pilot
results and NULL results. It submits each pending checkpoint as a vectorized
SQL statement, restores every source row, and returns the validated result
ordered by `row_id`. The loop is over checkpoints, never individual requests.

Start with a 2,048-input checkpoint. After the pilot, use measured throughput
to choose a checkpoint expected to finish in roughly 60-120 seconds, keeping
it within the approved in-memory budget. Media checkpoints normally need
5-20 files instead. `runner.checkpoint_size` may change without changing the
interpretation configuration. Do not tune by sending extra billable benchmark
requests without approval.

Events report committed unique inputs, successes, unresolved NULL results,
and active processing time excluding user approval waits. Emit a phase
message while a chunk runs; refine the ETA after two representative completed
checkpoints. Do not manufacture per-row progress inside a remote statement.

### 4. Local inspection and authorized output

```python
preview = runner.result().limit(10)
summary = conn.sql(
    "SELECT ai_result.category, count(*) FROM bulk_processed GROUP BY ALL"
)
```

The summary's `category` is an example; use fields from the agreed schema.
Neither query calls the API. Check source/output counts, keys, categories,
evidence, review flags, failed conversions, and summary denominators.

Follow `safe-data-io.md` for the exact authorized new output: exclusive
reservation, no overwrite, re-read validation, and separate cleanup consent
on failure. Flatten a stored structured result for CSV/Excel; prefer Parquet
when nested results, high row counts, or types make it more suitable.

## Failure and resume semantics

A failed statement raises; it never becomes an empty successful result.
Previously committed chunks remain in the helper's temporary tables. There
is no final output until the full mapping and restored row count validate.
After an explicit recovery decision in the same process, `run` skips those
committed chunks. Do not switch model/prompt/schema and reuse old mappings.
Execution-only limits such as concurrency or bounded transport retry settings
may be reduced during approved recovery; interpretation-affecting options
must stay identical.

A failed chunk or a provider retry may already have sent billable requests.
The guarantee is one materialized mapping per unique input under one
configuration, not exactly-once delivery to a remote provider. Identify the
unfinished scope, explain possible retry charges, and retain failed/unknown
status honestly. NULL model results stay visible for review and are not
automatically retried to force success.
