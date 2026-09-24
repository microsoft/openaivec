# Safe data input and output

Read this reference whenever a request names Excel, CSV/TSV, a relational
database, an output destination, or any operation that could change existing
data.

## Routing and conversation

Treat Excel, CSV, and supported relational sources as strong routing signals
when the user asks for an AI-suitable repeated operation such as
classification, extraction, normalization, translation, embedding, semantic
search, or intelligent fill. The user does not need to mention OpenAI,
openaivec, DuckDB, SQL, or a UDF.

Pure file conversion, row preview, database administration, workbook
formatting, and ordinary deterministic updates remain outside this skill.

Use the user's terminology in conversation:

- say "Excel workbook", "CSV file", "source table", and "new result table";
- do not volunteer the internal query engine, extension, UDF, or staging-table
  names;
- disclose implementation details accurately when the user asks, when a
  compatibility boundary needs explanation, or when an error depends on them.

## Supported boundary

Use only capabilities available in the installed DuckDB version:

| Source or destination | Scope |
| --- | --- |
| CSV/TSV | Native tabular read and write; confirm delimiter, header, encoding, and inferred types |
| Parquet | Native columnar read and write |
| JSON/NDJSON | Supported JSON read and write |
| Excel | `.xlsx` through the available Excel extension; `.xls` is unsupported |
| SQLite | Read/write through the available official SQLite extension |
| PostgreSQL | Read/write through the available official PostgreSQL extension |
| MySQL | Read/write through the available official MySQL extension |
| Other relational systems | Only when the installed DuckDB version has an available connector that supports the exact requested operation |

Do not install an unrelated parser or database client to widen this boundary.
If support is unavailable, ask the user to export the required rows to CSV or
Parquet, or to provide another already-supported source.

Excel support is for tabular values. Do not promise to preserve or edit
formulas, styles, charts, pivot tables, macros, external links, or other
workbook behavior. Prefer CSV or Parquet for outputs too large for a practical
worksheet.

## Authorization matrix

Authorization is operation-specific and target-specific:

| Requested operation | Minimum authorization | Required behavior |
| --- | --- | --- |
| Read | The user identifies or supplies the source | Open files and attached databases read-only where supported |
| Internal staging | No extra authorization | Use only an in-memory connection and temporary tables; clean up at process end |
| Create a new file/table | The user says save, write, export, or create and names the destination | Reserve or create exclusively and fail if the destination already exists |
| Append/insert | The user explicitly says append or insert and names the existing target | Validate schema, keys, duplicate policy, and row count before a transaction |
| Update/merge | The user explicitly says update or merge, names the target, and defines keys/scope | Show the proposed matched and changed row counts before a transaction |
| Overwrite/replace | The user explicitly says overwrite or replace and names the exact target | Propose a new version first; use an atomic swap only when supported |
| Delete/truncate/drop | The user explicitly names the destructive operation, exact target, and row/object scope | Show affected keys/counts, require immediate confirmation, and use rollback when supported |
| Install an extension | The user explicitly approves installation after the extension and reason are stated | Install only a supported extension from the configured trusted repository |

`save to results.csv` authorizes creating a new `results.csv`; it does not
authorize replacing an existing file. `write to analytics.results_new`
authorizes creating that new table; it does not authorize appending if the
table already exists. Ambiguous phrases such as "put it back" or "sync it"
are not mutation authorization.

Authorization does not carry over to another path, table, schema, database, or
operation. If any resolved target differs from the request, stop.

A persistent staging table, backup, snapshot, or copied workbook is another
write. Name it and obtain explicit create and cleanup authorization separately;
do not infer that permission from an overwrite or deletion request.
Creating a missing parent directory or database schema is also a separate
write and requires explicit authorization.

Creating a new output does not authorize deleting an incomplete output after a
failure. Ask whether failure cleanup is allowed; otherwise leave the exact
newly created path in place, label it incomplete, and report it.

## Read-only workflow

1. Start with an in-memory connection.
2. Disable automatic extension installation before probing capabilities:

```python
import duckdb

conn = duckdb.connect()
conn.execute("SET autoinstall_known_extensions = false")
available_extensions = conn.sql(
    """
    SELECT extension_name, installed, loaded
    FROM duckdb_extensions()
    ORDER BY extension_name
    """
).fetchall()
```

3. Inspect file existence or database metadata without writing. Do not create a
   sidecar database or local cache file.
4. For a relational source, use a read-only attachment when the connector
   supports it and use a least-privilege read account.
5. Keep passwords, tokens, and complete connection strings in environment
   variables or a secret manager. Never paste them into chat, print them, log
   them, or embed them in generated SQL.
6. Read only required columns and rows. Preserve a stable business key and use
   predicates to avoid copying an entire remote database unnecessarily.
7. Inspect schema, row counts, NULL counts, distinct billable inputs, and only
   approved sample values before any OpenAI call.

Loading an already-installed extension into the current process is ephemeral.
Downloading or installing an extension is a network request and persistent
local change, so it requires explicit approval.

Keep batching caches in process memory. Do not silently enable persistent
caching or disk spill. If the workload cannot complete in memory, state the
estimated temporary-space need, request approval to create and later remove a
new dedicated scratch directory, keep it separate from every
source/destination, and delete only that exact approved directory after
validation or failure.

## Safe output workflow

The default proposal is a new, versioned destination:

- `survey_enriched_20260924T183656.csv` instead of `survey.csv`;
- `survey_enriched_20260924T183656.xlsx` instead of the source workbook;
- `analytics.ticket_analysis_20260924` instead of updating
  `operations.tickets`.

Before any persistent write:

1. Resolve and display the exact user-facing destination and operation.
2. Check whether it exists without changing it.
3. If it exists and overwrite/append/update was not explicitly authorized,
   stop. Do not choose another existing target silently.
4. Materialize and validate the complete result in temporary in-memory tables.
5. Verify source/output row counts, key uniqueness, NULL behavior, allowed
   values, and rejected rows.
6. Immediately before a new file write, reserve the exact path with an
   operating-system exclusive create. Do not rely on the earlier existence
   check because another process could create the path between the check and
   write.
7. For a new database table, use ordinary `CREATE TABLE` inside the authorized
   catalog/schema, never `CREATE OR REPLACE`; creation must fail if it exists.
8. Use session-temporary staging. If persistent staging is required, obtain
   separate authorization for its exact name and later cleanup.
9. Re-read the new output and repeat critical row-count and schema checks.
10. Report the destination, operation, succeeded/rejected counts, and whether
   the source was untouched.

Do not use `CREATE OR REPLACE`, `INSERT OR REPLACE`, `COPY ... OVERWRITE`,
`DROP`, `TRUNCATE`, or a write directly to an existing user path as a
convenience.

DuckDB file writers can replace an existing file. Reserve a new path first:

```python
import os
from pathlib import Path


def reserve_new_file(path: Path) -> None:
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(descriptor)
```

Only call the writer after `reserve_new_file` succeeds. If writing or
validation fails, do not delete the incomplete path unless that exact cleanup
was explicitly authorized.

## File-specific safeguards

### Excel

- Accept `.xlsx`; reject legacy `.xls`.
- Require or discover the intended sheet without modifying the workbook.
- Treat formula cells as source values exposed by the reader; do not claim to
  preserve formula logic in a generated workbook.
- Write a new data-only workbook. Never use the source workbook as the output
  path unless the user explicitly requests overwrite after being warned that
  workbook features may not be preserved.
- Follow [Excel support setup](excel-setup.md). Check without side effects; if
  support is missing, explain the impact and choices in plain language and
  install the official signed component only after explicit consent.

### CSV and TSV

- Inspect delimiter, quote, escape, header, encoding, date, decimal, and NULL
  conventions. Do not silently accept a faulty parse.
- Preserve the source file. Write to a new path with explicit header and
  delimiter settings.
- Re-read the generated file and compare row count, column names, and key
  uniqueness before reporting success.

### Parquet and JSON

- Keep schema-union behavior explicit for multi-file reads.
- Write to a new file or new partition root and fail if it already exists.
- Do not delete old partitions as part of a successful write unless deletion
  was separately authorized.

## Relational database safeguards

PostgreSQL, MySQL, and SQLite official extensions can read and write, but
write capability is not permission to mutate.

For reads:

- attach read-only where supported;
- use a read-only database identity;
- qualify catalog, schema, and table names;
- avoid `USE` or changing the default catalog; and
- push filters and projections to the source.

For an explicitly authorized new table:

1. Use a new run-specific table name in the authorized schema.
2. Fail if it exists.
3. Create and populate it inside a transaction when the connector provides the
   required atomicity.
4. Validate counts, schema, constraints, and sample keys before commit.
5. Roll back on any validation failure.

For append, update, merge, or delete:

1. Produce a read-only diff with exact matched, changed, inserted, unchanged,
   rejected, and deleted counts.
2. Validate that every target row is selected by a stable key; reject a
   non-keyed broad mutation.
3. Offer a recoverable copy or target-native snapshot where available; create
   it only after separate authorization for its exact name and cleanup.
4. Execute the smallest change inside a transaction.
5. Validate before commit and roll back on mismatch.
6. If the connector cannot guarantee the needed atomicity or rollback, do not
   perform the in-place mutation; offer a new table instead.

Never create databases, users, roles, grants, network rules, or server
resources. Those are database administration and remain outside this skill.

## User-facing safety statements

For a read-only request:

> I will read only the named source and use the requested columns. No file or
> database table will be created, updated, overwritten, or deleted.

For an authorized new output:

> You asked for a new result at the named destination. I will stop rather than
> replace it if that destination already exists, and I will leave the source
> unchanged.

For an existing-object mutation:

> Changing the existing target is riskier. The safest option is a new versioned
> output. Before any in-place change, I will show the exact target and affected
> rows and require explicit authorization for that operation.

## References

- [DuckDB CSV import](https://duckdb.org/docs/current/data/csv/overview.html)
- [DuckDB Excel extension](https://duckdb.org/docs/current/core_extensions/excel.html)
- [DuckDB PostgreSQL extension](https://duckdb.org/docs/current/core_extensions/postgres/overview.html)
- [DuckDB MySQL extension](https://duckdb.org/docs/current/core_extensions/mysql.html)
- [DuckDB SQLite extension](https://duckdb.org/docs/current/core_extensions/sqlite.html)
- [DuckDB extension overview](https://duckdb.org/docs/current/extensions/overview.html)
- [DuckDB attachment options](https://duckdb.org/docs/current/sql/statements/attach.html)
- [DuckDB transactions](https://duckdb.org/docs/current/sql/statements/transactions.html)
