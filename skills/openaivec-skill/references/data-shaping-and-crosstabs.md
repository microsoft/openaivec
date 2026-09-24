# Business-safe data shaping and cross-tabs

Use this reference after the request has routed to this skill because it
contains repeated interpretive work. DuckDB may perform deterministic
profiling, cleaning, joining, grouping, and cross-tabs internally before or
after the AI-backed step.

Pure preview, sorting, arithmetic, format conversion, or cross-tab requests
without an openaivec-suitable interpretive operation remain outside this
skill. Internal use of DuckDB does not widen the routing boundary.

## Required order

1. Inspect technical facts without changing the source.
2. Establish the business meaning of rows, columns, codes, units, and dates.
3. Record the agreed population, dimensions, measures, and denominators.
4. Shape data in temporary views or tables while preserving raw values and
   stable keys.
5. Materialize any AI-derived fields exactly once.
6. Aggregate the materialized result locally.
7. Reconcile counts and totals before presenting or writing anything.

Do not start an aggregation while a material ambiguity can change its meaning.

## Business-semantics gate

Technical metadata can establish data type, NULL rate, distinct count, range,
and candidate keys. It cannot establish business meaning. A column called
`amount`, `date`, `status`, `region`, `customer`, or `count` is not
self-defining.

Before shaping or aggregating, record this contract:

| Item | Question to resolve |
| --- | --- |
| Row grain | What does one row represent: an order, order line, shipment, case event, survey response, or something else? |
| Stable key | Which field or field combination identifies that business object? |
| Population | Which rows belong in the result, and which cancellations, tests, reversals, or duplicates are excluded? |
| Dimensions | What do grouping codes mean, and are labels mutually exclusive and stable over time? |
| Measures | Is a value additive, a balance, a rate, an average, a score, or an identifier that must never be summed? |
| Time | Which business event does the date represent, in which timezone and fiscal calendar? |
| Units | Which currency, quantity unit, tax basis, scale, and sign convention apply? |
| Missing values | Does NULL, blank, zero, `unknown`, or `N/A` have a distinct business meaning? |
| Join and deduplication | Which relationship and tie-breaker make a join or latest-record selection valid? |
| Output | Which counts, totals, percentages, ordering, and display labels does the user need? |

An authoritative data dictionary, schema comment, or definition supplied by
the user can answer an item. Column names, guessed acronyms, and value patterns
cannot.

## How to ask when meaning is ambiguous

Ask only when the ambiguity can materially change filtering, grouping,
joining, deduplication, a measure, or a denominator. Do not burden the user
with implementation-only questions.

1. Ask the highest-impact unresolved question first, normally row grain.
2. Ask exactly one question at a time.
3. When two or more meanings are plausible, use a single-select question with
   two to five mutually exclusive choices.
4. Always allow **Other / none of these** with free-text input. If the harness
   automatically adds a free-text option, use it rather than duplicating an
   explicit `Other` choice.
5. Ask in the user's language. Phrase choices in business terms and explain
   any code only in parentheses.
6. Do not mark an option recommended unless authoritative metadata makes it
   materially more likely; avoid anchoring the user to a guess.
7. Record the answer, then ask the next unresolved question in a later turn.
   Never bundle grain, amount, date, and NULL semantics into one question.

Example question:

> What does one row in this table represent?

Single-select choices:

- One customer order
- One product line within an order
- One shipment
- Other / none of these — describe it

After that answer, a separate question may ask:

> Which business amount should this report total?

Single-select choices:

- Gross invoiced amount before discounts
- Net invoiced amount after discounts
- Cash received
- Other / none of these — describe it

For a column such as `region_code`, candidate choices might be customer
location, shipping destination, sales organization, or reporting region. Show
only candidates supported by surrounding metadata or user-approved samples.

If there are no defensible candidates, ask one focused free-text question
instead. If the user does not know, limit the result to unambiguous counts or
present separately labelled scenarios. Never silently choose one meaning.

## Profile before transformation

Use an in-memory connection. Read-only profiling may include:

```sql
DESCRIBE source_raw;

SELECT
    count(*) AS row_count,
    count(DISTINCT record_id) AS distinct_record_ids,
    count(*) FILTER (WHERE record_id IS NULL) AS missing_record_ids,
    count(*) FILTER (
        WHERE region_code IS NULL
           OR trim(CAST(region_code AS VARCHAR)) = ''
    ) AS missing_region_codes
FROM source_raw;
```

Inspect schema, counts, distinct counts, NULLs, ranges, and code frequencies
before raw samples. Minimum and maximum values, rare codes, and free text may
still be sensitive; do not expose them in conversation without approval.

Profile candidate dimensions in long form:

```sql
SELECT region_code, count(*) AS row_count
FROM source_raw
GROUP BY region_code
ORDER BY row_count DESC, region_code;
```

High cardinality, mixed types, unexpected NULLs, or a non-unique proposed key
are reasons to clarify or repair the contract before remote processing.

## Basic deterministic shaping

Create temporary views or tables and keep raw values beside derived values:

```sql
CREATE TEMP VIEW shaped AS
SELECT
    record_id,
    region_code AS region_code_raw,
    NULLIF(trim(CAST(region_code AS VARCHAR)), '') AS region_code,
    amount_text AS amount_raw,
    TRY_CAST(amount_text AS DECIMAL(18, 2)) AS amount,
    event_date_text AS event_date_raw,
    TRY_CAST(event_date_text AS DATE) AS business_date,
    comment_text
FROM source_raw;
```

Then quantify rejected conversions:

```sql
SELECT
    count(*) FILTER (
        WHERE amount_raw IS NOT NULL AND amount IS NULL
    ) AS rejected_amounts,
    count(*) FILTER (
        WHERE event_date_raw IS NOT NULL AND business_date IS NULL
    ) AS rejected_dates
FROM shaped;
```

Use `TRY_CAST` so invalid source values become visible validation failures
rather than aborting midway. Do not remove currency signs, convert timezones,
translate codes, reinterpret zero as NULL, or combine units until the business
contract defines the operation.

Allowed local shaping includes:

- selecting and clearly renaming columns;
- trimming whitespace and normalizing confirmed missing markers;
- casting with rejected-value counts;
- parsing dates using an agreed format and business timezone;
- filtering to an agreed population;
- mapping codes through an approved lookup table;
- projecting fields from a materialized structured AI result;
- joining under a validated relationship;
- deduplicating under an agreed business key and tie-breaker; and
- creating long-form summaries and cross-tabs.

Preserve stable keys and lineage columns through every step. Do not overwrite
the only copy of a raw value with a cleaned or inferred value.

## Joins and deduplication

Before a join, verify uniqueness on the side expected to be one row per key:

```sql
SELECT customer_id, count(*) AS rows_per_key
FROM customer_dimension
GROUP BY customer_id
HAVING count(*) > 1;
```

Stop on unexpected duplicates. A many-to-many join can multiply rows and
inflate every later total.

Deduplicate only after the business key and tie-breaker are confirmed:

```sql
CREATE TEMP TABLE latest_cases AS
SELECT * EXCLUDE (row_rank)
FROM (
    SELECT
        *,
        row_number() OVER (
            PARTITION BY case_id
            ORDER BY updated_at DESC, source_row DESC
        ) AS row_rank
    FROM shaped
)
WHERE row_rank = 1;
```

Do not assume "latest" is correct merely because an update timestamp exists.
Reconcile row counts before and after every join or deduplication.

## Materialize AI results before aggregation

Run the AI-backed function only on the approved distinct inputs, materialize
the mapping once, and join it back to preserved source rows. Project the
structured fields into a temporary table:

```sql
CREATE TEMP TABLE enriched AS
SELECT
    p.record_id,
    p.region_code,
    p.business_date,
    p.amount,
    p.ai_result.sentiment AS sentiment,
    p.ai_result.theme AS theme
FROM processed AS p;
```

Never place a remote UDF inside `GROUP BY`, `PIVOT`, a window expression, or a
query that may be re-executed. All summaries after this point must be local.

## Long-form aggregation first

Long form is the safest default because new or rare categories remain rows
instead of silently changing the output schema:

```sql
SELECT
    region_code,
    sentiment,
    count(*) AS response_count
FROM enriched
WHERE sentiment IS NOT NULL
GROUP BY region_code, sentiment
ORDER BY region_code, sentiment;
```

Use exact distinct counts when the metric is a business entity count:

```sql
SELECT
    region_code,
    count(DISTINCT record_id) AS distinct_responses
FROM enriched
GROUP BY region_code
ORDER BY region_code;
```

Do not substitute row counts for customer, order, case, or document counts
unless the row grain proves they are equivalent.

## Fixed cross-tabs with explicit denominators

For stable categories, conditional aggregation produces predictable columns
and makes denominators visible:

```sql
SELECT
    region_code,
    count(*) AS all_rows,
    count(sentiment) AS classified_rows,
    count(*) FILTER (WHERE sentiment = 'positive') AS positive_rows,
    count(*) FILTER (WHERE sentiment = 'neutral') AS neutral_rows,
    count(*) FILTER (WHERE sentiment = 'negative') AS negative_rows,
    round(
        100.0 * count(*) FILTER (WHERE sentiment = 'positive')
        / NULLIF(count(sentiment), 0),
        1
    ) AS positive_pct_of_classified
FROM enriched
GROUP BY region_code
ORDER BY region_code;
```

Report both `all_rows` and `classified_rows`; otherwise a high NULL or failure
rate can make percentages misleading. Define whether unknown categories belong
in the denominator before calculating a rate.

DuckDB `PIVOT` is useful when the user wants a matrix:

```sql
PIVOT (
    SELECT region_code, sentiment
    FROM enriched
    WHERE sentiment IS NOT NULL
)
ON sentiment IN ('positive', 'neutral', 'negative')
USING count(*)
GROUP BY region_code
ORDER BY region_code;
```

Use an explicit `IN` list when a stable output schema matters. Keep a long-form
result when categories are high-cardinality, user-generated, or expected to
change.

## Measures, time, and totals

- Sum only additive measures at the chosen grain.
- Never sum identifiers, percentages, rates, balances from different dates,
  or values with mixed currency or units.
- Do not average pre-aggregated averages. Recompute from additive numerator
  and denominator when they are available.
- For weighted averages, confirm the weight and use
  `sum(value * weight) / NULLIF(sum(weight), 0)`.
- Resolve business date, timezone, and fiscal calendar before using
  `date_trunc`.
- Keep full-precision values internally and round only presentation columns.
- Use separate labelled totals for intentionally different populations rather
  than blending them.

## Validation and completion

Before presenting or writing a summary:

1. Reconcile source, shaped, enriched, included, excluded, and rejected rows.
2. Verify key uniqueness and expected join cardinality.
3. Reconcile each long-form or cross-tab total to its declared denominator.
4. Count NULL, unknown, failed, and unmapped categories explicitly.
5. Confirm that dimensions are mutually exclusive when the table assumes they
   are.
6. Check that time buckets, units, currency, signs, and filters match the
   agreed contract.
7. Confirm the source stayed unchanged and that any destination was separately
   authorized.

Report the agreed row grain, measure definitions, dimensions, filters,
denominators, rejected conversions, unmatched joins, unresolved ambiguities,
and validation counts in user language. Mention DuckDB or SQL only when the
user asks or troubleshooting requires it.

## References

- [DuckDB aggregate functions](https://duckdb.org/docs/current/sql/functions/aggregates.html)
- [DuckDB GROUP BY](https://duckdb.org/docs/current/sql/query_syntax/groupby.html)
- [DuckDB PIVOT](https://duckdb.org/docs/current/sql/statements/pivot.html)
- [DuckDB casting](https://duckdb.org/docs/current/sql/expressions/cast.html)
