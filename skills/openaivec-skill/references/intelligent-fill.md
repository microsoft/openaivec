# Intelligent fill

Use this reference when a user asks to infer a missing table value from the
other columns. This is contextual imputation with a few-shot prompt, not an
ordinary SQL default and not recovery of an unknown fact.

## Scope and safety

Use intelligent fill only when:

- the target has trustworthy non-NULL examples;
- the permitted predictor columns contain enough context to predict it;
- an inferred value is acceptable for the business process;
- uncertainty can remain `NULL` or enter a human-review queue; and
- the user has approved the provider, model, fields leaving the machine, cost,
  and destination.

Prefer deterministic DuckDB rules such as `COALESCE`, joins to reference data,
or arithmetic when they can produce the correct value. Do not use intelligent
fill for authoritative identifiers, account numbers, invoice or payment
amounts, legal or medical facts, safety-critical settings, or employment,
credit, insurance, payment, or eligibility decisions.

Exclude columns that leak the answer, contain post-outcome information, expose
unnecessary sensitive data, or have no predictive meaning. Keep stable row
keys outside the model input.

## What openaivec `fillna` does

`df.ai.fillna("target")`, `await df.aio.fillna("target")`, and
`openaivec.task.table.fillna(df, "target")` derive demonstrations from rows
where the target is known. Each demonstration is a complete JSON row with the
target replaced by `null`; its output is an object containing only `output`.

Important implementation limits:

- `max_examples` defaults to 8, but it is a maximum, not the number guaranteed
  to enter the prompt.
- Sampling is deterministic, but it is not business-stratified.
- Examples are bounded to 6,000 characters and an estimated 1,500 tokens.
- Duplicate or oversized examples can be skipped. If none fit, construction
  falls back to zero-shot instructions.
- Task construction is local by default. `improve_prompt=True` is a separate
  billable prompt-refinement call and must not be mixed into example-count
  experiments.
- The DataFrame accessors update only missing positions and leave the source
  DataFrame unchanged.

Eight is therefore a convenient initial cap, not a research-backed universal
optimum.

## How many examples

There is no fixed answer across columns, datasets, models, or model snapshots.
Choose the count empirically for the exact target and production distribution.

Use this candidate schedule:

1. Compare 1, 2, 4, and 8 examples.
2. If 8 still gives a material, predeclared improvement over 4 and the
   effective prompt count reaches 8, compare 12 and 16.
3. Select the smallest count that passes every overall and important-segment
   quality threshold.
4. Stop increasing the count when the prompt budget prevents the effective
   count from increasing, two consecutive increases fail the predeclared
   minimum improvement, or the task remains below the acceptance threshold.

The doubling schedule is an economical search procedure, not a claim that
these counts are universally optimal. Never run the full missing set merely to
decide the number of examples.

## Required evaluation procedure

### 1. Define the decision before calling the API

Record:

- target column and exact allowed output type or values;
- permitted predictor columns;
- model or deployment and pinned model snapshot when available;
- business segments that need separate reporting;
- primary metric, segment-level metrics, acceptance thresholds, and the
  minimum improvement that justifies more examples;
- deterministic baseline and human-review policy; and
- maximum evaluation and production cost.

Keep the same model, instructions, reasoning effort, validation retries, and
output schema for every candidate count. Evaluate prompt refinement, if
needed, only after choosing the example count.

### 2. Create a leakage-safe holdout

Use only rows whose target is known and verified. Split them into a candidate
example pool and a disjoint holdout. Mask the target in holdout inputs while
retaining the actual value separately for scoring.

- Make the masked holdout resemble the rows that are actually missing. Random
  masking can overstate quality when missingness is concentrated in a region,
  product, time period, workflow, or target value.
- Keep exact and near duplicates in the same side of the split.
- Do not include a row identifier or another column that directly encodes the
  answer.
- Cover common cases, rare but important cases, every allowed categorical
  value where feasible, and relevant regions, products, languages, or time
  periods.
- Keep a final untouched confirmation set when the dataset is large enough.
- If there are too few verified rows to cover important segments, do not
  impute; collect or review more examples.

The number of evaluation rows is separate from the few-shot example count.
Make the holdout large enough to represent each required segment and report
the score numerator, denominator, and uncertainty rather than claiming
precision from a tiny sample. Treat candidate differences within the measured
uncertainty as a tie and choose the smaller count.

### 3. Order a representative, nested example pool

Quality depends on which examples are selected, not only how many. Build a
deterministic order that interleaves target values and important business
segments. The 1-example set must be a subset of the 2-example set, which must
be a subset of the 4-example set, and so on. This keeps the selected sets
nested, although the current `fillna` implementation may deterministically
reorder each set when it builds the prompt.

Do not pass a large uncurated table and assume deterministic random sampling is
representative. For a controlled experiment, pass exactly the first `n`
preselected rows to `fillna(..., max_examples=n)`.

### 4. Run the same masked holdout for every count

The following evaluation skeleton assumes DuckDB has already produced
`exemplar_order` and `holdout` pandas DataFrames from approved columns. It uses
the task factory explicitly so evaluation examples and holdout rows remain
disjoint.

```python
from xml.etree import ElementTree

import pandas as pd

from openaivec import pandas_ext
from openaivec.task.table import fillna

target = "resolution_code"
model_columns = ["customer_segment", "product_family", "issue_type", target]
candidate_counts = [1, 2, 4, 8]

actual = holdout[target].copy()
masked_holdout = holdout[model_columns].copy()
masked_holdout[target] = None


def effective_example_count(instructions: str) -> int:
    if "<Examples>" not in instructions:
        return 0
    root = ElementTree.fromstring(instructions)
    return len(root.findall("./Examples/Example"))


scores = []
for count in candidate_counts:
    examples = exemplar_order.iloc[:count][model_columns].copy()
    task = fillna(examples, target, max_examples=count)
    predictions = masked_holdout.ai.task(
        task,
        batch_size=64,
        show_progress=False,
        reasoning={"effort": "none"},
    )
    predicted = pd.Series(
        [result.output for result in predictions],
        index=holdout.index,
    )
    correct = predicted.eq(actual)
    segment_exact_match = (
        holdout.assign(correct=correct).groupby("customer_segment", dropna=False)["correct"].mean().to_dict()
    )
    scores.append(
        {
            "requested_examples": count,
            "effective_examples": effective_example_count(task.instructions),
            "exact_match": correct.mean(),
            "null_rate": predicted.isna().mean(),
            "segment_exact_match": segment_exact_match,
        }
    )

score_table = pd.DataFrame(scores)
print(score_table)
```

Keep the evaluation batch size fixed across candidate example counts so the
comparison is operationally consistent. After selecting the prompt and example
count, the production DuckDB task uses `batch_size=None` for adaptive text
batching.

Counting `<Example>` elements is a diagnostic for the current prompt format.
Record it because `max_examples` alone does not prove how many examples fit.
Do not log the prompt when rows contain sensitive data.

For a production decision, repeat close candidate counts across multiple
predeclared permutations of the same representative example sets when
sufficient verified data and evaluation budget exist. Compare aggregate and
worst-segment results. If only one order can be tested, report order
sensitivity as an unmeasured limitation.

This loop intentionally performs billable evaluation calls. Run the
environment and authentication preflight first. If authentication returns
401/403, stop and follow
[authentication recovery](authentication.md); never ask the user to paste a
secret into chat.

### 5. Score for the target type

- **Categorical:** exact match, macro-F1 or balanced accuracy, invalid value
  rate, `NULL` rate, and metrics for every important segment.
- **Numeric:** mean or median absolute error, business-tolerance pass rate,
  range/unit violations, and comparison with a median, grouped median, or
  other deterministic baseline.
- **Boolean:** class-specific recall and precision, especially for the costly
  error direction.
- **Free text:** a fixed human rubric for factual support, usefulness, and
  unsupported invention. Do not use exact string match alone.

Reject a candidate that improves the average while materially harming a
required segment. If intelligent fill does not beat the deterministic baseline
or meet the acceptance gate, leave values `NULL`; do not conceal failure by
adding examples indefinitely.

### 6. Confirm and freeze

Run the selected count once on the untouched confirmation set when available.
Then freeze and record:

- exemplar row keys or a versioned exemplar query;
- requested and effective example counts;
- model/deployment and snapshot;
- ordered predictor columns and target type;
- evaluation dataset version and overall/segment metrics; and
- prompt hash, package version, and acceptance decision.

Re-run the evaluation after changing the model, exemplar data, column set,
prompt behavior, package version, or target taxonomy, and monitor accepted
fills against later human corrections.

## Production DuckDB pattern

For a bounded DataFrame already in memory, `df.ai.fillna(target,
max_examples=chosen_count)` is a convenience API when its automatic example
selection and default request settings are acceptable. After a controlled
evaluation, do not switch back to that convenience path: reuse the evaluated
`PreparedTask`, exemplar rows, and API settings with `.ai.task` or `.aio.task`
and assign by row position.

For large DuckDB inputs, do not load the full table into pandas. Use pandas
only for the chosen, bounded exemplar rows, then execute the evaluated prepared
task over distinct missing inputs in DuckDB.

`fillna()` returns a general response whose `output` may be an integer, float,
string, boolean, or `None`. DuckDB UDF return types are fixed, so replace that
general response format with a target-specific public `PreparedTask` before
registering the UDF:

```python
from typing import Literal

import duckdb
from pydantic import BaseModel, ConfigDict

from openaivec import PreparedTask
from openaivec.duckdb_ext import task_udf
from openaivec.task.table import fillna


class ResolutionFill(BaseModel):
    model_config = ConfigDict(extra="forbid")

    output: Literal["refund", "replace", "troubleshoot"] | None


chosen_count = 4
model_columns = [
    "customer_segment",
    "product_family",
    "issue_type",
    "resolution_code",
]

conn = duckdb.connect()
exemplars = conn.sql(
    """
    SELECT customer_segment, product_family, issue_type, resolution_code
    FROM approved_fill_examples
    ORDER BY exemplar_order
    LIMIT ?
    """,
    params=[chosen_count],
).df()

base_task = fillna(
    exemplars[model_columns],
    "resolution_code",
    max_examples=chosen_count,
)
typed_task = PreparedTask(
    instructions=base_task.instructions,
    response_format=ResolutionFill,
)

task_udf(
    conn,
    "ai_fill_resolution",
    task=typed_task,
    batch_size=None,
    max_concurrency=8,
    reasoning={"effort": "none"},
)
```

Before registration, verify that the effective example count and prompt hash
match the accepted evaluation run. Stop if the approved exemplar query now
returns different rows or the prompt budget changes the effective count.

The SQL JSON field order and names must match `model_columns`, with the target
set to `NULL`. Preserve the key outside the JSON:

```sql
CREATE TEMP TABLE fill_staged AS
SELECT
    ticket_id,
    resolution_code,
    CASE
        WHEN resolution_code IS NULL THEN to_json(struct_pack(
            customer_segment := customer_segment,
            product_family := product_family,
            issue_type := issue_type,
            resolution_code := CAST(NULL AS VARCHAR)
        ))
    END AS input_json
FROM source_tickets;

CREATE TEMP TABLE unique_fill_results AS
SELECT input_json, ai_fill_resolution(input_json) AS ai_result
FROM (
    SELECT DISTINCT input_json
    FROM fill_staged
    WHERE input_json IS NOT NULL
);

CREATE TEMP TABLE filled_tickets AS
SELECT
    s.ticket_id,
    CASE
        WHEN s.resolution_code IS NULL THEN r.ai_result.output
        ELSE s.resolution_code
    END AS resolution_code,
    s.resolution_code IS NULL AND r.ai_result.output IS NOT NULL AS was_imputed
FROM fill_staged AS s
LEFT JOIN unique_fill_results AS r USING (input_json);
```

Materialize `unique_fill_results` once before any preview or aggregation.
Validate source/output row counts, key uniqueness, allowed values, unresolved
`NULL` values, and `was_imputed` counts. Never overwrite the source target.
Only after an explicit write request, create a new user-facing file or table
under the rules in [safe data I/O](safe-data-io.md), retaining provenance so
inferred and observed values remain distinguishable.

## Research basis

The procedure deliberately avoids claiming a universal number of examples:

- [OpenAI prompt engineering](https://developers.openai.com/api/docs/guides/prompt-engineering)
  recommends pinning model snapshots and building evaluation suites because
  behavior varies by model and version.
- [OpenAI evaluation best practices](https://developers.openai.com/api/docs/guides/evaluation-best-practices)
  recommends defining the objective, dataset, metrics, comparison, and
  continuous evaluation, using production-like, typical, edge, and adversarial
  cases rather than subjective inspection.
- [What Makes Good In-Context Examples for GPT-3?](https://aclanthology.org/2022.deelio-1.10/)
  studies the effect of example selection.
- [Fantastically Ordered Prompts and Where to Find Them](https://aclanthology.org/2022.acl-long.556/)
  studies few-shot prompt order sensitivity.
- [Coverage-based Example Selection for In-Context Learning](https://aclanthology.org/2023.findings-emnlp.930/)
  motivates coverage-aware selection rather than relying only on an arbitrary
  count.

These sources support task-specific evaluation and careful example
selection/order; they do not establish one example count that is correct for
all intelligent-fill workloads.
