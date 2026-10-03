# Guided business experience

This is an instruction to the assistant, not a checklist for the business
user. The user should describe work once, make a few meaningful choices, and
review a result. They should not need to learn how the processing works.

## Detect the opportunity without an explicit invocation

Use the skill when the request or already-authorized source inspection shows
both repeated scope and text/document interpretation:

| What the assistant encounters | Business suggestion |
| --- | --- |
| Many survey, review, or inquiry rows with free text | Topic and sentiment classification |
| A table with long descriptions or notes | Extract fields or next actions into consistent columns |
| A folder of supported invoices, quotes, or reports | Extract visible facts into a review table |
| Many product descriptions in several languages | Consistent translation or category normalization |
| A document library and many questions | Match each question to relevant documents |
| Known and missing categories in otherwise complete rows | Evaluate contextual fill before filling blanks |

Recognize colloquial requests such as "do all of these at once", "make this
usable", "sort out the comments", "classify every description", or "process
this whole document folder", including equivalent requests in the user's
language. Do not require a library, AI, SQL, or skill name.

When a user merely asks to inspect a large table, respect that read-only
request. If the authorized inspection discovers suitable text, briefly offer
the relevant outcome with a single choice; do not start billable processing.
Do not activate for a numeric-only table, file copying, format conversion,
literal replacement, or workbook styling just because it is large.

## Ask less, do more

If the outcome is known, skip the outcome question. If the source is named,
skip the source question. Reuse an approved service and output convention
only within their authorized scope. Recommend a small set of columns instead
of presenting an empty technical schema form.

Use the harness's native question tool when available. Ask one question at a
time. A free-text facility supplied by the harness is enough; do not add a
duplicate catch-all option. Without a native choice tool, present short
numbered choices honestly, not fake buttons.

The four user-visible stages are:

1. **Outcome:** "What would help most?" Offer two to five relevant results,
   not libraries or algorithms.
2. **Source:** "Which source should I use?" Ask only for a missing authorized
   workbook/sheet, table/column, or folder. Inspect metadata locally.
3. **Preview:** Show a bounded result table after scoped consent. Ask whether
   its fields and meaning meet the need. Correct poor definitions before
   continuing.
4. **Run:** Show unique scope, receiving service, human-review rule, and
   authorized destination. Ask whether to run that exact plan.

Readiness, update, privacy, and write consent can add necessary questions.
Do not claim every run requires exactly four clicks. Avoid asking the user to
solve a technical problem just to preserve a four-step marketing claim.

## Example: discovered free-text table

Suppose authorized inspection finds 80,000 survey rows, 76,000 non-empty
comments, and 31,200 distinct comments. These are illustrative counts.

Assistant:

> This table has a large comment column. I can organize it in one run without
> changing the source. What would help most?

Choices:

- Classify topics and sentiment (recommended)
- Extract requests and next actions
- Produce one short summary per comment

After the user chooses classification, propose sentiment, topic, brief
summary, evidence where needed, and a review flag. Do not ask for Pydantic
types, model names, SQL, or concurrency.

Before the pilot:

> I will test eight distinct comments using the approved company AI service.
> Only the selected comment text is sent. This is billable usage; the full
> table will not run yet. May I run this preview?

After the pilot, show at most eight result rows, using approved/redacted source
samples and business labels. Preserve identifiers in the result even if
technical staging IDs are hidden from the conversation.

Choices:

- These categories work; continue to the full-run plan
- Change the categories before continuing
- Stop without running the full table

Then show the full-run plan:

> 80,000 source rows will be retained. There are 31,200 distinct comments;
> the eight accepted preview results will be reused. Empty comments stay
> empty, and uncertain results are flagged for review. The approved service
> receives only the selected text. I will create only the new result you
> authorized and will stop if that destination exists.

Do not imply that token counts are a price quote, that repeated source rows
are removed, or that a model's confidence value is a calibrated probability.

## Example: a document folder

Ask which visible facts are needed only if the outcome is unspecified.
Recommend a short review table appropriate to the document type. Invoice
facts might be supplier, invoice number, date, currency, total, and review
reason. Procurement or accounting staff make decisions and approve payment.

Explain unavailable or oversized files in ordinary terms. Do not hide skipped
files or silently substitute a different document parser. Media files are
processed with a bounded parallel path; do not promise the same throughput
as a short text table or deduplicate different paths as identical content.

## What stays hidden

The assistant handles:

- environment checks and approved project-local setup;
- SQL, Python, temporary tables, generated run IDs, and output schemas;
- public API choice, model options, batching, and concurrency;
- accepted-pilot reuse, duplicate mapping, and retry ownership;
- token estimation, checkpoint timing, and local grouped calculations.

No user needs to copy code, run a shell command, choose a batch size, or read
a stack trace. Technical references are for the assistant and administrators.
Disclose implementation accurately if asked, but do not turn a routine
business interaction into an engineering tutorial.

Never hide the data leaving the environment, receiving service, estimated
workload and cost uncertainty, setup changes, source/destination, unresolved
records, or the need for human review.

## Progress and recovery language

Send a short phase message before reading, processing, validating, and saving.
For a long run, show committed progress at measured checkpoints:

> Organizing comments: 12,000 of 31,200 distinct texts complete (38%).
> 90 need review. Processing continues; the source is unchanged.

Report actual elapsed processing time. An ETA is an estimate only after two
representative completed checkpoints. While a slow chunk is in flight, report
the last committed count and current phase, not invented incremental success.

On failure:

> Processing paused because the selected service is limiting requests.
> Completed results remain available in this session. No final output was
> written. I can resume the unfinished portion at a lower processing rate.

Offer one relevant recovery choice. Never silently change service, output,
or interpretation rules. A failed chunk may already have incurred charges;
say so before authorizing its retry. Authentication recovery, software
installation, or on-disk checkpoints still have their separate consent gates.

## Completion

Default to a short business handoff:

> The new review table is ready. All source rows and identifiers are retained,
> repeated text was processed once and restored to every matching row, and
> uncertain records are flagged. The original files are unchanged.

Include actual input/output counts, unique work, repeated evaluations avoided,
and unresolved/failed counts. Link the exact authorized new output only after
it has been re-read and verified. Put a compact review list beside the result
when requested; do not create additional files automatically.
