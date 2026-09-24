# openaivec skill

Portable [Agent Skills](https://agentskills.io/) guidance for applying
openaivec to large file sets and table columns. Users can work in terms of
Excel, CSV, Parquet, JSON, or supported relational sources; the skill handles
the vectorized execution layer internally.

The skill covers:

- batched text and structured Responses;
- extraction, prepared tasks, embeddings, and similarity search;
- Excel, CSV/TSV, Parquet, JSON, and supported relational input/output;
- informed, user-approved setup of the official Excel support component when
  it is not already available;
- read-only defaults and explicit gates for create, append, update, overwrite,
  and delete operations;
- local files plus openaivec-supported multimodal inputs;
- intelligent missing-value fill with measured few-shot example selection;
- deterministic DuckDB shaping and cross-tabs around materialized AI results,
  with a business-semantics clarification gate;
- adaptive text batching, global duplicate reuse, stable result restoration,
  and progress checkpoints for long-running work;
- business scenarios for customer support, product catalogs, documents,
  localization, incident reports, CRM notes, and semantic search;
- conversational recovery for OpenAI, Azure OpenAI, Entra ID, and Fabric
  authentication failures.

It intentionally does not route for standalone generic ETL, unsupported
database connectors, cloud resource provisioning, workbook formatting,
unsupported file parsing, or AI SDKs other than openaivec. Once an
interpretive workload routes here, basic deterministic shaping and aggregation
may be used internally around the materialized AI result.

## Business user experience

The assistant should keep implementation details in the background and:

1. restate the business outcome, source, fields, and requested destination;
2. ask one plain-language question at a time when a column meaning is unclear;
3. show which data may leave the environment, the unique workload, and the
   limits of any cost estimate;
4. run a small structured preview and agree on acceptance checks;
5. process duplicate inputs once while restoring every source row;
6. report regular checkpoints during long work; and
7. return validated counts, unresolved items, and a human-review-ready result
   without silently changing the source.

## Install from GitHub

Preview the package before installing it:

```bash
gh skill preview microsoft/openaivec openaivec-skill
```

Install it for a supported harness:

```bash
gh skill install microsoft/openaivec openaivec-skill \
  --agent github-copilot \
  --scope project
```

Replace `github-copilot` with another supported host such as `claude-code`,
`codex`, `cursor`, or `gemini-cli`. For an interactive multi-harness installer:

```bash
npx skills add microsoft/openaivec --skill openaivec-skill
```

To validate and install a local checkout:

```bash
gh skill publish --dry-run
gh skill install . openaivec-skill --from-local --agent github-copilot
```

## Use

The harness should load the skill automatically for requests such as:

> Classify the `review_text` column in all Parquet files under `data/reviews/`
> with OpenAI, preserve each source row, and write one result Parquet file.

It should also activate when the user names only the business operation and
source, for example:

> Read the comments from this Excel workbook, identify sentiment and themes,
> and write a new CSV without changing the workbook.

In clients that support explicit skill invocation, request
`/openaivec-skill`.

See [business scenarios](references/business-scenarios.md) for the scenario
catalog and runnable examples.

See [safe data I/O](references/safe-data-io.md) for supported-source
boundaries and mandatory authorization gates for persistent writes.

See [Excel support setup](references/excel-setup.md) for the plain-language
impact explanation, user choices, and consent-gated installation command.

See [intelligent fill](references/intelligent-fill.md) for the required
holdout evaluation used to choose the few-shot example count before imputing a
missing column.

See [data shaping and cross-tabs](references/data-shaping-and-crosstabs.md) for
local cleaning, joins, grouped summaries, pivot tables, denominator checks,
and one-question-at-a-time clarification of ambiguous business meanings.

## Maintainer release

Validate from the repository root:

```bash
gh skill publish --dry-run
```

After updating `metadata.version` in `SKILL.md` and merging to `main`, create a
matching tag on that exact commit:

```bash
git switch main
git pull --ff-only
git tag openaivec-skill-v1.0.0
git push origin openaivec-skill-v1.0.0
```

The `Publish Agent Skill` workflow validates the Agent Skills specification,
checks that the tag and frontmatter versions match, builds `.tar.gz` and `.zip`
packages with SHA-256 checksums, verifies a local harness installation, and
creates the GitHub Release. Rerunning a completed release is safe when all
expected assets already exist; an incomplete existing release fails without
overwriting assets.

The skill-specific tag deliberately does not match this repository's
`v*.*.*` PyPI release trigger.

## License

The portable skill package includes its own [MIT license](LICENSE).
