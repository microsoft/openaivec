# openaivec-skill 1.3.1 - OSS business bulk-processing assistant

This patch clarifies that openaivec-skill is **MIT-licensed open source**.
Installation guidance names its upstream repository without implying vendor
certification, endorsement, product status, or commercial support. The bulk
execution behavior introduced in 1.3.0 is unchanged.

## Process large tables and document folders through guided choices

Business users describe the outcome, select the source, review a small
preview, and approve the full run. The assistant handles code and processing
settings instead of asking users to write SQL or tune concurrency.

- **No explicit invocation required:** bulk text/document requests and suitable
  opportunities discovered during authorized inspection activate the guidance.
- **Text classification at scale:** organize surveys, reviews, support tickets,
  sales notes, and product descriptions by topic, sentiment, or other agreed
  categories. Extraction, summaries, translation, and semantic matching are
  also supported.
- **Efficient bulk execution:** reuse duplicate inputs and accepted preview
  results, retain every source row and identifier, and report checkpoints
  during long runs.
- **Business-language interaction:** meaningful choices and short previews,
  with implementation details kept in the background.
- **Explicit safety gates:** approve data transmission, billable previews and
  full runs, and exact new output destinations. No silent source changes,
  software installation, or persistent checkpoints.

Actual throughput and charges depend on the input and selected service.
This release does not promise a fixed speedup or route numeric-only
analytics, ordinary conversion, or workbook formatting to AI unnecessarily.

## Business-user installation

Open the intended project in an Agent Skills-compatible assistant such as
GitHub Copilot, Claude Code, Codex, or Cursor. Paste the following into chat.
**You do not need to run terminal commands.**

```text
Install the MIT-licensed openaivec-skill from its upstream repository,
microsoft/openaivec, for this workspace only. Select the latest stable
Skill release tagged openaivec-skill-vX.Y.Z, not a Python-library release
or untagged branch, and pin installation to that exact tag.
Explain the files and scope, stop if an existing Skill would be replaced,
and ask before installing. After approval, use this assistant's supported
Skill installer, perform installation yourself, and verify the source,
version, and workspace-only scope. Tell me if I need a new chat.
Do not open business data, install processing software, configure secrets,
or make an AI request yet. Do not ask me to run commands; prepare an
administrator handoff if direct installation is unavailable.
```

Review the explanation and approve installation. Start a new chat if the
assistant asks you to. If an approved AI connection or processing software is
missing, the assistant explains the impact and asks before preparing it.
Never paste API keys or passwords into chat.

After installation, describe the task normally:

```text
Classify all the comments in this workbook by topic and sentiment.
Preserve the original response IDs and every row; leave the workbook
unchanged. Recommend useful result columns and show a small preview.
Guide me through the choices, and ask before sending data, running the
full set, or saving a new result.
```

You do not need to name the skill each time. Discovering a suitable large
text table or document folder does not authorize paid processing by itself.

See the [business-user getting-started guide](https://microsoft.github.io/openaivec/agent-skill/)
for installation variants, readiness checks, and practical examples.

## Assistant and maintainer details

The release adds guided-conversation and measured-performance references,
an in-memory bulk runner with separate pilot/full-run gates, deterministic
checkpoint mapping, accepted-pilot reuse, ordered row restoration, and
offline regression coverage. Committed chunks can be reused in the same
process; cross-process checkpoints require separate authorization.
Remote retries can still incur charges; the helper does not claim
exactly-once delivery to the provider.

The portable archives include the English guide, references, helper scripts,
evaluation cases, release notes, and MIT license.
This is an **Agent Skill release**, not a new Python-library release.
