# Excel support setup

Read this reference before processing an `.xlsx` workbook. Excel support is
frequent, but installation is still a persistent local change and requires an
informed user choice.

## Decision flow

1. Check without network access or local changes:

   ```bash
   python scripts/manage_excel_extension.py check
   ```

2. If the output says `Excel support: installed`, do not reinstall or ask for
   installation consent. Continue with the separate data privacy, remote-call,
   and write-authorization workflow.
3. If it says `Excel support: not installed`, explain the impact and options
   below in plain language.
4. Ask one focused question and wait for the user's choice.
5. Install only if the user chooses installation.

Do not trigger installation merely because a path ends in `.xlsx`.

## Plain-language explanation

Adapt this explanation to the user's language:

> This environment does not yet have the optional feature needed to read Excel
> `.xlsx` tables directly. I can add the official Excel support component.
>
> What will change:
>
> - one signed component made for the installed data-processing software will
>   be downloaded from its official distribution site;
> - it will remain in this environment's local component storage, so future
>   sessions can reuse it;
> - the download uses network access and a small amount of local disk space;
>   the exact location and size depend on this environment; and
> - the component runs with the same local file permissions as this process,
>   which is why I will accept only the official signed version.
>
> What will not happen during installation:
>
> - Microsoft Excel itself will not be installed;
> - no workbook will be opened, changed, deleted, or uploaded;
> - no workbook contents will be sent to OpenAI or another model provider; and
> - no result file or database table will be created.
>
> Your organization may block software downloads or require an administrator.
> If you prefer not to install anything, you can provide the required sheet as
> CSV or Parquet instead.

Do not replace this explanation with only "a dependency is required" or
"install the Excel extension." The user must understand the practical effect.

## Ask exactly one question

Use the harness's structured question UI when available. Localize the wording,
but offer these three choices:

1. **Add the official Excel support component locally (Recommended)**
2. **I will provide CSV or Parquet instead**
3. **Stop for now**

Do not preselect consent, infer consent from the original Excel request, or
bundle this question with approval to send workbook data to an AI provider.

## Installation after approval

Only after the user chooses the first option, run:

```bash
python scripts/manage_excel_extension.py install --accept-local-install
```

The command:

- disables automatic extension installation;
- rejects community-signed and unsigned extensions for this operation;
- installs `excel` explicitly from DuckDB's official `core` repository;
- loads it under normal signature verification; and
- verifies that Excel support is both installed and loaded.

If support is already installed, the command loads and verifies it without
downloading another copy. It does not use `FORCE INSTALL`, replace an existing
installation, open a workbook, or remove anything.

Translate the successful result into user language:

> Excel table support is now ready. The workbook has not been opened or
> changed yet. Before reading its contents, I will separately confirm which
> sheet and columns may be processed and whether any data may be sent to the
> selected AI service.

Installation approval authorizes only this component installation. It does not
authorize:

- reading or sampling workbook values;
- sending workbook values to a model provider;
- creating a result file;
- overwriting the source workbook;
- updating the installed component later; or
- uninstalling or deleting the component.

Each later action follows its own privacy and write gate.

## If installation fails

Do not retry repeatedly, switch repositories, allow unsigned code, or install
`openpyxl`, another spreadsheet library, or a system package as a hidden
fallback.

Explain the likely category without overwhelming the user:

- **Network or company policy:** the official download site may be blocked.
- **Local permission:** this environment may not allow local component
  installation.
- **Platform/version availability:** a compatible official build may not be
  available for this environment.
- **Existing installation problem:** an installed copy may not pass loading or
  signature checks.

Offer these options:

1. Ask the environment administrator to allow the official component.
2. Export the required worksheet to CSV or Parquet and continue without Excel
   support.
3. Stop without changing the environment.

Include the exact technical error only after the plain-language summary, and
never expose unrelated local paths or secrets.

## Update, replacement, and removal

An update, forced reinstall, replacement, or removal is a separate persistent
change. Do not perform it under the original installation consent.

- Do not use `FORCE INSTALL`.
- Do not delete extension files directly.
- Do not change the configured extension directory.
- If a repair is required, explain the new change and alternatives and obtain
  separate explicit authorization.
- Prefer help from the environment administrator when the existing component
  is centrally managed.

## Workbook boundary after setup

Excel support covers tabular `.xlsx` values. It does not expand the skill to
legacy `.xls`, formulas, styles, charts, macros, pivot tables, external links,
or workbook automation. Follow [safe data I/O](safe-data-io.md) for read-only
access, sheet selection, new outputs, and overwrite protection.

## References

- [DuckDB Excel extension](https://duckdb.org/docs/current/core_extensions/excel.html)
- [Installing DuckDB extensions](https://duckdb.org/docs/current/extensions/installing_extensions.html)
- [Securing DuckDB extensions](https://duckdb.org/docs/current/operations_manual/securing_duckdb/securing_extensions.html)
