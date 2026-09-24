# Pandas Extension

## Filling missing values

`df.ai.fillna("column")` and `await df.aio.fillna("column")` fill only missing
cells and preserve index labels, including duplicate labels. They construct a
local few-shot task by default: **no prompt-improvement API call** occurs before
row processing. The default is eight deterministically sampled examples, with
at most 6,000 characters and an estimated 1,500 tokens of example content.
Large individual rows that do not fit either budget are skipped; if none fit,
the task uses zero-shot instructions.
Input examples are complete JSON rows with the target value set to null, just
like the runtime input. Outputs contain only an `output` field.

For optional LLM prompt refinement, prepare the task explicitly with
`openaivec.task.table.fillna(df, "column", improve_prompt=True)` before
executing it with `df[df["column"].isna()].ai.task(task)` or awaiting
`df[df["column"].isna()].aio.task(task)`. Refinement is synchronous, opt-in,
and uses the same bounded examples; it is not run inside async `fillna`.
When applying the task results yourself, assign by **row position**, not index
label, to avoid overwriting nonmissing rows sharing a label.

For explicitly managed `responses_with_cache`, `task_with_cache`, or
`parse_with_cache` calls, include `None` in the cache's value type: a response
without a parsed result is cached as `None` by both sync and async accessors.

## Configuration

::: openaivec.pandas_ext._config

## Series Accessor (`.ai`)

::: openaivec.pandas_ext._series_sync.OpenAIVecSeriesAccessor

## DataFrame Accessor (`.ai`)

::: openaivec.pandas_ext._dataframe_sync.OpenAIVecDataFrameAccessor

## Async Series Accessor (`.aio`)

::: openaivec.pandas_ext._series_async.AsyncOpenAIVecSeriesAccessor

## Async DataFrame Accessor (`.aio`)

::: openaivec.pandas_ext._dataframe_async.AsyncOpenAIVecDataFrameAccessor