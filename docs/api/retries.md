# Transport Retries

Transport retries have one owner. By default, `retry_policy=None` delegates
them entirely to the supplied OpenAI SDK client. Its `max_retries`, timeout,
authentication, transport, and ownership remain unchanged. There is no outer
12-attempt retry loop. For example, an SDK client with `max_retries=2` makes at
most three HTTP attempts for one request; `max_retries=0` makes one.

Use `RetryPolicy` to replace the SDK's transport retry settings for an operation:

```python
from openai import OpenAI
from openaivec import BatchResponses, RetryPolicy

with OpenAI() as client:
    responses = BatchResponses.of(
        client,
        "gpt-4.1-mini",
        "Summarize each input",
        retry_policy=RetryPolicy(
            max_attempts=3,
            initial_delay=0.5,
            max_delay=8.0,
            max_elapsed=60.0,
        ),
    )
    result = responses.parse(["The order arrived on time."])
```

The policy is accepted by synchronous and asynchronous Responses, Embeddings,
schema inference, pandas accessors, and Spark/DuckDB UDF factories. It is an
openaivec control, never an API request parameter. Per-call client settings
disable SDK retries without mutating or closing the caller's client or shared
transport. The caller remains responsible for closing that client.

## Attempts and Delays

`max_attempts=3` counts the initial HTTP attempt. Use `max_attempts=1` for
fail-fast transport behavior. Connection errors, timeouts, HTTP 408, 409, 429,
and 5xx errors are retryable; a server `x-should-retry: false` header disables
status retries. Other API errors propagate immediately.

Explicit policies use full-jitter exponential delays, starting with a ceiling
of `min(initial_delay, max_delay)` and doubling up to `max_delay`. Both delay
limits are finite and nonnegative. Zero disables the delay. Unlike SDK-owned
retries, explicit policies use these local delay limits, not `Retry-After`.
Exhaustion logs only the exception type and attempt count, not prompts or keys.

## Deadlines

`max_elapsed=None` leaves the elapsed-time budget unset. A positive finite
value measures monotonic elapsed seconds across requests and retry waits:

- Responses: one batched request and all its validation corrections.
- Schema inference: one inference call and all its validation corrections.
- Embeddings: one cache batch, including token planning and provider-limit
  subrequests. Splitting does not restart the budget.
- Multimodal Responses: each individual Responses API request. File loading
  and uploads are outside this budget.

The deadline is not a limit for an entire `parse()` or `create()` operation,
its cache queue time, or all Spark partitions. Schema-less parsing gives
inference and each extraction batch separate budgets. Use an application-level
deadline when the entire workflow must be bounded.

Async requests are cancelled and awaited when the deadline expires. Sync
requests cap each HTTP timeout phase by the remaining budget and reject late
results, but cannot forcibly interrupt a blocking transport. A smaller explicit
request timeout is preserved. Deadline exhaustion raises `TimeoutError`; an
async cancellation is propagated without retrying, and cache workers are drained.

## Validation Is Separate

`max_validation_retries=3` controls additional schema/ID corrections for batched
Responses. Schema inference's `max_retries=8` counts total inference attempts.
Neither is a transport retry count. With three HTTP attempts and three
additional response corrections, at most twelve HTTP attempts can occur for
that batch, unless its elapsed-time deadline is reached first.

Schema-less pandas and Spark parsing forward the same policy to both inference
and extraction. Set `max_validation_retries=0` and `max_retries=1` when no
validation correction is desired. Use `retry_policy=RetryPolicy(max_attempts=1)`
or configure the SDK with `max_retries=0` to disable transport retries too.

::: openaivec.RetryPolicy