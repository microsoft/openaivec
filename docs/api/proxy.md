# proxy

Batching proxies for order-preserving, cached batch mapping.

This module provides two helpers:

- BatchingMapProxy: thread-safe synchronous batching with caching and de-duplication.
- AsyncBatchingMapProxy: asyncio-friendly batching with optional concurrency limits.

Both proxies accept the mapping function as the second argument to map(). The function must:

- Accept a list of inputs and return a list of outputs in the same order.
- Be pure relative to a single call (side effects should be idempotent or safe).

## Synchronous usage (BatchingMapProxy)

```python
from typing import List
from openaivec.proxy import BatchingMapProxy

# Define your batch mapping function
def fetch_many(keys: List[int]) -> List[str]:
    # Example: echo values as strings
    return [f"val:{k}" for k in keys]

# Create proxy with an optional batch size hint
proxy = BatchingMapProxy[int, str](batch_size=3)

# Map items using the proxy. Duplicates are de-duplicated and order preserved.
items = [1, 2, 2, 3, 4, 4, 5]
outputs = proxy.map(items, fetch_many)
assert outputs == ["val:1", "val:2", "val:2", "val:3", "val:4", "val:4", "val:5"]

# Cache is reused across calls
outputs2 = proxy.map([5, 4, 3, 2, 1], fetch_many)
assert outputs2 == ["val:5", "val:4", "val:3", "val:2", "val:1"]
```

### Notes

- If `batch_size` is None or <= 0, all unique items are processed in a single call.
- Under concurrency, the proxy prevents duplicate work by coordinating in-flight keys.

## Asynchronous usage (AsyncBatchingMapProxy)

```python
import asyncio
from typing import List
from openaivec.proxy import AsyncBatchingMapProxy

# Define your async batch mapping function
async def fetch_many_async(keys: List[int]) -> List[str]:
    # Simulate I/O
    await asyncio.sleep(0.01)
    return [f"val:{k}" for k in keys]

# Create proxy with batch size and an optional concurrency cap for map_func calls
proxy = AsyncBatchingMapProxy[int, str](batch_size=3, max_concurrency=2)

async def main():
    items = [1, 2, 3, 4, 5]
    out = await proxy.map(items, fetch_many_async)
    assert out == ["val:1", "val:2", "val:3", "val:4", "val:5"]

    # Overlapping requests deduplicate work and share results via the cache
    r1 = proxy.map([1, 2, 3, 4], fetch_many_async)
    r2 = proxy.map([3, 4, 5], fetch_many_async)
    o1, o2 = await asyncio.gather(r1, r2)
    assert o1 == ["val:1", "val:2", "val:3", "val:4"]
    assert o2 == ["val:3", "val:4", "val:5"]

asyncio.run(main())
```

### Notes

- `max_concurrency` limits concurrent invocations of `map_func` across overlapping `map()` calls.
- The proxy rechecks the cache immediately before each batch call to avoid redundant work.

## API summary

```python
class BatchingMapProxy[S: Hashable, T]:
    batch_size: int | None

    def map(self, items: list[S], map_func: Callable[[list[S]], list[T]]) -> list[T]:
        ...

class AsyncBatchingMapProxy[S: Hashable, T]:
    batch_size: int | None
    max_concurrency: int

    async def map(self, items: list[S], map_func: Callable[[list[S]], Awaitable[list[T]]]) -> list[T]:
        ...
```

Implementation details:

- Inputs are de-duplicated with first-occurrence order preserved.
- Cache is filled atomically and shared across calls.
- In-flight keys are coordinated (threading.Event / asyncio.Event) to prevent duplicated computation.
- Errors from `map_func` propagate; in-flight keys are released to avoid deadlocks.
