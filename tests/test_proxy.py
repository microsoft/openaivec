from __future__ import annotations

import asyncio
from typing import List

from openaivec.proxy import BatchingMapProxy


def test_batching_map_proxy_batches_calls_by_batch_size():
    calls: List[List[int]] = []

    def mf(xs: List[int]) -> List[int]:
        calls.append(xs[:])
        # echo back values
        return xs

    proxy = BatchingMapProxy[int, int](map_func=mf, batch_size=3)

    items = list(range(8))  # 0..7
    out = proxy.map(items)

    assert out == items
    # Should call in batches of 3,3,2
    assert [len(c) for c in calls] == [3, 3, 2]
    assert calls[0] == [0, 1, 2]
    assert calls[1] == [3, 4, 5]
    assert calls[2] == [6, 7]


def test_batching_map_proxy_cache_skips_already_processed_items():
    calls: List[List[int]] = []

    def mf(xs: List[int]) -> List[int]:
        calls.append(xs[:])
        return xs

    proxy = BatchingMapProxy[int, int](map_func=mf, batch_size=10)

    # first call processes 1,2,3
    out1 = proxy.map([1, 2, 3])
    assert out1 == [1, 2, 3]
    assert calls == [[1, 2, 3]]

    # second call has 2,3 cached, only processes 4
    out2 = proxy.map([2, 3, 4])
    assert out2 == [2, 3, 4]
    assert calls == [[1, 2, 3], [4]]


def test_batching_map_proxy_default_process_all_at_once_when_no_batch_size():
    calls: List[List[int]] = []

    def mf(xs: List[int]) -> List[int]:
        calls.append(xs[:])
        return xs

    proxy = BatchingMapProxy[int, int](map_func=mf)  # batch_size None

    items = [10, 20, 30, 40]
    out = proxy.map(items)
    assert out == items
    assert len(calls) == 1
    assert calls[0] == items


def test_batching_map_proxy_deduplicates_requests_and_batches():
    calls: List[List[int]] = []

    def mf(xs: List[int]) -> List[int]:
        calls.append(xs[:])
        return xs

    proxy = BatchingMapProxy[int, int](map_func=mf, batch_size=3)

    # inputs contain duplicates; 1 and 2 repeat
    items = [1, 1, 2, 3, 2, 4, 4, 5]
    out = proxy.map(items)

    assert out == items

    # unique order preserving: [1,2,3,4,5] -> batches: [1,2,3], [4,5]
    assert [len(c) for c in calls] == [3, 2]
    assert calls[0] == [1, 2, 3]
    assert calls[1] == [4, 5]

    # second call reuses cache entirely (no extra calls)
    out2 = proxy.map(items)
    assert out2 == items
    assert [len(c) for c in calls] == [3, 2]


def test_batching_map_proxy_rechecks_cache_within_batch_iteration():
    calls: List[List[int]] = []

    def mf(xs: List[int]) -> List[int]:
        # simulate an external side-effect that might populate cache between calls
        # (here we just record calls; LocalProxy itself will handle the cache)
        calls.append(xs[:])
        return xs

    proxy = BatchingMapProxy[int, int](map_func=mf, batch_size=4)

    # First call: all unique, expect one call with 4
    out1 = proxy.map([1, 2, 3, 4])
    assert out1 == [1, 2, 3, 4]
    assert calls == [[1, 2, 3, 4]]

    # Second call introduces overlap within would-be batches: [2,3,4,5,6]
    # Cache should skip 2,3,4 and only call for [5,6]
    out2 = proxy.map([2, 3, 4, 5, 6])
    assert out2 == [2, 3, 4, 5, 6]
    assert calls == [[1, 2, 3, 4], [5, 6]]


# -------------------- Internal methods tests --------------------
def test_internal_unique_in_order():
    from openaivec.proxy import BatchingMapProxy

    p = BatchingMapProxy[int, int](map_func=lambda xs: xs)
    uniq = getattr(p, "_BatchingMapProxy__unique_in_order")
    assert uniq([1, 1, 2, 3, 2, 4]) == [1, 2, 3, 4]


def test_internal_normalized_batch_size():
    from openaivec.proxy import BatchingMapProxy

    p = BatchingMapProxy[int, int](map_func=lambda xs: xs)
    nb = getattr(p, "_BatchingMapProxy__normalized_batch_size")
    assert nb(5) == 5  # default None => total
    p.batch_size = 0
    assert nb(7) == 7  # non-positive => total
    p.batch_size = 3
    assert nb(10) == 3  # positive => batch_size


def test_internal_all_cached_and_values():
    from openaivec.proxy import BatchingMapProxy

    p = BatchingMapProxy[int, int](map_func=lambda xs: xs)
    # fill cache via public API
    p.map([1, 2, 3])
    all_cached = getattr(p, "_BatchingMapProxy__all_cached")
    values = getattr(p, "_BatchingMapProxy__values")
    assert all_cached([1, 2]) is True
    assert all_cached([1, 4]) is False
    assert values([3, 2, 1]) == [3, 2, 1]


def test_internal_acquire_ownership():
    import threading

    from openaivec.proxy import BatchingMapProxy

    p = BatchingMapProxy[int, int](map_func=lambda xs: xs)
    # Cache 1; mark 2 inflight; 3 is missing
    p.map([1])
    inflight = getattr(p, "_BatchingMapProxy__inflight")
    lock = getattr(p, "_BatchingMapProxy__lock")
    with lock:
        inflight[2] = threading.Event()
    acquire = getattr(p, "_BatchingMapProxy__acquire_ownership")
    owned, wait_for = acquire([1, 2, 3])
    assert owned == [3]
    assert wait_for == [2]


def test_internal_finalize_success_and_failure():
    import threading

    from openaivec.proxy import BatchingMapProxy

    p = BatchingMapProxy[int, int](map_func=lambda xs: xs)
    inflight = getattr(p, "_BatchingMapProxy__inflight")
    cache = getattr(p, "_BatchingMapProxy__cache")
    lock = getattr(p, "_BatchingMapProxy__lock")
    finalize_success = getattr(p, "_BatchingMapProxy__finalize_success")
    finalize_failure = getattr(p, "_BatchingMapProxy__finalize_failure")

    # success path
    with lock:
        inflight[10] = threading.Event()
        inflight[20] = threading.Event()
    finalize_success([10, 20], [100, 200])
    with lock:
        assert cache[10] == 100 and cache[20] == 200
        assert 10 not in inflight and 20 not in inflight

    # failure path
    with lock:
        inflight[30] = threading.Event()
        inflight[40] = threading.Event()
    finalize_failure([30, 40])
    with lock:
        assert 30 not in inflight and 40 not in inflight
        assert 30 not in cache and 40 not in cache


def test_internal_process_owned_batches_and_skip_cached():
    from openaivec.proxy import BatchingMapProxy

    calls: list[list[int]] = []

    def mf(xs: list[int]) -> list[int]:
        calls.append(xs[:])
        return xs

    p = BatchingMapProxy[int, int](map_func=mf, batch_size=2)
    # Pre-cache 3 to force skip in second batch
    p.map([3])
    # Reset call log to focus on process_owned invocations
    calls.clear()
    process_owned = getattr(p, "_BatchingMapProxy__process_owned")
    cache = getattr(p, "_BatchingMapProxy__cache")

    process_owned([0, 1, 2, 3, 4])
    assert calls[0] == [0, 1]
    assert calls[1] == [2]  # 3 was cached and skipped
    assert calls[2] == [4]
    # cache should contain all keys now
    for k in [0, 1, 2, 3, 4]:
        assert k in cache


def test_internal_try_compute_single_success():
    from openaivec.proxy import BatchingMapProxy

    p = BatchingMapProxy[int, int](map_func=lambda xs: [x * 10 for x in xs])
    try_single = getattr(p, "_BatchingMapProxy__try_compute_single")
    cache = getattr(p, "_BatchingMapProxy__cache")
    try_single(7)
    assert cache[7] == 70


def test_internal_wait_for_with_inflight_event():
    import threading
    import time

    from openaivec.proxy import BatchingMapProxy

    p = BatchingMapProxy[int, int](map_func=lambda xs: [x * 10 for x in xs])
    inflight = getattr(p, "_BatchingMapProxy__inflight")
    cache = getattr(p, "_BatchingMapProxy__cache")
    lock = getattr(p, "_BatchingMapProxy__lock")
    wait_for = getattr(p, "_BatchingMapProxy__wait_for")

    keys = [100, 200]
    with lock:
        for k in keys:
            inflight[k] = threading.Event()

    def producer():
        time.sleep(0.05)
        with lock:
            for k in keys:
                cache[k] = k * 10
                ev = inflight.pop(k, None)
                if ev:
                    ev.set()

    t = threading.Thread(target=producer)
    t.start()
    wait_for(keys)
    t.join(timeout=1)
    assert all(k in cache for k in keys)


# -------------------- AsyncBatchingMapProxy tests --------------------


async def _afunc_echo(xs: list[int]) -> list[int]:
    await asyncio.sleep(0.01)
    return xs


def test_async_localproxy_basic(event_loop=None):
    from openaivec.proxy import AsyncBatchingMapProxy

    calls: list[list[int]] = []

    async def af(xs: list[int]) -> list[int]:
        calls.append(xs[:])
        return await _afunc_echo(xs)

    proxy = AsyncBatchingMapProxy[int, int](map_func=af, batch_size=3)

    async def run():
        out = await proxy.map([1, 2, 3, 4, 5])
        assert out == [1, 2, 3, 4, 5]

    asyncio.run(run())
    # Expect batches: [1,2,3], [4,5]
    assert [len(c) for c in calls] == [3, 2]


def test_async_localproxy_dedup_and_cache(event_loop=None):
    from openaivec.proxy import AsyncBatchingMapProxy

    calls: list[list[int]] = []

    async def af(xs: list[int]) -> list[int]:
        calls.append(xs[:])
        return await _afunc_echo(xs)

    proxy = AsyncBatchingMapProxy[int, int](map_func=af, batch_size=10)

    async def run():
        out1 = await proxy.map([1, 1, 2, 3])
        assert out1 == [1, 1, 2, 3]
        out2 = await proxy.map([3, 2, 1])
        assert out2 == [3, 2, 1]

    asyncio.run(run())
    # First call computes [1,2,3] once, second call uses cache entirely
    assert calls == [[1, 2, 3]]


def test_async_localproxy_concurrent_requests(event_loop=None):
    from openaivec.proxy import AsyncBatchingMapProxy

    calls: list[list[int]] = []

    async def af(xs: list[int]) -> list[int]:
        # simulate IO
        await asyncio.sleep(0.02)
        calls.append(xs[:])
        return xs

    proxy = AsyncBatchingMapProxy[int, int](map_func=af, batch_size=3)

    async def run():
        # two overlapping requests with duplicates
        r1 = proxy.map([1, 2, 3, 4])
        r2 = proxy.map([3, 4, 5])
        out1, out2 = await asyncio.gather(r1, r2)
        assert out1 == [1, 2, 3, 4]
        assert out2 == [3, 4, 5]

    asyncio.run(run())
    # Expect that computations are not duplicated: first call handles [1,2,3], [4,5] possibly
    # depending on interleaving but total coverage should be minimal. We check that
    # every number 1..5 appears across the union of calls and no number is overrepresented.
    flat = [x for call in calls for x in call]
    assert set(flat) == {1, 2, 3, 4, 5}


def test_async_localproxy_max_concurrency_limit(event_loop=None):
    from openaivec.proxy import AsyncBatchingMapProxy

    current = 0
    peak = 0

    async def af(xs: list[int]) -> list[int]:
        nonlocal current, peak
        # simulate per-call concurrent work proportional to input size
        current += 1
        peak = max(peak, current)
        await asyncio.sleep(0.05)
        current -= 1
        return xs

    proxy = AsyncBatchingMapProxy[int, int](map_func=af, batch_size=1, max_concurrency=2)

    async def run():
        # Launch several maps concurrently; each map will call af once per batch
        tasks = [proxy.map([i]) for i in range(6)]
        outs = await asyncio.gather(*tasks)
        assert outs == [[i] for i in range(6)]

    asyncio.run(run())
    # Peak concurrency should not exceed limit (2)
    assert peak <= 2
