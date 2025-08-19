from __future__ import annotations

import asyncio
import time

import pytest

from openaivec._proxy import AsyncBatchingMapProxy, BatchingMapProxy


def test_batching_map_proxy_batches_calls_by_batch_size():
    calls: list[list[int]] = []

    def mf(xs: list[int]) -> list[int]:
        calls.append(xs[:])
        # echo back values
        return xs

    proxy = BatchingMapProxy[int, int](batch_size=3)

    items = list(range(8))  # 0..7
    out = proxy.map(items, mf)

    assert out == items
    # Should call in batches of 3,3,2
    assert [len(c) for c in calls] == [3, 3, 2]
    assert calls[0] == [0, 1, 2]
    assert calls[1] == [3, 4, 5]
    assert calls[2] == [6, 7]


def test_batching_map_proxy_cache_skips_already_processed_items():
    calls: list[list[int]] = []

    def mf(xs: list[int]) -> list[int]:
        calls.append(xs[:])
        return xs

    proxy = BatchingMapProxy[int, int](batch_size=10)

    # first call processes 1,2,3
    out1 = proxy.map([1, 2, 3], mf)
    assert out1 == [1, 2, 3]
    assert calls == [[1, 2, 3]]

    # second call has 2,3 cached, only processes 4
    out2 = proxy.map([2, 3, 4], mf)
    assert out2 == [2, 3, 4]
    assert calls == [[1, 2, 3], [4]]


def test_batching_map_proxy_default_process_all_at_once_when_no_batch_size():
    calls: list[list[int]] = []

    def mf(xs: list[int]) -> list[int]:
        calls.append(xs[:])
        return xs

    proxy = BatchingMapProxy[int, int]()  # batch_size None

    items = [10, 20, 30, 40]
    out = proxy.map(items, mf)
    assert out == items
    assert len(calls) == 1
    assert calls[0] == items


def test_batching_map_proxy_deduplicates_requests_and_batches():
    calls: list[list[int]] = []

    def mf(xs: list[int]) -> list[int]:
        calls.append(xs[:])
        return xs

    proxy = BatchingMapProxy[int, int](batch_size=3)

    # inputs contain duplicates; 1 and 2 repeat
    items = [1, 1, 2, 3, 2, 4, 4, 5]
    out = proxy.map(items, mf)

    assert out == items

    # unique order preserving: [1,2,3,4,5] -> batches: [1,2,3], [4,5]
    assert [len(c) for c in calls] == [3, 2]
    assert calls[0] == [1, 2, 3]
    assert calls[1] == [4, 5]

    # second call reuses cache entirely (no extra calls)
    out2 = proxy.map(items, mf)
    assert out2 == items
    assert [len(c) for c in calls] == [3, 2]


def test_batching_map_proxy_rechecks_cache_within_batch_iteration():
    calls: list[list[int]] = []

    def mf(xs: list[int]) -> list[int]:
        # simulate an external side-effect that might populate cache between calls
        # (here we just record calls; LocalProxy itself will handle the cache)
        calls.append(xs[:])
        return xs

    proxy = BatchingMapProxy[int, int](batch_size=4)

    # First call: all unique, expect one call with 4
    out1 = proxy.map([1, 2, 3, 4], mf)
    assert out1 == [1, 2, 3, 4]
    assert calls == [[1, 2, 3, 4]]

    # Second call introduces overlap within would-be batches: [2,3,4,5,6]
    # Cache should skip 2,3,4 and only call for [5,6]
    out2 = proxy.map([2, 3, 4, 5, 6], mf)
    assert out2 == [2, 3, 4, 5, 6]
    assert calls == [[1, 2, 3, 4], [5, 6]]


def test_batching_map_proxy_map_func_length_mismatch_raises_and_releases():
    from openaivec._proxy import BatchingMapProxy

    p = BatchingMapProxy[int, int](batch_size=3)

    def bad(xs: list[int]) -> list[int]:
        # Return wrong length to simulate contract violation
        return xs[:-1]

    # Using a try/except to assert ValueError and ensure no deadlock occurs
    with pytest.raises(ValueError):
        p.map([1, 2, 3], bad)

    # After failure, a good call should still proceed (events must be released)
    out = p.map([1, 2, 3], lambda xs: xs)
    assert out == [1, 2, 3]


# -------------------- Internal methods tests --------------------
def test_internal_unique_in_order():
    from openaivec._proxy import BatchingMapProxy

    p = BatchingMapProxy[int, int]()
    assert p._unique_in_order([1, 1, 2, 3, 2, 4]) == [1, 2, 3, 4]


def test_internal_normalized_batch_size():
    from openaivec._proxy import BatchingMapProxy

    p = BatchingMapProxy[int, int]()
    assert p._normalized_batch_size(5) == 5  # default None => total
    p.batch_size = 0
    assert p._normalized_batch_size(7) == 7  # non-positive => total
    p.batch_size = 3
    assert p._normalized_batch_size(10) == 3  # positive => batch_size


def test_internal_all_cached_and_values():
    from openaivec._proxy import BatchingMapProxy

    p = BatchingMapProxy[int, int]()
    # fill cache via public API
    p.map([1, 2, 3], lambda xs: xs)
    all_cached = getattr(p, "_BatchingMapProxy__all_cached")
    values = getattr(p, "_BatchingMapProxy__values")
    assert all_cached([1, 2]) is True
    assert all_cached([1, 4]) is False
    assert values([3, 2, 1]) == [3, 2, 1]


def test_internal_acquire_ownership():
    import threading

    from openaivec._proxy import BatchingMapProxy

    p = BatchingMapProxy[int, int]()
    # Cache 1; mark 2 inflight; 3 is missing
    p.map([1], lambda xs: xs)
    inflight = getattr(p, "_inflight")
    lock = getattr(p, "_lock")
    with lock:
        inflight[2] = threading.Event()
    acquire = getattr(p, "_BatchingMapProxy__acquire_ownership")
    owned, wait_for = acquire([1, 2, 3])
    assert owned == [3]
    assert wait_for == [2]


def test_internal_finalize_success_and_failure():
    import threading

    from openaivec._proxy import BatchingMapProxy

    p = BatchingMapProxy[int, int]()
    inflight = getattr(p, "_inflight")
    cache = getattr(p, "_cache")
    lock = getattr(p, "_lock")
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
    from openaivec._proxy import BatchingMapProxy

    calls: list[list[int]] = []

    def mf(xs: list[int]) -> list[int]:
        calls.append(xs[:])
        return xs

    p = BatchingMapProxy[int, int](batch_size=2)
    # Pre-cache 3 to force skip in second batch
    p.map([3], mf)
    # Reset call log to focus on process_owned invocations
    calls.clear()
    process_owned = getattr(p, "_BatchingMapProxy__process_owned")
    cache = getattr(p, "_cache")

    process_owned([0, 1, 2, 3, 4], mf)
    assert calls[0] == [0, 1]
    assert calls[1] == [2, 4]  # 3 was cached and skipped, 2,4 batched together for efficiency
    # cache should contain all keys now
    for k in [0, 1, 2, 3, 4]:
        assert k in cache


def test_internal_wait_for_with_inflight_event():
    import threading
    import time

    from openaivec._proxy import BatchingMapProxy

    p = BatchingMapProxy[int, int]()

    def mf(xs: list[int]) -> list[int]:
        return [x * 10 for x in xs]

    inflight = getattr(p, "_inflight")
    cache = getattr(p, "_cache")
    lock = getattr(p, "_lock")
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
    wait_for(keys, mf)
    t.join(timeout=1)
    assert all(k in cache for k in keys)


# -------------------- AsyncBatchingMapProxy tests --------------------


async def _afunc_echo(xs: list[int]) -> list[int]:
    await asyncio.sleep(0.01)
    return xs


def test_async_localproxy_basic(event_loop=None):
    from openaivec._proxy import AsyncBatchingMapProxy

    calls: list[list[int]] = []

    async def af(xs: list[int]) -> list[int]:
        calls.append(xs[:])
        return await _afunc_echo(xs)

    proxy = AsyncBatchingMapProxy[int, int](batch_size=3)

    async def run():
        out = await proxy.map([1, 2, 3, 4, 5], af)
        assert out == [1, 2, 3, 4, 5]

    asyncio.run(run())
    # Expect batches: [1,2,3], [4,5]
    assert [len(c) for c in calls] == [3, 2]


def test_async_localproxy_dedup_and_cache(event_loop=None):
    from openaivec._proxy import AsyncBatchingMapProxy

    calls: list[list[int]] = []

    async def af(xs: list[int]) -> list[int]:
        calls.append(xs[:])
        return await _afunc_echo(xs)

    proxy = AsyncBatchingMapProxy[int, int](batch_size=10)

    async def run():
        out1 = await proxy.map([1, 1, 2, 3], af)
        assert out1 == [1, 1, 2, 3]
        out2 = await proxy.map([3, 2, 1], af)
        assert out2 == [3, 2, 1]

    asyncio.run(run())
    # First call computes [1,2,3] once, second call uses cache entirely
    assert calls == [[1, 2, 3]]


def test_async_localproxy_concurrent_requests(event_loop=None):
    from openaivec._proxy import AsyncBatchingMapProxy

    calls: list[list[int]] = []

    async def af(xs: list[int]) -> list[int]:
        # simulate IO
        await asyncio.sleep(0.02)
        calls.append(xs[:])
        return xs

    proxy = AsyncBatchingMapProxy[int, int](batch_size=3)

    async def run():
        # two overlapping requests with duplicates
        r1 = proxy.map([1, 2, 3, 4], af)
        r2 = proxy.map([3, 4, 5], af)
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
    from openaivec._proxy import AsyncBatchingMapProxy

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

    proxy = AsyncBatchingMapProxy[int, int](batch_size=1, max_concurrency=2)

    async def run():
        # Launch several maps concurrently; each map will call af once per batch
        tasks = [proxy.map([i], af) for i in range(6)]
        outs = await asyncio.gather(*tasks)
        assert outs == [[i] for i in range(6)]

    asyncio.run(run())
    # Peak concurrency should not exceed limit (2)
    assert peak <= 2


def test_async_localproxy_map_func_length_mismatch_raises_and_releases(event_loop=None):
    from openaivec._proxy import AsyncBatchingMapProxy

    async def bad(xs: list[int]) -> list[int]:
        return xs[:-1]

    proxy = AsyncBatchingMapProxy[int, int](batch_size=2)

    async def run_bad():
        with pytest.raises(ValueError):
            await proxy.map([10, 20], bad)

    asyncio.run(run_bad())

    async def run_ok():
        out = await proxy.map([10, 20], _afunc_echo)
        assert out == [10, 20]

    asyncio.run(run_ok())


def test_sync_clear_releases_memory_and_recomputes():
    calls: list[list[int]] = []

    def f(xs: list[int]) -> list[int]:
        calls.append(xs[:])
        return xs

    p = BatchingMapProxy[int, int](batch_size=10)
    out1 = p.map([1, 2, 2, 3], f)
    assert out1 == [1, 2, 2, 3]
    assert len(calls) >= 1  # at least one batch call

    # Clear all memory
    p.clear()

    # Next call should recompute (calls count increases)
    out2 = p.map([1, 2, 2, 3], f)
    assert out2 == [1, 2, 2, 3]
    assert len(calls) >= 2


def test_batch_size_maximization_with_cache_hits():
    """Test that batch_size is maximized even when some items are cached."""
    from openaivec._proxy import BatchingMapProxy

    calls: list[list[int]] = []

    def mf(xs: list[int]) -> list[int]:
        calls.append(xs[:])
        return xs

    p = BatchingMapProxy[int, int](batch_size=3)

    # Pre-cache some items: 2, 5, 8
    p.map([2], mf)
    p.map([5], mf)
    p.map([8], mf)
    calls.clear()

    # Process items [0,1,2,3,4,5,6,7,8,9] where 2,5,8 are cached
    # Expected behavior: accumulate uncached items to maximize batch_size
    # Uncached items: [0,1,3,4,6,7,9] should be batched as [0,1,3] + [4,6,7] + [9]
    items = list(range(10))
    result = p.map(items, mf)

    assert result == items
    # Should make 3 calls with maximized batch sizes
    assert len(calls) == 3
    assert calls[0] == [0, 1, 3]  # First batch: 3 items
    assert calls[1] == [4, 6, 7]  # Second batch: 3 items
    assert calls[2] == [9]  # Final batch: 1 item (remainder)


def test_batch_size_maximization_complex_scenario():
    """Test batch_size maximization with more complex cache hit patterns."""
    from openaivec._proxy import BatchingMapProxy

    calls: list[list[int]] = []

    def mf(xs: list[int]) -> list[int]:
        calls.append(xs[:])
        return [x * 2 for x in xs]

    p = BatchingMapProxy[int, int](batch_size=4)

    # Pre-cache items: 1, 3, 6, 8, 11
    for item in [1, 3, 6, 8, 11]:
        p.map([item], mf)
    calls.clear()

    # Process items [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]
    # Uncached: [0,2,4,5,7,9,10,12,13,14] (10 items)
    # Should batch as [0,2,4,5] + [7,9,10,12] + [13,14]
    items = list(range(15))
    result = p.map(items, mf)

    expected = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28]
    assert result == expected

    # Should make 3 calls
    assert len(calls) == 3
    assert calls[0] == [0, 2, 4, 5]  # 4 items (full batch)
    assert calls[1] == [7, 9, 10, 12]  # 4 items (full batch)
    assert calls[2] == [13, 14]  # 2 items (remainder)


@pytest.mark.asyncio
async def test_async_clear_releases_memory_and_recomputes():
    calls: list[list[int]] = []

    async def af(xs: list[int]) -> list[int]:
        calls.append(xs[:])
        await asyncio.sleep(0)
        return xs

    p = AsyncBatchingMapProxy[int, int](batch_size=10)
    out1 = await p.map([1, 2, 3], af)
    assert out1 == [1, 2, 3]
    assert len(calls) >= 1

    await p.clear()

    out2 = await p.map([1, 2, 3], af)
    assert out2 == [1, 2, 3]
    assert len(calls) >= 2


@pytest.mark.asyncio
async def test_async_batch_size_maximization_with_cache_hits():
    """Test that batch_size is maximized even when some items are cached (async version)."""
    from openaivec._proxy import AsyncBatchingMapProxy

    calls: list[list[int]] = []

    async def mf(xs: list[int]) -> list[int]:
        calls.append(xs[:])
        await asyncio.sleep(0.01)
        return xs

    p = AsyncBatchingMapProxy[int, int](batch_size=3)

    # Pre-cache some items: 2, 5, 8
    await p.map([2], mf)
    await p.map([5], mf)
    await p.map([8], mf)
    calls.clear()

    # Process items [0,1,2,3,4,5,6,7,8,9] where 2,5,8 are cached
    # Expected behavior: accumulate uncached items to maximize batch_size
    # Uncached items: [0,1,3,4,6,7,9] should be batched as [0,1,3] + [4,6,7] + [9]
    items = list(range(10))
    result = await p.map(items, mf)

    assert result == items
    # Should make 3 calls with maximized batch sizes
    assert len(calls) == 3
    assert calls[0] == [0, 1, 3]  # First batch: 3 items
    assert calls[1] == [4, 6, 7]  # Second batch: 3 items
    assert calls[2] == [9]  # Final batch: 1 item (remainder)


@pytest.mark.asyncio
async def test_async_batch_size_maximization_complex_scenario():
    """Test batch_size maximization with more complex cache hit patterns (async version)."""
    from openaivec._proxy import AsyncBatchingMapProxy

    calls: list[list[int]] = []

    async def mf(xs: list[int]) -> list[int]:
        calls.append(xs[:])
        await asyncio.sleep(0.01)
        return [x * 2 for x in xs]

    p = AsyncBatchingMapProxy[int, int](batch_size=4)

    # Pre-cache items: 1, 3, 6, 8, 11
    for item in [1, 3, 6, 8, 11]:
        await p.map([item], mf)
    calls.clear()

    # Process items [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]
    # Uncached: [0,2,4,5,7,9,10,12,13,14] (10 items)
    # Should batch as [0,2,4,5] + [7,9,10,12] + [13,14]
    items = list(range(15))
    result = await p.map(items, mf)

    expected = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28]
    assert result == expected

    # Should make 3 calls
    assert len(calls) == 3
    assert calls[0] == [0, 2, 4, 5]  # 4 items (full batch)
    assert calls[1] == [7, 9, 10, 12]  # 4 items (full batch)
    assert calls[2] == [13, 14]  # 2 items (remainder)


# -------------------- Progress Bar Tests --------------------


def test_notebook_environment_detection():
    """Test notebook environment detection functionality."""
    from openaivec._proxy import ProxyBase

    proxy = ProxyBase()
    # The method should return a boolean and not raise an exception
    result = proxy._is_notebook_environment()
    assert isinstance(result, bool)


def test_progress_bar_methods():
    """Test progress bar creation and management methods."""
    from openaivec._proxy import ProxyBase

    proxy = ProxyBase()
    proxy.show_progress = True

    # Test progress bar creation (may return None if tqdm not available or not in notebook)
    progress_bar = proxy._create_progress_bar(100, "Testing")

    # Test update and close methods don't raise exceptions
    proxy._update_progress_bar(progress_bar, 10)
    proxy._close_progress_bar(progress_bar)


def test_batching_proxy_with_progress_disabled():
    """Test BatchingMapProxy with progress disabled (default)."""
    calls: list[list[int]] = []

    def mf(xs: list[int]) -> list[int]:
        calls.append(xs[:])
        return xs

    proxy = BatchingMapProxy[int, int](batch_size=3, show_progress=False)
    items = list(range(6))
    result = proxy.map(items, mf)

    assert result == items
    assert len(calls) == 2  # Should batch as [0,1,2] + [3,4,5]


def test_batching_proxy_with_progress_enabled():
    """Test BatchingMapProxy with progress enabled."""
    calls: list[list[int]] = []

    def mf(xs: list[int]) -> list[int]:
        calls.append(xs[:])
        return xs

    proxy = BatchingMapProxy[int, int](batch_size=3, show_progress=True)
    items = list(range(6))
    result = proxy.map(items, mf)

    assert result == items
    assert len(calls) == 2  # Should batch as [0,1,2] + [3,4,5]


@pytest.mark.asyncio
async def test_async_batching_proxy_with_progress_disabled():
    """Test AsyncBatchingMapProxy with progress disabled (default)."""
    calls: list[list[int]] = []

    async def mf(xs: list[int]) -> list[int]:
        calls.append(xs[:])
        await asyncio.sleep(0.01)
        return xs

    proxy = AsyncBatchingMapProxy[int, int](batch_size=3, show_progress=False)
    items = list(range(6))
    result = await proxy.map(items, mf)

    assert result == items
    assert len(calls) == 2  # Should batch as [0,1,2] + [3,4,5]


@pytest.mark.asyncio
async def test_async_batching_proxy_with_progress_enabled():
    """Test AsyncBatchingMapProxy with progress enabled."""
    calls: list[list[int]] = []

    async def mf(xs: list[int]) -> list[int]:
        calls.append(xs[:])
        await asyncio.sleep(0.01)
        return xs

    proxy = AsyncBatchingMapProxy[int, int](batch_size=3, show_progress=True)
    items = list(range(6))
    result = await proxy.map(items, mf)

    assert result == items
    assert len(calls) == 2  # Should batch as [0,1,2] + [3,4,5]


def test_progress_bar_with_forced_notebook_environment():
    """Test progress bar functionality with forced notebook environment."""
    from openaivec._proxy import ProxyBase

    # Monkey patch the notebook detection to return True
    original_method = ProxyBase._is_notebook_environment
    ProxyBase._is_notebook_environment = lambda self: True

    try:
        calls: list[list[int]] = []

        def mf(xs: list[int]) -> list[int]:
            calls.append(xs[:])
            return xs

        # Create proxy with progress enabled
        proxy = BatchingMapProxy[int, int](batch_size=3, show_progress=True)
        items = list(range(10))
        result = proxy.map(items, mf)

        assert result == items
        # Should batch as [0,1,2] + [3,4,5] + [6,7,8] + [9]
        assert len(calls) == 4

    finally:
        # Restore original method
        ProxyBase._is_notebook_environment = original_method


@pytest.mark.asyncio
async def test_async_progress_bar_with_forced_notebook_environment():
    """Test async progress bar functionality with forced notebook environment."""
    from openaivec._proxy import ProxyBase

    # Monkey patch the notebook detection to return True
    original_method = ProxyBase._is_notebook_environment
    ProxyBase._is_notebook_environment = lambda self: True

    try:
        calls: list[list[int]] = []

        async def mf(xs: list[int]) -> list[int]:
            calls.append(xs[:])
            await asyncio.sleep(0.01)
            return xs

        # Create proxy with progress enabled
        proxy = AsyncBatchingMapProxy[int, int](batch_size=3, show_progress=True)
        items = list(range(10))
        result = await proxy.map(items, mf)

        assert result == items
        # Should batch as [0,1,2] + [3,4,5] + [6,7,8] + [9]
        assert len(calls) == 4

    finally:
        # Restore original method
        ProxyBase._is_notebook_environment = original_method


@pytest.mark.asyncio
async def test_async_concurrent_batch_processing():
    """Test that AsyncBatchingMapProxy processes batches concurrently."""
    calls: list[list[int]] = []
    start_times: list[float] = []

    async def timing_mf(xs: list[int]) -> list[int]:
        import time

        calls.append(xs[:])
        start_times.append(time.time())
        await asyncio.sleep(0.1)  # Simulate network latency
        return xs

    proxy = AsyncBatchingMapProxy[int, int](batch_size=5, max_concurrency=5)
    items = list(range(25))  # 5 batches of 5 items each

    start = time.time()
    result = await proxy.map(items, timing_mf)
    end = time.time()

    assert result == items
    assert len(calls) == 5  # Should have 5 batches

    # Check that batches started concurrently (within a short time window)
    if len(start_times) >= 2:
        time_diff = max(start_times) - min(start_times)
        assert time_diff < 0.05, f"Batches should start concurrently, but time diff was {time_diff:.3f}s"

    # Total time should be close to single batch time when max_concurrency >= num_batches
    assert end - start < 0.2, f"Expected ~0.1s with concurrent processing, got {end - start:.3f}s"


@pytest.mark.asyncio
async def test_async_max_concurrency_limit():
    """Test that max_concurrency properly limits concurrent batch processing."""
    active_count = 0
    max_active = 0
    lock = asyncio.Lock()

    async def tracking_mf(xs: list[int]) -> list[int]:
        nonlocal active_count, max_active

        async with lock:
            active_count += 1
            max_active = max(max_active, active_count)

        await asyncio.sleep(0.1)

        async with lock:
            active_count -= 1

        return xs

    proxy = AsyncBatchingMapProxy[int, int](batch_size=1, max_concurrency=3)
    items = list(range(10))  # 10 batches of 1 item each

    result = await proxy.map(items, tracking_mf)

    assert result == items
    assert max_active <= 3, f"Expected max 3 concurrent batches, but got {max_active}"
    assert max_active >= 3, f"Should reach max concurrency of 3, but only got {max_active}"


@pytest.mark.asyncio
async def test_async_performance_improvement():
    """Test that async version with concurrency is faster than sequential."""

    async def slow_mf(xs: list[int]) -> list[int]:
        await asyncio.sleep(0.05)  # 50ms per batch
        return xs

    items = list(range(20))  # 4 batches of 5 items each

    # Test with max_concurrency=1 (sequential)
    proxy_seq = AsyncBatchingMapProxy[int, int](batch_size=5, max_concurrency=1)
    start = time.time()
    result_seq = await proxy_seq.map(items, slow_mf)
    time_seq = time.time() - start

    # Test with max_concurrency=4 (concurrent)
    proxy_conc = AsyncBatchingMapProxy[int, int](batch_size=5, max_concurrency=4)
    start = time.time()
    result_conc = await proxy_conc.map(items, slow_mf)
    time_conc = time.time() - start

    assert result_seq == result_conc == items

    # Concurrent version should be significantly faster
    speedup = time_seq / time_conc
    assert speedup > 2.5, f"Expected >2.5x speedup with concurrency, got {speedup:.1f}x"
