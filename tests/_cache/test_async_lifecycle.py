import asyncio

import pytest

from openaivec._cache import AsyncBatchCache


@pytest.mark.asyncio
async def test_cancelled_owner_releases_keys_for_reuse():
    cache = AsyncBatchCache[str, str](batch_size=1, max_concurrency=1, show_progress=False)
    started = asyncio.Event()
    release = asyncio.Event()

    async def blocked(items: list[str]) -> list[str]:
        started.set()
        await release.wait()
        return items

    async def succeed(items: list[str]) -> list[str]:
        return items

    owner = asyncio.create_task(cache.map(["key"], blocked))
    try:
        await asyncio.wait_for(started.wait(), 2)
        owner.cancel()
        with pytest.raises(asyncio.CancelledError):
            await owner
        assert getattr(cache, "_inflight") == {}
        assert getattr(cache, "_active_calls") == 0
        assert await asyncio.wait_for(cache.map(["key"], succeed), 2) == ["key"]
    finally:
        release.set()
        owner.cancel()
        await asyncio.gather(owner, return_exceptions=True)
        await cache.clear()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["exception", "length"])
async def test_failed_map_drains_siblings_before_reuse(failure: str):
    cache = AsyncBatchCache[int, str](batch_size=1, max_concurrency=2, show_progress=False)
    started = asyncio.Event()
    stopped = asyncio.Event()
    release = asyncio.Event()
    workers: list[asyncio.Task[object]] = []

    async def mapper(items: list[int]) -> list[str]:
        current = asyncio.current_task()
        assert current is not None
        workers.append(current)
        if items == [0]:
            await started.wait()
            if failure == "length":
                return []
            raise ValueError("mapper failed")
        started.set()
        try:
            await release.wait()
            return ["old"]
        finally:
            stopped.set()

    async def succeed(items: list[int]) -> list[str]:
        return ["new"] * len(items)

    try:
        with pytest.raises(ValueError):
            await asyncio.wait_for(cache.map([0, 1], mapper), 2)
        assert stopped.is_set()
        assert all(worker.done() for worker in workers)
        assert getattr(cache, "_inflight") == {}
        assert getattr(cache, "_active_calls") == 0
        assert await asyncio.wait_for(cache.map([1], succeed), 2) == ["new"]
        release.set()
        await asyncio.gather(*workers, return_exceptions=True)
        assert cache.cache[1] == "new"
    finally:
        release.set()
        await asyncio.gather(*workers, return_exceptions=True)
        await cache.clear()


@pytest.mark.asyncio
async def test_cancellation_with_full_producer_queue():
    cache = AsyncBatchCache[int, int](batch_size=1, max_concurrency=1, show_progress=False)
    started = asyncio.Event()
    release = asyncio.Event()

    async def blocked(items: list[int]) -> list[int]:
        started.set()
        await release.wait()
        return items

    owner = asyncio.create_task(cache.map(list(range(20)), blocked))
    try:
        await asyncio.wait_for(started.wait(), 2)
        owner.cancel()
        done, _ = await asyncio.wait([owner], timeout=2)
        assert owner in done
        with pytest.raises(asyncio.CancelledError):
            await owner
        assert getattr(cache, "_inflight") == {}
        assert getattr(cache, "_active_calls") == 0
    finally:
        release.set()
        owner.cancel()
        await asyncio.gather(owner, return_exceptions=True)
        await cache.clear()


@pytest.mark.asyncio
async def test_cancellation_while_waiting_for_shared_semaphore():
    cache = AsyncBatchCache[str, str](batch_size=1, max_concurrency=1, show_progress=False)
    started = asyncio.Event()
    release = asyncio.Event()

    async def blocked(items: list[str]) -> list[str]:
        started.set()
        await release.wait()
        return items

    first = asyncio.create_task(cache.map(["first"], blocked))
    await asyncio.wait_for(started.wait(), 2)
    second = asyncio.create_task(cache.map(["second"], blocked))
    try:
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        semaphore = getattr(cache, "_AsyncBatchCache__sema")
        assert len(semaphore._waiters) == 1
        second.cancel()
        with pytest.raises(asyncio.CancelledError):
            await second
        assert list(getattr(cache, "_inflight")) == ["first"]
        assert getattr(cache, "_active_calls") == 1
        release.set()
        assert await asyncio.wait_for(first, 2) == ["first"]
        assert await asyncio.wait_for(cache.map(["second"], blocked), 2) == ["second"]
    finally:
        release.set()
        first.cancel()
        second.cancel()
        await asyncio.gather(first, second, return_exceptions=True)
        await cache.clear()


@pytest.mark.asyncio
async def test_cancellation_releases_rescued_keys_while_waiting_for_other_owner():
    cache = AsyncBatchCache[str, str](show_progress=False)
    inflight: dict[str, asyncio.Event] = getattr(cache, "_inflight")
    abandoned = asyncio.Event()
    busy = asyncio.Event()
    inflight.update(abandoned=abandoned, busy=busy)

    async def succeed(items: list[str]) -> list[str]:
        return items

    waiter = asyncio.create_task(cache.map(["abandoned", "busy"], succeed))
    try:
        await asyncio.sleep(0)
        del inflight["abandoned"]
        abandoned.set()
        await asyncio.sleep(0)
        rescued = inflight["abandoned"]
        assert rescued is not abandoned
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        assert rescued.is_set()
        assert inflight == {"busy": busy}
        assert await asyncio.wait_for(cache.map(["abandoned"], succeed), 2) == ["abandoned"]
    finally:
        waiter.cancel()
        await asyncio.gather(waiter, return_exceptions=True)
        await cache.clear()


@pytest.mark.asyncio
async def test_overlapping_waiter_recovers_after_owner_cancellation():
    cache = AsyncBatchCache[str, str](show_progress=False, max_cache_size=1)
    started = asyncio.Event()
    release = asyncio.Event()

    async def blocked(items: list[str]) -> list[str]:
        started.set()
        await release.wait()
        return items

    async def succeed(items: list[str]) -> list[str]:
        return ["new"] * len(items)

    owner = asyncio.create_task(cache.map(["first", "second"], blocked))
    await asyncio.wait_for(started.wait(), 2)
    waiter = asyncio.create_task(cache.map(["first", "second", "first"], succeed))
    try:
        await asyncio.sleep(0)
        owner.cancel()
        with pytest.raises(asyncio.CancelledError):
            await owner
        assert await asyncio.wait_for(waiter, 2) == ["new", "new", "new"]
        assert getattr(cache, "_inflight") == {}
        assert getattr(cache, "_active_calls") == 0
        assert len(cache.cache) == 1
    finally:
        release.set()
        owner.cancel()
        waiter.cancel()
        await asyncio.gather(owner, waiter, return_exceptions=True)
        await cache.clear()


@pytest.mark.asyncio
async def test_finalization_preserves_newer_ownership():
    cache = AsyncBatchCache[str, str](show_progress=False)
    older = asyncio.Event()
    newer = asyncio.Event()
    inflight: dict[str, asyncio.Event] = getattr(cache, "_inflight")
    inflight["key"] = newer
    await getattr(cache, "_AsyncBatchCache__finalize_success")({"key": older}, ["old"])
    await getattr(cache, "_AsyncBatchCache__finalize_failure")({"key": older})
    assert inflight == {"key": newer}
    assert "key" not in cache.cache
    assert older.is_set()
    assert not newer.is_set()
    await cache.clear()
