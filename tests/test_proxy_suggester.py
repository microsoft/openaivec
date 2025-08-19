"""Test cases for BatchSizeSuggester integration with proxies."""

import asyncio

import pytest

from openaivec._proxy import AsyncBatchingMapProxy, BatchingMapProxy


def test_sync_proxy_uses_suggester_when_batch_size_none():
    """Test that BatchingMapProxy uses suggester when batch_size is None."""
    proxy = BatchingMapProxy[int, str](batch_size=None)

    # Set a known batch size in the suggester
    proxy.suggester.current_batch_size = 5

    call_count = 0
    batch_sizes_used = []

    def map_func(items: list[int]) -> list[str]:
        nonlocal call_count
        call_count += 1
        batch_sizes_used.append(len(items))
        return [str(x) for x in items]

    # Process 20 items - should use suggester's batch size (5)
    items = list(range(20))
    result = proxy.map(items, map_func)

    assert result == [str(x) for x in items]
    assert call_count == 4  # 20 items / 5 batch_size = 4 calls
    assert all(size == 5 for size in batch_sizes_used)


def test_sync_proxy_suggester_adapts_batch_size():
    """Test that suggester adapts batch size based on execution time."""
    proxy = BatchingMapProxy[int, str](batch_size=None)

    # Configure suggester for testing
    proxy.suggester.min_batch_size = 2
    proxy.suggester.current_batch_size = 2
    proxy.suggester.min_duration = 0.001  # 1ms
    proxy.suggester.max_duration = 0.002  # 2ms
    proxy.suggester.sample_size = 2
    proxy.suggester.step_ratio = 0.5

    import time

    def fast_map_func(items: list[int]) -> list[str]:
        # Simulate very fast processing (should increase batch size)
        time.sleep(0.0001)  # 0.1ms
        return [str(x) for x in items]

    # First calls use initial batch size
    items = list(range(10))
    result1 = proxy.map(items, fast_map_func)
    assert result1 == [str(x) for x in items]

    # Manually call suggest_batch_size to trigger adaptation
    new_size = proxy.suggester.suggest_batch_size()

    # After enough samples, suggester should increase batch size
    items2 = list(range(10))
    result2 = proxy.map(items2, fast_map_func)
    assert result2 == [str(x) for x in items2]

    # Check that batch size increased
    assert new_size > 2 or proxy.suggester.current_batch_size > 2


def test_sync_proxy_respects_total_when_suggested_exceeds():
    """Test that normalized_batch_size doesn't exceed total items."""
    proxy = BatchingMapProxy[int, str](batch_size=None)

    # Set suggester to suggest a large batch size
    proxy.suggester.current_batch_size = 100

    call_count = 0
    batch_sizes_used = []

    def map_func(items: list[int]) -> list[str]:
        nonlocal call_count
        call_count += 1
        batch_sizes_used.append(len(items))
        return [str(x) for x in items]

    # Process only 10 items (less than suggester's 100)
    items = list(range(10))
    result = proxy.map(items, map_func)

    assert result == [str(x) for x in items]
    assert call_count == 1  # Should process all in one batch
    assert batch_sizes_used[0] == 10  # Should not exceed total


@pytest.mark.asyncio
async def test_async_proxy_uses_suggester_when_batch_size_none():
    """Test that AsyncBatchingMapProxy uses suggester when batch_size is None."""
    proxy = AsyncBatchingMapProxy[int, str](batch_size=None)

    # Set a known batch size in the suggester
    proxy.suggester.current_batch_size = 3

    call_count = 0
    batch_sizes_used = []

    async def map_func(items: list[int]) -> list[str]:
        nonlocal call_count
        call_count += 1
        batch_sizes_used.append(len(items))
        await asyncio.sleep(0.001)
        return [str(x) for x in items]

    # Process 12 items - should use suggester's batch size (3)
    items = list(range(12))
    result = await proxy.map(items, map_func)

    assert result == [str(x) for x in items]
    assert call_count == 4  # 12 items / 3 batch_size = 4 calls
    assert all(size == 3 for size in batch_sizes_used)


@pytest.mark.asyncio
async def test_async_proxy_suggester_with_concurrency():
    """Test that async proxy with suggester works with concurrent processing."""
    proxy = AsyncBatchingMapProxy[int, str](batch_size=None, max_concurrency=3)

    # Set a known batch size in the suggester
    proxy.suggester.current_batch_size = 2

    processing_times = []

    async def map_func(items: list[int]) -> list[str]:
        start = asyncio.get_event_loop().time()
        await asyncio.sleep(0.01)  # 10ms per batch
        processing_times.append((start, asyncio.get_event_loop().time()))
        return [str(x) for x in items]

    # Process 10 items - should create 5 batches of 2
    items = list(range(10))
    result = await proxy.map(items, map_func)

    assert result == [str(x) for x in items]
    assert len(processing_times) == 5  # 10 items / 2 batch_size = 5 calls

    # Check for concurrent execution (some batches should overlap in time)
    overlaps = 0
    for i in range(len(processing_times)):
        for j in range(i + 1, len(processing_times)):
            start_i, end_i = processing_times[i]
            start_j, end_j = processing_times[j]
            # Check if time ranges overlap
            if start_i < end_j and start_j < end_i:
                overlaps += 1

    assert overlaps > 0, "Expected some batches to run concurrently"


def test_sync_proxy_batch_size_overrides_suggester():
    """Test that explicit batch_size takes precedence over suggester."""
    proxy = BatchingMapProxy[int, str](batch_size=7)

    # Even though suggester might suggest different size
    proxy.suggester.current_batch_size = 3

    batch_sizes_used = []

    def map_func(items: list[int]) -> list[str]:
        batch_sizes_used.append(len(items))
        return [str(x) for x in items]

    # Process 20 items with explicit batch_size=7
    items = list(range(20))
    result = proxy.map(items, map_func)

    assert result == [str(x) for x in items]
    # Should use batch_size=7, not suggester's 3
    assert batch_sizes_used == [7, 7, 6]  # 7 + 7 + 6 = 20


def test_sync_proxy_zero_batch_size_processes_all():
    """Test that batch_size=0 processes all items at once."""
    proxy = BatchingMapProxy[int, str](batch_size=0)

    # Suggester should not be used
    proxy.suggester.current_batch_size = 5

    call_count = 0

    def map_func(items: list[int]) -> list[str]:
        nonlocal call_count
        call_count += 1
        return [str(x) for x in items]

    # Process 100 items with batch_size=0
    items = list(range(100))
    result = proxy.map(items, map_func)

    assert result == [str(x) for x in items]
    assert call_count == 1  # Should process all in one call
