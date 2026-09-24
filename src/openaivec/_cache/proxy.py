import asyncio
import threading
from collections.abc import Awaitable, Callable, Hashable
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar, cast

from openai import APIStatusError, BadRequestError

from openaivec._cache import BatchSizeSuggester
from openaivec._cache._backend import CacheBackend, InMemoryCacheBackend

__all__ = []

S = TypeVar("S", bound=Hashable)
T = TypeVar("T")
DEFAULT_MANAGED_CACHE_SIZE = 4096
_MAX_SIZE_SPLITS = 16
_SIZE_ERROR_CODES = frozenset(
    {"context_length_exceeded", "max_tokens_exceeded", "request_too_large", "input_too_large", "too_many_inputs"}
)


def _is_request_size_error(error: Exception) -> bool:
    if not isinstance(error, APIStatusError):
        return False
    if error.status_code == 413:
        return True
    if not isinstance(error, BadRequestError):
        return False
    body = error.body
    if isinstance(body, dict):
        details = body.get("error", body)
        if isinstance(details, dict) and details.get("code") in _SIZE_ERROR_CODES:
            return True
    return error.code in _SIZE_ERROR_CODES


def _default_cache_backend() -> InMemoryCacheBackend:
    """Factory that returns the default in-memory CacheBackend."""
    return InMemoryCacheBackend()


class BatchCacheBase(Generic[S, T]):
    """Common utilities shared by BatchCache and AsyncBatchCache.

    Provides order-preserving deduplication and batch size normalization that
    depend only on ``batch_size`` and do not touch concurrency primitives.

    Attributes:
        batch_size: Optional mini-batch size hint used by implementations to
            split work into chunks. When unset or non-positive, implementations
            should process the entire input in a single call.
        max_cache_size: Optional retention target applied after active calls
            drain. When set, implementations evict oldest cached items once
            no callers are still reading results.
    """

    batch_size: int | None  # subclasses may override via dataclass
    max_cache_size: int | None
    show_progress: bool  # Enable progress bar display
    suggester: BatchSizeSuggester  # Batch size optimization, initialized by subclasses

    @staticmethod
    def _touch_keys_unlocked(cache: CacheBackend[S, T], keys: list[S]) -> None:
        """Mark keys as recently used in an ordered cache."""
        unique = BatchCacheBase._unique_in_order(keys)
        touch_many = getattr(cache, "touch_many", None)
        if callable(touch_many):
            touch_many(unique)
            return
        for key in unique:
            if key in cache:
                cache.move_to_end(key)

    @staticmethod
    def _cached_values_unlocked(cache: CacheBackend[S, T], keys: list[S]) -> dict[S, T]:
        """Look up keys without changing their recency, using bulk I/O if available."""
        if not keys:
            return {}
        unique = BatchCacheBase._unique_in_order(keys)
        get_many = getattr(cache, "get_many", None)
        if callable(get_many):
            return cast(dict[S, T], get_many(unique))
        return {key: cache[key] for key in unique if key in cache}

    @staticmethod
    def _put_values_unlocked(cache: CacheBackend[S, T], items: list[tuple[S, T]]) -> None:
        """Store results in order, using bulk I/O if available."""
        if not items:
            return
        put_many = getattr(cache, "put_many", None)
        if callable(put_many):
            put_many(items)
            return
        for key, value in items:
            cache[key] = value
            cache.move_to_end(key)

    @staticmethod
    def _prune_cache_unlocked(cache: CacheBackend[S, T], max_cache_size: int | None) -> None:
        """Evict oldest cached items until the cache fits the configured limit."""
        if max_cache_size is None:
            return
        while len(cache) > max_cache_size:
            cache.pop_oldest()

    def _is_notebook_environment(self) -> bool:
        """Check whether the active IPython shell owns a notebook kernel.

        Returns:
            bool: True only with a kernel-backed shell. Installed packages and
                inherited notebook environment variables are not sufficient.
        """
        import importlib

        try:
            ipython_module = importlib.import_module("IPython")
        except ImportError:
            return False
        get_ipython = getattr(ipython_module, "get_ipython", None)
        ipython = get_ipython() if callable(get_ipython) else None
        return getattr(ipython, "kernel", None) is not None

    def _create_progress_bar(self, total: int, desc: str = "Processing batches") -> Any:
        """Create a progress bar if conditions are met.

        Args:
            total (int): Total number of items to process.
            desc (str): Description for the progress bar.

        Returns:
            Any: Progress bar instance or None if not available.
        """
        if not self.show_progress or not self._is_notebook_environment():
            return None

        # Prefer notebook-specific rendering first; fall back to auto
        # when widget backends are unavailable in the current runtime.
        for module_name in ("tqdm.notebook", "tqdm.auto"):
            try:
                if module_name == "tqdm.notebook":
                    from tqdm.notebook import tqdm as tqdm_progress
                else:
                    from tqdm.auto import tqdm as tqdm_progress
                return tqdm_progress(total=total, desc=desc, unit="item")
            except Exception:
                continue
        return None

    def _update_progress_bar(self, progress_bar: Any, increment: int) -> None:
        """Update progress bar with the given increment.

        Args:
            progress_bar (Any): Progress bar instance.
            increment (int): Number of items to increment.
        """
        if progress_bar is not None:
            progress_bar.update(increment)

    def _close_progress_bar(self, progress_bar: Any) -> None:
        """Close the progress bar.

        Args:
            progress_bar (Any): Progress bar instance.
        """
        if progress_bar is not None:
            progress_bar.close()

    @staticmethod
    def _unique_in_order(seq: list[S]) -> list[S]:
        """Return unique items preserving their first-occurrence order.

        Args:
            seq (list[S]): Sequence of items which may contain duplicates.

        Returns:
            list[S]: A new list containing each distinct item from ``seq`` exactly
            once, in the order of their first occurrence.
        """
        seen: set[S] = set()
        out: list[S] = []
        for x in seq:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    def _normalized_batch_size(self, total: int) -> int:
        """Compute the effective batch size used for processing.

        If ``batch_size`` is None, use the suggester to determine optimal batch size.
        If ``batch_size`` is non-positive, process the entire ``total`` in a single call.

        Args:
            total (int): Number of items intended to be processed.

        Returns:
            int: The positive batch size to use.
        """
        if self.batch_size and self.batch_size > 0:
            return self.batch_size
        elif self.batch_size is None:
            # Use suggester to determine optimal batch size
            suggested = self.suggester.suggest_batch_size()
            return min(suggested, total)  # Don't exceed total items
        else:
            # batch_size is 0 or negative, process all at once
            return total


@dataclass
class BatchCache(BatchCacheBase[S, T], Generic[S, T]):
    """Thread-safe local proxy that caches results of a mapping function.

    This proxy batches calls to the ``map_func`` you pass to ``map()`` (if
    ``batch_size`` is set),
    deduplicates inputs while preserving order, and ensures that concurrent calls do
    not duplicate work via an in-flight registry. All public behavior is preserved
    while minimizing redundant requests and maintaining input order in the output.
    Valid ``None`` results are cached like any other value.

    When ``batch_size=None``, automatic batch size optimization is enabled,
    dynamically adjusting batch sizes based on execution time to maintain optimal
    performance (targeting 30-60 seconds per batch). When ``max_cache_size`` is
    configured, pruning happens only after the last overlapping ``map()`` call
    exits so in-flight readers cannot lose freshly computed values.

    Example:
        ```python
        p = BatchCache[int, str](batch_size=3)

        def f(xs: list[int]) -> list[str]:
            return [f"v:{x}" for x in xs]

        p.map([1, 2, 2, 3, 4], f)
        # ['v:1', 'v:2', 'v:2', 'v:3', 'v:4']
        ```
    """

    # Number of items to process per call to map_func.
    # - If None (default): Enables automatic batch size optimization, dynamically adjusting
    #   based on execution time (targeting 30-60 seconds per batch)
    # - If positive integer: Fixed batch size
    # - If <= 0: Process all items at once
    batch_size: int | None = None
    max_cache_size: int | None = None
    show_progress: bool = True
    suggester: BatchSizeSuggester = field(default_factory=BatchSizeSuggester, repr=False)

    cache: CacheBackend[S, T] = field(default_factory=_default_cache_backend)

    # internals
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)
    _inflight: dict[S, threading.Event] = field(default_factory=dict, repr=False)
    _active_calls: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate optional cache-retention settings."""
        if self.max_cache_size is not None and self.max_cache_size < 1:
            raise ValueError("max_cache_size must be >= 1")

    def __all_cached(self, items: list[S]) -> bool:
        """Check whether all items are present in the cache.

        This method acquires the internal lock to perform a consistent check.

        Args:
            items (list[S]): Items to verify against the cache.

        Returns:
            bool: True if every item is already cached, False otherwise.
        """
        with self._lock:
            cached = self._cached_values_unlocked(self.cache, items)
            return all(x in cached for x in items)

    def __values(self, items: list[S]) -> list[T]:
        """Fetch cached values for ``items`` preserving the given order.

        This method acquires the internal lock while reading the cache.

        Args:
            items (list[S]): Items to retrieve from the cache.

        Returns:
            list[T]: The cached values corresponding to ``items`` in the same
            order.
        """
        with self._lock:
            cached = self._cached_values_unlocked(self.cache, items)
            values = [cached[x] for x in items]
            self._touch_keys_unlocked(self.cache, items)
            return values

    def __acquire_ownership(self, items: list[S]) -> tuple[list[S], list[S]]:
        """Acquire ownership for missing items and identify keys to wait for.

        For each unique item, if it's already cached, it is ignored. If it's
        currently being computed by another thread (in-flight), it is added to
        the wait list. Otherwise, this method marks the key as in-flight and
        considers it "owned" by the current thread.

        Args:
            items (list[S]): Unique items (order-preserving) to be processed.

        Returns:
            tuple[list[S], list[S]]: A tuple ``(owned, wait_for)`` where
            - ``owned`` are items this thread is responsible for computing.
            - ``wait_for`` are items that another thread is already computing.
        """
        owned: list[S] = []
        wait_for: list[S] = []
        with self._lock:
            cached = self._cached_values_unlocked(self.cache, items)
            for x in items:
                if x in cached:
                    continue
                if x in self._inflight:
                    wait_for.append(x)
                else:
                    self._inflight[x] = threading.Event()
                    owned.append(x)
        return owned, wait_for

    def __finalize_success(self, to_call: list[S], results: list[T]) -> None:
        """Populate cache with results and signal completion events.

        Args:
            to_call (list[S]): Items that were computed.
            results (list[T]): Results corresponding to ``to_call`` in order.
        """
        if len(results) != len(to_call):
            # Prevent deadlocks if map_func violates the contract.
            # Release waiters and surface a clear error.
            self.__finalize_failure(to_call)
            raise ValueError("map_func must return a list of results with the same length and order as inputs")
        with self._lock:
            self._put_values_unlocked(self.cache, list(zip(to_call, results)))
            for x in to_call:
                ev = self._inflight.pop(x, None)
                if ev:
                    ev.set()

    def __finalize_failure(self, to_call: list[S]) -> None:
        """Release in-flight events on failure to avoid deadlocks.

        Args:
            to_call (list[S]): Items that were intended to be computed when an
            error occurred.
        """
        with self._lock:
            for x in to_call:
                ev = self._inflight.pop(x, None)
                if ev:
                    ev.set()

    def clear(self) -> None:
        """Clear all cached results and release any in-flight waiters.

        Notes:
            - Intended to be called after all processing is finished.
            - Do not call concurrently with active map() calls to avoid
              unnecessary recomputation or racy wake-ups.
        """
        with self._lock:
            for ev in self._inflight.values():
                ev.set()
            self._inflight.clear()
            self.cache.clear()
            self._active_calls = 0

    def close(self) -> None:
        """Alias for clear()."""
        self.clear()

    def __process_owned(self, owned: list[S], map_func: Callable[[list[S]], list[T]]) -> None:
        """Process owned items in mini-batches and fill the cache.

        Before calling ``map_func`` for each batch, the cache is re-checked
        to skip any items that may have been filled in the meantime. Items
        are accumulated across multiple original batches to maximize batch
        size utilization when some items are cached. On exceptions raised
        by ``map_func``, all corresponding in-flight events are released
        to prevent deadlocks, and the exception is propagated.

        Args:
            owned (list[S]): Items for which the current thread has computation
            ownership.

        Raises:
            Exception: Propagates any exception raised by ``map_func``.
        """
        if not owned:
            return
        # Setup progress bar
        progress_bar = self._create_progress_bar(len(owned))

        # Accumulate uncached items to maximize batch size utilization
        pending_to_call: list[S] = []

        try:
            i = 0
            while i < len(owned):
                remaining = len(owned) - i
                current_batch_size = self._normalized_batch_size(remaining)
                batch = owned[i : i + current_batch_size]
                # Double-check cache right before processing
                with self._lock:
                    cached = self._cached_values_unlocked(self.cache, batch)
                    uncached_in_batch = [x for x in batch if x not in cached]

                pending_to_call.extend(uncached_in_batch)

                # Process accumulated items when we reach batch_size or at the end
                is_last_batch = i + current_batch_size >= len(owned)
                if len(pending_to_call) >= current_batch_size or (is_last_batch and pending_to_call):
                    # Take up to batch_size items to process
                    to_call = pending_to_call[:current_batch_size]
                    pending_to_call = pending_to_call[current_batch_size:]

                    results = self.__map_with_size_recovery(to_call, map_func)
                    self.__finalize_success(to_call, results)

                    # Update progress bar
                    self._update_progress_bar(progress_bar, len(to_call))

                # Move to next batch
                i += current_batch_size

            # Process any remaining items
            while pending_to_call:
                # Get dynamic batch size for remaining items
                remaining_batch_size = self._normalized_batch_size(len(pending_to_call))
                to_call = pending_to_call[:remaining_batch_size]
                pending_to_call = pending_to_call[remaining_batch_size:]

                results = self.__map_with_size_recovery(to_call, map_func)
                self.__finalize_success(to_call, results)

                # Update progress bar
                self._update_progress_bar(progress_bar, len(to_call))
        finally:
            self._close_progress_bar(progress_bar)

    def __map_with_size_recovery(
        self, items: list[S], map_func: Callable[[list[S]], list[T]], splits: int = 0
    ) -> list[T]:
        try:
            with self.suggester.record(len(items)):
                results = map_func(items)
            if len(results) != len(items):
                raise ValueError("map_func must return a list of results with the same length and order as inputs")
            return results
        except Exception as error:
            if not _is_request_size_error(error) or len(items) == 1 or splits >= _MAX_SIZE_SPLITS:
                raise
            if self.batch_size is None:
                self.suggester.reduce_after_size_error(len(items))
            middle = len(items) // 2
            return self.__map_with_size_recovery(items[:middle], map_func, splits + 1) + self.__map_with_size_recovery(
                items[middle:], map_func, splits + 1
            )

    def __wait_for(self, keys: list[S], map_func: Callable[[list[S]], list[T]]) -> None:
        """Wait for other threads to complete computations for the given keys.

        If a key is neither cached nor in-flight, this method now claims ownership
        for that key immediately (registers an in-flight Event) and defers the
        computation so that all such rescued keys can be processed together in a
        single batched call to ``map_func`` after the scan completes. This avoids
        high-cost single-item calls.

        Args:
            keys (list[S]): Items whose computations are owned by other threads.
        """
        rescued: list[S] = []  # keys we claim to batch-process
        try:
            for x in keys:
                while True:
                    waiter: threading.Event | None = None
                    with self._lock:
                        if x in self._cached_values_unlocked(self.cache, [x]):
                            break
                        waiter = self._inflight.get(x)
                        if waiter is None:
                            # Not cached and no one computing; claim ownership to batch later.
                            self._inflight[x] = threading.Event()
                            rescued.append(x)
                            break
                    # Someone else is computing; wait for completion.
                    waiter.wait()
            if rescued:
                self.__process_owned(rescued, map_func)
        finally:
            self.__finalize_failure(rescued)

    def __enter_map(self) -> None:
        """Track active map calls so cache pruning happens only after quiescence."""
        with self._lock:
            self._active_calls += 1

    def __exit_map(self) -> None:
        """Drop active-call count and prune cache once no callers remain."""
        with self._lock:
            self._active_calls -= 1
            if self._active_calls == 0:
                self._prune_cache_unlocked(self.cache, self.max_cache_size)

    # ---- public API ------------------------------------------------------
    def map(self, items: list[S], map_func: Callable[[list[S]], list[T]]) -> list[T]:
        """Map ``items`` to values using caching and optional mini-batching.

        This method is thread-safe. It deduplicates inputs while preserving order,
        coordinates concurrent work to prevent duplicate computation, and processes
        owned items in mini-batches determined by ``batch_size``. Before each batch
        call to ``map_func``, the cache is re-checked to avoid redundant requests.

        Args:
            items (list[S]): Input items to map.
            map_func (Callable[[list[S]], list[T]]): Function that maps a batch of
                items to their corresponding results. Must return results in the
                same order as inputs.

        Returns:
            list[T]: Mapped values corresponding to ``items`` in the same order.

        Raises:
            Exception: Propagates any exception raised by ``map_func``.

        Example:
            ```python
            proxy: BatchCache[int, str] = BatchCache(batch_size=2)
            calls: list[list[int]] = []

            def mapper(chunk: list[int]) -> list[str]:
                calls.append(chunk)
                return [f"v:{x}" for x in chunk]

            proxy.map([1, 2, 2, 3], mapper)
            # ['v:1', 'v:2', 'v:2', 'v:3']
            calls  # duplicate ``2`` is only computed once
            # [[1, 2], [3]]
            ```
        """
        self.__enter_map()
        try:
            if self.__all_cached(items):
                return self.__values(items)

            unique_items = self._unique_in_order(items)
            owned, wait_for = self.__acquire_ownership(unique_items)

            try:
                self.__process_owned(owned, map_func)
            finally:
                self.__finalize_failure(owned)
            self.__wait_for(wait_for, map_func)
            return self.__values(items)
        finally:
            self.__exit_map()


@dataclass
class AsyncBatchCache(BatchCacheBase[S, T], Generic[S, T]):
    """Asynchronous version of BatchCache for use with async functions.

    The ``map()`` method accepts an async ``map_func`` that may perform I/O and
    awaits it in mini-batches. It deduplicates inputs, maintains cache
    consistency, and coordinates concurrent coroutines to avoid duplicate work
    via an in-flight registry of asyncio events. Valid ``None`` results are
    cached like any other value.

    When ``batch_size=None``, automatic batch size optimization is enabled,
    dynamically adjusting batch sizes based on execution time to maintain optimal
    performance (targeting 30-60 seconds per batch). When ``max_cache_size`` is
    configured, pruning happens only after the last overlapping ``map()`` call
    exits so in-flight readers cannot lose freshly computed values.

    Example:
        ```python
        import asyncio

        p = AsyncBatchCache[int, str](batch_size=2)

        async def af(xs: list[int]) -> list[str]:
            await asyncio.sleep(0)
            return [f"v:{x}" for x in xs]

        async def run():
            return await p.map([1, 2, 3], af)

        asyncio.run(run())
        # ['v:1', 'v:2', 'v:3']
        ```
    """

    # Number of items to process per call to map_func.
    # - If None (default): Enables automatic batch size optimization, dynamically adjusting
    #   based on execution time (targeting 30-60 seconds per batch)
    # - If positive integer: Fixed batch size
    # - If <= 0: Process all items at once
    batch_size: int | None = None
    max_cache_size: int | None = None
    max_concurrency: int = 8
    show_progress: bool = True
    suggester: BatchSizeSuggester = field(default_factory=BatchSizeSuggester, repr=False)

    cache: CacheBackend[S, T] = field(default_factory=_default_cache_backend, repr=False)

    # internals
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)
    _inflight: dict[S, asyncio.Event] = field(default_factory=dict, repr=False)
    _active_calls: int = field(default=0, init=False, repr=False)
    __sema: asyncio.Semaphore = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize internal semaphore based on ``max_concurrency``.

        An ``asyncio.Semaphore`` is created to limit the number of concurrent
        ``map_func`` calls across overlapping ``map`` invocations.

        Notes:
            This method is invoked automatically by ``dataclasses`` after
            initialization and does not need to be called directly.
        """
        if self.max_concurrency < 1:
            raise ValueError("max_concurrency must be >= 1")
        if self.max_cache_size is not None and self.max_cache_size < 1:
            raise ValueError("max_cache_size must be >= 1")
        self.__sema = asyncio.Semaphore(self.max_concurrency)

    async def __all_cached(self, items: list[S]) -> bool:
        """Check whether all items are present in the cache.

        This method acquires the internal asyncio lock for a consistent view
        of the cache.

        Args:
            items (list[S]): Items to verify against the cache.

        Returns:
            bool: True if every item in ``items`` is already cached, False otherwise.
        """
        async with self._lock:
            cached = self._cached_values_unlocked(self.cache, items)
            return all(x in cached for x in items)

    async def __values(self, items: list[S]) -> list[T]:
        """Get cached values for ``items`` preserving their given order.

        The internal asyncio lock is held while reading the cache to preserve
        consistency under concurrency.

        Args:
            items (list[S]): Items to read from the cache.

        Returns:
            list[T]: Cached values corresponding to ``items`` in the same order.
        """
        async with self._lock:
            cached = self._cached_values_unlocked(self.cache, items)
            values = [cached[x] for x in items]
            self._touch_keys_unlocked(self.cache, items)
            return values

    async def __acquire_ownership(self, items: list[S]) -> tuple[dict[S, asyncio.Event], list[S]]:
        """Acquire ownership for missing keys and identify keys to wait for.

        Args:
            items (list[S]): Unique items (order-preserving) to be processed.

        Returns:
            tuple[dict[S, asyncio.Event], list[S]]: Ownership events for keys this
            coroutine should compute, and keys currently computed elsewhere.
        """
        owned: dict[S, asyncio.Event] = {}
        wait_for: list[S] = []
        async with self._lock:
            cached = self._cached_values_unlocked(self.cache, items)
            for x in items:
                if x in cached:
                    continue
                if x in self._inflight:
                    wait_for.append(x)
                else:
                    self._inflight[x] = asyncio.Event()
                    owned[x] = self._inflight[x]
        return owned, wait_for

    async def __finalize_success(self, owned: dict[S, asyncio.Event], results: list[T]) -> None:
        """Populate cache and signal completion for successfully computed keys.

        Args:
            owned (dict[S, asyncio.Event]): Ownership events for the recent batch.
            results (list[T]): Results corresponding to ``owned`` in order.
        """
        if len(results) != len(owned):
            raise ValueError("map_func must return a list of results with the same length and order as inputs")
        async with self._lock:
            current = [
                (key, result)
                for (key, event), result in zip(owned.items(), results)
                if self._inflight.get(key) is event
            ]
            self._put_values_unlocked(self.cache, current)
            for key, event in owned.items():
                if self._inflight.get(key) is event:
                    del self._inflight[key]
                event.set()

    async def __finalize_failure(self, owned: dict[S, asyncio.Event]) -> None:
        """Release in-flight events on failure to avoid deadlocks.

        Args:
            owned (dict[S, asyncio.Event]): Ownership events whose waiters must
                be released without disturbing newer ownership.
        """
        async with self._lock:
            for key, event in owned.items():
                if self._inflight.get(key) is event:
                    del self._inflight[key]
                event.set()

    async def clear(self) -> None:
        """Clear all cached results and release any in-flight waiters.

        Notes:
            - Intended to be awaited after all processing is finished.
            - Do not call concurrently with active map() calls to avoid
              unnecessary recomputation or racy wake-ups.
        """
        async with self._lock:
            for ev in self._inflight.values():
                ev.set()
            self._inflight.clear()
            self.cache.clear()
            self._active_calls = 0

    async def aclose(self) -> None:
        """Alias for clear()."""
        await self.clear()

    async def __process_owned(
        self, owned: dict[S, asyncio.Event], map_func: Callable[[list[S]], Awaitable[list[T]]]
    ) -> None:
        """Process owned keys using Producer-Consumer pattern with dynamic batch sizing.

        Args:
            owned (dict[S, asyncio.Event]): Keys and their computation ownership events.

        Raises:
            Exception: Propagates any exception raised by ``map_func``.
        """
        if not owned:
            return

        owned_keys = list(owned)
        progress_bar = self._create_progress_bar(len(owned))
        batch_queue: asyncio.Queue[dict[S, asyncio.Event] | None] = asyncio.Queue(maxsize=self.max_concurrency)

        async def producer() -> None:
            index = 0
            while index < len(owned_keys):
                remaining = len(owned_keys) - index
                batch_size = self._normalized_batch_size(remaining)
                batch = {key: owned[key] for key in owned_keys[index : index + batch_size]}
                await batch_queue.put(batch)
                index += batch_size
            for _ in range(self.max_concurrency):
                await batch_queue.put(None)

        async def consumer() -> None:
            while True:
                batch = await batch_queue.get()
                try:
                    if batch is None:
                        break
                    await self.__process_single_batch(batch, map_func, progress_bar)
                finally:
                    batch_queue.task_done()

        tasks = [asyncio.create_task(producer())]
        tasks.extend(asyncio.create_task(consumer()) for _ in range(self.max_concurrency))
        try:
            await asyncio.gather(*tasks)
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            try:
                await asyncio.gather(*tasks, return_exceptions=True)
            finally:
                self._close_progress_bar(progress_bar)

    async def __process_single_batch(
        self, owned: dict[S, asyncio.Event], map_func: Callable[[list[S]], Awaitable[list[T]]], progress_bar
    ) -> None:
        """Process a single batch with semaphore control."""
        to_call = list(owned)
        async with self.__sema:
            results = await self.__map_with_size_recovery(to_call, map_func)
            await self.__finalize_success(owned, results)

        self._update_progress_bar(progress_bar, len(to_call))

    async def __map_with_size_recovery(
        self, items: list[S], map_func: Callable[[list[S]], Awaitable[list[T]]], splits: int = 0
    ) -> list[T]:
        try:
            with self.suggester.record(len(items)):
                results = await map_func(items)
            if len(results) != len(items):
                raise ValueError("map_func must return a list of results with the same length and order as inputs")
            return results
        except Exception as error:
            if not _is_request_size_error(error) or len(items) == 1 or splits >= _MAX_SIZE_SPLITS:
                raise
            if self.batch_size is None:
                self.suggester.reduce_after_size_error(len(items))
            middle = len(items) // 2
            left = await self.__map_with_size_recovery(items[:middle], map_func, splits + 1)
            right = await self.__map_with_size_recovery(items[middle:], map_func, splits + 1)
            return left + right

    async def __wait_for(self, keys: list[S], map_func: Callable[[list[S]], Awaitable[list[T]]]) -> None:
        """Wait for computations owned by other coroutines to complete.

        If a key is neither cached nor in-flight, this method now claims ownership
        for that key immediately (registers an in-flight Event) and defers the
        computation so that all such rescued keys can be processed together in a
        single batched call to ``map_func`` after the scan completes. This avoids
        high-cost single-item calls.

        Args:
            keys (list[S]): Items whose computations are owned by other coroutines.
        """
        rescued: dict[S, asyncio.Event] = {}
        try:
            for key in keys:
                while True:
                    async with self._lock:
                        if key in self._cached_values_unlocked(self.cache, [key]):
                            break
                        waiter = self._inflight.get(key)
                        if waiter is None:
                            self._inflight[key] = asyncio.Event()
                            rescued[key] = self._inflight[key]
                            break
                    await waiter.wait()
            if rescued:
                await self.__process_owned(rescued, map_func)
        finally:
            await self.__finalize_failure(rescued)

    async def __enter_map(self) -> None:
        """Track active map calls so cache pruning happens only after quiescence."""
        async with self._lock:
            self._active_calls += 1

    async def __exit_map(self) -> None:
        """Drop active-call count and prune cache once no callers remain."""
        async with self._lock:
            self._active_calls -= 1
            if self._active_calls == 0:
                self._prune_cache_unlocked(self.cache, self.max_cache_size)

    # ---- public API ------------------------------------------------------
    async def map(self, items: list[S], map_func: Callable[[list[S]], Awaitable[list[T]]]) -> list[T]:
        """Async map with caching, de-duplication, and optional mini-batching.

        Args:
            items (list[S]): Input items to map.
            map_func (Callable[[list[S]], Awaitable[list[T]]]): Async function that
                maps a batch of items to their results, preserving input order.

        Returns:
            list[T]: Mapped values corresponding to ``items`` in the same order.

        Example:
            ```python
            import asyncio

            async def mapper(chunk: list[int]) -> list[str]:
                await asyncio.sleep(0)
                return [f"v:{x}" for x in chunk]

            proxy: AsyncBatchCache[int, str] = AsyncBatchCache(batch_size=2)
            asyncio.run(proxy.map([1, 1, 2], mapper))
            # ['v:1', 'v:1', 'v:2']
            ```
        """
        await self.__enter_map()
        try:
            if await self.__all_cached(items):
                return await self.__values(items)

            unique_items = self._unique_in_order(items)
            owned, wait_for = await self.__acquire_ownership(unique_items)

            try:
                await self.__process_owned(owned, map_func)
            finally:
                await self.__finalize_failure(owned)
            await self.__wait_for(wait_for, map_func)
            return await self.__values(items)
        finally:
            await self.__exit_map()
