import asyncio
import threading
from collections.abc import Awaitable, Callable, Hashable
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

from openaivec._optimize import BatchSizeSuggester

__all__ = []

S = TypeVar("S", bound=Hashable)
T = TypeVar("T")


class ProxyBase(Generic[S, T]):
    """Common utilities shared by BatchingMapProxy and AsyncBatchingMapProxy.

    Provides order-preserving deduplication and batch size normalization that
    depend only on ``batch_size`` and do not touch concurrency primitives.

    Attributes:
        batch_size: Optional mini-batch size hint used by implementations to
            split work into chunks. When unset or non-positive, implementations
            should process the entire input in a single call.
    """

    batch_size: int | None  # subclasses may override via dataclass
    show_progress: bool  # Enable progress bar display
    suggester: BatchSizeSuggester  # Batch size optimization, initialized by subclasses

    def _is_notebook_environment(self) -> bool:
        """Check if running in a Jupyter notebook environment.

        Returns:
            bool: True if running in a notebook, False otherwise.
        """
        try:
            from IPython.core.getipython import get_ipython

            ipython = get_ipython()
            if ipython is not None:
                # Check for different notebook environments
                class_name = ipython.__class__.__name__
                module_name = ipython.__class__.__module__

                # Standard Jupyter notebook/lab
                if class_name == "ZMQInteractiveShell":
                    return True

                # JupyterLab and newer environments
                if "zmq" in module_name.lower() or "jupyter" in module_name.lower():
                    return True

                # Google Colab
                if "google.colab" in module_name:
                    return True

                # VS Code notebooks and others
                if hasattr(ipython, "kernel"):
                    return True

        except ImportError:
            pass

        # Check for other notebook indicators
        # Check for common notebook environment variables
        import os
        import sys

        notebook_vars = [
            "JUPYTER_CONFIG_DIR",
            "JUPYTERLAB_DIR",
            "COLAB_GPU",
            "VSCODE_PID",  # VS Code
        ]

        for var in notebook_vars:
            if var in os.environ:
                return True

        # Check if running in IPython without terminal
        if "IPython" in sys.modules:
            try:
                # If we can import display from IPython, likely in notebook
                import importlib.util

                if importlib.util.find_spec("IPython.display") is not None:
                    return True
            except ImportError:
                pass

        return False

    def _create_progress_bar(self, total: int, desc: str = "Processing batches") -> Any:
        """Create a progress bar if conditions are met.

        Args:
            total (int): Total number of items to process.
            desc (str): Description for the progress bar.

        Returns:
            Any: Progress bar instance or None if not available.
        """
        try:
            from tqdm.auto import tqdm as tqdm_progress

            if self.show_progress and self._is_notebook_environment():
                return tqdm_progress(total=total, desc=desc, unit="item")
        except ImportError:
            pass
        return None

    def _update_progress_bar(self, progress_bar: Any, increment: int) -> None:
        """Update progress bar with the given increment.

        Args:
            progress_bar (Any): Progress bar instance.
            increment (int): Number of items to increment.
        """
        if progress_bar:
            progress_bar.update(increment)

    def _close_progress_bar(self, progress_bar: Any) -> None:
        """Close the progress bar.

        Args:
            progress_bar (Any): Progress bar instance.
        """
        if progress_bar:
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
class BatchingMapProxy(ProxyBase[S, T], Generic[S, T]):
    """Thread-safe local proxy that caches results of a mapping function.

    This proxy batches calls to the ``map_func`` you pass to ``map()`` (if
    ``batch_size`` is set),
    deduplicates inputs while preserving order, and ensures that concurrent calls do
    not duplicate work via an in-flight registry. All public behavior is preserved
    while minimizing redundant requests and maintaining input order in the output.

    When ``batch_size=None``, automatic batch size optimization is enabled,
    dynamically adjusting batch sizes based on execution time to maintain optimal
    performance (targeting 30-60 seconds per batch).

    Example:
        >>> p = BatchingMapProxy[int, str](batch_size=3)
        >>> def f(xs: list[int]) -> list[str]:
        ...     return [f"v:{x}" for x in xs]
        >>> p.map([1, 2, 2, 3, 4], f)
        ['v:1', 'v:2', 'v:2', 'v:3', 'v:4']
    """

    # Number of items to process per call to map_func.
    # - If None (default): Enables automatic batch size optimization, dynamically adjusting
    #   based on execution time (targeting 30-60 seconds per batch)
    # - If positive integer: Fixed batch size
    # - If <= 0: Process all items at once
    batch_size: int | None = None
    show_progress: bool = False
    suggester: BatchSizeSuggester = field(default_factory=BatchSizeSuggester, repr=False)

    # internals
    _cache: dict[S, T] = field(default_factory=dict)
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)
    _inflight: dict[S, threading.Event] = field(default_factory=dict, repr=False)

    def __all_cached(self, items: list[S]) -> bool:
        """Check whether all items are present in the cache.

        This method acquires the internal lock to perform a consistent check.

        Args:
            items (list[S]): Items to verify against the cache.

        Returns:
            bool: True if every item is already cached, False otherwise.
        """
        with self._lock:
            return all(x in self._cache for x in items)

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
            return [self._cache[x] for x in items]

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
            for x in items:
                if x in self._cache:
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
            for x, y in zip(to_call, results):
                self._cache[x] = y
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
            self._cache.clear()

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

        i = 0
        while i < len(owned):
            # Get dynamic batch size for each iteration
            current_batch_size = self._normalized_batch_size(len(owned))
            batch = owned[i : i + current_batch_size]
            # Double-check cache right before processing
            with self._lock:
                uncached_in_batch = [x for x in batch if x not in self._cache]

            pending_to_call.extend(uncached_in_batch)

            # Process accumulated items when we reach batch_size or at the end
            is_last_batch = i + current_batch_size >= len(owned)
            if len(pending_to_call) >= current_batch_size or (is_last_batch and pending_to_call):
                # Take up to batch_size items to process
                to_call = pending_to_call[:current_batch_size]
                pending_to_call = pending_to_call[current_batch_size:]

                try:
                    # Always measure execution time using suggester
                    with self.suggester.record(len(to_call)):
                        results = map_func(to_call)
                except Exception:
                    self.__finalize_failure(to_call)
                    raise
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

            try:
                with self.suggester.record(len(to_call)):
                    results = map_func(to_call)
            except Exception:
                self.__finalize_failure(to_call)
                raise
            self.__finalize_success(to_call, results)

            # Update progress bar
            self._update_progress_bar(progress_bar, len(to_call))

        # Close progress bar
        self._close_progress_bar(progress_bar)

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
        for x in keys:
            while True:
                with self._lock:
                    if x in self._cache:
                        break
                    ev = self._inflight.get(x)
                    if ev is None:
                        # Not cached and no one computing; claim ownership to batch later.
                        self._inflight[x] = threading.Event()
                        rescued.append(x)
                        break
                # Someone else is computing; wait for completion.
                ev.wait()
        # Batch-process rescued keys, if any
        if rescued:
            try:
                self.__process_owned(rescued, map_func)
            except Exception:
                # Ensure events are released on failure to avoid deadlock
                self.__finalize_failure(rescued)
                raise

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
        """
        if self.__all_cached(items):
            return self.__values(items)

        unique_items = self._unique_in_order(items)
        owned, wait_for = self.__acquire_ownership(unique_items)

        self.__process_owned(owned, map_func)
        self.__wait_for(wait_for, map_func)

        # Fetch results before purging None entries
        results = self.__values(items)

        # Remove None values from cache so they are recomputed on future calls
        with self._lock:
            if self._cache:  # micro-optimization
                for k in set(items):
                    try:
                        if self._cache.get(k, object()) is None:
                            del self._cache[k]
                    except KeyError:
                        pass

        return results


@dataclass
class AsyncBatchingMapProxy(ProxyBase[S, T], Generic[S, T]):
    """Asynchronous version of BatchingMapProxy for use with async functions.

    The ``map()`` method accepts an async ``map_func`` that may perform I/O and
    awaits it
    in mini-batches. It deduplicates inputs, maintains cache consistency, and
    coordinates concurrent coroutines to avoid duplicate work via an in-flight
    registry of asyncio events.

    When ``batch_size=None``, automatic batch size optimization is enabled,
    dynamically adjusting batch sizes based on execution time to maintain optimal
    performance (targeting 30-60 seconds per batch).

    Example:
        >>> import asyncio
        >>> from typing import List
        >>> p = AsyncBatchingMapProxy[int, str](batch_size=2)
        >>> async def af(xs: list[int]) -> list[str]:
        ...     await asyncio.sleep(0)
        ...     return [f"v:{x}" for x in xs]
        >>> async def run():
        ...     return await p.map([1, 2, 3], af)
        >>> asyncio.run(run())
        ['v:1', 'v:2', 'v:3']
    """

    # Number of items to process per call to map_func.
    # - If None (default): Enables automatic batch size optimization, dynamically adjusting
    #   based on execution time (targeting 30-60 seconds per batch)
    # - If positive integer: Fixed batch size
    # - If <= 0: Process all items at once
    batch_size: int | None = None
    max_concurrency: int = 8
    show_progress: bool = False
    suggester: BatchSizeSuggester = field(default_factory=BatchSizeSuggester, repr=False)

    # internals
    _cache: dict[S, T] = field(default_factory=dict, repr=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)
    _inflight: dict[S, asyncio.Event] = field(default_factory=dict, repr=False)
    __sema: asyncio.Semaphore | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize internal semaphore based on ``max_concurrency``.

        If ``max_concurrency`` is a positive integer, an ``asyncio.Semaphore``
        is created to limit the number of concurrent ``map_func`` calls across
        overlapping ``map`` invocations. When non-positive or ``None``, no
        semaphore is used and concurrency is unrestricted by this proxy.

        Notes:
            This method is invoked automatically by ``dataclasses`` after
            initialization and does not need to be called directly.
        """
        # Initialize semaphore if limiting is requested; non-positive disables limiting
        if self.max_concurrency and self.max_concurrency > 0:
            self.__sema = asyncio.Semaphore(self.max_concurrency)
        else:
            self.__sema = None

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
            return all(x in self._cache for x in items)

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
            return [self._cache[x] for x in items]

    async def __acquire_ownership(self, items: list[S]) -> tuple[list[S], list[S]]:
        """Acquire ownership for missing keys and identify keys to wait for.

        Args:
            items (list[S]): Unique items (order-preserving) to be processed.

        Returns:
            tuple[list[S], list[S]]: A tuple ``(owned, wait_for)`` where owned are
            keys this coroutine should compute, and wait_for are keys currently
            being computed elsewhere.
        """
        owned: list[S] = []
        wait_for: list[S] = []
        async with self._lock:
            for x in items:
                if x in self._cache:
                    continue
                if x in self._inflight:
                    wait_for.append(x)
                else:
                    self._inflight[x] = asyncio.Event()
                    owned.append(x)
        return owned, wait_for

    async def __finalize_success(self, to_call: list[S], results: list[T]) -> None:
        """Populate cache and signal completion for successfully computed keys.

        Args:
            to_call (list[S]): Items that were computed in the recent batch.
            results (list[T]): Results corresponding to ``to_call`` in order.
        """
        if len(results) != len(to_call):
            # Prevent deadlocks if map_func violates the contract.
            await self.__finalize_failure(to_call)
            raise ValueError("map_func must return a list of results with the same length and order as inputs")
        async with self._lock:
            for x, y in zip(to_call, results):
                self._cache[x] = y
                ev = self._inflight.pop(x, None)
                if ev:
                    ev.set()

    async def __finalize_failure(self, to_call: list[S]) -> None:
        """Release in-flight events on failure to avoid deadlocks.

        Args:
            to_call (list[S]): Items whose computation failed; their waiters will
            be released.
        """
        async with self._lock:
            for x in to_call:
                ev = self._inflight.pop(x, None)
                if ev:
                    ev.set()

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
            self._cache.clear()

    async def aclose(self) -> None:
        """Alias for clear()."""
        await self.clear()

    async def __process_owned(self, owned: list[S], map_func: Callable[[list[S]], Awaitable[list[T]]]) -> None:
        """Process owned keys using Producer-Consumer pattern with dynamic batch sizing.

        Args:
            owned (list[S]): Items for which this coroutine holds computation ownership.

        Raises:
            Exception: Propagates any exception raised by ``map_func``.
        """
        if not owned:
            return

        progress_bar = self._create_progress_bar(len(owned))
        batch_queue: asyncio.Queue = asyncio.Queue(maxsize=self.max_concurrency)

        async def producer():
            index = 0
            while index < len(owned):
                batch_size = self._normalized_batch_size(len(owned) - index)
                batch = owned[index : index + batch_size]
                await batch_queue.put(batch)
                index += batch_size
            # Send completion signals
            for _ in range(self.max_concurrency):
                await batch_queue.put(None)

        async def consumer():
            while True:
                batch = await batch_queue.get()
                try:
                    if batch is None:
                        break
                    await self.__process_single_batch(batch, map_func, progress_bar)
                finally:
                    batch_queue.task_done()

        await asyncio.gather(producer(), *[consumer() for _ in range(self.max_concurrency)])

        self._close_progress_bar(progress_bar)

    async def __process_single_batch(
        self, to_call: list[S], map_func: Callable[[list[S]], Awaitable[list[T]]], progress_bar
    ) -> None:
        """Process a single batch with semaphore control."""
        acquired = False
        try:
            if self.__sema:
                await self.__sema.acquire()
                acquired = True
            # Measure async map_func execution using suggester
            with self.suggester.record(len(to_call)):
                results = await map_func(to_call)
        except Exception:
            await self.__finalize_failure(to_call)
            raise
        finally:
            if self.__sema and acquired:
                self.__sema.release()
        await self.__finalize_success(to_call, results)

        # Update progress bar
        self._update_progress_bar(progress_bar, len(to_call))

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
        rescued: list[S] = []  # keys we claim to batch-process
        for x in keys:
            while True:
                async with self._lock:
                    if x in self._cache:
                        break
                    ev = self._inflight.get(x)
                    if ev is None:
                        # Not cached and no one computing; claim ownership to batch later.
                        self._inflight[x] = asyncio.Event()
                        rescued.append(x)
                        break
                # Someone else is computing; wait for completion.
                await ev.wait()
        # Batch-process rescued keys, if any
        if rescued:
            try:
                await self.__process_owned(rescued, map_func)
            except Exception:
                await self.__finalize_failure(rescued)
                raise

    # ---- public API ------------------------------------------------------
    async def map(self, items: list[S], map_func: Callable[[list[S]], Awaitable[list[T]]]) -> list[T]:
        """Async map with caching, de-duplication, and optional mini-batching.

        Args:
            items (list[S]): Input items to map.
            map_func (Callable[[list[S]], Awaitable[list[T]]]): Async function that
                maps a batch of items to their results, preserving input order.

        Returns:
            list[T]: Mapped values corresponding to ``items`` in the same order.
        """
        if await self.__all_cached(items):
            return await self.__values(items)

        unique_items = self._unique_in_order(items)
        owned, wait_for = await self.__acquire_ownership(unique_items)

        await self.__process_owned(owned, map_func)
        await self.__wait_for(wait_for, map_func)

        results = await self.__values(items)

        # Remove None values from cache after retrieval to avoid persisting incomplete results
        async with self._lock:
            if self._cache:
                for k in set(items):
                    if self._cache.get(k, object()) is None:
                        self._cache.pop(k, None)

        return results
