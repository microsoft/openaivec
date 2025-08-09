import threading
from collections.abc import Hashable
from dataclasses import dataclass, field
from typing import Callable, Dict, Generic, List, Optional, TypeVar

S = TypeVar("S", bound=Hashable)
T = TypeVar("T")


@dataclass
class LocalProxy(Generic[S, T]):
    """Thread-safe local proxy that caches results of a mapping function.

    This proxy batches calls to the provided ``map_func`` (if ``batch_size`` is set),
    deduplicates inputs while preserving order, and ensures that concurrent calls do
    not duplicate work via an in-flight registry. All public behavior is preserved
    while minimizing redundant requests and maintaining input order in the output.
    """

    map_func: Callable[[List[S]], List[T]]
    # Number of items to process per call to map_func. If None or <= 0, process all at once.
    batch_size: Optional[int] = None
    __cache: Dict[S, T] = field(default_factory=dict)
    # Thread-safety primitives (not part of public API)
    __lock: threading.RLock = field(default_factory=threading.RLock, repr=False)
    __inflight: Dict[S, threading.Event] = field(default_factory=dict, repr=False)

    # ---- private helpers -------------------------------------------------
    @staticmethod
    def __unique_in_order(seq: List[S]) -> List[S]:
        """Return unique items preserving their first-occurrence order.

        Args:
            seq: Sequence of items which may contain duplicates.

        Returns:
            A new list containing each distinct item from ``seq`` exactly once,
            in the order of their first occurrence.
        """
        seen: set[S] = set()
        out: List[S] = []
        for x in seq:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    def __normalized_batch_size(self, total: int) -> int:
        """Compute the effective batch size used for processing.

        If ``batch_size`` is not set or non-positive, the entire ``total`` is
        processed in a single call.

        Args:
            total: Number of items intended to be processed.

        Returns:
            The positive batch size to use.
        """
        return self.batch_size if (self.batch_size and self.batch_size > 0) else total

    def __all_cached(self, items: List[S]) -> bool:
        """Check whether all items are present in the cache.

        This method acquires the internal lock to perform a consistent check.

        Args:
            items: Items to verify against the cache.

        Returns:
            True if every item is already cached, False otherwise.
        """
        with self.__lock:
            return all(x in self.__cache for x in items)

    def __values(self, items: List[S]) -> List[T]:
        """Fetch cached values for ``items`` preserving the given order.

        This method acquires the internal lock while reading the cache.

        Args:
            items: Items to retrieve from the cache.

        Returns:
            The cached values corresponding to ``items`` in the same order.
        """
        with self.__lock:
            return [self.__cache[x] for x in items]

    def __acquire_ownership(self, items: List[S]) -> tuple[List[S], List[S]]:
        """Acquire ownership for missing items and identify keys to wait for.

        For each unique item, if it's already cached, it is ignored. If it's
        currently being computed by another thread (in-flight), it is added to
        the wait list. Otherwise, this method marks the key as in-flight and
        considers it "owned" by the current thread.

        Args:
            items: Unique items (order-preserving) to be processed.

        Returns:
            A tuple of two lists: ``(owned, wait_for)`` where
            - ``owned``: Items this thread is responsible for computing.
            - ``wait_for``: Items that another thread is already computing.
        """
        owned: List[S] = []
        wait_for: List[S] = []
        with self.__lock:
            for x in items:
                if x in self.__cache:
                    continue
                if x in self.__inflight:
                    wait_for.append(x)
                else:
                    self.__inflight[x] = threading.Event()
                    owned.append(x)
        return owned, wait_for

    def __finalize_success(self, to_call: List[S], results: List[T]) -> None:
        """Populate cache with results and signal completion events.

        Args:
            to_call: Items that were computed.
            results: Results corresponding to ``to_call`` in order.
        """
        with self.__lock:
            for x, y in zip(to_call, results):
                self.__cache[x] = y
                ev = self.__inflight.pop(x, None)
                if ev:
                    ev.set()

    def __finalize_failure(self, to_call: List[S]) -> None:
        """Release in-flight events on failure to avoid deadlocks.

        Args:
            to_call: Items that were intended to be computed when an error occurred.
        """
        with self.__lock:
            for x in to_call:
                ev = self.__inflight.pop(x, None)
                if ev:
                    ev.set()

    def __process_owned(self, owned: List[S]) -> None:
        """Process owned items in mini-batches and fill the cache.

        Before calling ``map_func`` for each batch, the cache is re-checked
        to skip any items that may have been filled in the meantime. On
        exceptions raised by ``map_func``, all corresponding in-flight events
        are released to prevent deadlocks, and the exception is propagated.

        Args:
            owned: Items for which the current thread has computation ownership.

        Raises:
            Exception: Propagates any exception raised by ``map_func``.
        """
        if not owned:
            return
        batch_size = self.__normalized_batch_size(len(owned))
        for i in range(0, len(owned), batch_size):
            batch = owned[i : i + batch_size]
            # Double-check cache right before calling map_func
            with self.__lock:
                to_call = [x for x in batch if x not in self.__cache]
            if not to_call:
                continue
            try:
                results = self.map_func(to_call)
            except Exception:
                self.__finalize_failure(to_call)
                raise
            self.__finalize_success(to_call, results)

    def __try_compute_single(self, x: S) -> None:
        """Compute a single missing key when no one else is in-flight.

        This is used as a race-recovery path when a key is neither cached nor
        registered as in-flight, ensuring progress without busy-waiting.

        Args:
            x: The item to compute.

        Raises:
            Exception: Propagates any exception raised by ``map_func``.
        """
        with self.__lock:
            if x in self.__cache:
                return
            if x not in self.__inflight:
                self.__inflight[x] = threading.Event()
        try:
            result = self.map_func([x])[0]
        except Exception:
            with self.__lock:
                ev = self.__inflight.pop(x, None)
                if ev:
                    ev.set()
            raise
        with self.__lock:
            self.__cache[x] = result
            ev = self.__inflight.pop(x, None)
            if ev:
                ev.set()

    def __wait_for(self, keys: List[S]) -> None:
        """Wait for other threads to complete computations for the given keys.

        If a key is neither cached nor in-flight, this method attempts to compute
        it inline via ``__try_compute_single`` to avoid indefinite waiting.

        Args:
            keys: Items whose computations are owned by other threads.
        """
        for x in keys:
            while True:
                with self.__lock:
                    if x in self.__cache:
                        break
                    ev = self.__inflight.get(x)
                if ev is not None:
                    ev.wait()
                else:
                    # No inflight and not cached; try compute here.
                    self.__try_compute_single(x)
                    break

    # ---- public API ------------------------------------------------------
    def map(self, items: List[S]) -> List[T]:
        """Map ``items`` to values using caching and optional mini-batching.

        This method is thread-safe. It deduplicates inputs while preserving order,
        coordinates concurrent work to prevent duplicate computation, and processes
        owned items in mini-batches determined by ``batch_size``. Before each batch
        call to ``map_func``, the cache is re-checked to avoid redundant requests.

        Args:
            items: Input items to map.

        Returns:
            A list of mapped values corresponding to ``items`` in the same order.

        Raises:
            Exception: Propagates any exception raised by ``map_func``.
        """
        if self.__all_cached(items):
            return self.__values(items)

        unique_items = self.__unique_in_order(items)
        owned, wait_for = self.__acquire_ownership(unique_items)

        self.__process_owned(owned)
        self.__wait_for(wait_for)

        return self.__values(items)
