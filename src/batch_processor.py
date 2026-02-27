"""Batch processing for efficient ML inference."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generic, List, Optional, TypeVar

import numpy as np

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")


@dataclass
class BatchRequest(Generic[T]):
    """A single item queued for batched processing."""

    request_id: str
    payload: T
    priority: int = 0           # higher = processed first within a batch
    submitted_at: float = field(default_factory=time.time)


@dataclass
class BatchResult(Generic[R]):
    """Result for a single batched item."""

    request_id: str
    result: Optional[R]
    latency_ms: float
    error: str = ""


class BatchProcessor(Generic[T, R]):
    """
    Accumulates individual inference requests and processes them as batches
    to maximise GPU/CPU throughput.

    Items are flushed either when the batch reaches *max_batch_size* or
    after *max_wait_ms* milliseconds, whichever comes first.

    Parameters
    ----------
    process_fn : callable
        Sync or async function that accepts a list of payloads and returns
        a list of results in the same order.
    max_batch_size : int
        Maximum items per batch.
    max_wait_ms : float
        Maximum time to wait before flushing an incomplete batch.
    num_workers : int
        Number of concurrent batch processing coroutines.
    """

    def __init__(
        self,
        process_fn: Callable[[List[T]], List[R]],
        max_batch_size: int = 32,
        max_wait_ms: float = 50.0,
        num_workers: int = 1,
    ) -> None:
        self.process_fn = process_fn
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self.num_workers = num_workers

        self._queue: asyncio.Queue = asyncio.Queue()
        self._pending: Dict[str, asyncio.Future] = {}
        self._running = False
        self._worker_tasks: List[asyncio.Task] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start background worker tasks."""
        if self._running:
            return
        self._running = True
        for _ in range(self.num_workers):
            task = asyncio.create_task(self._worker_loop())
            self._worker_tasks.append(task)
        logger.info("BatchProcessor started with %d worker(s).", self.num_workers)

    async def stop(self) -> None:
        """Drain the queue and stop worker tasks."""
        self._running = False
        for task in self._worker_tasks:
            task.cancel()
        await asyncio.gather(*self._worker_tasks, return_exceptions=True)
        self._worker_tasks.clear()
        logger.info("BatchProcessor stopped.")

    async def submit(self, payload: T, priority: int = 0) -> R:
        """
        Submit a single item for batched inference and await the result.

        Parameters
        ----------
        payload : T
            Data to process.
        priority : int
            Higher-priority items are placed earlier within the same batch.

        Returns
        -------
        R
            The corresponding result from ``process_fn``.
        """
        import uuid
        request_id = str(uuid.uuid4())
        loop = asyncio.get_event_loop()
        future: asyncio.Future = loop.create_future()
        req = BatchRequest(request_id=request_id, payload=payload, priority=priority)
        self._pending[request_id] = future
        await self._queue.put(req)
        return await future

    async def submit_many(self, payloads: List[T]) -> List[R]:
        """
        Submit multiple items and collect results in order.

        Parameters
        ----------
        payloads : list
            Items to process.

        Returns
        -------
        list of R
        """
        tasks = [self.submit(p) for p in payloads]
        return list(await asyncio.gather(*tasks))

    def stats(self) -> Dict[str, Any]:
        """Return current queue depth and pending request count."""
        return {
            "queue_depth": self._queue.qsize(),
            "pending_requests": len(self._pending),
            "workers": self.num_workers,
            "running": self._running,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _worker_loop(self) -> None:
        """Continuously drain queue items into batches and process them."""
        while self._running:
            batch: List[BatchRequest] = []
            deadline = time.perf_counter() + self.max_wait_ms / 1000

            # Collect up to max_batch_size items
            while len(batch) < self.max_batch_size:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                try:
                    req = await asyncio.wait_for(self._queue.get(), timeout=remaining)
                    batch.append(req)
                except asyncio.TimeoutError:
                    break

            if not batch:
                await asyncio.sleep(self.max_wait_ms / 2000)
                continue

            # Sort by priority (descending)
            batch.sort(key=lambda r: r.priority, reverse=True)
            await self._process_batch(batch)

    async def _process_batch(self, batch: List[BatchRequest]) -> None:
        payloads = [req.payload for req in batch]
        t0 = time.perf_counter()
        try:
            if asyncio.iscoroutinefunction(self.process_fn):
                results = await self.process_fn(payloads)
            else:
                loop = asyncio.get_event_loop()
                results = await loop.run_in_executor(None, self.process_fn, payloads)
            latency_ms = (time.perf_counter() - t0) * 1000
        except Exception as exc:  # noqa: BLE001
            latency_ms = (time.perf_counter() - t0) * 1000
            logger.error("Batch processing failed: %s", exc)
            for req in batch:
                future = self._pending.pop(req.request_id, None)
                if future and not future.done():
                    future.set_exception(exc)
            return

        for req, result in zip(batch, results):
            future = self._pending.pop(req.request_id, None)
            if future and not future.done():
                future.set_result(result)

        logger.debug("Processed batch of %d in %.1f ms.", len(batch), latency_ms)


def pad_batch(arrays: List[np.ndarray], pad_value: float = 0.0) -> np.ndarray:
    """
    Pad a list of 1-D arrays to the same length and stack them.

    Parameters
    ----------
    arrays : list of np.ndarray
        Variable-length arrays to pad.
    pad_value : float
        Value used for padding.

    Returns
    -------
    np.ndarray
        Shape (n_arrays, max_length).
    """
    max_len = max(len(a) for a in arrays)
    padded = np.full((len(arrays), max_len), pad_value, dtype=float)
    for i, arr in enumerate(arrays):
        padded[i, : len(arr)] = arr
    return padded
