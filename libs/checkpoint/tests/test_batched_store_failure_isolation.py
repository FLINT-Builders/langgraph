"""Coalesced async store calls must not share one caller's failure.

AsyncBatchedBaseStore drains every operation queued in the same loop turn into
one abatch. A search filter the store rejects has to fail only that caller.
Other gets, searches, puts, deletes, and namespace listings queued beside it
must finish, and their writes must be applied.

The first abatch after the gate opens waits until every caller has queued.
That makes the shared batch deterministic. Later batches run immediately.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterable
from typing import Any

from langgraph.store.base import Op, Result
from langgraph.store.base.batch import AsyncBatchedBaseStore
from langgraph.store.memory import InMemoryStore

NS = ("mem",)
KNOWN = "known"
DOOMED = "doomed"
WRITTEN = "written"


class _QueuedInMemoryStore(AsyncBatchedBaseStore):
    """Same shape as the server's BatchedStore: one queue over InMemoryStore."""

    def __init__(self) -> None:
        self._hold_next_batch = False
        self._batch_entered = asyncio.Event()
        self._batch_release = asyncio.Event()
        super().__init__()
        self._store = InMemoryStore()

    def batch(self, ops: Iterable[Op]) -> list[Result]:
        return self._store.batch(ops)

    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        if self._hold_next_batch:
            self._hold_next_batch = False
            self._batch_entered.set()
            await self._batch_release.wait()
        return await self._store.abatch(ops)


async def _run_in_one_batch(
    store: _QueuedInMemoryStore,
    calls: list[Callable[[], Awaitable[Any]]],
) -> list[Any]:
    store._batch_entered.clear()
    store._batch_release.clear()
    store._hold_next_batch = True
    warmup = asyncio.create_task(store.aget(("__batch_warmup__",), "missing"))
    try:
        await asyncio.wait_for(store._batch_entered.wait(), timeout=5)
        tasks = [asyncio.create_task(call()) for call in calls]
        deadline = asyncio.get_running_loop().time() + 5
        while True:
            await asyncio.sleep(0)
            still_running = sum(not task.done() for task in tasks)
            if store._aqueue.qsize() >= still_running:
                break
            if asyncio.get_running_loop().time() > deadline:
                raise TimeoutError(
                    f"timed out coalescing store ops; queued={store._aqueue.qsize()} "
                    f"running={still_running}"
                )
        store._batch_release.set()
        return list(
            await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True), timeout=5
            )
        )
    finally:
        if not store._batch_release.is_set():
            store._batch_release.set()
        await warmup


async def test_unsupported_search_filter_does_not_fail_coalesced_callers() -> None:
    store = _QueuedInMemoryStore()
    await store.aput(NS, KNOWN, {"field": "ok"})
    await store.aput(NS, DOOMED, {"field": "gone"})

    async def poison() -> Any:
        return await store.asearch(NS, filter={"field": {"$regex": "x"}})

    async def victim_get() -> Any:
        return await store.aget(NS, KNOWN)

    async def victim_search() -> Any:
        return await store.asearch(NS, filter={"field": "ok"})

    async def victim_put() -> Any:
        return await store.aput(NS, WRITTEN, {"field": "new"})

    async def victim_delete() -> Any:
        return await store.adelete(NS, DOOMED)

    async def victim_list() -> Any:
        return await store.alist_namespaces(prefix=NS)

    poisoned, got, found, put_result, deleted, listed = await _run_in_one_batch(
        store,
        [poison, victim_get, victim_search, victim_put, victim_delete, victim_list],
    )

    assert isinstance(poisoned, ValueError)
    assert "Unsupported operator" in str(poisoned)

    assert not isinstance(got, Exception), got
    assert got is not None
    assert got.key == KNOWN
    assert got.value == {"field": "ok"}

    assert not isinstance(found, Exception), found
    assert [item.key for item in found] == [KNOWN]

    assert put_result is None
    assert deleted is None
    assert not isinstance(listed, Exception), listed
    assert NS in listed

    written = await store.aget(NS, WRITTEN)
    assert written is not None
    assert written.value == {"field": "new"}
    assert await store.aget(NS, DOOMED) is None

    # A later clean read still works after a poisoned batch. The worker must
    # stay alive, and the bad search must not have erased the earlier write.
    again = await store.aget(NS, KNOWN)
    assert again is not None
    assert again.value == {"field": "ok"}
