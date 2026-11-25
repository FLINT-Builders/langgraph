"""Coalesced AsyncSqliteStore calls must not share one search's failure.

Search filters are queued without validation. Building a filter with an
unsupported operator raises while the shared batch still contains other
callers' gets, searches, puts, deletes, and namespace listings. That error
must reach only the search that submitted it, and the other writes must commit.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterable
from typing import Any

import aiosqlite
from langgraph.store.base import Op, Result

from langgraph.store.sqlite.aio import AsyncSqliteStore
from langgraph.store.sqlite.base import SqliteIndexConfig, TTLConfig

NS = ("mem",)
KNOWN = "known"
DOOMED = "doomed"
WRITTEN = "written"


class _GatedAsyncSqliteStore(AsyncSqliteStore):
    """AsyncSqliteStore whose next batch waits until the test releases it."""

    def __init__(
        self,
        conn: aiosqlite.Connection,
        *,
        deserializer: Any = None,
        index: SqliteIndexConfig | None = None,
        ttl: TTLConfig | None = None,
    ) -> None:
        self._hold_next_batch = False
        self._batch_entered = asyncio.Event()
        self._batch_release = asyncio.Event()
        super().__init__(conn, deserializer=deserializer, index=index, ttl=ttl)

    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        if self._hold_next_batch:
            self._hold_next_batch = False
            self._batch_entered.set()
            await self._batch_release.wait()
        return await super().abatch(ops)


async def _run_in_one_batch(
    store: _GatedAsyncSqliteStore,
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


async def test_unsupported_search_filter_does_not_fail_coalesced_sqlite_ops() -> None:
    async with _GatedAsyncSqliteStore.from_conn_string(":memory:") as store:
        await store.setup()
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

        (
            poisoned,
            got,
            found,
            put_result,
            deleted,
            listed,
        ) = await _run_in_one_batch(
            store,
            [
                poison,
                victim_get,
                victim_search,
                victim_put,
                victim_delete,
                victim_list,
            ],
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

        again = await store.aget(NS, KNOWN)
        assert again is not None
        assert again.value == {"field": "ok"}


async def test_bad_search_of_empty_prefix_does_not_drop_other_namespace_write() -> None:
    """SQLite rejects the operator while building SQL, before any row is read."""

    async with _GatedAsyncSqliteStore.from_conn_string(":memory:") as store:
        await store.setup()
        await store.aput(NS, KNOWN, {"field": "ok"})

        async def poison() -> Any:
            return await store.asearch(("nobody",), filter={"field": {"$regex": "x"}})

        async def victim_put() -> Any:
            return await store.aput(NS, WRITTEN, {"field": "new"})

        poisoned, put_result = await _run_in_one_batch(store, [poison, victim_put])

        assert isinstance(poisoned, ValueError)
        assert "Unsupported operator" in str(poisoned)
        assert put_result is None
        written = await store.aget(NS, WRITTEN)
        assert written is not None
        assert written.value == {"field": "new"}
        assert await store.aget(NS, KNOWN) is not None
