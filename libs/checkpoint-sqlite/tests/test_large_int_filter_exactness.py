"""Exact integer comparisons for SQLite store search filters.

These tests exercise `SqliteStore.search` and `AsyncSqliteStore.asearch`,
including vector search, which builds the same numeric SQL. They fail while
integer filters are coerced with `float` or `CAST(... AS REAL)`, because
integers above 2**53 that are still distinct in Python can collapse to one
IEEE-754 double. A passing run returns only the stored item whose JSON
integer is the one named by the filter.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager

from langgraph.store.base import SearchItem

from langgraph.store.sqlite import AsyncSqliteStore, SqliteStore
from langgraph.store.sqlite.base import SqliteIndexConfig
from tests.test_store import CharacterEmbeddings

# 2**53 is the last integer that every smaller integer can also represent.
# The next integer is a different Python int and a different SQLite integer,
# but it rounds to the same float.
EXACT = 9007199254740992
NEIGHBOR = 9007199254740993
NEG_EXACT = -9007199254740992
NEG_NEIGHBOR = -9007199254740993
INT64_MAX = 9223372036854775807
INT64_PREV = 9223372036854775806


def _keys(results: list[SearchItem]) -> set[str]:
    return {item.key for item in results}


def _assert_only(
    results: list[SearchItem], key: str, field: str, expected: int
) -> None:
    assert _keys(results) == {key}
    assert len(results) == 1
    found = results[0].value[field]
    assert found == expected
    assert type(found) is int


def _assert_only_value(results: list[SearchItem], key: str, expected: dict) -> None:
    assert _keys(results) == {key}
    assert len(results) == 1
    assert results[0].value == expected


@contextmanager
def _sync_store() -> Iterator[SqliteStore]:
    with SqliteStore.from_conn_string(":memory:") as store:
        store.setup()
        yield store


@asynccontextmanager
async def _async_store() -> AsyncIterator[AsyncSqliteStore]:
    async with AsyncSqliteStore.from_conn_string(":memory:") as store:
        await store.setup()
        yield store


def _put_pair(
    store: SqliteStore, namespace: tuple[str, ...], field: str, left: int, right: int
) -> None:
    store.put(namespace, "left", {field: left, "text": "shared memory"})
    store.put(namespace, "right", {field: right, "text": "shared memory"})


async def _aput_pair(
    store: AsyncSqliteStore,
    namespace: tuple[str, ...],
    field: str,
    left: int,
    right: int,
) -> None:
    await store.aput(namespace, "left", {field: left, "text": "shared memory"})
    await store.aput(namespace, "right", {field: right, "text": "shared memory"})


def _assert_equality_filters(
    store: SqliteStore, namespace: tuple[str, ...], field: str, left: int, right: int
) -> None:
    _assert_only(store.search(namespace, filter={field: left}), "left", field, left)
    _assert_only(store.search(namespace, filter={field: right}), "right", field, right)
    _assert_only(
        store.search(namespace, filter={field: {"$eq": left}}),
        "left",
        field,
        left,
    )
    _assert_only(
        store.search(namespace, filter={field: {"$eq": right}}),
        "right",
        field,
        right,
    )


async def _assert_async_equality_filters(
    store: AsyncSqliteStore,
    namespace: tuple[str, ...],
    field: str,
    left: int,
    right: int,
) -> None:
    _assert_only(
        await store.asearch(namespace, filter={field: left}), "left", field, left
    )
    _assert_only(
        await store.asearch(namespace, filter={field: right}),
        "right",
        field,
        right,
    )
    _assert_only(
        await store.asearch(namespace, filter={field: {"$eq": left}}),
        "left",
        field,
        left,
    )
    _assert_only(
        await store.asearch(namespace, filter={field: {"$eq": right}}),
        "right",
        field,
        right,
    )


def _assert_other_operators(
    store: SqliteStore, namespace: tuple[str, ...], field: str, low: int, high: int
) -> None:
    """`low` and `high` are the numeric values, independent of row names."""
    by_value = {
        item.value[field]: item.key
        for item in store.search(namespace, filter=None, limit=10)
    }
    assert set(by_value) == {low, high}

    _assert_only(
        store.search(namespace, filter={field: {"$ne": low}}),
        by_value[high],
        field,
        high,
    )
    _assert_only(
        store.search(namespace, filter={field: {"$ne": high}}),
        by_value[low],
        field,
        low,
    )
    _assert_only(
        store.search(namespace, filter={field: {"$gt": low}}),
        by_value[high],
        field,
        high,
    )
    assert _keys(store.search(namespace, filter={field: {"$gt": high}})) == set()
    _assert_only(
        store.search(namespace, filter={field: {"$gte": high}}),
        by_value[high],
        field,
        high,
    )
    assert _keys(store.search(namespace, filter={field: {"$gte": low}})) == {
        "left",
        "right",
    }
    _assert_only(
        store.search(namespace, filter={field: {"$lt": high}}),
        by_value[low],
        field,
        low,
    )
    assert _keys(store.search(namespace, filter={field: {"$lt": low}})) == set()
    _assert_only(
        store.search(namespace, filter={field: {"$lte": low}}),
        by_value[low],
        field,
        low,
    )
    assert _keys(store.search(namespace, filter={field: {"$lte": high}})) == {
        "left",
        "right",
    }


async def _assert_async_other_operators(
    store: AsyncSqliteStore,
    namespace: tuple[str, ...],
    field: str,
    low: int,
    high: int,
) -> None:
    by_value = {
        item.value[field]: item.key
        for item in await store.asearch(namespace, filter=None, limit=10)
    }
    assert set(by_value) == {low, high}

    _assert_only(
        await store.asearch(namespace, filter={field: {"$ne": low}}),
        by_value[high],
        field,
        high,
    )
    _assert_only(
        await store.asearch(namespace, filter={field: {"$ne": high}}),
        by_value[low],
        field,
        low,
    )
    _assert_only(
        await store.asearch(namespace, filter={field: {"$gt": low}}),
        by_value[high],
        field,
        high,
    )
    assert _keys(await store.asearch(namespace, filter={field: {"$gt": high}})) == set()
    _assert_only(
        await store.asearch(namespace, filter={field: {"$gte": high}}),
        by_value[high],
        field,
        high,
    )
    assert _keys(await store.asearch(namespace, filter={field: {"$gte": low}})) == {
        "left",
        "right",
    }
    _assert_only(
        await store.asearch(namespace, filter={field: {"$lt": high}}),
        by_value[low],
        field,
        low,
    )
    assert _keys(await store.asearch(namespace, filter={field: {"$lt": low}})) == set()
    _assert_only(
        await store.asearch(namespace, filter={field: {"$lte": low}}),
        by_value[low],
        field,
        low,
    )
    assert _keys(await store.asearch(namespace, filter={field: {"$lte": high}})) == {
        "left",
        "right",
    }


def test_sync_search_distinguishes_integers_above_2_to_the_53() -> None:
    with _sync_store() as store:
        namespace = ("memories",)
        _put_pair(store, namespace, "owner", EXACT, NEIGHBOR)
        # Integers that are exactly representable still match one row.
        store.put(("small",), "one", {"owner": 1})
        store.put(("small",), "two", {"owner": 2})
        _assert_only(store.search(("small",), filter={"owner": 1}), "one", "owner", 1)
        _assert_only(store.search(("small",), filter={"owner": 2}), "two", "owner", 2)

        _assert_equality_filters(store, namespace, "owner", EXACT, NEIGHBOR)
        _assert_other_operators(store, namespace, "owner", EXACT, NEIGHBOR)


def test_sync_search_distinguishes_negative_neighbors() -> None:
    with _sync_store() as store:
        namespace = ("memories",)
        _put_pair(store, namespace, "owner", NEG_EXACT, NEG_NEIGHBOR)
        _assert_equality_filters(store, namespace, "owner", NEG_EXACT, NEG_NEIGHBOR)
        _assert_other_operators(store, namespace, "owner", NEG_NEIGHBOR, NEG_EXACT)


def test_sync_search_distinguishes_int64_boundary() -> None:
    with _sync_store() as store:
        namespace = ("memories",)
        _put_pair(store, namespace, "owner", INT64_PREV, INT64_MAX)
        _assert_equality_filters(store, namespace, "owner", INT64_PREV, INT64_MAX)
        _assert_other_operators(store, namespace, "owner", INT64_PREV, INT64_MAX)


def test_sync_search_distinguishes_nested_integer_field() -> None:
    with _sync_store() as store:
        namespace = ("memories",)
        store.put(namespace, "left", {"meta": {"owner": EXACT}})
        store.put(namespace, "right", {"meta": {"owner": NEIGHBOR}})
        _assert_only_value(
            store.search(namespace, filter={"meta.owner": NEIGHBOR}),
            "right",
            {"meta": {"owner": NEIGHBOR}},
        )
        _assert_only_value(
            store.search(namespace, filter={"meta.owner": {"$eq": EXACT}}),
            "left",
            {"meta": {"owner": EXACT}},
        )
        _assert_only_value(
            store.search(namespace, filter={"meta.owner": {"$gt": EXACT}}),
            "right",
            {"meta": {"owner": NEIGHBOR}},
        )


def test_sync_vector_search_uses_the_same_integer_comparison() -> None:
    embeddings = CharacterEmbeddings(dims=50)
    index: SqliteIndexConfig = {
        "dims": embeddings.dims,
        "embed": embeddings,
        "text_fields": ["text"],
    }
    with SqliteStore.from_conn_string(":memory:", index=index) as store:
        store.setup()
        namespace = ("memories",)
        _put_pair(store, namespace, "owner", EXACT, NEIGHBOR)
        unfiltered = store.search(namespace, query="shared memory", limit=10)
        assert _keys(unfiltered) == {"left", "right"}
        _assert_only(
            store.search(
                namespace,
                query="shared memory",
                filter={"owner": NEIGHBOR},
                limit=10,
            ),
            "right",
            "owner",
            NEIGHBOR,
        )
        _assert_only(
            store.search(
                namespace,
                query="shared memory",
                filter={"owner": {"$lte": EXACT}},
                limit=10,
            ),
            "left",
            "owner",
            EXACT,
        )


async def test_async_search_distinguishes_integers_above_2_to_the_53() -> None:
    async with _async_store() as store:
        namespace = ("memories",)
        await _aput_pair(store, namespace, "owner", EXACT, NEIGHBOR)
        await store.aput(("small",), "one", {"owner": 1})
        await store.aput(("small",), "two", {"owner": 2})
        _assert_only(
            await store.asearch(("small",), filter={"owner": 1}),
            "one",
            "owner",
            1,
        )
        _assert_only(
            await store.asearch(("small",), filter={"owner": 2}),
            "two",
            "owner",
            2,
        )
        await _assert_async_equality_filters(store, namespace, "owner", EXACT, NEIGHBOR)
        await _assert_async_other_operators(store, namespace, "owner", EXACT, NEIGHBOR)


async def test_async_search_distinguishes_negative_neighbors() -> None:
    async with _async_store() as store:
        namespace = ("memories",)
        await _aput_pair(store, namespace, "owner", NEG_EXACT, NEG_NEIGHBOR)
        await _assert_async_equality_filters(
            store, namespace, "owner", NEG_EXACT, NEG_NEIGHBOR
        )
        await _assert_async_other_operators(
            store, namespace, "owner", NEG_NEIGHBOR, NEG_EXACT
        )


async def test_async_search_distinguishes_int64_boundary() -> None:
    async with _async_store() as store:
        namespace = ("memories",)
        await _aput_pair(store, namespace, "owner", INT64_PREV, INT64_MAX)
        await _assert_async_equality_filters(
            store, namespace, "owner", INT64_PREV, INT64_MAX
        )
        await _assert_async_other_operators(
            store, namespace, "owner", INT64_PREV, INT64_MAX
        )


async def test_async_search_distinguishes_nested_integer_field() -> None:
    async with _async_store() as store:
        namespace = ("memories",)
        await store.aput(namespace, "left", {"meta": {"owner": EXACT}})
        await store.aput(namespace, "right", {"meta": {"owner": NEIGHBOR}})
        _assert_only_value(
            await store.asearch(namespace, filter={"meta.owner": NEIGHBOR}),
            "right",
            {"meta": {"owner": NEIGHBOR}},
        )
        _assert_only_value(
            await store.asearch(namespace, filter={"meta.owner": {"$eq": EXACT}}),
            "left",
            {"meta": {"owner": EXACT}},
        )
        _assert_only_value(
            await store.asearch(namespace, filter={"meta.owner": {"$gt": EXACT}}),
            "right",
            {"meta": {"owner": NEIGHBOR}},
        )


async def test_async_vector_search_uses_the_same_integer_comparison() -> None:
    embeddings = CharacterEmbeddings(dims=50)
    index: SqliteIndexConfig = {
        "dims": embeddings.dims,
        "embed": embeddings,
        "text_fields": ["text"],
    }
    async with AsyncSqliteStore.from_conn_string(":memory:", index=index) as store:
        await store.setup()
        namespace = ("memories",)
        await _aput_pair(store, namespace, "owner", EXACT, NEIGHBOR)
        unfiltered = await store.asearch(namespace, query="shared memory", limit=10)
        assert _keys(unfiltered) == {"left", "right"}
        _assert_only(
            await store.asearch(
                namespace,
                query="shared memory",
                filter={"owner": NEIGHBOR},
                limit=10,
            ),
            "right",
            "owner",
            NEIGHBOR,
        )
        _assert_only(
            await store.asearch(
                namespace,
                query="shared memory",
                filter={"owner": {"$lte": EXACT}},
                limit=10,
            ),
            "left",
            "owner",
            EXACT,
        )
