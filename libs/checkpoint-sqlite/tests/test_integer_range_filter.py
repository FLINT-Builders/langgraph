"""Integer range filters on the SQLite stores must use exact integer ordering.

SqliteStore and AsyncSqliteStore cast range comparisons through SQLite REAL,
which collapses distinct integers above 2**53. These tests call put/search
and expect the mathematical result. Nested object filters are not covered:
search rejects them with ValueError before comparing.
"""

from __future__ import annotations

import pytest

from langgraph.store.sqlite import SqliteStore
from langgraph.store.sqlite.aio import AsyncSqliteStore

# 2**53 is the point where adjacent integers collapse to one float.
LIMIT = 9007199254740992
ABOVE = 9007199254740993
NEG_LIMIT = -9007199254740992
NEG_ABOVE = -9007199254740993


def _keys(items: list) -> set[str]:
    return {item.key for item in items}


def test_get_preserves_integer_above_2_to_the_53() -> None:
    with SqliteStore.from_conn_string(":memory:") as store:
        store.setup()
        store.put(("records",), "over", {"amount": ABOVE})

        item = store.get(("records",), "over")

        assert item is not None
        assert item.value["amount"] == ABOVE
        assert type(item.value["amount"]) is int


def test_search_lte_excludes_integer_above_limit() -> None:
    with SqliteStore.from_conn_string(":memory:") as store:
        store.setup()
        store.put(("records",), "over", {"amount": ABOVE})
        store.put(("records",), "at-cap", {"amount": LIMIT})
        store.put(("records",), "under", {"amount": LIMIT - 1})

        found = store.search(("records",), filter={"amount": {"$lte": LIMIT}}, limit=10)

        assert _keys(found) == {"at-cap", "under"}


def test_search_gt_includes_integer_above_limit() -> None:
    with SqliteStore.from_conn_string(":memory:") as store:
        store.setup()
        store.put(("records",), "over", {"amount": ABOVE})
        store.put(("records",), "at-cap", {"amount": LIMIT})

        found = store.search(("records",), filter={"amount": {"$gt": LIMIT}}, limit=10)

        assert _keys(found) == {"over"}


def test_search_gte_excludes_integer_below_larger_bound() -> None:
    with SqliteStore.from_conn_string(":memory:") as store:
        store.setup()
        store.put(("records",), "over", {"amount": ABOVE})
        store.put(("records",), "at-cap", {"amount": LIMIT})

        found = store.search(("records",), filter={"amount": {"$gte": ABOVE}}, limit=10)

        assert _keys(found) == {"over"}


def test_search_lt_includes_integer_below_larger_bound() -> None:
    with SqliteStore.from_conn_string(":memory:") as store:
        store.setup()
        store.put(("records",), "over", {"amount": ABOVE})
        store.put(("records",), "at-cap", {"amount": LIMIT})
        store.put(("records",), "under", {"amount": LIMIT - 1})

        found = store.search(("records",), filter={"amount": {"$lt": ABOVE}}, limit=10)

        assert _keys(found) == {"at-cap", "under"}


def test_search_eq_matches_only_the_same_integer() -> None:
    """Equality also passes the bound through float before SQLite compares it."""
    with SqliteStore.from_conn_string(":memory:") as store:
        store.setup()
        store.put(("records",), "over", {"amount": ABOVE})
        store.put(("records",), "at-cap", {"amount": LIMIT})

        found = store.search(("records",), filter={"amount": {"$eq": ABOVE}}, limit=10)

        assert _keys(found) == {"over"}


def test_search_ne_rejects_only_the_same_integer() -> None:
    with SqliteStore.from_conn_string(":memory:") as store:
        store.setup()
        store.put(("records",), "over", {"amount": ABOVE})
        store.put(("records",), "at-cap", {"amount": LIMIT})

        found = store.search(("records",), filter={"amount": {"$ne": ABOVE}}, limit=10)

        assert _keys(found) == {"at-cap"}


def test_search_negative_lt_includes_more_negative_integer() -> None:
    with SqliteStore.from_conn_string(":memory:") as store:
        store.setup()
        store.put(("records",), "more-neg", {"amount": NEG_ABOVE})
        store.put(("records",), "at-cap", {"amount": NEG_LIMIT})

        found = store.search(
            ("records",), filter={"amount": {"$lt": NEG_LIMIT}}, limit=10
        )

        assert _keys(found) == {"more-neg"}


def test_search_negative_gte_excludes_more_negative_integer() -> None:
    with SqliteStore.from_conn_string(":memory:") as store:
        store.setup()
        store.put(("records",), "more-neg", {"amount": NEG_ABOVE})
        store.put(("records",), "at-cap", {"amount": NEG_LIMIT})

        found = store.search(
            ("records",), filter={"amount": {"$gte": NEG_LIMIT}}, limit=10
        )

        assert _keys(found) == {"at-cap"}


def test_search_conjunction_accepts_only_integers_inside_window() -> None:
    with SqliteStore.from_conn_string(":memory:") as store:
        store.setup()
        store.put(("records",), "over", {"amount": ABOVE})
        store.put(("records",), "at-cap", {"amount": LIMIT})
        store.put(("records",), "under", {"amount": LIMIT - 1})

        found = store.search(
            ("records",),
            filter={"amount": {"$gt": LIMIT, "$lte": ABOVE}},
            limit=10,
        )

        assert _keys(found) == {"over"}


@pytest.mark.parametrize(
    ("operator", "bound", "stored", "should_match"),
    [
        ("$lte", LIMIT, ABOVE, False),
        ("$gt", LIMIT, ABOVE, True),
        ("$gte", ABOVE, LIMIT, False),
        ("$lt", ABOVE, LIMIT, True),
        ("$eq", ABOVE, ABOVE, True),
        ("$eq", ABOVE, LIMIT, False),
        ("$ne", ABOVE, ABOVE, False),
        ("$ne", ABOVE, LIMIT, True),
        ("$lt", NEG_LIMIT, NEG_ABOVE, True),
        ("$gte", NEG_LIMIT, NEG_ABOVE, False),
        ("$gt", 2.5, 3, True),
        ("$lte", 3.5, 3, True),
    ],
)
async def test_asearch_integer_and_float_bounds(
    operator: str,
    bound: int | float,
    stored: int | float,
    should_match: bool,
) -> None:
    async with AsyncSqliteStore.from_conn_string(":memory:") as store:
        await store.setup()
        await store.aput(("records",), "item", {"amount": stored})

        found = await store.asearch(
            ("records",), filter={"amount": {operator: bound}}, limit=10
        )

        assert _keys(found) == ({"item"} if should_match else set())


async def test_asearch_lte_excludes_integer_above_limit() -> None:
    async with AsyncSqliteStore.from_conn_string(":memory:") as store:
        await store.setup()
        await store.aput(("records",), "over", {"amount": ABOVE})
        await store.aput(("records",), "at-cap", {"amount": LIMIT})

        found = await store.asearch(
            ("records",), filter={"amount": {"$lte": LIMIT}}, limit=10
        )

        assert _keys(found) == {"at-cap"}


async def test_asearch_gt_includes_integer_above_limit() -> None:
    async with AsyncSqliteStore.from_conn_string(":memory:") as store:
        await store.setup()
        await store.aput(("records",), "over", {"amount": ABOVE})
        await store.aput(("records",), "at-cap", {"amount": LIMIT})

        found = await store.asearch(
            ("records",), filter={"amount": {"$gt": LIMIT}}, limit=10
        )

        assert _keys(found) == {"over"}
