"""Integer range filters on InMemoryStore must use exact integer ordering.

Integers above 2**53 stay distinct in the store, but the search filter
coerces both sides with float before $gt, $gte, $lt, and $lte. These tests
call put and search (and the async pair) and expect the mathematical result.
"""

from __future__ import annotations

import pytest

from langgraph.store.memory import InMemoryStore

# 2**53 is the point where adjacent integers collapse to one float.
LIMIT = 9007199254740992
ABOVE = 9007199254740993
NEG_LIMIT = -9007199254740992
NEG_ABOVE = -9007199254740993


def _keys(items: list) -> set[str]:
    return {item.key for item in items}


def test_get_preserves_integer_above_2_to_the_53() -> None:
    """The stored field is still the original int when search compares it."""
    store = InMemoryStore()
    store.put(("records",), "over", {"amount": ABOVE})

    item = store.get(("records",), "over")

    assert item is not None
    assert item.value["amount"] == ABOVE
    assert type(item.value["amount"]) is int


def test_search_lte_excludes_integer_above_limit() -> None:
    """A maximum check must not return a stored int larger than the bound."""
    store = InMemoryStore()
    store.put(("records",), "over", {"amount": ABOVE})
    store.put(("records",), "at-cap", {"amount": LIMIT})
    store.put(("records",), "under", {"amount": LIMIT - 1})

    found = store.search(("records",), filter={"amount": {"$lte": LIMIT}})

    assert _keys(found) == {"at-cap", "under"}


def test_search_gt_includes_integer_above_limit() -> None:
    """A greater-than check must return a stored int larger than the bound."""
    store = InMemoryStore()
    store.put(("records",), "over", {"amount": ABOVE})
    store.put(("records",), "at-cap", {"amount": LIMIT})

    found = store.search(("records",), filter={"amount": {"$gt": LIMIT}})

    assert _keys(found) == {"over"}


def test_search_gte_excludes_integer_below_larger_bound() -> None:
    """$gte must not treat a smaller int as equal to a larger bound."""
    store = InMemoryStore()
    store.put(("records",), "over", {"amount": ABOVE})
    store.put(("records",), "at-cap", {"amount": LIMIT})

    found = store.search(("records",), filter={"amount": {"$gte": ABOVE}})

    assert _keys(found) == {"over"}


def test_search_lt_includes_integer_below_larger_bound() -> None:
    """$lt must return a stored int that is smaller than the bound."""
    store = InMemoryStore()
    store.put(("records",), "over", {"amount": ABOVE})
    store.put(("records",), "at-cap", {"amount": LIMIT})
    store.put(("records",), "under", {"amount": LIMIT - 1})

    found = store.search(("records",), filter={"amount": {"$lt": ABOVE}})

    assert _keys(found) == {"at-cap", "under"}


def test_search_negative_lt_includes_more_negative_integer() -> None:
    store = InMemoryStore()
    store.put(("records",), "more-neg", {"amount": NEG_ABOVE})
    store.put(("records",), "at-cap", {"amount": NEG_LIMIT})

    found = store.search(("records",), filter={"amount": {"$lt": NEG_LIMIT}})

    assert _keys(found) == {"more-neg"}


def test_search_negative_gte_excludes_more_negative_integer() -> None:
    store = InMemoryStore()
    store.put(("records",), "more-neg", {"amount": NEG_ABOVE})
    store.put(("records",), "at-cap", {"amount": NEG_LIMIT})

    found = store.search(("records",), filter={"amount": {"$gte": NEG_LIMIT}})

    assert _keys(found) == {"at-cap"}


def test_search_nested_lte_excludes_integer_above_limit() -> None:
    """Operators inside a nested object use the same comparison."""
    store = InMemoryStore()
    store.put(("records",), "over", {"meta": {"amount": ABOVE}})
    store.put(("records",), "at-cap", {"meta": {"amount": LIMIT}})

    found = store.search(("records",), filter={"meta": {"amount": {"$lte": LIMIT}}})

    assert _keys(found) == {"at-cap"}


def test_search_conjunction_accepts_only_integers_inside_window() -> None:
    """Both operators in one field filter are applied to the stored int."""
    store = InMemoryStore()
    store.put(("records",), "over", {"amount": ABOVE})
    store.put(("records",), "at-cap", {"amount": LIMIT})
    store.put(("records",), "under", {"amount": LIMIT - 1})

    found = store.search(
        ("records",),
        filter={"amount": {"$gt": LIMIT, "$lte": ABOVE}},
    )

    assert _keys(found) == {"over"}


@pytest.mark.parametrize(
    ("operator", "bound", "stored", "should_match"),
    [
        ("$lte", LIMIT, ABOVE, False),
        ("$gt", LIMIT, ABOVE, True),
        ("$gte", ABOVE, LIMIT, False),
        ("$lt", ABOVE, LIMIT, True),
        ("$lte", LIMIT, LIMIT, True),
        ("$gt", LIMIT, LIMIT, False),
        ("$gte", LIMIT, LIMIT, True),
        ("$lt", LIMIT, LIMIT, False),
        ("$lt", NEG_LIMIT, NEG_ABOVE, True),
        ("$gte", NEG_LIMIT, NEG_ABOVE, False),
        ("$lte", NEG_LIMIT, NEG_ABOVE, True),
        ("$gt", NEG_LIMIT, NEG_ABOVE, False),
        ("$gt", 2.5, 3, True),
        ("$lte", 3.5, 3, True),
        ("$lt", 3, 3.5, False),
        ("$gte", 4, 3.5, False),
        ("$eq", LIMIT, ABOVE, False),
        ("$ne", LIMIT, ABOVE, True),
    ],
)
@pytest.mark.parametrize("async_search", [False, True])
async def test_search_integer_and_float_bounds(
    operator: str,
    bound: int | float,
    stored: int | float,
    should_match: bool,
    async_search: bool,
) -> None:
    store = InMemoryStore()
    store.put(("records",), "item", {"amount": stored})
    filt = {"amount": {operator: bound}}

    if async_search:
        found = await store.asearch(("records",), filter=filt)
    else:
        found = store.search(("records",), filter=filt)

    assert _keys(found) == ({"item"} if should_match else set())


async def test_asearch_lte_excludes_integer_above_limit() -> None:
    store = InMemoryStore()
    store.put(("records",), "over", {"amount": ABOVE})
    store.put(("records",), "at-cap", {"amount": LIMIT})

    found = await store.asearch(("records",), filter={"amount": {"$lte": LIMIT}})

    assert _keys(found) == {"at-cap"}


async def test_asearch_gt_includes_integer_above_limit() -> None:
    store = InMemoryStore()
    store.put(("records",), "over", {"amount": ABOVE})
    store.put(("records",), "at-cap", {"amount": LIMIT})

    found = await store.asearch(("records",), filter={"amount": {"$gt": LIMIT}})

    assert _keys(found) == {"over"}


async def test_asearch_gte_excludes_integer_below_larger_bound() -> None:
    store = InMemoryStore()
    store.put(("records",), "over", {"amount": ABOVE})
    store.put(("records",), "at-cap", {"amount": LIMIT})

    found = await store.asearch(("records",), filter={"amount": {"$gte": ABOVE}})

    assert _keys(found) == {"over"}


async def test_asearch_lt_includes_integer_below_larger_bound() -> None:
    store = InMemoryStore()
    store.put(("records",), "over", {"amount": ABOVE})
    store.put(("records",), "at-cap", {"amount": LIMIT})

    found = await store.asearch(("records",), filter={"amount": {"$lt": ABOVE}})

    assert _keys(found) == {"at-cap"}
