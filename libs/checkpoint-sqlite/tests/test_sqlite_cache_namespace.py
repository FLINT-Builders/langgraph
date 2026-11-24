"""Namespace tuples that differ only by comma placement must stay distinct.

SqliteCache stores a namespace by joining its segments with commas and rebuilds
the tuple by splitting on every comma. These tests are the isolation contract
for get, set, and clear (and their async wrappers). They fail while a comma, or
the character used to escape a comma, inside a segment can alias another tuple.
"""

from __future__ import annotations

from pathlib import Path

from langgraph.cache.base import FullKey, Namespace

from langgraph.cache.sqlite import SqliteCache

# Each pair is two namespace tuples that must never share a row.
# Tuples that contain a raw comma alias today because segments are joined and
# split on ",". Tuples that contain "%" must stay distinct from the tuple a
# comma-encoding would produce: rewriting "," to "%2C" without encoding "%"
# maps ("a%2Cb",) and ("a", "b") onto one stored string.
_ISOLATED_PAIRS: tuple[tuple[Namespace, Namespace], ...] = (
    (("acme,users",), ("acme", "users")),
    (("a", "b,c"), ("a", "b", "c")),
    (("a,b", "c"), ("a", "b", "c")),
    (("a,b", "c"), ("a", "b,c")),
    (("a%,b",), ("a%", "b")),
    (("a%2Cb",), ("a", "b")),
    (("a%2Cb",), ("a,b",)),
)


def _cache(path: Path) -> SqliteCache:
    return SqliteCache(path=str(path))


def _assert_pair_isolated(
    cache: SqliteCache, left: Namespace, right: Namespace, entry: str
) -> None:
    left_key: FullKey = (left, entry)
    right_key: FullKey = (right, entry)

    cache.set({left_key: ("left", None)})
    assert cache.get([left_key]) == {left_key: "left"}

    cache.set({right_key: ("right", None)})
    assert cache.get([right_key]) == {right_key: "right"}
    # A later write to the other tuple must not replace this row.
    assert cache.get([left_key]) == {left_key: "left"}

    cache.clear([left])
    assert cache.get([left_key]) == {}
    assert cache.get([right_key]) == {right_key: "right"}


def test_namespace_tuples_with_commas_do_not_alias(tmp_path: Path) -> None:
    """get, set, and clear must keep comma-distinct namespace tuples apart."""
    for index, (left, right) in enumerate(_ISOLATED_PAIRS):
        cache = _cache(tmp_path / f"pair-{index}.sqlite")
        _assert_pair_isolated(cache, left, right, entry=f"tool:{index}")

    # A comma in the entry name is part of that entry, not another namespace segment.
    cache = _cache(tmp_path / "entry.sqlite")
    entry_key: FullKey = (("acme",), "users,tool")
    other_key: FullKey = (("acme", "users"), "tool")
    cache.set({entry_key: ("entry", None), other_key: ("other", None)})
    assert cache.get([entry_key]) == {entry_key: "entry"}
    assert cache.get([other_key]) == {other_key: "other"}
    cache.clear([("acme",)])
    assert cache.get([entry_key]) == {}
    assert cache.get([other_key]) == {other_key: "other"}


async def _assert_pair_isolated_async(
    cache: SqliteCache, left: Namespace, right: Namespace, entry: str
) -> None:
    left_key: FullKey = (left, entry)
    right_key: FullKey = (right, entry)

    await cache.aset({left_key: ("left", None)})
    assert await cache.aget([left_key]) == {left_key: "left"}

    await cache.aset({right_key: ("right", None)})
    assert await cache.aget([right_key]) == {right_key: "right"}
    assert await cache.aget([left_key]) == {left_key: "left"}

    await cache.aclear([left])
    assert await cache.aget([left_key]) == {}
    assert await cache.aget([right_key]) == {right_key: "right"}


async def test_async_namespace_tuples_with_commas_do_not_alias(tmp_path: Path) -> None:
    """aget, aset, and aclear must use the same non-aliasing namespace encoding."""
    for index, (left, right) in enumerate(_ISOLATED_PAIRS):
        cache = _cache(tmp_path / f"async-{index}.sqlite")
        await _assert_pair_isolated_async(cache, left, right, entry=f"tool:{index}")
