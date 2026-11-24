"""A shared SqliteCache must not serve one namespace's row as another node's output.

These tests drive the cache through Pregel, which is the path that stores and
reloads node writes. They fail while a namespace segment that contains a comma
is stored under the same row as the tuple produced by splitting that comma.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from langgraph.cache.sqlite import SqliteCache
from typing_extensions import TypedDict
from xxhash import xxh3_128_hexdigest

from langgraph._internal._constants import CACHE_NS_WRITES
from langgraph.func import entrypoint, task
from langgraph.graph import END, START, StateGraph
from langgraph.pregel._call import identifier
from langgraph.types import CachePolicy


class _State(TypedDict, total=False):
    query: str
    answer: str


def _fixed_key(_value: object) -> str:
    return "shared-entry"


def _entry_key() -> str:
    # Pregel hashes the cache policy key before writing the cache entry.
    return xxh3_128_hexdigest(_fixed_key(None).encode())


def _runtime_namespace(app: Any, node_name: str) -> tuple[str, ...]:
    """Namespace Pregel uses for this node, including the ``__dynamic__`` fallback."""
    return (
        CACHE_NS_WRITES,
        identifier(app.nodes[node_name]) or "__dynamic__",
        node_name,
    )


def _aliased_namespace(app: Any, node_name: str) -> tuple[str, ...]:
    """One-segment collapse of the node namespace, split apart again by commas."""
    namespace = _runtime_namespace(app, node_name)
    return (namespace[0], f"{namespace[1]},{namespace[2]}")


def _compile_victim(cache: SqliteCache, lookup: Any) -> Any:
    graph = StateGraph(_State)
    graph.add_node("users", lookup, cache_policy=CachePolicy(key_func=_fixed_key))
    graph.add_edge(START, "users")
    graph.add_edge("users", END)
    return graph.compile(cache=cache)


def test_graph_run_ignores_value_stored_under_comma_aliased_namespace(
    tmp_path: Path,
) -> None:
    """set(("...lookup,users",), entry) must not become the "users" node's output."""
    calls = {"n": 0}

    def lookup(state: _State) -> dict[str, str]:
        calls["n"] += 1
        return {"answer": f"real:{state['query']}"}

    cache = SqliteCache(path=str(tmp_path / "cache.sqlite"))
    app = _compile_victim(cache, lookup)
    # One segment whose comma splits into the runtime identifier and node name.
    cache.set(
        {
            (_aliased_namespace(app, "users"), _entry_key()): (
                [("answer", "POISONED")],
                None,
            )
        }
    )

    result = app.invoke({"query": "weather", "answer": ""})

    assert calls["n"] == 1
    assert result["answer"] == "real:weather"


def test_clearing_comma_aliased_namespace_does_not_drop_cached_node(
    tmp_path: Path,
) -> None:
    """clear of the aliased tuple must leave the node's own cached writes in place."""
    calls = {"n": 0}

    def lookup(state: _State) -> dict[str, str]:
        calls["n"] += 1
        return {"answer": f"real:{state['query']}"}

    cache = SqliteCache(path=str(tmp_path / "cache.sqlite"))
    app = _compile_victim(cache, lookup)
    aliased_ns = _aliased_namespace(app, "users")

    first = app.invoke({"query": "weather", "answer": ""})
    assert calls["n"] == 1
    assert first["answer"] == "real:weather"

    cache.clear([aliased_ns])
    calls["n"] = 0
    second = app.invoke({"query": "weather", "answer": ""})

    assert calls["n"] == 0
    assert second["answer"] == "real:weather"


def test_functional_task_with_comma_in_identifier_does_not_replace_node_cache(
    tmp_path: Path,
) -> None:
    """A task run must not write its result into another node's namespace.

    Functional tasks cache under (writes, identifier). A node caches under
    (writes, identifier, node name). Putting a comma in the task identifier
    makes those two tuples join to one string today, so the node run treats
    the task result as its own cached writes and does not execute.
    """
    calls = {"n": 0}

    def lookup(state: _State) -> dict[str, str]:
        calls["n"] += 1
        return {"answer": "real"}

    def evil(state: dict[str, Any]) -> dict[str, str]:
        return {"answer": "FROM_EVIL"}

    evil.__module__ = lookup.__module__
    evil.__qualname__ = lookup.__qualname__ + ",users"

    cache = SqliteCache(path=str(tmp_path / "cache.sqlite"))
    evil_task = task(evil, cache_policy=CachePolicy(key_func=_fixed_key))

    @entrypoint(cache=cache)
    def attacker(state: dict[str, Any]) -> dict[str, str]:
        return evil_task(state).result()

    attacker.invoke({"query": "q", "answer": ""})
    app = _compile_victim(cache, lookup)
    result = app.invoke({"query": "q", "answer": ""})

    assert calls["n"] == 1
    assert result["answer"] == "real"


@pytest.mark.anyio
async def test_async_graph_run_ignores_value_stored_under_comma_aliased_namespace(
    tmp_path: Path,
) -> None:
    """aget/aset must not apply a comma-aliased row on an async graph run."""
    calls = {"n": 0}

    async def lookup(state: _State) -> dict[str, str]:
        calls["n"] += 1
        return {"answer": f"real:{state['query']}"}

    cache = SqliteCache(path=str(tmp_path / "cache.sqlite"))
    app = _compile_victim(cache, lookup)
    await cache.aset(
        {
            (_aliased_namespace(app, "users"), _entry_key()): (
                [("answer", "POISONED")],
                None,
            )
        }
    )

    result = await app.ainvoke({"query": "weather", "answer": ""})

    assert calls["n"] == 1
    assert result["answer"] == "real:weather"
