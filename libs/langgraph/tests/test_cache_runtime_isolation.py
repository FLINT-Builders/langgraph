"""Cache keys must not reuse another caller's writes.

Nodes and tasks that opt into `CachePolicy()` are cached by channel input
alone. Runtime, config, store, and `previous` are injected when the node
runs and are not part of that key. These tests are the spec for a fix: they
fail while a later caller with the same channel input and a different
injected identity receives the earlier caller's writes.

A correct fix either includes every injected value the node reads in the
default cache key, or does not cache a node whose signature accepts those
values unless the caller supplies a key function that accounts for them.
Nodes that read only their channel input must still be cached.
"""

from __future__ import annotations

import asyncio
import operator
from dataclasses import dataclass
from typing import Annotated, Any

import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.cache.memory import InMemoryCache
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from typing_extensions import TypedDict

from langgraph.func import entrypoint, task
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.types import CachePolicy, Send

RECORDS = {"user-a": "dossier-a", "user-b": "dossier-b"}


@dataclass
class Context:
    user_id: str


class State(TypedDict, total=False):
    query: str
    record: str


class FanoutState(TypedDict):
    query: str
    parts: Annotated[list[str], operator.add]


class _AuthUser:
    """Stand-in for the authenticated user object the server places on config."""

    def __init__(self, identity: str) -> None:
        self.identity = identity

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _AuthUser) and other.identity == self.identity

    def __hash__(self) -> int:
        return hash(self.identity)


def _auth_config(identity: str) -> RunnableConfig:
    return {
        "configurable": {
            "assistant_id": "assistant",
            "graph_id": "graph",
            "langgraph_auth_user": _AuthUser(identity),
        }
    }


def _user_config(identity: str) -> RunnableConfig:
    return {"configurable": {"user_id": identity}}


def _compile_lookup(node: Any, *, cache: InMemoryCache | None = None) -> Any:
    graph = StateGraph(State, context_schema=Context)
    graph.add_node("lookup", node, cache_policy=CachePolicy())
    graph.add_edge(START, "lookup")
    graph.add_edge("lookup", END)
    return graph.compile(cache=cache or InMemoryCache())


def test_cached_node_does_not_reuse_writes_across_runtime_context() -> None:
    seen: list[str] = []

    def lookup(state: State, runtime: Runtime[Context]) -> dict[str, str]:
        user_id = runtime.context.user_id
        seen.append(user_id)
        return {"record": RECORDS[user_id]}

    graph = _compile_lookup(lookup)
    first = graph.invoke({"query": "same"}, context=Context("user-a"))
    second = graph.invoke({"query": "same"}, context=Context("user-b"))

    assert first["record"] == "dossier-a"
    assert second["record"] == "dossier-b"
    assert seen == ["user-a", "user-b"]


def test_cached_node_does_not_reuse_writes_across_runtime_context_async() -> None:
    async def run() -> None:
        seen: list[str] = []

        async def lookup(state: State, runtime: Runtime[Context]) -> dict[str, str]:
            user_id = runtime.context.user_id
            seen.append(user_id)
            return {"record": RECORDS[user_id]}

        graph = _compile_lookup(lookup)
        first = await graph.ainvoke({"query": "same"}, context=Context("user-a"))
        second = await graph.ainvoke({"query": "same"}, context=Context("user-b"))

        assert first["record"] == "dossier-a"
        assert second["record"] == "dossier-b"
        assert seen == ["user-a", "user-b"]

    asyncio.run(run())


def test_cached_node_does_not_reuse_writes_across_config() -> None:
    seen: list[str] = []

    def lookup(state: State, config: RunnableConfig) -> dict[str, str]:
        user_id = config["configurable"]["user_id"]
        seen.append(user_id)
        return {"record": RECORDS[user_id]}

    graph = _compile_lookup(lookup)
    first = graph.invoke({"query": "same"}, _user_config("user-a"))
    second = graph.invoke({"query": "same"}, _user_config("user-b"))

    assert first["record"] == "dossier-a"
    assert second["record"] == "dossier-b"
    assert seen == ["user-a", "user-b"]


def test_cached_node_does_not_reuse_writes_across_auth_user() -> None:
    seen: list[str] = []

    def lookup(state: State, runtime: Runtime[Context]) -> dict[str, str]:
        server_info = runtime.server_info
        assert server_info is not None and server_info.user is not None
        user_id = server_info.user.identity
        seen.append(user_id)
        return {"record": RECORDS[user_id]}

    graph = _compile_lookup(lookup)
    first = graph.invoke({"query": "same"}, _auth_config("user-a"))
    second = graph.invoke({"query": "same"}, _auth_config("user-b"))

    assert first["record"] == "dossier-a"
    assert second["record"] == "dossier-b"
    assert seen == ["user-a", "user-b"]


def test_cached_node_does_not_reuse_writes_across_stores() -> None:
    seen: list[str] = []

    def lookup(state: State, store: BaseStore) -> dict[str, str]:
        item = store.get(("records",), state["query"])
        assert item is not None
        seen.append(item.value["owner"])
        return {"record": item.value["body"]}

    def build(store: InMemoryStore, cache: InMemoryCache) -> Any:
        graph = StateGraph(State)
        graph.add_node("lookup", lookup, cache_policy=CachePolicy())
        graph.add_edge(START, "lookup")
        graph.add_edge("lookup", END)
        return graph.compile(cache=cache, store=store)

    cache = InMemoryCache()
    store_a = InMemoryStore()
    store_b = InMemoryStore()
    store_a.put(("records",), "acct", {"owner": "user-a", "body": "dossier-a"})
    store_b.put(("records",), "acct", {"owner": "user-b", "body": "dossier-b"})
    first = build(store_a, cache).invoke({"query": "acct"})
    second = build(store_b, cache).invoke({"query": "acct"})

    assert first["record"] == "dossier-a"
    assert second["record"] == "dossier-b"
    assert seen == ["user-a", "user-b"]


def test_node_defaults_cache_policy_does_not_reuse_writes_across_context() -> None:
    seen: list[str] = []

    def lookup(state: State, runtime: Runtime[Context]) -> dict[str, str]:
        user_id = runtime.context.user_id
        seen.append(user_id)
        return {"record": RECORDS[user_id]}

    graph = StateGraph(State, context_schema=Context)
    graph.set_node_defaults(cache_policy=CachePolicy())
    graph.add_node("lookup", lookup)
    graph.add_edge(START, "lookup")
    graph.add_edge("lookup", END)
    app = graph.compile(cache=InMemoryCache())

    first = app.invoke({"query": "same"}, context=Context("user-a"))
    second = app.invoke({"query": "same"}, context=Context("user-b"))

    assert first["record"] == "dossier-a"
    assert second["record"] == "dossier-b"
    assert seen == ["user-a", "user-b"]


def test_cached_send_does_not_reuse_writes_across_runtime_context() -> None:
    seen: list[str] = []

    def fanout(state: State) -> Send:
        return Send("lookup", {"query": state["query"]})

    def lookup(state: State, runtime: Runtime[Context]) -> dict[str, str]:
        user_id = runtime.context.user_id
        seen.append(user_id)
        return {"record": RECORDS[user_id]}

    graph = StateGraph(State, context_schema=Context)
    graph.add_node("lookup", lookup, cache_policy=CachePolicy())
    graph.add_conditional_edges(START, fanout)
    graph.add_edge("lookup", END)
    app = graph.compile(cache=InMemoryCache())

    first = app.invoke({"query": "same"}, context=Context("user-a"))
    second = app.invoke({"query": "same"}, context=Context("user-b"))

    assert first["record"] == "dossier-a"
    assert second["record"] == "dossier-b"
    assert seen == ["user-a", "user-b"]


def test_cached_task_does_not_reuse_writes_across_runtime_context() -> None:
    seen: list[str] = []

    @task(cache_policy=CachePolicy())
    def lookup(query: str, runtime: Runtime[Context]) -> str:
        seen.append(runtime.context.user_id)
        return RECORDS[runtime.context.user_id]

    @entrypoint(cache=InMemoryCache(), context_schema=Context)
    def workflow(query: str) -> str:
        return lookup(query).result()

    first = workflow.invoke("same", context=Context("user-a"))
    second = workflow.invoke("same", context=Context("user-b"))

    assert first == "dossier-a"
    assert second == "dossier-b"
    assert seen == ["user-a", "user-b"]


def test_cached_task_does_not_reuse_writes_across_runtime_context_async() -> None:
    async def run() -> None:
        seen: list[str] = []

        @task(cache_policy=CachePolicy())
        async def lookup(query: str, runtime: Runtime[Context]) -> str:
            seen.append(runtime.context.user_id)
            return RECORDS[runtime.context.user_id]

        @entrypoint(cache=InMemoryCache(), context_schema=Context)
        async def workflow(query: str) -> str:
            return await lookup(query)

        first = await workflow.ainvoke("same", context=Context("user-a"))
        second = await workflow.ainvoke("same", context=Context("user-b"))

        assert first == "dossier-a"
        assert second == "dossier-b"
        assert seen == ["user-a", "user-b"]

    asyncio.run(run())


def test_entrypoint_cache_policy_does_not_reuse_writes_across_runtime_context() -> None:
    seen: list[str] = []

    @entrypoint(
        cache=InMemoryCache(),
        cache_policy=CachePolicy(),
        context_schema=Context,
    )
    def workflow(query: str, runtime: Runtime[Context]) -> str:
        seen.append(runtime.context.user_id)
        return RECORDS[runtime.context.user_id]

    first = workflow.invoke("same", context=Context("user-a"))
    second = workflow.invoke("same", context=Context("user-b"))

    assert first == "dossier-a"
    assert second == "dossier-b"
    assert seen == ["user-a", "user-b"]


def test_entrypoint_cache_policy_does_not_reuse_writes_across_previous() -> None:
    seen: list[int] = []

    @entrypoint(
        checkpointer=InMemorySaver(),
        cache=InMemoryCache(),
        cache_policy=CachePolicy(),
    )
    def workflow(n: int, *, previous: int | None = None) -> entrypoint.final[int, int]:
        base = 0 if previous is None else previous
        seen.append(base)
        return entrypoint.final(value=base, save=base + 10)

    config = {"configurable": {"thread_id": "thread-a"}}
    assert workflow.invoke(1, config) == 0
    assert workflow.invoke(1, config) == 10
    assert seen == [0, 10]


def test_cache_still_hits_when_node_reads_only_channel_input() -> None:
    seen: list[str] = []

    def lookup(state: State) -> dict[str, str]:
        seen.append(state["query"])
        return {"record": "dossier-shared"}

    graph = _compile_lookup(lookup)
    first = graph.invoke({"query": "same"})
    second = graph.invoke({"query": "same"})

    assert first["record"] == "dossier-shared"
    assert second["record"] == "dossier-shared"
    assert seen == ["same"]


def test_stream_does_not_replay_another_callers_updates() -> None:
    seen: list[str] = []

    def lookup(state: State, runtime: Runtime[Context]) -> dict[str, str]:
        user_id = runtime.context.user_id
        seen.append(user_id)
        return {"record": RECORDS[user_id]}

    graph = _compile_lookup(lookup)
    first = [
        chunk
        for chunk in graph.stream(
            {"query": "same"}, context=Context("user-a"), stream_mode="updates"
        )
    ]
    second = [
        chunk
        for chunk in graph.stream(
            {"query": "same"}, context=Context("user-b"), stream_mode="updates"
        )
    ]

    assert first == [{"lookup": {"record": "dossier-a"}}]
    assert second == [{"lookup": {"record": "dossier-b"}}]
    assert seen == ["user-a", "user-b"]


@pytest.mark.anyio
async def test_astream_does_not_replay_another_callers_updates() -> None:
    seen: list[str] = []

    async def lookup(state: State, runtime: Runtime[Context]) -> dict[str, str]:
        user_id = runtime.context.user_id
        seen.append(user_id)
        return {"record": RECORDS[user_id]}

    graph = _compile_lookup(lookup)
    first = [
        chunk
        async for chunk in graph.astream(
            {"query": "same"}, context=Context("user-a"), stream_mode="updates"
        )
    ]
    second = [
        chunk
        async for chunk in graph.astream(
            {"query": "same"}, context=Context("user-b"), stream_mode="updates"
        )
    ]

    assert first == [{"lookup": {"record": "dossier-a"}}]
    assert second == [{"lookup": {"record": "dossier-b"}}]
    assert seen == ["user-a", "user-b"]


def test_send_fanout_aggregates_per_caller() -> None:
    """Several sends in one run share a cache, and a later caller does not."""

    seen: list[str] = []

    def fanout(state: FanoutState) -> list[Send]:
        return [
            Send("lookup", {"query": state["query"], "part": part})
            for part in ("x", "y")
        ]

    def lookup(
        state: dict[str, str], runtime: Runtime[Context]
    ) -> dict[str, list[str]]:
        user_id = runtime.context.user_id
        seen.append(f"{user_id}:{state['part']}")
        return {"parts": [f"{RECORDS[user_id]}:{state['part']}"]}

    graph = StateGraph(FanoutState, context_schema=Context)
    graph.add_node("lookup", lookup, cache_policy=CachePolicy())
    graph.add_conditional_edges(START, fanout)
    graph.add_edge("lookup", END)
    app = graph.compile(cache=InMemoryCache())

    first = app.invoke({"query": "same", "parts": []}, context=Context("user-a"))
    second = app.invoke({"query": "same", "parts": []}, context=Context("user-b"))

    assert sorted(first["parts"]) == ["dossier-a:x", "dossier-a:y"]
    assert sorted(second["parts"]) == ["dossier-b:x", "dossier-b:y"]
    assert sorted(seen) == [
        "user-a:x",
        "user-a:y",
        "user-b:x",
        "user-b:y",
    ]
