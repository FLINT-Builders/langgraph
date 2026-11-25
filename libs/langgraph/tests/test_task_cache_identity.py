"""Distinct callables must not reuse each other's task-cache entries.

`identifier()` is only `module` plus `qualname`. Functional tasks use that
string, or `__dynamic__` when it is missing, as the whole cache namespace.
Factory-built functions, lambdas, and partials therefore share a namespace,
and a shared cache serves one callable's writes as another's.

These tests are the spec for that isolation. They fail while the collision
is present and pass once each distinct callable has its own namespace.
Repeating one callable must still hit its own cache.

Not covered here:

- `prepare_node_error_handler_task` builds the same style of key, but
  `StateGraph` sets error-handler `cache_policy` to `None` and does not copy
  the builder default onto handler nodes, and `compile()` does not set a
  graph-level policy. The handler cache lookup is not reachable from the
  public graph API.
- `RedisCache` stores the same `(namespace, key)` pair. It is not executed
  here; `InMemoryCache` and `SqliteCache` are.
"""

from __future__ import annotations

import functools
from typing import Any

import pytest
from langgraph.cache.base import BaseCache
from langgraph.cache.memory import InMemoryCache
from langgraph.cache.sqlite import SqliteCache
from typing_extensions import TypedDict

from langgraph.func import entrypoint, task
from langgraph.graph import END, START, StateGraph
from langgraph.types import CachePolicy, Send

pytestmark = pytest.mark.anyio


@pytest.fixture(params=["memory", "sqlite"])
def shared_cache(request: pytest.FixtureRequest) -> BaseCache:
    if request.param == "memory":
        return InMemoryCache()
    if request.param == "sqlite":
        return SqliteCache(path=":memory:")
    raise AssertionError(request.param)


def _make_task_workflow(
    prefix: str,
    calls: list[str],
    cache: BaseCache,
    *,
    task_name: str | None = None,
) -> tuple[Any, Any]:
    @task(name=task_name, cache_policy=CachePolicy())
    def call_model(prompt: str) -> str:
        calls.append(prefix)
        return f"{prefix}:{prompt}"

    @entrypoint(cache=cache)
    def run(prompt: str) -> str:
        return call_model(prompt).result()

    return run, call_model


def _make_async_task_workflow(
    prefix: str,
    calls: list[str],
    cache: BaseCache,
) -> tuple[Any, Any]:
    @task(cache_policy=CachePolicy())
    async def call_model(prompt: str) -> str:
        calls.append(prefix)
        return f"{prefix}:{prompt}"

    @entrypoint(cache=cache)
    async def run(prompt: str) -> str:
        return await call_model(prompt)

    return run, call_model


def _make_lambda_workflow(prefix: str, calls: list[str], cache: BaseCache) -> Any:
    call_model = task(cache_policy=CachePolicy())(
        lambda prompt, prefix=prefix: (
            calls.append(prefix),
            f"{prefix}:{prompt}",
        )[1]
    )

    @entrypoint(cache=cache)
    def run(prompt: str) -> str:
        return call_model(prompt).result()

    return run


def _make_async_lambda_workflow(prefix: str, calls: list[str], cache: BaseCache) -> Any:
    async def call_model(prompt: str, prefix: str = prefix) -> str:
        calls.append(prefix)
        return f"{prefix}:{prompt}"

    # Same module and qualname as every other lambda created here.
    call_model.__name__ = "<lambda>"
    call_model.__qualname__ = (
        _make_async_lambda_workflow.__qualname__ + ".<locals>.<lambda>"
    )
    wrapped = task(cache_policy=CachePolicy())(call_model)

    @entrypoint(cache=cache)
    async def run(prompt: str) -> str:
        return await wrapped(prompt)

    return run


def _impl(prompt: str, prefix: str, calls: list[str]) -> str:
    calls.append(prefix)
    return f"{prefix}:{prompt}"


def _make_partial_workflow(prefix: str, calls: list[str], cache: BaseCache) -> Any:
    call_model = task(cache_policy=CachePolicy())(
        functools.partial(_impl, prefix=prefix, calls=calls)
    )

    @entrypoint(cache=cache)
    def run(prompt: str) -> str:
        return call_model(prompt).result()

    return run


def _make_dynamic_workflow(prefix: str, calls: list[str], cache: BaseCache) -> Any:
    def call_model(prompt: str) -> str:
        calls.append(prefix)
        return f"{prefix}:{prompt}"

    # Distinct qualnames, no module: both fall back to `__dynamic__`.
    call_model.__module__ = None  # type: ignore[assignment]
    call_model.__name__ = f"dyn_{prefix}"
    call_model.__qualname__ = f"dyn_{prefix}"
    wrapped = task(cache_policy=CachePolicy())(call_model)

    @entrypoint(cache=cache)
    def run(prompt: str) -> str:
        return wrapped(prompt).result()

    return run


class _State(TypedDict, total=False):
    prompt: str
    out: str


def _make_state_graph(prefix: str, calls: list[str], cache: BaseCache) -> Any:
    def model(state: _State) -> _State:
        calls.append(prefix)
        return {"out": f"{prefix}:{state['prompt']}"}

    graph = StateGraph(_State)
    graph.add_node("model", model, cache_policy=CachePolicy())
    graph.add_edge(START, "model")
    graph.add_edge("model", END)
    return graph.compile(cache=cache)


def _make_send_graph(prefix: str, calls: list[str], cache: BaseCache) -> Any:
    def model(state: _State) -> _State:
        calls.append(prefix)
        return {"out": f"{prefix}:{state['prompt']}"}

    def route(state: _State) -> list[Send]:
        return [Send("model", {"prompt": state["prompt"]})]

    graph = StateGraph(_State)
    graph.add_node("model", model, cache_policy=CachePolicy())
    graph.add_edge("model", END)
    graph.set_conditional_entry_point(route, path_map=["model"])
    return graph.compile(cache=cache)


def _make_cached_entrypoint(prefix: str, calls: list[str], cache: BaseCache) -> Any:
    @entrypoint(cache=cache, cache_policy=CachePolicy())
    def run(prompt: str) -> str:
        calls.append(prefix)
        return f"{prefix}:{prompt}"

    return run


def test_factory_tasks_do_not_share_cached_writes(shared_cache: BaseCache) -> None:
    calls: list[str] = []
    run_a, _task_a = _make_task_workflow("alpha", calls, shared_cache)
    run_b, _task_b = _make_task_workflow("beta", calls, shared_cache)

    assert run_a.invoke("same-prompt") == "alpha:same-prompt"
    assert run_b.invoke("same-prompt") == "beta:same-prompt"
    assert calls == ["alpha", "beta"]


def test_distinct_task_names_do_not_share_cached_writes(
    shared_cache: BaseCache,
) -> None:
    calls: list[str] = []
    run_a, _task_a = _make_task_workflow(
        "alpha", calls, shared_cache, task_name="call_alpha"
    )
    run_b, _task_b = _make_task_workflow(
        "beta", calls, shared_cache, task_name="call_beta"
    )

    assert run_a.invoke("same-prompt") == "alpha:same-prompt"
    assert run_b.invoke("same-prompt") == "beta:same-prompt"
    assert calls == ["alpha", "beta"]


def test_lambda_tasks_do_not_share_cached_writes(shared_cache: BaseCache) -> None:
    calls: list[str] = []
    run_a = _make_lambda_workflow("alpha", calls, shared_cache)
    run_b = _make_lambda_workflow("beta", calls, shared_cache)

    assert run_a.invoke("same-prompt") == "alpha:same-prompt"
    assert run_b.invoke("same-prompt") == "beta:same-prompt"
    assert calls == ["alpha", "beta"]


def test_partial_tasks_do_not_share_cached_writes(shared_cache: BaseCache) -> None:
    calls: list[str] = []
    run_a = _make_partial_workflow("alpha", calls, shared_cache)
    run_b = _make_partial_workflow("beta", calls, shared_cache)

    assert run_a.invoke("same-prompt") == "alpha:same-prompt"
    assert run_b.invoke("same-prompt") == "beta:same-prompt"
    assert calls == ["alpha", "beta"]


def test_moduleless_tasks_do_not_share_cached_writes(
    shared_cache: BaseCache,
) -> None:
    calls: list[str] = []
    run_a = _make_dynamic_workflow("alpha", calls, shared_cache)
    run_b = _make_dynamic_workflow("beta", calls, shared_cache)

    assert run_a.invoke("same-prompt") == "alpha:same-prompt"
    assert run_b.invoke("same-prompt") == "beta:same-prompt"
    assert calls == ["alpha", "beta"]


def test_repeated_call_of_one_task_is_cached(shared_cache: BaseCache) -> None:
    calls: list[str] = []
    run, _task = _make_task_workflow("alpha", calls, shared_cache)

    assert run.invoke("one") == "alpha:one"
    assert run.invoke("one") == "alpha:one"
    assert run.invoke("two") == "alpha:two"
    assert calls == ["alpha", "alpha"]


def test_clear_cache_does_not_drop_a_different_factory_task(
    shared_cache: BaseCache,
) -> None:
    calls: list[str] = []
    run_a, task_a = _make_task_workflow("alpha", calls, shared_cache)
    run_b, _task_b = _make_task_workflow("beta", calls, shared_cache)

    assert run_a.invoke("prompt-a") == "alpha:prompt-a"
    assert run_b.invoke("prompt-b") == "beta:prompt-b"
    task_a.clear_cache(shared_cache)

    assert run_a.invoke("prompt-a") == "alpha:prompt-a"
    assert run_b.invoke("prompt-b") == "beta:prompt-b"
    assert calls == ["alpha", "beta", "alpha"]


async def test_async_factory_tasks_do_not_share_cached_writes(
    shared_cache: BaseCache,
) -> None:
    calls: list[str] = []
    run_a, _task_a = _make_async_task_workflow("alpha", calls, shared_cache)
    run_b, _task_b = _make_async_task_workflow("beta", calls, shared_cache)

    assert await run_a.ainvoke("same-prompt") == "alpha:same-prompt"
    assert await run_b.ainvoke("same-prompt") == "beta:same-prompt"
    assert calls == ["alpha", "beta"]


async def test_async_lambda_tasks_do_not_share_cached_writes(
    shared_cache: BaseCache,
) -> None:
    calls: list[str] = []
    run_a = _make_async_lambda_workflow("alpha", calls, shared_cache)
    run_b = _make_async_lambda_workflow("beta", calls, shared_cache)

    assert await run_a.ainvoke("same-prompt") == "alpha:same-prompt"
    assert await run_b.ainvoke("same-prompt") == "beta:same-prompt"
    assert calls == ["alpha", "beta"]


async def test_aclear_cache_does_not_drop_a_different_factory_task(
    shared_cache: BaseCache,
) -> None:
    calls: list[str] = []
    run_a, task_a = _make_async_task_workflow("alpha", calls, shared_cache)
    run_b, _task_b = _make_async_task_workflow("beta", calls, shared_cache)

    assert await run_a.ainvoke("prompt-a") == "alpha:prompt-a"
    assert await run_b.ainvoke("prompt-b") == "beta:prompt-b"
    await task_a.aclear_cache(shared_cache)

    assert await run_a.ainvoke("prompt-a") == "alpha:prompt-a"
    assert await run_b.ainvoke("prompt-b") == "beta:prompt-b"
    assert calls == ["alpha", "beta", "alpha"]


def test_factory_state_graph_nodes_do_not_share_cached_writes(
    shared_cache: BaseCache,
) -> None:
    calls: list[str] = []
    graph_a = _make_state_graph("alpha", calls, shared_cache)
    graph_b = _make_state_graph("beta", calls, shared_cache)

    assert graph_a.invoke({"prompt": "same-prompt"})["out"] == "alpha:same-prompt"
    assert graph_b.invoke({"prompt": "same-prompt"})["out"] == "beta:same-prompt"
    assert calls == ["alpha", "beta"]


def test_same_state_graph_function_still_hits_its_own_cache(
    shared_cache: BaseCache,
) -> None:
    calls: list[str] = []

    def model(state: _State) -> _State:
        calls.append("shared")
        return {"out": f"shared:{state['prompt']}"}

    def compile_graph() -> Any:
        graph = StateGraph(_State)
        graph.add_node("model", model, cache_policy=CachePolicy())
        graph.add_edge(START, "model")
        graph.add_edge("model", END)
        return graph.compile(cache=shared_cache)

    graph_a = compile_graph()
    graph_b = compile_graph()
    assert graph_a.invoke({"prompt": "same-prompt"})["out"] == "shared:same-prompt"
    assert graph_b.invoke({"prompt": "same-prompt"})["out"] == "shared:same-prompt"
    assert calls == ["shared"]


def test_send_nodes_do_not_share_cached_writes(shared_cache: BaseCache) -> None:
    calls: list[str] = []
    graph_a = _make_send_graph("alpha", calls, shared_cache)
    graph_b = _make_send_graph("beta", calls, shared_cache)

    assert graph_a.invoke({"prompt": "same-prompt"})["out"] == "alpha:same-prompt"
    assert graph_b.invoke({"prompt": "same-prompt"})["out"] == "beta:same-prompt"
    assert calls == ["alpha", "beta"]


def test_factory_entrypoint_cache_policy_does_not_share_writes(
    shared_cache: BaseCache,
) -> None:
    calls: list[str] = []
    run_a = _make_cached_entrypoint("alpha", calls, shared_cache)
    run_b = _make_cached_entrypoint("beta", calls, shared_cache)

    assert run_a.invoke("same-prompt") == "alpha:same-prompt"
    assert run_b.invoke("same-prompt") == "beta:same-prompt"
    assert calls == ["alpha", "beta"]


def test_graph_clear_cache_does_not_drop_another_graphs_node(
    shared_cache: BaseCache,
) -> None:
    calls: list[str] = []
    graph_a = _make_state_graph("alpha", calls, shared_cache)
    graph_b = _make_state_graph("beta", calls, shared_cache)

    assert graph_a.invoke({"prompt": "prompt-a"})["out"] == "alpha:prompt-a"
    assert graph_b.invoke({"prompt": "prompt-b"})["out"] == "beta:prompt-b"
    graph_a.clear_cache()

    assert graph_a.invoke({"prompt": "prompt-a"})["out"] == "alpha:prompt-a"
    assert graph_b.invoke({"prompt": "prompt-b"})["out"] == "beta:prompt-b"
    assert calls == ["alpha", "beta", "alpha"]


async def test_async_state_graph_nodes_do_not_share_cached_writes(
    shared_cache: BaseCache,
) -> None:
    calls: list[str] = []
    graph_a = _make_state_graph("alpha", calls, shared_cache)
    graph_b = _make_state_graph("beta", calls, shared_cache)

    result_a = await graph_a.ainvoke({"prompt": "same-prompt"})
    result_b = await graph_b.ainvoke({"prompt": "same-prompt"})
    assert result_a["out"] == "alpha:same-prompt"
    assert result_b["out"] == "beta:same-prompt"
    assert calls == ["alpha", "beta"]


async def test_async_send_nodes_do_not_share_cached_writes(
    shared_cache: BaseCache,
) -> None:
    calls: list[str] = []
    graph_a = _make_send_graph("alpha", calls, shared_cache)
    graph_b = _make_send_graph("beta", calls, shared_cache)

    result_a = await graph_a.ainvoke({"prompt": "same-prompt"})
    result_b = await graph_b.ainvoke({"prompt": "same-prompt"})
    assert result_a["out"] == "alpha:same-prompt"
    assert result_b["out"] == "beta:same-prompt"
    assert calls == ["alpha", "beta"]


def test_distinct_node_names_do_not_share_cache(shared_cache: BaseCache) -> None:
    """Different node names already separate graph-node cache entries."""
    calls: list[str] = []

    def build(node_name: str, prefix: str) -> Any:
        def model(state: _State) -> _State:
            calls.append(prefix)
            return {"out": f"{prefix}:{state['prompt']}"}

        # Force the same qualname while keeping the node names different.
        model.__qualname__ = "build.<locals>.model"
        graph = StateGraph(_State)
        graph.add_node(node_name, model, cache_policy=CachePolicy())
        graph.add_edge(START, node_name)
        graph.add_edge(node_name, END)
        return graph.compile(cache=shared_cache)

    graph_a = build("alpha_node", "alpha")
    graph_b = build("beta_node", "beta")
    assert graph_a.invoke({"prompt": "same-prompt"})["out"] == "alpha:same-prompt"
    assert graph_b.invoke({"prompt": "same-prompt"})["out"] == "beta:same-prompt"
    assert calls == ["alpha", "beta"]
