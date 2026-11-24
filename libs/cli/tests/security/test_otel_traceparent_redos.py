"""Attacker-controlled traceparent values must not stall run startup.

`langgraph dev` (langgraph-api 0.10.0, the inmem extra) stores a W3C
traceparent on the run and later parses it with
`opentelemetry.trace.propagation.tracecontext.TraceContextTextMapPropagator`
from opentelemetry-api. Through 1.41.0 that parser's traceparent regular
expression backtracks quadratically on a value that has a trace-context
prefix, a long run of spaces, and an embedded newline.

These tests build that value as JSON, send it through the run-creation
paths the server actually uses, and then call `restore_otel_trace_context`,
which is what `langgraph_api.worker.worker` calls on the stored configurable
before it streams the graph. They fail while opentelemetry-api 1.41.0 is
installed and pass once that parser no longer backtracks (1.45.2 or later).

Paths covered:

* `POST /runs` and `POST /threads/{thread_id}/runs`. `ApiRequest.json`
  strips `__otel_traceparent__` from `config.configurable` and `context`,
  then `create_valid_run` copies `checkpoint` onto the configurable.
* Protocol v2 `run.start` (`POST /threads/{thread_id}/commands` and the
  websocket on the same path). Both parse JSON with `orjson` and call
  `ThreadRunManager.handle_command`, which copies `config.configurable`
  through without the reserved-key strip.

Not covered by a separate test: MCP `tools/call` and A2A `SendMessage`
pass `context` into `runs.create`. That field is stripped by
`sanitize_reserved_keys` before `create_valid_run`, and neither caller
sets `checkpoint`. Cron scheduling and `POST /runs/batch` call the same
`create_valid_run` as the REST test.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
import time
import uuid
from typing import Any, TypedDict

import orjson
import pytest

# Config is read at import. Set the inmem server environment before importing
# langgraph_api. OTEL must be on: restore_otel_trace_context returns immediately
# when it is off, which is the default unless this flag or an OTLP endpoint is set.
os.environ.setdefault("DATABASE_URI", ":memory:")
os.environ.setdefault("REDIS_URI", "fake")
os.environ.setdefault("MIGRATIONS_PATH", "__inmem")
os.environ.setdefault("LANGGRAPH_RUNTIME_EDITION", "inmem")
os.environ.setdefault("LANGGRAPH_DISABLE_FILE_PERSISTENCE", "true")
os.environ.setdefault("LANGSMITH_LANGGRAPH_API_VARIANT", "local_dev")
os.environ["OTEL_ENABLED"] = "true"
# The inmem runtime builds its cache path from the process cwd at import.
os.chdir(tempfile.mkdtemp(prefix="otel-traceparent-"))

from langgraph.graph import END, START, StateGraph  # noqa: E402
from langgraph_api.api.event_streaming import _make_manager  # noqa: E402
from langgraph_api.graph import register_graph  # noqa: E402
from langgraph_api.models.run import create_valid_run  # noqa: E402
from langgraph_api.otel_context import (  # noqa: E402
    OTEL_TRACEPARENT_KEY,
    restore_otel_trace_context,
)
from langgraph_api.route import _json_loads  # noqa: E402
from langgraph_api.validation import RunCreateStateful, RunCreateStateless  # noqa: E402
from langgraph_runtime.database import connect  # noqa: E402

# 1.41.0 spends well over a second here. 1.45.2 returns in well under a millisecond.
_SPACE_COUNT = 32_000
_MAX_EXTRACT_SECONDS = 0.25

pytestmark = pytest.mark.usefixtures("_workdir")


class _State(TypedDict):
    x: str


def _node(state: _State) -> _State:
    return state


def _malicious_traceparent() -> str:
    """trace-context prefix, a long space run, and an embedded newline."""
    prefix = "00-" + ("ab" * 16) + "-" + ("cd" * 8) + "-01-"
    return prefix + (" " * _SPACE_COUNT) + "\nX"


@pytest.fixture(scope="module")
def _workdir(tmp_path_factory: pytest.TempPathFactory) -> None:
    """Keep the inmem runtime's relative cache files out of the repo."""
    os.chdir(tmp_path_factory.mktemp("otel-traceparent"))


@pytest.fixture(scope="module")
def _graph_registered(_workdir: None) -> str:
    builder: StateGraph = StateGraph(_State)
    builder.add_node("n", _node)
    builder.add_edge(START, "n")
    builder.add_edge("n", END)
    graph = builder.compile()

    async def _register() -> None:
        await register_graph("agent", graph, {})

    asyncio.run(_register())
    return "agent"


def _assert_extract_is_fast(configurable: dict[str, Any], malicious: str) -> None:
    assert configurable.get(OTEL_TRACEPARENT_KEY) == malicious
    started = time.perf_counter()
    with restore_otel_trace_context(configurable):
        pass
    elapsed = time.perf_counter() - started
    assert elapsed < _MAX_EXTRACT_SECONDS, (
        f"restore_otel_trace_context took {elapsed:.3f}s on a "
        f"{len(malicious)}-byte traceparent; TraceContextTextMapPropagator.extract "
        "is backtracking"
    )


def test_rest_run_checkpoint_traceparent_does_not_stall_extract(
    _graph_registered: str,
) -> None:
    """checkpoint survives reserved-key stripping and is what the worker parses."""
    malicious = _malicious_traceparent()
    body = {
        "assistant_id": _graph_registered,
        "input": {"x": "hi"},
        "config": {"configurable": {OTEL_TRACEPARENT_KEY: malicious}},
        "context": {OTEL_TRACEPARENT_KEY: malicious},
        "checkpoint": {
            "thread_id": str(uuid.uuid4()),
            OTEL_TRACEPARENT_KEY: malicious,
        },
    }

    raw = bytearray(orjson.dumps(body))
    # Same parser ApiRequest.json uses for POST /runs and the stateful run routes.
    stateless = _json_loads(raw, RunCreateStateless)
    stateful = _json_loads(raw, RunCreateStateful)
    for parsed in (stateless, stateful):
        assert (
            parsed.get("config", {}).get("configurable", {}).get(OTEL_TRACEPARENT_KEY)
            is None
        )
        assert parsed.get("context", {}).get(OTEL_TRACEPARENT_KEY) is None
        assert parsed["checkpoint"][OTEL_TRACEPARENT_KEY] == malicious

    async def _create() -> dict[str, Any]:
        async with connect() as conn:
            return await create_valid_run(conn, None, stateless, {})

    run = asyncio.run(_create())
    _assert_extract_is_fast(
        run["kwargs"]["config"]["configurable"],
        malicious,
    )


def test_event_streaming_run_start_traceparent_does_not_stall_extract(
    _graph_registered: str,
) -> None:
    """run.start copies config.configurable without stripping the reserved key."""
    malicious = _malicious_traceparent()
    command = {
        "id": 1,
        "method": "run.start",
        "params": {
            "assistant_id": _graph_registered,
            "input": {"x": "hi"},
            "config": {"configurable": {OTEL_TRACEPARENT_KEY: malicious}},
        },
    }

    # HTTP commands and the websocket both orjson.loads the frame, then handle_command.
    parsed = orjson.loads(orjson.dumps(command))

    async def _start() -> dict[str, Any]:
        manager = _make_manager(str(uuid.uuid4()))
        try:
            response = await manager.handle_command(parsed)
        finally:
            await manager.close()
        return response

    response = asyncio.run(_start())
    assert response.get("type") == "success", response

    async def _stored() -> dict[str, Any]:
        async with connect() as conn:
            runs = [
                run
                for run in conn.store["runs"]
                if str(run["run_id"]) == response["result"]["run_id"]
            ]
        assert len(runs) == 1
        return runs[0]["kwargs"]["config"]["configurable"]

    _assert_extract_is_fast(asyncio.run(_stored()), malicious)
