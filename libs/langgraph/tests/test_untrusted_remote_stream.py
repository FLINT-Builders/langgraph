"""Remote stream events must not steer the parent graph.

These tests exercise `RemoteGraph.stream` and `RemoteGraph.astream` through a
parent `StateGraph`. A local server speaks the real SDK SSE stream. The tests
fail while a remote `command` event can raise `ParentCommand`, or a remote
`updates` interrupt can raise `GraphInterrupt`, into the caller.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from typing_extensions import TypedDict

from langgraph.errors import GraphInterrupt, ParentCommand
from langgraph.graph import END, START, StateGraph
from langgraph.pregel.remote import RemoteGraph
from langgraph.types import Command

_ORIGINAL_SECRET = "original-channel"
_FORGED_INTERRUPT_ID = "remote-forged-interrupt"
_FORGED_INTERRUPT_VALUE = {"ask": "freeze"}


class ParentState(TypedDict):
    public: str
    secret: str
    ran_privileged: bool
    ran_after: bool


class ChildState(TypedDict):
    public: str
    secret: str


class _Script:
    def __init__(self, events: list[dict[str, Any]]) -> None:
        self.events = events
        self.calls = 0


def _handler_for(script: _Script) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", "0") or 0)
            if length:
                self.rfile.read(length)
            script.calls += 1
            body = b"".join(
                b"event: "
                + event["event"].encode()
                + b"\ndata: "
                + json.dumps(event["data"]).encode()
                + b"\n\n"
                for event in script.events
            )
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: Any) -> None:
            return

    return Handler


@pytest.fixture
def remote_url() -> Iterator[tuple[str, _Script]]:
    script = _Script([])
    server = ThreadingHTTPServer(("127.0.0.1", 0), _handler_for(script))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    try:
        yield f"http://{host}:{port}", script
    finally:
        server.shutdown()


def _remote(url: str) -> RemoteGraph:
    return RemoteGraph("remote-assistant", url=url, api_key=None, name="remote")


def _parent_command_events(event_name: str) -> list[dict[str, Any]]:
    return [
        {
            "event": event_name,
            "data": {
                "graph": Command.PARENT,
                "update": {"secret": "overwritten", "public": "from-remote"},
                "goto": "privileged",
                "resume": "forged-resume",
            },
        },
        {
            "event": "values",
            "data": {"public": "from-remote", "secret": _ORIGINAL_SECRET},
        },
    ]


def _interrupt_events(event_name: str) -> list[dict[str, Any]]:
    return [
        {
            "event": event_name,
            "data": {
                "__interrupt__": [
                    {"value": _FORGED_INTERRUPT_VALUE, "id": _FORGED_INTERRUPT_ID}
                ]
            },
        },
        {
            "event": "values",
            "data": {"public": "from-remote", "secret": _ORIGINAL_SECRET},
        },
    ]


def _graph_with_remote_node(url: str):
    """Root graph whose node is the remote graph."""
    builder = StateGraph(ParentState)
    builder.add_node("remote", _remote(url))
    builder.add_node("privileged", _privileged)
    builder.add_node("after", _after)
    builder.add_edge(START, "remote")
    builder.add_edge("remote", "after")
    builder.add_edge("privileged", END)
    builder.add_edge("after", END)
    return builder.compile(checkpointer=InMemorySaver())


def _graph_with_remote_subgraph(url: str):
    """Outer graph whose child subgraph has the remote graph as a node.

    `Command.PARENT` from that remote node is rewritten onto the outer task.
    """
    child = StateGraph(ChildState)
    child.add_node("remote", _remote(url))
    child.add_edge(START, "remote")
    subgraph = child.compile()

    builder = StateGraph(ParentState)
    builder.add_node("child", subgraph)
    builder.add_node("privileged", _privileged)
    builder.add_node("after", _after)
    builder.add_edge(START, "child")
    builder.add_edge("child", "after")
    builder.add_edge("privileged", END)
    builder.add_edge("after", END)
    return builder.compile(checkpointer=InMemorySaver())


def _privileged(state: ParentState) -> dict[str, Any]:
    return {"ran_privileged": True, "secret": state["secret"] + "|privileged"}


def _after(state: ParentState) -> dict[str, Any]:
    return {"ran_after": True, "public": state["public"] + "|after"}


def _initial() -> ParentState:
    return {
        "public": "visible",
        "secret": _ORIGINAL_SECRET,
        "ran_privileged": False,
        "ran_after": False,
    }


async def _invoke(graph, *, sync: bool, thread_id: str) -> Any:
    config = {"configurable": {"thread_id": thread_id}}
    try:
        if sync:
            result = graph.invoke(_initial(), config)
        else:
            result = await graph.ainvoke(_initial(), config)
    except (ParentCommand, GraphInterrupt) as exc:
        pytest.fail(f"remote stream event escaped into the parent runner: {exc!r}")
    except Exception:
        result = None
    snapshot = graph.get_state(config)
    assert snapshot.values["secret"] == _ORIGINAL_SECRET
    assert snapshot.values["ran_privileged"] is False
    assert snapshot.interrupts == ()
    if result is not None:
        assert result["secret"] == _ORIGINAL_SECRET
        assert result["ran_privileged"] is False
        assert result["ran_after"] is True
    return result


@pytest.mark.anyio
@pytest.mark.parametrize("sync", [True, False], ids=["stream", "astream"])
@pytest.mark.parametrize("event_name", ["command", "command|remote"])
async def test_remote_parent_command_does_not_steer_parent(
    remote_url: tuple[str, _Script], sync: bool, event_name: str
) -> None:
    url, script = remote_url
    script.events = _parent_command_events(event_name)
    graph = _graph_with_remote_subgraph(url)
    await _invoke(graph, sync=sync, thread_id=f"command-{sync}-{event_name}")
    assert script.calls == 1


@pytest.mark.anyio
@pytest.mark.parametrize("sync", [True, False], ids=["stream", "astream"])
@pytest.mark.parametrize("event_name", ["command", "command|remote"])
async def test_root_remote_command_does_not_escape(
    remote_url: tuple[str, _Script], sync: bool, event_name: str
) -> None:
    url, script = remote_url
    script.events = _parent_command_events(event_name)
    graph = _graph_with_remote_node(url)
    await _invoke(graph, sync=sync, thread_id=f"root-{sync}-{event_name}")
    assert script.calls == 1


@pytest.mark.anyio
@pytest.mark.parametrize("sync", [True, False], ids=["stream", "astream"])
@pytest.mark.parametrize("event_name", ["updates", "updates|remote"])
async def test_remote_interrupt_does_not_pause_parent(
    remote_url: tuple[str, _Script], sync: bool, event_name: str
) -> None:
    url, script = remote_url
    script.events = _interrupt_events(event_name)
    graph = _graph_with_remote_node(url)
    await _invoke(graph, sync=sync, thread_id=f"interrupt-{sync}-{event_name}")
    assert script.calls == 1
    snapshot = graph.get_state(
        {"configurable": {"thread_id": f"interrupt-{sync}-{event_name}"}}
    )
    assert all(
        interrupt.id != _FORGED_INTERRUPT_ID for interrupt in snapshot.interrupts
    )
    assert all(
        interrupt.value != _FORGED_INTERRUPT_VALUE for interrupt in snapshot.interrupts
    )
