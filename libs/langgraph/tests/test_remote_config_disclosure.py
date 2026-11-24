"""Sensitive run-config values must not be sent to a remote graph.

These tests fail while `RemoteGraph` copies string and number entries from
`configurable` and `metadata` into the outbound run request. They pass once
those entries are removed before the request is built.

The placeholder below is not a credential. It only marks a value that must
not leave the process.
"""

from __future__ import annotations

import asyncio
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import orjson
import pytest
from typing_extensions import TypedDict

from langgraph.graph import START, StateGraph
from langgraph.pregel.remote import RemoteGraph

PLACEHOLDER = "PLACEHOLDER_CONFIG_VALUE"
NUMERIC_SENTINEL = 424242

_SENSITIVE_CONFIGURABLE: dict[str, Any] = {
    "model_name": "example-model",
    "api_token": PLACEHOLDER,
    "openai_api_key": PLACEHOLDER,
    "db_password": PLACEHOLDER,
    "client_secret": PLACEHOLDER,
    "authorization": PLACEHOLDER,
    "API_TOKEN": PLACEHOLDER,
    "pin_password": NUMERIC_SENTINEL,
    "wrapped": {"access_token": PLACEHOLDER},
    "api_keys": [PLACEHOLDER],
    "thread_id": "thread-under-test",
    "checkpoint_id": "checkpoint-under-test",
    "checkpoint_ns": "ns-under-test",
    "checkpoint_map": {"ns-under-test": "checkpoint-under-test"},
}

_SENSITIVE_METADATA: dict[str, Any] = {
    "run_label": "example-run",
    "user_token": PLACEHOLDER,
    "session_secret": PLACEHOLDER,
}


class _State(TypedDict):
    text: str


class _RecordingServer(ThreadingHTTPServer):
    bodies: list[tuple[str, bytes]]


class _RecordingHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: Any) -> None:
        return

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        server = self.server
        assert isinstance(server, _RecordingServer)
        server.bodies.append((self.path, body))

        if self.path.endswith("/runs/stream"):
            payload = b'event: values\ndata: {"text":"ok"}\n\n'
            self._write(200, "text/event-stream", payload)
            return
        if self.path.endswith("/commands"):
            command = orjson.loads(body)
            payload = orjson.dumps(
                {
                    "id": command.get("id"),
                    "result": {"run_id": "run-under-test"},
                }
            )
            self._write(200, "application/json", payload)
            return
        if self.path.endswith("/stream/events"):
            self._write(200, "text/event-stream", b": keepalive\n\n")
            return
        self._write(404, "text/plain", b"not found")

    def _write(self, status: int, content_type: str, payload: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(payload)


@pytest.fixture
def remote_url() -> Any:
    server = _RecordingServer(("127.0.0.1", 0), _RecordingHandler)
    server.bodies = []
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    try:
        yield server, f"http://{host}:{port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def _caller_config() -> dict[str, Any]:
    return {
        "configurable": dict(_SENSITIVE_CONFIGURABLE),
        "metadata": dict(_SENSITIVE_METADATA),
    }


def _remote(url: str, *, config: dict[str, Any] | None = None) -> RemoteGraph:
    return RemoteGraph(
        "assistant-under-test",
        url=url,
        api_key=None,
        config=config,
    )


def _parent(remote: RemoteGraph) -> Any:
    return (
        StateGraph(_State)
        .add_node("remote", remote)
        .add_edge(START, "remote")
        .compile()
    )


def _assert_not_disclosed(bodies: list[tuple[str, bytes]]) -> None:
    assert bodies, "the remote server received no request"
    leaked_paths = [
        path
        for path, body in bodies
        if PLACEHOLDER.encode() in body or str(NUMERIC_SENTINEL).encode() in body
    ]
    assert leaked_paths == []


def test_parent_invoke_does_not_send_sensitive_config(remote_url: Any) -> None:
    server, url = remote_url
    graph = _parent(_remote(url))
    graph.invoke({"text": "hi"}, _caller_config())
    _assert_not_disclosed(server.bodies)


def test_parent_ainvoke_does_not_send_sensitive_config(remote_url: Any) -> None:
    server, url = remote_url
    graph = _parent(_remote(url))

    async def _run() -> None:
        await graph.ainvoke({"text": "hi"}, _caller_config())

    asyncio.run(_run())
    _assert_not_disclosed(server.bodies)


def test_remote_invoke_does_not_send_sensitive_config(remote_url: Any) -> None:
    server, url = remote_url
    _remote(url).invoke({"text": "hi"}, _caller_config())
    _assert_not_disclosed(server.bodies)


def test_remote_stream_does_not_send_sensitive_config(remote_url: Any) -> None:
    server, url = remote_url
    list(_remote(url).stream({"text": "hi"}, _caller_config()))
    _assert_not_disclosed(server.bodies)


def test_remote_ainvoke_does_not_send_sensitive_config(remote_url: Any) -> None:
    server, url = remote_url
    remote = _remote(url)

    async def _run() -> None:
        await remote.ainvoke({"text": "hi"}, _caller_config())

    asyncio.run(_run())
    _assert_not_disclosed(server.bodies)


def test_remote_astream_does_not_send_sensitive_config(remote_url: Any) -> None:
    server, url = remote_url
    remote = _remote(url)

    async def _run() -> None:
        async for _chunk in remote.astream({"text": "hi"}, _caller_config()):
            pass

    asyncio.run(_run())
    _assert_not_disclosed(server.bodies)


def test_remote_stream_events_v3_does_not_send_sensitive_config(
    remote_url: Any,
) -> None:
    server, url = remote_url
    with _remote(url).stream_events(
        {"text": "hi"},
        _caller_config(),
        version="v3",
    ):
        pass
    _assert_not_disclosed(server.bodies)


def test_remote_astream_events_v3_does_not_send_sensitive_config(
    remote_url: Any,
) -> None:
    server, url = remote_url
    remote = _remote(url)

    async def _run() -> None:
        async with await remote.astream_events(
            {"text": "hi"},
            _caller_config(),
            version="v3",
        ):
            pass

    asyncio.run(_run())
    _assert_not_disclosed(server.bodies)


def test_constructor_config_is_not_sent(remote_url: Any) -> None:
    server, url = remote_url
    remote = _remote(url, config=_caller_config())
    remote.invoke({"text": "hi"}, {"configurable": {"model_name": "example-model"}})
    _assert_not_disclosed(server.bodies)


def test_with_config_is_not_sent(remote_url: Any) -> None:
    server, url = remote_url
    remote = _remote(url).with_config(_caller_config())
    remote.invoke({"text": "hi"})
    _assert_not_disclosed(server.bodies)
