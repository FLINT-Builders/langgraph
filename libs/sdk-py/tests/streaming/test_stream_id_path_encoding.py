"""Caller-supplied thread and assistant ids on the v3 stream helpers must stay
one encoded path segment.

`client.threads.stream` forwards those ids into `thread.output`,
`thread.values`, and `thread.agent.get_tree` (async and sync). Those helpers
build a GET path. The path that httpx actually sends, and the JSON the helper
returns, must be the encoded thread-state or assistant-graph resource — not a
path produced by leaving the id unencoded.

These tests fail until both ids are passed through `_quote_path_param` in the
async and sync stream modules.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import httpx
import pytest

from langgraph_sdk._async.client import LangGraphClient
from langgraph_sdk._shared.utilities import _get_headers, _quote_path_param
from langgraph_sdk._sync.client import SyncLangGraphClient

# Obvious placeholder. Not a credential.
_API_KEY = "placeholder-api-key"
_BASE_URL = "http://langgraph.test"

# Ids that include a slash, dot segments, and a query or fragment marker.
# A raw interpolation lets the HTTP stack retarget the GET.
_THREAD_IDS = [
    "../../store/items?namespace=memories&key=note#",
    "../../store/items?namespace=memories&key=note?",
    "other-thread/state#",
]
_ASSISTANT_IDS = [
    "../../store/items?namespace=memories&key=note#",
    "../../store/items?namespace=memories&key=note?",
    "other-assistant/graph#",
]

_INTENDED_VALUES = {"source": "intended-thread"}
_FOREIGN_VALUES = {"source": "foreign-endpoint"}
_INTENDED_GRAPH = {"nodes": [{"id": "intended-assistant"}]}
_FOREIGN_GRAPH = {
    "namespace": ["memories"],
    "key": "note",
    "value": {"secret": "foreign-graph"},
}


def _thread_state_path(thread_id: str) -> str:
    return f"/threads/{_quote_path_param(thread_id)}/state"


def _assistant_graph_path(assistant_id: str) -> str:
    return f"/assistants/{_quote_path_param(assistant_id)}/graph"


@dataclass(frozen=True)
class _Captured:
    method: str
    raw_path: str
    api_key: str | None


def _handler(
    captured: list[_Captured],
    *,
    thread_id: str,
    assistant_id: str,
) -> Callable[[httpx.Request], httpx.Response]:
    """Serve the encoded resources and a different body for every other GET.

    POST /commands and the lifecycle SSE are answered so entering the stream
    and, where required, `run.start` can complete. They are not the requests
    under test.
    """
    state_path = _thread_state_path(thread_id)
    graph_path = _assistant_graph_path(assistant_id)

    def handler(request: httpx.Request) -> httpx.Response:
        raw_path = request.url.raw_path.decode("ascii")
        captured.append(
            _Captured(
                method=request.method,
                raw_path=raw_path,
                api_key=request.headers.get("x-api-key"),
            )
        )
        if request.method == "POST" and raw_path.endswith("/commands"):
            return httpx.Response(
                200,
                json={"id": 1, "type": "success", "result": {"run_id": "run-1"}},
            )
        if request.method == "POST":
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=b"",
            )
        if raw_path == state_path:
            body: dict = {
                "values": _INTENDED_VALUES,
                "next": [],
                "tasks": [],
            }
        elif raw_path == graph_path:
            body = dict(_INTENDED_GRAPH)
        elif raw_path.split("?", 1)[0].endswith("/graph"):
            body = dict(_FOREIGN_GRAPH)
        else:
            body = {"values": _FOREIGN_VALUES, "next": [], "tasks": []}
        return httpx.Response(200, json=body)

    return handler


def _gets(captured: list[_Captured]) -> list[_Captured]:
    return [item for item in captured if item.method == "GET"]


def _assert_single_get(captured: list[_Captured], expected_path: str) -> _Captured:
    gets = _gets(captured)
    assert [item.raw_path for item in gets] == [expected_path], captured
    assert gets[0].api_key == _API_KEY
    segment = expected_path.split("/")[2]
    assert "/" not in segment
    assert "?" not in segment
    assert "#" not in segment
    return gets[0]


@pytest.mark.parametrize("thread_id", _THREAD_IDS)
async def test_async_output_stays_on_encoded_thread_state(thread_id: str) -> None:
    """Explicit thread id, no run: `thread.output` GETs that thread's state."""
    captured: list[_Captured] = []
    transport = httpx.MockTransport(
        _handler(captured, thread_id=thread_id, assistant_id="agent")
    )
    async with httpx.AsyncClient(
        transport=transport,
        base_url=_BASE_URL,
        headers=_get_headers(_API_KEY, None),
        trust_env=False,
    ) as raw:
        client = LangGraphClient(raw)
        async with client.threads.stream(
            thread_id=thread_id, assistant_id="agent"
        ) as thread:
            result = await thread.output

    _assert_single_get(captured, _thread_state_path(thread_id))
    assert result == _INTENDED_VALUES


@pytest.mark.parametrize("thread_id", _THREAD_IDS)
async def test_async_values_stays_on_encoded_thread_state(thread_id: str) -> None:
    """The first `thread.values` snapshot is the same thread-state GET."""
    captured: list[_Captured] = []
    transport = httpx.MockTransport(
        _handler(captured, thread_id=thread_id, assistant_id="agent")
    )
    async with httpx.AsyncClient(
        transport=transport,
        base_url=_BASE_URL,
        headers=_get_headers(_API_KEY, None),
        trust_env=False,
    ) as raw:
        client = LangGraphClient(raw)
        async with client.threads.stream(
            thread_id=thread_id, assistant_id="agent"
        ) as thread:
            result = None
            async for snapshot in thread.values:
                result = snapshot
                break

    _assert_single_get(captured, _thread_state_path(thread_id))
    assert result == _INTENDED_VALUES


@pytest.mark.parametrize("assistant_id", _ASSISTANT_IDS)
async def test_async_graph_tree_stays_on_encoded_assistant(
    assistant_id: str,
) -> None:
    """`thread.agent.get_tree` GETs that assistant's graph and returns it."""
    captured: list[_Captured] = []
    transport = httpx.MockTransport(
        _handler(captured, thread_id="thread-1", assistant_id=assistant_id)
    )
    async with httpx.AsyncClient(
        transport=transport,
        base_url=_BASE_URL,
        headers=_get_headers(_API_KEY, None),
        trust_env=False,
    ) as raw:
        client = LangGraphClient(raw)
        async with client.threads.stream(
            thread_id="thread-1", assistant_id=assistant_id
        ) as thread:
            result = await thread.agent.get_tree()

    _assert_single_get(captured, _assistant_graph_path(assistant_id))
    assert result == _INTENDED_GRAPH


@pytest.mark.parametrize("thread_id", _THREAD_IDS)
def test_sync_output_stays_on_encoded_thread_state(thread_id: str) -> None:
    captured: list[_Captured] = []
    transport = httpx.MockTransport(
        _handler(captured, thread_id=thread_id, assistant_id="agent")
    )
    with httpx.Client(
        transport=transport,
        base_url=_BASE_URL,
        headers=_get_headers(_API_KEY, None),
        trust_env=False,
    ) as raw:
        client = SyncLangGraphClient(raw)
        with client.threads.stream(thread_id=thread_id, assistant_id="agent") as thread:
            result = thread.output

    _assert_single_get(captured, _thread_state_path(thread_id))
    assert result == _INTENDED_VALUES


@pytest.mark.parametrize("thread_id", _THREAD_IDS)
def test_sync_values_stays_on_encoded_thread_state(thread_id: str) -> None:
    """Sync `thread.values` fetches state after `run.start` opens the gate."""
    captured: list[_Captured] = []
    transport = httpx.MockTransport(
        _handler(captured, thread_id=thread_id, assistant_id="agent")
    )
    with httpx.Client(
        transport=transport,
        base_url=_BASE_URL,
        headers=_get_headers(_API_KEY, None),
        trust_env=False,
    ) as raw:
        client = SyncLangGraphClient(raw)
        with client.threads.stream(thread_id=thread_id, assistant_id="agent") as thread:
            thread.run.start(input={})
            result = None
            for snapshot in thread.values:
                result = snapshot
                break

    _assert_single_get(captured, _thread_state_path(thread_id))
    assert result == _INTENDED_VALUES


@pytest.mark.parametrize("assistant_id", _ASSISTANT_IDS)
def test_sync_graph_tree_stays_on_encoded_assistant(assistant_id: str) -> None:
    captured: list[_Captured] = []
    transport = httpx.MockTransport(
        _handler(captured, thread_id="thread-1", assistant_id=assistant_id)
    )
    with httpx.Client(
        transport=transport,
        base_url=_BASE_URL,
        headers=_get_headers(_API_KEY, None),
        trust_env=False,
    ) as raw:
        client = SyncLangGraphClient(raw)
        with client.threads.stream(
            thread_id="thread-1", assistant_id=assistant_id
        ) as thread:
            result = thread.agent.get_tree()

    _assert_single_get(captured, _assistant_graph_path(assistant_id))
    assert result == _INTENDED_GRAPH


async def test_async_plain_thread_id_output_is_unchanged() -> None:
    """A normal id still addresses `/threads/{id}/state` and returns its values."""
    thread_id = "thread-1"
    captured: list[_Captured] = []
    transport = httpx.MockTransport(
        _handler(captured, thread_id=thread_id, assistant_id="agent")
    )
    async with httpx.AsyncClient(
        transport=transport,
        base_url=_BASE_URL,
        headers=_get_headers(_API_KEY, None),
        trust_env=False,
    ) as raw:
        client = LangGraphClient(raw)
        async with client.threads.stream(
            thread_id=thread_id, assistant_id="agent"
        ) as thread:
            result = await thread.output

    _assert_single_get(captured, "/threads/thread-1/state")
    assert result == _INTENDED_VALUES
