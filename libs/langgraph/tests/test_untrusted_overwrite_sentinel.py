"""Caller-supplied JSON must not replace reducer-channel state.

`Overwrite` instances returned by graph code are a trusted API. A plain dict
that merely looks like one — `{"type": "__overwrite__", "value": ...}` or
`{"__overwrite__": ...}` — is ordinary channel data when it arrives as graph
input, as a `Command` update, or as `update_state` values. Those calls are
how a client JSON body is applied (the LangGraph API server forwards `input`
and `command` through to `invoke` / `astream` / `aupdate_state`).

These tests fail while that dict is still treated as an overwrite. They pass
once those caller-supplied shapes no longer replace the stored value. A
refusal must leave the previous checkpoint intact, and a node that already
ran must have seen that same intact state.
"""

from __future__ import annotations

import json
import operator
from typing import Annotated, Any, Literal
from uuid import uuid4

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from typing_extensions import TypedDict

from langgraph.channels.delta import DeltaChannel
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import _messages_delta_reducer, add_messages
from langgraph.types import Command

pytestmark = pytest.mark.anyio

Shape = Literal["type_tag", "type_tag_extra", "single_key"]
Channel = Literal["messages", "notes", "budget", "profile"]
Kind = Literal["binop", "delta"]
How = Literal[
    "invoke",
    "stream",
    "command",
    "update_state",
    "update_state_as_input",
    "ainvoke",
    "astream",
    "acommand",
    "aupdate_state",
    "aupdate_state_as_input",
]

_UPDATE_HOWS = {
    "update_state",
    "update_state_as_input",
    "aupdate_state",
    "aupdate_state_as_input",
}


def _sentinel(shape: Shape, value: Any) -> dict[str, Any]:
    """Build a JSON object the channel currently treats as Overwrite."""
    if shape == "type_tag":
        body: dict[str, Any] = {"type": "__overwrite__", "value": value}
    elif shape == "type_tag_extra":
        # Extra keys still match `_get_overwrite` (`"value" in value`).
        body = {
            "type": "__overwrite__",
            "value": value,
            "source": "client-json",
        }
    elif shape == "single_key":
        body = {"__overwrite__": value}
    else:
        raise AssertionError(shape)
    return json.loads(json.dumps(body))


def _merge(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    merged = dict(left)
    merged.update(right)
    return merged


def _notes_delta(state: list[str], writes: list[Any]) -> list[str]:
    items = list(state)
    for write in writes:
        if isinstance(write, list):
            items.extend(write)
        else:
            items.append(write)
    return items


def _budget_delta(state: int, writes: list[int]) -> int:
    total = state
    for write in writes:
        total += write
    return total


def _profile_delta(
    state: dict[str, Any], writes: list[dict[str, Any]]
) -> dict[str, Any]:
    merged = dict(state)
    for write in writes:
        merged.update(write)
    return merged


def _seed() -> dict[str, Any]:
    return {
        "messages": [
            {"role": "system", "content": "SECRET_SYSTEM", "id": "sys-1"},
            {
                "role": "tool",
                "content": "SECRET_TOOL",
                "tool_call_id": "call-1",
                "id": "tool-1",
            },
            {"role": "user", "content": "PARTICIPANT_A", "id": "user-a"},
        ],
        "notes": ["system-note", "tool-note"],
        "budget": 7,
        "profile": {"role": "admin", "name": "alice"},
    }


def _attack(shape: Shape, channel: Channel) -> dict[str, Any]:
    replacement: dict[Channel, Any] = {
        "messages": [
            {"role": "user", "content": "ATTACK_ONLY", "id": "atk"},
        ],
        "notes": ["attacker-note"],
        "budget": 0,
        "profile": {"role": "user"},
    }
    return {channel: _sentinel(shape, replacement[channel])}


def _follow_up() -> dict[str, Any]:
    """A later participant turn. Reducers must merge this, not replace."""
    return {
        "messages": [
            {"role": "user", "content": "PARTICIPANT_B", "id": "user-b"},
        ],
        "notes": ["participant-b-note"],
        "budget": 1,
        "profile": {"title": "member"},
    }


def _message_contents(messages: list[Any]) -> list[Any]:
    contents: list[Any] = []
    for message in messages:
        if isinstance(message, dict):
            contents.append(message.get("content"))
        else:
            contents.append(getattr(message, "content", None))
    return contents


def _intact(channel: Channel, snapshot: dict[str, Any]) -> bool:
    if channel == "messages":
        contents = _message_contents(snapshot["messages"])
        return (
            "SECRET_SYSTEM" in contents
            and "SECRET_TOOL" in contents
            and "PARTICIPANT_A" in contents
        )
    if channel == "notes":
        notes = list(snapshot["notes"])
        return notes[:2] == ["system-note", "tool-note"]
    if channel == "budget":
        return snapshot["budget"] >= 7
    profile = snapshot["profile"]
    return profile.get("name") == "alice" and profile.get("role") == "admin"


def _compile(kind: Kind, observed: list[dict[str, Any]]):
    if kind == "binop":

        class State(TypedDict):
            messages: Annotated[list, add_messages]
            notes: Annotated[list, operator.add]
            budget: Annotated[int, operator.add]
            profile: Annotated[dict, _merge]

    else:

        class State(TypedDict):
            messages: Annotated[list, DeltaChannel(_messages_delta_reducer)]
            notes: Annotated[list, DeltaChannel(_notes_delta)]
            budget: Annotated[int, DeltaChannel(_budget_delta)]
            profile: Annotated[dict, DeltaChannel(_profile_delta)]

    def agent(state: State) -> dict[str, Any]:
        observed.append(
            {
                "messages": list(state["messages"]),
                "notes": list(state["notes"]),
                "budget": state["budget"],
                "profile": dict(state["profile"]),
            }
        )
        return {
            "messages": [
                {
                    "role": "assistant",
                    "content": "from-agent",
                    "id": f"agent-{len(observed)}",
                }
            ]
        }

    graph = StateGraph(State)
    graph.add_node("agent", agent)
    graph.add_edge(START, "agent")
    graph.add_edge("agent", END)
    return graph.compile(checkpointer=InMemorySaver())


def _assert_channel_preserved(
    channel: Channel,
    values: dict[str, Any],
    observed: list[dict[str, Any]],
) -> None:
    snapshots = [*observed, values]
    assert snapshots, "expected a checkpoint to read back"
    for snapshot in snapshots:
        assert _intact(channel, snapshot), (channel, snapshot)


async def _apply(graph, config: dict, payload: dict | None, how: How) -> None:
    if how == "invoke":
        graph.invoke(payload, config)
    elif how == "stream":
        list(graph.stream(payload, config))
    elif how == "command":
        graph.invoke(Command(update=payload), config)
    elif how == "update_state":
        graph.update_state(config, payload)
    elif how == "update_state_as_input":
        graph.update_state(config, payload, as_node="__input__")
    elif how == "ainvoke":
        await graph.ainvoke(payload, config)
    elif how == "astream":
        async for _chunk in graph.astream(payload, config):
            pass
    elif how == "acommand":
        await graph.ainvoke(Command(update=payload), config)
    elif how == "aupdate_state":
        await graph.aupdate_state(config, payload)
    elif how == "aupdate_state_as_input":
        await graph.aupdate_state(config, payload, as_node="__input__")
    else:
        raise AssertionError(how)


async def _exercise(
    kind: Kind,
    channel: Channel,
    shape: Shape,
    how: How,
    *,
    payload: dict[str, Any] | None = None,
) -> None:
    observed: list[dict[str, Any]] = []
    graph = _compile(kind, observed)
    config = {"configurable": {"thread_id": str(uuid4())}}
    await _apply(graph, config, _seed(), "ainvoke" if how.startswith("a") else "invoke")
    seeded = graph.get_state(config).values
    assert _intact(channel, seeded), seeded
    observed.clear()

    attack = payload if payload is not None else _attack(shape, channel)
    try:
        await _apply(graph, config, attack, how)
    except Exception:
        # Refusal is only successful when nothing was replaced first.
        pass

    if how in _UPDATE_HOWS:
        # `update_state` does not run the node. Read the checkpoint back
        # through the next execution. `as_node="__input__"` only writes the
        # start channel; the sentinel is fanned out to reducers on resume.
        observed.clear()
        follow_how = "ainvoke" if how.startswith("a") else "invoke"
        follow_payload = (
            None
            if how in {"update_state_as_input", "aupdate_state_as_input"}
            else _follow_up()
        )
        try:
            await _apply(graph, config, follow_payload, follow_how)
        except Exception:
            pass

    _assert_channel_preserved(channel, graph.get_state(config).values, observed)


@pytest.mark.parametrize("kind", ["binop", "delta"])
@pytest.mark.parametrize("channel", ["messages", "notes", "budget", "profile"])
@pytest.mark.parametrize("shape", ["type_tag", "type_tag_extra", "single_key"])
@pytest.mark.parametrize(
    "how",
    ["invoke", "stream", "command", "update_state", "update_state_as_input"],
)
async def test_caller_json_sentinel_does_not_replace_reducer_state(
    kind: Kind, channel: Channel, shape: Shape, how: How
) -> None:
    await _exercise(kind, channel, shape, how)


@pytest.mark.parametrize("kind", ["binop", "delta"])
@pytest.mark.parametrize("channel", ["messages", "budget"])
@pytest.mark.parametrize("shape", ["type_tag", "type_tag_extra", "single_key"])
@pytest.mark.parametrize(
    "how",
    ["ainvoke", "astream", "acommand", "aupdate_state", "aupdate_state_as_input"],
)
async def test_async_caller_json_sentinel_does_not_replace_reducer_state(
    kind: Kind, channel: Channel, shape: Shape, how: How
) -> None:
    await _exercise(kind, channel, shape, how)


@pytest.mark.parametrize("kind", ["binop", "delta"])
async def test_legitimate_follow_up_keeps_prior_history(kind: Kind) -> None:
    """A normal second turn merges. This is the behavior the sentinel bypasses."""
    observed: list[dict[str, Any]] = []
    graph = _compile(kind, observed)
    config = {"configurable": {"thread_id": str(uuid4())}}
    graph.invoke(_seed(), config)
    observed.clear()
    graph.invoke(_follow_up(), config)
    values = graph.get_state(config).values
    for channel in ("messages", "notes", "budget", "profile"):
        _assert_channel_preserved(channel, values, observed)
    assert "PARTICIPANT_B" in _message_contents(values["messages"])
    assert values["notes"][-1] == "participant-b-note"
    assert values["budget"] == 8
    assert values["profile"]["title"] == "member"


@pytest.mark.parametrize("kind", ["binop", "delta"])
async def test_profile_dict_with_type_and_value_is_not_an_overwrite(
    kind: Kind,
) -> None:
    """A dict that merely has `type` and `value` keys is channel data."""
    observed: list[dict[str, Any]] = []
    graph = _compile(kind, observed)
    config = {"configurable": {"thread_id": str(uuid4())}}
    graph.invoke(_seed(), config)
    observed.clear()
    graph.invoke(
        {"profile": {"type": "human", "value": "hi"}},
        config,
    )
    values = graph.get_state(config).values
    _assert_channel_preserved("profile", values, observed)
    assert values["profile"]["type"] == "human"
    assert values["profile"]["value"] == "hi"
