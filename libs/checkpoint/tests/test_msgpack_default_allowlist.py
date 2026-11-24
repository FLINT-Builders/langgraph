"""Spec for the default msgpack constructor allowlist.

These tests fail while `JsonPlusSerializer` treats an omitted allowlist as
"construct every type". They pass once the default refuses unlisted
constructors, returns the raw constructor argument, and does not call the
named object. `allowed_msgpack_modules=True` remains an explicit opt-in
that still performs the call.

The callable used here records that it ran. A fix that calls it and then
discards the return value still fails.
"""

from __future__ import annotations

import os
from typing import Any

import ormsgpack
import pytest

from langgraph.cache.base import FullKey
from langgraph.cache.memory import InMemoryCache
from langgraph.cache.redis import RedisCache
from langgraph.checkpoint.base import empty_checkpoint
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde import _msgpack as _lg_msgpack
from langgraph.checkpoint.serde.encrypted import EncryptedSerializer
from langgraph.checkpoint.serde.jsonplus import (
    EXT_CONSTRUCTOR_KW_ARGS,
    EXT_CONSTRUCTOR_POS_ARGS,
    EXT_CONSTRUCTOR_SINGLE_ARG,
    EXT_METHOD_SINGLE_ARG,
    EXT_PYDANTIC_V1,
    EXT_PYDANTIC_V2,
    JsonPlusSerializer,
    _msgpack_enc,
)

_MODULE = __name__
_CALLS: list[tuple[Any, ...]] = []

# Distinct from the callable's return value, so a successful call cannot
# be mistaken for the raw argument the hook is supposed to hand back.
_SINGLE_ARG = "marker-single"
_POS_ARGS = ["marker-pos"]
_KWARGS = {"value": "marker-kw"}
_METHOD_ARG = "marker-method"


def single_target(arg: object) -> str:
    _CALLS.append(("single", arg))
    return "executed-single"


def pos_target(*args: object) -> str:
    _CALLS.append(("pos", args))
    return "executed-pos"


def kw_target(**kwargs: object) -> str:
    _CALLS.append(("kw", tuple(sorted(kwargs.items()))))
    return "executed-kw"


class _IdentityCipher:
    def encrypt(self, plaintext: bytes) -> tuple[str, bytes]:
        return "identity", plaintext

    def decrypt(self, ciphername: str, ciphertext: bytes) -> bytes:
        return ciphertext


class _StoredBytesRedis:
    """Stand-in for `Redis.mget` from redis-py 8.1.0.

    `Redis.mget` (`redis/commands/core.py`) returns values in key order.
    With the default `decode_responses=False` (`redis/client.py`), the RESP2
    parser (`redis/_parsers/resp2.py`) hands bulk strings to `Encoder.decode`
    (`redis/_parsers/encoders.py`), which returns those bytes unchanged.
    """

    def __init__(self) -> None:
        self.stored: dict[str, bytes] = {}

    def mget(self, keys: list[str]) -> list[bytes | None]:
        return [self.stored.get(key) for key in keys]


def _ext(code: int, body: tuple[Any, ...]) -> bytes:
    return ormsgpack.packb(
        ormsgpack.Ext(code, _msgpack_enc(body)),
        option=ormsgpack.OPT_NON_STR_KEYS,
    )


def _pos_payload() -> bytes:
    return _ext(EXT_CONSTRUCTOR_POS_ARGS, (_MODULE, "pos_target", _POS_ARGS))


def _default_serializer() -> JsonPlusSerializer:
    """Build the serializer the way an unset strict-mode flag does."""
    assert os.environ.get("LANGGRAPH_STRICT_MSGPACK") is None
    previous = _lg_msgpack.STRICT_MSGPACK_ENABLED
    _lg_msgpack.STRICT_MSGPACK_ENABLED = False
    try:
        return JsonPlusSerializer()
    finally:
        _lg_msgpack.STRICT_MSGPACK_ENABLED = previous


def _assert_blocked(result: Any, raw: Any) -> None:
    assert _CALLS == []
    assert result == raw


@pytest.mark.parametrize(
    ("code", "body", "raw"),
    [
        (
            EXT_CONSTRUCTOR_SINGLE_ARG,
            (_MODULE, "single_target", _SINGLE_ARG),
            _SINGLE_ARG,
        ),
        (
            EXT_CONSTRUCTOR_POS_ARGS,
            (_MODULE, "pos_target", _POS_ARGS),
            _POS_ARGS,
        ),
        (
            EXT_CONSTRUCTOR_KW_ARGS,
            (_MODULE, "kw_target", _KWARGS),
            _KWARGS,
        ),
        (
            EXT_PYDANTIC_V1,
            (_MODULE, "kw_target", _KWARGS),
            _KWARGS,
        ),
        (
            EXT_PYDANTIC_V2,
            (_MODULE, "kw_target", _KWARGS),
            _KWARGS,
        ),
    ],
)
def test_default_constructor_extension_does_not_call(
    code: int, body: tuple[Any, ...], raw: Any
) -> None:
    serde = _default_serializer()
    _CALLS.clear()
    result = serde.loads_typed(("msgpack", _ext(code, body)))
    _assert_blocked(result, raw)


def test_default_nested_extension_does_not_call() -> None:
    serde = _default_serializer()
    _CALLS.clear()
    payload = ormsgpack.packb(
        {
            "channel_values": {
                "chan": ormsgpack.Ext(
                    EXT_CONSTRUCTOR_POS_ARGS,
                    _msgpack_enc((_MODULE, "pos_target", _POS_ARGS)),
                )
            }
        },
        option=ormsgpack.OPT_NON_STR_KEYS,
    )
    result = serde.loads_typed(("msgpack", payload))
    _assert_blocked(result["channel_values"]["chan"], _POS_ARGS)


def test_default_method_extension_does_not_call() -> None:
    """Method extensions are not on the permissive constructor path."""
    serde = _default_serializer()
    _CALLS.clear()
    payload = _ext(
        EXT_METHOD_SINGLE_ARG,
        (_MODULE, "single_target", _METHOD_ARG, "single_target"),
    )
    result = serde.loads_typed(("msgpack", payload))
    _assert_blocked(result, _METHOD_ARG)


def test_explicit_allow_all_still_calls() -> None:
    serde = JsonPlusSerializer(allowed_msgpack_modules=True)
    _CALLS.clear()
    result = serde.loads_typed(("msgpack", _pos_payload()))
    assert _CALLS == [("pos", ("marker-pos",))]
    assert result == "executed-pos"


def test_explicit_allowlist_entry_still_calls() -> None:
    serde = JsonPlusSerializer(allowed_msgpack_modules=[(_MODULE, "pos_target")])
    _CALLS.clear()
    result = serde.loads_typed(("msgpack", _pos_payload()))
    assert _CALLS == [("pos", ("marker-pos",))]
    assert result == "executed-pos"


def test_default_serializer_still_constructs_safe_types() -> None:
    serde = _default_serializer()
    value = {1, 2, 3}
    assert serde.loads_typed(serde.dumps_typed(value)) == value


def test_encrypted_passthrough_uses_default_allowlist() -> None:
    serde = EncryptedSerializer(_IdentityCipher())
    _CALLS.clear()
    result = serde.loads_typed(("msgpack", _pos_payload()))
    _assert_blocked(result, _POS_ARGS)


def test_inmemory_saver_loads_do_not_call() -> None:
    saver = InMemorySaver()
    parent = empty_checkpoint()
    parent["id"] = "0002-parent"
    parent["channel_values"] = {"chan": "plain"}
    parent["channel_versions"] = {"chan": "1"}
    child = empty_checkpoint()
    child["id"] = "0003-child"
    thread = "thread-1"
    ns = ""
    base = {"configurable": {"thread_id": thread, "checkpoint_ns": ns}}
    saver.put(base, parent, {"source": "input", "step": -1}, {"chan": "1"})
    saver.put(
        {
            **base,
            "configurable": {**base["configurable"], "checkpoint_id": parent["id"]},
        },
        child,
        {"source": "loop", "step": 0},
        {},
    )
    payload = _pos_payload()
    saver.blobs[(thread, ns, "chan", "1")] = ("msgpack", payload)
    saved_parent = {
        "configurable": {
            "thread_id": thread,
            "checkpoint_ns": ns,
            "checkpoint_id": parent["id"],
        }
    }
    saver.put_writes(saved_parent, [("chan", "plain")], "task-1")
    outer = saver.writes[(thread, ns, parent["id"])]
    for key, (task_id, channel, _serialized, task_path) in list(outer.items()):
        outer[key] = (task_id, channel, ("msgpack", payload), task_path)

    _CALLS.clear()
    loaded = saver.get_tuple(saved_parent)
    listed = list(saver.list(saved_parent))
    history = saver.get_delta_channel_history(
        config={
            "configurable": {
                "thread_id": thread,
                "checkpoint_ns": ns,
                "checkpoint_id": child["id"],
            }
        },
        channels=["chan"],
    )
    assert loaded is not None
    assert loaded.pending_writes is not None
    assert listed[0].pending_writes is not None
    assert _CALLS == []
    assert loaded.checkpoint["channel_values"]["chan"] == _POS_ARGS
    assert loaded.pending_writes[0][2] == _POS_ARGS
    assert listed[0].checkpoint["channel_values"]["chan"] == _POS_ARGS
    assert listed[0].pending_writes[0][2] == _POS_ARGS
    assert history["chan"]["seed"] == _POS_ARGS
    assert history["chan"]["writes"][0][2] == _POS_ARGS


def test_inmemory_cache_get_does_not_call() -> None:
    cache = InMemoryCache()
    ns = ("graph",)
    key = "node"
    cache._cache[ns] = {key: ("msgpack", _pos_payload(), None)}
    _CALLS.clear()
    found = cache.get([(ns, key)])
    _assert_blocked(found[(ns, key)], _POS_ARGS)


def test_redis_cache_get_does_not_call() -> None:
    client = _StoredBytesRedis()
    cache = RedisCache(client)
    full_key: FullKey = (("graph",), "node")
    redis_key = cache._make_key(full_key[0], full_key[1])
    client.stored[redis_key] = b"msgpack:" + _pos_payload()
    _CALLS.clear()
    found = cache.get([full_key])
    _assert_blocked(found[full_key], _POS_ARGS)
