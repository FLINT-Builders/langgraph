"""Postgres loaders pass stored blob bytes to the default msgpack serializer.

`PostgresSaver.get_tuple` and `list` obtain `(channel, type, blob)` and
write rows from `SELECT_SQL`, then call `_load_blobs` and `_load_writes`.
`get_delta_channel_history` calls `_build_delta_channels_writes_history`.
Those methods are what this file runs. The shallow saver's inline
`pending_sends` loads (`shallow.py` get/list, sync and async) are the
same `loads_typed((type, blob))` call and are not given their own test.
"""

from __future__ import annotations

from typing import Any

import ormsgpack
import pytest
from langgraph.checkpoint.serde.jsonplus import (
    EXT_CONSTRUCTOR_POS_ARGS,
    _msgpack_enc,
)
from langgraph.checkpoint.serde.types import TASKS

from langgraph.checkpoint.postgres.base import BasePostgresSaver

pytestmark = pytest.mark.no_db

_MODULE = __name__
_CALLS: list[tuple[Any, ...]] = []
_POS_ARGS = ["marker-pos"]


def pos_target(*args: object) -> str:
    _CALLS.append(("pos", args))
    return "executed-pos"


def _pos_payload() -> bytes:
    return ormsgpack.packb(
        ormsgpack.Ext(
            EXT_CONSTRUCTOR_POS_ARGS,
            _msgpack_enc((_MODULE, "pos_target", _POS_ARGS)),
        ),
        option=ormsgpack.OPT_NON_STR_KEYS,
    )


def test_postgres_blob_and_write_loaders_do_not_call() -> None:
    saver = BasePostgresSaver()
    payload = _pos_payload()
    _CALLS.clear()
    blobs = saver._load_blobs([(b"chan", b"msgpack", payload)])
    writes = saver._load_writes([(b"task-1", b"notes", b"msgpack", payload)])
    assert _CALLS == []
    assert blobs == {"chan": _POS_ARGS}
    assert writes == [("task-1", "notes", _POS_ARGS)]


def test_postgres_delta_history_does_not_call() -> None:
    saver = BasePostgresSaver()
    payload = _pos_payload()
    _CALLS.clear()
    history = saver._build_delta_channels_writes_history(
        channels=["chan"],
        chain_by_ch={"chan": ["cp-parent"]},
        seed_ver_by_ch={"chan": "1"},
        seed_inline_by_ch={},
        stage2_rows=[
            {
                "_kind": "b",
                "channel": "chan",
                "version": "1",
                "type": "msgpack",
                "blob": payload,
            },
            {
                "_kind": "w",
                "channel": "chan",
                "checkpoint_id": "cp-parent",
                "type": "msgpack",
                "blob": payload,
                "task_id": "task-1",
                "idx": 0,
            },
        ],
    )
    assert _CALLS == []
    assert history["chan"]["seed"] == _POS_ARGS
    assert history["chan"]["writes"] == [("task-1", "chan", _POS_ARGS)]


def test_postgres_pending_send_migration_does_not_call() -> None:
    saver = BasePostgresSaver()
    checkpoint: dict[str, Any] = {"channel_versions": {}}
    channel_values: list[tuple[bytes, bytes, bytes]] = []
    _CALLS.clear()
    saver._migrate_pending_sends(
        [(b"msgpack", _pos_payload())],
        checkpoint,
        channel_values,
    )
    assert _CALLS == []
    assert channel_values
    _channel, enc, blob = channel_values[0]
    assert _channel == TASKS.encode()
    migrated = saver.serde.loads_typed((enc.decode(), blob))
    assert migrated == [_POS_ARGS]
    assert _CALLS == []
