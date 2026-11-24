"""Checkpoint and cache rows stored by the sqlite savers reach the default loader.

`AsyncSqliteSaver.alist` and `aget_delta_channel_history` are the same
`loads_typed` calls as the sync saver and the shared `_delta` helpers
exercised here. `EXT_NUMPY_ARRAY` is a separate reconstruction path and
is not part of this allowlist.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any

import aiosqlite
import ormsgpack
from langgraph.checkpoint.base import empty_checkpoint
from langgraph.checkpoint.serde.jsonplus import (
    EXT_CONSTRUCTOR_POS_ARGS,
    JsonPlusSerializer,
    _msgpack_enc,
)

from langgraph.cache.sqlite import SqliteCache
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.sqlite._delta import (
    build_delta_channels_writes_history,
    step_walk_with_row,
)
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

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


def _checkpoint_blob(saver: SqliteSaver) -> tuple[str, bytes]:
    checkpoint = empty_checkpoint()
    checkpoint["id"] = "cp-1"
    checkpoint["channel_values"] = {
        "chan": ormsgpack.Ext(
            EXT_CONSTRUCTOR_POS_ARGS,
            _msgpack_enc((_MODULE, "pos_target", _POS_ARGS)),
        )
    }
    checkpoint["channel_versions"] = {"chan": "1"}
    return saver.serde.dumps_typed(checkpoint)


def test_sqlite_saver_get_and_list_do_not_call() -> None:
    conn = sqlite3.connect(":memory:")
    saver = SqliteSaver(conn)
    saver.setup()
    type_, checkpoint_blob = _checkpoint_blob(saver)
    conn.execute(
        "INSERT INTO checkpoints (thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, type, checkpoint, metadata) VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("thread-1", "", "cp-1", None, type_, checkpoint_blob, json.dumps({}).encode()),
    )
    conn.execute(
        "INSERT INTO writes (thread_id, checkpoint_ns, checkpoint_id, task_id, idx, channel, type, value) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        ("thread-1", "", "cp-1", "task-1", 0, "notes", "msgpack", _pos_payload()),
    )
    conn.commit()
    config = {
        "configurable": {
            "thread_id": "thread-1",
            "checkpoint_ns": "",
            "checkpoint_id": "cp-1",
        }
    }
    _CALLS.clear()
    loaded = saver.get_tuple(config)
    listed = list(saver.list(config))
    assert loaded is not None
    assert loaded.pending_writes is not None
    assert listed[0].pending_writes is not None
    assert _CALLS == []
    assert loaded.checkpoint["channel_values"]["chan"] == _POS_ARGS
    assert loaded.pending_writes[0][2] == _POS_ARGS
    assert listed[0].checkpoint["channel_values"]["chan"] == _POS_ARGS
    assert listed[0].pending_writes[0][2] == _POS_ARGS
    conn.close()


def test_sqlite_delta_helpers_do_not_call() -> None:
    serde = JsonPlusSerializer()
    checkpoint = empty_checkpoint()
    checkpoint["id"] = "0001-parent"
    checkpoint["channel_values"] = {
        "chan": ormsgpack.Ext(
            EXT_CONSTRUCTOR_POS_ARGS,
            _msgpack_enc((_MODULE, "pos_target", _POS_ARGS)),
        )
    }
    type_, blob = serde.dumps_typed(checkpoint)
    chain_by_ch: dict[str, list[str]] = {"chan": []}
    seed_val_by_ch: dict[str, Any] = {}
    seeded: set[str] = set()
    _CALLS.clear()
    step_walk_with_row(
        cid="0002-child",
        parent_cid="0001-parent",
        type_tag="null",
        blob=b"",
        target_id="0002-child",
        serde=serde,
        chain_by_ch=chain_by_ch,
        seed_val_by_ch=seed_val_by_ch,
        walk_state={},
        seeded=seeded,
        channels=["chan"],
    )
    step_walk_with_row(
        cid="0001-parent",
        parent_cid=None,
        type_tag=type_,
        blob=blob,
        target_id="0002-child",
        serde=serde,
        chain_by_ch=chain_by_ch,
        seed_val_by_ch=seed_val_by_ch,
        walk_state={"started": True, "cur_cid": "0001-parent", "active": {"chan"}},
        seeded=seeded,
        channels=["chan"],
    )
    history = build_delta_channels_writes_history(
        channels=["chan"],
        chain_by_ch=chain_by_ch,
        seed_val_by_ch=seed_val_by_ch,
        seeded=seeded,
        stage2_rows=[("0001-parent", "chan", "task-1", 0, "msgpack", _pos_payload())],
        serde=serde,
    )
    assert _CALLS == []
    assert history["chan"]["seed"] == _POS_ARGS
    assert history["chan"]["writes"] == [("task-1", "chan", _POS_ARGS)]


def test_sqlite_cache_get_does_not_call() -> None:
    cache = SqliteCache(path=":memory:")
    cache._conn.execute(
        "INSERT INTO cache (ns, key, expiry, encoding, val) VALUES (?, ?, ?, ?, ?)",
        ("graph", "node", None, "msgpack", _pos_payload()),
    )
    cache._conn.commit()
    _CALLS.clear()
    found = cache.get([(("graph",), "node")])
    assert _CALLS == []
    assert found[(("graph",), "node")] == _POS_ARGS
    cache._conn.close()


async def test_async_sqlite_saver_aget_tuple_does_not_call() -> None:
    async with aiosqlite.connect(":memory:") as conn:
        saver = AsyncSqliteSaver(conn)
        await saver.setup()
        type_, checkpoint_blob = _checkpoint_blob(saver)
        await conn.execute(
            "INSERT INTO checkpoints (thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, type, checkpoint, metadata) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                "thread-1",
                "",
                "cp-1",
                None,
                type_,
                checkpoint_blob,
                json.dumps({}).encode(),
            ),
        )
        await conn.execute(
            "INSERT INTO writes (thread_id, checkpoint_ns, checkpoint_id, task_id, idx, channel, type, value) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("thread-1", "", "cp-1", "task-1", 0, "notes", "msgpack", _pos_payload()),
        )
        await conn.commit()
        _CALLS.clear()
        loaded = await saver.aget_tuple(
            {
                "configurable": {
                    "thread_id": "thread-1",
                    "checkpoint_ns": "",
                    "checkpoint_id": "cp-1",
                }
            }
        )
        assert loaded is not None
        assert _CALLS == []
        assert loaded.checkpoint["channel_values"]["chan"] == _POS_ARGS
        assert loaded.pending_writes is not None
        assert loaded.pending_writes[0][2] == _POS_ARGS
