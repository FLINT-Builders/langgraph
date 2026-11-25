"""SQLite checkpoints must authenticate the row a ciphertext is stored in.

These tests save with `EncryptedSerializer.from_pycryptodome_aes`, then copy
ciphertext with SQL the way a database writer who does not hold the key
would. They fail while `SqliteSaver` and `AsyncSqliteSaver` decrypt the copy.
They pass once that load is rejected.

`SqliteSaver` stores metadata as plaintext JSON, so metadata is not an
encrypted record and is not covered here. `RedisCache` is not covered here.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Iterator
from contextlib import closing
from typing import Any

import aiosqlite
import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import empty_checkpoint
from langgraph.checkpoint.serde.encrypted import EncryptedSerializer

from langgraph.cache.sqlite import SqliteCache
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

# Obvious test placeholder. Not a deployed key.
_PLACEHOLDER_AES_KEY = b"placeholder-key!"


def _serde() -> EncryptedSerializer:
    assert len(_PLACEHOLDER_AES_KEY) in (16, 24, 32)
    return EncryptedSerializer.from_pycryptodome_aes(key=_PLACEHOLDER_AES_KEY)


def _config(
    thread_id: str,
    *,
    checkpoint_ns: str = "",
    checkpoint_id: str | None = None,
) -> RunnableConfig:
    configurable: dict[str, Any] = {
        "thread_id": thread_id,
        "checkpoint_ns": checkpoint_ns,
    }
    if checkpoint_id is not None:
        configurable["checkpoint_id"] = checkpoint_id
    return {"configurable": configurable}


def _assert_rejected(load: Callable[[], Any], foreign_plaintext: str) -> None:
    try:
        loaded = load()
    except Exception as exc:
        assert foreign_plaintext not in repr(exc)
        return
    raise AssertionError(
        f"ciphertext stored under a different record was decrypted: {loaded!r}"
    )


@pytest.fixture
def saver() -> Iterator[SqliteSaver]:
    with closing(sqlite3.connect(":memory:", check_same_thread=False)) as conn:
        yield SqliteSaver(conn, serde=_serde())


def _put(
    saver: SqliteSaver,
    thread_id: str,
    values: dict[str, Any],
    *,
    checkpoint_ns: str = "",
    parent_id: str | None = None,
) -> RunnableConfig:
    checkpoint = empty_checkpoint()
    checkpoint["channel_values"] = values
    checkpoint["channel_versions"] = {key: "1" for key in values}
    saved = saver.put(
        _config(thread_id, checkpoint_ns=checkpoint_ns, checkpoint_id=parent_id),
        checkpoint,
        {},
        checkpoint["channel_versions"],
    )
    row = saver.conn.execute(
        "SELECT type FROM checkpoints WHERE thread_id = ? AND checkpoint_ns = ? AND checkpoint_id = ?",
        (thread_id, checkpoint_ns, saved["configurable"]["checkpoint_id"]),
    ).fetchone()
    assert row is not None
    assert row[0].endswith("+aes")
    return saved


def _copy_checkpoint(
    conn: sqlite3.Connection,
    src: tuple[str, str, str],
    dst: tuple[str, str, str],
) -> None:
    conn.execute(
        """
        UPDATE checkpoints
        SET type = (
                SELECT type FROM checkpoints
                WHERE thread_id = ? AND checkpoint_ns = ? AND checkpoint_id = ?
            ),
            checkpoint = (
                SELECT checkpoint FROM checkpoints
                WHERE thread_id = ? AND checkpoint_ns = ? AND checkpoint_id = ?
            )
        WHERE thread_id = ? AND checkpoint_ns = ? AND checkpoint_id = ?
        """,
        (*src, *src, *dst),
    )
    conn.commit()


def _copy_write(
    conn: sqlite3.Connection,
    src: tuple[str, str, str, str, int],
    dst: tuple[str, str, str, str, int],
) -> None:
    conn.execute(
        """
        UPDATE writes
        SET type = (
                SELECT type FROM writes
                WHERE thread_id = ? AND checkpoint_ns = ? AND checkpoint_id = ?
                    AND task_id = ? AND idx = ?
            ),
            value = (
                SELECT value FROM writes
                WHERE thread_id = ? AND checkpoint_ns = ? AND checkpoint_id = ?
                    AND task_id = ? AND idx = ?
            )
        WHERE thread_id = ? AND checkpoint_ns = ? AND checkpoint_id = ?
            AND task_id = ? AND idx = ?
        """,
        (*src, *src, *dst),
    )
    conn.commit()


def test_owner_checkpoint_round_trip(saver: SqliteSaver) -> None:
    saved = _put(saver, "owner", {"messages": {"text": "owner-state"}})

    loaded = saver.get_tuple(saved)

    assert loaded is not None
    assert loaded.checkpoint["channel_values"]["messages"] == {"text": "owner-state"}


def test_cross_thread_checkpoint_is_rejected(saver: SqliteSaver) -> None:
    _put(saver, "victim", {"messages": {"text": "victim-state"}})
    attacker = _put(saver, "attacker", {"messages": {"text": "attacker-state"}})
    own = saver.get_tuple(attacker)
    assert own is not None
    assert own.checkpoint["channel_values"]["messages"]["text"] == "attacker-state"
    victim_id = saver.conn.execute(
        "SELECT checkpoint_id FROM checkpoints WHERE thread_id = ?",
        ("victim",),
    ).fetchone()[0]

    _copy_checkpoint(
        saver.conn,
        ("victim", "", victim_id),
        ("attacker", "", attacker["configurable"]["checkpoint_id"]),
    )

    _assert_rejected(lambda: saver.get_tuple(_config("attacker")), "victim-state")


def test_cross_namespace_checkpoint_is_rejected(saver: SqliteSaver) -> None:
    source = _put(
        saver,
        "thread",
        {"messages": {"text": "ns-a-state"}},
        checkpoint_ns="a",
    )
    _put(
        saver,
        "thread",
        {"messages": {"text": "ns-b-state"}},
        checkpoint_ns="b",
    )
    destination_id = saver.conn.execute(
        "SELECT checkpoint_id FROM checkpoints WHERE thread_id = ? AND checkpoint_ns = ?",
        ("thread", "b"),
    ).fetchone()[0]

    _copy_checkpoint(
        saver.conn,
        ("thread", "a", source["configurable"]["checkpoint_id"]),
        ("thread", "b", destination_id),
    )

    _assert_rejected(
        lambda: saver.get_tuple(_config("thread", checkpoint_ns="b")),
        "ns-a-state",
    )


def test_old_checkpoint_copied_onto_a_newer_row_is_rejected(saver: SqliteSaver) -> None:
    parent = _put(saver, "thread", {"messages": {"text": "old-state"}})
    child = _put(
        saver,
        "thread",
        {"messages": {"text": "new-state"}},
        parent_id=parent["configurable"]["checkpoint_id"],
    )
    own = saver.get_tuple(child)
    assert own is not None
    assert own.checkpoint["channel_values"]["messages"]["text"] == "new-state"

    _copy_checkpoint(
        saver.conn,
        ("thread", "", parent["configurable"]["checkpoint_id"]),
        ("thread", "", child["configurable"]["checkpoint_id"]),
    )

    _assert_rejected(lambda: saver.get_tuple(child), "old-state")


def test_cross_thread_write_is_rejected(saver: SqliteSaver) -> None:
    victim = _put(saver, "victim", {"messages": {"text": "blob"}})
    attacker = _put(saver, "attacker", {"messages": {"text": "blob"}})
    saver.put_writes(victim, [("messages", {"text": "victim-write"})], "task-1")
    saver.put_writes(attacker, [("messages", {"text": "attacker-write"})], "task-1")
    own = saver.get_tuple(attacker)
    assert own is not None
    assert own.pending_writes[0][2] == {"text": "attacker-write"}

    _copy_write(
        saver.conn,
        ("victim", "", victim["configurable"]["checkpoint_id"], "task-1", 0),
        ("attacker", "", attacker["configurable"]["checkpoint_id"], "task-1", 0),
    )

    _assert_rejected(lambda: saver.get_tuple(attacker), "victim-write")


def test_cross_checkpoint_write_is_rejected(saver: SqliteSaver) -> None:
    parent = _put(saver, "thread", {"messages": {"text": "blob"}})
    child = _put(
        saver,
        "thread",
        {"messages": {"text": "blob"}},
        parent_id=parent["configurable"]["checkpoint_id"],
    )
    saver.put_writes(parent, [("messages", {"text": "parent-write"})], "task-1")
    saver.put_writes(child, [("messages", {"text": "child-write"})], "task-1")

    _copy_write(
        saver.conn,
        ("thread", "", parent["configurable"]["checkpoint_id"], "task-1", 0),
        ("thread", "", child["configurable"]["checkpoint_id"], "task-1", 0),
    )

    _assert_rejected(lambda: saver.get_tuple(child), "parent-write")


def test_cross_task_write_is_rejected(saver: SqliteSaver) -> None:
    saved = _put(saver, "thread", {"messages": {"text": "blob"}})
    checkpoint_id = saved["configurable"]["checkpoint_id"]
    saver.put_writes(saved, [("messages", {"text": "task-a-write"})], "task-a")
    saver.put_writes(saved, [("messages", {"text": "task-b-write"})], "task-b")

    _copy_write(
        saver.conn,
        ("thread", "", checkpoint_id, "task-a", 0),
        ("thread", "", checkpoint_id, "task-b", 0),
    )

    _assert_rejected(lambda: saver.get_tuple(saved), "task-a-write")


def test_list_rejects_cross_thread_checkpoint(saver: SqliteSaver) -> None:
    victim = _put(saver, "victim", {"messages": {"text": "victim-listed"}})
    _put(saver, "attacker", {"messages": {"text": "attacker-listed"}})
    attacker_id = saver.conn.execute(
        "SELECT checkpoint_id FROM checkpoints WHERE thread_id = ?",
        ("attacker",),
    ).fetchone()[0]
    _copy_checkpoint(
        saver.conn,
        ("victim", "", victim["configurable"]["checkpoint_id"]),
        ("attacker", "", attacker_id),
    )

    _assert_rejected(
        lambda: list(saver.list(_config("attacker"))),
        "victim-listed",
    )


def test_delta_seed_checkpoint_is_rejected_when_copied(saver: SqliteSaver) -> None:
    victim_parent = _put(saver, "victim", {"messages": {"text": "victim-seed"}})
    _put(
        saver,
        "victim",
        {"messages": {"text": "victim-child"}},
        parent_id=victim_parent["configurable"]["checkpoint_id"],
    )
    attacker_parent = _put(saver, "attacker", {"messages": {"text": "attacker-seed"}})
    attacker_child = _put(
        saver,
        "attacker",
        {"messages": {"text": "attacker-child"}},
        parent_id=attacker_parent["configurable"]["checkpoint_id"],
    )
    own = saver.get_delta_channel_history(config=attacker_child, channels=["messages"])
    assert own["messages"]["seed"]["text"] == "attacker-seed"

    _copy_checkpoint(
        saver.conn,
        ("victim", "", victim_parent["configurable"]["checkpoint_id"]),
        ("attacker", "", attacker_parent["configurable"]["checkpoint_id"]),
    )

    _assert_rejected(
        lambda: saver.get_delta_channel_history(
            config=attacker_child, channels=["messages"]
        ),
        "victim-seed",
    )


def test_delta_write_is_rejected_when_copied(saver: SqliteSaver) -> None:
    victim_parent = _put(saver, "victim", {"messages": {"text": "seed"}})
    _put(
        saver,
        "victim",
        {"messages": {"text": "child"}},
        parent_id=victim_parent["configurable"]["checkpoint_id"],
    )
    saver.put_writes(
        victim_parent, [("messages", {"text": "victim-delta-write"})], "task-1"
    )
    attacker_parent = _put(saver, "attacker", {"messages": {"text": "seed"}})
    attacker_child = _put(
        saver,
        "attacker",
        {"messages": {"text": "child"}},
        parent_id=attacker_parent["configurable"]["checkpoint_id"],
    )
    saver.put_writes(
        attacker_parent, [("messages", {"text": "attacker-delta-write"})], "task-1"
    )

    _copy_write(
        saver.conn,
        ("victim", "", victim_parent["configurable"]["checkpoint_id"], "task-1", 0),
        (
            "attacker",
            "",
            attacker_parent["configurable"]["checkpoint_id"],
            "task-1",
            0,
        ),
    )

    _assert_rejected(
        lambda: saver.get_delta_channel_history(
            config=attacker_child, channels=["messages"]
        ),
        "victim-delta-write",
    )


async def test_async_sqlite_rejects_cross_thread_checkpoint() -> None:
    async with aiosqlite.connect(":memory:") as conn:
        saver = AsyncSqliteSaver(conn, serde=_serde())
        victim = empty_checkpoint()
        victim["channel_values"] = {"messages": {"text": "victim-async"}}
        victim["channel_versions"] = {"messages": "1"}
        attacker = empty_checkpoint()
        attacker["channel_values"] = {"messages": {"text": "attacker-async"}}
        attacker["channel_versions"] = {"messages": "1"}
        victim_saved = await saver.aput(
            _config("victim"), victim, {}, victim["channel_versions"]
        )
        attacker_saved = await saver.aput(
            _config("attacker"), attacker, {}, attacker["channel_versions"]
        )
        await conn.execute(
            """
            UPDATE checkpoints
            SET type = (
                    SELECT type FROM checkpoints
                    WHERE thread_id = ? AND checkpoint_ns = ? AND checkpoint_id = ?
                ),
                checkpoint = (
                    SELECT checkpoint FROM checkpoints
                    WHERE thread_id = ? AND checkpoint_ns = ? AND checkpoint_id = ?
                )
            WHERE thread_id = ? AND checkpoint_ns = ? AND checkpoint_id = ?
            """,
            (
                "victim",
                "",
                victim_saved["configurable"]["checkpoint_id"],
                "victim",
                "",
                victim_saved["configurable"]["checkpoint_id"],
                "attacker",
                "",
                attacker_saved["configurable"]["checkpoint_id"],
            ),
        )
        await conn.commit()

        try:
            loaded = await saver.aget_tuple(_config("attacker"))
        except Exception as exc:
            assert "victim-async" not in repr(exc)
            return
        raise AssertionError(
            f"ciphertext stored under a different record was decrypted: {loaded!r}"
        )


def test_sqlite_cache_value_is_bound_to_namespace_and_key(tmp_path: Any) -> None:
    cache = SqliteCache(path=str(tmp_path / "cache.sqlite"), serde=_serde())
    victim_key = (("users",), "victim")
    attacker_key = (("users",), "attacker")
    cache.set(
        {
            victim_key: ({"text": "victim-cache"}, None),
            attacker_key: ({"text": "attacker-cache"}, None),
        }
    )
    assert cache.get([attacker_key])[attacker_key] == {"text": "attacker-cache"}
    cache._conn.execute(
        """
        UPDATE cache
        SET encoding = (SELECT encoding FROM cache WHERE ns = ? AND key = ?),
            val = (SELECT val FROM cache WHERE ns = ? AND key = ?)
        WHERE ns = ? AND key = ?
        """,
        ("users", "victim", "users", "victim", "users", "attacker"),
    )
    cache._conn.commit()

    _assert_rejected(lambda: cache.get([attacker_key]), "victim-cache")
