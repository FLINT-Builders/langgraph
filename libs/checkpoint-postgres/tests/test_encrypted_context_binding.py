"""Postgres checkpoints must authenticate the row a ciphertext is stored in.

These tests save dict channel values with `EncryptedSerializer` (those values
are written to `checkpoint_blobs`; primitive channel values stay inline in the
checkpoint JSON and are outside this check), then copy `type` and `blob` with
SQL. They fail while `PostgresSaver` or `ShallowPostgresSaver` decrypts the
copy. They pass once that load is rejected.

Not covered here: `AsyncShallowPostgresSaver.aget_tuple` repeats the pending
sends `loads_typed` call in `shallow.py`, and `RedisCache`.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any
from uuid import uuid4

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import empty_checkpoint
from langgraph.checkpoint.serde.encrypted import EncryptedSerializer
from langgraph.checkpoint.serde.types import TASKS
from psycopg import AsyncConnection, Connection
from psycopg.rows import dict_row

from langgraph.checkpoint.postgres import PostgresSaver, ShallowPostgresSaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from tests.conftest import DEFAULT_POSTGRES_URI

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


@contextmanager
def _database() -> Iterator[str]:
    name = f"ctx{uuid4().hex[:12]}"
    with Connection.connect(DEFAULT_POSTGRES_URI, autocommit=True) as admin:
        admin.execute(f"CREATE DATABASE {name}")
    try:
        yield DEFAULT_POSTGRES_URI + name
    finally:
        with Connection.connect(DEFAULT_POSTGRES_URI, autocommit=True) as admin:
            admin.execute(
                "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                "WHERE datname = %s AND pid <> pg_backend_pid()",
                (name,),
            )
            admin.execute(f"DROP DATABASE {name}")


@contextmanager
def _postgres(cls: type[PostgresSaver] | type[ShallowPostgresSaver]) -> Iterator[Any]:
    with _database() as uri:
        with Connection.connect(
            uri, autocommit=True, prepare_threshold=0, row_factory=dict_row
        ) as conn:
            saver = cls(conn, serde=_serde())
            saver.setup()
            yield saver


def _put(
    saver: Any,
    thread_id: str,
    values: dict[str, Any],
    versions: dict[str, str],
    *,
    checkpoint_ns: str = "",
    parent_id: str | None = None,
) -> RunnableConfig:
    checkpoint = empty_checkpoint()
    checkpoint["channel_values"] = values
    checkpoint["channel_versions"] = dict(versions)
    return saver.put(
        _config(thread_id, checkpoint_ns=checkpoint_ns, checkpoint_id=parent_id),
        checkpoint,
        {},
        versions,
    )


def _copy_blob(
    saver: Any,
    src: tuple[str, str, str, str],
    dst: tuple[str, str, str, str],
) -> None:
    with saver._cursor() as cur:
        cur.execute(
            """
            UPDATE checkpoint_blobs AS dst
            SET type = src.type, blob = src.blob
            FROM checkpoint_blobs AS src
            WHERE src.thread_id = %s AND src.checkpoint_ns = %s
                AND src.channel = %s AND src.version = %s
                AND dst.thread_id = %s AND dst.checkpoint_ns = %s
                AND dst.channel = %s AND dst.version = %s
            """,
            (*src, *dst),
        )


def _copy_write(
    saver: Any,
    src: tuple[str, str, str, str, int],
    dst: tuple[str, str, str, str, int],
) -> None:
    with saver._cursor() as cur:
        cur.execute(
            """
            UPDATE checkpoint_writes AS dst
            SET type = src.type, blob = src.blob
            FROM checkpoint_writes AS src
            WHERE src.thread_id = %s AND src.checkpoint_ns = %s
                AND src.checkpoint_id = %s AND src.task_id = %s AND src.idx = %s
                AND dst.thread_id = %s AND dst.checkpoint_ns = %s
                AND dst.checkpoint_id = %s AND dst.task_id = %s AND dst.idx = %s
            """,
            (*src, *dst),
        )


def _blob_type(saver: Any, thread_id: str, channel: str, version: str) -> str:
    with saver._cursor() as cur:
        cur.execute(
            """
            SELECT type FROM checkpoint_blobs
            WHERE thread_id = %s AND channel = %s AND version = %s
            """,
            (thread_id, channel, version),
        )
        row = cur.fetchone()
    assert row is not None
    return row["type"]


def test_owner_blob_round_trip() -> None:
    with _postgres(PostgresSaver) as saver:
        saved = _put(
            saver,
            "owner",
            {"messages": {"text": "owner-state"}},
            {"messages": "1"},
        )
        assert _blob_type(saver, "owner", "messages", "1").endswith("+aes")

        loaded = saver.get_tuple(saved)

        assert loaded is not None
        assert loaded.checkpoint["channel_values"]["messages"] == {
            "text": "owner-state"
        }


def test_cross_thread_blob_is_rejected() -> None:
    with _postgres(PostgresSaver) as saver:
        _put(saver, "victim", {"messages": {"text": "victim-state"}}, {"messages": "1"})
        attacker = _put(
            saver,
            "attacker",
            {"messages": {"text": "attacker-state"}},
            {"messages": "1"},
        )
        own = saver.get_tuple(attacker)
        assert own is not None
        assert own.checkpoint["channel_values"]["messages"]["text"] == "attacker-state"

        _copy_blob(
            saver,
            ("victim", "", "messages", "1"),
            ("attacker", "", "messages", "1"),
        )

        _assert_rejected(lambda: saver.get_tuple(_config("attacker")), "victim-state")


def test_cross_namespace_blob_is_rejected() -> None:
    with _postgres(PostgresSaver) as saver:
        _put(
            saver,
            "thread",
            {"messages": {"text": "ns-a-state"}},
            {"messages": "1"},
            checkpoint_ns="a",
        )
        _put(
            saver,
            "thread",
            {"messages": {"text": "ns-b-state"}},
            {"messages": "1"},
            checkpoint_ns="b",
        )
        _copy_blob(
            saver,
            ("thread", "a", "messages", "1"),
            ("thread", "b", "messages", "1"),
        )

        _assert_rejected(
            lambda: saver.get_tuple(_config("thread", checkpoint_ns="b")),
            "ns-a-state",
        )


def test_cross_channel_blob_is_rejected() -> None:
    with _postgres(PostgresSaver) as saver:
        saved = _put(
            saver,
            "thread",
            {
                "messages": {"text": "messages-state"},
                "tools": {"text": "tools-state"},
            },
            {"messages": "1", "tools": "2"},
        )
        own = saver.get_tuple(saved)
        assert own is not None
        assert own.checkpoint["channel_values"]["tools"]["text"] == "tools-state"

        _copy_blob(
            saver,
            ("thread", "", "messages", "1"),
            ("thread", "", "tools", "2"),
        )

        _assert_rejected(lambda: saver.get_tuple(saved), "messages-state")


def test_cross_version_blob_is_rejected() -> None:
    with _postgres(PostgresSaver) as saver:
        parent = _put(
            saver, "thread", {"messages": {"text": "old-state"}}, {"messages": "1"}
        )
        child = _put(
            saver,
            "thread",
            {"messages": {"text": "new-state"}},
            {"messages": "2"},
            parent_id=parent["configurable"]["checkpoint_id"],
        )
        own = saver.get_tuple(child)
        assert own is not None
        assert own.checkpoint["channel_values"]["messages"]["text"] == "new-state"

        _copy_blob(
            saver,
            ("thread", "", "messages", "1"),
            ("thread", "", "messages", "2"),
        )

        _assert_rejected(lambda: saver.get_tuple(child), "old-state")


def test_cross_thread_write_is_rejected() -> None:
    with _postgres(PostgresSaver) as saver:
        victim = _put(
            saver, "victim", {"messages": {"text": "blob"}}, {"messages": "1"}
        )
        attacker = _put(
            saver, "attacker", {"messages": {"text": "blob"}}, {"messages": "1"}
        )
        saver.put_writes(victim, [("messages", {"text": "victim-write"})], "task-1")
        saver.put_writes(attacker, [("messages", {"text": "attacker-write"})], "task-1")
        own = saver.get_tuple(attacker)
        assert own is not None
        assert own.pending_writes[0][2] == {"text": "attacker-write"}

        _copy_write(
            saver,
            ("victim", "", victim["configurable"]["checkpoint_id"], "task-1", 0),
            ("attacker", "", attacker["configurable"]["checkpoint_id"], "task-1", 0),
        )

        _assert_rejected(lambda: saver.get_tuple(attacker), "victim-write")


def test_cross_checkpoint_write_is_rejected() -> None:
    with _postgres(PostgresSaver) as saver:
        parent = _put(
            saver, "thread", {"messages": {"text": "blob"}}, {"messages": "1"}
        )
        child = _put(
            saver,
            "thread",
            {"messages": {"text": "blob"}},
            {"messages": "2"},
            parent_id=parent["configurable"]["checkpoint_id"],
        )
        saver.put_writes(parent, [("messages", {"text": "parent-write"})], "task-1")
        saver.put_writes(child, [("messages", {"text": "child-write"})], "task-1")

        _copy_write(
            saver,
            ("thread", "", parent["configurable"]["checkpoint_id"], "task-1", 0),
            ("thread", "", child["configurable"]["checkpoint_id"], "task-1", 0),
        )

        _assert_rejected(lambda: saver.get_tuple(child), "parent-write")


def test_cross_task_write_is_rejected() -> None:
    with _postgres(PostgresSaver) as saver:
        saved = _put(saver, "thread", {"messages": {"text": "blob"}}, {"messages": "1"})
        checkpoint_id = saved["configurable"]["checkpoint_id"]
        saver.put_writes(saved, [("messages", {"text": "task-a-write"})], "task-a")
        saver.put_writes(saved, [("messages", {"text": "task-b-write"})], "task-b")

        _copy_write(
            saver,
            ("thread", "", checkpoint_id, "task-a", 0),
            ("thread", "", checkpoint_id, "task-b", 0),
        )

        _assert_rejected(lambda: saver.get_tuple(saved), "task-a-write")


def test_list_rejects_cross_thread_blob() -> None:
    with _postgres(PostgresSaver) as saver:
        _put(
            saver, "victim", {"messages": {"text": "victim-listed"}}, {"messages": "1"}
        )
        _put(
            saver,
            "attacker",
            {"messages": {"text": "attacker-listed"}},
            {"messages": "1"},
        )
        _copy_blob(
            saver,
            ("victim", "", "messages", "1"),
            ("attacker", "", "messages", "1"),
        )

        _assert_rejected(
            lambda: list(saver.list(_config("attacker"))),
            "victim-listed",
        )


def test_pending_send_migration_rejects_copied_write() -> None:
    """Child checkpoints use v < 4, so get_tuple decrypts parent TASKS writes."""
    with _postgres(PostgresSaver) as saver:
        victim_parent = _put(
            saver, "victim", {"messages": {"text": "blob"}}, {"messages": "1"}
        )
        _put(
            saver,
            "victim",
            {"messages": {"text": "child"}},
            {"messages": "2"},
            parent_id=victim_parent["configurable"]["checkpoint_id"],
        )
        saver.put_writes(victim_parent, [(TASKS, [{"text": "victim-send"}])], "task-1")
        attacker_parent = _put(
            saver, "attacker", {"messages": {"text": "blob"}}, {"messages": "1"}
        )
        attacker_child = _put(
            saver,
            "attacker",
            {"messages": {"text": "child"}},
            {"messages": "2"},
            parent_id=attacker_parent["configurable"]["checkpoint_id"],
        )
        saver.put_writes(
            attacker_parent, [(TASKS, [{"text": "attacker-send"}])], "task-1"
        )

        _copy_write(
            saver,
            (
                "victim",
                "",
                victim_parent["configurable"]["checkpoint_id"],
                "task-1",
                0,
            ),
            (
                "attacker",
                "",
                attacker_parent["configurable"]["checkpoint_id"],
                "task-1",
                0,
            ),
        )

        _assert_rejected(lambda: saver.get_tuple(attacker_child), "victim-send")


def test_delta_seed_blob_is_rejected_when_copied() -> None:
    with _postgres(PostgresSaver) as saver:
        victim_parent = _put(
            saver, "victim", {"messages": {"text": "victim-seed"}}, {"messages": "1"}
        )
        _put(
            saver,
            "victim",
            {"messages": {"text": "victim-child"}},
            {"messages": "2"},
            parent_id=victim_parent["configurable"]["checkpoint_id"],
        )
        attacker_parent = _put(
            saver,
            "attacker",
            {"messages": {"text": "attacker-seed"}},
            {"messages": "1"},
        )
        attacker_child = _put(
            saver,
            "attacker",
            {"messages": {"text": "attacker-child"}},
            {"messages": "2"},
            parent_id=attacker_parent["configurable"]["checkpoint_id"],
        )
        own = saver.get_delta_channel_history(
            config=attacker_child, channels=["messages"]
        )
        assert own["messages"]["seed"] == {"text": "attacker-seed"}

        _copy_blob(
            saver,
            ("victim", "", "messages", "1"),
            ("attacker", "", "messages", "1"),
        )

        _assert_rejected(
            lambda: saver.get_delta_channel_history(
                config=attacker_child, channels=["messages"]
            ),
            "victim-seed",
        )


def test_delta_write_is_rejected_when_copied() -> None:
    with _postgres(PostgresSaver) as saver:
        victim_parent = _put(
            saver, "victim", {"messages": {"text": "seed"}}, {"messages": "1"}
        )
        _put(
            saver,
            "victim",
            {"messages": {"text": "child"}},
            {"messages": "2"},
            parent_id=victim_parent["configurable"]["checkpoint_id"],
        )
        saver.put_writes(
            victim_parent, [("messages", {"text": "victim-delta-write"})], "task-1"
        )
        attacker_parent = _put(
            saver, "attacker", {"messages": {"text": "seed"}}, {"messages": "1"}
        )
        attacker_child = _put(
            saver,
            "attacker",
            {"messages": {"text": "child"}},
            {"messages": "2"},
            parent_id=attacker_parent["configurable"]["checkpoint_id"],
        )
        saver.put_writes(
            attacker_parent,
            [("messages", {"text": "attacker-delta-write"})],
            "task-1",
        )

        _copy_write(
            saver,
            (
                "victim",
                "",
                victim_parent["configurable"]["checkpoint_id"],
                "task-1",
                0,
            ),
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


def test_shallow_cross_thread_blob_is_rejected() -> None:
    with _postgres(ShallowPostgresSaver) as saver:
        _put(
            saver, "victim", {"messages": {"text": "victim-shallow"}}, {"messages": "1"}
        )
        attacker = _put(
            saver,
            "attacker",
            {"messages": {"text": "attacker-shallow"}},
            {"messages": "1"},
        )
        own = saver.get_tuple(attacker)
        assert own is not None
        assert (
            own.checkpoint["channel_values"]["messages"]["text"] == "attacker-shallow"
        )
        with saver._cursor() as cur:
            cur.execute(
                """
                UPDATE checkpoint_blobs AS dst
                SET type = src.type, blob = src.blob
                FROM checkpoint_blobs AS src
                WHERE src.thread_id = %s AND src.channel = %s
                    AND dst.thread_id = %s AND dst.channel = %s
                """,
                ("victim", "messages", "attacker", "messages"),
            )

        _assert_rejected(
            lambda: saver.get_tuple(_config("attacker")),
            "victim-shallow",
        )


def test_shallow_pending_send_is_rejected() -> None:
    with _postgres(ShallowPostgresSaver) as saver:
        victim = _put(
            saver, "victim", {"messages": {"text": "blob"}}, {"messages": "1"}
        )
        attacker = _put(
            saver, "attacker", {"messages": {"text": "blob"}}, {"messages": "1"}
        )
        saver.put_writes(victim, [(TASKS, [{"text": "victim-send"}])], "task-1")
        saver.put_writes(attacker, [(TASKS, [{"text": "attacker-send"}])], "task-1")
        with saver._cursor() as cur:
            cur.execute(
                """
                UPDATE checkpoint_writes AS dst
                SET type = src.type, blob = src.blob
                FROM checkpoint_writes AS src
                WHERE src.thread_id = %s AND src.channel = %s
                    AND dst.thread_id = %s AND dst.channel = %s
                """,
                ("victim", TASKS, "attacker", TASKS),
            )

        _assert_rejected(lambda: saver.get_tuple(attacker), "victim-send")


async def test_async_postgres_rejects_cross_thread_blob() -> None:
    with _database() as uri:
        async with await AsyncConnection.connect(
            uri, autocommit=True, prepare_threshold=0, row_factory=dict_row
        ) as conn:
            saver = AsyncPostgresSaver(conn, serde=_serde())
            await saver.setup()
            victim = empty_checkpoint()
            victim["channel_values"] = {"messages": {"text": "victim-async"}}
            victim["channel_versions"] = {"messages": "1"}
            attacker = empty_checkpoint()
            attacker["channel_values"] = {"messages": {"text": "attacker-async"}}
            attacker["channel_versions"] = {"messages": "1"}
            await saver.aput(_config("victim"), victim, {}, victim["channel_versions"])
            await saver.aput(
                _config("attacker"), attacker, {}, attacker["channel_versions"]
            )
            async with saver._cursor() as cur:
                await cur.execute(
                    """
                    UPDATE checkpoint_blobs AS dst
                    SET type = src.type, blob = src.blob
                    FROM checkpoint_blobs AS src
                    WHERE src.thread_id = %s AND src.channel = %s AND src.version = %s
                        AND dst.thread_id = %s AND dst.channel = %s AND dst.version = %s
                    """,
                    ("victim", "messages", "1", "attacker", "messages", "1"),
                )

            try:
                loaded = await saver.aget_tuple(_config("attacker"))
            except Exception as exc:
                assert "victim-async" not in repr(exc)
                return
            raise AssertionError(
                f"ciphertext stored under a different record was decrypted: {loaded!r}"
            )
