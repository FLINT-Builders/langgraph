"""Primitive channel values must not bypass an encrypting serializer.

Graph state fields stored in a `LastValue` channel are copied unchanged into
`checkpoint["channel_values"]` (`LastValue.checkpoint`) and handed to
`PostgresSaver.put` / `AsyncPostgresSaver.aput`. Those methods keep `None`,
`str`, `int`, `float`, and `bool` inline in the `checkpoints.checkpoint` JSONB
column and only send other values through `serde`.

These tests fail while that split is in place and an `EncryptedSerializer` is
configured: a reader of the checkpoints table can recover the primitive
fields without the cipher. They pass once every channel value is stored only
as serializer output in `checkpoint_blobs` and `get_tuple` / `aget_tuple`
still return the original values.

`ShallowPostgresSaver` and `AsyncShallowPostgresSaver` already pass every
channel value to the serializer, and `put_writes` / `aput_writes` do the same
for writes. Those paths are not covered here.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import asynccontextmanager, contextmanager
from typing import Any
from uuid import uuid4

from langgraph.checkpoint.base import Checkpoint
from langgraph.checkpoint.serde.encrypted import EncryptedSerializer
from psycopg import AsyncConnection, Connection
from psycopg.rows import DictRow, dict_row

from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from tests.conftest import DEFAULT_POSTGRES_URI

# Obvious placeholder. Not a key used by any deployment.
_AES_KEY = b"placeholder-key!"

USER_QUERY = "customer-query-visible-without-the-cipher"
NESTED_SUMMARY = "nested-summary-must-stay-ciphertext"
COUNT = 42424242
SCORE = 8675309.125

_PRIMITIVE_CHANNELS = ("user_query", "count", "score", "flag", "enabled", "note")
_ALL_CHANNELS = (*_PRIMITIVE_CHANNELS, "payload")


def _serde() -> EncryptedSerializer:
    assert len(_AES_KEY) == 16
    return EncryptedSerializer.from_pycryptodome_aes(key=_AES_KEY)


def _checkpoint() -> tuple[Checkpoint, dict[str, str]]:
    """Checkpoint shape produced for a graph whose channels hold these values."""
    versions = {
        "user_query": "00000000000000000000000000000001.0000000000000001",
        "count": "00000000000000000000000000000001.0000000000000002",
        "score": "00000000000000000000000000000001.0000000000000003",
        "flag": "00000000000000000000000000000001.0000000000000004",
        "enabled": "00000000000000000000000000000001.0000000000000005",
        "note": "00000000000000000000000000000001.0000000000000006",
        "payload": "00000000000000000000000000000001.0000000000000007",
    }
    checkpoint: Checkpoint = {
        "v": 4,
        "id": str(uuid4()),
        "ts": "2024-05-04T06:32:42.235444+00:00",
        "channel_values": {
            "user_query": USER_QUERY,
            "count": COUNT,
            "score": SCORE,
            "flag": False,
            "enabled": True,
            "note": None,
            "payload": {"summary": NESTED_SUMMARY},
        },
        "channel_versions": versions,
        "versions_seen": {},
        "updated_channels": None,
    }
    return checkpoint, versions


def _assert_round_trip(channel_values: Mapping[str, Any]) -> None:
    assert channel_values["user_query"] == USER_QUERY
    assert channel_values["count"] == COUNT
    assert channel_values["score"] == SCORE
    assert channel_values["flag"] is False
    assert channel_values["enabled"] is True
    assert channel_values["note"] is None
    assert channel_values["payload"] == {"summary": NESTED_SUMMARY}


def _assert_not_readable_without_the_cipher(
    checkpoint_text: str,
    channel_values: Mapping[str, Any],
    blobs: Sequence[Mapping[str, Any]],
) -> None:
    """A checkpoints-table reader must not see primitive channel values.

    Each primitive channel must also have a blob whose type is the
    encrypting serializer's `type+cipher` tag, and whose bytes do not
    contain the plaintext strings.
    """
    assert USER_QUERY not in checkpoint_text
    assert NESTED_SUMMARY not in checkpoint_text
    assert str(COUNT) not in checkpoint_text
    assert "8675309.125" not in checkpoint_text

    assert channel_values.get("user_query") != USER_QUERY
    assert channel_values.get("count") != COUNT
    assert channel_values.get("score") != SCORE
    assert channel_values.get("flag") is not False
    assert channel_values.get("enabled") is not True
    # JSON null is the inlined form of None. Absence, or a non-null
    # encrypted stand-in, is required.
    assert "note" not in channel_values or channel_values["note"] is not None

    by_channel = {row["channel"]: row for row in blobs}
    for channel in _ALL_CHANNELS:
        assert channel in by_channel, channel
        blob_type = by_channel[channel]["type"]
        assert isinstance(blob_type, str)
        assert "+" in blob_type
        raw = bytes(by_channel[channel]["blob"] or b"")
        assert USER_QUERY.encode() not in raw
        assert NESTED_SUMMARY.encode() not in raw


@contextmanager
def _sync_saver() -> Iterator[tuple[PostgresSaver, Connection[DictRow]]]:
    database = f"test_{uuid4().hex[:16]}"
    with Connection.connect(DEFAULT_POSTGRES_URI, autocommit=True) as admin:
        admin.execute(f"CREATE DATABASE {database}")
    try:
        with Connection.connect(
            DEFAULT_POSTGRES_URI + database,
            autocommit=True,
            prepare_threshold=0,
            row_factory=dict_row,
        ) as conn:
            saver = PostgresSaver(conn, serde=_serde())
            saver.setup()
            yield saver, conn
    finally:
        with Connection.connect(DEFAULT_POSTGRES_URI, autocommit=True) as admin:
            admin.execute(f"DROP DATABASE {database}")


@asynccontextmanager
async def _async_saver():
    database = f"test_{uuid4().hex[:16]}"
    async with await AsyncConnection.connect(
        DEFAULT_POSTGRES_URI, autocommit=True
    ) as admin:
        await admin.execute(f"CREATE DATABASE {database}")
    try:
        async with await AsyncConnection.connect(
            DEFAULT_POSTGRES_URI + database,
            autocommit=True,
            prepare_threshold=0,
            row_factory=dict_row,
        ) as conn:
            saver = AsyncPostgresSaver(conn, serde=_serde())
            await saver.setup()
            yield saver, conn
    finally:
        async with await AsyncConnection.connect(
            DEFAULT_POSTGRES_URI, autocommit=True
        ) as admin:
            await admin.execute(f"DROP DATABASE {database}")


def _stored_rows(
    conn: Connection[DictRow], thread_id: str
) -> tuple[str, Mapping[str, Any], list[dict[str, Any]]]:
    row = conn.execute(
        """
        SELECT checkpoint::text AS checkpoint_text, checkpoint
        FROM checkpoints
        WHERE thread_id = %s
        """,
        (thread_id,),
    ).fetchone()
    assert row is not None
    blobs = conn.execute(
        """
        SELECT channel, type, blob
        FROM checkpoint_blobs
        WHERE thread_id = %s
        """,
        (thread_id,),
    ).fetchall()
    channel_values = row["checkpoint"].get("channel_values") or {}
    return row["checkpoint_text"], channel_values, list(blobs)


async def _astored_rows(
    conn: AsyncConnection[DictRow], thread_id: str
) -> tuple[str, Mapping[str, Any], list[dict[str, Any]]]:
    cur = await conn.execute(
        """
        SELECT checkpoint::text AS checkpoint_text, checkpoint
        FROM checkpoints
        WHERE thread_id = %s
        """,
        (thread_id,),
    )
    row = await cur.fetchone()
    assert row is not None
    blob_cur = await conn.execute(
        """
        SELECT channel, type, blob
        FROM checkpoint_blobs
        WHERE thread_id = %s
        """,
        (thread_id,),
    )
    blobs = await blob_cur.fetchall()
    channel_values = row["checkpoint"].get("channel_values") or {}
    return row["checkpoint_text"], channel_values, list(blobs)


def test_postgres_saver_put_encrypts_primitive_channel_values() -> None:
    thread_id = f"thread-{uuid4().hex}"
    checkpoint, versions = _checkpoint()
    config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}

    with _sync_saver() as (saver, conn):
        saved = saver.put(
            config,
            checkpoint,
            {"source": "loop", "step": 1, "parents": {}},
            versions,
        )
        loaded = saver.get_tuple(saved)
        assert loaded is not None
        _assert_round_trip(loaded.checkpoint["channel_values"])

        checkpoint_text, channel_values, blobs = _stored_rows(conn, thread_id)
        _assert_not_readable_without_the_cipher(checkpoint_text, channel_values, blobs)


async def test_async_postgres_saver_aput_encrypts_primitive_channel_values() -> None:
    thread_id = f"thread-{uuid4().hex}"
    checkpoint, versions = _checkpoint()
    config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}

    async with _async_saver() as (saver, conn):
        saved = await saver.aput(
            config,
            checkpoint,
            {"source": "loop", "step": 1, "parents": {}},
            versions,
        )
        loaded = await saver.aget_tuple(saved)
        assert loaded is not None
        _assert_round_trip(loaded.checkpoint["channel_values"])

        checkpoint_text, channel_values, blobs = await _astored_rows(conn, thread_id)
        _assert_not_readable_without_the_cipher(checkpoint_text, channel_values, blobs)
