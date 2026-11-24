"""Spec for the encrypted serializer plaintext type-tag bypass.

These tests fail while `EncryptedSerializer.loads_typed` treats a type tag
with no cipher suffix as trusted plaintext. They pass once that serializer
rejects those records unless `allow_unencrypted=True`, and once an
unsupported cipher name is rejected by a check that still runs under
`python -O`.

The attacker in these tests can replace stored type tags and blobs. They
do not have the AES key used to construct the serializer.
"""

# ruff: noqa: E402

from __future__ import annotations

import logging
import pickle
import sqlite3
import subprocess
import sys
import warnings
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import aiosqlite
import ormsgpack
import pytest
from langchain_core.runnables import RunnableConfig

_LIBS = Path(__file__).resolve().parents[2]
for _extra in ("checkpoint-sqlite", "checkpoint-postgres"):
    _path = str(_LIBS / _extra)
    if _path not in sys.path:
        sys.path.insert(0, _path)

from langgraph.cache.sqlite import SqliteCache
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.postgres.base import BasePostgresSaver
from langgraph.checkpoint.postgres.shallow import (
    AsyncShallowPostgresSaver,
    ShallowPostgresSaver,
)
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from langgraph.cache.memory import InMemoryCache
from langgraph.cache.redis import RedisCache
from langgraph.checkpoint.base import empty_checkpoint
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.encrypted import EncryptedSerializer
from langgraph.checkpoint.serde.jsonplus import (
    EXT_CONSTRUCTOR_KW_ARGS,
    EXT_CONSTRUCTOR_POS_ARGS,
    EXT_CONSTRUCTOR_SINGLE_ARG,
    JsonPlusSerializer,
    _msgpack_enc,
)

# Obvious placeholder. Not a credential from this repository.
_AES_KEY = b"0123456789abcdef"
_REJECTED = object()
_SINK_CALLS: list[tuple[tuple[Any, ...], dict[str, Any]]] = []


def plaintext_bypass_sink(*args: Any, **kwargs: Any) -> str:
    """Callable named by attacker-controlled msgpack ext records."""
    _SINK_CALLS.append((args, kwargs))
    return "executed"


def _ext(code: int, payload: tuple[Any, ...]) -> bytes:
    return ormsgpack.packb(
        ormsgpack.Ext(code, _msgpack_enc(payload)),
        option=ormsgpack.OPT_NON_STR_KEYS,
    )


def _single_gadget() -> bytes:
    return _ext(
        EXT_CONSTRUCTOR_SINGLE_ARG,
        (__name__, "plaintext_bypass_sink", "owned"),
    )


def _positional_gadget() -> bytes:
    return _ext(
        EXT_CONSTRUCTOR_POS_ARGS,
        (__name__, "plaintext_bypass_sink", ("owned",)),
    )


def _keyword_gadget() -> bytes:
    return _ext(
        EXT_CONSTRUCTOR_KW_ARGS,
        (__name__, "plaintext_bypass_sink", {"note": "owned"}),
    )


def _forged_state_blob() -> bytes:
    return ormsgpack.packb(
        {"messages": [{"role": "human", "content": "forged"}]},
        option=ormsgpack.OPT_NON_STR_KEYS,
    )


def _allowlisted_set_blob() -> bytes:
    return _ext(
        EXT_CONSTRUCTOR_SINGLE_ARG,
        ("builtins", "set", ["forged-channel-state"]),
    )


def _aes_serde(
    *,
    inner: JsonPlusSerializer | None = None,
    allow_unencrypted: bool = False,
) -> EncryptedSerializer:
    kwargs: dict[str, Any] = {"key": _AES_KEY}
    if allow_unencrypted:
        kwargs["allow_unencrypted"] = True
    if inner is None:
        return EncryptedSerializer.from_pycryptodome_aes(**kwargs)
    return EncryptedSerializer.from_pycryptodome_aes(inner, **kwargs)


def _assert_rejected(load: Callable[[], Any]) -> None:
    _SINK_CALLS.clear()
    try:
        value = load()
    except Exception:
        value = _REJECTED
    assert _SINK_CALLS == []
    assert value is _REJECTED


def _config(thread_id: str = "thread-1") -> RunnableConfig:
    return {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}


def _checkpoint(values: dict[str, Any] | None = None) -> dict[str, Any]:
    checkpoint = empty_checkpoint()
    checkpoint["channel_values"] = dict(values or {})
    checkpoint["channel_versions"] = {key: "1" for key in checkpoint["channel_values"]}
    return checkpoint


def _base_row(**overrides: Any) -> dict[str, Any]:
    row = {
        "thread_id": "thread-1",
        "checkpoint_ns": "",
        "checkpoint_id": "cp-1",
        "parent_checkpoint_id": None,
        "metadata": {},
        "checkpoint": {
            "v": 4,
            "id": "cp-1",
            "ts": "2024-05-04T00:00:00+00:00",
            "channel_values": {},
            "channel_versions": {"messages": "1"},
            "versions_seen": {},
        },
        "channel_values": [],
        "pending_writes": [],
        "pending_sends": [],
    }
    row.update(overrides)
    return row


class _Cursor:
    def __init__(
        self,
        *,
        fetchone: Sequence[Any] = (),
        fetchall: Sequence[Any] = (),
        iterate: Sequence[Any] = (),
    ) -> None:
        self._fetchone = list(fetchone)
        self._fetchall = list(fetchall)
        self._iterate = list(iterate)

    def execute(self, *args: Any, **kwargs: Any) -> None:
        return None

    def fetchone(self) -> Any:
        if not self._fetchone:
            return None
        return self._fetchone.pop(0)

    def fetchall(self) -> list[Any]:
        return list(self._fetchall)

    def __iter__(self) -> Iterator[Any]:
        return iter(self._iterate)


class _CursorContext:
    def __init__(self, cursor: _Cursor) -> None:
        self._cursor = cursor

    def __enter__(self) -> _Cursor:
        return self._cursor

    def __exit__(self, *exc: object) -> bool:
        return False


class _AsyncCursor:
    """Cursor whose execute/fetch methods are awaitable, matching psycopg's async API."""

    def __init__(
        self,
        *,
        fetchone: Sequence[Any] = (),
        iterate: Sequence[Any] = (),
    ) -> None:
        self._fetchone = list(fetchone)
        self._iterate = list(iterate)

    async def execute(self, *args: Any, **kwargs: Any) -> None:
        return None

    async def fetchone(self) -> Any:
        if not self._fetchone:
            return None
        return self._fetchone.pop(0)

    def __aiter__(self) -> _AsyncCursor:
        return self

    async def __anext__(self) -> Any:
        if not self._iterate:
            raise StopAsyncIteration
        return self._iterate.pop(0)


class _AsyncCursorContext:
    def __init__(self, cursor: _AsyncCursor) -> None:
        self._cursor = cursor

    async def __aenter__(self) -> _AsyncCursor:
        return self._cursor

    async def __aexit__(self, *exc: object) -> bool:
        return False


def _bind_cursor(saver: Any, cursor: _Cursor) -> None:
    def _cursor(*args: Any, **kwargs: Any) -> _CursorContext:
        return _CursorContext(cursor)

    saver._cursor = _cursor


def _bind_async_cursor(saver: Any, cursor: _AsyncCursor) -> None:
    def _cursor(*args: Any, **kwargs: Any) -> _AsyncCursorContext:
        return _AsyncCursorContext(cursor)

    saver._cursor = _cursor


@contextmanager
def _sqlite_saver(serde: EncryptedSerializer) -> Iterator[SqliteSaver]:
    conn = sqlite3.connect(":memory:")
    try:
        yield SqliteSaver(conn, serde=serde)
    finally:
        conn.close()


class _Mget:
    def __init__(self, values: dict[str, bytes]) -> None:
        self.values = values

    def mget(self, keys: Sequence[str]) -> list[bytes | None]:
        return [self.values.get(key) for key in keys]


class TestEncryptedSerializerRejectsPlaintext:
    def test_default_factory_is_permissive_msgpack(self) -> None:
        serde = _aes_serde()
        assert serde.serde._allowed_msgpack_modules is True

    @pytest.mark.parametrize(
        "blob",
        [_single_gadget(), _positional_gadget(), _keyword_gadget()],
        ids=["single", "positional", "keyword"],
    )
    def test_plaintext_msgpack_ext_is_not_called(self, blob: bytes) -> None:
        serde = _aes_serde()
        _assert_rejected(lambda: serde.loads_typed(("msgpack", blob)))

    def test_plaintext_msgpack_cannot_forge_state(self) -> None:
        serde = _aes_serde()
        _assert_rejected(lambda: serde.loads_typed(("msgpack", _forged_state_blob())))

    def test_strict_mode_plaintext_cannot_forge_allowlisted_state(self) -> None:
        serde = _aes_serde(inner=JsonPlusSerializer(allowed_msgpack_modules=None))
        _assert_rejected(
            lambda: serde.loads_typed(("msgpack", _allowlisted_set_blob()))
        )

    @pytest.mark.parametrize(
        ("type_", "blob"),
        [
            ("json", b'{"forged": true}'),
            ("bytes", b"forged"),
            ("bytearray", b"forged"),
            ("null", b""),
        ],
    )
    def test_other_plaintext_type_tags_are_rejected(
        self, type_: str, blob: bytes
    ) -> None:
        serde = _aes_serde()
        _assert_rejected(lambda: serde.loads_typed((type_, blob)))

    def test_plaintext_pickle_is_rejected_before_inner_loads(self) -> None:
        serde = _aes_serde(inner=JsonPlusSerializer(pickle_fallback=True))
        blob = pickle.dumps({"forged": True})
        _assert_rejected(lambda: serde.loads_typed(("pickle", blob)))

    def test_cipher_suffix_does_not_deserialize_unauthenticated_body(self) -> None:
        serde = _aes_serde()
        _assert_rejected(lambda: serde.loads_typed(("msgpack+aes", _single_gadget())))

    def test_legitimate_aes_roundtrip_still_works(self) -> None:
        serde = _aes_serde()
        original = {"messages": ["trusted"]}
        dumped = serde.dumps_typed(original)
        assert dumped[0].endswith("+aes")
        assert b"trusted" not in dumped[1]
        assert serde.loads_typed(dumped) == original

    def test_allow_unencrypted_opts_in_and_warns(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        _SINK_CALLS.clear()
        caplog.set_level(logging.WARNING)
        serde = _aes_serde(allow_unencrypted=True)
        assert getattr(serde, "allow_unencrypted", False) is True
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = serde.loads_typed(("msgpack", _single_gadget()))
        assert _SINK_CALLS
        assert result == "executed"
        text = caplog.text + " ".join(str(item.message) for item in caught)
        assert "unencrypted" in text.lower()
        typ, _blob = serde.dumps_typed({"ok": True})
        assert typ.endswith("+aes")

    def test_unsupported_cipher_name_rejected_under_optimize(self) -> None:
        script = """
import sys
from langgraph.checkpoint.serde.encrypted import EncryptedSerializer

key = b"0123456789abcdef"
serde = EncryptedSerializer.from_pycryptodome_aes(key=key)
typ, blob = serde.dumps_typed({"kept": True})
if serde.loads_typed((typ, blob)) != {"kept": True}:
    sys.exit(2)
try:
    loaded = serde.loads_typed(("msgpack+not-aes", blob))
except Exception:
    sys.exit(0)
else:
    sys.exit(1 if loaded == {"kept": True} else 3)
"""
        completed = subprocess.run(
            [sys.executable, "-O", "-c", script],
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, (
            f"stdout={completed.stdout!r} stderr={completed.stderr!r} "
            f"code={completed.returncode}"
        )


class TestSaverLoadPaths:
    def test_sqlite_get_tuple_rejects_plaintext_checkpoint(self) -> None:
        serde = _aes_serde()
        with _sqlite_saver(serde) as saver:
            saved = saver.put(
                _config(),
                _checkpoint({"note": "trusted"}),
                {},
                {"note": "1"},
            )
            stored = saver.conn.execute(
                "SELECT type FROM checkpoints WHERE checkpoint_id = ?",
                (saved["configurable"]["checkpoint_id"],),
            ).fetchone()
            assert stored is not None and "+aes" in stored[0]
            with saver.cursor() as cur:
                cur.execute(
                    "UPDATE checkpoints SET type = ?, checkpoint = ? WHERE checkpoint_id = ?",
                    (
                        "msgpack",
                        _single_gadget(),
                        saved["configurable"]["checkpoint_id"],
                    ),
                )
            _assert_rejected(lambda: saver.get_tuple(_config()))

    def test_sqlite_get_tuple_rejects_plaintext_write(self) -> None:
        serde = _aes_serde()
        with _sqlite_saver(serde) as saver:
            saved = saver.put(_config(), _checkpoint(), {}, {})
            saver.put_writes(saved, [("messages", {"ok": True})], "task-1")
            with saver.cursor() as cur:
                cur.execute(
                    "UPDATE writes SET type = ?, value = ? WHERE task_id = ?",
                    ("msgpack", _positional_gadget(), "task-1"),
                )
            _assert_rejected(lambda: saver.get_tuple(saved))

    def test_sqlite_list_rejects_forged_checkpoint_state(self) -> None:
        serde = _aes_serde()
        with _sqlite_saver(serde) as saver:
            saved = saver.put(_config(), _checkpoint(), {}, {})
            forged = ormsgpack.packb(
                {
                    "v": 2,
                    "id": saved["configurable"]["checkpoint_id"],
                    "ts": "2024-05-04T00:00:00+00:00",
                    "channel_values": {"messages": {"injected": True}},
                    "channel_versions": {},
                    "versions_seen": {},
                }
            )
            with saver.cursor() as cur:
                cur.execute(
                    "UPDATE checkpoints SET type = ?, checkpoint = ?",
                    ("msgpack", forged),
                )

            def _list() -> Any:
                return list(saver.list(_config()))

            _assert_rejected(_list)

    def test_sqlite_delta_history_rejects_plaintext_parent(self) -> None:
        serde = _aes_serde()
        with _sqlite_saver(serde) as saver:
            parent = _checkpoint({"messages": {"role": "human"}})
            parent["id"] = "00000000-0000-0000-0000-000000000001"
            child = _checkpoint()
            child["id"] = "00000000-0000-0000-0000-000000000002"
            parent_cfg = saver.put(_config(), parent, {}, {"messages": "1"})
            child_cfg = saver.put(parent_cfg, child, {}, {})
            with saver.cursor() as cur:
                cur.execute(
                    "UPDATE checkpoints SET type = ?, checkpoint = ? WHERE checkpoint_id = ?",
                    (
                        "msgpack",
                        _keyword_gadget(),
                        parent_cfg["configurable"]["checkpoint_id"],
                    ),
                )
            _assert_rejected(
                lambda: saver.get_delta_channel_history(
                    config=child_cfg, channels=["messages"]
                )
            )

    async def test_async_sqlite_aget_tuple_rejects_plaintext_checkpoint(self) -> None:
        serde = _aes_serde()
        async with aiosqlite.connect(":memory:") as conn:
            saver = AsyncSqliteSaver(conn, serde=serde)
            saved = await saver.aput(_config(), _checkpoint(), {}, {})
            await conn.execute(
                "UPDATE checkpoints SET type = ?, checkpoint = ? WHERE checkpoint_id = ?",
                ("msgpack", _single_gadget(), saved["configurable"]["checkpoint_id"]),
            )
            await conn.commit()
            _SINK_CALLS.clear()
            try:
                value = await saver.aget_tuple(_config())
            except Exception:
                value = _REJECTED
            assert _SINK_CALLS == []
            assert value is _REJECTED

    def test_sqlite_cache_rejects_plaintext_encoding(self, tmp_path: Path) -> None:
        serde = _aes_serde()
        cache = SqliteCache(path=str(tmp_path / "cache.sqlite"), serde=serde)
        cache.set({((), "k"): ({"ok": True}, None)})
        cache._conn.execute(
            "UPDATE cache SET encoding = ?, val = ?",
            ("msgpack", _single_gadget()),
        )
        cache._conn.commit()
        _assert_rejected(lambda: cache.get([((), "k")]))

    def test_memory_saver_rejects_plaintext_blob(self) -> None:
        serde = _aes_serde()
        saver = InMemorySaver(serde=serde)
        saved = saver.put(
            _config(),
            _checkpoint({"messages": {"role": "human"}}),
            {},
            {"messages": "1"},
        )
        blob_key = ("thread-1", "", "messages", "1")
        assert saver.blobs[blob_key][0].endswith("+aes")
        saver.blobs[blob_key] = ("msgpack", _single_gadget())
        _assert_rejected(lambda: saver.get_tuple(saved))

    def test_memory_saver_rejects_plaintext_checkpoint_document(self) -> None:
        serde = _aes_serde()
        saver = InMemorySaver(serde=serde)
        saved = saver.put(_config(), _checkpoint(), {}, {})
        checkpoint_id = saved["configurable"]["checkpoint_id"]
        _, metadata, parent = saver.storage["thread-1"][""][checkpoint_id]
        forged = ormsgpack.packb(
            {
                "v": 2,
                "id": checkpoint_id,
                "ts": "2024-05-04T00:00:00+00:00",
                "channel_values": {},
                "channel_versions": {},
                "versions_seen": {},
                "injected": True,
            }
        )
        saver.storage["thread-1"][""][checkpoint_id] = (
            ("msgpack", forged),
            metadata,
            parent,
        )
        _assert_rejected(lambda: saver.get(saved))

    def test_memory_saver_rejects_plaintext_write(self) -> None:
        serde = _aes_serde()
        saver = InMemorySaver(serde=serde)
        saved = saver.put(_config(), _checkpoint(), {}, {})
        saver.put_writes(saved, [("messages", {"ok": True})], "task-1")
        outer = ("thread-1", "", saved["configurable"]["checkpoint_id"])
        inner_key = next(iter(saver.writes[outer]))
        task_id, channel, _serialized, task_path = saver.writes[outer][inner_key]
        saver.writes[outer][inner_key] = (
            task_id,
            channel,
            ("msgpack", _positional_gadget()),
            task_path,
        )
        _assert_rejected(lambda: list(saver.list(saved)))

    def test_memory_delta_history_rejects_plaintext_blob(self) -> None:
        serde = _aes_serde()
        saver = InMemorySaver(serde=serde)
        parent_cfg = saver.put(
            _config(),
            _checkpoint({"messages": {"role": "human"}}),
            {},
            {"messages": "1"},
        )
        child_cfg = saver.put(parent_cfg, _checkpoint(), {}, {})
        blob_key = ("thread-1", "", "messages", "1")
        saver.blobs[blob_key] = ("msgpack", _allowlisted_set_blob())
        _assert_rejected(
            lambda: saver.get_delta_channel_history(
                config=child_cfg, channels=["messages"]
            )
        )

    def test_memory_cache_rejects_plaintext_encoding(self) -> None:
        serde = _aes_serde()
        cache: InMemoryCache[Any] = InMemoryCache(serde=serde)
        cache.set({((), "k"): ({"ok": True}, None)})
        encoding, blob, expiry = cache._cache[()]["k"]
        assert encoding.endswith("+aes")
        cache._cache[()]["k"] = ("msgpack", _single_gadget(), expiry)
        _assert_rejected(lambda: cache.get([((), "k")]))

    def test_redis_cache_rejects_plaintext_encoding(self) -> None:
        serde = _aes_serde()
        key = "langgraph:cache:k"
        cache: RedisCache[Any] = RedisCache(
            _Mget({key: b"msgpack:" + _single_gadget()}),
            serde=serde,
        )
        _assert_rejected(lambda: cache.get([((), "k")]))

    def test_postgres_get_tuple_rejects_plaintext_blob(self) -> None:
        serde = _aes_serde()
        saver = PostgresSaver.__new__(PostgresSaver)
        BasePostgresSaver.__init__(saver, serde=serde)
        row = _base_row(
            channel_values=[(b"messages", b"msgpack", _single_gadget())],
        )
        _bind_cursor(saver, _Cursor(fetchone=[row]))
        _assert_rejected(lambda: saver.get_tuple(_config()))

    def test_postgres_get_tuple_rejects_plaintext_pending_write(self) -> None:
        serde = _aes_serde()
        saver = PostgresSaver.__new__(PostgresSaver)
        BasePostgresSaver.__init__(saver, serde=serde)
        row = _base_row(
            pending_writes=[(b"task-1", b"messages", b"msgpack", _positional_gadget())],
        )
        _bind_cursor(saver, _Cursor(fetchone=[row]))
        _assert_rejected(lambda: saver.get_tuple(_config()))

    def test_postgres_migration_rejects_plaintext_pending_send(self) -> None:
        serde = _aes_serde()
        saver = PostgresSaver.__new__(PostgresSaver)
        BasePostgresSaver.__init__(saver, serde=serde)
        checkpoint = _base_row()["checkpoint"]
        checkpoint = {**checkpoint, "v": 3, "channel_versions": {}}
        row = _base_row(
            parent_checkpoint_id="parent-1",
            checkpoint=checkpoint,
            channel_values=[],
        )
        _bind_cursor(
            saver,
            _Cursor(
                fetchone=[row, {"sends": [(b"msgpack", _keyword_gadget())]}],
            ),
        )
        _assert_rejected(lambda: saver.get_tuple(_config()))

    def test_postgres_list_rejects_plaintext_blob(self) -> None:
        serde = _aes_serde()
        saver = PostgresSaver.__new__(PostgresSaver)
        BasePostgresSaver.__init__(saver, serde=serde)
        row = _base_row(
            channel_values=[(b"messages", b"msgpack", _forged_state_blob())]
        )
        _bind_cursor(saver, _Cursor(fetchall=[row]))

        def _list() -> Any:
            return list(saver.list(_config()))

        _assert_rejected(_list)

    def test_postgres_delta_seed_rejects_plaintext_blob(self) -> None:
        serde = _aes_serde()
        saver = PostgresSaver.__new__(PostgresSaver)
        BasePostgresSaver.__init__(saver, serde=serde)
        _assert_rejected(
            lambda: saver._build_delta_channels_writes_history(
                channels=["messages"],
                chain_by_ch={"messages": []},
                seed_ver_by_ch={"messages": "1"},
                seed_inline_by_ch={},
                stage2_rows=[
                    {
                        "_kind": "b",
                        "channel": "messages",
                        "version": "1",
                        "type": "msgpack",
                        "blob": _single_gadget(),
                    }
                ],
            )
        )

    def test_postgres_delta_write_rejects_plaintext_blob(self) -> None:
        serde = _aes_serde()
        saver = PostgresSaver.__new__(PostgresSaver)
        BasePostgresSaver.__init__(saver, serde=serde)
        _assert_rejected(
            lambda: saver._build_delta_channels_writes_history(
                channels=["messages"],
                chain_by_ch={"messages": ["cp-1"]},
                seed_ver_by_ch={"messages": None},
                seed_inline_by_ch={},
                stage2_rows=[
                    {
                        "_kind": "w",
                        "checkpoint_id": "cp-1",
                        "channel": "messages",
                        "task_id": "task-1",
                        "idx": 0,
                        "type": "msgpack",
                        "blob": _positional_gadget(),
                    }
                ],
            )
        )

    def test_shallow_postgres_get_tuple_rejects_plaintext_send(self) -> None:
        serde = _aes_serde()
        saver = ShallowPostgresSaver.__new__(ShallowPostgresSaver)
        BasePostgresSaver.__init__(saver, serde=serde)
        row = _base_row(pending_sends=[(b"msgpack", _keyword_gadget())])
        _bind_cursor(saver, _Cursor(iterate=[row]))
        _assert_rejected(lambda: saver.get_tuple(_config()))

    async def test_async_postgres_aget_tuple_rejects_plaintext_blob(self) -> None:
        serde = _aes_serde()
        saver = AsyncPostgresSaver.__new__(AsyncPostgresSaver)
        BasePostgresSaver.__init__(saver, serde=serde)
        row = _base_row(channel_values=[(b"messages", b"msgpack", _single_gadget())])
        _bind_async_cursor(saver, _AsyncCursor(fetchone=[row]))
        _SINK_CALLS.clear()
        try:
            value = await saver.aget_tuple(_config())
        except Exception:
            value = _REJECTED
        assert _SINK_CALLS == []
        assert value is _REJECTED

    async def test_async_shallow_get_tuple_rejects_plaintext_blob(self) -> None:
        serde = _aes_serde()
        saver = AsyncShallowPostgresSaver.__new__(AsyncShallowPostgresSaver)
        BasePostgresSaver.__init__(saver, serde=serde)
        row = _base_row(
            channel_values=[(b"messages", b"msgpack", _forged_state_blob())]
        )
        _bind_async_cursor(saver, _AsyncCursor(iterate=[row]))
        _SINK_CALLS.clear()
        try:
            value = await saver.aget_tuple(_config())
        except Exception:
            value = _REJECTED
        assert _SINK_CALLS == []
        assert value is _REJECTED
