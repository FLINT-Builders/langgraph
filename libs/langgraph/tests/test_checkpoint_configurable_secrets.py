"""Sensitive run-config values must not be stored in checkpoint metadata.

These tests fail while the vulnerability is present. `get_checkpoint_metadata`
copies primitive `configurable` entries into the metadata saved with every
checkpoint, and the SQLite savers then JSON-encode that metadata outside the
configured serializer. They pass once those values are omitted.

Postgres savers are not executed here (no database server). They persist the
same `get_serializable_checkpoint_metadata` object through psycopg `Jsonb`,
and `test_postgres_jsonb_metadata_payload_omits_sensitive_configurable` checks
that payload with the dumper those savers use.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from collections.abc import Mapping
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    empty_checkpoint,
    get_checkpoint_metadata,
    get_serializable_checkpoint_metadata,
)
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.encrypted import EncryptedSerializer
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from psycopg.types.json import Jsonb, JsonbDumper
from typing_extensions import TypedDict

from langgraph.graph import END, START, StateGraph

# Obvious placeholders. These are not credentials.
_BENIGN_MODEL = "placeholder-model"
_BENIGN_ATTEMPT = 3
_BENIGN_RUN = "placeholder-run"
_PRIVATE = "placeholder-private"

_SENSITIVE: dict[str, Any] = {
    "api_key": "placeholder-api-key",
    "access_token": "placeholder-token",
    "client_secret": "placeholder-secret",
    "db_password": "placeholder-password",
    "authorization": "placeholder-auth",
    "API_KEY": "placeholder-api-key-upper",
    "My_Secret": "placeholder-mixed-secret",
    "signing_key": "",
    "refresh_token": 7,
}

_SENSITIVE_TEXTS = tuple(
    value for value in _SENSITIVE.values() if isinstance(value, str) and value
)


class _PrefixCipher:
    """Marks bytes the serializer handled. Not an encryption algorithm."""

    def encrypt(self, plaintext: bytes) -> tuple[str, bytes]:
        return "prefix", b"ENC:" + plaintext

    def decrypt(self, ciphername: str, ciphertext: bytes) -> bytes:
        if ciphername != "prefix" or not ciphertext.startswith(b"ENC:"):
            raise ValueError("ciphertext was not produced by the test cipher")
        return ciphertext[4:]


def _run_config() -> RunnableConfig:
    return {
        "configurable": {
            "thread_id": "thread-1",
            "model_name": _BENIGN_MODEL,
            "attempt": _BENIGN_ATTEMPT,
            "__private": _PRIVATE,
            **_SENSITIVE,
        },
        "metadata": {"run_label": _BENIGN_RUN},
    }


def _saver_config() -> RunnableConfig:
    config = _run_config()
    config["configurable"] = {**config["configurable"], "checkpoint_ns": ""}
    return config


def _assert_clean(metadata: Mapping[str, Any] | None) -> None:
    assert metadata is not None
    for name in _SENSITIVE:
        assert name not in metadata
    rendered = json.dumps(metadata)
    for text in _SENSITIVE_TEXTS:
        assert text not in rendered
    assert _PRIVATE not in rendered
    assert "__private" not in metadata
    assert "thread_id" not in metadata
    assert metadata["model_name"] == _BENIGN_MODEL
    assert metadata["attempt"] == _BENIGN_ATTEMPT
    assert metadata["run_label"] == _BENIGN_RUN


def _as_bytes(value: Any) -> bytes:
    if isinstance(value, memoryview):
        return value.tobytes()
    if isinstance(value, bytes):
        return value
    if isinstance(value, str):
        return value.encode()
    raise AssertionError(f"unexpected metadata column type: {type(value)!r}")


def _assert_raw_metadata_clean(raw: bytes) -> None:
    for text in _SENSITIVE_TEXTS:
        assert text.encode() not in raw
    assert _PRIVATE.encode() not in raw


def _assert_sqlite_rows(conn: sqlite3.Connection, *, encrypted: bool = False) -> None:
    rows = list(conn.execute("SELECT type, checkpoint, metadata FROM checkpoints"))
    assert rows
    for stored_type, checkpoint, metadata in rows:
        raw = _as_bytes(metadata)
        _assert_raw_metadata_clean(raw)
        parsed = json.loads(raw.decode())
        _assert_clean(parsed)
        if encrypted:
            assert isinstance(stored_type, str) and "+prefix" in stored_type
            assert _as_bytes(checkpoint).startswith(b"ENC:")


class _State(TypedDict):
    value: int


def _add_one(state: _State) -> _State:
    return {"value": state["value"] + 1}


def _graph(saver: Any) -> Any:
    builder = StateGraph(_State)
    builder.add_node("add_one", _add_one)
    builder.add_edge(START, "add_one")
    builder.add_edge("add_one", END)
    return builder.compile(checkpointer=saver)


def _put_metadata() -> dict[str, Any]:
    return {"source": "input", "step": 1, "writes": {"ignored": True}}


def test_get_checkpoint_metadata_omits_sensitive_configurable() -> None:
    metadata = get_checkpoint_metadata(_saver_config(), _put_metadata())
    _assert_clean(metadata)
    assert metadata["source"] == "input"
    assert metadata["step"] == 1
    assert "writes" in metadata


def test_get_serializable_checkpoint_metadata_omits_sensitive_configurable() -> None:
    metadata = get_serializable_checkpoint_metadata(_saver_config(), _put_metadata())
    _assert_clean(metadata)
    assert "writes" not in metadata


def test_postgres_jsonb_metadata_payload_omits_sensitive_configurable() -> None:
    """Bytes ``Jsonb`` would send for the metadata argument of a Postgres put."""
    metadata = get_serializable_checkpoint_metadata(_saver_config(), _put_metadata())
    raw = JsonbDumper(Jsonb).dump(Jsonb(metadata))
    assert isinstance(raw, bytes)
    _assert_raw_metadata_clean(raw)
    _assert_clean(json.loads(raw.decode()))


def test_memory_put_omits_sensitive_configurable() -> None:
    saver = InMemorySaver()
    config = _saver_config()
    saver.put(config, empty_checkpoint(), _put_metadata(), {})
    saved = saver.get_tuple(config)
    assert saved is not None
    _assert_clean(saved.metadata)
    for namespaces in saver.storage.values():
        for checkpoints in namespaces.values():
            for _checkpoint_id, (
                _checkpoint,
                metadata_blob,
                _parent,
            ) in checkpoints.items():
                _assert_raw_metadata_clean(metadata_blob[1])
                _assert_clean(saver.serde.loads_typed(metadata_blob))


def test_memory_aput_omits_sensitive_configurable() -> None:
    async def inner() -> None:
        saver = InMemorySaver()
        config = _saver_config()
        await saver.aput(config, empty_checkpoint(), _put_metadata(), {})
        saved = await saver.aget_tuple(config)
        assert saved is not None
        _assert_clean(saved.metadata)

    asyncio.run(inner())


def test_sqlite_put_omits_sensitive_configurable() -> None:
    with SqliteSaver.from_conn_string(":memory:") as saver:
        config = _saver_config()
        saver.put(config, empty_checkpoint(), _put_metadata(), {})
        saved = saver.get_tuple(config)
        assert saved is not None
        _assert_clean(saved.metadata)
        listed = list(saver.list(config))
        assert len(listed) == 1
        _assert_clean(listed[0].metadata)
        _assert_sqlite_rows(saver.conn)


def test_async_sqlite_aput_omits_sensitive_configurable(tmp_path: Any) -> None:
    path = tmp_path / "async-put.sqlite"

    async def inner() -> None:
        async with AsyncSqliteSaver.from_conn_string(str(path)) as saver:
            config = _saver_config()
            await saver.aput(config, empty_checkpoint(), _put_metadata(), {})
            saved = await saver.aget_tuple(config)
            assert saved is not None
            _assert_clean(saved.metadata)

    asyncio.run(inner())
    conn = sqlite3.connect(path)
    try:
        _assert_sqlite_rows(conn)
    finally:
        conn.close()


def test_encrypted_sqlite_put_omits_sensitive_configurable() -> None:
    serde = EncryptedSerializer(_PrefixCipher())
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    try:
        saver = SqliteSaver(conn, serde=serde)
        config = _saver_config()
        saver.put(config, empty_checkpoint(), _put_metadata(), {})
        saved = saver.get_tuple(config)
        assert saved is not None
        _assert_clean(saved.metadata)
        _assert_sqlite_rows(conn, encrypted=True)
    finally:
        conn.close()


def test_graph_invoke_sqlite_omits_sensitive_configurable(tmp_path: Any) -> None:
    path = tmp_path / "invoke.sqlite"
    config = _run_config()
    with SqliteSaver.from_conn_string(str(path)) as saver:
        graph = _graph(saver)
        assert graph.invoke({"value": 1}, config) == {"value": 2}
        _assert_clean(graph.get_state(config).metadata)
        history = list(graph.get_state_history(config))
        assert history
        for snapshot in history:
            _assert_clean(snapshot.metadata)
        _assert_sqlite_rows(saver.conn)
    reread = sqlite3.connect(path)
    try:
        _assert_sqlite_rows(reread)
    finally:
        reread.close()


def test_graph_ainvoke_async_sqlite_omits_sensitive_configurable(tmp_path: Any) -> None:
    path = tmp_path / "ainvoke.sqlite"
    config = _run_config()

    async def inner() -> None:
        async with AsyncSqliteSaver.from_conn_string(str(path)) as saver:
            graph = _graph(saver)
            assert await graph.ainvoke({"value": 1}, config) == {"value": 2}
            _assert_clean((await graph.aget_state(config)).metadata)
            history = [item async for item in graph.aget_state_history(config)]
            assert history
            for snapshot in history:
                _assert_clean(snapshot.metadata)

    asyncio.run(inner())
    conn = sqlite3.connect(path)
    try:
        _assert_sqlite_rows(conn)
    finally:
        conn.close()


def test_graph_invoke_encrypted_sqlite_omits_sensitive_configurable(
    tmp_path: Any,
) -> None:
    path = tmp_path / "encrypted.sqlite"
    config = _run_config()
    conn = sqlite3.connect(path, check_same_thread=False)
    try:
        saver = SqliteSaver(conn, serde=EncryptedSerializer(_PrefixCipher()))
        graph = _graph(saver)
        assert graph.invoke({"value": 1}, config) == {"value": 2}
        _assert_clean(graph.get_state(config).metadata)
        history = list(graph.get_state_history(config))
        assert history
        for snapshot in history:
            _assert_clean(snapshot.metadata)
        _assert_sqlite_rows(conn, encrypted=True)
    finally:
        conn.close()


def test_graph_invoke_memory_omits_sensitive_configurable() -> None:
    saver = InMemorySaver()
    config = _run_config()
    graph = _graph(saver)
    assert graph.invoke({"value": 1}, config) == {"value": 2}
    _assert_clean(graph.get_state(config).metadata)
    history = list(graph.get_state_history(config))
    assert history
    for snapshot in history:
        _assert_clean(snapshot.metadata)
    for namespaces in saver.storage.values():
        for checkpoints in namespaces.values():
            for _checkpoint_id, (
                _checkpoint,
                metadata_blob,
                _parent,
            ) in checkpoints.items():
                _assert_raw_metadata_clean(metadata_blob[1])
