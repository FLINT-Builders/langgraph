"""Namespace labels that contain dots must not alias a different tuple.

`("users", "alice.smith")` and `("users", "alice", "smith")` are different
namespaces. The SQLite store joins segments with `.` and splits on every `.`,
so a dotted label reads and writes the other namespace's row.

`put` / `aput` already reject dotted labels. These tests cover the operations
that still apply the unescaped encoding: `get`, `search`, `list_namespaces`,
`delete`, and `batch` / `abatch`, including vector search and TTL refresh.
A fix may keep rejecting dotted labels, or it may escape them. Either way the
dotted label must not observe or modify the split namespace.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from langgraph.store.base import InvalidNamespaceError, PutOp

from langgraph.store.sqlite import AsyncSqliteStore, SqliteStore
from langgraph.store.sqlite.base import SqliteIndexConfig
from tests.test_store import CharacterEmbeddings

SPLIT = ("users", "alice", "smith")
DOTTED = ("users", "alice.smith")
KEY = "memory"
VICTIM = {"secret": "victim-memory"}
ATTACKER = {"secret": "attacker-memory"}


@pytest.fixture
def store() -> Iterator[SqliteStore]:
    with SqliteStore.from_conn_string(":memory:") as store:
        store.setup()
        yield store


def _read(store: SqliteStore, namespace: tuple[str, ...], key: str):
    try:
        return store.get(namespace, key)
    except InvalidNamespaceError:
        return None


def _search(store: SqliteStore, prefix: tuple[str, ...]):
    try:
        return store.search(prefix, limit=100)
    except InvalidNamespaceError:
        return []


def _list(store: SqliteStore, **kwargs):
    try:
        return store.list_namespaces(limit=100, **kwargs)
    except InvalidNamespaceError:
        return []


def _delete(store: SqliteStore, namespace: tuple[str, ...], key: str) -> None:
    try:
        store.delete(namespace, key)
    except InvalidNamespaceError:
        return


def _batch_put(
    store: SqliteStore, namespace: tuple[str, ...], key: str, value: dict
) -> None:
    try:
        store.batch([PutOp(namespace, key, value)])
    except InvalidNamespaceError:
        return


def test_put_rejects_dotted_namespace_without_writing(store: SqliteStore) -> None:
    store.put(SPLIT, KEY, VICTIM)
    with pytest.raises(InvalidNamespaceError):
        store.put(DOTTED, KEY, ATTACKER)
    item = store.get(SPLIT, KEY)
    assert item is not None
    assert item.namespace == SPLIT
    assert item.value == VICTIM


def test_get_dotted_namespace_does_not_read_split_namespace(store: SqliteStore) -> None:
    store.put(SPLIT, KEY, VICTIM)
    assert _read(store, DOTTED, KEY) is None
    item = store.get(SPLIT, KEY)
    assert item is not None
    assert item.namespace == SPLIT
    assert item.value == VICTIM


def test_search_dotted_prefix_does_not_return_split_namespace(
    store: SqliteStore,
) -> None:
    store.put(SPLIT, KEY, VICTIM)
    # A real descendant prefix still matches. A dotted label is not that prefix.
    assert {item.namespace for item in store.search(("users", "alice"), limit=100)} == {
        SPLIT
    }
    assert _search(store, DOTTED) == []


def test_list_namespaces_dotted_patterns_do_not_return_split_namespace(
    store: SqliteStore,
) -> None:
    store.put(SPLIT, KEY, VICTIM)
    assert SPLIT not in _list(store, prefix=DOTTED)
    assert SPLIT not in _list(store, suffix=("alice.smith",))
    assert SPLIT not in _list(store, prefix=("*", "alice.smith"))
    assert ("users", "alice") not in _list(store, prefix=DOTTED, max_depth=2)
    # The same row is still listed under its real namespace.
    assert SPLIT in store.list_namespaces(prefix=("users", "alice"), limit=100)


def test_delete_dotted_namespace_does_not_delete_split_namespace(
    store: SqliteStore,
) -> None:
    store.put(SPLIT, KEY, VICTIM)
    _delete(store, DOTTED, KEY)
    item = store.get(SPLIT, KEY)
    assert item is not None
    assert item.value == VICTIM


def test_batch_put_dotted_namespace_does_not_overwrite_split_namespace(
    store: SqliteStore,
) -> None:
    store.put(SPLIT, KEY, VICTIM)
    _batch_put(store, DOTTED, KEY, ATTACKER)
    item = store.get(SPLIT, KEY)
    assert item is not None
    assert item.namespace == SPLIT
    assert item.value == VICTIM

    _batch_put(store, DOTTED, "other", ATTACKER)
    assert store.get(SPLIT, "other") is None
    dotted = _read(store, DOTTED, "other")
    if dotted is not None:
        assert dotted.namespace == DOTTED
        assert dotted.value == ATTACKER
        assert DOTTED in store.list_namespaces(limit=100)
    assert SPLIT in store.list_namespaces(limit=100)


def test_vector_search_dotted_prefix_does_not_return_split_namespace() -> None:
    embeddings = CharacterEmbeddings(dims=32)
    index: SqliteIndexConfig = {
        "dims": embeddings.dims,
        "embed": embeddings,
        "text_fields": ["text"],
    }
    with SqliteStore.from_conn_string(":memory:", index=index) as store:
        store.setup()
        store.put(SPLIT, KEY, {"text": "victim memory"})
        try:
            hits = store.search(DOTTED, query="victim memory", limit=10)
        except InvalidNamespaceError:
            hits = []
        assert hits == []
        own = store.search(SPLIT, query="victim memory", limit=10)
        assert [item.namespace for item in own] == [SPLIT]


def test_ttl_refresh_on_dotted_read_does_not_touch_split_row() -> None:
    with SqliteStore.from_conn_string(
        ":memory:", ttl={"refresh_on_read": True}
    ) as store:
        store.setup()
        store.put(SPLIT, KEY, VICTIM, ttl=1000)
        store.conn.execute(
            "UPDATE store SET expires_at = ? WHERE prefix = ? AND key = ?",
            ("2000-01-01 00:00:00", "users.alice.smith", KEY),
        )
        try:
            store.get(DOTTED, KEY, refresh_ttl=True)
        except InvalidNamespaceError:
            pass
        try:
            store.search(DOTTED, limit=10, refresh_ttl=True)
        except InvalidNamespaceError:
            pass
        row = store.conn.execute(
            "SELECT expires_at FROM store WHERE prefix = ? AND key = ?",
            ("users.alice.smith", KEY),
        ).fetchone()
        assert row is not None
        assert row[0] == "2000-01-01 00:00:00"


async def _aread(store: AsyncSqliteStore, namespace: tuple[str, ...], key: str):
    try:
        return await store.aget(namespace, key)
    except InvalidNamespaceError:
        return None


async def _asearch(store: AsyncSqliteStore, prefix: tuple[str, ...]):
    try:
        return await store.asearch(prefix, limit=100)
    except InvalidNamespaceError:
        return []


async def _alist(store: AsyncSqliteStore, **kwargs):
    try:
        return await store.alist_namespaces(limit=100, **kwargs)
    except InvalidNamespaceError:
        return []


@pytest.fixture
async def async_store():
    async with AsyncSqliteStore.from_conn_string(":memory:") as store:
        await store.setup()
        yield store


async def test_async_dotted_namespace_does_not_alias_split_namespace(
    async_store: AsyncSqliteStore,
) -> None:
    await async_store.aput(SPLIT, KEY, VICTIM)
    with pytest.raises(InvalidNamespaceError):
        await async_store.aput(DOTTED, KEY, ATTACKER)

    assert await _aread(async_store, DOTTED, KEY) is None
    item = await async_store.aget(SPLIT, KEY)
    assert item is not None
    assert item.namespace == SPLIT
    assert item.value == VICTIM

    assert await _asearch(async_store, DOTTED) == []
    assert SPLIT not in await _alist(async_store, prefix=DOTTED)
    assert SPLIT not in await _alist(async_store, suffix=("alice.smith",))
    assert SPLIT not in await _alist(async_store, prefix=("*", "alice.smith"))

    try:
        await async_store.adelete(DOTTED, KEY)
    except InvalidNamespaceError:
        pass
    item = await async_store.aget(SPLIT, KEY)
    assert item is not None
    assert item.value == VICTIM

    try:
        await async_store.abatch([PutOp(DOTTED, KEY, ATTACKER)])
    except InvalidNamespaceError:
        pass
    item = await async_store.aget(SPLIT, KEY)
    assert item is not None
    assert item.namespace == SPLIT
    assert item.value == VICTIM


async def test_async_ttl_refresh_on_dotted_read_does_not_touch_split_row() -> None:
    async with AsyncSqliteStore.from_conn_string(
        ":memory:", ttl={"refresh_on_read": True}
    ) as store:
        await store.setup()
        await store.aput(SPLIT, KEY, VICTIM, ttl=1000)
        await store.conn.execute(
            "UPDATE store SET expires_at = ? WHERE prefix = ? AND key = ?",
            ("2000-01-01 00:00:00", "users.alice.smith", KEY),
        )
        await store.conn.commit()
        try:
            await store.aget(DOTTED, KEY, refresh_ttl=True)
        except InvalidNamespaceError:
            pass
        try:
            await store.asearch(DOTTED, limit=10, refresh_ttl=True)
        except InvalidNamespaceError:
            pass
        async with store.conn.execute(
            "SELECT expires_at FROM store WHERE prefix = ? AND key = ?",
            ("users.alice.smith", KEY),
        ) as cur:
            row = await cur.fetchone()
        assert row is not None
        assert row[0] == "2000-01-01 00:00:00"


def test_vector_batch_put_dotted_namespace_does_not_overwrite_split_vectors() -> None:
    embeddings = CharacterEmbeddings(dims=32)
    index: SqliteIndexConfig = {
        "dims": embeddings.dims,
        "embed": embeddings,
        "text_fields": ["text"],
    }
    with SqliteStore.from_conn_string(":memory:", index=index) as store:
        store.setup()
        store.put(SPLIT, KEY, {"text": "victim memory"})
        _batch_put(store, DOTTED, KEY, {"text": "attacker memory"})
        own = store.search(SPLIT, query="victim memory", limit=10)
        assert [item.value["text"] for item in own] == ["victim memory"]
        try:
            dotted_hits = store.search(DOTTED, query="attacker memory", limit=10)
        except InvalidNamespaceError:
            dotted_hits = []
        assert all(item.namespace != SPLIT for item in dotted_hits)
        assert all(item.value.get("text") != "victim memory" for item in dotted_hits)
