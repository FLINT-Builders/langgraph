# type: ignore
"""Namespace segments that contain dots must not alias other tuples.

These tests fail while a dot inside a label is stored and queried as the
namespace separator. A dotted label such as ("tenant", "alice.smith") and the
split tuple ("tenant", "alice", "smith") have to stay independent for get,
put, delete, search, list_namespaces, and batch, on both the sync and async
stores.

Rejecting a dotted label is acceptable only when the rejection happens before
any read, write, or TTL refresh. Accepting it is acceptable only when the
stored row round-trips as that same tuple.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import Any
from uuid import uuid4

import pytest
from langgraph.store.base import GetOp, InvalidNamespaceError, PutOp
from psycopg import AsyncConnection, Connection

from langgraph.store.postgres import AsyncPostgresStore, PostgresStore
from tests.conftest import DEFAULT_URI

SPLIT = ("tenant", "alice", "smith")
DOTTED = ("tenant", "alice.smith")
KEY = "memory"
VICTIM = {"owner": "split"}
DOTTED_VALUE = {"owner": "dotted"}


def _database_url(database: str) -> str:
    uri_parts = DEFAULT_URI.split("/")
    uri_base = "/".join(uri_parts[:-1])
    query_params = ""
    if "?" in uri_parts[-1]:
        _, query_params = uri_parts[-1].split("?", 1)
        query_params = "?" + query_params
    return f"{uri_base}/{database}{query_params}"


@pytest.fixture
def store() -> Iterator[PostgresStore]:
    database = f"test_{uuid4().hex[:16]}"
    conn_string = _database_url(database)
    with Connection.connect(DEFAULT_URI, autocommit=True) as conn:
        conn.execute(f"CREATE DATABASE {database}")
    try:
        with PostgresStore.from_conn_string(conn_string) as store:
            store.setup()
            yield store
    finally:
        with Connection.connect(DEFAULT_URI, autocommit=True) as conn:
            conn.execute(f"DROP DATABASE {database} WITH (FORCE)")


@pytest.fixture
async def async_store() -> AsyncIterator[AsyncPostgresStore]:
    database = f"test_{uuid4().hex[:16]}"
    conn_string = _database_url(database)
    async with await AsyncConnection.connect(DEFAULT_URI, autocommit=True) as conn:
        await conn.execute(f"CREATE DATABASE {database}")
    try:
        async with AsyncPostgresStore.from_conn_string(conn_string) as store:
            await store.setup()
            yield store
    finally:
        async with await AsyncConnection.connect(DEFAULT_URI, autocommit=True) as conn:
            await conn.execute(f"DROP DATABASE {database} WITH (FORCE)")


def _expires_at(store: PostgresStore, key: str) -> Any:
    with store._cursor() as cur:
        cur.execute("SELECT expires_at FROM store WHERE key = %s", (key,))
        row = cur.fetchone()
    if row is None:
        return None
    return row["expires_at"]


async def _aexpires_at(store: AsyncPostgresStore, key: str) -> Any:
    async with store._cursor() as cur:
        await cur.execute("SELECT expires_at FROM store WHERE key = %s", (key,))
        row = await cur.fetchone()
    if row is None:
        return None
    return row["expires_at"]


def _get(store: PostgresStore, namespace: tuple[str, ...], key: str) -> Any:
    try:
        return store.get(namespace, key)
    except InvalidNamespaceError:
        return None


async def _aget(store: AsyncPostgresStore, namespace: tuple[str, ...], key: str) -> Any:
    try:
        return await store.aget(namespace, key)
    except InvalidNamespaceError:
        return None


def _search(store: PostgresStore, prefix: tuple[str, ...], **kwargs: Any) -> list[Any]:
    try:
        return list(store.search(prefix, limit=100, **kwargs))
    except InvalidNamespaceError:
        return []


async def _asearch(
    store: AsyncPostgresStore, prefix: tuple[str, ...], **kwargs: Any
) -> list[Any]:
    try:
        return list(await store.asearch(prefix, limit=100, **kwargs))
    except InvalidNamespaceError:
        return []


def _delete(store: PostgresStore, namespace: tuple[str, ...], key: str) -> None:
    try:
        store.delete(namespace, key)
    except InvalidNamespaceError:
        return


async def _adelete(
    store: AsyncPostgresStore, namespace: tuple[str, ...], key: str
) -> None:
    try:
        await store.adelete(namespace, key)
    except InvalidNamespaceError:
        return


def _list_namespaces(store: PostgresStore, **kwargs: Any) -> list[tuple[str, ...]]:
    try:
        return list(store.list_namespaces(limit=100, **kwargs))
    except InvalidNamespaceError:
        return []


async def _alist_namespaces(
    store: AsyncPostgresStore, **kwargs: Any
) -> list[tuple[str, ...]]:
    try:
        return list(await store.alist_namespaces(limit=100, **kwargs))
    except InvalidNamespaceError:
        return []


def _store(
    store: PostgresStore, namespace: tuple[str, ...], key: str, value: dict
) -> bool:
    """Store `value` or return False when the namespace is rejected first.

    `put` validates labels. `batch` is the public operation `put` delegates to
    after that check, and it currently accepts the same labels. Either path
    may reject, but neither may write before refusing.
    """
    try:
        store.put(namespace, key, value)
        return True
    except InvalidNamespaceError:
        pass
    try:
        store.batch([PutOp(namespace=namespace, key=key, value=value)])
    except InvalidNamespaceError:
        return False
    # A rejection that still inserted a row is a failed refusal.
    item = _get(store, namespace, key)
    if item is None:
        return False
    return True


async def _astore(
    store: AsyncPostgresStore, namespace: tuple[str, ...], key: str, value: dict
) -> bool:
    try:
        await store.aput(namespace, key, value)
        return True
    except InvalidNamespaceError:
        pass
    try:
        await store.abatch([PutOp(namespace=namespace, key=key, value=value)])
    except InvalidNamespaceError:
        return False
    item = await _aget(store, namespace, key)
    if item is None:
        return False
    return True


def test_dotted_get_does_not_read_or_refresh_split_namespace(
    store: PostgresStore,
) -> None:
    store.put(SPLIT, KEY, VICTIM, ttl=60)
    before = _expires_at(store, KEY)
    assert before is not None

    item = _get(store, DOTTED, KEY)
    after = _expires_at(store, KEY)

    assert item is None
    assert after == before
    assert store.get(SPLIT, KEY).value == VICTIM


def test_dotted_delete_does_not_remove_split_namespace(store: PostgresStore) -> None:
    store.put(SPLIT, KEY, VICTIM)
    _delete(store, DOTTED, KEY)
    assert store.get(SPLIT, KEY).value == VICTIM


def test_dotted_search_does_not_return_split_namespace(store: PostgresStore) -> None:
    store.put(SPLIT, KEY, VICTIM, ttl=60)
    before = _expires_at(store, KEY)

    assert _search(store, DOTTED) == []
    assert _search(store, DOTTED, filter={"owner": "split"}) == []
    assert _expires_at(store, KEY) == before
    assert store.get(SPLIT, KEY).value == VICTIM


def test_dotted_batch_get_and_delete_do_not_touch_split_namespace(
    store: PostgresStore,
) -> None:
    store.put(SPLIT, KEY, VICTIM)
    try:
        results = store.batch([GetOp(namespace=DOTTED, key=KEY, refresh_ttl=False)])
    except InvalidNamespaceError:
        results = [None]
    assert results == [None]

    try:
        store.batch([PutOp(namespace=DOTTED, key=KEY, value=None)])
    except InvalidNamespaceError:
        pass
    assert store.get(SPLIT, KEY).value == VICTIM


def test_prefix_search_does_not_treat_dotted_label_as_descendant(
    store: PostgresStore,
) -> None:
    stored = _store(store, DOTTED, KEY, DOTTED_VALUE)
    single = ("alice.smith",)
    single_stored = _store(store, single, KEY, {"owner": "single"})

    tenant_hits = _search(store, ("tenant", "alice"))
    assert [item.namespace for item in tenant_hits] == []
    assert _search(store, ("alice",)) == []

    if stored:
        exact = _search(store, DOTTED)
        assert [item.namespace for item in exact] == [DOTTED]
        assert exact[0].value == DOTTED_VALUE
    if single_stored:
        exact_single = _search(store, single)
        assert [item.namespace for item in exact_single] == [single]
        assert exact_single[0].value == {"owner": "single"}


def test_put_on_split_namespace_does_not_overwrite_dotted_label(
    store: PostgresStore,
) -> None:
    stored = _store(store, DOTTED, KEY, DOTTED_VALUE)
    store.put(SPLIT, KEY, VICTIM)

    assert store.get(SPLIT, KEY).value == VICTIM
    item = _get(store, DOTTED, KEY)
    if stored:
        assert item is not None
        assert item.namespace == DOTTED
        assert item.value == DOTTED_VALUE
    else:
        assert item is None


def test_list_namespaces_does_not_split_dotted_labels(store: PostgresStore) -> None:
    stored = _store(store, DOTTED, KEY, DOTTED_VALUE)

    by_prefix = _list_namespaces(store, prefix=DOTTED)
    by_parent = _list_namespaces(store, prefix=("tenant", "alice"))
    by_suffix = _list_namespaces(store, suffix=("smith",))
    by_depth = _list_namespaces(store, prefix=("tenant",), max_depth=2)

    assert SPLIT not in by_prefix
    assert SPLIT not in by_parent
    assert SPLIT not in by_suffix
    assert ("tenant", "alice") not in by_depth
    if stored:
        assert by_prefix == [DOTTED]
        assert by_parent == []
        assert by_suffix == []
        assert DOTTED in by_depth
    else:
        assert by_prefix == []
        assert by_parent == []
        assert by_suffix == []
        assert by_depth == []


def test_distinct_dotted_groupings_do_not_share_a_row(store: PostgresStore) -> None:
    shapes = [
        (("a.b", "c"), {"owner": "ab-c"}),
        (("a", "b.c"), {"owner": "a-bc"}),
        (("a", "b", "c"), {"owner": "a-b-c"}),
    ]
    stored = {ns: _store(store, ns, KEY, value) for ns, value in shapes}

    for namespace, value in shapes:
        item = _get(store, namespace, KEY)
        if stored[namespace]:
            assert item is not None
            assert item.namespace == namespace
            assert item.value == value
        else:
            assert item is None


def test_legitimate_descendant_search_still_matches(store: PostgresStore) -> None:
    child = ("tenant", "alice", "extra")
    store.put(child, KEY, {"owner": "child"})
    assert child in {
        item.namespace for item in store.search(("tenant", "alice"), limit=100)
    }
    assert child in {item.namespace for item in store.search(("tenant",), limit=100)}
    assert child in set(store.list_namespaces(prefix=("tenant", "alice")))
    assert child in set(store.list_namespaces(prefix=("tenant",)))


async def test_async_dotted_get_does_not_read_or_refresh_split_namespace(
    async_store: AsyncPostgresStore,
) -> None:
    await async_store.aput(SPLIT, KEY, VICTIM, ttl=60)
    before = await _aexpires_at(async_store, KEY)
    assert before is not None

    item = await _aget(async_store, DOTTED, KEY)
    after = await _aexpires_at(async_store, KEY)

    assert item is None
    assert after == before
    assert (await async_store.aget(SPLIT, KEY)).value == VICTIM


async def test_async_dotted_delete_does_not_remove_split_namespace(
    async_store: AsyncPostgresStore,
) -> None:
    await async_store.aput(SPLIT, KEY, VICTIM)
    await _adelete(async_store, DOTTED, KEY)
    assert (await async_store.aget(SPLIT, KEY)).value == VICTIM


async def test_async_dotted_search_does_not_return_split_namespace(
    async_store: AsyncPostgresStore,
) -> None:
    await async_store.aput(SPLIT, KEY, VICTIM, ttl=60)
    before = await _aexpires_at(async_store, KEY)

    assert await _asearch(async_store, DOTTED) == []
    assert await _asearch(async_store, DOTTED, filter={"owner": "split"}) == []
    assert await _aexpires_at(async_store, KEY) == before
    assert (await async_store.aget(SPLIT, KEY)).value == VICTIM


async def test_async_dotted_batch_get_and_delete_do_not_touch_split_namespace(
    async_store: AsyncPostgresStore,
) -> None:
    await async_store.aput(SPLIT, KEY, VICTIM)
    try:
        results = await async_store.abatch(
            [GetOp(namespace=DOTTED, key=KEY, refresh_ttl=False)]
        )
    except InvalidNamespaceError:
        results = [None]
    assert results == [None]

    try:
        await async_store.abatch([PutOp(namespace=DOTTED, key=KEY, value=None)])
    except InvalidNamespaceError:
        pass
    assert (await async_store.aget(SPLIT, KEY)).value == VICTIM


async def test_async_prefix_search_does_not_treat_dotted_label_as_descendant(
    async_store: AsyncPostgresStore,
) -> None:
    stored = await _astore(async_store, DOTTED, KEY, DOTTED_VALUE)

    tenant_hits = await _asearch(async_store, ("tenant", "alice"))
    assert [item.namespace for item in tenant_hits] == []
    if stored:
        exact = await _asearch(async_store, DOTTED)
        assert [item.namespace for item in exact] == [DOTTED]
        assert exact[0].value == DOTTED_VALUE


async def test_async_put_on_split_namespace_does_not_overwrite_dotted_label(
    async_store: AsyncPostgresStore,
) -> None:
    stored = await _astore(async_store, DOTTED, KEY, DOTTED_VALUE)
    await async_store.aput(SPLIT, KEY, VICTIM)

    assert (await async_store.aget(SPLIT, KEY)).value == VICTIM
    item = await _aget(async_store, DOTTED, KEY)
    if stored:
        assert item is not None
        assert item.namespace == DOTTED
        assert item.value == DOTTED_VALUE
    else:
        assert item is None


async def test_async_list_namespaces_does_not_split_dotted_labels(
    async_store: AsyncPostgresStore,
) -> None:
    stored = await _astore(async_store, DOTTED, KEY, DOTTED_VALUE)

    by_prefix = await _alist_namespaces(async_store, prefix=DOTTED)
    by_parent = await _alist_namespaces(async_store, prefix=("tenant", "alice"))
    by_suffix = await _alist_namespaces(async_store, suffix=("smith",))
    by_depth = await _alist_namespaces(async_store, prefix=("tenant",), max_depth=2)

    assert SPLIT not in by_prefix
    assert SPLIT not in by_parent
    assert SPLIT not in by_suffix
    assert ("tenant", "alice") not in by_depth
    if stored:
        assert by_prefix == [DOTTED]
        assert by_parent == []
        assert by_suffix == []
        assert DOTTED in by_depth
    else:
        assert by_prefix == []
        assert by_parent == []
        assert by_suffix == []
        assert by_depth == []


async def test_async_distinct_dotted_groupings_do_not_share_a_row(
    async_store: AsyncPostgresStore,
) -> None:
    shapes = [
        (("a.b", "c"), {"owner": "ab-c"}),
        (("a", "b.c"), {"owner": "a-bc"}),
        (("a", "b", "c"), {"owner": "a-b-c"}),
    ]
    stored = {ns: await _astore(async_store, ns, KEY, value) for ns, value in shapes}

    for namespace, value in shapes:
        item = await _aget(async_store, namespace, KEY)
        if stored[namespace]:
            assert item is not None
            assert item.namespace == namespace
            assert item.value == value
        else:
            assert item is None
