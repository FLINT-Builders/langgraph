"""Numeric search filters must compare JSON numbers as numbers.

Postgres store search turns `$gt`, `$gte`, `$lt`, and `$lte` into text
comparisons (`value->>key` compared to `str(filter)`). These tests call the
public put/search APIs and expect numeric ordering, matching the in-memory
and SQLite stores.

They fail while that text comparison is in place. A stored score of 50 matches
`$gt` 100, and a stored score of 10 does not match `$gt` 3.2. The same
operators are checked on the plain search query, the TTL-refresh wrapper, and
the vector-search query, for both the sync and async stores.

Pipeline and pool connections are not given their own cases. They execute the
SQL built by `BasePostgresStore._prepare_batch_search_queries`.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import contextmanager
from typing import Any
from uuid import uuid4

from psycopg import AsyncConnection, Connection

from langgraph.store.postgres import AsyncPostgresStore, PostgresStore
from langgraph.store.postgres.base import PostgresIndexConfig
from tests.conftest import DEFAULT_URI, CharacterEmbeddings

NAMESPACE = ("scores",)
OTHER_NAMESPACE = ("other",)
LABEL = "shared-label"

# Values chosen so lexicographic order and numeric order disagree.
# Includes the reported cases: 50 compared with 100, and 10 compared with 3.2.
ROWS: tuple[tuple[str, float], ...] = (
    ("n_neg_20", -20),
    ("n_neg_1", -1),
    ("n_2", 2),
    ("n_3_2", 3.2),
    ("n_9_5", 9.5),
    ("n_10", 10),
    ("n_50", 50),
    ("n_100", 100),
    ("n_150", 150),
    ("n_999", 999),
    ("n_1000", 1000),
)

OPERATORS = ("$gt", "$gte", "$lt", "$lte")
# 100 and 3.2 are the thresholds from the reported examples.
THRESHOLDS = (100, 3.2, 10, 9.5, -3, 1000, 50)


def _expected_keys(operator: str, threshold: float) -> set[str]:
    def matches(value: float) -> bool:
        if operator == "$gt":
            return value > threshold
        if operator == "$gte":
            return value >= threshold
        if operator == "$lt":
            return value < threshold
        if operator == "$lte":
            return value <= threshold
        raise AssertionError(operator)

    return {key for key, value in ROWS if matches(value)}


def _collect_mismatches(
    search: Callable[[str, float], set[str]], *, label: str
) -> list[str]:
    mismatches: list[str] = []
    for operator in OPERATORS:
        for threshold in THRESHOLDS:
            got = search(operator, threshold)
            expected = _expected_keys(operator, threshold)
            if got != expected:
                mismatches.append(
                    f"{label}{operator} {threshold}: got {sorted(got)}, "
                    f"expected {sorted(expected)}"
                )
    return mismatches


def _documents() -> list[tuple[tuple[str, ...], str, dict[str, Any]]]:
    docs: list[tuple[tuple[str, ...], str, dict[str, Any]]] = [
        (NAMESPACE, key, {"score": score, "label": LABEL}) for key, score in ROWS
    ]
    # No numeric field. A numeric comparison yields NULL and must not match.
    docs.append((NAMESPACE, "unscored", {"label": LABEL, "note": "no score"}))
    # Different namespace. A high score here must not satisfy the searched prefix.
    docs.append((OTHER_NAMESPACE, "decoy", {"score": 100000, "label": LABEL}))
    return docs


def _admin_and_db() -> tuple[str, str, str]:
    database = f"test_{uuid4().hex[:16]}"
    uri_parts = DEFAULT_URI.split("/")
    uri_base = "/".join(uri_parts[:-1])
    query_params = ""
    if "?" in uri_parts[-1]:
        _, query_params = uri_parts[-1].split("?", 1)
        query_params = "?" + query_params
    return DEFAULT_URI, f"{uri_base}/{database}{query_params}", database


def _vector_index() -> PostgresIndexConfig:
    embeddings = CharacterEmbeddings(dims=8)
    return {
        "dims": embeddings.dims,
        "embed": embeddings,
        "fields": ["label"],
        "distance_type": "cosine",
        "ann_index_config": {"vector_type": "vector", "kind": "hnsw"},
    }


def _drop_database(admin_conn_string: str, database: str) -> None:
    with Connection.connect(admin_conn_string, autocommit=True) as conn:
        conn.execute(
            "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
            "WHERE datname = %s AND pid <> pg_backend_pid()",
            (database,),
        )
        conn.execute(f"DROP DATABASE IF EXISTS {database}")


@contextmanager
def _open_sync_store(
    *, index: PostgresIndexConfig | None = None
) -> Iterator[PostgresStore]:
    admin_conn_string, conn_string, database = _admin_and_db()
    with Connection.connect(admin_conn_string, autocommit=True) as conn:
        conn.execute(f"CREATE DATABASE {database}")
    try:
        with PostgresStore.from_conn_string(conn_string, index=index) as store:
            store.setup()
            yield store
    finally:
        _drop_database(admin_conn_string, database)


async def _open_async_store(
    *, index: PostgresIndexConfig | None = None
) -> AsyncIterator[AsyncPostgresStore]:
    admin_conn_string, conn_string, database = _admin_and_db()
    async with await AsyncConnection.connect(
        admin_conn_string, autocommit=True
    ) as conn:
        await conn.execute(f"CREATE DATABASE {database}")
    try:
        async with AsyncPostgresStore.from_conn_string(
            conn_string, index=index
        ) as store:
            await store.setup()
            yield store
    finally:
        _drop_database(admin_conn_string, database)


def _seed_sync(store: PostgresStore) -> None:
    for namespace, key, value in _documents():
        store.put(namespace, key, value, index=None)


async def _seed_async(store: AsyncPostgresStore) -> None:
    for namespace, key, value in _documents():
        await store.aput(namespace, key, value, index=None)


def test_sync_search_compares_numeric_filters_as_numbers() -> None:
    """Plain search, with and without the TTL refresh wrapper."""
    with _open_sync_store() as store:
        _seed_sync(store)

        mismatches: list[str] = []
        for refresh_ttl in (False, True):

            def search(
                operator: str, threshold: float, *, refresh_ttl: bool = refresh_ttl
            ) -> set[str]:
                rows = store.search(
                    NAMESPACE,
                    filter={"score": {operator: threshold}},
                    limit=50,
                    refresh_ttl=refresh_ttl,
                )
                return {row.key for row in rows}

            mismatches.extend(
                _collect_mismatches(search, label=f"refresh_ttl={refresh_ttl} ")
            )
        assert not mismatches, "numeric filters compared as text:\n" + "\n".join(
            mismatches
        )


def test_sync_vector_search_compares_numeric_filters_as_numbers() -> None:
    """Vector search applies the same filter inside its scored query."""
    with _open_sync_store(index=_vector_index()) as store:
        _seed_sync(store)

        mismatches: list[str] = []
        for refresh_ttl in (False, True):

            def search(
                operator: str, threshold: float, *, refresh_ttl: bool = refresh_ttl
            ) -> set[str]:
                rows = store.search(
                    NAMESPACE,
                    query=LABEL,
                    filter={"score": {operator: threshold}},
                    limit=50,
                    refresh_ttl=refresh_ttl,
                )
                return {row.key for row in rows}

            mismatches.extend(
                _collect_mismatches(search, label=f"refresh_ttl={refresh_ttl} ")
            )
        assert not mismatches, "numeric filters compared as text:\n" + "\n".join(
            mismatches
        )


async def _assert_async_filters(
    store: AsyncPostgresStore, *, query: str | None
) -> None:
    mismatches: list[str] = []
    for refresh_ttl in (False, True):
        for operator in OPERATORS:
            for threshold in THRESHOLDS:
                rows = await store.asearch(
                    NAMESPACE,
                    query=query,
                    filter={"score": {operator: threshold}},
                    limit=50,
                    refresh_ttl=refresh_ttl,
                )
                got = {row.key for row in rows}
                expected = _expected_keys(operator, threshold)
                if got != expected:
                    mismatches.append(
                        f"{operator} {threshold} refresh_ttl={refresh_ttl}: "
                        f"got {sorted(got)}, expected {sorted(expected)}"
                    )
    assert not mismatches, "numeric filters compared as text:\n" + "\n".join(mismatches)


async def test_async_search_compares_numeric_filters_as_numbers() -> None:
    """Async store search uses the same filter SQL."""
    async for store in _open_async_store():
        await _seed_async(store)
        await _assert_async_filters(store, query=None)
        break


async def test_async_vector_search_compares_numeric_filters_as_numbers() -> None:
    """Async vector search applies the same numeric filter."""
    async for store in _open_async_store(index=_vector_index()):
        await _seed_async(store)
        await _assert_async_filters(store, query=LABEL)
        break
