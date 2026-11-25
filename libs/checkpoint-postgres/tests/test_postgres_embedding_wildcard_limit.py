"""Bound wildcard expansion in the Postgres store before embeddings are requested.

``PostgresStore.batch`` and ``AsyncPostgresStore.abatch`` both build one
embedding input and one ``store_vectors`` row per wildcard match, with no
maximum. These tests drive that put path and require the embedding call and
the vector insert to stay within ``MAX_EMBEDDED_TEXTS_PER_PUT``.

The database cursor records the statement the store executes. It does not
decide how many texts were extracted. A live server is not required: the
expansion and the ``embed_documents`` call happen in the store before
``cursor.execute``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest
from langchain_core.embeddings import Embeddings
from langgraph.store.base import PutOp

from langgraph.store.postgres import PostgresStore
from langgraph.store.postgres.aio import AsyncPostgresStore

# One put may not embed or stage more vector rows than this, per put and per path.
MAX_EMBEDDED_TEXTS_PER_PUT = 32
_OVER_CAP = MAX_EMBEDDED_TEXTS_PER_PUT * 4


@pytest.fixture(autouse=True)
def clear_test_db() -> None:
    """Do not open the shared Postgres fixture from ``tests/conftest.py``.

    These tests call the store's put implementation with a recording cursor.
    They do not need a database server.
    """
    return None


class RecordingEmbeddings(Embeddings):
    def __init__(self) -> None:
        self.batches: list[list[str]] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        copied = list(texts)
        self.batches.append(copied)
        return [[1.0, float(i)] for i, _ in enumerate(copied)]

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0]

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed_documents(texts)

    async def aembed_query(self, text: str) -> list[float]:
        return self.embed_query(text)

    def embedded_texts(self) -> list[str]:
        return [text for batch in self.batches for text in batch]


class RecordingCursor:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Sequence[Any] | None]] = []

    def execute(self, query: str, params: Sequence[Any] | None = None) -> None:
        self.calls.append((query, params))


class AsyncRecordingCursor:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Sequence[Any] | None]] = []

    async def execute(self, query: str, params: Sequence[Any] | None = None) -> None:
        self.calls.append((query, params))


def _texts(count: int, prefix: str = "chunk") -> list[str]:
    return [f"{prefix}-{i:05d}" for i in range(count)]


def _vector_rows(calls: list[tuple[str, Sequence[Any] | None]]) -> int:
    total = 0
    for query, params in calls:
        if "store_vectors" not in query or not params:
            continue
        # Each vector row is (prefix, key, field_name, embedding).
        total += len(params) // 4
    return total


def _assert_bounded(
    embeddings: RecordingEmbeddings,
    calls: list[tuple[str, Sequence[Any] | None]],
) -> None:
    embedded = embeddings.embedded_texts()
    stored = _vector_rows(calls)
    assert len(embedded) <= MAX_EMBEDDED_TEXTS_PER_PUT, (
        f"embed_documents received {len(embedded)} texts; "
        f"cap is {MAX_EMBEDDED_TEXTS_PER_PUT}"
    )
    assert all(len(batch) <= MAX_EMBEDDED_TEXTS_PER_PUT for batch in embeddings.batches)
    assert stored <= MAX_EMBEDDED_TEXTS_PER_PUT, (
        f"vector insert staged {stored} rows; cap is {MAX_EMBEDDED_TEXTS_PER_PUT}"
    )


def _sync_store(
    embeddings: RecordingEmbeddings, fields: list[str] | None = None
) -> PostgresStore:
    config: dict = {"dims": 2, "embed": embeddings}
    if fields is not None:
        config["fields"] = fields
    return PostgresStore(conn=object(), index=config)  # type: ignore[arg-type]


def _put_op(value: dict, index: list[str] | None) -> list[tuple[int, PutOp]]:
    return [(0, PutOp(("docs",), "item", value, index=index))]


def test_sync_put_array_wildcard_does_not_embed_every_element() -> None:
    embeddings = RecordingEmbeddings()
    store = _sync_store(embeddings)
    cursor = RecordingCursor()
    ops = _put_op({"items": _texts(_OVER_CAP)}, ["items[*]"])

    try:
        store._batch_put_ops(ops, cursor)  # type: ignore[arg-type]
    except Exception:
        pass

    _assert_bounded(embeddings, cursor.calls)


async def test_async_put_array_wildcard_does_not_embed_every_element() -> None:
    embeddings = RecordingEmbeddings()
    store = AsyncPostgresStore(conn=object(), index={"dims": 2, "embed": embeddings})  # type: ignore[arg-type]
    cursor = AsyncRecordingCursor()
    ops = _put_op({"items": _texts(_OVER_CAP)}, ["items[*]"])

    try:
        await store._batch_put_ops(ops, cursor)  # type: ignore[arg-type]
    except Exception:
        pass

    _assert_bounded(embeddings, cursor.calls)


def test_sync_put_nested_and_object_wildcards_stay_bounded() -> None:
    embeddings = RecordingEmbeddings()
    store = _sync_store(embeddings)
    cursor = RecordingCursor()
    per_side = MAX_EMBEDDED_TEXTS_PER_PUT
    value = {
        "chapters": [
            {"sections": _texts(per_side, prefix=f"c{chapter}")}
            for chapter in range(per_side)
        ],
        "meta": {f"field-{i:05d}": f"value-{i:05d}" for i in range(_OVER_CAP)},
    }

    try:
        store._batch_put_ops(
            _put_op(value, ["chapters[*].sections[*]", "*"]),
            cursor,  # type: ignore[arg-type]
        )
    except Exception:
        pass

    _assert_bounded(embeddings, cursor.calls)


def test_configured_wildcard_field_does_not_embed_every_element() -> None:
    embeddings = RecordingEmbeddings()
    store = _sync_store(embeddings, fields=["items[*]"])
    cursor = RecordingCursor()

    try:
        store._batch_put_ops(
            _put_op({"items": _texts(_OVER_CAP)}, None),
            cursor,  # type: ignore[arg-type]
        )
    except Exception:
        pass

    _assert_bounded(embeddings, cursor.calls)


def test_small_array_wildcard_is_still_embedded_and_inserted() -> None:
    embeddings = RecordingEmbeddings()
    store = _sync_store(embeddings)
    cursor = RecordingCursor()
    expected = _texts(3)

    store._batch_put_ops(
        _put_op({"items": expected}, ["items[*]"]),
        cursor,  # type: ignore[arg-type]
    )

    assert embeddings.embedded_texts() == expected
    assert _vector_rows(cursor.calls) == len(expected)


def test_index_false_skips_embeddings_and_vector_insert() -> None:
    embeddings = RecordingEmbeddings()
    store = _sync_store(embeddings)
    cursor = RecordingCursor()

    store._batch_put_ops(
        [(0, PutOp(("docs",), "item", {"items": _texts(_OVER_CAP)}, index=False))],
        cursor,  # type: ignore[arg-type]
    )

    assert embeddings.embedded_texts() == []
    assert _vector_rows(cursor.calls) == 0
