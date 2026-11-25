"""Bound wildcard expansion in the SQLite store before embeddings are requested.

``SqliteStore.put`` and ``AsyncSqliteStore.aput`` extract every wildcard match
and insert one ``store_vectors`` row per match. These tests run that path
against a real SQLite database and require the embedding call and the inserted
rows to stay within ``MAX_EMBEDDED_TEXTS_PER_PUT``.
"""

from __future__ import annotations

import sqlite3

from langchain_core.embeddings import Embeddings

from langgraph.store.sqlite import SqliteStore
from langgraph.store.sqlite.aio import AsyncSqliteStore
from langgraph.store.sqlite.base import SqliteIndexConfig

# One put may not embed or insert more texts than this, per put and per path.
MAX_EMBEDDED_TEXTS_PER_PUT = 32
_OVER_CAP = MAX_EMBEDDED_TEXTS_PER_PUT * 4


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


def _texts(count: int, prefix: str = "chunk") -> list[str]:
    return [f"{prefix}-{i:05d}" for i in range(count)]


def _index(
    embeddings: RecordingEmbeddings, fields: list[str] | None = None
) -> SqliteIndexConfig:
    config: SqliteIndexConfig = {"dims": 2, "embed": embeddings}
    if fields is not None:
        config["text_fields"] = fields
    return config


def _vector_count(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT COUNT(*) FROM store_vectors").fetchone()
    return int(row[0])


def _assert_bounded(embeddings: RecordingEmbeddings, stored: int) -> None:
    embedded = embeddings.embedded_texts()
    assert len(embedded) <= MAX_EMBEDDED_TEXTS_PER_PUT, (
        f"embed_documents received {len(embedded)} texts; "
        f"cap is {MAX_EMBEDDED_TEXTS_PER_PUT}"
    )
    assert all(len(batch) <= MAX_EMBEDDED_TEXTS_PER_PUT for batch in embeddings.batches)
    assert stored <= MAX_EMBEDDED_TEXTS_PER_PUT, (
        f"inserted {stored} vector rows; cap is {MAX_EMBEDDED_TEXTS_PER_PUT}"
    )


def _open_sync(
    embeddings: RecordingEmbeddings, fields: list[str] | None = None
) -> SqliteStore:
    store = SqliteStore(
        sqlite3.connect(":memory:", check_same_thread=False, isolation_level=None),
        index=_index(embeddings, fields),
    )
    store.setup()
    return store


def test_put_array_wildcard_does_not_embed_or_insert_every_element() -> None:
    embeddings = RecordingEmbeddings()
    store = _open_sync(embeddings)
    try:
        store.put(("docs",), "item", {"items": _texts(_OVER_CAP)}, index=["items[*]"])
    except Exception:
        pass
    _assert_bounded(embeddings, _vector_count(store.conn))
    store.conn.close()


def test_put_nested_wildcard_does_not_embed_or_insert_every_element() -> None:
    embeddings = RecordingEmbeddings()
    store = _open_sync(embeddings)
    per_side = MAX_EMBEDDED_TEXTS_PER_PUT
    value = {
        "chapters": [
            {"sections": _texts(per_side, prefix=f"c{chapter}")}
            for chapter in range(per_side)
        ]
    }
    try:
        store.put(("docs",), "book", value, index=["chapters[*].sections[*]"])
    except Exception:
        pass
    _assert_bounded(embeddings, _vector_count(store.conn))
    store.conn.close()


def test_configured_wildcard_field_does_not_embed_or_insert_every_element() -> None:
    embeddings = RecordingEmbeddings()
    store = _open_sync(embeddings, fields=["items[*]"])
    try:
        store.put(("docs",), "configured", {"items": _texts(_OVER_CAP)})
    except Exception:
        pass
    _assert_bounded(embeddings, _vector_count(store.conn))
    store.conn.close()


def test_small_array_wildcard_is_still_embedded_and_inserted() -> None:
    embeddings = RecordingEmbeddings()
    store = _open_sync(embeddings)
    expected = _texts(3)
    store.put(("docs",), "small", {"items": expected}, index=["items[*]"])
    assert embeddings.embedded_texts() == expected
    assert _vector_count(store.conn) == len(expected)
    store.conn.close()


def test_index_false_skips_embeddings_and_vector_rows() -> None:
    embeddings = RecordingEmbeddings()
    store = _open_sync(embeddings)
    store.put(("docs",), "plain", {"items": _texts(_OVER_CAP)}, index=False)
    assert embeddings.embedded_texts() == []
    assert _vector_count(store.conn) == 0
    store.conn.close()


async def test_aput_array_wildcard_does_not_embed_or_insert_every_element() -> None:
    embeddings = RecordingEmbeddings()
    async with AsyncSqliteStore.from_conn_string(
        ":memory:", index=_index(embeddings)
    ) as store:
        await store.setup()

        async def _put() -> None:
            await store.aput(
                ("docs",), "item", {"items": _texts(_OVER_CAP)}, index=["items[*]"]
            )

        try:
            await _put()
        except Exception:
            pass
        cursor = await store.conn.execute("SELECT COUNT(*) FROM store_vectors")
        row = await cursor.fetchone()
        stored = int(row[0]) if row is not None else 0
        _assert_bounded(embeddings, stored)
