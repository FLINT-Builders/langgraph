"""Bound wildcard expansion before embeddings are requested.

A store configured for vector search must not turn one ``put`` / ``aput`` into
an unbounded ``embed_documents`` call or an unbounded set of stored vectors.
Caller-supplied index paths such as ``items[*]`` and store-configured field
paths share that extraction loop.

These tests are the spec for that bound. They fail while every matching text
is embedded and stored. A fix should reject or truncate the expansion before
calling the embedding function. A wildcard that matches at most
``MAX_EMBEDDED_TEXTS_PER_PUT`` texts must still be embedded in full.
"""

from __future__ import annotations

from collections.abc import Callable

from langchain_core.embeddings import Embeddings

from langgraph.store.memory import InMemoryStore

# One put may not embed or retain more texts than this, per put and per path.
MAX_EMBEDDED_TEXTS_PER_PUT = 32
_OVER_CAP = MAX_EMBEDDED_TEXTS_PER_PUT * 4


class RecordingEmbeddings(Embeddings):
    """Records every text the store actually submits for embedding."""

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


def _store(
    embeddings: RecordingEmbeddings, fields: list[str] | None = None
) -> InMemoryStore:
    config: dict = {"dims": 2, "embed": embeddings}
    if fields is not None:
        config["fields"] = fields
    return InMemoryStore(index=config)


def _vector_count(store: InMemoryStore, namespace: tuple[str, ...], key: str) -> int:
    return len(store._vectors[namespace].get(key, {}))


def _invoke(operation: Callable[[], None]) -> None:
    try:
        operation()
    except Exception:
        return


def _assert_bounded(
    embeddings: RecordingEmbeddings,
    store: InMemoryStore,
    namespace: tuple[str, ...],
    key: str,
) -> None:
    embedded = embeddings.embedded_texts()
    stored = _vector_count(store, namespace, key)
    assert len(embedded) <= MAX_EMBEDDED_TEXTS_PER_PUT, (
        f"embed_documents received {len(embedded)} texts; "
        f"cap is {MAX_EMBEDDED_TEXTS_PER_PUT}"
    )
    assert all(len(batch) <= MAX_EMBEDDED_TEXTS_PER_PUT for batch in embeddings.batches)
    assert stored <= MAX_EMBEDDED_TEXTS_PER_PUT, (
        f"stored {stored} vectors; cap is {MAX_EMBEDDED_TEXTS_PER_PUT}"
    )


def test_put_array_wildcard_does_not_embed_every_element() -> None:
    embeddings = RecordingEmbeddings()
    store = _store(embeddings)
    namespace = ("docs",)
    key = "item"
    value = {"items": _texts(_OVER_CAP)}

    _invoke(lambda: store.put(namespace, key, value, index=["items[*]"]))

    _assert_bounded(embeddings, store, namespace, key)


async def test_aput_array_wildcard_does_not_embed_every_element() -> None:
    embeddings = RecordingEmbeddings()
    store = _store(embeddings)
    namespace = ("docs",)
    key = "item"
    value = {"items": _texts(_OVER_CAP)}

    async def _put() -> None:
        await store.aput(namespace, key, value, index=["items[*]"])

    try:
        await _put()
    except Exception:
        pass

    _assert_bounded(embeddings, store, namespace, key)


def test_put_nested_wildcard_does_not_embed_every_element() -> None:
    embeddings = RecordingEmbeddings()
    store = _store(embeddings)
    namespace = ("docs",)
    key = "book"
    per_side = MAX_EMBEDDED_TEXTS_PER_PUT
    value = {
        "chapters": [
            {"sections": _texts(per_side, prefix=f"c{chapter}")}
            for chapter in range(per_side)
        ]
    }

    _invoke(lambda: store.put(namespace, key, value, index=["chapters[*].sections[*]"]))

    _assert_bounded(embeddings, store, namespace, key)


def test_put_object_wildcard_does_not_embed_every_value() -> None:
    embeddings = RecordingEmbeddings()
    store = _store(embeddings)
    namespace = ("docs",)
    key = "obj"
    value = {f"field-{i:05d}": f"value-{i:05d}" for i in range(_OVER_CAP)}

    _invoke(lambda: store.put(namespace, key, value, index=["*"]))

    _assert_bounded(embeddings, store, namespace, key)


def test_put_multiple_wildcard_paths_share_one_put_cap() -> None:
    embeddings = RecordingEmbeddings()
    store = _store(embeddings)
    namespace = ("docs",)
    key = "pair"
    value = {
        "left": _texts(_OVER_CAP, prefix="left"),
        "right": _texts(_OVER_CAP, prefix="right"),
    }

    _invoke(lambda: store.put(namespace, key, value, index=["left[*]", "right[*]"]))

    _assert_bounded(embeddings, store, namespace, key)


def test_configured_wildcard_field_does_not_embed_every_element() -> None:
    embeddings = RecordingEmbeddings()
    store = _store(embeddings, fields=["items[*]"])
    namespace = ("docs",)
    key = "configured"
    value = {"items": _texts(_OVER_CAP)}

    _invoke(lambda: store.put(namespace, key, value))

    _assert_bounded(embeddings, store, namespace, key)


def test_small_array_wildcard_is_still_embedded() -> None:
    embeddings = RecordingEmbeddings()
    store = _store(embeddings)
    namespace = ("docs",)
    key = "small"
    expected = _texts(3)

    store.put(namespace, key, {"items": expected}, index=["items[*]"])

    assert embeddings.embedded_texts() == expected
    assert _vector_count(store, namespace, key) == len(expected)


def test_single_index_does_not_expand_the_rest_of_the_array() -> None:
    embeddings = RecordingEmbeddings()
    store = _store(embeddings)
    namespace = ("docs",)
    key = "one"
    items = _texts(_OVER_CAP)

    store.put(namespace, key, {"items": items}, index=["items[0]"])

    assert embeddings.embedded_texts() == [items[0]]
    assert _vector_count(store, namespace, key) == 1


def test_index_false_skips_embeddings() -> None:
    embeddings = RecordingEmbeddings()
    store = _store(embeddings)
    namespace = ("docs",)
    key = "plain"

    store.put(namespace, key, {"items": _texts(_OVER_CAP)}, index=False)

    assert embeddings.embedded_texts() == []
    assert _vector_count(store, namespace, key) == 0
