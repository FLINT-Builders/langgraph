"""Storage location must be part of the authenticated data for encrypted checkpoints.

`EncryptedSerializer.from_pycryptodome_aes` encrypts with one shared key and
AES-EAX, and neither the cipher nor the saver passes the record identity in.
These tests copy a genuinely encrypted value into a different record and load
it through the saver or cache the product uses. They fail while that load
returns the foreign plaintext. They pass once the load rejects the copied
ciphertext and does not reveal the plaintext.

Call sites that use the same serializer and are not exercised here:

- `langgraph.cache.redis.RedisCache` (needs a Redis server; `get` also swallows
  deserializer errors)
- `langgraph.checkpoint.sqlite` and `langgraph.checkpoint.postgres` savers,
  covered in those packages' tests
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from langchain_core.runnables import RunnableConfig

from langgraph.cache.memory import InMemoryCache
from langgraph.checkpoint.base import empty_checkpoint
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.encrypted import EncryptedSerializer

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


def _put(
    saver: InMemorySaver,
    thread_id: str,
    values: dict[str, Any],
    versions: dict[str, str],
    *,
    checkpoint_ns: str = "",
    parent_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> RunnableConfig:
    checkpoint = empty_checkpoint()
    checkpoint["channel_values"] = values
    checkpoint["channel_versions"] = dict(versions)
    saved = saver.put(
        _config(thread_id, checkpoint_ns=checkpoint_ns, checkpoint_id=parent_id),
        checkpoint,
        metadata or {},
        versions,
    )
    for channel, version in versions.items():
        type_tag, _blob = saver.blobs[(thread_id, checkpoint_ns, channel, version)]
        assert type_tag.endswith("+aes")
    return saved


def _assert_rejected(load: Callable[[], Any], foreign_plaintext: str) -> None:
    """A copied ciphertext must not decrypt under the destination record."""
    try:
        loaded = load()
    except Exception as exc:
        assert foreign_plaintext not in repr(exc)
        return
    raise AssertionError(
        f"ciphertext stored under a different record was decrypted: {loaded!r}"
    )


def test_owner_blob_round_trip() -> None:
    saver = InMemorySaver(serde=_serde())
    saved = _put(
        saver,
        "owner",
        {"messages": {"text": "owner-state"}},
        {"messages": "1"},
    )

    loaded = saver.get_tuple(saved)

    assert loaded is not None
    assert loaded.checkpoint["channel_values"]["messages"] == {"text": "owner-state"}


def test_cross_thread_blob_is_rejected() -> None:
    saver = InMemorySaver(serde=_serde())
    _put(saver, "victim", {"messages": {"text": "victim-state"}}, {"messages": "1"})
    attacker = _put(
        saver, "attacker", {"messages": {"text": "attacker-state"}}, {"messages": "1"}
    )
    own = saver.get_tuple(attacker)
    assert own is not None
    assert own.checkpoint["channel_values"]["messages"]["text"] == "attacker-state"

    saver.blobs[("attacker", "", "messages", "1")] = saver.blobs[
        ("victim", "", "messages", "1")
    ]

    _assert_rejected(
        lambda: saver.get_tuple(_config("attacker")),
        "victim-state",
    )


def test_cross_namespace_blob_is_rejected() -> None:
    saver = InMemorySaver(serde=_serde())
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
    saver.blobs[("thread", "b", "messages", "1")] = saver.blobs[
        ("thread", "a", "messages", "1")
    ]

    _assert_rejected(
        lambda: saver.get_tuple(_config("thread", checkpoint_ns="b")),
        "ns-a-state",
    )


def test_cross_channel_blob_is_rejected() -> None:
    saver = InMemorySaver(serde=_serde())
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

    saver.blobs[("thread", "", "tools", "2")] = saver.blobs[
        ("thread", "", "messages", "1")
    ]

    _assert_rejected(lambda: saver.get_tuple(saved), "messages-state")


def test_cross_version_blob_is_rejected() -> None:
    saver = InMemorySaver(serde=_serde())
    parent = _put(
        saver,
        "thread",
        {"messages": {"text": "old-state"}},
        {"messages": "1"},
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

    saver.blobs[("thread", "", "messages", "2")] = saver.blobs[
        ("thread", "", "messages", "1")
    ]

    _assert_rejected(lambda: saver.get_tuple(child), "old-state")


def test_cross_thread_write_is_rejected() -> None:
    saver = InMemorySaver(serde=_serde())
    victim = _put(saver, "victim", {"messages": {"text": "blob"}}, {"messages": "1"})
    attacker = _put(
        saver, "attacker", {"messages": {"text": "blob"}}, {"messages": "1"}
    )
    saver.put_writes(victim, [("messages", {"text": "victim-write"})], "task-1")
    saver.put_writes(attacker, [("messages", {"text": "attacker-write"})], "task-1")
    victim_id = victim["configurable"]["checkpoint_id"]
    attacker_id = attacker["configurable"]["checkpoint_id"]
    copied = saver.writes[("victim", "", victim_id)][("task-1", 0)]
    current = saver.writes[("attacker", "", attacker_id)][("task-1", 0)]
    saver.writes[("attacker", "", attacker_id)][("task-1", 0)] = (
        current[0],
        current[1],
        copied[2],
        current[3],
    )

    _assert_rejected(lambda: saver.get_tuple(attacker), "victim-write")


def test_cross_task_write_is_rejected() -> None:
    saver = InMemorySaver(serde=_serde())
    saved = _put(saver, "thread", {"messages": {"text": "blob"}}, {"messages": "1"})
    checkpoint_id = saved["configurable"]["checkpoint_id"]
    saver.put_writes(saved, [("messages", {"text": "task-a-write"})], "task-a")
    saver.put_writes(saved, [("messages", {"text": "task-b-write"})], "task-b")
    copied = saver.writes[("thread", "", checkpoint_id)][("task-a", 0)]
    current = saver.writes[("thread", "", checkpoint_id)][("task-b", 0)]
    saver.writes[("thread", "", checkpoint_id)][("task-b", 0)] = (
        current[0],
        current[1],
        copied[2],
        current[3],
    )

    _assert_rejected(lambda: saver.get_tuple(saved), "task-a-write")


def test_cross_write_index_is_rejected() -> None:
    saver = InMemorySaver(serde=_serde())
    saved = _put(saver, "thread", {"messages": {"text": "blob"}}, {"messages": "1"})
    checkpoint_id = saved["configurable"]["checkpoint_id"]
    saver.put_writes(
        saved,
        [
            ("messages", {"text": "idx-0-write"}),
            ("tools", {"text": "idx-1-write"}),
        ],
        "task-1",
    )
    copied = saver.writes[("thread", "", checkpoint_id)][("task-1", 0)]
    current = saver.writes[("thread", "", checkpoint_id)][("task-1", 1)]
    saver.writes[("thread", "", checkpoint_id)][("task-1", 1)] = (
        current[0],
        current[1],
        copied[2],
        current[3],
    )

    _assert_rejected(lambda: saver.get_tuple(saved), "idx-0-write")


def test_list_rejects_cross_thread_blob() -> None:
    saver = InMemorySaver(serde=_serde())
    _put(saver, "victim", {"messages": {"text": "victim-listed"}}, {"messages": "1"})
    _put(
        saver, "attacker", {"messages": {"text": "attacker-listed"}}, {"messages": "1"}
    )
    saver.blobs[("attacker", "", "messages", "1")] = saver.blobs[
        ("victim", "", "messages", "1")
    ]

    _assert_rejected(
        lambda: list(saver.list(_config("attacker"))),
        "victim-listed",
    )


def test_metadata_ciphertext_is_bound_to_its_checkpoint() -> None:
    saver = InMemorySaver(serde=_serde())
    victim = _put(
        saver,
        "victim",
        {"messages": {"text": "blob"}},
        {"messages": "1"},
        metadata={"note": "victim-metadata"},
    )
    attacker = _put(
        saver,
        "attacker",
        {"messages": {"text": "blob"}},
        {"messages": "1"},
        metadata={"note": "attacker-metadata"},
    )
    victim_id = victim["configurable"]["checkpoint_id"]
    attacker_id = attacker["configurable"]["checkpoint_id"]
    src = saver.storage["victim"][""][victim_id]
    dst = saver.storage["attacker"][""][attacker_id]
    assert src[1][0].endswith("+aes")
    saver.storage["attacker"][""][attacker_id] = (dst[0], src[1], dst[2])

    _assert_rejected(lambda: saver.get_tuple(attacker), "victim-metadata")


def test_delta_seed_blob_is_rejected_when_copied_across_threads() -> None:
    saver = InMemorySaver(serde=_serde())
    victim_parent = _put(
        saver,
        "victim",
        {"messages": {"text": "victim-seed"}},
        {"messages": "1"},
    )
    victim_child = _put(
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
    _ = victim_child
    own = saver.get_delta_channel_history(config=attacker_child, channels=["messages"])[
        "messages"
    ]
    assert own["seed"] == {"text": "attacker-seed"}

    saver.blobs[("attacker", "", "messages", "1")] = saver.blobs[
        ("victim", "", "messages", "1")
    ]

    _assert_rejected(
        lambda: saver.get_delta_channel_history(
            config=attacker_child, channels=["messages"]
        ),
        "victim-seed",
    )


def test_delta_write_is_rejected_when_copied_across_threads() -> None:
    saver = InMemorySaver(serde=_serde())
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
        attacker_parent, [("messages", {"text": "attacker-delta-write"})], "task-1"
    )
    victim_id = victim_parent["configurable"]["checkpoint_id"]
    attacker_id = attacker_parent["configurable"]["checkpoint_id"]
    copied = saver.writes[("victim", "", victim_id)][("task-1", 0)]
    current = saver.writes[("attacker", "", attacker_id)][("task-1", 0)]
    saver.writes[("attacker", "", attacker_id)][("task-1", 0)] = (
        current[0],
        current[1],
        copied[2],
        current[3],
    )

    _assert_rejected(
        lambda: saver.get_delta_channel_history(
            config=attacker_child, channels=["messages"]
        ),
        "victim-delta-write",
    )


def test_cache_value_is_bound_to_its_key() -> None:
    cache = InMemoryCache(serde=_serde())
    victim_key = (("users",), "victim")
    attacker_key = (("users",), "attacker")
    cache.set(
        {
            victim_key: ({"text": "victim-cache"}, None),
            attacker_key: ({"text": "attacker-cache"}, None),
        }
    )
    assert cache.get([attacker_key])[attacker_key] == {"text": "attacker-cache"}

    attacker_ns, attacker_item = attacker_key
    victim_ns, victim_item = victim_key
    expiry = cache._cache[attacker_ns][attacker_item][2]
    cache._cache[attacker_ns][attacker_item] = (
        cache._cache[victim_ns][victim_item][0],
        cache._cache[victim_ns][victim_item][1],
        expiry,
    )

    _assert_rejected(lambda: cache.get([attacker_key]), "victim-cache")


def test_cache_value_is_bound_to_its_namespace() -> None:
    cache = InMemoryCache(serde=_serde())
    victim_key = (("users",), "item")
    other_ns = (("other",), "item")
    cache.set(
        {
            victim_key: ({"text": "victim-cache"}, None),
            other_ns: ({"text": "other-ns-cache"}, None),
        }
    )
    moved = cache._cache[victim_key[0]][victim_key[1]]
    cache._cache[other_ns[0]][other_ns[1]] = (
        moved[0],
        moved[1],
        cache._cache[other_ns[0]][other_ns[1]][2],
    )

    _assert_rejected(lambda: cache.get([other_ns]), "victim-cache")


async def test_aget_tuple_rejects_copied_blob() -> None:
    saver = InMemorySaver(serde=_serde())
    _put(saver, "victim", {"messages": {"text": "victim-async"}}, {"messages": "1"})
    _put(saver, "attacker", {"messages": {"text": "attacker-async"}}, {"messages": "1"})
    saver.blobs[("attacker", "", "messages", "1")] = saver.blobs[
        ("victim", "", "messages", "1")
    ]

    try:
        loaded = await saver.aget_tuple(_config("attacker"))
    except Exception as exc:
        assert "victim-async" not in repr(exc)
        return
    raise AssertionError(
        f"ciphertext stored under a different record was decrypted: {loaded!r}"
    )
