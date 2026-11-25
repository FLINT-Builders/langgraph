from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

RecordContext = Sequence[str | int | float | None]
"""Identity of the storage record a serialized value is written to.

Savers and caches pass the record's key (thread ID, namespace, channel,
version, task ID, index, ...) so a serializer that supports it can bind the
serialized bytes to that record and refuse them anywhere else.
"""


class UntypedSerializerProtocol(Protocol):
    """Protocol for serialization and deserialization of objects."""

    def dumps(self, obj: Any) -> bytes: ...

    def loads(self, data: bytes) -> Any: ...


@runtime_checkable
class SerializerProtocol(Protocol):
    """Protocol for serialization and deserialization of objects.

    - `dumps_typed`: Serialize an object to a tuple `(type, bytes)`.
    - `loads_typed`: Deserialize an object from a tuple `(type, bytes)`.

    Valid implementations include the `pickle`, `json` and `orjson` modules.
    """

    def dumps_typed(self, obj: Any) -> tuple[str, bytes]: ...

    def loads_typed(self, data: tuple[str, bytes]) -> Any: ...


class SerializerCompat(SerializerProtocol):
    def __init__(self, serde: UntypedSerializerProtocol) -> None:
        self.serde = serde

    def dumps_typed(self, obj: Any) -> tuple[str, bytes]:
        return type(obj).__name__, self.serde.dumps(obj)

    def loads_typed(self, data: tuple[str, bytes]) -> Any:
        return self.serde.loads(data[1])


def maybe_add_typed_methods(
    serde: SerializerProtocol | UntypedSerializerProtocol,
) -> SerializerProtocol:
    """Wrap serde old serde implementations in a class with loads_typed and dumps_typed for backwards compatibility."""

    if not isinstance(serde, SerializerProtocol):
        return SerializerCompat(serde)

    return serde


def dumps_typed_bound(
    serde: SerializerProtocol, obj: Any, context: RecordContext
) -> tuple[str, bytes]:
    """Serialize `obj` for the storage record identified by `context`.

    Serializers that implement `dumps_typed_bound` (such as
    `EncryptedSerializer`) bind the output to `context`; others fall back to
    `dumps_typed`.
    """
    bound = getattr(serde, "dumps_typed_bound", None)
    if bound is None:
        return serde.dumps_typed(obj)
    return bound(obj, context)


def loads_typed_bound(
    serde: SerializerProtocol, data: tuple[str, bytes], context: RecordContext
) -> Any:
    """Deserialize `data` read from the storage record identified by `context`.

    Serializers that implement `loads_typed_bound` (such as
    `EncryptedSerializer`) reject data that was bound to a different record;
    others fall back to `loads_typed`.
    """
    bound = getattr(serde, "loads_typed_bound", None)
    if bound is None:
        return serde.loads_typed(data)
    return bound(data, context)


def checkpoint_context(
    thread_id: str, checkpoint_ns: str, checkpoint_id: str
) -> RecordContext:
    return ("checkpoint", thread_id, checkpoint_ns, checkpoint_id)


def metadata_context(
    thread_id: str, checkpoint_ns: str, checkpoint_id: str
) -> RecordContext:
    return ("metadata", thread_id, checkpoint_ns, checkpoint_id)


def blob_context(
    thread_id: str,
    checkpoint_ns: str,
    channel: str,
    version: str | int | float | None,
) -> RecordContext:
    return ("blob", thread_id, checkpoint_ns, channel, version)


def write_context(
    thread_id: str,
    checkpoint_ns: str,
    checkpoint_id: str,
    task_id: str,
    idx: int | str,
    channel: str,
) -> RecordContext:
    return ("write", thread_id, checkpoint_ns, checkpoint_id, task_id, idx, channel)


def cache_context(*key: str) -> RecordContext:
    return ("cache", *key)


class CipherProtocol(Protocol):
    """Protocol for encryption and decryption of data.

    - `encrypt`: Encrypt plaintext.
    - `decrypt`: Decrypt ciphertext.
    """

    def encrypt(self, plaintext: bytes) -> tuple[str, bytes]:
        """Encrypt plaintext. Returns a tuple `(cipher name, ciphertext)`."""
        ...

    def decrypt(self, ciphername: str, ciphertext: bytes) -> bytes:
        """Decrypt ciphertext. Returns the plaintext."""
        ...
