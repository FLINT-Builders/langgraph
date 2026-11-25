from __future__ import annotations

import os
import pickle
import threading
import weakref
from collections.abc import Hashable, Iterable, Mapping, Sequence
from dataclasses import fields, is_dataclass
from hashlib import blake2b
from typing import Any


def _freeze(obj: Any, depth: int = 10) -> Hashable:
    if isinstance(obj, Hashable) or depth <= 0:
        # already hashable, no need to freeze
        return obj
    elif isinstance(obj, Mapping):
        # sort keys so {"a":1,"b":2} == {"b":2,"a":1}
        return tuple(sorted((k, _freeze(v, depth - 1)) for k, v in obj.items()))
    elif isinstance(obj, Sequence):
        return tuple(_freeze(x, depth - 1) for x in obj)
    # numpy / pandas etc. can provide their own .tobytes()
    elif hasattr(obj, "tobytes"):
        return (
            type(obj).__name__,
            obj.tobytes(),
            obj.shape if hasattr(obj, "shape") else None,
        )
    return obj  # strings, ints, dataclasses with frozen=True, etc.


def default_cache_key(*args: Any, **kwargs: Any) -> str | bytes:
    """Default cache key function that uses the arguments and keyword arguments to generate a hashable key."""
    # protocol 5 strikes a good balance between speed and size
    return pickle.dumps((_freeze(args), _freeze(kwargs)), protocol=5, fix_imports=False)


_OBJECT_TOKENS: dict[int, tuple[weakref.ref, bytes]] = {}
_OBJECT_TOKENS_LOCK = threading.Lock()


def _forget_token(key: int, ref: weakref.ref) -> None:
    with _OBJECT_TOKENS_LOCK:
        entry = _OBJECT_TOKENS.get(key)
        if entry is not None and entry[0] is ref:
            del _OBJECT_TOKENS[key]


def object_token(obj: Any) -> bytes | None:
    """Return a random token that is stable for the lifetime of `obj`.

    Tokens are never derived from `id(obj)` alone, so two objects can never
    share a token, even across processes. Returns `None` for objects that
    cannot be weakly referenced.
    """
    key = id(obj)
    with _OBJECT_TOKENS_LOCK:
        entry = _OBJECT_TOKENS.get(key)
        if entry is not None and entry[0]() is obj:
            return entry[1]
        try:
            ref = weakref.ref(obj, lambda r, key=key: _forget_token(key, r))
        except TypeError:
            return None
        token = os.urandom(16)
        _OBJECT_TOKENS[key] = (ref, token)
        return token


def _length_prefixed(data: bytes) -> bytes:
    return len(data).to_bytes(8, "big") + data


def _encode_parts(tag: bytes, parts: Iterable[bytes | None]) -> bytes | None:
    out = [tag]
    for part in parts:
        if part is None:
            return None
        out.append(_length_prefixed(part))
    return b"".join(out)


def _encode_scope_value(obj: Any, depth: int = 4) -> bytes | None:
    if obj is None:
        return b"n"
    try:
        return b"v" + pickle.dumps(_freeze(obj), protocol=5, fix_imports=False)
    except Exception:
        pass
    if depth > 0:
        if is_dataclass(obj) and not isinstance(obj, type):
            cls = type(obj)
            return _encode_parts(
                b"d",
                [
                    f"{cls.__module__}.{cls.__qualname__}".encode(),
                    *(
                        _encode_scope_value(getattr(obj, f.name, None), depth - 1)
                        for f in fields(obj)
                    ),
                ],
            )
        if isinstance(obj, Mapping):
            items: list[bytes] = []
            for k, v in obj.items():
                item = _encode_parts(
                    b"i",
                    [
                        _encode_scope_value(k, depth - 1),
                        _encode_scope_value(v, depth - 1),
                    ],
                )
                if item is None:
                    return None
                items.append(item)
            return _encode_parts(b"m", sorted(items))
        if isinstance(obj, (list, tuple)):
            return _encode_parts(b"l", [_encode_scope_value(x, depth - 1) for x in obj])
    token = object_token(obj)
    return None if token is None else b"o" + token


def auth_user_scope(user: Any) -> Any:
    """Reduce an authenticated user object to the fields that identify it."""
    if user is None:
        return None
    try:
        identity = user.identity
    except Exception:
        return user
    if not isinstance(identity, str):
        return user
    try:
        permissions = tuple(sorted(str(p) for p in (user.permissions or ())))
    except Exception:
        permissions = None
    try:
        is_authenticated = bool(user.is_authenticated)
    except Exception:
        is_authenticated = None
    return ("auth_user", identity, is_authenticated, permissions)


def cache_scope_digest(parts: Sequence[tuple[str, Any]]) -> bytes | None:
    """Digest the values that scope a cached task to the caller.

    Returns `None` when a value cannot be encoded such that two different
    callers are guaranteed to produce different digests; such tasks must not
    be cached.
    """
    digest = blake2b(digest_size=16)
    for name, value in parts:
        encoded = _encode_scope_value(value)
        if encoded is None:
            return None
        digest.update(_length_prefixed(name.encode()))
        digest.update(_length_prefixed(encoded))
    return digest.digest()


def cache_key_digest(scope: bytes, args_key: str | bytes) -> str:
    digest = blake2b(_length_prefixed(scope), digest_size=16)
    digest.update(args_key.encode() if isinstance(args_key, str) else args_key)
    return digest.hexdigest()
