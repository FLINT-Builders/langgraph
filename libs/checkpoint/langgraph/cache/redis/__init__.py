from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from langgraph.cache.base import BaseCache, FullKey, Namespace, ValueT
from langgraph.checkpoint.serde.base import SerializerProtocol

# `%` is the escape character, `:` separates segments, and the rest are Redis
# glob metacharacters, so encoded parts can be matched literally by `KEYS`.
_ESCAPES = {
    "%": "%25",
    ":": "%3A",
    "*": "%2A",
    "?": "%3F",
    "[": "%5B",
    "]": "%5D",
    "\\": "%5C",
}
_UNESCAPES = {v: k for k, v in _ESCAPES.items()}
_ESCAPE_TABLE = str.maketrans(_ESCAPES)
_GLOB_ESCAPE_TABLE = str.maketrans({c: f"\\{c}" for c in "*?[]\\"})


def _encode_part(part: str) -> str:
    return part.translate(_ESCAPE_TABLE)


def _decode_part(part: str) -> str:
    out: list[str] = []
    i = 0
    while i < len(part):
        token = part[i : i + 3]
        if part[i] == "%" and token in _UNESCAPES:
            out.append(_UNESCAPES[token])
            i += 3
        else:
            out.append(part[i])
            i += 1
    return "".join(out)


class RedisCache(BaseCache[ValueT]):
    """Redis-based cache implementation with TTL support."""

    def __init__(
        self,
        redis: Any,
        *,
        serde: SerializerProtocol | None = None,
        prefix: str = "langgraph:cache:",
    ) -> None:
        """Initialize the cache with a Redis client.

        Args:
            redis: Redis client instance (sync or async)
            serde: Serializer to use for values
            prefix: Key prefix for all cached values
        """
        super().__init__(serde=serde)
        self.redis = redis
        self.prefix = prefix

    def _ns_prefix(self, ns: Namespace) -> str:
        """Return the part of every Redis key in `ns` that precedes the entry key.

        Each segment is encoded and followed by `:`, so distinct namespaces never
        share a prefix of the same length.
        """
        return self.prefix + "".join(f"{_encode_part(segment)}:" for segment in ns)

    def _make_key(self, ns: Namespace, key: str) -> str:
        """Create a Redis key from namespace and key."""
        return self._ns_prefix(ns) + _encode_part(key)

    def _parse_key(self, redis_key: str) -> tuple[Namespace, str]:
        """Parse a Redis key back to namespace and key."""
        if not redis_key.startswith(self.prefix):
            raise ValueError(
                f"Key {redis_key} does not start with prefix {self.prefix}"
            )

        parts = redis_key[len(self.prefix) :].split(":")
        return (
            tuple(_decode_part(part) for part in parts[:-1]),
            _decode_part(parts[-1]),
        )

    def get(self, keys: Sequence[FullKey]) -> dict[FullKey, ValueT]:
        """Get the cached values for the given keys."""
        if not keys:
            return {}

        # Build Redis keys
        redis_keys = [self._make_key(ns, key) for ns, key in keys]

        # Get values from Redis using MGET
        try:
            raw_values = self.redis.mget(redis_keys)
        except Exception:
            # If Redis is unavailable, return empty dict
            return {}

        values: dict[FullKey, ValueT] = {}
        for i, raw_value in enumerate(raw_values):
            if raw_value is not None:
                try:
                    # Deserialize the value
                    encoding, data = raw_value.split(b":", 1)
                    values[keys[i]] = self.serde.loads_typed((encoding.decode(), data))
                except Exception:
                    # Skip corrupted entries
                    continue

        return values

    async def aget(self, keys: Sequence[FullKey]) -> dict[FullKey, ValueT]:
        """Asynchronously get the cached values for the given keys."""
        return self.get(keys)

    def set(self, mapping: Mapping[FullKey, tuple[ValueT, int | None]]) -> None:
        """Set the cached values for the given keys and TTLs."""
        if not mapping:
            return

        # Use pipeline for efficient batch operations
        pipe = self.redis.pipeline()

        for (ns, key), (value, ttl) in mapping.items():
            redis_key = self._make_key(ns, key)
            encoding, data = self.serde.dumps_typed(value)

            # Store as "encoding:data" format
            serialized_value = f"{encoding}:".encode() + data

            if ttl is not None:
                pipe.setex(redis_key, ttl, serialized_value)
            else:
                pipe.set(redis_key, serialized_value)

        try:
            pipe.execute()
        except Exception:
            # Silently fail if Redis is unavailable
            pass

    async def aset(self, mapping: Mapping[FullKey, tuple[ValueT, int | None]]) -> None:
        """Asynchronously set the cached values for the given keys and TTLs."""
        self.set(mapping)

    def clear(self, namespaces: Sequence[Namespace] | None = None) -> None:
        """Delete the cached values for the given namespaces.
        If no namespaces are provided, clear all cached values."""
        try:
            if namespaces is None:
                # Clear all keys with our prefix
                pattern = f"{self.prefix.translate(_GLOB_ESCAPE_TABLE)}*"
                keys = self.redis.keys(pattern)
                if keys:
                    self.redis.delete(*keys)
            else:
                # Clear specific namespaces
                keys_to_delete = []
                for ns in namespaces:
                    ns_prefix = self._ns_prefix(ns)
                    ns_prefix_bytes = ns_prefix.encode()
                    pattern = f"{ns_prefix.translate(_GLOB_ESCAPE_TABLE)}*"
                    for redis_key in self.redis.keys(pattern):
                        # The pattern also matches keys of nested namespaces;
                        # an entry of `ns` itself has no `:` after its prefix.
                        if isinstance(redis_key, bytes):
                            rest_has_sep = b":" in redis_key[len(ns_prefix_bytes) :]
                        else:
                            rest_has_sep = ":" in redis_key[len(ns_prefix) :]
                        if not rest_has_sep:
                            keys_to_delete.append(redis_key)

                if keys_to_delete:
                    self.redis.delete(*keys_to_delete)
        except Exception:
            # Silently fail if Redis is unavailable
            pass

    async def aclear(self, namespaces: Sequence[Namespace] | None = None) -> None:
        """Asynchronously delete the cached values for the given namespaces.
        If no namespaces are provided, clear all cached values."""
        self.clear(namespaces)
