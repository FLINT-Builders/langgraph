import hashlib
import hmac
import os
from typing import Any

from langgraph.checkpoint.serde.base import (
    CipherProtocol,
    RecordContext,
    SerializerProtocol,
)
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

# Bound plaintexts start with this marker followed by the SHA-256 digest of the
# record context. The marker is encrypted and authenticated with the value, so
# it cannot be added or stripped without the key. `0xc1` is never the first byte
# of msgpack, JSON or pickle output, so an unbound plaintext is not mistaken for
# a bound one.
_CONTEXT_MAGIC = b"\xc1lgctx1"
_CONTEXT_HEADER_LEN = len(_CONTEXT_MAGIC) + hashlib.sha256().digest_size


def _context_header(context: RecordContext) -> bytes:
    digest = hashlib.sha256()
    for part in context:
        encoded = ("" if part is None else str(part)).encode("utf-8", "surrogatepass")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return _CONTEXT_MAGIC + digest.digest()


class EncryptedSerializer(SerializerProtocol):
    """Serializer that encrypts and decrypts data using an encryption protocol.

    Savers and caches call `dumps_typed_bound` / `loads_typed_bound` with the
    key of the record a value is stored in. The record key is authenticated
    together with the value, so ciphertext copied into another record (another
    thread, namespace, channel, version, checkpoint, task or cache key) fails to
    load.

    Args:
        cipher: The cipher used to encrypt and decrypt values. It must be an
            authenticated cipher (such as AES-EAX or AES-GCM), otherwise the
            record binding can be forged.
        serde: The serializer used for the plaintext.
        allow_unbound_ciphertext: Also accept, in `loads_typed_bound`,
            ciphertext written without a record binding (by versions of this
            serializer that predate it). Such ciphertext can be moved between
            records undetected, so only enable this while re-encrypting existing
            data.
    """

    def __init__(
        self,
        cipher: CipherProtocol,
        serde: SerializerProtocol = JsonPlusSerializer(),
        *,
        allow_unbound_ciphertext: bool = False,
    ) -> None:
        self.cipher = cipher
        self.serde = serde
        self.allow_unbound_ciphertext = allow_unbound_ciphertext

    def dumps_typed(self, obj: Any) -> tuple[str, bytes]:
        """Serialize an object to a tuple `(type, bytes)` and encrypt the bytes."""
        # serialize data
        typ, data = self.serde.dumps_typed(obj)
        # encrypt data
        ciphername, ciphertext = self.cipher.encrypt(data)
        # add cipher name to type
        return f"{typ}+{ciphername}", ciphertext

    def loads_typed(self, data: tuple[str, bytes]) -> Any:
        enc_cipher, ciphertext = data
        # unencrypted data
        if "+" not in enc_cipher:
            return self.serde.loads_typed(data)
        typ, plaintext = self._decrypt(enc_cipher, ciphertext)
        if plaintext.startswith(_CONTEXT_MAGIC):
            raise ValueError(
                "Encrypted value is bound to a storage record and can only be "
                "loaded with `loads_typed_bound`."
            )
        # deserialize data
        return self.serde.loads_typed((typ, plaintext))

    def dumps_typed_bound(self, obj: Any, context: RecordContext) -> tuple[str, bytes]:
        """Serialize and encrypt an object, binding it to the record `context`."""
        typ, data = self.serde.dumps_typed(obj)
        ciphername, ciphertext = self.cipher.encrypt(_context_header(context) + data)
        return f"{typ}+{ciphername}", ciphertext

    def loads_typed_bound(self, data: tuple[str, bytes], context: RecordContext) -> Any:
        """Decrypt and deserialize a value read from the record `context`.

        Raises `ValueError` if the value was encrypted for a different record.
        """
        enc_cipher, ciphertext = data
        # unencrypted data
        if "+" not in enc_cipher:
            return self.serde.loads_typed(data)
        typ, plaintext = self._decrypt(enc_cipher, ciphertext)
        if plaintext.startswith(_CONTEXT_MAGIC):
            if not hmac.compare_digest(
                plaintext[:_CONTEXT_HEADER_LEN], _context_header(context)
            ):
                raise ValueError(
                    "Encrypted value was not written to this storage record."
                )
            return self.serde.loads_typed((typ, plaintext[_CONTEXT_HEADER_LEN:]))
        if not self.allow_unbound_ciphertext:
            raise ValueError(
                "Encrypted value is not bound to a storage record. Pass "
                "`allow_unbound_ciphertext=True` to read values written by "
                "earlier versions while re-encrypting them."
            )
        return self.serde.loads_typed((typ, plaintext))

    def _decrypt(self, enc_cipher: str, ciphertext: bytes) -> tuple[str, bytes]:
        # extract cipher name
        typ, ciphername = enc_cipher.split("+", 1)
        # decrypt data
        return typ, self.cipher.decrypt(ciphername, ciphertext)

    @classmethod
    def from_pycryptodome_aes(
        cls,
        serde: SerializerProtocol = JsonPlusSerializer(),
        *,
        allow_unbound_ciphertext: bool = False,
        **kwargs: Any,
    ) -> "EncryptedSerializer":
        """Create an `EncryptedSerializer` using AES encryption."""
        try:
            from Crypto.Cipher import AES
        except ImportError:
            raise ImportError(
                "Pycryptodome is not installed. Please install it with `pip install pycryptodome`."
            ) from None

        # check if AES key is provided
        if "key" in kwargs:
            key: bytes = kwargs.pop("key")
        else:
            key_str = os.getenv("LANGGRAPH_AES_KEY")
            if key_str is None:
                raise ValueError("LANGGRAPH_AES_KEY environment variable is not set.")
            key = key_str.encode()
            if len(key) not in (16, 24, 32):
                raise ValueError("LANGGRAPH_AES_KEY must be 16, 24, or 32 bytes long.")

        # set default mode to EAX if not provided
        if kwargs.get("mode") is None:
            kwargs["mode"] = AES.MODE_EAX

        class PycryptodomeAesCipher(CipherProtocol):
            def encrypt(self, plaintext: bytes) -> tuple[str, bytes]:
                cipher = AES.new(key, **kwargs)
                ciphertext, tag = cipher.encrypt_and_digest(plaintext)
                return "aes", cipher.nonce + tag + ciphertext

            def decrypt(self, ciphername: str, ciphertext: bytes) -> bytes:
                assert ciphername == "aes", f"Unsupported cipher: {ciphername}"
                nonce = ciphertext[:16]
                tag = ciphertext[16:32]
                actual_ciphertext = ciphertext[32:]

                cipher = AES.new(key, **kwargs, nonce=nonce)
                return cipher.decrypt_and_verify(actual_ciphertext, tag)

        return cls(
            PycryptodomeAesCipher(),
            serde,
            allow_unbound_ciphertext=allow_unbound_ciphertext,
        )
