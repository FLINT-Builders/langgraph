"""The SDK HTTPS clients must not trust the distrusted emSign ECC Root CA - G3.

certifi before 2026.10.18 ships that root. httpx loads it as a trust anchor
when these clients build their default TLS context. These tests fail until
the context those clients actually use no longer contains the root.
"""

from __future__ import annotations

import hashlib
import ssl

import pytest

from langgraph_sdk import get_client, get_sync_client

# SHA-256 of the emSign ECC Root CA - G3 certificate in the certifi bundles
# pinned by this repository (2026.1.4 through 2026.5.20).
_EMSIGN_ECC_ROOT_G3_SHA256 = (
    "86a1ecba089c4a8d3bbe2734c612ba341d813e043cf9e8a862cd5c57a36bbe6b"
)


def _assert_verified_context_excludes_emsign(ctx: ssl.SSLContext) -> None:
    assert ctx.verify_mode == ssl.CERT_REQUIRED
    assert ctx.check_hostname is True
    fingerprints = {
        hashlib.sha256(der).hexdigest() for der in ctx.get_ca_certs(binary_form=True)
    }
    assert _EMSIGN_ECC_ROOT_G3_SHA256 not in fingerprints


@pytest.fixture(autouse=True)
def _clear_ssl_env(monkeypatch: pytest.MonkeyPatch) -> None:
    # httpx uses SSL_CERT_FILE / SSL_CERT_DIR instead of certifi when set.
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    monkeypatch.delenv("SSL_CERT_DIR", raising=False)


def test_sync_client_does_not_trust_emsign_ecc_root_g3() -> None:
    client = get_sync_client(url="https://example.com", api_key=None)
    context = client.http.client._transport._pool._ssl_context
    _assert_verified_context_excludes_emsign(context)


def test_async_client_does_not_trust_emsign_ecc_root_g3() -> None:
    client = get_client(url="https://example.com", api_key=None)
    context = client.http.client._transport._pool._ssl_context
    _assert_verified_context_excludes_emsign(context)
