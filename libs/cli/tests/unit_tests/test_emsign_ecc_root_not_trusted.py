"""CLI HTTPS clients must not trust the distrusted emSign ECC Root CA - G3.

certifi before 2026.10.18 ships that root. These clients build httpx
transports with default TLS verification, which loads that bundle. These
tests fail until the context those clients actually use no longer contains
the root.
"""

from __future__ import annotations

import hashlib
import ssl

import httpx
import pytest

from langgraph_cli.host_backend import HostBackendClient

# SHA-256 of the emSign ECC Root CA - G3 certificate in the certifi bundles
# pinned by this repository (2026.1.4 through 2026.5.20).
_EMSIGN_ECC_ROOT_G3_SHA256 = (
    "86a1ecba089c4a8d3bbe2734c612ba341d813e043cf9e8a862cd5c57a36bbe6b"
)

# Obvious placeholder. HostBackendClient requires a non-empty API key argument.
_PLACEHOLDER_API_KEY = "placeholder-api-key"


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


def test_host_backend_client_does_not_trust_emsign_ecc_root_g3() -> None:
    client = HostBackendClient("https://example.com", _PLACEHOLDER_API_KEY)
    context = client._client._transport._pool._ssl_context
    _assert_verified_context_excludes_emsign(context)


def test_pypi_version_fetch_does_not_trust_emsign_ecc_root_g3() -> None:
    # langgraph_cli.config._get_pypi_versions calls httpx.get(url, timeout=10)
    # with no verify argument, which builds this default client.
    with httpx.Client(timeout=10) as client:
        context = client._transport._pool._ssl_context
    _assert_verified_context_excludes_emsign(context)
