"""
Client credential extraction for the OAuth token endpoint.

RFC 6749 section 2.3.1 defines two ways a client may authenticate at the token
endpoint: an ``Authorization: Basic`` header (``client_secret_basic``) or
credentials in the request body (``client_secret_post``). This module turns
either form into a single ``ClientCredentials`` value, and reports the ways the
two can disagree so the caller can reject the request instead of guessing.
"""

import base64
import binascii
from dataclasses import dataclass
from enum import Enum
from urllib.parse import unquote_plus

from .constants import (
    AUTH_SCHEME_BASIC,
    BASIC_CREDENTIALS_SEPARATOR,
)


class CredentialError(Enum):
    """Why a set of presented client credentials could not be accepted."""

    MALFORMED_HEADER = "malformed_header"
    CLIENT_ID_MISMATCH = "client_id_mismatch"
    CLIENT_SECRET_MISMATCH = "client_secret_mismatch"


@dataclass(frozen=True)
class ClientCredentials:
    """Client credentials presented at the token endpoint."""

    client_id: str | None = None
    client_secret: str | None = None
    error: CredentialError | None = None

    @property
    def is_valid(self) -> bool:
        """True when the presented credentials are internally consistent."""
        return self.error is None


def parse_basic_auth(header_value: str | None) -> tuple[str | None, str | None]:
    """
    Parse an ``Authorization: Basic`` header into (client_id, client_secret).

    Per RFC 6749 section 2.3.1 both halves are form-urlencoded before being
    base64'd, so they are unquoted here.

    Args:
        header_value: Raw Authorization header value, or None

    Returns:
        (client_id, client_secret), both None when the header is absent or uses
        a different scheme

    Raises:
        ValueError: If the header claims Basic but cannot be decoded
    """
    if not header_value:
        return None, None

    scheme, _, encoded = header_value.partition(" ")
    if scheme.lower() != AUTH_SCHEME_BASIC:
        # Some other scheme (e.g. Bearer) — not client authentication.
        return None, None

    encoded = encoded.strip()
    if not encoded:
        raise ValueError("Basic credentials missing")

    try:
        decoded = base64.b64decode(encoded, validate=True).decode("utf-8")
    except (binascii.Error, ValueError, UnicodeDecodeError) as exc:
        raise ValueError("Basic credentials are not valid base64") from exc

    if BASIC_CREDENTIALS_SEPARATOR not in decoded:
        raise ValueError("Basic credentials must be client_id:client_secret")

    client_id, _, client_secret = decoded.partition(BASIC_CREDENTIALS_SEPARATOR)
    if not client_id:
        raise ValueError("Basic credentials missing client_id")

    # RFC 6749 2.3.1 applies application/x-www-form-urlencoded, which encodes
    # space as "+" — so unquote_plus, not unquote.
    return unquote_plus(client_id), unquote_plus(client_secret)


def extract_client_credentials(
    header_value: str | None,
    body_client_id: str | None,
    body_client_secret: str | None,
) -> ClientCredentials:
    """
    Resolve the client credentials for a token request.

    Header credentials take precedence, but a body value that contradicts the
    header is an error rather than something to silently drop.

    Args:
        header_value: Raw Authorization header value, or None
        body_client_id: client_id from the request body, or None
        body_client_secret: client_secret from the request body, or None

    Returns:
        The resolved credentials, or one carrying an ``error``
    """
    try:
        header_client_id, header_client_secret = parse_basic_auth(header_value)
    except ValueError:
        return ClientCredentials(error=CredentialError.MALFORMED_HEADER)

    if header_client_id is not None and body_client_id and body_client_id != header_client_id:
        return ClientCredentials(error=CredentialError.CLIENT_ID_MISMATCH)

    if header_client_secret and body_client_secret and body_client_secret != header_client_secret:
        return ClientCredentials(error=CredentialError.CLIENT_SECRET_MISMATCH)

    return ClientCredentials(
        client_id=header_client_id or body_client_id,
        # An empty secret carries no proof of anything; treat it as absent.
        client_secret=header_client_secret or body_client_secret or None,
    )


__all__ = [
    "ClientCredentials",
    "CredentialError",
    "extract_client_credentials",
    "parse_basic_auth",
]
