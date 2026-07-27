"""
Redirect URI validation and construction.

The redirect URI is the one piece of client-supplied data the authorization
endpoint hands straight back to a browser, so it gets two checks:

- at registration, the scheme must be one a browser will navigate to safely —
  ``javascript:`` and ``data:`` URIs registered here would later be rendered as
  links on the callback page (RFC 9700 section 4.1)
- at redirect time, parameters must be appended to whatever query string the
  registered URI already carries, rather than blindly appended after a ``?``
"""

from urllib.parse import urlencode, urlparse

from .constants import DANGEROUS_URI_SCHEMES, QUERY_SEPARATOR, QUERY_START

__all__ = ["build_redirect_url", "is_safe_redirect_uri"]


def is_safe_redirect_uri(redirect_uri: str) -> bool:
    """
    Report whether a redirect URI is safe to register and later navigate to.

    Rejects schemes that execute in the browser rather than navigating, and
    relative URIs (RFC 6749 section 3.1.2 requires an absolute URI).

    Args:
        redirect_uri: The URI a client wants to register

    Returns:
        True if the URI is absolute and uses a navigable scheme
    """
    if not redirect_uri or not redirect_uri.strip():
        return False

    try:
        parsed = urlparse(redirect_uri)
    except ValueError:
        return False

    scheme = parsed.scheme.lower()
    if not scheme:
        # Relative URI — not an absolute redirect target.
        return False

    if scheme in DANGEROUS_URI_SCHEMES:
        return False

    # An absolute URI needs somewhere to go: either a network location
    # (https://app.example/cb) or an opaque path for native app schemes
    # (com.example.app:/callback).
    return bool(parsed.netloc or parsed.path)


def build_redirect_url(redirect_uri: str, params: dict[str, str]) -> str:
    """
    Append query parameters to a redirect URI, preserving any it already has.

    Args:
        redirect_uri: The client's registered redirect URI
        params: Parameters to append

    Returns:
        The URI to redirect to
    """
    if not params:
        return redirect_uri

    separator = QUERY_SEPARATOR if QUERY_START in redirect_uri else QUERY_START
    return f"{redirect_uri}{separator}{urlencode(params)}"
