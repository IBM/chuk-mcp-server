"""
Constant-time comparison for OAuth credential material.

``secrets.compare_digest`` raises ``TypeError`` when handed a ``str`` containing
non-ASCII characters, and every credential this server compares (client secrets,
PKCE verifiers and challenges) arrives from the network. Comparing the UTF-8
encodings instead keeps the comparison constant-time while accepting any input.
"""

import secrets

__all__ = ["secure_compare"]


def secure_compare(expected: str | None, presented: str | None) -> bool:
    """
    Compare two credential strings in constant time.

    Args:
        expected: The stored value
        presented: The value supplied by the caller

    Returns:
        True if both are present and equal
    """
    if expected is None or presented is None:
        return False

    return secrets.compare_digest(expected.encode("utf-8"), presented.encode("utf-8"))
