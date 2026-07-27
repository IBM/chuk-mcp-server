"""
Backwards-compatibility helpers for OAuth extension points.

``BaseOAuthProvider`` and ``BaseTokenStore`` are documented extension points, so
implementations written against an older release will not accept keyword
arguments added later (``client_secret``, ``client_id``). These helpers let the
built-in middleware and provider pass the new arguments when the implementation
understands them and fall back to the old call shape when it does not.
"""

import inspect
from collections.abc import Callable
from functools import lru_cache
from typing import Any


@lru_cache(maxsize=256)
def _accepts(func: Any, name: str) -> bool:
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):  # pragma: no cover - builtins / C callables
        # Can't introspect it; assume the modern signature and let the call raise.
        return True

    for parameter in signature.parameters.values():
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            return True
        if parameter.name == name and parameter.kind is not inspect.Parameter.POSITIONAL_ONLY:
            return True

    return False


def supports_keyword(func: Callable[..., Any], name: str) -> bool:
    """
    Report whether ``func`` can be called with the keyword argument ``name``.

    For bound methods the underlying function is inspected, so the cache never
    keeps a provider or token store instance alive.
    """
    target = inspect.unwrap(getattr(func, "__func__", func))

    try:
        return bool(_accepts(target, name))
    except TypeError:  # unhashable callable — check without caching
        return bool(_accepts.__wrapped__(target, name))


__all__ = ["supports_keyword"]
