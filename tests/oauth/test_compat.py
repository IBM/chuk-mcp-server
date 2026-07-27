"""Tests for the OAuth extension point compatibility helpers."""

import functools
from unittest.mock import patch

from chuk_mcp_server.oauth.compat import _accepts, supports_keyword


class ModernProvider:
    """Stand-in for a provider written against the current interface."""

    async def exchange(self, code, client_id, client_secret=None):
        return code, client_id, client_secret


class LegacyProvider:
    """Stand-in for a provider written before client authentication existed."""

    async def exchange(self, code, client_id):
        return code, client_id


class KwargsProvider:
    """A provider that forwards everything."""

    async def exchange(self, code, **kwargs):
        return code, kwargs


class TestSupportsKeyword:
    """Test supports_keyword."""

    def test_plain_function_with_keyword(self):
        def fn(a, client_secret=None):
            return a, client_secret

        assert supports_keyword(fn, "client_secret")

    def test_plain_function_without_keyword(self):
        def fn(a):
            return a

        assert not supports_keyword(fn, "client_secret")

    def test_bound_method_with_keyword(self):
        assert supports_keyword(ModernProvider().exchange, "client_secret")

    def test_bound_method_without_keyword(self):
        assert not supports_keyword(LegacyProvider().exchange, "client_secret")

    def test_var_keyword_accepts_anything(self):
        assert supports_keyword(KwargsProvider().exchange, "client_secret")
        assert supports_keyword(KwargsProvider().exchange, "anything_at_all")

    def test_required_positional_or_keyword_counts(self):
        def fn(client_secret):
            return client_secret

        assert supports_keyword(fn, "client_secret")

    def test_positional_only_does_not_count(self):
        def fn(client_secret, /):
            return client_secret

        assert not supports_keyword(fn, "client_secret")

    def test_unwraps_decorated_functions(self):
        def decorate(f):
            @functools.wraps(f)
            def wrapper(*args, **kwargs):
                return f(*args, **kwargs)

            return wrapper

        @decorate
        def fn(a, client_secret=None):
            return a, client_secret

        assert supports_keyword(fn, "client_secret")

    def test_builtin_keyword_only_params_are_found(self):
        # print(*values, sep=..., end=...) — keyword-only params still count.
        assert supports_keyword(print, "sep")
        assert not supports_keyword(print, "client_secret")

    def test_uninspectable_callable_falls_back_to_permissive(self):
        # Some C callables have no signature; assume the modern one and let the
        # call itself raise rather than silently dropping the credential.
        def fn(a, client_secret=None):
            return a, client_secret

        _accepts.cache_clear()
        with patch("chuk_mcp_server.oauth.compat.inspect.signature", side_effect=ValueError):
            assert supports_keyword(fn, "client_secret")
        _accepts.cache_clear()

    def test_different_instances_share_result(self):
        # Caching keys on the underlying function, not the bound instance.
        assert supports_keyword(ModernProvider().exchange, "client_secret")
        assert supports_keyword(ModernProvider().exchange, "client_secret")
        assert not supports_keyword(LegacyProvider().exchange, "client_secret")

    def test_unhashable_callable(self):
        class Unhashable:
            __hash__ = None  # type: ignore[assignment]

            def __call__(self, a, client_secret=None):
                return a, client_secret

        assert supports_keyword(Unhashable(), "client_secret")
