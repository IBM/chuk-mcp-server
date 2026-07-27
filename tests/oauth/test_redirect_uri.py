"""Tests for redirect URI validation and construction."""

import pytest

from chuk_mcp_server.oauth.redirect_uri import build_redirect_url, is_safe_redirect_uri


class TestIsSafeRedirectUri:
    """Only absolute URIs with a navigable scheme may be registered."""

    @pytest.mark.parametrize(
        "uri",
        [
            "https://app.example/callback",
            "http://localhost:9999/callback",
            "http://127.0.0.1:8080/cb",
            "https://app.example/cb?existing=1",
            "com.example.app:/oauth/callback",
            "myapp://callback",
        ],
    )
    def test_accepts_navigable_uris(self, uri):
        assert is_safe_redirect_uri(uri)

    @pytest.mark.parametrize(
        "uri",
        [
            "javascript:alert(1)",
            "JavaScript:alert(1)",
            "  javascript:alert(1)",
            "data:text/html,<script>alert(1)</script>",
            "vbscript:msgbox(1)",
            "file:///etc/passwd",
            "blob:https://app.example/uuid",
            "about:blank",
        ],
    )
    def test_rejects_executable_schemes(self, uri):
        assert not is_safe_redirect_uri(uri)

    @pytest.mark.parametrize("uri", ["", "   ", "/callback", "callback", "//app.example/cb"])
    def test_rejects_non_absolute_uris(self, uri):
        assert not is_safe_redirect_uri(uri)

    def test_rejects_scheme_with_no_target(self):
        assert not is_safe_redirect_uri("https:")

    def test_rejects_unparseable_uri(self):
        # An invalid IPv6 literal makes urlparse raise.
        assert not is_safe_redirect_uri("https://[oops/cb")


class TestBuildRedirectUrl:
    """Parameters must be appended to whatever query string already exists."""

    def test_appends_to_a_uri_without_a_query(self):
        url = build_redirect_url("https://app.example/cb", {"code": "abc"})

        assert url == "https://app.example/cb?code=abc"

    def test_appends_to_a_uri_with_a_query(self):
        # A second "?" would corrupt the URL and drop the code.
        url = build_redirect_url("https://app.example/cb?tenant=acme", {"code": "abc"})

        assert url == "https://app.example/cb?tenant=acme&code=abc"

    def test_encodes_parameter_values(self):
        url = build_redirect_url("https://app.example/cb", {"error_description": "Authorization failed"})

        assert "Authorization+failed" in url
        assert " " not in url

    def test_no_params_returns_the_uri_unchanged(self):
        assert build_redirect_url("https://app.example/cb", {}) == "https://app.example/cb"

    def test_preserves_multiple_parameters(self):
        url = build_redirect_url("https://app.example/cb", {"code": "abc", "state": "xyz"})

        assert "code=abc" in url
        assert "state=xyz" in url
        assert url.count("?") == 1
