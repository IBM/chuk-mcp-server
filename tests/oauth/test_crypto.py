"""Tests for constant-time credential comparison."""

from chuk_mcp_server.oauth.crypto import secure_compare


class TestSecureCompare:
    """secure_compare must accept any input and never raise."""

    def test_equal_values(self):
        assert secure_compare("s3cret", "s3cret")

    def test_different_values(self):
        assert not secure_compare("s3cret", "other")

    def test_empty_strings_are_equal(self):
        assert secure_compare("", "")

    def test_empty_versus_value(self):
        assert not secure_compare("s3cret", "")
        assert not secure_compare("", "s3cret")

    def test_none_is_never_equal(self):
        assert not secure_compare(None, "s3cret")
        assert not secure_compare("s3cret", None)
        assert not secure_compare(None, None)

    def test_non_ascii_does_not_raise(self):
        # secrets.compare_digest rejects non-ASCII str; encoding first avoids it.
        assert not secure_compare("s3cret", "pásswörd")
        assert not secure_compare("pásswörd", "s3cret")

    def test_equal_non_ascii_values(self):
        assert secure_compare("pásswörd", "pásswörd")

    def test_emoji(self):
        assert secure_compare("🔐token", "🔐token")
        assert not secure_compare("🔐token", "🔓token")

    def test_differing_lengths(self):
        assert not secure_compare("short", "a-much-longer-value")
