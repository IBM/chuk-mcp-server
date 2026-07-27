"""Tests for token endpoint client credential extraction (RFC 6749 section 2.3.1)."""

import base64

import pytest

from chuk_mcp_server.oauth.client_auth import (
    ClientCredentials,
    CredentialError,
    extract_client_credentials,
    parse_basic_auth,
)


def basic_header(client_id: str, client_secret: str) -> str:
    """Build an Authorization: Basic header value."""
    encoded = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
    return f"Basic {encoded}"


class TestParseBasicAuth:
    """Test parse_basic_auth."""

    def test_none_header(self):
        assert parse_basic_auth(None) == (None, None)

    def test_empty_header(self):
        assert parse_basic_auth("") == (None, None)

    def test_valid_credentials(self):
        assert parse_basic_auth(basic_header("abc", "s3cret")) == ("abc", "s3cret")

    def test_scheme_is_case_insensitive(self):
        encoded = base64.b64encode(b"abc:s3cret").decode()
        assert parse_basic_auth(f"basic {encoded}") == ("abc", "s3cret")
        assert parse_basic_auth(f"BASIC {encoded}") == ("abc", "s3cret")

    def test_other_scheme_ignored(self):
        # A Bearer token is not client authentication.
        assert parse_basic_auth("Bearer some-access-token") == (None, None)

    def test_percent_encoded_values_are_unquoted(self):
        # RFC 6749 2.3.1 form-urlencodes both halves before base64.
        header = basic_header("client%20id", "secret%2Fwith%2Fslashes")
        assert parse_basic_auth(header) == ("client id", "secret/with/slashes")

    def test_empty_secret_is_allowed(self):
        assert parse_basic_auth(basic_header("abc", "")) == ("abc", "")

    def test_secret_containing_colon(self):
        # Only the first colon separates the two halves.
        header = basic_header("abc", "a:b:c")
        assert parse_basic_auth(header) == ("abc", "a:b:c")

    def test_missing_credentials_raises(self):
        with pytest.raises(ValueError, match="missing"):
            parse_basic_auth("Basic ")

    def test_invalid_base64_raises(self):
        with pytest.raises(ValueError, match="base64"):
            parse_basic_auth("Basic not-valid-base64!!!")

    def test_non_utf8_raises(self):
        encoded = base64.b64encode(b"\xff\xfe\xfd").decode()
        with pytest.raises(ValueError, match="base64"):
            parse_basic_auth(f"Basic {encoded}")

    def test_missing_colon_raises(self):
        encoded = base64.b64encode(b"no-colon-here").decode()
        with pytest.raises(ValueError, match="client_id:client_secret"):
            parse_basic_auth(f"Basic {encoded}")

    def test_missing_client_id_raises(self):
        encoded = base64.b64encode(b":only-a-secret").decode()
        with pytest.raises(ValueError, match="client_id"):
            parse_basic_auth(f"Basic {encoded}")


class TestExtractClientCredentials:
    """Test extract_client_credentials."""

    def test_nothing_presented(self):
        creds = extract_client_credentials(None, None, None)
        assert creds.is_valid
        assert creds.client_id is None
        assert creds.client_secret is None

    def test_body_only(self):
        creds = extract_client_credentials(None, "abc", "s3cret")
        assert creds.is_valid
        assert creds.client_id == "abc"
        assert creds.client_secret == "s3cret"

    def test_header_only(self):
        creds = extract_client_credentials(basic_header("abc", "s3cret"), None, None)
        assert creds.is_valid
        assert creds.client_id == "abc"
        assert creds.client_secret == "s3cret"

    def test_header_and_matching_body(self):
        creds = extract_client_credentials(basic_header("abc", "s3cret"), "abc", "s3cret")
        assert creds.is_valid
        assert creds.client_id == "abc"
        assert creds.client_secret == "s3cret"

    def test_public_client_sends_id_only(self):
        creds = extract_client_credentials(None, "abc", None)
        assert creds.is_valid
        assert creds.client_id == "abc"
        assert creds.client_secret is None

    def test_empty_body_secret_is_treated_as_absent(self):
        creds = extract_client_credentials(None, "abc", "")
        assert creds.is_valid
        assert creds.client_secret is None

    def test_malformed_header(self):
        creds = extract_client_credentials("Basic !!!not-base64", None, None)
        assert not creds.is_valid
        assert creds.error is CredentialError.MALFORMED_HEADER

    def test_client_id_mismatch(self):
        creds = extract_client_credentials(basic_header("abc", "s3cret"), "different", None)
        assert not creds.is_valid
        assert creds.error is CredentialError.CLIENT_ID_MISMATCH

    def test_client_secret_mismatch(self):
        creds = extract_client_credentials(basic_header("abc", "s3cret"), "abc", "other-secret")
        assert not creds.is_valid
        assert creds.error is CredentialError.CLIENT_SECRET_MISMATCH

    def test_header_secret_wins_when_body_omits_it(self):
        creds = extract_client_credentials(basic_header("abc", "s3cret"), "abc", None)
        assert creds.is_valid
        assert creds.client_secret == "s3cret"

    def test_body_secret_used_when_header_secret_empty(self):
        creds = extract_client_credentials(basic_header("abc", ""), "abc", "body-secret")
        assert creds.is_valid
        assert creds.client_secret == "body-secret"


class TestClientCredentials:
    """Test the ClientCredentials value object."""

    def test_is_valid_without_error(self):
        assert ClientCredentials(client_id="abc").is_valid

    def test_is_not_valid_with_error(self):
        assert not ClientCredentials(error=CredentialError.MALFORMED_HEADER).is_valid

    def test_is_frozen(self):
        creds = ClientCredentials(client_id="abc")
        with pytest.raises(AttributeError):
            creds.client_id = "changed"  # type: ignore[misc]
