"""
Regression tests for OAuth client authentication at the token endpoint.

These cover the guarantees the server's RFC 8414 metadata advertises:

- a client that registers as confidential must present its secret to exchange an
  authorization code or refresh a token (RFC 6749 sections 3.2.1 and 6)
- a wrong secret is rejected, whether or not one was required
- a refresh token is bound to the client it was issued to
- a PKCE challenge is verified even when the client omits code_challenge_method
"""

import base64
import hashlib
import json
from unittest.mock import AsyncMock, Mock

import pytest
from starlette.requests import Request

from chuk_mcp_server.oauth.base_provider import BaseOAuthProvider
from chuk_mcp_server.oauth.constants import (
    AUTH_METHOD_CLIENT_SECRET_BASIC,
    AUTH_METHOD_CLIENT_SECRET_POST,
    AUTH_METHOD_NONE,
    CODE_CHALLENGE_PLAIN,
    CODE_CHALLENGE_S256,
    ERROR_INVALID_CLIENT,
    ERROR_INVALID_REQUEST,
    GRANT_AUTHORIZATION_CODE,
    GRANT_REFRESH_TOKEN,
    HTTP_BAD_REQUEST,
    HTTP_INTERNAL_SERVER_ERROR,
    HTTP_UNAUTHORIZED,
)
from chuk_mcp_server.oauth.middleware import OAuthMiddleware
from chuk_mcp_server.oauth.models import OAuthToken, TokenError
from chuk_mcp_server.oauth.token_store import TokenStore

from .test_token_store import MockSession

REDIRECT_URI = "http://localhost:9999/callback"
USER_ID = "victim-user-1"


@pytest.fixture
def store(monkeypatch):
    """A TokenStore backed by an in-memory session."""
    session = MockSession()
    monkeypatch.setattr(
        "chuk_mcp_server.oauth.token_store.get_session",
        lambda: session,
    )
    return TokenStore(sandbox_id="test-client-auth")


def s256_challenge(verifier: str) -> str:
    """Compute the S256 challenge for a verifier."""
    digest = hashlib.sha256(verifier.encode()).digest()
    return base64.urlsafe_b64encode(digest).decode().rstrip("=")


async def register(store: TokenStore, auth_method: str) -> tuple[str, str]:
    """Register a client and return (client_id, client_secret)."""
    credentials = await store.register_client(
        client_name="Test App",
        redirect_uris=[REDIRECT_URI],
        token_endpoint_auth_method=auth_method,
    )
    return credentials["client_id"], credentials["client_secret"]


class TestRegistrationRecordsAuthMethod:
    """A client's declared auth method must survive registration."""

    @pytest.mark.asyncio
    async def test_declared_method_is_persisted(self, store):
        client_id, _ = await register(store, AUTH_METHOD_CLIENT_SECRET_POST)

        client = await store.get_client(client_id)
        assert client is not None
        assert client.token_endpoint_auth_method == AUTH_METHOD_CLIENT_SECRET_POST

    @pytest.mark.asyncio
    async def test_basic_method_is_persisted(self, store):
        client_id, _ = await register(store, AUTH_METHOD_CLIENT_SECRET_BASIC)

        client = await store.get_client(client_id)
        assert client.token_endpoint_auth_method == AUTH_METHOD_CLIENT_SECRET_BASIC

    @pytest.mark.asyncio
    async def test_undeclared_client_is_public(self, store):
        credentials = await store.register_client(
            client_name="Test App",
            redirect_uris=[REDIRECT_URI],
        )

        client = await store.get_client(credentials["client_id"])
        assert client.token_endpoint_auth_method == AUTH_METHOD_NONE

    @pytest.mark.asyncio
    async def test_registration_returns_the_method(self, store):
        credentials = await store.register_client(
            client_name="Test App",
            redirect_uris=[REDIRECT_URI],
            token_endpoint_auth_method=AUTH_METHOD_CLIENT_SECRET_POST,
        )

        assert credentials["token_endpoint_auth_method"] == AUTH_METHOD_CLIENT_SECRET_POST

    @pytest.mark.asyncio
    async def test_unknown_client_is_not_found(self, store):
        assert await store.get_client("never-registered") is None


class TestConfidentialClientAuthentication:
    """Confidential clients must present a matching secret."""

    @pytest.mark.asyncio
    async def test_correct_secret_authenticates(self, store):
        client_id, client_secret = await register(store, AUTH_METHOD_CLIENT_SECRET_POST)

        assert await store.authenticate_client(client_id, client_secret)

    @pytest.mark.asyncio
    async def test_missing_secret_is_rejected(self, store):
        client_id, _ = await register(store, AUTH_METHOD_CLIENT_SECRET_POST)

        assert not await store.authenticate_client(client_id)
        assert not await store.authenticate_client(client_id, None)

    @pytest.mark.asyncio
    async def test_wrong_secret_is_rejected(self, store):
        client_id, _ = await register(store, AUTH_METHOD_CLIENT_SECRET_POST)

        assert not await store.authenticate_client(client_id, "not-the-secret")

    @pytest.mark.asyncio
    async def test_empty_secret_is_rejected(self, store):
        client_id, _ = await register(store, AUTH_METHOD_CLIENT_SECRET_BASIC)

        assert not await store.authenticate_client(client_id, "")

    @pytest.mark.asyncio
    async def test_basic_client_also_requires_a_secret(self, store):
        client_id, client_secret = await register(store, AUTH_METHOD_CLIENT_SECRET_BASIC)

        assert not await store.authenticate_client(client_id)
        assert await store.authenticate_client(client_id, client_secret)

    @pytest.mark.asyncio
    async def test_unregistered_client_is_rejected(self, store):
        assert not await store.authenticate_client("never-registered", "any-secret")

    @pytest.mark.asyncio
    async def test_redirect_uri_is_still_checked(self, store):
        client_id, client_secret = await register(store, AUTH_METHOD_CLIENT_SECRET_POST)

        assert await store.authenticate_client(client_id, client_secret, REDIRECT_URI)
        assert not await store.authenticate_client(client_id, client_secret, "http://evil.example/cb")


class TestPublicClientAuthentication:
    """Public clients may omit a secret, but a wrong one is never accepted."""

    @pytest.mark.asyncio
    async def test_no_secret_is_accepted(self, store):
        client_id, _ = await register(store, AUTH_METHOD_NONE)

        assert await store.authenticate_client(client_id)

    @pytest.mark.asyncio
    async def test_wrong_secret_is_still_rejected(self, store):
        client_id, _ = await register(store, AUTH_METHOD_NONE)

        assert not await store.authenticate_client(client_id, "not-the-secret")

    @pytest.mark.asyncio
    async def test_correct_secret_is_accepted(self, store):
        client_id, client_secret = await register(store, AUTH_METHOD_NONE)

        assert await store.authenticate_client(client_id, client_secret)


class TestRefreshTokenClientBinding:
    """A refresh token belongs to one client (RFC 6749 section 6)."""

    @pytest.mark.asyncio
    async def test_issuing_client_can_refresh(self, store):
        _, refresh_token = await store.create_access_token(USER_ID, "client-a")

        assert await store.refresh_access_token(refresh_token, client_id="client-a") is not None

    @pytest.mark.asyncio
    async def test_other_client_cannot_refresh(self, store):
        _, refresh_token = await store.create_access_token(USER_ID, "client-a")

        assert await store.refresh_access_token(refresh_token, client_id="client-b") is None

    @pytest.mark.asyncio
    async def test_rejected_refresh_does_not_consume_the_token(self, store):
        _, refresh_token = await store.create_access_token(USER_ID, "client-a")

        assert await store.refresh_access_token(refresh_token, client_id="client-b") is None
        # The legitimate client can still use it.
        assert await store.refresh_access_token(refresh_token, client_id="client-a") is not None

    @pytest.mark.asyncio
    async def test_omitted_client_id_keeps_legacy_behaviour(self, store):
        _, refresh_token = await store.create_access_token(USER_ID, "client-a")

        assert await store.refresh_access_token(refresh_token) is not None

    @pytest.mark.asyncio
    async def test_unknown_refresh_token(self, store):
        assert await store.refresh_access_token("never-issued", client_id="client-a") is None


class TestPkceEnforcement:
    """A recorded challenge must always be verified."""

    async def _code(self, store, **kwargs):
        return await store.create_authorization_code(
            user_id=USER_ID,
            client_id="client-a",
            redirect_uri=REDIRECT_URI,
            **kwargs,
        )

    async def _validate(self, store, code, code_verifier=None):
        return await store.validate_authorization_code(
            code=code,
            client_id="client-a",
            redirect_uri=REDIRECT_URI,
            code_verifier=code_verifier,
        )

    @pytest.mark.asyncio
    async def test_s256_requires_matching_verifier(self, store):
        verifier = "a" * 64
        code = await self._code(
            store,
            code_challenge=s256_challenge(verifier),
            code_challenge_method=CODE_CHALLENGE_S256,
        )

        assert await self._validate(store, code, verifier) is not None

    @pytest.mark.asyncio
    async def test_s256_rejects_wrong_verifier(self, store):
        code = await self._code(
            store,
            code_challenge=s256_challenge("a" * 64),
            code_challenge_method=CODE_CHALLENGE_S256,
        )

        assert await self._validate(store, code, "b" * 64) is None

    @pytest.mark.asyncio
    async def test_s256_rejects_missing_verifier(self, store):
        code = await self._code(
            store,
            code_challenge=s256_challenge("a" * 64),
            code_challenge_method=CODE_CHALLENGE_S256,
        )

        assert await self._validate(store, code) is None

    @pytest.mark.asyncio
    async def test_plain_requires_matching_verifier(self, store):
        code = await self._code(
            store,
            code_challenge="the-verifier",
            code_challenge_method=CODE_CHALLENGE_PLAIN,
        )

        assert await self._validate(store, code, "the-verifier") is not None

    @pytest.mark.asyncio
    async def test_plain_rejects_wrong_verifier(self, store):
        code = await self._code(
            store,
            code_challenge="the-verifier",
            code_challenge_method=CODE_CHALLENGE_PLAIN,
        )

        assert await self._validate(store, code, "wrong") is None

    @pytest.mark.asyncio
    async def test_omitted_method_defaults_to_plain(self, store):
        # RFC 7636 4.3. Previously an absent method meant no verification at all.
        code = await self._code(store, code_challenge="the-verifier")

        assert await self._validate(store, code, "the-verifier") is not None

    @pytest.mark.asyncio
    async def test_omitted_method_rejects_arbitrary_verifier(self, store):
        code = await self._code(store, code_challenge="the-verifier")

        assert await self._validate(store, code, "anything-at-all") is None

    @pytest.mark.asyncio
    async def test_unknown_method_fails_closed(self, store):
        code = await self._code(
            store,
            code_challenge="the-verifier",
            code_challenge_method="S512",
        )

        assert await self._validate(store, code, "the-verifier") is None
        assert await self._validate(store, code, "anything-at-all") is None

    @pytest.mark.asyncio
    async def test_no_challenge_needs_no_verifier(self, store):
        code = await self._code(store)

        assert await self._validate(store, code) is not None


# ============================================================================
# Token endpoint (HTTP layer)
# ============================================================================


class RecordingProvider(BaseOAuthProvider):
    """Provider that records the credentials the middleware hands it."""

    def __init__(self, expected_secret: str | None = None, redirect_uri_registered: bool = False):
        self.expected_secret = expected_secret
        self.redirect_uri_registered = redirect_uri_registered
        self.received: dict[str, object] = {}

    async def validate_redirect_uri(self, client_id, redirect_uri):
        return self.redirect_uri_registered

    async def exchange_authorization_code(self, code, client_id, redirect_uri, code_verifier=None, client_secret=None):
        self.received = {
            "code": code,
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "code_verifier": code_verifier,
            "client_secret": client_secret,
        }
        if self.expected_secret is not None and client_secret != self.expected_secret:
            raise TokenError(ERROR_INVALID_CLIENT, "Client authentication failed")
        return OAuthToken(access_token="issued-access-token", refresh_token="issued-refresh-token")

    async def exchange_refresh_token(self, refresh_token, client_id, scope=None, client_secret=None):
        self.received = {
            "refresh_token": refresh_token,
            "client_id": client_id,
            "scope": scope,
            "client_secret": client_secret,
        }
        if self.expected_secret is not None and client_secret != self.expected_secret:
            raise TokenError(ERROR_INVALID_CLIENT, "Client authentication failed")
        return OAuthToken(access_token="refreshed-access-token")

    async def authorize(self, params):
        return {"code": "auth-code", "state": params.state}

    async def validate_access_token(self, token):
        return {"user_id": USER_ID}

    async def register_client(self, client_metadata):  # pragma: no cover - unused here
        raise NotImplementedError


class LegacyProvider(BaseOAuthProvider):
    """Provider written before client authentication existed."""

    def __init__(self):
        self.called = False

    async def exchange_authorization_code(self, code, client_id, redirect_uri, code_verifier=None):
        self.called = True
        return OAuthToken(access_token="legacy-access-token")

    async def exchange_refresh_token(self, refresh_token, client_id, scope=None):
        self.called = True
        return OAuthToken(access_token="legacy-access-token")

    async def authorize(self, params):  # pragma: no cover - unused here
        raise NotImplementedError

    async def validate_access_token(self, token):  # pragma: no cover - unused here
        raise NotImplementedError

    async def register_client(self, client_metadata):  # pragma: no cover - unused here
        raise NotImplementedError


def build_middleware(provider):
    """Wire a provider into OAuthMiddleware with a stubbed MCP server."""
    mcp_server = Mock()
    mcp_server.endpoint = Mock(return_value=lambda f: f)
    return OAuthMiddleware(mcp_server=mcp_server, provider=provider)


def token_request(form: dict, headers: dict | None = None) -> Request:
    """Build a token endpoint request."""
    request = Mock(spec=Request)
    request.form = AsyncMock(return_value=form)
    request.headers = headers or {}
    return request


def basic_header(client_id: str, client_secret: str) -> str:
    """Build an Authorization: Basic header value."""
    encoded = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
    return f"Basic {encoded}"


def code_form(**overrides) -> dict:
    """Build an authorization_code token request body."""
    return {
        "grant_type": GRANT_AUTHORIZATION_CODE,
        "code": "the-code",
        "client_id": "client-a",
        "redirect_uri": REDIRECT_URI,
        **overrides,
    }


class TestTokenEndpointCredentialExtraction:
    """The token endpoint must read a secret from either RFC 6749 location."""

    @pytest.mark.asyncio
    async def test_secret_from_post_body(self):
        provider = RecordingProvider()
        middleware = build_middleware(provider)

        response = await middleware._token_endpoint(token_request(code_form(client_secret="s3cret")))

        assert response.status_code == 200
        assert provider.received["client_secret"] == "s3cret"

    @pytest.mark.asyncio
    async def test_secret_from_basic_header(self):
        provider = RecordingProvider()
        middleware = build_middleware(provider)

        response = await middleware._token_endpoint(
            token_request(
                {"grant_type": GRANT_AUTHORIZATION_CODE, "code": "the-code", "redirect_uri": REDIRECT_URI},
                {"authorization": basic_header("client-a", "s3cret")},
            )
        )

        assert response.status_code == 200
        assert provider.received["client_id"] == "client-a"
        assert provider.received["client_secret"] == "s3cret"

    @pytest.mark.asyncio
    async def test_no_secret_reaches_provider_as_none(self):
        provider = RecordingProvider()
        middleware = build_middleware(provider)

        await middleware._token_endpoint(token_request(code_form()))

        assert provider.received["client_secret"] is None

    @pytest.mark.asyncio
    async def test_refresh_grant_forwards_the_secret(self):
        provider = RecordingProvider()
        middleware = build_middleware(provider)

        response = await middleware._token_endpoint(
            token_request(
                {
                    "grant_type": GRANT_REFRESH_TOKEN,
                    "refresh_token": "the-refresh-token",
                    "client_id": "client-a",
                    "client_secret": "s3cret",
                }
            )
        )

        assert response.status_code == 200
        assert provider.received["client_secret"] == "s3cret"

    @pytest.mark.asyncio
    async def test_malformed_basic_header_is_unauthorized(self):
        middleware = build_middleware(RecordingProvider())

        response = await middleware._token_endpoint(
            token_request(code_form(), {"authorization": "Basic !!!not-base64"})
        )

        assert response.status_code == HTTP_UNAUTHORIZED
        assert response.headers["www-authenticate"] == 'Basic realm="oauth"'
        assert json.loads(response.body)["error"] == ERROR_INVALID_CLIENT

    @pytest.mark.asyncio
    async def test_client_id_mismatch_is_a_bad_request(self):
        middleware = build_middleware(RecordingProvider())

        response = await middleware._token_endpoint(
            token_request(code_form(client_id="other"), {"authorization": basic_header("client-a", "s3cret")})
        )

        assert response.status_code == HTTP_BAD_REQUEST
        assert json.loads(response.body)["error"] == ERROR_INVALID_REQUEST

    @pytest.mark.asyncio
    async def test_conflicting_secrets_are_unauthorized(self):
        middleware = build_middleware(RecordingProvider())

        response = await middleware._token_endpoint(
            token_request(
                code_form(client_secret="body-secret"),
                {"authorization": basic_header("client-a", "header-secret")},
            )
        )

        assert response.status_code == HTTP_UNAUTHORIZED
        assert json.loads(response.body)["error"] == ERROR_INVALID_CLIENT


class TestTokenEndpointRejection:
    """A provider that refuses the client must produce a 401."""

    @pytest.mark.asyncio
    async def test_wrong_secret_is_unauthorized(self):
        middleware = build_middleware(RecordingProvider(expected_secret="right-secret"))

        response = await middleware._token_endpoint(token_request(code_form(client_secret="wrong-secret")))

        assert response.status_code == HTTP_UNAUTHORIZED
        assert response.headers["www-authenticate"] == 'Basic realm="oauth"'
        assert json.loads(response.body)["error"] == ERROR_INVALID_CLIENT

    @pytest.mark.asyncio
    async def test_missing_secret_is_unauthorized(self):
        middleware = build_middleware(RecordingProvider(expected_secret="right-secret"))

        response = await middleware._token_endpoint(token_request(code_form()))

        assert response.status_code == HTTP_UNAUTHORIZED

    @pytest.mark.asyncio
    async def test_refresh_grant_rejection_is_unauthorized(self):
        middleware = build_middleware(RecordingProvider(expected_secret="right-secret"))

        response = await middleware._token_endpoint(
            token_request(
                {
                    "grant_type": GRANT_REFRESH_TOKEN,
                    "refresh_token": "the-refresh-token",
                    "client_id": "client-a",
                }
            )
        )

        assert response.status_code == HTTP_UNAUTHORIZED


class TestLegacyProviderCompatibility:
    """Providers written against the old interface keep working."""

    @pytest.mark.asyncio
    async def test_legacy_provider_still_exchanges_without_a_secret(self):
        provider = LegacyProvider()
        middleware = build_middleware(provider)

        response = await middleware._token_endpoint(token_request(code_form()))

        assert response.status_code == 200
        assert provider.called

    @pytest.mark.asyncio
    async def test_legacy_provider_still_refreshes_without_a_secret(self):
        provider = LegacyProvider()
        middleware = build_middleware(provider)

        response = await middleware._token_endpoint(
            token_request(
                {
                    "grant_type": GRANT_REFRESH_TOKEN,
                    "refresh_token": "the-refresh-token",
                    "client_id": "client-a",
                }
            )
        )

        assert response.status_code == 200
        assert provider.called

    @pytest.mark.asyncio
    async def test_secret_sent_to_a_legacy_provider_is_refused_not_ignored(self):
        # Silently dropping the credential would be worse than failing loudly.
        provider = LegacyProvider()
        middleware = build_middleware(provider)

        response = await middleware._token_endpoint(token_request(code_form(client_secret="s3cret")))

        assert response.status_code == HTTP_UNAUTHORIZED
        assert not provider.called


class TestAuthorizeEndpointPkceMethod:
    """A challenge without a method must still be recorded as verifiable."""

    def test_explicit_s256_is_kept(self):
        assert OAuthMiddleware._resolve_code_challenge_method("chal", CODE_CHALLENGE_S256) == CODE_CHALLENGE_S256

    def test_explicit_plain_is_kept(self):
        assert OAuthMiddleware._resolve_code_challenge_method("chal", CODE_CHALLENGE_PLAIN) == CODE_CHALLENGE_PLAIN

    def test_omitted_method_becomes_plain(self):
        assert OAuthMiddleware._resolve_code_challenge_method("chal", None) == CODE_CHALLENGE_PLAIN

    def test_no_challenge_means_no_method(self):
        assert OAuthMiddleware._resolve_code_challenge_method(None, None) is None

    def test_unsupported_method_is_rejected(self):
        with pytest.raises(ValueError, match="Unsupported code_challenge_method"):
            OAuthMiddleware._resolve_code_challenge_method("chal", "S512")


def authorize_request(**overrides) -> Request:
    """Build an authorization endpoint request."""
    request = Mock(spec=Request)
    request.query_params = {
        "response_type": "code",
        "client_id": "client-a",
        "redirect_uri": REDIRECT_URI,
        **overrides,
    }
    return request


class TestAuthorizeEndpointRejectsBadPkce:
    """An unsupported PKCE method must not silently produce an unverifiable code."""

    @pytest.mark.asyncio
    async def test_unsupported_method_redirects_with_an_error(self):
        provider = RecordingProvider(redirect_uri_registered=True)
        middleware = build_middleware(provider)

        response = await middleware._authorize_endpoint(
            authorize_request(code_challenge="chal", code_challenge_method="S512", state="mcp-state")
        )

        assert response.status_code in (302, 307)
        assert response.headers["location"].startswith(REDIRECT_URI)
        assert "error=server_error" in response.headers["location"]
        assert "state=mcp-state" in response.headers["location"]

    @pytest.mark.asyncio
    async def test_error_page_when_no_redirect_uri(self):
        middleware = build_middleware(RecordingProvider())

        request = Mock(spec=Request)
        request.query_params = {"response_type": "code", "client_id": "client-a"}

        response = await middleware._authorize_endpoint(request)

        assert response.status_code == HTTP_BAD_REQUEST
        assert b"Authorization Error" in response.body


class TestAuthorizeErrorRedirectIsValidated:
    """RFC 6749 4.1.2.1 — never auto-redirect an error to an unvalidated URI."""

    @pytest.mark.asyncio
    async def test_unregistered_redirect_uri_gets_an_error_page(self):
        # The attacker's URI is not registered, so the error must not be sent there.
        provider = RecordingProvider(redirect_uri_registered=False)
        middleware = build_middleware(provider)

        response = await middleware._authorize_endpoint(
            authorize_request(
                redirect_uri="http://evil.example/steal",
                code_challenge_method="S512",
                state="mcp-state",
            )
        )

        assert response.status_code == HTTP_BAD_REQUEST
        assert b"Authorization Error" in response.body
        assert "location" not in response.headers

    @pytest.mark.asyncio
    async def test_registered_redirect_uri_still_receives_the_error(self):
        provider = RecordingProvider(redirect_uri_registered=True)
        middleware = build_middleware(provider)

        response = await middleware._authorize_endpoint(
            authorize_request(redirect_uri=REDIRECT_URI, code_challenge_method="S512", state="mcp-state")
        )

        assert response.status_code in (302, 307)
        assert response.headers["location"].startswith(REDIRECT_URI)
        assert "error=server_error" in response.headers["location"]

    @pytest.mark.asyncio
    async def test_missing_client_id_gets_an_error_page(self):
        # Without a client_id there is nothing to validate the URI against.
        provider = RecordingProvider(redirect_uri_registered=True)
        middleware = build_middleware(provider)

        request = Mock(spec=Request)
        request.query_params = {"response_type": "code", "redirect_uri": "http://evil.example/steal"}

        response = await middleware._authorize_endpoint(request)

        assert response.status_code == HTTP_BAD_REQUEST
        assert "location" not in response.headers

    @pytest.mark.asyncio
    async def test_a_raising_validator_refuses_the_redirect(self):
        provider = RecordingProvider(redirect_uri_registered=True)
        provider.validate_redirect_uri = AsyncMock(side_effect=RuntimeError("store down"))
        middleware = build_middleware(provider)

        response = await middleware._authorize_endpoint(authorize_request(code_challenge_method="S512"))

        assert response.status_code == HTTP_BAD_REQUEST
        assert "location" not in response.headers


class TestExternalCallbackLinkSafety:
    """The callback page renders the redirect URI as a link, so re-check it."""

    def _middleware_with_callback(self, redirect_uri):
        provider = RecordingProvider()
        provider.handle_external_callback = AsyncMock(
            return_value={"code": "the-code", "state": "mcp-state", "redirect_uri": redirect_uri}
        )
        return build_middleware(provider)

    def _callback_request(self):
        request = Mock(spec=Request)
        request.query_params = {"code": "google-code", "state": "google-state"}
        return request

    @pytest.mark.asyncio
    async def test_safe_redirect_uri_renders_the_page(self):
        middleware = self._middleware_with_callback(REDIRECT_URI)

        response = await middleware._external_callback_endpoint(self._callback_request())

        assert response.status_code == 200
        assert b"Authorization Successful" in response.body

    @pytest.mark.asyncio
    async def test_javascript_uri_is_refused(self):
        # Defence in depth: registration rejects these, but a client stored before
        # that check existed must not get a javascript: link rendered for it.
        middleware = self._middleware_with_callback("javascript:alert(document.cookie)")

        response = await middleware._external_callback_endpoint(self._callback_request())

        assert response.status_code == HTTP_INTERNAL_SERVER_ERROR
        assert b"javascript:" not in response.body
        assert b"Authorization Error" in response.body

    @pytest.mark.asyncio
    async def test_data_uri_is_refused(self):
        middleware = self._middleware_with_callback("data:text/html,<script>alert(1)</script>")

        response = await middleware._external_callback_endpoint(self._callback_request())

        assert response.status_code == HTTP_INTERNAL_SERVER_ERROR
        assert b"script" not in response.body

    @pytest.mark.asyncio
    async def test_code_is_appended_to_an_existing_query_string(self):
        middleware = self._middleware_with_callback("https://app.example/cb?tenant=acme")

        response = await middleware._external_callback_endpoint(self._callback_request())

        assert b"tenant=acme&amp;code=the-code" in response.body
