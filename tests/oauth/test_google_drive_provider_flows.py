"""
Tests for GoogleDriveOAuthProvider's OAuth Authorization Server behaviour.

Covers the MCP-side flows — authorize, token exchange, refresh, token validation,
client registration and the Google callback — against a fake token store, so no
Google credentials or network access are needed.
"""

from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from chuk_mcp_server.oauth.constants import (
    AUTH_METHOD_CLIENT_SECRET_BASIC,
    AUTH_METHOD_CLIENT_SECRET_POST,
    AUTH_METHOD_NONE,
    ERROR_INVALID_CLIENT,
    ERROR_INVALID_CLIENT_METADATA,
    ERROR_INVALID_GRANT,
    ERROR_INVALID_REDIRECT_URI,
    ERROR_INVALID_TOKEN,
    PROVIDER_GOOGLE_DRIVE,
)
from chuk_mcp_server.oauth.models import (
    AuthorizationParams,
    AuthorizeError,
    RegistrationError,
    TokenError,
)

REDIRECT_URI = "http://localhost:9999/callback"
USER_ID = "google-user-1"
CLIENT_ID = "client-a"
CLIENT_SECRET = "client-a-secret"


class FakeTokenStore:
    """In-memory token store implementing the current interface."""

    sandbox_id = "test-sandbox"

    def __init__(self):
        self.clients: dict[str, dict] = {}
        self.codes: dict[str, dict] = {}
        self.external: dict[str, dict] = {}
        self.external_expired = False
        self.access_tokens: dict[str, dict] = {}
        self.authenticate_result = True
        self.validate_code_result: dict | None = None
        self.refresh_result: tuple[str, str] | None = ("new-access", "new-refresh")
        self.refresh_calls: list[tuple] = []
        self.registered: list[dict] = []
        self.linked: list[dict] = []
        self.created_codes: list[dict] = []
        self.updated_external: list[dict] = []

    async def register_client(self, client_name, redirect_uris, token_endpoint_auth_method=AUTH_METHOD_NONE):
        self.registered.append(
            {
                "client_name": client_name,
                "redirect_uris": redirect_uris,
                "token_endpoint_auth_method": token_endpoint_auth_method,
            }
        )
        return {
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
            "token_endpoint_auth_method": token_endpoint_auth_method,
        }

    async def validate_client(self, client_id, client_secret=None, redirect_uri=None):
        return self.authenticate_result

    async def authenticate_client(self, client_id, client_secret=None, redirect_uri=None):
        return self.authenticate_result

    async def validate_authorization_code(self, code, client_id, redirect_uri, code_verifier=None):
        return self.validate_code_result

    async def create_authorization_code(
        self, user_id, client_id, redirect_uri, scope=None, code_challenge=None, code_challenge_method=None
    ):
        self.created_codes.append(
            {
                "user_id": user_id,
                "client_id": client_id,
                "redirect_uri": redirect_uri,
                "scope": scope,
                "code_challenge": code_challenge,
                "code_challenge_method": code_challenge_method,
            }
        )
        return "issued-code"

    async def create_access_token(self, user_id, client_id, scope=None):
        return ("issued-access", "issued-refresh")

    async def validate_access_token(self, token):
        return self.access_tokens.get(token)

    async def refresh_access_token(self, refresh_token, client_id=None):
        self.refresh_calls.append((refresh_token, client_id))
        return self.refresh_result

    async def get_external_token(self, user_id, provider="external"):
        return self.external.get(user_id)

    async def is_external_token_expired(self, user_id, provider="external"):
        return self.external_expired

    async def link_external_token(self, user_id, access_token, refresh_token=None, expires_in=None, provider=None):
        self.linked.append({"user_id": user_id, "access_token": access_token, "provider": provider})
        self.external[user_id] = {"access_token": access_token, "refresh_token": refresh_token}

    async def update_external_token(self, user_id, access_token, refresh_token=None, expires_in=None, provider=None):
        self.updated_external.append({"user_id": user_id, "access_token": access_token})
        self.external[user_id] = {"access_token": access_token, "refresh_token": refresh_token}


class LegacyTokenStore(FakeTokenStore):
    """A token store predating client-bound refresh and auth-method registration."""

    async def register_client(self, client_name, redirect_uris):
        self.registered.append({"client_name": client_name, "redirect_uris": redirect_uris})
        return {"client_id": CLIENT_ID, "client_secret": CLIENT_SECRET}

    async def refresh_access_token(self, refresh_token):
        self.refresh_calls.append((refresh_token,))
        return self.refresh_result

    # Predates authenticate_client entirely.
    authenticate_client = None


@pytest.fixture
def store():
    return FakeTokenStore()


@pytest.fixture
def provider(store):
    """A provider wired to the fake store, with the Google client stubbed."""
    with patch("chuk_mcp_server.oauth.providers.google_drive.httpx", MagicMock()):
        from chuk_mcp_server.oauth.providers.google_drive import GoogleDriveOAuthProvider

        instance = GoogleDriveOAuthProvider(
            google_client_id="google-id",
            google_client_secret="google-secret",
            google_redirect_uri="http://localhost:8000/oauth/callback",
            token_store=store,
        )
    instance.google_client = Mock()
    instance.google_client.get_authorization_url = Mock(return_value="https://accounts.google.com/auth?state=x")
    return instance


def auth_params(**overrides):
    return AuthorizationParams(
        response_type="code",
        client_id=CLIENT_ID,
        redirect_uri=REDIRECT_URI,
        **overrides,
    )


class TestAuthorize:
    """authorize() validates the client and routes to Google when needed."""

    @pytest.mark.asyncio
    async def test_rejects_unknown_client(self, provider, store):
        store.authenticate_result = False

        with pytest.raises(AuthorizeError) as exc:
            await provider.authorize(auth_params())

        assert exc.value.error == ERROR_INVALID_CLIENT

    @pytest.mark.asyncio
    async def test_redirects_to_google_when_not_linked(self, provider):
        result = await provider.authorize(auth_params(state="mcp-state"))

        assert result["requires_external_authorization"] is True
        assert result["authorization_url"].startswith("https://accounts.google.com/")
        assert result["state"] in provider._pending_authorizations

    @pytest.mark.asyncio
    async def test_pending_authorization_records_client_details(self, provider):
        result = await provider.authorize(
            auth_params(state="mcp-state", scope="drive", code_challenge="chal", code_challenge_method="S256")
        )

        pending = provider._pending_authorizations[result["state"]]
        assert pending["mcp_client_id"] == CLIENT_ID
        assert pending["mcp_redirect_uri"] == REDIRECT_URI
        assert pending["mcp_state"] == "mcp-state"
        assert pending["mcp_scope"] == "drive"
        assert pending["mcp_code_challenge"] == "chal"
        assert pending["mcp_code_challenge_method"] == "S256"

    @pytest.mark.asyncio
    async def test_issues_code_when_already_linked(self, provider, store):
        provider._pending_authorizations["mcp-state"] = {"user_id": USER_ID}
        store.external[USER_ID] = {"access_token": "google-token"}

        result = await provider.authorize(auth_params(state="mcp-state", scope="drive"))

        assert result["code"] == "issued-code"
        assert result["state"] == "mcp-state"
        # The pending entry is consumed.
        assert "mcp-state" not in provider._pending_authorizations
        assert store.created_codes[0]["user_id"] == USER_ID

    @pytest.mark.asyncio
    async def test_falls_back_to_google_when_link_expired(self, provider, store):
        provider._pending_authorizations["mcp-state"] = {"user_id": USER_ID}
        store.external[USER_ID] = {"access_token": "google-token"}
        store.external_expired = True

        result = await provider.authorize(auth_params(state="mcp-state"))

        assert result["requires_external_authorization"] is True


class TestExchangeAuthorizationCode:
    """Token exchange authenticates the client, then validates the code."""

    @pytest.mark.asyncio
    async def test_rejects_unauthenticated_client(self, provider, store):
        store.authenticate_result = False

        with pytest.raises(TokenError) as exc:
            await provider.exchange_authorization_code(
                code="the-code", client_id=CLIENT_ID, redirect_uri=REDIRECT_URI, client_secret="wrong"
            )

        assert exc.value.error == ERROR_INVALID_CLIENT

    @pytest.mark.asyncio
    async def test_client_is_authenticated_before_the_code_is_consumed(self, provider, store):
        store.authenticate_result = False
        store.validate_code_result = {"user_id": USER_ID, "scope": None}

        with pytest.raises(TokenError):
            await provider.exchange_authorization_code(code="the-code", client_id=CLIENT_ID, redirect_uri=REDIRECT_URI)

    @pytest.mark.asyncio
    async def test_rejects_invalid_code(self, provider, store):
        store.validate_code_result = None

        with pytest.raises(TokenError) as exc:
            await provider.exchange_authorization_code(code="the-code", client_id=CLIENT_ID, redirect_uri=REDIRECT_URI)

        assert exc.value.error == ERROR_INVALID_GRANT

    @pytest.mark.asyncio
    async def test_issues_tokens_for_a_valid_code(self, provider, store):
        store.validate_code_result = {"user_id": USER_ID, "scope": "drive"}

        token = await provider.exchange_authorization_code(
            code="the-code",
            client_id=CLIENT_ID,
            redirect_uri=REDIRECT_URI,
            client_secret=CLIENT_SECRET,
        )

        assert token.access_token == "issued-access"
        assert token.refresh_token == "issued-refresh"
        assert token.token_type == "Bearer"
        assert token.scope == "drive"

    @pytest.mark.asyncio
    async def test_falls_back_to_validate_client_on_legacy_stores(self, provider):
        provider.token_store = LegacyTokenStore()
        provider.token_store.validate_code_result = {"user_id": USER_ID, "scope": None}

        token = await provider.exchange_authorization_code(
            code="the-code", client_id=CLIENT_ID, redirect_uri=REDIRECT_URI
        )

        assert token.access_token == "issued-access"


class TestExchangeRefreshToken:
    """Refresh authenticates the client and binds the token to it."""

    @pytest.mark.asyncio
    async def test_rejects_unauthenticated_client(self, provider, store):
        store.authenticate_result = False

        with pytest.raises(TokenError) as exc:
            await provider.exchange_refresh_token(refresh_token="the-token", client_id=CLIENT_ID)

        assert exc.value.error == ERROR_INVALID_CLIENT

    @pytest.mark.asyncio
    async def test_passes_client_id_to_the_store(self, provider, store):
        await provider.exchange_refresh_token(refresh_token="the-token", client_id=CLIENT_ID)

        assert store.refresh_calls == [("the-token", CLIENT_ID)]

    @pytest.mark.asyncio
    async def test_rejects_unknown_refresh_token(self, provider, store):
        store.refresh_result = None

        with pytest.raises(TokenError) as exc:
            await provider.exchange_refresh_token(refresh_token="the-token", client_id=CLIENT_ID)

        assert exc.value.error == ERROR_INVALID_GRANT

    @pytest.mark.asyncio
    async def test_returns_rotated_tokens(self, provider):
        token = await provider.exchange_refresh_token(refresh_token="the-token", client_id=CLIENT_ID, scope="drive")

        assert token.access_token == "new-access"
        assert token.refresh_token == "new-refresh"
        assert token.scope == "drive"

    @pytest.mark.asyncio
    async def test_legacy_store_is_called_without_client_id(self, provider, caplog):
        provider.token_store = LegacyTokenStore()

        token = await provider.exchange_refresh_token(refresh_token="the-token", client_id=CLIENT_ID)

        assert provider.token_store.refresh_calls == [("the-token",)]
        assert token.access_token == "new-access"
        assert "client-bound refresh" in caplog.text


class TestValidateAccessToken:
    """Access token validation also surfaces the linked Google token."""

    @pytest.mark.asyncio
    async def test_rejects_unknown_token(self, provider):
        with pytest.raises(TokenError) as exc:
            await provider.validate_access_token("nope")

        assert exc.value.error == ERROR_INVALID_TOKEN

    @pytest.mark.asyncio
    async def test_requires_a_linked_google_account(self, provider, store):
        store.access_tokens["good"] = {"user_id": USER_ID, "client_id": CLIENT_ID}

        with pytest.raises(TokenError) as exc:
            await provider.validate_access_token("good")

        assert exc.value.error == "insufficient_scope"

    @pytest.mark.asyncio
    async def test_returns_external_token(self, provider, store):
        store.access_tokens["good"] = {"user_id": USER_ID, "client_id": CLIENT_ID}
        store.external[USER_ID] = {"access_token": "google-token", "refresh_token": "google-refresh"}

        result = await provider.validate_access_token("good")

        assert result["user_id"] == USER_ID
        assert result["external_access_token"] == "google-token"
        assert result["external_refresh_token"] == "google-refresh"

    @pytest.mark.asyncio
    async def test_refreshes_an_expired_google_token(self, provider, store):
        store.access_tokens["good"] = {"user_id": USER_ID, "client_id": CLIENT_ID}
        store.external[USER_ID] = {"access_token": "old-token", "refresh_token": "google-refresh"}
        store.external_expired = True
        provider.google_client.refresh_access_token = AsyncMock(
            return_value={"access_token": "fresh-token", "expires_in": 3600}
        )

        result = await provider.validate_access_token("good")

        assert result["external_access_token"] == "fresh-token"
        assert store.updated_external[0]["access_token"] == "fresh-token"

    @pytest.mark.asyncio
    async def test_expired_without_refresh_token_fails(self, provider, store):
        store.access_tokens["good"] = {"user_id": USER_ID, "client_id": CLIENT_ID}
        store.external[USER_ID] = {"access_token": "old-token"}
        store.external_expired = True

        with pytest.raises(TokenError) as exc:
            await provider.validate_access_token("good")

        assert exc.value.error == ERROR_INVALID_TOKEN

    @pytest.mark.asyncio
    async def test_google_refresh_failure_is_reported(self, provider, store):
        store.access_tokens["good"] = {"user_id": USER_ID, "client_id": CLIENT_ID}
        store.external[USER_ID] = {"access_token": "old-token", "refresh_token": "google-refresh"}
        store.external_expired = True
        provider.google_client.refresh_access_token = AsyncMock(side_effect=RuntimeError("google is down"))

        with pytest.raises(TokenError) as exc:
            await provider.validate_access_token("good")

        assert exc.value.error == ERROR_INVALID_TOKEN


class TestRegisterClient:
    """Registration honours the client's declared auth method."""

    @pytest.mark.asyncio
    async def test_requires_a_redirect_uri(self, provider):
        with pytest.raises(RegistrationError) as exc:
            await provider.register_client({"client_name": "App"})

        assert exc.value.error == ERROR_INVALID_REDIRECT_URI

    @pytest.mark.asyncio
    async def test_undeclared_client_registers_as_public(self, provider, store):
        info = await provider.register_client({"client_name": "App", "redirect_uris": [REDIRECT_URI]})

        assert info.token_endpoint_auth_method == AUTH_METHOD_NONE
        assert store.registered[0]["token_endpoint_auth_method"] == AUTH_METHOD_NONE

    @pytest.mark.asyncio
    async def test_declared_method_is_forwarded(self, provider, store):
        info = await provider.register_client(
            {
                "client_name": "App",
                "redirect_uris": [REDIRECT_URI],
                "token_endpoint_auth_method": AUTH_METHOD_CLIENT_SECRET_POST,
            }
        )

        assert info.token_endpoint_auth_method == AUTH_METHOD_CLIENT_SECRET_POST
        assert store.registered[0]["token_endpoint_auth_method"] == AUTH_METHOD_CLIENT_SECRET_POST

    @pytest.mark.asyncio
    async def test_basic_method_is_forwarded(self, provider, store):
        await provider.register_client(
            {
                "client_name": "App",
                "redirect_uris": [REDIRECT_URI],
                "token_endpoint_auth_method": AUTH_METHOD_CLIENT_SECRET_BASIC,
            }
        )

        assert store.registered[0]["token_endpoint_auth_method"] == AUTH_METHOD_CLIENT_SECRET_BASIC

    @pytest.mark.asyncio
    async def test_unsupported_method_is_rejected(self, provider):
        with pytest.raises(RegistrationError) as exc:
            await provider.register_client(
                {
                    "client_name": "App",
                    "redirect_uris": [REDIRECT_URI],
                    "token_endpoint_auth_method": "private_key_jwt",
                }
            )

        assert exc.value.error == ERROR_INVALID_CLIENT_METADATA

    @pytest.mark.asyncio
    async def test_returns_credentials(self, provider):
        info = await provider.register_client({"client_name": "App", "redirect_uris": [REDIRECT_URI]})

        assert info.client_id == CLIENT_ID
        assert info.client_secret == CLIENT_SECRET
        assert info.client_name == "App"
        assert info.redirect_uris == [REDIRECT_URI]

    @pytest.mark.asyncio
    async def test_legacy_store_registers_public_clients(self, provider):
        provider.token_store = LegacyTokenStore()

        info = await provider.register_client({"client_name": "App", "redirect_uris": [REDIRECT_URI]})

        assert info.token_endpoint_auth_method == AUTH_METHOD_NONE

    @pytest.mark.asyncio
    async def test_legacy_store_refuses_confidential_clients(self, provider):
        # Better to refuse than to issue a secret that could never be enforced.
        provider.token_store = LegacyTokenStore()

        with pytest.raises(RegistrationError) as exc:
            await provider.register_client(
                {
                    "client_name": "App",
                    "redirect_uris": [REDIRECT_URI],
                    "token_endpoint_auth_method": AUTH_METHOD_CLIENT_SECRET_POST,
                }
            )

        assert exc.value.error == ERROR_INVALID_CLIENT_METADATA


class TestHandleExternalCallback:
    """The Google callback links the account and mints an MCP code."""

    def _pending(self, provider, state="google-state"):
        provider._pending_authorizations[state] = {
            "mcp_client_id": CLIENT_ID,
            "mcp_redirect_uri": REDIRECT_URI,
            "mcp_state": "mcp-state",
            "mcp_scope": "drive",
            "mcp_code_challenge": "chal",
            "mcp_code_challenge_method": "S256",
        }
        return state

    @pytest.mark.asyncio
    async def test_unknown_state_is_rejected(self, provider):
        with pytest.raises(ValueError, match="Invalid or expired state"):
            await provider.handle_external_callback("google-code", "unknown-state")

    @pytest.mark.asyncio
    async def test_google_token_exchange_failure(self, provider):
        state = self._pending(provider)
        provider.google_client.exchange_code_for_token = AsyncMock(side_effect=RuntimeError("boom"))

        with pytest.raises(ValueError, match="Google token exchange failed"):
            await provider.handle_external_callback("google-code", state)

    @pytest.mark.asyncio
    async def test_user_info_failure(self, provider):
        state = self._pending(provider)
        provider.google_client.exchange_code_for_token = AsyncMock(return_value={"access_token": "g-token"})
        provider.google_client.get_user_info = AsyncMock(side_effect=RuntimeError("boom"))

        with pytest.raises(ValueError, match="Failed to get Google user info"):
            await provider.handle_external_callback("google-code", state)

    @pytest.mark.asyncio
    async def test_successful_callback(self, provider, store):
        state = self._pending(provider)
        provider.google_client.exchange_code_for_token = AsyncMock(
            return_value={"access_token": "g-token", "refresh_token": "g-refresh", "expires_in": 3600}
        )
        provider.google_client.get_user_info = AsyncMock(return_value={"sub": USER_ID})

        result = await provider.handle_external_callback("google-code", state)

        assert result["code"] == "issued-code"
        assert result["state"] == "mcp-state"
        assert result["redirect_uri"] == REDIRECT_URI
        assert store.linked[0]["user_id"] == USER_ID
        assert store.linked[0]["provider"] == PROVIDER_GOOGLE_DRIVE
        # The PKCE challenge from the original request is carried through.
        assert store.created_codes[0]["code_challenge"] == "chal"
        assert store.created_codes[0]["code_challenge_method"] == "S256"
        assert state not in provider._pending_authorizations


class TestProviderConstruction:
    """Constructor wiring."""

    def test_creates_a_default_token_store(self):
        with patch("chuk_mcp_server.oauth.providers.google_drive.httpx", MagicMock()):
            from chuk_mcp_server.oauth.providers.google_drive import GoogleDriveOAuthProvider
            from chuk_mcp_server.oauth.token_store import TokenStore

            instance = GoogleDriveOAuthProvider(
                google_client_id="google-id",
                google_client_secret="google-secret",
                google_redirect_uri="http://localhost:8000/oauth/callback",
                sandbox_id="custom-sandbox",
            )

        assert isinstance(instance.token_store, TokenStore)
        assert instance.token_store.sandbox_id == "custom-sandbox"
        assert instance._pending_authorizations == {}


class TestRedirectUriRegistrationHardening:
    """Registration must not accept a URI the callback page would render as a link."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "uri",
        [
            "javascript:alert(document.cookie)",
            "data:text/html,<script>alert(1)</script>",
            "vbscript:msgbox(1)",
            "file:///etc/passwd",
            "/relative/callback",
            "   ",
        ],
    )
    async def test_dangerous_redirect_uris_are_rejected(self, provider, uri):
        with pytest.raises(RegistrationError) as exc:
            await provider.register_client({"client_name": "App", "redirect_uris": [uri]})

        assert exc.value.error == ERROR_INVALID_REDIRECT_URI

    @pytest.mark.asyncio
    async def test_one_bad_uri_rejects_the_whole_registration(self, provider, store):
        with pytest.raises(RegistrationError):
            await provider.register_client(
                {"client_name": "App", "redirect_uris": [REDIRECT_URI, "javascript:alert(1)"]}
            )

        assert store.registered == []

    @pytest.mark.asyncio
    async def test_non_string_redirect_uri_is_rejected(self, provider):
        with pytest.raises(RegistrationError) as exc:
            await provider.register_client({"client_name": "App", "redirect_uris": [{"not": "a string"}]})

        assert exc.value.error == ERROR_INVALID_REDIRECT_URI

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "uri",
        ["https://app.example/callback", "http://localhost:9999/callback", "com.example.app:/cb"],
    )
    async def test_navigable_uris_are_accepted(self, provider, uri):
        info = await provider.register_client({"client_name": "App", "redirect_uris": [uri]})

        assert info.redirect_uris == [uri]


class TestAccessTokenLifetime:
    """expires_in must match the TTL the token store actually applies."""

    @pytest.mark.asyncio
    async def test_reports_the_stores_ttl(self, provider, store):
        store.access_token_ttl = 900
        store.validate_code_result = {"user_id": USER_ID, "scope": None}

        token = await provider.exchange_authorization_code(
            code="the-code", client_id=CLIENT_ID, redirect_uri=REDIRECT_URI
        )

        assert token.expires_in == 900

    @pytest.mark.asyncio
    async def test_refresh_reports_the_stores_ttl(self, provider, store):
        store.access_token_ttl = 900

        token = await provider.exchange_refresh_token(refresh_token="the-token", client_id=CLIENT_ID)

        assert token.expires_in == 900

    @pytest.mark.asyncio
    async def test_falls_back_when_the_store_has_no_ttl(self, provider, store):
        store.validate_code_result = {"user_id": USER_ID, "scope": None}

        token = await provider.exchange_authorization_code(
            code="the-code", client_id=CLIENT_ID, redirect_uri=REDIRECT_URI
        )

        assert token.expires_in == 3600


class TestValidateRedirectUriHook:
    """The default hook answers from the token store, and fails closed."""

    @pytest.mark.asyncio
    async def test_registered_pairing_is_accepted(self, provider, store):
        store.authenticate_result = True

        assert await provider.validate_redirect_uri(CLIENT_ID, REDIRECT_URI)

    @pytest.mark.asyncio
    async def test_unregistered_pairing_is_rejected(self, provider, store):
        store.authenticate_result = False

        assert not await provider.validate_redirect_uri(CLIENT_ID, "http://evil.example/cb")

    @pytest.mark.asyncio
    async def test_a_raising_store_fails_closed(self, provider):
        provider.token_store.validate_client = AsyncMock(side_effect=RuntimeError("store down"))

        assert not await provider.validate_redirect_uri(CLIENT_ID, REDIRECT_URI)

    @pytest.mark.asyncio
    async def test_a_provider_without_a_token_store_fails_closed(self, provider):
        del provider.token_store

        assert not await provider.validate_redirect_uri(CLIENT_ID, REDIRECT_URI)
