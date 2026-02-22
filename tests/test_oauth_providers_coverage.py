#!/usr/bin/env python3
"""Tests for OAuth providers to improve coverage.

Covers:
- src/chuk_mcp_server/oauth/providers/__init__.py  (ImportError branch)
- src/chuk_mcp_server/oauth/providers/google_drive.py (lines 241-561)
"""

import importlib
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from chuk_mcp_server.oauth.models import (
    AuthorizationParams,
    AuthorizeError,
    OAuthToken,
    RegistrationError,
    TokenError,
)
from chuk_mcp_server.oauth.providers.google_drive import (
    GoogleDriveOAuthClient,
    GoogleDriveOAuthProvider,
)

# ===========================================================================
# providers/__init__.py — ImportError fallback branch
# ===========================================================================


class TestProvidersInit:
    """Cover the except ImportError: pass branch in providers/__init__.py."""

    def test_providers_init_without_google_drive(self):
        """When google_drive import fails, __all__ should be empty."""
        import chuk_mcp_server.oauth.providers as providers_mod

        # Simulate google_drive import failure
        with patch.dict(
            "sys.modules",
            {"chuk_mcp_server.oauth.providers.google_drive": None},
        ):
            importlib.reload(providers_mod)
            assert "GoogleDriveOAuthProvider" not in providers_mod.__all__
            assert "GoogleDriveOAuthClient" not in providers_mod.__all__

        # Restore normal state
        importlib.reload(providers_mod)
        assert "GoogleDriveOAuthProvider" in providers_mod.__all__
        assert "GoogleDriveOAuthClient" in providers_mod.__all__

    def test_providers_init_with_google_drive(self):
        """Normal case: google_drive is importable."""
        import chuk_mcp_server.oauth.providers as providers_mod

        importlib.reload(providers_mod)
        assert "GoogleDriveOAuthProvider" in providers_mod.__all__
        assert "GoogleDriveOAuthClient" in providers_mod.__all__


# ===========================================================================
# GoogleDriveOAuthClient
# ===========================================================================


class TestGoogleDriveOAuthClient:
    """Test the GoogleDriveOAuthClient methods."""

    @pytest.fixture()
    def client(self):
        return GoogleDriveOAuthClient(
            client_id="test-client-id",
            client_secret="test-client-secret",
            redirect_uri="http://localhost:8000/oauth/callback",
        )

    def test_init(self, client):
        assert client.client_id == "test-client-id"
        assert client.client_secret == "test-client-secret"
        assert client.redirect_uri == "http://localhost:8000/oauth/callback"

    def test_get_authorization_url(self, client):
        url = client.get_authorization_url(state="test-state-123")
        assert url.startswith("https://accounts.google.com/o/oauth2/v2/auth?")
        assert "client_id=test-client-id" in url
        assert "state=test-state-123" in url
        assert "access_type=offline" in url
        assert "response_type=code" in url

    @pytest.mark.asyncio
    async def test_exchange_code_for_token(self, client):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "access_token": "ya29.xxx",
            "refresh_token": "1//xxx",
            "expires_in": 3600,
            "token_type": "Bearer",
        }
        mock_response.raise_for_status = MagicMock()

        mock_http_client = AsyncMock()
        mock_http_client.post.return_value = mock_response
        mock_http_client.__aenter__ = AsyncMock(return_value=mock_http_client)
        mock_http_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_http_client):
            result = await client.exchange_code_for_token("auth-code-123")

        assert result["access_token"] == "ya29.xxx"
        assert result["refresh_token"] == "1//xxx"
        mock_http_client.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_refresh_access_token(self, client):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "access_token": "ya29.new",
            "expires_in": 3600,
            "token_type": "Bearer",
        }
        mock_response.raise_for_status = MagicMock()

        mock_http_client = AsyncMock()
        mock_http_client.post.return_value = mock_response
        mock_http_client.__aenter__ = AsyncMock(return_value=mock_http_client)
        mock_http_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_http_client):
            result = await client.refresh_access_token("1//refresh-token")

        assert result["access_token"] == "ya29.new"
        mock_http_client.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_user_info(self, client):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "sub": "user-123",
            "name": "Test User",
            "email": "test@example.com",
        }
        mock_response.raise_for_status = MagicMock()

        mock_http_client = AsyncMock()
        mock_http_client.get.return_value = mock_response
        mock_http_client.__aenter__ = AsyncMock(return_value=mock_http_client)
        mock_http_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_http_client):
            result = await client.get_user_info("ya29.xxx")

        assert result["sub"] == "user-123"
        assert result["email"] == "test@example.com"
        mock_http_client.get.assert_called_once()


# ===========================================================================
# GoogleDriveOAuthProvider
# ===========================================================================


class TestGoogleDriveOAuthProvider:
    """Test GoogleDriveOAuthProvider methods (lines 241-561)."""

    @pytest.fixture()
    def mock_token_store(self):
        """Create a mock token store with all required methods."""
        store = AsyncMock()
        store.sandbox_id = "test-sandbox"
        store.validate_client = AsyncMock(return_value=True)
        store.get_external_token = AsyncMock(return_value=None)
        store.is_external_token_expired = AsyncMock(return_value=False)
        store.create_authorization_code = AsyncMock(return_value="auth-code-abc")
        store.validate_authorization_code = AsyncMock(
            return_value={
                "user_id": "user-123",
                "client_id": "client-abc",
                "scope": "read write",
            }
        )
        store.create_access_token = AsyncMock(return_value=("access-token-xyz", "refresh-token-xyz"))
        store.validate_access_token = AsyncMock(
            return_value={
                "user_id": "user-123",
                "client_id": "client-abc",
                "scope": "read write",
            }
        )
        store.refresh_access_token = AsyncMock(return_value=("new-access-token", "new-refresh-token"))
        store.register_client = AsyncMock(
            return_value={
                "client_id": "new-client-id",
                "client_secret": "new-client-secret",
            }
        )
        store.link_external_token = AsyncMock()
        store.update_external_token = AsyncMock()
        return store

    @pytest.fixture()
    def provider(self, mock_token_store):
        """Create a GoogleDriveOAuthProvider with mocked token store."""
        p = GoogleDriveOAuthProvider(
            google_client_id="google-client-id",
            google_client_secret="google-client-secret",
            google_redirect_uri="http://localhost:8000/oauth/callback",
            token_store=mock_token_store,
        )
        return p

    def test_init_with_token_store(self, provider, mock_token_store):
        assert provider.token_store is mock_token_store
        assert provider.google_client.client_id == "google-client-id"

    def test_init_without_token_store(self):
        """When no token_store is given, a default TokenStore is created."""
        p = GoogleDriveOAuthProvider(
            google_client_id="gid",
            google_client_secret="gsec",
            google_redirect_uri="http://localhost/cb",
        )
        assert p.token_store is not None

    # -----------------------------------------------------------------------
    # authorize()
    # -----------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_authorize_invalid_client(self, provider, mock_token_store):
        """authorize() should raise AuthorizeError for invalid client."""
        mock_token_store.validate_client.return_value = False
        params = AuthorizationParams(
            response_type="code",
            client_id="bad-client",
            redirect_uri="http://evil.com/callback",
        )
        with pytest.raises(AuthorizeError):
            await provider.authorize(params)

    @pytest.mark.asyncio
    async def test_authorize_redirects_to_google(self, provider, mock_token_store):
        """When no pending user, authorize() should redirect to Google."""
        params = AuthorizationParams(
            response_type="code",
            client_id="client-abc",
            redirect_uri="http://localhost/callback",
            state="mcp-state-1",
        )
        result = await provider.authorize(params)
        assert result.get("requires_external_authorization") is True
        assert "authorization_url" in result
        assert "accounts.google.com" in result["authorization_url"]

    @pytest.mark.asyncio
    async def test_authorize_with_existing_google_token(self, provider, mock_token_store):
        """When user has valid Google token, return authorization code directly."""
        state = "my-state"
        # Set up pending auth with a user_id
        provider._pending_authorizations[state] = {"user_id": "user-123"}

        # Token store returns valid google token
        mock_token_store.get_external_token.return_value = {
            "access_token": "ya29.valid",
            "refresh_token": "1//refresh",
        }
        mock_token_store.is_external_token_expired.return_value = False

        params = AuthorizationParams(
            response_type="code",
            client_id="client-abc",
            redirect_uri="http://localhost/callback",
            state=state,
        )
        result = await provider.authorize(params)
        assert "code" in result
        assert result["code"] == "auth-code-abc"
        # Pending auth should be cleaned up
        assert state not in provider._pending_authorizations

    @pytest.mark.asyncio
    async def test_authorize_with_no_state(self, provider, mock_token_store):
        """authorize() with state=None should still work (defaults to empty string)."""
        params = AuthorizationParams(
            response_type="code",
            client_id="client-abc",
            redirect_uri="http://localhost/callback",
            state=None,
        )
        result = await provider.authorize(params)
        assert result.get("requires_external_authorization") is True

    # -----------------------------------------------------------------------
    # exchange_authorization_code()
    # -----------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_exchange_authorization_code_success(self, provider, mock_token_store):
        token = await provider.exchange_authorization_code(
            code="valid-code-123",
            client_id="client-abc",
            redirect_uri="http://localhost/callback",
        )
        assert isinstance(token, OAuthToken)
        assert token.access_token == "access-token-xyz"
        assert token.refresh_token == "refresh-token-xyz"
        assert token.token_type == "Bearer"

    @pytest.mark.asyncio
    async def test_exchange_authorization_code_invalid(self, provider, mock_token_store):
        """Invalid code should raise TokenError."""
        mock_token_store.validate_authorization_code.return_value = None
        with pytest.raises(TokenError):
            await provider.exchange_authorization_code(
                code="bad-code",
                client_id="client-abc",
                redirect_uri="http://localhost/callback",
            )

    @pytest.mark.asyncio
    async def test_exchange_authorization_code_with_verifier(self, provider, mock_token_store):
        token = await provider.exchange_authorization_code(
            code="valid-code",
            client_id="client-abc",
            redirect_uri="http://localhost/callback",
            code_verifier="my-code-verifier",
        )
        assert isinstance(token, OAuthToken)
        mock_token_store.validate_authorization_code.assert_called_once_with(
            code="valid-code",
            client_id="client-abc",
            redirect_uri="http://localhost/callback",
            code_verifier="my-code-verifier",
        )

    # -----------------------------------------------------------------------
    # exchange_refresh_token()
    # -----------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_exchange_refresh_token_success(self, provider, mock_token_store):
        token = await provider.exchange_refresh_token(
            refresh_token="refresh-abc",
            client_id="client-abc",
        )
        assert isinstance(token, OAuthToken)
        assert token.access_token == "new-access-token"
        assert token.refresh_token == "new-refresh-token"

    @pytest.mark.asyncio
    async def test_exchange_refresh_token_invalid(self, provider, mock_token_store):
        mock_token_store.refresh_access_token.return_value = None
        with pytest.raises(TokenError):
            await provider.exchange_refresh_token(
                refresh_token="bad-refresh",
                client_id="client-abc",
            )

    @pytest.mark.asyncio
    async def test_exchange_refresh_token_with_scope(self, provider, mock_token_store):
        token = await provider.exchange_refresh_token(
            refresh_token="refresh-abc",
            client_id="client-abc",
            scope="read",
        )
        assert token.scope == "read"

    # -----------------------------------------------------------------------
    # validate_access_token()
    # -----------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_validate_access_token_success(self, provider, mock_token_store):
        mock_token_store.get_external_token.return_value = {
            "access_token": "ya29.google",
            "refresh_token": "1//refresh",
        }
        mock_token_store.is_external_token_expired.return_value = False

        result = await provider.validate_access_token("valid-mcp-token")
        assert result["user_id"] == "user-123"
        assert result["external_access_token"] == "ya29.google"
        assert result["external_refresh_token"] == "1//refresh"

    @pytest.mark.asyncio
    async def test_validate_access_token_invalid_mcp_token(self, provider, mock_token_store):
        mock_token_store.validate_access_token.return_value = None
        with pytest.raises(TokenError):
            await provider.validate_access_token("bad-mcp-token")

    @pytest.mark.asyncio
    async def test_validate_access_token_no_google_token(self, provider, mock_token_store):
        """No linked Google Drive token raises TokenError with insufficient_scope."""
        mock_token_store.get_external_token.return_value = None
        with pytest.raises(TokenError):
            await provider.validate_access_token("valid-mcp-token")

    @pytest.mark.asyncio
    async def test_validate_access_token_refreshes_expired_google_token(self, provider, mock_token_store):
        """Expired Google token should be refreshed automatically."""
        mock_token_store.get_external_token.side_effect = [
            # First call: expired token with refresh_token
            {"access_token": "ya29.old", "refresh_token": "1//refresh"},
            # Second call after update: refreshed token
            {"access_token": "ya29.new", "refresh_token": "1//refresh"},
        ]
        mock_token_store.is_external_token_expired.return_value = True

        # Mock the google client refresh
        provider.google_client.refresh_access_token = AsyncMock(
            return_value={
                "access_token": "ya29.new",
                "expires_in": 3600,
            }
        )

        result = await provider.validate_access_token("valid-mcp-token")
        assert result["external_access_token"] == "ya29.new"
        mock_token_store.update_external_token.assert_called_once()

    @pytest.mark.asyncio
    async def test_validate_access_token_refresh_fails(self, provider, mock_token_store):
        """If Google token refresh fails, raise TokenError."""
        mock_token_store.get_external_token.return_value = {
            "access_token": "ya29.old",
            "refresh_token": "1//refresh",
        }
        mock_token_store.is_external_token_expired.return_value = True

        provider.google_client.refresh_access_token = AsyncMock(side_effect=Exception("Google API error"))

        with pytest.raises(TokenError):
            await provider.validate_access_token("valid-mcp-token")

    @pytest.mark.asyncio
    async def test_validate_access_token_expired_no_refresh_token(self, provider, mock_token_store):
        """Expired Google token without refresh_token raises TokenError."""
        mock_token_store.get_external_token.return_value = {
            "access_token": "ya29.old",
            # No refresh_token
        }
        mock_token_store.is_external_token_expired.return_value = True

        with pytest.raises(TokenError):
            await provider.validate_access_token("valid-mcp-token")

    # -----------------------------------------------------------------------
    # register_client()
    # -----------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_register_client_success(self, provider, mock_token_store):
        result = await provider.register_client(
            client_metadata={
                "client_name": "My App",
                "redirect_uris": ["http://localhost/callback"],
            }
        )
        assert result.client_id == "new-client-id"
        assert result.client_secret == "new-client-secret"
        assert result.client_name == "My App"
        assert result.redirect_uris == ["http://localhost/callback"]

    @pytest.mark.asyncio
    async def test_register_client_no_redirect_uris(self, provider, mock_token_store):
        with pytest.raises(RegistrationError):
            await provider.register_client(
                client_metadata={
                    "client_name": "Bad App",
                    "redirect_uris": [],
                }
            )

    @pytest.mark.asyncio
    async def test_register_client_missing_redirect_uris(self, provider, mock_token_store):
        with pytest.raises(RegistrationError):
            await provider.register_client(
                client_metadata={
                    "client_name": "Bad App",
                }
            )

    @pytest.mark.asyncio
    async def test_register_client_default_name(self, provider, mock_token_store):
        result = await provider.register_client(
            client_metadata={
                "redirect_uris": ["http://localhost/callback"],
            }
        )
        assert result.client_name == "Unknown Client"

    # -----------------------------------------------------------------------
    # handle_external_callback()
    # -----------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_handle_external_callback_success(self, provider, mock_token_store):
        """Successful Google callback should create MCP authorization code."""
        state = "google-state-xyz"
        provider._pending_authorizations[state] = {
            "mcp_client_id": "client-abc",
            "mcp_redirect_uri": "http://localhost/callback",
            "mcp_state": "mcp-state-1",
            "mcp_scope": "read write",
            "mcp_code_challenge": None,
            "mcp_code_challenge_method": None,
        }

        # Mock Google token exchange and user info
        provider.google_client.exchange_code_for_token = AsyncMock(
            return_value={
                "access_token": "ya29.new",
                "refresh_token": "1//new-refresh",
                "expires_in": 3600,
            }
        )
        provider.google_client.get_user_info = AsyncMock(
            return_value={
                "sub": "google-user-456",
                "name": "Test User",
                "email": "test@gmail.com",
            }
        )

        result = await provider.handle_external_callback(
            code="google-auth-code",
            state=state,
        )

        assert "code" in result
        assert result["code"] == "auth-code-abc"
        assert result["state"] == "mcp-state-1"
        assert result["redirect_uri"] == "http://localhost/callback"
        # Pending auth should be cleaned up
        assert state not in provider._pending_authorizations
        # Token should be linked
        mock_token_store.link_external_token.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_external_callback_invalid_state(self, provider):
        with pytest.raises(ValueError, match="Invalid or expired state"):
            await provider.handle_external_callback(
                code="some-code",
                state="unknown-state",
            )

    @pytest.mark.asyncio
    async def test_handle_external_callback_google_exchange_fails(self, provider):
        state = "google-state-fail"
        provider._pending_authorizations[state] = {
            "mcp_client_id": "client-abc",
            "mcp_redirect_uri": "http://localhost/callback",
            "mcp_state": "s",
            "mcp_scope": None,
            "mcp_code_challenge": None,
            "mcp_code_challenge_method": None,
        }

        provider.google_client.exchange_code_for_token = AsyncMock(
            side_effect=httpx.HTTPStatusError(
                "401 Unauthorized",
                request=MagicMock(),
                response=MagicMock(),
            )
        )

        with pytest.raises(ValueError, match="Google token exchange failed"):
            await provider.handle_external_callback(
                code="bad-code",
                state=state,
            )

    @pytest.mark.asyncio
    async def test_handle_external_callback_user_info_fails(self, provider):
        state = "google-state-userinfo-fail"
        provider._pending_authorizations[state] = {
            "mcp_client_id": "client-abc",
            "mcp_redirect_uri": "http://localhost/callback",
            "mcp_state": "s",
            "mcp_scope": None,
            "mcp_code_challenge": None,
            "mcp_code_challenge_method": None,
        }

        provider.google_client.exchange_code_for_token = AsyncMock(
            return_value={
                "access_token": "ya29.xxx",
                "refresh_token": "1//xxx",
                "expires_in": 3600,
            }
        )
        provider.google_client.get_user_info = AsyncMock(side_effect=Exception("User info API error"))

        with pytest.raises(ValueError, match="Failed to get Google user info"):
            await provider.handle_external_callback(
                code="valid-code",
                state=state,
            )
