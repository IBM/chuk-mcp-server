"""
Token storage using chuk-sessions.

Stores OAuth tokens using chuk-sessions library with pluggable backends:
- Memory backend (default): For development and testing
- Redis backend: For persistent/distributed use

This provides:
- Secure storage (Redis for distributed deployments)
- Automatic TTL/expiration
- Multi-tenant isolation via sandbox_id
- High performance (1.8M ops/sec memory, 20K ops/sec Redis)
- Horizontally scalable

Configuration:
    Backend Configuration:
        SESSION_PROVIDER=redis
        SESSION_REDIS_URL=redis://localhost:6379/0

    Token TTL Configuration (in seconds):
        OAUTH_AUTH_CODE_TTL=300           # Authorization code lifetime (default: 5 minutes)
        OAUTH_ACCESS_TOKEN_TTL=900        # Access token lifetime (default: 15 minutes)
        OAUTH_REFRESH_TOKEN_TTL=86400     # Refresh token lifetime (default: 1 day)
        OAUTH_CLIENT_REGISTRATION_TTL=31536000  # Client registration lifetime (default: 1 year)
        OAUTH_EXTERNAL_TOKEN_TTL=86400    # External provider token lifetime (default: 1 day)
"""

import hashlib
import os
import secrets
from datetime import datetime, timedelta
from typing import Any

import orjson
from chuk_sessions import get_session

from .base_token_store import BaseTokenStore
from .constants import (
    CODE_CHALLENGE_PLAIN,
    CODE_CHALLENGE_S256,
    CONFIDENTIAL_AUTH_METHODS,
    DEFAULT_AUTH_METHOD,
    ENV_ACCESS_TOKEN_TTL,
    ENV_AUTH_CODE_TTL,
    ENV_CLIENT_REGISTRATION_TTL,
    ENV_EXTERNAL_TOKEN_TTL,
    ENV_PENDING_AUTH_TTL,
    ENV_REFRESH_TOKEN_TTL,
    TOKEN_EXPIRY_BUFFER_MINUTES,
    TTL_ACCESS_TOKEN,
    TTL_AUTH_CODE,
    TTL_CLIENT_REGISTRATION,
    TTL_EXTERNAL_TOKEN,
    TTL_PENDING_AUTH,
    TTL_REFRESH_TOKEN,
)
from .token_models import (
    AccessTokenData,
    AuthorizationCodeData,
    ClientData,
    ExternalTokenData,
    RefreshTokenData,
)


class TokenStore(BaseTokenStore):
    """
    Token store backed by chuk-sessions.

    Supports both memory (development) and Redis (distributed) backends.
    Backend is configured via environment variables (see module docstring).

    Session Keys:
    - auth_code:{code} -> authorization code data
    - access_token:{token} -> MCP access token data
    - refresh_token:{token} -> MCP refresh token data
    - linkedin_token:{user_id} -> LinkedIn token data
    - client:{client_id} -> registered client data
    """

    def __init__(self, sandbox_id: str = "chuk-mcp-linkedin"):
        """
        Initialize session-based token store.

        Args:
            sandbox_id: Sandbox ID for multi-tenant isolation (prefixed to all keys)
        """
        self.sandbox_id = sandbox_id

        # Load TTL configuration from environment with defaults
        self.auth_code_ttl = int(os.getenv(ENV_AUTH_CODE_TTL, str(TTL_AUTH_CODE)))
        self.access_token_ttl = int(os.getenv(ENV_ACCESS_TOKEN_TTL, str(TTL_ACCESS_TOKEN)))
        self.refresh_token_ttl = int(os.getenv(ENV_REFRESH_TOKEN_TTL, str(TTL_REFRESH_TOKEN)))
        self.client_registration_ttl = int(os.getenv(ENV_CLIENT_REGISTRATION_TTL, str(TTL_CLIENT_REGISTRATION)))
        self.external_token_ttl = int(os.getenv(ENV_EXTERNAL_TOKEN_TTL, str(TTL_EXTERNAL_TOKEN)))
        self.pending_auth_ttl = int(os.getenv(ENV_PENDING_AUTH_TTL, str(TTL_PENDING_AUTH)))

    # ============================================================================
    # MCP Authorization Codes
    # ============================================================================

    async def create_authorization_code(
        self,
        user_id: str,
        client_id: str,
        redirect_uri: str,
        scope: str | None = None,
        code_challenge: str | None = None,
        code_challenge_method: str | None = None,
    ) -> str:
        """
        Create authorization code for MCP client.

        Args:
            user_id: User identifier (from LinkedIn OAuth)
            client_id: MCP client ID
            redirect_uri: Redirect URI for callback
            scope: Requested scopes
            code_challenge: PKCE code challenge
            code_challenge_method: PKCE method (S256 or plain)

        Returns:
            Authorization code
        """
        code = secrets.token_urlsafe(32)

        code_data = AuthorizationCodeData(
            user_id=user_id,
            client_id=client_id,
            redirect_uri=redirect_uri,
            scope=scope,
            code_challenge=code_challenge,
            code_challenge_method=code_challenge_method,
        )

        # Store with configurable TTL (default: 5 minutes)
        async with get_session() as session:
            await session.setex(
                f"{self.sandbox_id}:auth_code:{code}",
                self.auth_code_ttl,
                code_data.to_json_bytes(),
            )

        return code

    async def validate_authorization_code(
        self,
        code: str,
        client_id: str,
        redirect_uri: str,
        code_verifier: str | None = None,
    ) -> dict[str, Any] | None:
        """
        Validate authorization code and return user info.

        Args:
            code: Authorization code
            client_id: MCP client ID
            redirect_uri: Redirect URI (must match)
            code_verifier: PKCE code verifier

        Returns:
            User info dict or None if invalid
        """
        async with get_session() as session:
            code_json = await session.get(f"{self.sandbox_id}:auth_code:{code}")
            if not code_json:
                return None

            code_data = AuthorizationCodeData.from_dict(orjson.loads(code_json))

            # Validate client_id and redirect_uri
            if code_data.client_id != client_id:
                return None
            if code_data.redirect_uri != redirect_uri:
                return None

            # Validate PKCE if present
            if code_data.code_challenge:
                if not code_verifier:
                    return None

                # RFC 7636 section 4.3: an absent code_challenge_method means "plain".
                # Never fall through without verifying — an unrecognised method must
                # fail closed rather than silently accept any verifier.
                challenge_method = code_data.code_challenge_method or CODE_CHALLENGE_PLAIN

                # Verify code challenge
                if challenge_method == CODE_CHALLENGE_S256:
                    # PKCE uses base64url encoding, not hex
                    import base64

                    verifier_hash = hashlib.sha256(code_verifier.encode()).digest()
                    verifier_challenge = base64.urlsafe_b64encode(verifier_hash).decode().rstrip("=")
                    if not secrets.compare_digest(verifier_challenge, code_data.code_challenge):
                        return None
                elif challenge_method == CODE_CHALLENGE_PLAIN:
                    if not secrets.compare_digest(code_verifier, code_data.code_challenge):
                        return None
                else:
                    return None

            # Code is valid, consume it (one-time use)
            await session.delete(f"{self.sandbox_id}:auth_code:{code}")

            return {
                "user_id": code_data.user_id,
                "client_id": code_data.client_id,
                "scope": code_data.scope,
            }

    # ============================================================================
    # MCP Access Tokens
    # ============================================================================

    async def create_access_token(
        self,
        user_id: str,
        client_id: str,
        scope: str | None = None,
    ) -> tuple[str, str]:
        """
        Create MCP access token and refresh token.

        Args:
            user_id: User identifier
            client_id: MCP client ID
            scope: Granted scopes

        Returns:
            Tuple of (access_token, refresh_token)
        """
        import logging

        logger = logging.getLogger(__name__)

        access_token = secrets.token_urlsafe(32)
        refresh_token = secrets.token_urlsafe(32)

        access_data = AccessTokenData(
            user_id=user_id,
            client_id=client_id,
            scope=scope,
        )

        refresh_data = RefreshTokenData(
            user_id=user_id,
            client_id=client_id,
            scope=scope,
            access_token=access_token,
        )

        async with get_session() as session:
            # Store access token with configurable TTL (default: 15 minutes)
            access_key = f"{self.sandbox_id}:access_token:{access_token}"
            await session.setex(
                access_key,
                self.access_token_ttl,
                access_data.to_json_bytes(),
            )
            logger.info(f"✓ Stored access token with key: {access_key} (TTL: {self.access_token_ttl}s)")

            # Store refresh token with configurable TTL (default: 1 day)
            refresh_key = f"{self.sandbox_id}:refresh_token:{refresh_token}"
            await session.setex(
                refresh_key,
                self.refresh_token_ttl,
                refresh_data.to_json_bytes(),
            )
            logger.debug(f"✓ Stored refresh token with key: {refresh_key} (TTL: {self.refresh_token_ttl}s)")

        return access_token, refresh_token

    async def validate_access_token(self, token: str) -> dict[str, Any] | None:
        """
        Validate MCP access token.

        Args:
            token: Access token

        Returns:
            Token info dict or None if invalid/expired
        """
        import logging

        logger = logging.getLogger(__name__)
        key = f"{self.sandbox_id}:access_token:{token}"
        logger.debug(f"🔍 Looking up token with key: {key}")

        async with get_session() as session:
            token_json = await session.get(key)
            if not token_json:
                logger.warning(f"❌ Token not found in session store (key: {key})")
                return None

            token_data = AccessTokenData.from_dict(orjson.loads(token_json))
            logger.debug(f"✓ Token found: user_id={token_data.user_id}")
            return token_data.to_dict()

    async def refresh_access_token(
        self,
        refresh_token: str,
        client_id: str | None = None,
    ) -> tuple[str, str] | None:
        """
        Refresh MCP access token using refresh token.

        Args:
            refresh_token: Refresh token
            client_id: Client presenting the token. When supplied it must match the
                client the refresh token was issued to (RFC 6749 section 6), so a
                leaked token cannot be redeemed by a different client. Optional only
                for backwards compatibility with existing callers.

        Returns:
            New (access_token, refresh_token) tuple or None if invalid
        """
        async with get_session() as session:
            refresh_json = await session.get(f"{self.sandbox_id}:refresh_token:{refresh_token}")
            if not refresh_json:
                return None

            refresh_data = RefreshTokenData.from_dict(orjson.loads(refresh_json))

            # Bind the refresh token to the client it was issued to
            if client_id is not None and refresh_data.client_id != client_id:
                return None

            # Revoke old access token
            old_access_token = refresh_data.access_token
            if old_access_token:
                await session.delete(f"{self.sandbox_id}:access_token:{old_access_token}")

            # Create new tokens
            new_access_token, new_refresh_token = await self.create_access_token(
                refresh_data.user_id,
                refresh_data.client_id,
                refresh_data.scope,
            )

            # Revoke old refresh token (token rotation)
            await session.delete(f"{self.sandbox_id}:refresh_token:{refresh_token}")

            return new_access_token, new_refresh_token

    # ============================================================================
    # External Provider Tokens (e.g., LinkedIn, GitHub, Google)
    # ============================================================================

    async def link_external_token(
        self,
        user_id: str,
        access_token: str,
        refresh_token: str | None = None,
        expires_in: int | None = None,
        provider: str = "external",
    ) -> None:
        """
        Link external OAuth provider tokens to MCP user.

        Args:
            user_id: MCP user identifier
            access_token: External provider access token
            refresh_token: External provider refresh token (if available)
            expires_in: Token lifetime in seconds (uses OAUTH_EXTERNAL_TOKEN_TTL if not provided)
            provider: Provider name for namespacing (e.g., 'linkedin', 'github')
        """
        # Use provided expires_in or fall back to configured default
        ttl = expires_in if expires_in is not None else self.external_token_ttl

        # Calculate expiration timestamp
        expires_at = (datetime.now() + timedelta(seconds=ttl)).isoformat()

        token_data = ExternalTokenData(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_at=expires_at,
            expires_in=ttl,
        )

        async with get_session() as session:
            await session.setex(
                f"{self.sandbox_id}:{provider}_token:{user_id}",
                ttl,
                token_data.to_json_bytes(),
            )

    async def get_external_token(self, user_id: str, provider: str = "external") -> dict[str, Any] | None:
        """
        Get external provider token for MCP user.

        Args:
            user_id: MCP user identifier
            provider: Provider name (e.g., 'linkedin', 'github')

        Returns:
            External token info or None
        """
        async with get_session() as session:
            token_json = await session.get(f"{self.sandbox_id}:{provider}_token:{user_id}")
            if not token_json:
                return None

            token_data = ExternalTokenData.from_dict(orjson.loads(token_json))
            return token_data.to_dict()

    async def update_external_token(
        self,
        user_id: str,
        access_token: str,
        refresh_token: str | None = None,
        expires_in: int | None = None,
        provider: str = "external",
    ) -> None:
        """Update external provider token (after refresh)."""
        await self.link_external_token(user_id, access_token, refresh_token, expires_in, provider)

    async def is_external_token_expired(self, user_id: str, provider: str = "external") -> bool:
        """
        Check if external provider token is expired.

        Checks the expires_at timestamp to determine if refresh is needed.
        We refresh proactively 5 minutes before actual expiration.

        Args:
            user_id: MCP user identifier
            provider: Provider name (e.g., 'linkedin', 'github')

        Returns:
            True if expired or expiring soon
        """
        async with get_session() as session:
            token_json = await session.get(f"{self.sandbox_id}:{provider}_token:{user_id}")
            if not token_json:
                return True

            token_data = ExternalTokenData.from_dict(orjson.loads(token_json))
            return token_data.is_expired(buffer_minutes=TOKEN_EXPIRY_BUFFER_MINUTES)

    # ============================================================================
    # Client Registration
    # ============================================================================

    async def register_client(
        self,
        client_name: str,
        redirect_uris: list[str],
        token_endpoint_auth_method: str = DEFAULT_AUTH_METHOD,
    ) -> dict[str, str]:
        """
        Register a new MCP client.

        Args:
            client_name: Client name
            redirect_uris: List of valid redirect URIs
            token_endpoint_auth_method: How the client will authenticate at the
                token endpoint (RFC 7591). Clients declaring client_secret_post or
                client_secret_basic must present their secret to exchange a code or
                refresh a token; "none" registers a public client.

        Returns:
            Dict with client_id, client_secret and token_endpoint_auth_method
        """
        client_id = secrets.token_urlsafe(16)
        client_secret = secrets.token_urlsafe(32)

        client_data = ClientData(
            client_name=client_name,
            client_secret=client_secret,
            redirect_uris=redirect_uris,
            token_endpoint_auth_method=token_endpoint_auth_method,
        )

        # Store with configurable TTL (default: 1 year)
        async with get_session() as session:
            await session.setex(
                f"{self.sandbox_id}:client:{client_id}",
                self.client_registration_ttl,
                client_data.to_json_bytes(),
            )

        return {
            "client_id": client_id,
            "client_secret": client_secret,
            "token_endpoint_auth_method": client_data.token_endpoint_auth_method,
        }

    async def get_client(self, client_id: str) -> ClientData | None:
        """
        Load a registered client.

        Args:
            client_id: Client ID

        Returns:
            Client data, or None if the client is not registered
        """
        async with get_session() as session:
            client_json = await session.get(f"{self.sandbox_id}:client:{client_id}")
            if not client_json:
                return None

            return ClientData.from_dict(orjson.loads(client_json))

    async def validate_client(
        self,
        client_id: str,
        client_secret: str | None = None,
        redirect_uri: str | None = None,
    ) -> bool:
        """
        Validate MCP client credentials.

        A supplied secret is always checked, but this method does not require one.
        Use :meth:`authenticate_client` at the token endpoint, where the client's
        registered token_endpoint_auth_method decides whether a secret is mandatory.

        Args:
            client_id: Client ID
            client_secret: Client secret (for confidential clients)
            redirect_uri: Redirect URI to validate

        Returns:
            True if valid
        """
        client_data = await self.get_client(client_id)
        if not client_data:
            return False

        # Validate secret if provided
        if client_secret is not None:
            if not secrets.compare_digest(client_data.client_secret, client_secret):
                return False

        # Validate redirect URI if provided
        if redirect_uri is not None:
            if redirect_uri not in client_data.redirect_uris:
                return False

        return True

    async def authenticate_client(
        self,
        client_id: str,
        client_secret: str | None = None,
        redirect_uri: str | None = None,
    ) -> bool:
        """
        Authenticate a client at the token endpoint (RFC 6749 section 3.2.1).

        Unlike :meth:`validate_client`, this enforces the client's registered
        token_endpoint_auth_method: a client that registered as client_secret_post
        or client_secret_basic must present a matching secret. Public clients
        ("none") may omit the secret, but a secret that is supplied and wrong is
        always rejected.

        Args:
            client_id: Client ID
            client_secret: Client secret presented at the token endpoint
            redirect_uri: Redirect URI to validate

        Returns:
            True if the client is authenticated
        """
        client_data = await self.get_client(client_id)
        if not client_data:
            return False

        requires_secret = client_data.token_endpoint_auth_method in CONFIDENTIAL_AUTH_METHODS
        if requires_secret and not client_secret:
            return False

        if client_secret is not None:
            if not secrets.compare_digest(client_data.client_secret, client_secret):
                return False

        if redirect_uri is not None:
            if redirect_uri not in client_data.redirect_uris:
                return False

        return True

    # ============================================================================
    # Pending Authorization Storage (for OAuth state management)
    # ============================================================================

    async def store_pending_authorization(
        self,
        state: str,
        data: dict[str, Any],
    ) -> None:
        """
        Store pending authorization data for OAuth state management.

        This is used to persist OAuth state across server restarts and
        multiple instances. The state parameter links the authorization
        request to the callback.

        Args:
            state: State parameter (used as key)
            data: Authorization data to store (client_id, redirect_uri, etc.)
        """
        async with get_session() as session:
            pending_data: bytes = orjson.dumps(data)
            await session.setex(
                f"{self.sandbox_id}:pending_auth:{state}",
                self.pending_auth_ttl,
                pending_data,
            )

    async def get_pending_authorization(
        self,
        state: str,
    ) -> dict[str, Any] | None:
        """
        Retrieve pending authorization data by state.

        Args:
            state: State parameter

        Returns:
            Authorization data dict or None if not found/expired
        """
        async with get_session() as session:
            data_json = await session.get(f"{self.sandbox_id}:pending_auth:{state}")
            if not data_json:
                return None
            result: dict[str, Any] = orjson.loads(data_json)
            return result

    async def delete_pending_authorization(
        self,
        state: str,
    ) -> bool:
        """
        Delete pending authorization (after successful callback).

        Args:
            state: State parameter

        Returns:
            True if deleted, False if not found
        """
        async with get_session() as session:
            key = f"{self.sandbox_id}:pending_auth:{state}"
            exists = await session.get(key)
            if exists:
                await session.delete(key)
                return True
            return False
