"""
Base token store interface for OAuth tokens.

Defines the interface that all token store implementations must follow.
"""

from abc import ABC, abstractmethod
from typing import Any

from .constants import DEFAULT_AUTH_METHOD


class BaseTokenStore(ABC):
    """
    Abstract base class for token storage implementations.

    All methods are async to support both file-based and Redis-based storage.
    """

    # ============================================================================
    # MCP Authorization Codes
    # ============================================================================

    @abstractmethod
    async def create_authorization_code(
        self,
        user_id: str,
        client_id: str,
        redirect_uri: str,
        scope: str | None = None,
        code_challenge: str | None = None,
        code_challenge_method: str | None = None,
    ) -> str:
        """Create authorization code for MCP client."""
        pass

    @abstractmethod
    async def validate_authorization_code(
        self,
        code: str,
        client_id: str,
        redirect_uri: str,
        code_verifier: str | None = None,
    ) -> dict[str, Any] | None:
        """Validate authorization code and return user info."""
        pass

    # ============================================================================
    # MCP Access Tokens
    # ============================================================================

    @abstractmethod
    async def create_access_token(
        self,
        user_id: str,
        client_id: str,
        scope: str | None = None,
    ) -> tuple[str, str]:
        """Create MCP access token and refresh token."""
        pass

    @abstractmethod
    async def validate_access_token(self, token: str) -> dict[str, Any] | None:
        """Validate MCP access token."""
        pass

    @abstractmethod
    async def refresh_access_token(
        self,
        refresh_token: str,
        client_id: str | None = None,
    ) -> tuple[str, str] | None:
        """
        Refresh MCP access token using refresh token.

        When ``client_id`` is supplied it must match the client the refresh token
        was issued to (RFC 6749 section 6). It is optional so that token stores
        written against an earlier release keep working.
        """
        pass

    # ============================================================================
    # External Provider Tokens (e.g., LinkedIn, GitHub, Google)
    # ============================================================================

    @abstractmethod
    async def link_external_token(
        self,
        user_id: str,
        access_token: str,
        refresh_token: str | None = None,
        expires_in: int | None = None,
        provider: str = "external",
    ) -> None:
        """Link external OAuth provider tokens to MCP user."""
        pass

    @abstractmethod
    async def get_external_token(self, user_id: str, provider: str = "external") -> dict[str, Any] | None:
        """Get external provider token for MCP user."""
        pass

    @abstractmethod
    async def update_external_token(
        self,
        user_id: str,
        access_token: str,
        refresh_token: str | None = None,
        expires_in: int | None = None,
        provider: str = "external",
    ) -> None:
        """Update external provider token (after refresh)."""
        pass

    @abstractmethod
    async def is_external_token_expired(self, user_id: str, provider: str = "external") -> bool:
        """Check if external provider token is expired."""
        pass

    # ============================================================================
    # Client Registration
    # ============================================================================

    @abstractmethod
    async def register_client(
        self,
        client_name: str,
        redirect_uris: list[str],
        token_endpoint_auth_method: str = DEFAULT_AUTH_METHOD,
    ) -> dict[str, str]:
        """
        Register a new MCP client.

        ``token_endpoint_auth_method`` is the RFC 7591 declaration of how the
        client will authenticate at the token endpoint. It has a default so that
        token stores written against an earlier release keep working.
        """
        pass

    @abstractmethod
    async def validate_client(
        self,
        client_id: str,
        client_secret: str | None = None,
        redirect_uri: str | None = None,
    ) -> bool:
        """Validate MCP client credentials, checking the secret only if supplied."""
        pass

    async def authenticate_client(
        self,
        client_id: str,
        client_secret: str | None = None,
        redirect_uri: str | None = None,
    ) -> bool:
        """
        Authenticate a client at the token endpoint (RFC 6749 section 3.2.1).

        Implementations that persist a client's registered
        token_endpoint_auth_method should override this to require a secret from
        confidential clients. The default delegates to :meth:`validate_client`,
        which preserves the behaviour of stores that do not track the method.
        """
        return await self.validate_client(
            client_id,
            client_secret=client_secret,
            redirect_uri=redirect_uri,
        )
