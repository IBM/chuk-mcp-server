"""
Base OAuth Authorization Server Provider for chuk-mcp-server.

Pure Python implementation without mcp library dependencies.
"""

from abc import ABC, abstractmethod
from typing import Any

from .models import (
    AuthorizationParams,
    OAuthClientInfo,
    OAuthToken,
)


class BaseOAuthProvider(ABC):
    """
    Base class for OAuth Authorization Server implementations.

    This is a pure chuk-mcp-server implementation that doesn't depend
    on the mcp library's auth modules.
    """

    @abstractmethod
    async def authorize(
        self,
        params: AuthorizationParams,
    ) -> dict[str, Any]:
        """
        Handle authorization request from OAuth client.

        Args:
            params: Authorization parameters from client

        Returns:
            Dict with authorization_code or redirect information

        Raises:
            AuthorizeError: If authorization fails
        """
        pass

    @abstractmethod
    async def exchange_authorization_code(
        self,
        code: str,
        client_id: str,
        redirect_uri: str,
        code_verifier: str | None = None,
        client_secret: str | None = None,
    ) -> OAuthToken:
        """
        Exchange authorization code for access token.

        Implementations MUST authenticate the client when it registered with a
        confidential token_endpoint_auth_method (client_secret_post or
        client_secret_basic) — see RFC 6749 section 3.2.1. Public clients
        register as "none" and may exchange without a secret.

        Args:
            code: Authorization code
            client_id: Client ID
            redirect_uri: Redirect URI (must match)
            code_verifier: PKCE code verifier
            client_secret: Client secret presented at the token endpoint, from
                either the request body or an Authorization: Basic header

        Returns:
            OAuth token

        Raises:
            TokenError: If exchange fails
        """
        pass

    @abstractmethod
    async def exchange_refresh_token(
        self,
        refresh_token: str,
        client_id: str,
        scope: str | None = None,
        client_secret: str | None = None,
    ) -> OAuthToken:
        """
        Refresh access token using refresh token.

        Implementations MUST authenticate confidential clients (RFC 6749
        section 6) and MUST reject a refresh token that was issued to a
        different client than the one presenting it.

        Args:
            refresh_token: Refresh token
            client_id: Client ID
            scope: Optional scope (must be subset of original)
            client_secret: Client secret presented at the token endpoint

        Returns:
            New OAuth token

        Raises:
            TokenError: If refresh fails
        """
        pass

    @abstractmethod
    async def validate_access_token(
        self,
        token: str,
    ) -> dict[str, Any]:
        """
        Validate and load access token.

        Args:
            token: Access token

        Returns:
            Token data

        Raises:
            TokenError: If token is invalid
        """
        pass

    @abstractmethod
    async def register_client(
        self,
        client_metadata: dict[str, Any],
    ) -> OAuthClientInfo:
        """
        Register a new OAuth client.

        Args:
            client_metadata: Client registration metadata

        Returns:
            Client information with credentials

        Raises:
            RegistrationError: If registration fails
        """
        pass

    async def revoke_token(
        self,
        token: str,
        token_type_hint: str | None = None,
    ) -> None:
        """
        Revoke a token.

        Args:
            token: Token to revoke
            token_type_hint: Type of token (access_token or refresh_token)
        """
        # Optional: Override to implement token revocation
        return None
