"""
Generic OAuth 2.1 implementation for ChukMCPServer.

Provides a complete OAuth 2.1 authorization server with:
- PKCE support (RFC 7636)
- Dynamic client registration (RFC 7591)
- Session-based token storage via chuk-sessions
- Pluggable OAuth provider support

Usage:
    from chuk_mcp_server.oauth import OAuthMiddleware, TokenStore
    from your_app.oauth import YourOAuthProvider

    # Create your provider
    provider = YourOAuthProvider(
        client_id="your_client_id",
        client_secret="your_client_secret",
        redirect_uri="http://localhost:8000/oauth/callback",
    )

    # Add OAuth to your MCP server
    oauth = OAuthMiddleware(
        mcp_server=mcp,
        provider=provider,
        oauth_server_url="http://localhost:8000",
        scopes_supported=["your.scope"],
        provider_name="Your Service",
    )
"""

from .base_provider import BaseOAuthProvider
from .base_token_store import BaseTokenStore
from .compat import supports_keyword
from .crypto import secure_compare
from .helpers import configure_storage_from_oauth, setup_google_drive_oauth
from .middleware import OAuthMiddleware
from .models import (
    AuthorizationParams,
    AuthorizeError,
    OAuthClientInfo,
    OAuthToken,
    RegistrationError,
    TokenError,
)
from .redirect_uri import build_redirect_url, is_safe_redirect_uri
from .token_store import TokenStore

__all__ = [
    "OAuthMiddleware",
    "AuthorizationParams",
    "OAuthToken",
    "OAuthClientInfo",
    "AuthorizeError",
    "TokenError",
    "RegistrationError",
    "TokenStore",
    "BaseOAuthProvider",
    "BaseTokenStore",
    "setup_google_drive_oauth",
    "configure_storage_from_oauth",
    # Helpers for provider authors
    "supports_keyword",
    "secure_compare",
    "is_safe_redirect_uri",
    "build_redirect_url",
]
