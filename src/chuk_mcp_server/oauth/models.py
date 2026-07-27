"""
OAuth data models for chuk-mcp-server.

Pure Python OAuth models without mcp library dependencies.
"""

from dataclasses import dataclass
from typing import Literal

from .constants import DEFAULT_AUTH_METHOD


@dataclass
class AuthorizationParams:
    """Parameters for OAuth authorization request"""

    response_type: str  # "code"
    client_id: str
    redirect_uri: str
    scope: str | None = None
    state: str | None = None
    code_challenge: str | None = None
    code_challenge_method: Literal["S256", "plain"] | None = None


@dataclass
class OAuthToken:
    """OAuth token response"""

    access_token: str
    token_type: str = "Bearer"
    expires_in: int = 3600
    refresh_token: str | None = None
    scope: str | None = None


@dataclass
class OAuthClientInfo:
    """OAuth client registration information"""

    client_id: str
    client_secret: str
    client_name: str
    redirect_uris: list[str]
    token_endpoint_auth_method: str = DEFAULT_AUTH_METHOD


class OAuthError(Exception):
    """Base class for OAuth errors"""

    def __init__(self, error: str, error_description: str | None = None):
        self.error = error
        self.error_description = error_description
        super().__init__(f"{error}: {error_description}" if error_description else error)


class AuthorizeError(OAuthError):
    """Error during authorization"""

    pass


class TokenError(OAuthError):
    """Error during token exchange"""

    pass


class RegistrationError(OAuthError):
    """Error during client registration"""

    pass
