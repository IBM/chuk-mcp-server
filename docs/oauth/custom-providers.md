# Custom OAuth Providers

Create your own OAuth provider to integrate with any service that supports OAuth 2.0/2.1.

## Overview

Custom providers allow you to integrate with:
- LinkedIn
- GitHub
- Dropbox
- Slack
- Any OAuth 2.0/2.1 compliant service

## Provider Structure

A custom provider implements `BaseOAuthProvider` and handles:
1. Authorization requests from MCP clients
2. Redirects to external OAuth provider
3. Callback from external provider
4. Token exchange and storage
5. Token refresh and validation

## Basic Provider

```python
from chuk_mcp_server.oauth import (
    BaseOAuthProvider,
    AuthorizationParams,
    OAuthToken,
    OAuthClientInfo,
    TokenStore,
)
from typing import Any
import secrets

class MyServiceProvider(BaseOAuthProvider):
    """OAuth provider for MyService."""

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        redirect_uri: str,
        oauth_server_url: str = "http://localhost:8000",
        sandbox_id: str = "my-service",
    ):
        self.oauth_server_url = oauth_server_url
        self.token_store = TokenStore(sandbox_id=sandbox_id)

        # Your service's OAuth client
        self.service_client = MyServiceOAuthClient(
            client_id=client_id,
            client_secret=client_secret,
            redirect_uri=redirect_uri,
        )

        self._pending_authorizations = {}

    async def authorize(self, params: AuthorizationParams) -> dict[str, Any]:
        """Handle authorization request from MCP client."""
        # Validate MCP client
        if not await self.token_store.validate_client(
            params.client_id,
            redirect_uri=params.redirect_uri,
        ):
            from chuk_mcp_server.oauth.models import AuthorizeError
            raise AuthorizeError(
                error="invalid_client",
                error_description="Invalid client_id or redirect_uri",
            )

        # Generate state for OAuth flow
        state = secrets.token_urlsafe(32)

        self._pending_authorizations[state] = {
            "mcp_client_id": params.client_id,
            "mcp_redirect_uri": params.redirect_uri,
            "mcp_state": params.state,
            "mcp_scope": params.scope,
            "mcp_code_challenge": params.code_challenge,
            "mcp_code_challenge_method": params.code_challenge_method,
        }

        # Get authorization URL from your service
        auth_url = self.service_client.get_authorization_url(state=state)

        return {
            "authorization_url": auth_url,
            "state": state,
            "requires_external_authorization": True,
        }

    async def handle_external_callback(
        self, code: str, state: str
    ) -> dict[str, Any]:
        """Handle callback from external OAuth provider."""
        pending = self._pending_authorizations.get(state)
        if not pending:
            raise ValueError("Invalid or expired state parameter")

        # Exchange code for token at external service
        service_token = await self.service_client.exchange_code_for_token(code)

        # Get user info from external service
        user_info = await self.service_client.get_user_info(
            service_token["access_token"]
        )
        user_id = user_info["id"]

        # Store external token
        await self.token_store.link_external_token(
            user_id=user_id,
            access_token=service_token["access_token"],
            refresh_token=service_token.get("refresh_token"),
            expires_in=service_token.get("expires_in", 3600),
            provider="my-service",
        )

        # Create MCP authorization code
        mcp_code = await self.token_store.create_authorization_code(
            user_id=user_id,
            client_id=pending["mcp_client_id"],
            redirect_uri=pending["mcp_redirect_uri"],
            scope=pending["mcp_scope"],
            code_challenge=pending["mcp_code_challenge"],
            code_challenge_method=pending["mcp_code_challenge_method"],
        )

        # Clean up
        del self._pending_authorizations[state]

        return {
            "code": mcp_code,
            "state": pending["mcp_state"],
            "redirect_uri": pending["mcp_redirect_uri"],
        }

    async def exchange_authorization_code(
        self,
        code: str,
        client_id: str,
        redirect_uri: str,
        code_verifier: str | None = None,
        client_secret: str | None = None,
    ) -> OAuthToken:
        """Exchange authorization code for access token."""
        # Authenticate the client FIRST (RFC 6749 §3.2.1). authenticate_client
        # requires a secret from clients that registered as client_secret_post or
        # client_secret_basic, and rejects a wrong secret from any client.
        if not await self.token_store.authenticate_client(
            client_id,
            client_secret=client_secret,
        ):
            from chuk_mcp_server.oauth.models import TokenError
            raise TokenError(
                error="invalid_client",
                error_description="Client authentication failed",
            )

        # Validate code
        code_data = await self.token_store.validate_authorization_code(
            code=code,
            client_id=client_id,
            redirect_uri=redirect_uri,
            code_verifier=code_verifier,
        )

        if not code_data:
            from chuk_mcp_server.oauth.models import TokenError
            raise TokenError(
                error="invalid_grant",
                error_description="Invalid authorization code",
            )

        # Create tokens
        access_token, refresh_token = await self.token_store.create_access_token(
            user_id=code_data["user_id"],
            client_id=client_id,
            scope=code_data["scope"],
        )

        return OAuthToken(
            access_token=access_token,
            token_type="Bearer",
            expires_in=3600,
            refresh_token=refresh_token,
            scope=code_data["scope"],
        )

    async def exchange_refresh_token(
        self,
        refresh_token: str,
        client_id: str,
        scope: str | None = None,
        client_secret: str | None = None,
    ) -> OAuthToken:
        """Refresh an access token."""
        # Authenticate the client, then pass client_id to the store so a leaked
        # refresh token cannot be redeemed by a different client (RFC 6749 §6).
        if not await self.token_store.authenticate_client(
            client_id,
            client_secret=client_secret,
        ):
            from chuk_mcp_server.oauth.models import TokenError
            raise TokenError(
                error="invalid_client",
                error_description="Client authentication failed",
            )

        # Get refresh token data
        token_data = await self.token_store.get_refresh_token_data(
            refresh_token, client_id
        )

        if not token_data:
            from chuk_mcp_server.oauth.models import TokenError
            raise TokenError(error="invalid_grant")

        # Create new access token
        access_token, new_refresh = await self.token_store.create_access_token(
            user_id=token_data["user_id"],
            client_id=client_id,
            scope=scope or token_data["scope"],
        )

        return OAuthToken(
            access_token=access_token,
            token_type="Bearer",
            expires_in=3600,
            refresh_token=new_refresh,
        )

    async def validate_access_token(self, access_token: str) -> dict[str, Any]:
        """Validate an access token."""
        token_data = await self.token_store.validate_access_token(access_token)

        if not token_data:
            from chuk_mcp_server.oauth.models import TokenError
            raise TokenError(error="invalid_token")

        # Get external token
        external_token = await self.token_store.get_external_token(
            token_data["user_id"], "my-service"
        )

        # Refresh if expired
        if await self.token_store.is_external_token_expired(
            token_data["user_id"], "my-service"
        ):
            if external_token.get("refresh_token"):
                new_token = await self.service_client.refresh_access_token(
                    external_token["refresh_token"]
                )
                await self.token_store.update_external_token(
                    token_data["user_id"],
                    "my-service",
                    new_token["access_token"],
                    new_token.get("expires_in", 3600),
                )
                external_token = new_token

        return {
            **token_data,
            "external_access_token": external_token["access_token"],
        }

    async def register_client(
        self, client_metadata: dict[str, Any]
    ) -> OAuthClientInfo:
        """Register a new MCP client."""
        # Honour the client's RFC 7591 token_endpoint_auth_method declaration.
        # Dropping it would leave a confidential client's secret unenforceable.
        auth_method = client_metadata.get("token_endpoint_auth_method", "none")

        credentials = await self.token_store.register_client(
            client_name=client_metadata.get("client_name", "Unknown"),
            redirect_uris=client_metadata.get("redirect_uris", []),
            token_endpoint_auth_method=auth_method,
        )

        return OAuthClientInfo(
            client_id=credentials["client_id"],
            client_secret=credentials["client_secret"],
            client_name=client_metadata.get("client_name", "Unknown"),
            redirect_uris=client_metadata.get("redirect_uris", []),
            token_endpoint_auth_method=auth_method,
        )
```

### Client authentication requirements

Anything you build on `BaseOAuthProvider` is an OAuth Authorization Server, so it
must hold up its end of what the server's RFC 8414 metadata advertises:

- **Authenticate the client at the token endpoint before honouring a grant.** The
  middleware extracts `client_secret` from either the request body
  (`client_secret_post`) or an `Authorization: Basic` header
  (`client_secret_basic`) and passes it to your provider. Check it with
  `token_store.authenticate_client()`, which requires a secret from clients that
  registered as confidential and rejects a wrong secret from anyone.
- **Persist the client's declared `token_endpoint_auth_method`.** A secret you
  issue but never record a requirement for is a secret you can never enforce.
- **Bind refresh tokens to their client.** Pass `client_id` to
  `token_store.refresh_access_token()` so a leaked token is useless to anyone else.
- **Never let a PKCE challenge go unverified.** If a code carries a
  `code_challenge`, the token exchange must verify it — RFC 7636 §4.3 says an
  absent `code_challenge_method` means `plain`, not "skip the check".
- **Validate redirect URIs at registration.** Use
  `chuk_mcp_server.oauth.redirect_uri.is_safe_redirect_uri()` to reject
  `javascript:`, `data:` and relative URIs. Registration is open, and the
  callback page renders the registered URI as a link.
- **Report the real token lifetime.** `expires_in` must match the TTL your token
  store applies, or clients will cache tokens past their expiry.

Raise `TokenError(error="invalid_client", ...)` when authentication fails; the
middleware turns that into a `401` with a `WWW-Authenticate: Basic` header.

### `validate_redirect_uri`

Before the authorization endpoint redirects an *error* back to a client, it asks
your provider whether the redirect URI is actually registered — RFC 6749 §4.1.2.1
forbids automatically navigating to an unvalidated URI. The default
implementation on `BaseOAuthProvider` answers from `self.token_store`, so the
documented provider pattern needs no extra work. Override it if you store
clients elsewhere:

```python
    async def validate_redirect_uri(self, client_id: str, redirect_uri: str) -> bool:
        client = await my_client_registry.get(client_id)
        return bool(client and redirect_uri in client.redirect_uris)
```

Returning `False` is always safe: the error is rendered as a page instead of
being redirected. A provider that cannot answer **must** return `False`.

## Service OAuth Client

You also need a client for the external service:

```python
import httpx
from typing import Any

class MyServiceOAuthClient:
    """OAuth client for MyService API."""

    def __init__(self, client_id: str, client_secret: str, redirect_uri: str):
        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect_uri = redirect_uri

    def get_authorization_url(self, state: str) -> str:
        """Generate authorization URL."""
        from urllib.parse import urlencode

        params = {
            "client_id": self.client_id,
            "redirect_uri": self.redirect_uri,
            "response_type": "code",
            "state": state,
            "scope": "read write",  # Your service's scopes
        }

        return f"https://myservice.com/oauth/authorize?{urlencode(params)}"

    async def exchange_code_for_token(self, code: str) -> dict[str, Any]:
        """Exchange authorization code for access token."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://myservice.com/oauth/token",
                data={
                    "grant_type": "authorization_code",
                    "code": code,
                    "client_id": self.client_id,
                    "client_secret": self.client_secret,
                    "redirect_uri": self.redirect_uri,
                },
            )
            response.raise_for_status()
            result: dict[str, Any] = response.json()
            return result

    async def refresh_access_token(self, refresh_token: str) -> dict[str, Any]:
        """Refresh an access token."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://myservice.com/oauth/token",
                data={
                    "grant_type": "refresh_token",
                    "refresh_token": refresh_token,
                    "client_id": self.client_id,
                    "client_secret": self.client_secret,
                },
            )
            response.raise_for_status()
            result: dict[str, Any] = response.json()
            return result

    async def get_user_info(self, access_token: str) -> dict[str, Any]:
        """Get user information."""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://myservice.com/api/v1/user",
                headers={"Authorization": f"Bearer {access_token}"},
            )
            response.raise_for_status()
            result: dict[str, Any] = response.json()
            return result
```

## Using Your Provider

```python
from chuk_mcp_server import ChukMCPServer
from chuk_mcp_server.oauth import OAuthMiddleware
from my_provider import MyServiceProvider

mcp = ChukMCPServer("my-app")

def setup_oauth():
    provider = MyServiceProvider(
        client_id="your-client-id",
        client_secret="your-client-secret",
        redirect_uri="http://localhost:8000/oauth/callback",
        oauth_server_url="http://localhost:8000",
    )

    return OAuthMiddleware(
        mcp_server=mcp,
        provider=provider,
        oauth_server_url="http://localhost:8000",
        scopes_supported=["read", "write"],
        provider_name="My Service",
    )

mcp.run(transport="http", port=8000, post_register_hook=setup_oauth)
```

## Real-World Examples

### LinkedIn Provider

See [chuk-mcp-linkedin](https://github.com/chrishayuk/chuk-mcp-linkedin) for a complete implementation.

### Key Features
- Token refresh handling
- User profile access
- Post publishing
- Connection management

## Next Steps

- [Protected Tools](protected-tools.md) - Use your provider with tools
- [Token Management](tokens.md) - Understand token lifecycle
- [Google Drive Provider](google-drive.md) - Reference implementation
