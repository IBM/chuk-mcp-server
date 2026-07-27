"""
Generic OAuth middleware for ChukMCPServer.

Adds OAuth endpoints to any MCP server:
- OAuth Authorization Server discovery (/.well-known/oauth-authorization-server) - RFC 8414
- OAuth Protected Resource metadata (/.well-known/oauth-protected-resource) - RFC 9728
- Authorization endpoint (/oauth/authorize)
- Token endpoint (/oauth/token)
- External provider callback (/oauth/callback)
- Client registration (/oauth/register)

Works with any OAuth provider that implements BaseOAuthProvider.
"""

import html
import logging
from typing import Any, Literal, cast
from urllib.parse import urlencode

from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, RedirectResponse

from .base_provider import BaseOAuthProvider
from .client_auth import CredentialError, extract_client_credentials
from .compat import supports_keyword
from .constants import (
    AUTH_METHOD_CLIENT_SECRET_BASIC,
    AUTH_METHOD_CLIENT_SECRET_POST,
    AUTH_METHOD_NONE,
    BASIC_AUTH_REALM,
    CODE_CHALLENGE_PLAIN,
    CODE_CHALLENGE_S256,
    DEFAULT_AUTH_METHOD,
    ERROR_INVALID_CLIENT,
    ERROR_INVALID_CLIENT_METADATA,
    ERROR_INVALID_REQUEST,
    ERROR_SERVER_ERROR,
    ERROR_UNSUPPORTED_GRANT_TYPE,
    GRANT_AUTHORIZATION_CODE,
    GRANT_REFRESH_TOKEN,
    HEADER_AUTHORIZATION,
    HEADER_WWW_AUTHENTICATE,
    HTTP_BAD_REQUEST,
    HTTP_CREATED,
    HTTP_INTERNAL_SERVER_ERROR,
    HTTP_UNAUTHORIZED,
    PARAM_ACCESS_TOKEN,
    PARAM_CLIENT_ID,
    PARAM_CLIENT_NAME,
    PARAM_CLIENT_SECRET,
    PARAM_CODE,
    PARAM_CODE_CHALLENGE,
    PARAM_CODE_CHALLENGE_METHOD,
    PARAM_CODE_VERIFIER,
    PARAM_ERROR,
    PARAM_ERROR_DESCRIPTION,
    PARAM_EXPIRES_IN,
    PARAM_GRANT_TYPE,
    PARAM_REDIRECT_URI,
    PARAM_REDIRECT_URIS,
    PARAM_REFRESH_TOKEN,
    PARAM_RESPONSE_TYPE,
    PARAM_SCOPE,
    PARAM_STATE,
    PARAM_TOKEN_ENDPOINT_AUTH_METHOD,
    PARAM_TOKEN_TYPE,
    PATH_AUTHORIZATION_SERVER_METADATA,
    PATH_AUTHORIZE,
    PATH_PROTECTED_RESOURCE,
    PATH_REGISTER,
    PATH_TOKEN,
    RESPONSE_TYPE_CODE,
)
from .models import AuthorizationParams, RegistrationError, TokenError

logger = logging.getLogger(__name__)


class OAuthMiddleware:
    """
    Generic OAuth middleware for ChukMCPServer.

    Implements the MCP OAuth specification with pluggable provider support.
    """

    def __init__(
        self,
        mcp_server: Any,  # ChukMCPServer instance
        provider: BaseOAuthProvider,
        oauth_server_url: str = "http://localhost:8000",
        callback_path: str = "/oauth/callback",
        scopes_supported: list[str] | None = None,
        service_documentation: str | None = None,
        provider_name: str = "OAuth Provider",
    ):
        """
        Initialize generic OAuth middleware.

        Args:
            mcp_server: ChukMCPServer instance
            provider: OAuth provider instance (must implement BaseOAuthProvider)
            oauth_server_url: This OAuth server's base URL
            callback_path: Path for external provider callback (default: /oauth/callback)
            scopes_supported: List of supported scopes for metadata
            service_documentation: URL to service documentation
            provider_name: Human-readable name of the OAuth provider (e.g., "LinkedIn", "GitHub")
        """
        self.mcp = mcp_server
        self.provider = provider
        self.oauth_server_url = oauth_server_url
        self.callback_path = callback_path
        self.scopes_supported = scopes_supported or []
        self.service_documentation = service_documentation
        self.provider_name = provider_name

        # Register OAuth endpoints
        self._register_endpoints()

    def _register_endpoints(self) -> None:
        """Register OAuth endpoints with the MCP server."""

        # OAuth server metadata (RFC 8414)
        @self.mcp.endpoint(PATH_AUTHORIZATION_SERVER_METADATA, methods=["GET"])  # type: ignore[untyped-decorator]
        async def oauth_metadata(request: Request) -> JSONResponse:
            """OAuth Authorization Server Metadata endpoint."""
            return await self._metadata_endpoint(request)

        # Protected Resource Metadata (RFC 9728)
        @self.mcp.endpoint(PATH_PROTECTED_RESOURCE, methods=["GET"])  # type: ignore[untyped-decorator]
        async def protected_resource_metadata(request: Request) -> JSONResponse:
            """OAuth Protected Resource Metadata endpoint."""
            return await self._protected_resource_endpoint(request)

        # OAuth authorize endpoint
        @self.mcp.endpoint(PATH_AUTHORIZE, methods=["GET"])  # type: ignore[untyped-decorator]
        async def oauth_authorize(request: Request) -> Any:
            """OAuth authorization endpoint."""
            return await self._authorize_endpoint(request)

        # OAuth token endpoint
        @self.mcp.endpoint(PATH_TOKEN, methods=["POST"])  # type: ignore[untyped-decorator]
        async def oauth_token(request: Request) -> JSONResponse:
            """OAuth token endpoint."""
            return await self._token_endpoint(request)

        # Client registration endpoint (RFC 7591)
        @self.mcp.endpoint(PATH_REGISTER, methods=["POST"])  # type: ignore[untyped-decorator]
        async def oauth_register(request: Request) -> JSONResponse:
            """Dynamic client registration endpoint."""
            return await self._register_endpoint(request)

        # External OAuth provider callback
        @self.mcp.endpoint(self.callback_path, methods=["GET"])  # type: ignore[untyped-decorator]
        async def external_callback(request: Request) -> Any:
            """External OAuth provider callback endpoint."""
            return await self._external_callback_endpoint(request)

    # ============================================================================
    # OAuth Endpoints
    # ============================================================================

    async def _metadata_endpoint(self, request: Request) -> JSONResponse:
        """
        OAuth Authorization Server Metadata endpoint (RFC 8414).

        Returns server metadata for OAuth discovery.
        """
        metadata = {
            "issuer": self.oauth_server_url,
            "authorization_endpoint": f"{self.oauth_server_url}{PATH_AUTHORIZE}",
            "token_endpoint": f"{self.oauth_server_url}{PATH_TOKEN}",
            "registration_endpoint": f"{self.oauth_server_url}{PATH_REGISTER}",
            "grant_types_supported": [GRANT_AUTHORIZATION_CODE, GRANT_REFRESH_TOKEN],
            "response_types_supported": [RESPONSE_TYPE_CODE],
            "token_endpoint_auth_methods_supported": [
                AUTH_METHOD_CLIENT_SECRET_POST,
                AUTH_METHOD_CLIENT_SECRET_BASIC,
                AUTH_METHOD_NONE,
            ],
            "code_challenge_methods_supported": [CODE_CHALLENGE_S256, CODE_CHALLENGE_PLAIN],
        }

        # Add optional metadata
        if self.scopes_supported:
            metadata["scopes_supported"] = self.scopes_supported

        if self.service_documentation:
            metadata["service_documentation"] = self.service_documentation

        return JSONResponse(metadata)

    async def _protected_resource_endpoint(self, request: Request) -> JSONResponse:
        """
        OAuth Protected Resource Metadata endpoint (RFC 9728).

        Returns information about this protected resource, including
        which authorization servers can issue tokens for it.
        """
        metadata = {
            "resource": self.oauth_server_url,
            "authorization_servers": [self.oauth_server_url],
            "bearer_methods_supported": ["header"],
            "resource_signing_alg_values_supported": ["RS256"],
        }

        # Add optional metadata
        if self.scopes_supported:
            metadata["scopes_supported"] = self.scopes_supported

        if self.service_documentation:
            metadata["resource_documentation"] = self.service_documentation

        return JSONResponse(metadata, headers={"Access-Control-Allow-Origin": "*"})

    @staticmethod
    def _resolve_code_challenge_method(
        code_challenge: str | None,
        requested_method: str | None,
    ) -> Literal["S256", "plain"] | None:
        """
        Decide which PKCE method to record against an authorization code.

        RFC 7636 section 4.3: when a challenge is present but no method is named,
        the method is "plain". Recording None instead would leave the challenge
        unverifiable at the token endpoint, so a challenge always gets a method.

        Raises:
            ValueError: If an unrecognised method is requested
        """
        if requested_method in (CODE_CHALLENGE_S256, CODE_CHALLENGE_PLAIN):
            return cast(Literal["S256", "plain"], requested_method)

        if requested_method:
            raise ValueError(f"Unsupported code_challenge_method: {requested_method}")

        if not code_challenge:
            return None

        return cast(Literal["S256", "plain"], CODE_CHALLENGE_PLAIN)

    async def _authorize_endpoint(self, request: Request) -> Any:
        """
        OAuth authorization endpoint.

        Handles authorization requests from MCP clients.
        May redirect to external provider for authentication.
        """
        params: dict[str, str] = {}
        try:
            # Parse authorization parameters
            params = dict(request.query_params)

            code_challenge = params.get(PARAM_CODE_CHALLENGE)
            code_challenge_method = self._resolve_code_challenge_method(
                code_challenge,
                params.get(PARAM_CODE_CHALLENGE_METHOD),
            )

            auth_params = AuthorizationParams(
                response_type=params.get(PARAM_RESPONSE_TYPE, RESPONSE_TYPE_CODE),
                client_id=params[PARAM_CLIENT_ID],
                redirect_uri=params[PARAM_REDIRECT_URI],
                scope=params.get(PARAM_SCOPE),
                state=params.get(PARAM_STATE),
                code_challenge=code_challenge,
                code_challenge_method=code_challenge_method,
            )

            # Process authorization request
            result = await self.provider.authorize(auth_params)

            # Check if we need external provider authorization
            if result.get("requires_external_authorization"):
                # Redirect to external provider
                return RedirectResponse(result["authorization_url"])

            # We have authorization code, redirect back to client
            redirect_params = {
                PARAM_CODE: result["code"],
            }
            if result.get(PARAM_STATE):
                redirect_params[PARAM_STATE] = result[PARAM_STATE]

            redirect_url = f"{auth_params.redirect_uri}?{urlencode(redirect_params)}"
            return RedirectResponse(redirect_url)

        except Exception:
            # Attempt to redirect with error, but only if redirect_uri is available
            try:
                redirect_uri = params.get(PARAM_REDIRECT_URI)
                if redirect_uri:
                    error_params = {
                        PARAM_ERROR: ERROR_SERVER_ERROR,
                        PARAM_ERROR_DESCRIPTION: "Authorization failed",
                    }
                    if params.get(PARAM_STATE):
                        error_params[PARAM_STATE] = params[PARAM_STATE]

                    error_url = f"{redirect_uri}?{urlencode(error_params)}"
                    return RedirectResponse(error_url)
            except Exception:
                logger.debug("Failed to redirect with error response", exc_info=True)

            # Fallback: render error as HTML page (no redirect)
            return HTMLResponse(
                """
                <html>
                    <head><title>Authorization Error</title></head>
                    <body>
                        <h1>Authorization Error</h1>
                        <p>An unexpected error occurred during authorization.</p>
                    </body>
                </html>
                """,
                status_code=HTTP_BAD_REQUEST,
            )

    async def _token_endpoint(self, request: Request) -> JSONResponse:
        """
        OAuth token endpoint.

        Exchanges authorization code for access token.
        Also handles refresh token requests.
        """
        try:
            # Parse form data
            form_data = await request.form()

            # Helper to extract string values from form
            def get_form_str(key: str) -> str | None:
                val = form_data.get(key)
                return str(val) if val is not None else None

            grant_type = get_form_str(PARAM_GRANT_TYPE)

            logger.debug(f"🔐 Token exchange request - grant_type: {grant_type}")
            logger.debug(f"🔐 Token exchange - client_id: {get_form_str(PARAM_CLIENT_ID)}")
            code_val = get_form_str(PARAM_CODE)
            logger.debug(f"🔐 Token exchange - code: {code_val[:20]}..." if code_val else "no code")
            logger.debug(f"🔐 Token exchange - redirect_uri: {get_form_str(PARAM_REDIRECT_URI)}")
            logger.debug(f"🔐 Token exchange - code_verifier present: {bool(get_form_str(PARAM_CODE_VERIFIER))}")

            # Client credentials may arrive in the body (client_secret_post) or in
            # an Authorization: Basic header (client_secret_basic) — RFC 6749 2.3.1.
            credentials = extract_client_credentials(
                request.headers.get(HEADER_AUTHORIZATION),
                get_form_str(PARAM_CLIENT_ID),
                get_form_str(PARAM_CLIENT_SECRET),
            )
            if not credentials.is_valid:
                return self._credential_error_response(credentials.error)

            client_id = credentials.client_id
            client_secret = credentials.client_secret

            if grant_type == GRANT_AUTHORIZATION_CODE:
                # Exchange authorization code for token
                code = get_form_str(PARAM_CODE)
                redirect_uri = get_form_str(PARAM_REDIRECT_URI)
                code_verifier = get_form_str(PARAM_CODE_VERIFIER)

                if not code or not client_id or not redirect_uri:
                    return JSONResponse(
                        {
                            PARAM_ERROR: ERROR_INVALID_REQUEST,
                            PARAM_ERROR_DESCRIPTION: "Missing required parameters",
                        },
                        status_code=HTTP_BAD_REQUEST,
                    )

                token = await self._call_provider(
                    self.provider.exchange_authorization_code,
                    client_secret,
                    code=code,
                    client_id=client_id,
                    redirect_uri=redirect_uri,
                    code_verifier=code_verifier,
                )
            elif grant_type == GRANT_REFRESH_TOKEN:
                # Refresh access token
                refresh_token = get_form_str(PARAM_REFRESH_TOKEN)
                scope = get_form_str(PARAM_SCOPE)

                if not refresh_token or not client_id:
                    return JSONResponse(
                        {
                            PARAM_ERROR: ERROR_INVALID_REQUEST,
                            PARAM_ERROR_DESCRIPTION: "Missing required parameters",
                        },
                        status_code=HTTP_BAD_REQUEST,
                    )

                token = await self._call_provider(
                    self.provider.exchange_refresh_token,
                    client_secret,
                    refresh_token=refresh_token,
                    client_id=client_id,
                    scope=scope,
                )
            else:
                return JSONResponse(
                    {
                        PARAM_ERROR: ERROR_UNSUPPORTED_GRANT_TYPE,
                        PARAM_ERROR_DESCRIPTION: f"Grant type {grant_type} not supported",
                    },
                    status_code=HTTP_BAD_REQUEST,
                )

            # Return token response
            return JSONResponse(
                {
                    PARAM_ACCESS_TOKEN: str(token.access_token),
                    PARAM_TOKEN_TYPE: token.token_type,
                    PARAM_EXPIRES_IN: token.expires_in,
                    PARAM_REFRESH_TOKEN: str(token.refresh_token) if token.refresh_token else None,
                    PARAM_SCOPE: token.scope,
                }
            )

        except TokenError as e:
            logger.warning(f"Token exchange rejected: {e.error}")
            if e.error == ERROR_INVALID_CLIENT:
                return self._invalid_client_response("Client authentication failed")
            return JSONResponse(
                {
                    PARAM_ERROR: e.error,
                    PARAM_ERROR_DESCRIPTION: "Token exchange failed",
                },
                status_code=HTTP_BAD_REQUEST,
            )

        except Exception as e:
            logger.error(f"Token exchange failed: {type(e).__name__}", exc_info=True)
            return JSONResponse(
                {
                    PARAM_ERROR: ERROR_INVALID_REQUEST,
                    PARAM_ERROR_DESCRIPTION: "Token exchange failed",
                },
                status_code=HTTP_BAD_REQUEST,
            )

    @staticmethod
    def _invalid_client_response(description: str) -> JSONResponse:
        """Build an RFC 6749 section 5.2 invalid_client response."""
        return JSONResponse(
            {PARAM_ERROR: ERROR_INVALID_CLIENT, PARAM_ERROR_DESCRIPTION: description},
            status_code=HTTP_UNAUTHORIZED,
            headers={HEADER_WWW_AUTHENTICATE: BASIC_AUTH_REALM},
        )

    @classmethod
    def _credential_error_response(cls, error: CredentialError | None) -> JSONResponse:
        """Map a credential extraction failure onto an OAuth error response."""
        if error is CredentialError.CLIENT_ID_MISMATCH:
            # Not an authentication failure — the request contradicts itself.
            return JSONResponse(
                {
                    PARAM_ERROR: ERROR_INVALID_REQUEST,
                    PARAM_ERROR_DESCRIPTION: "client_id mismatch between Authorization header and request body",
                },
                status_code=HTTP_BAD_REQUEST,
            )

        descriptions = {
            CredentialError.MALFORMED_HEADER: "Malformed Authorization header",
            CredentialError.CLIENT_SECRET_MISMATCH: "Conflicting client credentials",
        }
        description = descriptions.get(error) if error else None
        return cls._invalid_client_response(description or "Client authentication failed")

    @staticmethod
    async def _call_provider(
        method: Any,
        client_secret: str | None,
        **kwargs: Any,
    ) -> Any:
        """
        Invoke a provider token method, passing client_secret when it is accepted.

        Providers written against an earlier release do not declare client_secret.
        Those cannot authenticate their clients, so a secret sent to one is refused
        rather than silently ignored.
        """
        if supports_keyword(method, PARAM_CLIENT_SECRET):
            return await method(client_secret=client_secret, **kwargs)

        if client_secret is not None:
            logger.error(
                "Provider %s cannot verify client_secret; rejecting the request rather "
                "than ignoring the credential. Add a client_secret parameter to it.",
                getattr(method, "__qualname__", method),
            )
            raise TokenError(
                error=ERROR_INVALID_CLIENT,
                error_description="Client authentication is not supported by this provider",
            )

        logger.warning(
            "Provider %s does not accept client_secret; confidential clients cannot be "
            "authenticated at the token endpoint.",
            getattr(method, "__qualname__", method),
        )
        return await method(**kwargs)

    async def _register_endpoint(self, request: Request) -> JSONResponse:
        """
        Dynamic client registration endpoint (RFC 7591).

        Allows MCP clients to register dynamically.
        """
        try:
            # Parse registration request
            body = await request.json()

            # Register client
            client_info = await self.provider.register_client(body)

            return JSONResponse(
                {
                    PARAM_CLIENT_ID: client_info.client_id,
                    PARAM_CLIENT_SECRET: client_info.client_secret,
                    PARAM_CLIENT_NAME: client_info.client_name,
                    PARAM_REDIRECT_URIS: client_info.redirect_uris,
                    # Echo back what was actually recorded, so a client can tell
                    # whether its requested method was honoured (RFC 7591 3.2.1).
                    PARAM_TOKEN_ENDPOINT_AUTH_METHOD: getattr(
                        client_info, PARAM_TOKEN_ENDPOINT_AUTH_METHOD, DEFAULT_AUTH_METHOD
                    ),
                },
                status_code=HTTP_CREATED,
            )

        except RegistrationError as e:
            logger.warning(f"Client registration rejected: {e.error}")
            return JSONResponse(
                {
                    PARAM_ERROR: e.error,
                    PARAM_ERROR_DESCRIPTION: e.error_description or "Client registration failed",
                },
                status_code=HTTP_BAD_REQUEST,
            )

        except Exception as e:
            logger.error(f"Client registration failed: {type(e).__name__}", exc_info=True)
            return JSONResponse(
                {
                    PARAM_ERROR: ERROR_INVALID_CLIENT_METADATA,
                    PARAM_ERROR_DESCRIPTION: "Client registration failed",
                },
                status_code=HTTP_BAD_REQUEST,
            )

    async def _external_callback_endpoint(self, request: Request) -> Any:
        """
        External OAuth provider callback endpoint.

        Handles the callback from external provider after user authorization.
        Completes the external OAuth flow and redirects back to MCP client.
        """
        try:
            # Get authorization code and state
            code = request.query_params.get("code")
            state = request.query_params.get("state")
            error = request.query_params.get(PARAM_ERROR)

            if error:
                # External authorization failed
                return HTMLResponse(
                    f"""
                    <html>
                        <head><title>Authorization Failed</title></head>
                        <body>
                            <h1>{html.escape(self.provider_name)} Authorization Failed</h1>
                            <p>Error: {html.escape(error)}</p>
                            <p>Description: {html.escape(request.query_params.get("error_description", "Unknown error"))}</p>
                        </body>
                    </html>
                    """,
                    status_code=HTTP_BAD_REQUEST,
                )

            if not code or not state:
                return HTMLResponse(
                    """
                    <html>
                        <head><title>Invalid Request</title></head>
                        <body>
                            <h1>Invalid OAuth Callback</h1>
                            <p>Missing required parameters</p>
                        </body>
                    </html>
                    """,
                    status_code=HTTP_BAD_REQUEST,
                )

            # Handle external provider callback
            # Provider should implement handle_external_callback method
            if not hasattr(self.provider, "handle_external_callback"):
                raise NotImplementedError(
                    f"Provider {type(self.provider).__name__} must implement handle_external_callback method"
                )

            result = await self.provider.handle_external_callback(code, state)

            # Redirect back to MCP client with authorization code
            redirect_params = {
                PARAM_CODE: result["code"],
            }
            if result.get(PARAM_STATE):
                redirect_params[PARAM_STATE] = result[PARAM_STATE]

            redirect_url = f"{result['redirect_uri']}?{urlencode(redirect_params)}"

            # Return success page with auto-redirect
            escaped_redirect = html.escape(redirect_url)
            escaped_provider = html.escape(self.provider_name)
            return HTMLResponse(
                f"""
                <html>
                    <head>
                        <title>Authorization Successful</title>
                        <meta http-equiv="refresh" content="3;url={escaped_redirect}">
                    </head>
                    <body>
                        <h1>{escaped_provider} Authorization Successful!</h1>
                        <p>Your {escaped_provider} account has been linked.</p>
                        <p>Redirecting back to the application...</p>
                        <p>If not redirected, <a href="{escaped_redirect}">click here</a>.</p>
                    </body>
                </html>
                """
            )

        except Exception as e:
            logger.error(f"Authorization callback error: {e}")
            return HTMLResponse(
                """
                <html>
                    <head><title>Error</title></head>
                    <body>
                        <h1>Authorization Error</h1>
                        <p>An unexpected error occurred during authorization.</p>
                    </body>
                </html>
                """,
                status_code=HTTP_INTERNAL_SERVER_ERROR,
            )
