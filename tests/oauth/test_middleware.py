"""Tests for OAuth middleware."""

from unittest.mock import AsyncMock, Mock

import pytest
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, RedirectResponse

from chuk_mcp_server.oauth.base_provider import BaseOAuthProvider
from chuk_mcp_server.oauth.middleware import OAuthMiddleware
from chuk_mcp_server.oauth.models import (
    AuthorizeError,
    OAuthClientInfo,
    OAuthToken,
    RegistrationError,
    TokenError,
)


class MockOAuthProvider(BaseOAuthProvider):
    """Mock OAuth provider for testing.

    Deliberately keeps the pre-client-authentication method signatures so these
    tests also cover the middleware's backwards-compatibility path for providers
    written against an earlier release. See TestClientAuthentication for a
    provider that accepts client_secret.
    """

    def __init__(self):
        self.authorize_called = False
        self.exchange_code_called = False
        self.exchange_refresh_called = False
        self.validate_token_called = False
        self.register_client_called = False

    async def authorize(self, params):
        self.authorize_called = True
        return {"code": "test_auth_code_123", "state": params.state}

    async def exchange_authorization_code(self, code, client_id, redirect_uri, code_verifier=None):
        self.exchange_code_called = True
        return OAuthToken(
            access_token="test_access_token",
            refresh_token="test_refresh_token",
            expires_in=3600,
        )

    async def exchange_refresh_token(self, refresh_token, client_id, scope=None):
        self.exchange_refresh_called = True
        return OAuthToken(access_token="new_access_token", expires_in=3600)

    async def validate_access_token(self, token):
        self.validate_token_called = True
        return {"user_id": "test_user", "client_id": "test_client"}

    async def register_client(self, client_metadata):
        self.register_client_called = True
        return OAuthClientInfo(
            client_id="new_client_id",
            client_secret="new_client_secret",
            client_name=client_metadata.get("client_name", "Test Client"),
            redirect_uris=client_metadata.get("redirect_uris", []),
        )


class TestOAuthMiddleware:
    """Test OAuthMiddleware class."""

    def test_init_basic(self):
        """Test basic OAuthMiddleware initialization."""
        mock_server = Mock()
        mock_server.endpoint = Mock(return_value=lambda f: f)
        provider = MockOAuthProvider()

        middleware = OAuthMiddleware(
            mcp_server=mock_server,
            provider=provider,
            oauth_server_url="http://localhost:8000",
        )

        assert middleware.mcp == mock_server
        assert middleware.provider == provider
        assert middleware.oauth_server_url == "http://localhost:8000"
        assert middleware.callback_path == "/oauth/callback"
        assert middleware.scopes_supported == []
        assert middleware.service_documentation is None
        assert middleware.provider_name == "OAuth Provider"

    def test_init_with_custom_values(self):
        """Test OAuthMiddleware initialization with custom values."""
        mock_server = Mock()
        mock_server.endpoint = Mock(return_value=lambda f: f)
        provider = MockOAuthProvider()

        middleware = OAuthMiddleware(
            mcp_server=mock_server,
            provider=provider,
            oauth_server_url="https://api.example.com",
            callback_path="/custom/callback",
            scopes_supported=["read", "write"],
            service_documentation="https://docs.example.com",
            provider_name="LinkedIn",
        )

        assert middleware.oauth_server_url == "https://api.example.com"
        assert middleware.callback_path == "/custom/callback"
        assert middleware.scopes_supported == ["read", "write"]
        assert middleware.service_documentation == "https://docs.example.com"
        assert middleware.provider_name == "LinkedIn"

    def test_register_endpoints_called(self):
        """Test that _register_endpoints is called during initialization."""
        mock_server = Mock()
        endpoint_calls = []

        def mock_endpoint(path, methods=None):
            def decorator(func):
                endpoint_calls.append((path, methods))
                return func

            return decorator

        mock_server.endpoint = mock_endpoint
        provider = MockOAuthProvider()

        OAuthMiddleware(
            mcp_server=mock_server,
            provider=provider,
        )

        # Verify all endpoints were registered
        expected_paths = [
            "/.well-known/oauth-authorization-server",
            "/oauth/authorize",
            "/oauth/token",
            "/oauth/register",
            "/oauth/callback",
        ]
        registered_paths = [path for path, _ in endpoint_calls]
        for path in expected_paths:
            assert path in registered_paths

    @pytest.mark.asyncio
    async def test_metadata_endpoint(self):
        """Test OAuth metadata endpoint."""
        mock_server = Mock()
        mock_server.endpoint = Mock(return_value=lambda f: f)
        provider = MockOAuthProvider()

        middleware = OAuthMiddleware(
            mcp_server=mock_server,
            provider=provider,
            oauth_server_url="http://localhost:8000",
            scopes_supported=["read", "write"],
        )

        # Create mock request
        mock_request = Mock(spec=Request)

        # Call metadata endpoint
        response = await middleware._metadata_endpoint(mock_request)

        assert isinstance(response, JSONResponse)
        # Parse the response body
        import json

        body = json.loads(response.body)
        assert body["issuer"] == "http://localhost:8000"
        assert body["authorization_endpoint"] == "http://localhost:8000/oauth/authorize"
        assert body["token_endpoint"] == "http://localhost:8000/oauth/token"
        assert "read" in body["scopes_supported"]
        assert "write" in body["scopes_supported"]

    @pytest.mark.asyncio
    async def test_authorize_endpoint_success(self):
        """Test successful authorization endpoint."""
        mock_server = Mock()
        mock_server.endpoint = Mock(return_value=lambda f: f)
        provider = MockOAuthProvider()

        middleware = OAuthMiddleware(
            mcp_server=mock_server,
            provider=provider,
        )

        # Create mock request with query params
        mock_request = Mock(spec=Request)
        mock_request.query_params = {
            "response_type": "code",
            "client_id": "test_client",
            "redirect_uri": "http://localhost/callback",
            "state": "random_state",
        }

        # Call authorize endpoint
        response = await middleware._authorize_endpoint(mock_request)

        assert provider.authorize_called
        assert isinstance(response, RedirectResponse)
        assert "code=test_auth_code_123" in response.headers["location"]
        assert "state=random_state" in response.headers["location"]

    @pytest.mark.asyncio
    async def test_authorize_endpoint_with_pkce(self):
        """Test authorization endpoint with PKCE."""
        mock_server = Mock()
        mock_server.endpoint = Mock(return_value=lambda f: f)
        provider = MockOAuthProvider()

        middleware = OAuthMiddleware(
            mcp_server=mock_server,
            provider=provider,
        )

        mock_request = Mock(spec=Request)
        mock_request.query_params = {
            "response_type": "code",
            "client_id": "test_client",
            "redirect_uri": "http://localhost/callback",
            "code_challenge": "test_challenge",
            "code_challenge_method": "S256",
        }

        response = await middleware._authorize_endpoint(mock_request)

        assert provider.authorize_called
        assert isinstance(response, RedirectResponse)

    @pytest.mark.asyncio
    async def test_authorize_endpoint_error(self):
        """Test authorization endpoint with error."""
        mock_server = Mock()
        mock_server.endpoint = Mock(return_value=lambda f: f)

        # Provider that raises an error
        provider = MockOAuthProvider()

        async def authorize_error(params):
            raise AuthorizeError("access_denied", "User denied access")

        provider.authorize = authorize_error

        middleware = OAuthMiddleware(
            mcp_server=mock_server,
            provider=provider,
        )

        mock_request = Mock(spec=Request)
        mock_request.query_params = {
            "response_type": "code",
            "client_id": "test_client",
            "redirect_uri": "http://localhost/callback",
        }

        response = await middleware._authorize_endpoint(mock_request)

        assert isinstance(response, RedirectResponse)
        # Middleware wraps all exceptions as server_error with sanitized description
        assert "error=server_error" in response.headers["location"]
        assert "Authorization+failed" in response.headers["location"]

    @pytest.mark.asyncio
    async def test_token_endpoint_authorization_code_grant(self):
        """Test token endpoint with authorization_code grant."""
        mock_server = Mock()
        mock_server.endpoint = Mock(return_value=lambda f: f)
        provider = MockOAuthProvider()

        middleware = OAuthMiddleware(
            mcp_server=mock_server,
            provider=provider,
        )

        # Create mock request with form data
        mock_request = Mock(spec=Request)
        form_data = {
            "grant_type": "authorization_code",
            "code": "test_code",
            "client_id": "test_client",
            "redirect_uri": "http://localhost/callback",
        }
        mock_request.form = AsyncMock(return_value=form_data)
        mock_request.headers = {}

        response = await middleware._token_endpoint(mock_request)

        assert provider.exchange_code_called
        assert isinstance(response, JSONResponse)
        import json

        body = json.loads(response.body)
        assert body["access_token"] == "test_access_token"
        assert body["token_type"] == "Bearer"

    @pytest.mark.asyncio
    async def test_token_endpoint_refresh_token_grant(self):
        """Test token endpoint with refresh_token grant."""
        mock_server = Mock()
        mock_server.endpoint = Mock(return_value=lambda f: f)
        provider = MockOAuthProvider()

        middleware = OAuthMiddleware(
            mcp_server=mock_server,
            provider=provider,
        )

        mock_request = Mock(spec=Request)
        form_data = {
            "grant_type": "refresh_token",
            "refresh_token": "test_refresh",
            "client_id": "test_client",
        }
        mock_request.form = AsyncMock(return_value=form_data)
        mock_request.headers = {}

        response = await middleware._token_endpoint(mock_request)

        assert provider.exchange_refresh_called
        assert isinstance(response, JSONResponse)

    @pytest.mark.asyncio
    async def test_token_endpoint_unsupported_grant(self):
        """Test token endpoint with unsupported grant type."""
        mock_server = Mock()
        mock_server.endpoint = Mock(return_value=lambda f: f)
        provider = MockOAuthProvider()

        middleware = OAuthMiddleware(
            mcp_server=mock_server,
            provider=provider,
        )

        mock_request = Mock(spec=Request)
        form_data = {
            "grant_type": "unsupported_grant",
        }
        mock_request.form = AsyncMock(return_value=form_data)
        mock_request.headers = {}

        response = await middleware._token_endpoint(mock_request)

        assert isinstance(response, JSONResponse)
        assert response.status_code == 400
        import json

        body = json.loads(response.body)
        assert body["error"] == "unsupported_grant_type"

    @pytest.mark.asyncio
    async def test_token_endpoint_missing_parameters(self):
        """Test token endpoint with missing parameters."""
        mock_server = Mock()
        mock_server.endpoint = Mock(return_value=lambda f: f)
        provider = MockOAuthProvider()

        middleware = OAuthMiddleware(
            mcp_server=mock_server,
            provider=provider,
        )

        mock_request = Mock(spec=Request)
        form_data = {
            "grant_type": "authorization_code",
            # Missing code, client_id, redirect_uri
        }
        mock_request.form = AsyncMock(return_value=form_data)
        mock_request.headers = {}

        response = await middleware._token_endpoint(mock_request)

        assert isinstance(response, JSONResponse)
        assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_register_endpoint_success(self):
        """Test client registration endpoint success."""
        mock_server = Mock()
        mock_server.endpoint = Mock(return_value=lambda f: f)
        provider = MockOAuthProvider()

        middleware = OAuthMiddleware(
            mcp_server=mock_server,
            provider=provider,
        )

        mock_request = Mock(spec=Request)
        mock_request.json = AsyncMock(
            return_value={
                "client_name": "My Application",
                "redirect_uris": ["http://localhost/callback"],
            }
        )

        response = await middleware._register_endpoint(mock_request)

        assert provider.register_client_called
        assert isinstance(response, JSONResponse)
        import json

        body = json.loads(response.body)
        assert body["client_id"] == "new_client_id"
        assert body["client_secret"] == "new_client_secret"

    @pytest.mark.asyncio
    async def test_register_endpoint_error(self):
        """Test client registration endpoint with error."""
        mock_server = Mock()
        mock_server.endpoint = Mock(return_value=lambda f: f)

        provider = MockOAuthProvider()

        async def register_error(client_metadata):
            raise RegistrationError("invalid_redirect_uri", "Invalid redirect URI")

        provider.register_client = register_error

        middleware = OAuthMiddleware(
            mcp_server=mock_server,
            provider=provider,
        )

        mock_request = Mock(spec=Request)
        mock_request.json = AsyncMock(return_value={"client_name": "Test"})

        response = await middleware._register_endpoint(mock_request)

        assert isinstance(response, JSONResponse)
        assert response.status_code == 400
        import json

        body = json.loads(response.body)
        # RegistrationError carries an RFC 7591 error code, which is surfaced as-is
        # so the client can tell why registration was refused.
        assert body["error"] == "invalid_redirect_uri"
        assert body["error_description"] == "Invalid redirect URI"

    @pytest.mark.asyncio
    async def test_external_callback_success(self):
        """Test external callback endpoint success."""
        mock_server = Mock()
        mock_server.endpoint = Mock(return_value=lambda f: f)
        provider = MockOAuthProvider()

        # Add handle_external_callback method
        async def handle_external_callback(code, state):
            return {"code": "mcp_code_123", "state": state, "redirect_uri": "http://localhost/app/callback"}

        provider.handle_external_callback = handle_external_callback

        middleware = OAuthMiddleware(
            mcp_server=mock_server,
            provider=provider,
            provider_name="LinkedIn",
        )

        mock_request = Mock(spec=Request)
        mock_request.query_params = {
            "code": "external_code_123",
            "state": "test_state",
        }

        response = await middleware._external_callback_endpoint(mock_request)

        assert isinstance(response, HTMLResponse)
        assert "LinkedIn Authorization Successful" in str(response.body)
        assert "mcp_code_123" in str(response.body)

    @pytest.mark.asyncio
    async def test_external_callback_with_error(self):
        """Test external callback endpoint with error from provider."""
        mock_server = Mock()
        mock_server.endpoint = Mock(return_value=lambda f: f)
        provider = MockOAuthProvider()

        middleware = OAuthMiddleware(
            mcp_server=mock_server,
            provider=provider,
        )

        mock_request = Mock(spec=Request)
        mock_request.query_params = {
            "error": "access_denied",
            "error_description": "User denied access",
        }

        response = await middleware._external_callback_endpoint(mock_request)

        assert isinstance(response, HTMLResponse)
        assert response.status_code == 400
        assert "access_denied" in str(response.body)

    @pytest.mark.asyncio
    async def test_external_callback_missing_params(self):
        """Test external callback endpoint with missing parameters."""
        mock_server = Mock()
        mock_server.endpoint = Mock(return_value=lambda f: f)
        provider = MockOAuthProvider()

        middleware = OAuthMiddleware(
            mcp_server=mock_server,
            provider=provider,
        )

        mock_request = Mock(spec=Request)
        mock_request.query_params = {}

        response = await middleware._external_callback_endpoint(mock_request)

        assert isinstance(response, HTMLResponse)
        assert response.status_code == 400
        assert "Invalid OAuth Callback" in str(response.body)

    @pytest.mark.asyncio
    async def test_external_callback_not_implemented(self):
        """Test external callback when provider doesn't implement it."""
        mock_server = Mock()
        mock_server.endpoint = Mock(return_value=lambda f: f)
        provider = MockOAuthProvider()
        # Remove the method if it exists
        if hasattr(provider, "handle_external_callback"):
            delattr(provider, "handle_external_callback")

        middleware = OAuthMiddleware(
            mcp_server=mock_server,
            provider=provider,
        )

        mock_request = Mock(spec=Request)
        mock_request.query_params = {
            "code": "test_code",
            "state": "test_state",
        }

        response = await middleware._external_callback_endpoint(mock_request)

        assert isinstance(response, HTMLResponse)
        assert response.status_code == 500

    @pytest.mark.asyncio
    async def test_authorize_endpoint_external_redirect(self):
        """Test authorize endpoint that requires external authorization."""
        mock_server = Mock()
        mock_server.endpoint = Mock(return_value=lambda f: f)
        provider = MockOAuthProvider()

        async def authorize_external(params):
            return {
                "requires_external_authorization": True,
                "authorization_url": "https://provider.com/oauth/authorize?client_id=123",
            }

        provider.authorize = authorize_external

        middleware = OAuthMiddleware(
            mcp_server=mock_server,
            provider=provider,
        )

        mock_request = Mock(spec=Request)
        mock_request.query_params = {
            "response_type": "code",
            "client_id": "test_client",
            "redirect_uri": "http://localhost/callback",
        }

        response = await middleware._authorize_endpoint(mock_request)

        assert isinstance(response, RedirectResponse)
        assert "provider.com" in response.headers["location"]

    @pytest.mark.asyncio
    async def test_token_endpoint_with_code_verifier(self):
        """Test token endpoint with PKCE code verifier."""
        mock_server = Mock()
        mock_server.endpoint = Mock(return_value=lambda f: f)
        provider = MockOAuthProvider()

        middleware = OAuthMiddleware(
            mcp_server=mock_server,
            provider=provider,
        )

        mock_request = Mock(spec=Request)
        form_data = {
            "grant_type": "authorization_code",
            "code": "test_code",
            "client_id": "test_client",
            "redirect_uri": "http://localhost/callback",
            "code_verifier": "test_verifier",
        }
        mock_request.form = AsyncMock(return_value=form_data)
        mock_request.headers = {}

        response = await middleware._token_endpoint(mock_request)

        assert provider.exchange_code_called
        assert isinstance(response, JSONResponse)

    @pytest.mark.asyncio
    async def test_token_endpoint_error_handling(self):
        """Test token endpoint error handling."""
        mock_server = Mock()
        mock_server.endpoint = Mock(return_value=lambda f: f)
        provider = MockOAuthProvider()

        async def exchange_error(code, client_id, redirect_uri, code_verifier=None):
            raise TokenError("invalid_grant", "Code expired")

        provider.exchange_authorization_code = exchange_error

        middleware = OAuthMiddleware(
            mcp_server=mock_server,
            provider=provider,
        )

        mock_request = Mock(spec=Request)
        form_data = {
            "grant_type": "authorization_code",
            "code": "expired_code",
            "client_id": "test_client",
            "redirect_uri": "http://localhost/callback",
        }
        mock_request.form = AsyncMock(return_value=form_data)
        mock_request.headers = {}

        response = await middleware._token_endpoint(mock_request)

        assert isinstance(response, JSONResponse)
        assert response.status_code == 400
        import json

        body = json.loads(response.body)
        # A TokenError's own code is surfaced; the description stays sanitized.
        assert body["error"] == "invalid_grant"
        assert body["error_description"] == "Token exchange failed"

    @pytest.mark.asyncio
    async def test_authorize_with_state(self):
        """Test authorize endpoint preserves state parameter."""
        mock_server = Mock()
        mock_server.endpoint = Mock(return_value=lambda f: f)
        provider = MockOAuthProvider()

        middleware = OAuthMiddleware(
            mcp_server=mock_server,
            provider=provider,
        )

        mock_request = Mock(spec=Request)
        mock_request.query_params = {
            "response_type": "code",
            "client_id": "test_client",
            "redirect_uri": "http://localhost/callback",
            "state": "custom_state_123",
        }

        response = await middleware._authorize_endpoint(mock_request)

        assert isinstance(response, RedirectResponse)
        assert "state=custom_state_123" in response.headers["location"]

    @pytest.mark.asyncio
    async def test_authorize_error_with_state(self):
        """Test authorize endpoint error handling preserves state parameter."""
        mock_server = Mock()
        mock_server.endpoint = Mock(return_value=lambda f: f)
        provider = MockOAuthProvider()

        async def authorize_error(params):
            raise Exception("Something went wrong")

        provider.authorize = authorize_error

        middleware = OAuthMiddleware(
            mcp_server=mock_server,
            provider=provider,
        )

        mock_request = Mock(spec=Request)
        mock_request.query_params = {
            "response_type": "code",
            "client_id": "test_client",
            "redirect_uri": "http://localhost/callback",
            "state": "test_state_456",
        }

        response = await middleware._authorize_endpoint(mock_request)

        assert isinstance(response, RedirectResponse)
        assert "state=test_state_456" in response.headers["location"]
        assert "error=server_error" in response.headers["location"]

    @pytest.mark.asyncio
    async def test_metadata_endpoint_with_scopes_and_docs(self):
        """Test metadata endpoint with scopes and service documentation."""
        mock_server = Mock()
        mock_server.endpoint = Mock(return_value=lambda f: f)
        provider = MockOAuthProvider()

        middleware = OAuthMiddleware(
            mcp_server=mock_server,
            provider=provider,
            oauth_server_url="http://localhost:8000",
            scopes_supported=["read", "write", "admin"],
            service_documentation="https://docs.example.com/api",
        )

        mock_request = Mock(spec=Request)
        response = await middleware._metadata_endpoint(mock_request)

        assert isinstance(response, JSONResponse)
        import json

        body = json.loads(response.body)
        assert body["scopes_supported"] == ["read", "write", "admin"]
        assert body["service_documentation"] == "https://docs.example.com/api"

    @pytest.mark.asyncio
    async def test_protected_resource_endpoint(self):
        """Test protected resource metadata endpoint."""
        mock_server = Mock()
        mock_server.endpoint = Mock(return_value=lambda f: f)
        provider = MockOAuthProvider()

        middleware = OAuthMiddleware(
            mcp_server=mock_server,
            provider=provider,
            oauth_server_url="http://localhost:8000",
            scopes_supported=["read", "write"],
            service_documentation="https://docs.example.com/api",
        )

        mock_request = Mock(spec=Request)
        response = await middleware._protected_resource_endpoint(mock_request)

        assert isinstance(response, JSONResponse)
        import json

        body = json.loads(response.body)
        assert body["resource"] == "http://localhost:8000"
        assert body["authorization_servers"] == ["http://localhost:8000"]
        assert body["bearer_methods_supported"] == ["header"]
        assert body["resource_signing_alg_values_supported"] == ["RS256"]
        assert body["scopes_supported"] == ["read", "write"]
        assert body["resource_documentation"] == "https://docs.example.com/api"
        assert response.headers["access-control-allow-origin"] == "*"

    @pytest.mark.asyncio
    async def test_protected_resource_endpoint_minimal(self):
        """Test protected resource endpoint without optional metadata."""
        mock_server = Mock()
        mock_server.endpoint = Mock(return_value=lambda f: f)
        provider = MockOAuthProvider()

        middleware = OAuthMiddleware(
            mcp_server=mock_server,
            provider=provider,
            oauth_server_url="http://localhost:8000",
        )

        mock_request = Mock(spec=Request)
        response = await middleware._protected_resource_endpoint(mock_request)

        assert isinstance(response, JSONResponse)
        import json

        body = json.loads(response.body)
        assert body["resource"] == "http://localhost:8000"
        assert "scopes_supported" not in body
        assert "resource_documentation" not in body

    @pytest.mark.asyncio
    async def test_token_endpoint_refresh_missing_params(self):
        """Test token endpoint refresh grant with missing parameters."""
        mock_server = Mock()
        mock_server.endpoint = Mock(return_value=lambda f: f)
        provider = MockOAuthProvider()

        middleware = OAuthMiddleware(
            mcp_server=mock_server,
            provider=provider,
        )

        # Test missing refresh_token
        mock_request = Mock(spec=Request)
        form_data = {
            "grant_type": "refresh_token",
            "client_id": "test_client",
            # Missing refresh_token
        }
        mock_request.form = AsyncMock(return_value=form_data)
        mock_request.headers = {}

        response = await middleware._token_endpoint(mock_request)

        assert isinstance(response, JSONResponse)
        assert response.status_code == 400
        import json

        body = json.loads(response.body)
        assert body["error"] == "invalid_request"

        # Test missing client_id
        mock_request = Mock(spec=Request)
        form_data = {
            "grant_type": "refresh_token",
            "refresh_token": "test_refresh",
            # Missing client_id
        }
        mock_request.form = AsyncMock(return_value=form_data)
        mock_request.headers = {}

        response = await middleware._token_endpoint(mock_request)

        assert isinstance(response, JSONResponse)
        assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_external_callback_without_state_in_result(self):
        """Test external callback when result doesn't include state."""
        mock_server = Mock()
        mock_server.endpoint = Mock(return_value=lambda f: f)
        provider = MockOAuthProvider()

        # Provider returns result without state
        async def handle_external_callback(code, state):
            return {
                "code": "mcp_code_123",
                # No state in result
                "redirect_uri": "http://localhost/app/callback",
            }

        provider.handle_external_callback = handle_external_callback

        middleware = OAuthMiddleware(
            mcp_server=mock_server,
            provider=provider,
            provider_name="Test Provider",
        )

        mock_request = Mock(spec=Request)
        mock_request.query_params = {
            "code": "external_code_123",
            "state": "test_state",
        }

        response = await middleware._external_callback_endpoint(mock_request)

        assert isinstance(response, HTMLResponse)
        assert "Test Provider Authorization Successful" in str(response.body)
        # State should not be in the redirect URL
        assert "state=" not in str(response.body)

    @pytest.mark.asyncio
    async def test_registered_endpoints_are_callable(self):
        """Test that registered endpoint decorators work when called through the server."""
        mock_server = Mock()
        registered_functions = {}

        def mock_endpoint(path, methods=None):
            def decorator(func):
                registered_functions[path] = func
                return func

            return decorator

        mock_server.endpoint = mock_endpoint
        provider = MockOAuthProvider()

        OAuthMiddleware(
            mcp_server=mock_server,
            provider=provider,
            scopes_supported=["read"],
            service_documentation="https://docs.example.com",
        )

        # Test that all endpoints were registered and are callable
        assert "/.well-known/oauth-authorization-server" in registered_functions
        assert "/.well-known/oauth-protected-resource" in registered_functions
        assert "/oauth/authorize" in registered_functions
        assert "/oauth/token" in registered_functions
        assert "/oauth/register" in registered_functions
        assert "/oauth/callback" in registered_functions

        # Test calling the registered functions through the decorator
        mock_request = Mock(spec=Request)
        mock_request.query_params = {}

        # Call metadata endpoint through registered function
        metadata_func = registered_functions["/.well-known/oauth-authorization-server"]
        response = await metadata_func(mock_request)
        assert isinstance(response, JSONResponse)

        # Call protected resource endpoint through registered function
        protected_func = registered_functions["/.well-known/oauth-protected-resource"]
        response = await protected_func(mock_request)
        assert isinstance(response, JSONResponse)

        # Call authorize endpoint through registered function
        mock_request.query_params = {
            "response_type": "code",
            "client_id": "test_client",
            "redirect_uri": "http://localhost/callback",
        }
        authorize_func = registered_functions["/oauth/authorize"]
        response = await authorize_func(mock_request)
        assert isinstance(response, RedirectResponse)

        # Call token endpoint through registered function
        mock_request.headers = {}
        mock_request.form = AsyncMock(
            return_value={
                "grant_type": "authorization_code",
                "code": "test_code",
                "client_id": "test_client",
                "redirect_uri": "http://localhost/callback",
            }
        )
        token_func = registered_functions["/oauth/token"]
        response = await token_func(mock_request)
        assert isinstance(response, JSONResponse)

        # Call register endpoint through registered function
        mock_request.json = AsyncMock(
            return_value={
                "client_name": "Test Client",
                "redirect_uris": ["http://localhost/callback"],
            }
        )
        register_func = registered_functions["/oauth/register"]
        response = await register_func(mock_request)
        assert isinstance(response, JSONResponse)

        # Call external callback endpoint through registered function
        async def handle_external_callback(code, state):
            return {"code": "mcp_code", "redirect_uri": "http://localhost/callback"}

        provider.handle_external_callback = handle_external_callback
        mock_request.query_params = {
            "code": "external_code",
            "state": "test_state",
        }
        callback_func = registered_functions["/oauth/callback"]
        response = await callback_func(mock_request)
        assert isinstance(response, HTMLResponse)
