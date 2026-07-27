#!/usr/bin/env python3
"""
OAuth constants — grant types, endpoints, TTL values, error types, and environment variables.
"""

# ---------------------------------------------------------------------------
# OAuth grant types
# ---------------------------------------------------------------------------
GRANT_AUTHORIZATION_CODE = "authorization_code"
GRANT_REFRESH_TOKEN = "refresh_token"

# Client authentication methods
AUTH_METHOD_CLIENT_SECRET_POST = "client_secret_post"
AUTH_METHOD_CLIENT_SECRET_BASIC = "client_secret_basic"
AUTH_METHOD_NONE = "none"

SUPPORTED_AUTH_METHODS = (
    AUTH_METHOD_CLIENT_SECRET_POST,
    AUTH_METHOD_CLIENT_SECRET_BASIC,
    AUTH_METHOD_NONE,
)

# Methods that oblige the client to authenticate at the token endpoint.
CONFIDENTIAL_AUTH_METHODS = (
    AUTH_METHOD_CLIENT_SECRET_POST,
    AUTH_METHOD_CLIENT_SECRET_BASIC,
)

# Default for clients that do not declare token_endpoint_auth_method, and for
# clients registered before the field was persisted. RFC 7591 defaults to
# client_secret_basic, but MCP clients are overwhelmingly public + PKCE, and
# defaulting to confidential would reject every already-registered client.
DEFAULT_AUTH_METHOD = AUTH_METHOD_NONE


# ---------------------------------------------------------------------------
# Code challenge methods
# ---------------------------------------------------------------------------
CODE_CHALLENGE_S256 = "S256"
CODE_CHALLENGE_PLAIN = "plain"


# ---------------------------------------------------------------------------
# OAuth response types
# ---------------------------------------------------------------------------
RESPONSE_TYPE_CODE = "code"


# ---------------------------------------------------------------------------
# OAuth endpoint paths
# ---------------------------------------------------------------------------
PATH_AUTHORIZATION_SERVER_METADATA = "/.well-known/oauth-authorization-server"
PATH_PROTECTED_RESOURCE = "/.well-known/oauth-protected-resource"
PATH_AUTHORIZE = "/oauth/authorize"
PATH_TOKEN = "/oauth/token"
PATH_REGISTER = "/oauth/register"
PATH_CALLBACK = "/oauth/callback"


# ---------------------------------------------------------------------------
# Token parameter names
# ---------------------------------------------------------------------------
PARAM_GRANT_TYPE = "grant_type"
PARAM_CLIENT_ID = "client_id"
PARAM_CLIENT_SECRET = "client_secret"
PARAM_TOKEN_ENDPOINT_AUTH_METHOD = "token_endpoint_auth_method"
PARAM_REDIRECT_URI = "redirect_uri"
PARAM_CODE_VERIFIER = "code_verifier"
PARAM_CODE_CHALLENGE = "code_challenge"
PARAM_CODE_CHALLENGE_METHOD = "code_challenge_method"
PARAM_REFRESH_TOKEN = "refresh_token"
PARAM_CODE = "code"
PARAM_STATE = "state"
PARAM_SCOPE = "scope"
PARAM_RESPONSE_TYPE = "response_type"
PARAM_CLIENT_NAME = "client_name"
PARAM_REDIRECT_URIS = "redirect_uris"
PARAM_ERROR = "error"
PARAM_ERROR_DESCRIPTION = "error_description"

# Token response fields
PARAM_ACCESS_TOKEN = "access_token"
PARAM_TOKEN_TYPE = "token_type"
PARAM_EXPIRES_IN = "expires_in"


# ---------------------------------------------------------------------------
# HTTP client authentication (RFC 6749 section 2.3.1)
# ---------------------------------------------------------------------------
HEADER_AUTHORIZATION = "authorization"
HEADER_WWW_AUTHENTICATE = "WWW-Authenticate"
AUTH_SCHEME_BASIC = "basic"
BASIC_AUTH_REALM = 'Basic realm="oauth"'
BASIC_CREDENTIALS_SEPARATOR = ":"


# ---------------------------------------------------------------------------
# Redirect URI handling
# ---------------------------------------------------------------------------
QUERY_START = "?"
QUERY_SEPARATOR = "&"

# Schemes a browser executes rather than navigates to. A client must not be able
# to register one, because the callback page renders the redirect URI as a link.
DANGEROUS_URI_SCHEMES = frozenset(
    {
        "javascript",
        "data",
        "vbscript",
        "file",
        "blob",
        "about",
    }
)


# ---------------------------------------------------------------------------
# HTTP status codes used by the OAuth endpoints
# ---------------------------------------------------------------------------
HTTP_OK = 200
HTTP_CREATED = 201
HTTP_BAD_REQUEST = 400
HTTP_UNAUTHORIZED = 401
HTTP_INTERNAL_SERVER_ERROR = 500


# ---------------------------------------------------------------------------
# OAuth error types
# ---------------------------------------------------------------------------
ERROR_INVALID_REQUEST = "invalid_request"
ERROR_INVALID_GRANT = "invalid_grant"
ERROR_INVALID_CLIENT = "invalid_client"
ERROR_INVALID_TOKEN = "invalid_token"
ERROR_UNSUPPORTED_GRANT_TYPE = "unsupported_grant_type"
ERROR_INSUFFICIENT_SCOPE = "insufficient_scope"
ERROR_INVALID_REDIRECT_URI = "invalid_redirect_uri"
ERROR_INVALID_CLIENT_METADATA = "invalid_client_metadata"
ERROR_SERVER_ERROR = "server_error"


# ---------------------------------------------------------------------------
# TTL defaults (seconds)
# ---------------------------------------------------------------------------
TTL_AUTH_CODE = 300  # 5 minutes
TTL_ACCESS_TOKEN = 900  # 15 minutes
TTL_REFRESH_TOKEN = 86400  # 1 day
TTL_CLIENT_REGISTRATION = 31536000  # 1 year
TTL_EXTERNAL_TOKEN = 86400  # 1 day
TTL_PENDING_AUTH = 600  # 10 minutes
TOKEN_EXPIRY_BUFFER_MINUTES = 5


# ---------------------------------------------------------------------------
# TTL environment variable names
# ---------------------------------------------------------------------------
ENV_AUTH_CODE_TTL = "OAUTH_AUTH_CODE_TTL"
ENV_ACCESS_TOKEN_TTL = "OAUTH_ACCESS_TOKEN_TTL"
ENV_REFRESH_TOKEN_TTL = "OAUTH_REFRESH_TOKEN_TTL"
ENV_CLIENT_REGISTRATION_TTL = "OAUTH_CLIENT_REGISTRATION_TTL"
ENV_EXTERNAL_TOKEN_TTL = "OAUTH_EXTERNAL_TOKEN_TTL"
ENV_PENDING_AUTH_TTL = "OAUTH_PENDING_AUTH_TTL"
ENV_OAUTH_SERVER_URL = "OAUTH_SERVER_URL"


# ---------------------------------------------------------------------------
# Google OAuth environment variables
# ---------------------------------------------------------------------------
ENV_GOOGLE_CLIENT_ID = "GOOGLE_CLIENT_ID"
ENV_GOOGLE_CLIENT_SECRET = "GOOGLE_CLIENT_SECRET"
ENV_GOOGLE_REDIRECT_URI = "GOOGLE_REDIRECT_URI"
ENV_GOOGLE_DRIVE_ROOT_FOLDER = "GOOGLE_DRIVE_ROOT_FOLDER"
GOOGLE_DEFAULT_ROOT_FOLDER = "CHUK"

# Google OAuth URLs
GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_URL = "https://www.googleapis.com/oauth2/v3/userinfo"


# ---------------------------------------------------------------------------
# Provider names
# ---------------------------------------------------------------------------
PROVIDER_GOOGLE_DRIVE = "google_drive"
PROVIDER_GOOGLE = "google"


# ---------------------------------------------------------------------------
# Token expiry default (seconds)
# ---------------------------------------------------------------------------
DEFAULT_TOKEN_EXPIRY = 3600  # 1 hour
