#!/usr/bin/env python3
"""
Endpoint constants - Single source of truth for magic strings used across endpoints.
"""

from enum import IntEnum


# ---------------------------------------------------------------------------
# HTTP status codes
# ---------------------------------------------------------------------------
class HttpStatus(IntEnum):
    OK = 200
    ACCEPTED = 202
    NO_CONTENT = 204
    BAD_REQUEST = 400
    NOT_FOUND = 404
    METHOD_NOT_ALLOWED = 405
    INTERNAL_SERVER_ERROR = 500


# ---------------------------------------------------------------------------
# JSON-RPC error codes
# ---------------------------------------------------------------------------
class JsonRpcErrorCode(IntEnum):
    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    INTERNAL_ERROR = -32603


JSONRPC_VERSION = "2.0"


# ---------------------------------------------------------------------------
# Content types
# ---------------------------------------------------------------------------
CONTENT_TYPE_JSON = "application/json"
CONTENT_TYPE_MARKDOWN = "text/markdown"
CONTENT_TYPE_SSE = "text/event-stream"


# ---------------------------------------------------------------------------
# Header names
# ---------------------------------------------------------------------------
HEADER_CORS_ORIGIN = "Access-Control-Allow-Origin"
HEADER_CORS_METHODS = "Access-Control-Allow-Methods"
HEADER_CORS_HEADERS = "Access-Control-Allow-Headers"
HEADER_CORS_MAX_AGE = "Access-Control-Max-Age"
HEADER_CACHE_CONTROL = "Cache-Control"
HEADER_CONTENT_TYPE = "Content-Type"
HEADER_CONNECTION = "Connection"
HEADER_ALLOW = "Allow"
HEADER_MCP_SESSION_ID = "Mcp-Session-Id"
HEADER_ACCEPT = "accept"
HEADER_AUTHORIZATION = "authorization"
HEADER_HOST = "host"


# ---------------------------------------------------------------------------
# Header values
# ---------------------------------------------------------------------------
CORS_ALLOW_ALL = "*"
CACHE_NO_CACHE = "no-cache"
CACHE_SHORT = "public, max-age=300"
CACHE_LONG = "public, max-age=3600, immutable"
CONNECTION_KEEP_ALIVE = "keep-alive"


# ---------------------------------------------------------------------------
# Pre-computed header combinations (shared across endpoints)
# ---------------------------------------------------------------------------
HEADERS_CORS_NOCACHE: dict[str, str] = {
    HEADER_CORS_ORIGIN: CORS_ALLOW_ALL,
    HEADER_CACHE_CONTROL: CACHE_NO_CACHE,
    HEADER_CONTENT_TYPE: CONTENT_TYPE_JSON,
}

HEADERS_CORS_SHORT_CACHE: dict[str, str] = {
    HEADER_CORS_ORIGIN: CORS_ALLOW_ALL,
    HEADER_CACHE_CONTROL: CACHE_SHORT,
    HEADER_CONTENT_TYPE: CONTENT_TYPE_JSON,
}

HEADERS_CORS_LONG_CACHE: dict[str, str] = {
    HEADER_CORS_ORIGIN: CORS_ALLOW_ALL,
    HEADER_CACHE_CONTROL: CACHE_LONG,
    HEADER_CONTENT_TYPE: CONTENT_TYPE_JSON,
}

HEADERS_CORS_ONLY: dict[str, str] = {
    HEADER_CORS_ORIGIN: CORS_ALLOW_ALL,
}

HEADERS_INFO: dict[str, str] = {
    HEADER_CORS_ORIGIN: CORS_ALLOW_ALL,
    HEADER_CACHE_CONTROL: CACHE_SHORT,
}


# ---------------------------------------------------------------------------
# Framework / protocol identity
# ---------------------------------------------------------------------------
SERVER_NAME = "ChukMCPServer"
POWERED_BY = "chuk_mcp"
FRAMEWORK_DESCRIPTION = "ChukMCPServer with chuk_mcp"
MCP_PROTOCOL_VERSION = "2025-03-26"
MCP_PROTOCOL_FULL = "MCP 2025-03-26"
MCP_TRANSPORT = "HTTP with SSE"

# MCP 2025-11-25 headers
HEADER_MCP_PROTOCOL_VERSION = "MCP-Protocol-Version"
HEADER_LAST_EVENT_ID = "Last-Event-ID"
METHOD_DELETE = "DELETE"


# ---------------------------------------------------------------------------
# SSE event framing
# ---------------------------------------------------------------------------
SSE_EVENT_MESSAGE = "event: message\r\n"
SSE_EVENT_ERROR = "event: error\r\n"
SSE_LINE_END = "\r\n"


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------
BEARER_PREFIX = "bearer "


# ---------------------------------------------------------------------------
# HTTP methods
# ---------------------------------------------------------------------------
METHOD_GET = "GET"
METHOD_POST = "POST"
METHOD_OPTIONS = "OPTIONS"


# ---------------------------------------------------------------------------
# Error messages
# ---------------------------------------------------------------------------
ERROR_METHOD_NOT_ALLOWED = "Method not allowed"
ERROR_BAD_REQUEST = "Bad request"
ERROR_NOT_FOUND = "Not found"
ERROR_INTERNAL = "Internal server error"
ERROR_EMPTY_BODY = "Empty request body"
ERROR_INVALID_JSON = "Invalid JSON format"
ERROR_JSON_PARSE = "JSON parsing error"
ERROR_MISSING_SESSION = "Bad Request: Missing session ID"


# ---------------------------------------------------------------------------
# Error types (for structured error responses)
# ---------------------------------------------------------------------------
ERROR_TYPE_BAD_REQUEST = "bad_request"
ERROR_TYPE_NOT_FOUND = "not_found"
ERROR_TYPE_METHOD_NOT_ALLOWED = "method_not_allowed"
ERROR_TYPE_INTERNAL = "internal_error"


# ---------------------------------------------------------------------------
# Status strings
# ---------------------------------------------------------------------------
STATUS_HEALTHY = "healthy"
STATUS_PONG = "pong"
STATUS_READY = "ready"
STATUS_SUCCESS = "success"


# ---------------------------------------------------------------------------
# URL paths
# ---------------------------------------------------------------------------
PATH_MCP = "/mcp"
PATH_HEALTH = "/health"
PATH_PING = "/ping"
PATH_VERSION = "/version"
PATH_INFO = "/"
PATH_DOCS = "/docs"
