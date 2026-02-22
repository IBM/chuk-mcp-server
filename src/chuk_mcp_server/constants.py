#!/usr/bin/env python3
"""
Top-level constants shared across the chuk_mcp_server package.
"""

import re
from enum import IntEnum

# ---------------------------------------------------------------------------
# JSON-RPC
# ---------------------------------------------------------------------------
JSONRPC_VERSION = "2.0"
JSONRPC_KEY = "jsonrpc"

# JSON-RPC message keys
KEY_METHOD = "method"
KEY_PARAMS = "params"
KEY_ID = "id"
KEY_RESULT = "result"
KEY_ERROR = "error"


class JsonRpcError(IntEnum):
    """Standard JSON-RPC 2.0 error codes."""

    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603


# ---------------------------------------------------------------------------
# MCP protocol
# ---------------------------------------------------------------------------
MCP_PROTOCOL_VERSION_2025_06 = "2025-06-18"
MCP_PROTOCOL_VERSION_2025_03 = "2025-03-26"
MCP_DEFAULT_PROTOCOL_VERSION = MCP_PROTOCOL_VERSION_2025_06


# MCP method names
class McpMethod:
    INITIALIZE = "initialize"
    INITIALIZED = "notifications/initialized"
    PING = "ping"
    TOOLS_LIST = "tools/list"
    TOOLS_CALL = "tools/call"
    RESOURCES_LIST = "resources/list"
    RESOURCES_READ = "resources/read"
    PROMPTS_LIST = "prompts/list"
    PROMPTS_GET = "prompts/get"
    LOGGING_SET_LEVEL = "logging/setLevel"
    SAMPLING_CREATE_MESSAGE = "sampling/createMessage"
    ELICITATION_CREATE = "elicitation/create"
    NOTIFICATIONS_PROGRESS = "notifications/progress"
    ROOTS_LIST = "roots/list"
    NOTIFICATIONS_ROOTS_LIST_CHANGED = "notifications/roots/list_changed"
    RESOURCES_SUBSCRIBE = "resources/subscribe"
    RESOURCES_UNSUBSCRIBE = "resources/unsubscribe"
    NOTIFICATIONS_RESOURCES_UPDATED = "notifications/resources/updated"
    COMPLETION_COMPLETE = "completion/complete"
    RESOURCES_TEMPLATES_LIST = "resources/templates/list"
    NOTIFICATIONS_CANCELLED = "notifications/cancelled"
    NOTIFICATIONS_MESSAGE = "notifications/message"


# MCP initialize parameter keys
KEY_CLIENT_INFO = "clientInfo"
KEY_PROTOCOL_VERSION = "protocolVersion"
KEY_SERVER_INFO = "serverInfo"
KEY_CAPABILITIES = "capabilities"


# ---------------------------------------------------------------------------
# MCP metadata attributes (set by decorators, read by core)
# ---------------------------------------------------------------------------
ATTR_MCP_TOOL = "_mcp_tool"
ATTR_MCP_RESOURCE = "_mcp_resource"
ATTR_MCP_PROMPT = "_mcp_prompt"
ATTR_REQUIRES_AUTH = "_requires_auth"
ATTR_AUTH_SCOPES = "_auth_scopes"

# OAuth injection parameter names
PARAM_EXTERNAL_ACCESS_TOKEN = "_external_access_token"
PARAM_USER_ID = "_user_id"


# ---------------------------------------------------------------------------
# Component types
# ---------------------------------------------------------------------------
COMPONENT_TOOL = "tool"
COMPONENT_RESOURCE = "resource"
COMPONENT_PROMPT = "prompt"
DEFAULT_COMPONENT_VERSION = "1.0.0"


# ---------------------------------------------------------------------------
# Content types
# ---------------------------------------------------------------------------
CONTENT_TYPE_JSON = "application/json"
CONTENT_TYPE_PLAIN = "text/plain"
CONTENT_TYPE_MARKDOWN = "text/markdown"
CONTENT_TYPE_SSE = "text/event-stream"


# ---------------------------------------------------------------------------
# Common HTTP headers
# ---------------------------------------------------------------------------
HEADER_CORS_ORIGIN = "Access-Control-Allow-Origin"
HEADER_CORS_METHODS = "Access-Control-Allow-Methods"
HEADER_CORS_HEADERS = "Access-Control-Allow-Headers"
HEADER_CORS_MAX_AGE = "Access-Control-Max-Age"
HEADER_CONTENT_TYPE = "Content-Type"
HEADER_MCP_SESSION_ID = "Mcp-Session-Id"
CORS_ALLOW_ALL = "*"


# ---------------------------------------------------------------------------
# Environment variables
# ---------------------------------------------------------------------------
ENV_MCP_LOG_LEVEL = "MCP_LOG_LEVEL"
ENV_MCP_TRANSPORT = "MCP_TRANSPORT"
ENV_MCP_STDIO = "MCP_STDIO"
ENV_USE_STDIO = "USE_STDIO"
ENV_MCP_SERVER_NAME = "MCP_SERVER_NAME"
ENV_MCP_SERVER_VERSION = "MCP_SERVER_VERSION"
ENV_PORT = "PORT"


# ---------------------------------------------------------------------------
# Logging level strings
# ---------------------------------------------------------------------------
LOG_DEBUG = "debug"
LOG_INFO = "info"
LOG_NOTICE = "notice"
LOG_WARNING = "warning"
LOG_ERROR = "error"
LOG_CRITICAL = "critical"
LOG_ALERT = "alert"
LOG_EMERGENCY = "emergency"


# ---------------------------------------------------------------------------
# Network defaults
# ---------------------------------------------------------------------------
DEFAULT_HOST = "0.0.0.0"
DEFAULT_ENCODING = "utf-8"
DEFAULT_PORT = 8000


# ---------------------------------------------------------------------------
# Server identity
# ---------------------------------------------------------------------------
SERVER_NAME = "ChukMCPServer"
POWERED_BY = "chuk_mcp"
FRAMEWORK_DESCRIPTION = "ChukMCPServer with chuk_mcp"
PACKAGE_LOGGER = "chuk_mcp_server"


# ---------------------------------------------------------------------------
# Pagination
# ---------------------------------------------------------------------------
KEY_CURSOR = "cursor"
KEY_NEXT_CURSOR = "nextCursor"
DEFAULT_PAGE_SIZE = 100


# ---------------------------------------------------------------------------
# Tool annotations keys (MCP 2025-03-26)
# ---------------------------------------------------------------------------
KEY_ANNOTATIONS = "annotations"
KEY_READ_ONLY_HINT = "readOnlyHint"
KEY_DESTRUCTIVE_HINT = "destructiveHint"
KEY_IDEMPOTENT_HINT = "idempotentHint"
KEY_OPEN_WORLD_HINT = "openWorldHint"


# ---------------------------------------------------------------------------
# MCP error codes beyond JSON-RPC standard
# ---------------------------------------------------------------------------
MCP_ERROR_RESOURCE_NOT_FOUND = -32002
MCP_ERROR_URL_ELICITATION_REQUIRED = -32042


# ---------------------------------------------------------------------------
# Tool name validation (MCP 2025-11-25)
# ---------------------------------------------------------------------------
TOOL_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9_\-\.]{1,128}$")


# ---------------------------------------------------------------------------
# Tasks system (MCP 2025-11-25)
# ---------------------------------------------------------------------------
class McpTaskMethod:
    TASKS_GET = "tasks/get"
    TASKS_RESULT = "tasks/result"
    TASKS_LIST = "tasks/list"
    TASKS_CANCEL = "tasks/cancel"
    NOTIFICATIONS_TASKS_STATUS = "notifications/tasks/status"


# ---------------------------------------------------------------------------
# MCP protocol version (2025-11-25)
# ---------------------------------------------------------------------------
MCP_PROTOCOL_VERSION_2025_11 = "2025-11-25"
HEADER_MCP_PROTOCOL_VERSION = "MCP-Protocol-Version"
HEADER_LAST_EVENT_ID = "Last-Event-ID"


# ---------------------------------------------------------------------------
# Request validation limits (Phase 5: Production Hardening)
# ---------------------------------------------------------------------------
MAX_REQUEST_BODY_BYTES = 10 * 1024 * 1024  # 10 MB
MAX_ARGUMENT_KEYS = 100
MAX_PENDING_REQUESTS = 100


# ---------------------------------------------------------------------------
# Rate limiting (Phase 5: Production Hardening)
# ---------------------------------------------------------------------------
DEFAULT_RATE_LIMIT_RPS = 100.0
DEFAULT_RATE_LIMIT_BURST = 200.0
