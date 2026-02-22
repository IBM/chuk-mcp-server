# chuk-mcp-server Technical Specification

**Package**: `chuk-mcp-server` v0.21.0
**License**: MIT
**Python**: >= 3.11
**MCP Protocol**: `2025-11-25` (default), `2025-06-18` (supported), `2025-03-26` (supported)

---

## Table of Contents

1. [Overview](#overview)
2. [Server API](#server-api)
3. [Decorators](#decorators)
4. [Context API](#context-api)
5. [MCP Protocol](#mcp-protocol)
6. [Transport Modes](#transport-modes)
7. [HTTP Endpoints](#http-endpoints)
8. [Type System](#type-system)
9. [OAuth](#oauth)
10. [Sampling](#sampling)
11. [Elicitation](#elicitation)
12. [Progress Notifications](#progress-notifications)
13. [Logging Notifications](#logging-notifications)
14. [Roots](#roots)
15. [Resource Subscriptions](#resource-subscriptions)
16. [Completions](#completions)
17. [Tasks](#tasks)
18. [Artifacts](#artifacts)
19. [Server Composition](#server-composition)
20. [Configuration](#configuration)
21. [Testing](#testing)
22. [OpenAPI](#openapi)
23. [CLI](#cli)

---

## Overview

`chuk-mcp-server` is a zero-configuration MCP (Model Context Protocol) framework for building MCP servers in Python. It provides decorator-based registration of tools, resources, and prompts, automatic environment detection (local, Docker, cloud), and support for both HTTP and STDIO transports with bidirectional sampling.

### Minimal Example

```python
from chuk_mcp_server import tool, resource, run

@tool
def hello(name: str) -> str:
    return f"Hello, {name}!"

@resource("config://settings")
def get_settings() -> dict:
    return {"app": "my_app"}

if __name__ == "__main__":
    run()
```

### Core Dependencies

| Package | Purpose |
|---------|---------|
| `chuk-mcp` | MCP type definitions and content helpers |
| `starlette` | HTTP application framework |
| `uvicorn` | ASGI server |
| `orjson` | High-performance JSON serialization |
| `httptools` | Fast HTTP parsing |

---

## Server API

### Constructor

```python
ChukMCPServer(
    name: str | None = None,
    version: str = "1.0.0",
    title: str | None = None,
    description: str | None = None,
    icons: list[dict[str, Any]] | None = None,
    website_url: str | None = None,
    capabilities: ServerCapabilities | None = None,
    tools: bool = True,
    resources: bool = True,
    prompts: bool = False,
    logging: bool = False,
    completions: bool = False,
    experimental: dict[str, Any] | None = None,
    host: str | None = None,
    port: int | None = None,
    debug: bool | None = None,
    transport: str | None = None,
    proxy_config: dict[str, Any] | None = None,
    tool_modules_config: dict[str, Any] | None = None,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str \| None` | Auto-detected | Server name; inferred from project if `None` |
| `version` | `str` | `"1.0.0"` | Server version string |
| `title` | `str \| None` | `None` | Human-readable server title (MCP 2025-11-25) |
| `description` | `str \| None` | `None` | Server description (MCP 2025-11-25) |
| `icons` | `list[dict] \| None` | `None` | Server icons for UI rendering (MCP 2025-11-25). Each dict has `uri` and `mimeType` |
| `website_url` | `str \| None` | `None` | Server website URL (MCP 2025-11-25) |
| `capabilities` | `ServerCapabilities \| None` | Auto-configured | MCP capabilities; built from boolean flags if `None` |
| `tools` | `bool` | `True` | Enable the tools capability |
| `resources` | `bool` | `True` | Enable the resources capability |
| `prompts` | `bool` | `False` | Enable the prompts capability |
| `logging` | `bool` | `False` | Enable the logging capability |
| `completions` | `bool` | `False` | Enable the completions capability |
| `experimental` | `dict \| None` | `None` | Experimental capabilities dict |
| `host` | `str \| None` | Auto-detected | Host to bind; uses SmartConfig if `None` |
| `port` | `int \| None` | Auto-detected | Port to bind; uses SmartConfig if `None` |
| `debug` | `bool \| None` | Auto-detected | Debug mode; uses SmartConfig if `None` |
| `transport` | `str \| None` | Auto-detected | Transport mode: `"http"` or `"stdio"` |
| `proxy_config` | `dict \| None` | `None` | Multi-server proxy configuration |
| `tool_modules_config` | `dict \| None` | `None` | Dynamic tool module loading configuration |

### Factory Functions

```python
create_mcp_server(name: str | None = None, **kwargs) -> ChukMCPServer
```

Factory function; equivalent to `ChukMCPServer(name=name, **kwargs)`.

```python
quick_server(name: str | None = None) -> ChukMCPServer
```

Creates a minimal server for prototyping. Defaults to `name="Quick Server"` and `version="0.1.0"`.

```python
get_mcp_server() -> ChukMCPServer
```

Returns the global singleton `ChukMCPServer` instance; creates one if it does not exist.

```python
run(transport: str = "http", **kwargs) -> None
```

Runs the global server. Delegates to `server.run()` for HTTP or `server.run_stdio()` for STDIO.

### Running the Server

```python
mcp.run(
    host: str | None = None,
    port: int | None = None,
    debug: bool | None = None,
    stdio: bool | None = None,
    log_level: str = "warning",
    post_register_hook: Callable | None = None,
    reload: bool = False,
    inspect: bool = False,
)
```

Starts the server in HTTP mode (default) or STDIO mode. All parameters fall back to SmartConfig auto-detection if `None`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `host` | `str \| None` | SmartConfig | Bind address |
| `port` | `int \| None` | SmartConfig | Bind port |
| `debug` | `bool \| None` | SmartConfig | Enable debug logging |
| `stdio` | `bool \| None` | Auto-detect | Force STDIO mode if `True` |
| `log_level` | `str` | `"warning"` | Log level: debug, info, warning, error, critical |
| `post_register_hook` | `Callable \| None` | `None` | Callback invoked after default endpoints are registered |
| `reload` | `bool` | `False` | Enable uvicorn hot reload (restarts on file changes) |
| `inspect` | `bool` | `False` | Open MCP Inspector in browser on server start |

```python
mcp.run_stdio(debug: bool | None = None, log_level: str = "warning")
```

Starts the server in STDIO mode explicitly. Logging is directed to stderr to keep stdout clean for JSON-RPC messages.

### Registration Methods

#### Decorator-Based (Instance)

```python
@mcp.tool
def my_tool(x: int) -> str: ...

@mcp.tool(name="custom_name", description="Custom description")
def my_tool(x: int) -> str: ...

@mcp.resource("config://settings")
def get_settings() -> dict: ...

@mcp.resource("file://readme", name="Readme", description="...", mime_type="text/markdown")
def get_readme() -> str: ...

@mcp.prompt
def code_review(code: str) -> str: ...

@mcp.prompt(name="custom_prompt", description="Custom prompt")
def my_prompt(topic: str) -> str: ...
```

#### Manual Registration

```python
mcp.add_tool(tool_handler: ToolHandler, **kwargs)
mcp.add_resource(resource_handler: ResourceHandler, **kwargs)
mcp.add_prompt(prompt_handler: PromptHandler, **kwargs)
mcp.add_endpoint(path: str, handler: Callable, methods: list[str] | None = None, **kwargs)
```

#### Function-to-Handler Registration

```python
mcp.register_function_as_tool(func, name=None, description=None) -> ToolHandler
mcp.register_function_as_resource(func, uri, name=None, description=None, mime_type="text/plain") -> ResourceHandler
mcp.register_function_as_prompt(func, name=None, description=None) -> PromptHandler
```

### Custom HTTP Endpoints

```python
@mcp.endpoint("/api/data", methods=["GET", "POST"])
async def data_handler(request: Request) -> Response:
    return Response('{"data": "example"}')
```

### Introspection

```python
mcp.get_tools() -> list[ToolHandler]
mcp.get_resources() -> list[ResourceHandler]
mcp.get_prompts() -> list[PromptHandler]
mcp.get_endpoints() -> list[dict]
mcp.info() -> dict[str, Any]
mcp.get_smart_config() -> dict
mcp.get_composition_stats() -> dict
```

### Cleanup

```python
mcp.clear_tools()
mcp.clear_resources()
mcp.clear_prompts()
mcp.clear_endpoints()
mcp.clear_all()
```

---

## Decorators

Standalone decorators register handlers on a global registry. When a `ChukMCPServer` is instantiated, it adopts all globally registered handlers and clears the global registry.

### `@tool`

```python
from chuk_mcp_server import tool

@tool
def hello(name: str) -> str:
    """Greet a user."""
    return f"Hello, {name}!"

@tool(name="add_numbers", description="Add two numbers")
def add(x: int, y: int = 0) -> int:
    return x + y

@tool(
    name="db.lookup",
    read_only_hint=True,
    idempotent_hint=True,
    output_schema={"type": "object", "properties": {"id": {"type": "string"}}},
    icons=[{"uri": "https://example.com/search.svg", "mimeType": "image/svg+xml"}],
)
def lookup(query: str) -> dict:
    """Lookup with annotations, structured output, and icons."""
    return {"id": "123"}
```

| Kwarg | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | `str \| None` | `func.__name__` | Tool name (1-128 chars, alphanumeric + `_`/`-`/`.`) |
| `description` | `str \| None` | `func.__doc__` | Tool description |
| `read_only_hint` | `bool \| None` | `None` | Hint: tool only reads data, no side effects (MCP 2025-03-26) |
| `destructive_hint` | `bool \| None` | `None` | Hint: tool may perform destructive operations (MCP 2025-03-26) |
| `idempotent_hint` | `bool \| None` | `None` | Hint: repeated calls with same args have same effect (MCP 2025-03-26) |
| `open_world_hint` | `bool \| None` | `None` | Hint: tool interacts with external systems (MCP 2025-03-26) |
| `output_schema` | `dict \| None` | `None` | JSON Schema for structured tool output (MCP 2025-06-18) |
| `icons` | `list[dict] \| None` | `None` | Icons for UI rendering; each dict has `uri` and `mimeType` (MCP 2025-11-25) |

Supports both `@tool` and `@tool()` syntax. Async functions are supported.

### `@resource`

```python
from chuk_mcp_server import resource

@resource("config://settings")
def get_settings() -> dict:
    return {"app": "my_app", "version": "1.0"}

@resource("file://readme", mime_type="text/markdown",
          icons=[{"uri": "https://example.com/doc.svg", "mimeType": "image/svg+xml"}])
def get_readme() -> str:
    return "# My Application"
```

| Kwarg | Type | Default | Description |
|-------|------|---------|-------------|
| `uri` | `str` | **required** | Resource URI (positional) |
| `name` | `str \| None` | Derived from `func.__name__` | Resource display name |
| `description` | `str \| None` | `func.__doc__` | Resource description |
| `mime_type` | `str` | `"text/plain"` | MIME type for the resource content |
| `icons` | `list[dict] \| None` | `None` | Icons for UI rendering (MCP 2025-11-25) |

### `@resource_template`

```python
from chuk_mcp_server import resource_template

@resource_template("users://{user_id}/profile")
def get_user_profile(user_id: str) -> dict:
    """Get a user's profile by ID."""
    return {"user_id": user_id, "name": "Example"}

@resource_template("files://{path}", mime_type="application/octet-stream",
                   icons=[{"uri": "https://example.com/file.svg", "mimeType": "image/svg+xml"}])
def get_file(path: str) -> str:
    return f"Contents of {path}"
```

| Kwarg | Type | Default | Description |
|-------|------|---------|-------------|
| `uri_template` | `str` | **required** | RFC 6570 URI template (positional) |
| `name` | `str \| None` | Derived from `func.__name__` | Template display name |
| `description` | `str \| None` | `func.__doc__` | Template description |
| `mime_type` | `str \| None` | `None` | MIME type for the resource content |
| `icons` | `list[dict] \| None` | `None` | Icons for UI rendering (MCP 2025-11-25) |

Discoverable via `resources/templates/list`. Template parameters are extracted from the URI template and passed as kwargs to the handler function.

### `@prompt`

```python
from chuk_mcp_server import prompt

@prompt
def code_review(code: str, language: str = "python") -> str:
    return f"Please review this {language} code:\n\n{code}"
```

| Kwarg | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | `str \| None` | `func.__name__` | Prompt name in MCP |
| `description` | `str \| None` | `func.__doc__` | Prompt description |

Supports both `@prompt` and `@prompt()` syntax.

### `@requires_auth`

```python
from chuk_mcp_server import requires_auth

@requires_auth(scopes=["drive.file"])
async def protected_tool(_external_access_token: str | None = None) -> str:
    # Use _external_access_token to call external API
    ...
```

| Kwarg | Type | Default | Description |
|-------|------|---------|-------------|
| `scopes` | `list[str] \| None` | `None` | Required OAuth scopes |

Marks a tool as requiring OAuth authorization. The protocol handler validates the token before execution and injects `_external_access_token` and `_user_id` into the tool arguments.

---

## Context API

The context API provides thread-safe, async-safe context storage for request-scoped data using Python `contextvars`. All functions are importable from `chuk_mcp_server` or `chuk_mcp_server.context`.

### Session

```python
get_session_id() -> str | None
set_session_id(session_id: str | None) -> None
require_session_id() -> str  # raises RuntimeError if not set
```

### User

```python
get_user_id() -> str | None
set_user_id(user_id: str | None) -> None
require_user_id() -> str  # raises PermissionError if not set
```

### Progress Token

```python
get_progress_token() -> str | int | None
set_progress_token(token: str | int | None) -> None
```

### Metadata

```python
get_metadata() -> dict[str, Any]         # returns a copy
set_metadata(metadata: dict[str, Any]) -> None
update_metadata(key: str, value: Any) -> None
clear_metadata() -> None
```

### HTTP Request

```python
get_http_request() -> Scope | None       # Starlette ASGI scope
```

### Sampling

```python
get_sampling_fn() -> Callable | None
set_sampling_fn(fn: Callable | None) -> None

async create_message(
    messages: list[Any],
    max_tokens: int = 1000,
    system_prompt: str | None = None,
    temperature: float | None = None,
    model_preferences: Any = None,
    stop_sequences: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    include_context: str | None = None,
) -> dict[str, Any]
```

See the [Sampling](#sampling) section for details.

### Elicitation

```python
get_elicitation_fn() -> Callable | None
set_elicitation_fn(fn: Callable | None) -> None

async create_elicitation(
    message: str,
    schema: dict[str, Any],
    title: str | None = None,
    description: str | None = None,
) -> dict[str, Any]
```

See the [Elicitation](#elicitation) section for details.

### Progress Notifications

```python
get_progress_notify_fn() -> Callable | None
set_progress_notify_fn(fn: Callable | None) -> None

async send_progress(
    progress: float,
    total: float | None = None,
    message: str | None = None,
) -> None
```

See the [Progress Notifications](#progress-notifications) section for details.

### Roots

```python
get_roots_fn() -> Callable | None
set_roots_fn(fn: Callable | None) -> None

async list_roots() -> list[dict[str, Any]]
```

See the [Roots](#roots) section for details.

### Logging Notifications

```python
get_log_fn() -> Callable | None
set_log_fn(fn: Callable | None) -> None

async send_log(
    level: str,
    data: Any,
    logger_name: str | None = None,
) -> None
```

Sends `notifications/message` to the client. Levels: `debug`, `info`, `notice`, `warning`, `error`, `critical`, `alert`, `emergency`. Silent no-op if unavailable.

### Resource Links

```python
get_resource_links() -> list[dict[str, Any]] | None
set_resource_links(links: list[dict[str, Any]] | None) -> None

add_resource_link(
    uri: str,
    name: str | None = None,
    description: str | None = None,
    mime_type: str | None = None,
) -> None
```

Accumulates `ResourceLink` references during tool execution. The protocol handler attaches them to the tool result as `_meta.links`.

### Utilities

```python
clear_all() -> None                      # reset all context variables
get_current_context() -> dict[str, Any]  # snapshot of all context values
```

### RequestContext (Async Context Manager)

```python
async with RequestContext(
    session_id="abc",
    user_id="user@example.com",
    progress_token="token-123",
    metadata={"key": "value"},
) as ctx:
    # Context is active here
    assert get_session_id() == "abc"
# Previous context is restored here
```

Supports nesting; inner contexts take precedence and outer contexts are restored on exit.

---

## MCP Protocol

### Supported Methods

| Method | Direction | Description |
|--------|-----------|-------------|
| `initialize` | client -> server | Initialize MCP session; returns server info and capabilities |
| `notifications/initialized` | client -> server | Client confirms session is ready (notification, no response) |
| `ping` | client -> server | Connectivity check; returns empty result |
| `tools/list` | client -> server | List all registered tools with schemas (supports pagination) |
| `tools/call` | client -> server | Execute a tool with arguments (supports structured output) |
| `resources/list` | client -> server | List all registered resources (supports pagination) |
| `resources/read` | client -> server | Read a resource by URI |
| `resources/subscribe` | client -> server | Subscribe to resource update notifications |
| `resources/unsubscribe` | client -> server | Unsubscribe from resource update notifications |
| `resources/templates/list` | client -> server | List resource templates (RFC 6570, supports pagination) |
| `prompts/list` | client -> server | List all registered prompts (supports pagination) |
| `prompts/get` | client -> server | Get a prompt with arguments |
| `logging/setLevel` | client -> server | Set server logging level |
| `completion/complete` | client -> server | Request argument auto-completion for resources or prompts |
| `sampling/createMessage` | server -> client | Request LLM sampling from the client (supports tool calling) |
| `elicitation/create` | server -> client | Request structured user input from the client (form mode) |
| `roots/list` | server -> client | Request client's filesystem roots |
| `tasks/get` | client -> server | Get task status by ID |
| `tasks/result` | client -> server | Get task result when completed |
| `tasks/list` | client -> server | List all tasks for current session |
| `tasks/cancel` | client -> server | Cancel a running task |
| `notifications/progress` | server -> client | Send progress update during tool execution |
| `notifications/message` | server -> client | Send log message notification to client |
| `notifications/cancelled` | bidirectional | Cancel an in-flight request |
| `notifications/tasks/status` | server -> client | Task status change notification |
| `notifications/roots/list_changed` | client -> server | Client's roots list has changed |
| `notifications/resources/updated` | server -> client | A subscribed resource has been updated |

### Protocol Versions

| Version | Status |
|---------|--------|
| `2025-11-25` | Default (latest) |
| `2025-06-18` | Supported |
| `2025-03-26` | Supported |

### JSON-RPC

All messages use JSON-RPC 2.0. The wire format is:

```json
{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "tools/call",
    "params": {
        "name": "hello",
        "arguments": {"name": "world"}
    }
}
```

### Error Codes

| Code | Name | Description |
|------|------|-------------|
| `-32700` | Parse Error | Invalid JSON |
| `-32600` | Invalid Request | Malformed JSON-RPC |
| `-32601` | Method Not Found | Unknown MCP method |
| `-32602` | Invalid Params | Invalid method parameters |
| `-32603` | Internal Error | Server-side error |
| `-32042` | URL Elicitation Required | Tool requires user interaction at an external URL (MCP 2025-11-25) |

### Session Management

Sessions are created on `initialize` and tracked by the `SessionManager`. Configuration:

- Maximum sessions: 1000 (evicts oldest when at capacity)
- Session expiry: 3600 seconds of inactivity
- Cleanup runs every 100 session creations
- Protected sessions: Sessions with active SSE streams skip eviction
- Eviction callback: Cleans up `_resource_subscriptions`, `_sse_event_buffers`, `_sse_event_counters`, and rate limiter buckets

Session IDs are returned in the `Mcp-Session-Id` HTTP header and must be included in subsequent requests.

### Request Validation

All incoming requests are validated at the transport and protocol layers:

| Limit | Value | Location |
|-------|-------|----------|
| Max request body size | 10 MB (`MAX_REQUEST_BODY_BYTES`) | HTTP endpoint, STDIO transport |
| Max argument keys | 100 (`MAX_ARGUMENT_KEYS`) | Protocol handler (`_handle_tools_call`) |
| Arguments type | Must be `dict` | Protocol handler (`_handle_tools_call`) |
| Max pending requests | 100 (`MAX_PENDING_REQUESTS`) | STDIO transport (`_send_and_receive`) |

### Rate Limiting

Per-session token bucket rate limiting is available via the `rate_limit_rps` parameter on `MCPProtocolHandler`. When enabled, each session gets an independent bucket that refills at the configured rate. Disabled by default (no overhead).

```python
MCPProtocolHandler(server_info, capabilities, rate_limit_rps=100.0)
# burst = rate_limit_rps * 2 = 200.0
```

### Graceful Shutdown

`MCPProtocolHandler.shutdown(timeout=5.0)` performs an orderly shutdown:

1. Waits up to `timeout` seconds for in-flight requests to complete
2. Cancels any remaining in-flight request tasks
3. Clears all sessions, task store, subscriptions, SSE state, and rate limiter

---

## Transport Modes

### HTTP Transport

The HTTP transport uses Starlette with Uvicorn and provides:

- Full CORS support (all origins, GET/POST/OPTIONS)
- `Mcp-Session-Id` header exposed to clients
- Preflight cache: 86400 seconds
- Context middleware for request lifecycle

**Performance Configuration:**

| Setting | Value |
|---------|-------|
| HTTP parser | httptools |
| Event loop | uvloop (non-Windows) |
| Backlog | 4096 |
| Concurrency limit | 2000 |
| Keep-alive timeout | 60s |
| Workers | 1 |

### STDIO Transport

Two STDIO implementations are provided:

**`StdioTransport`** (async): Uses `asyncio.StreamReader` for non-blocking stdin reads. Suitable for long-running async servers.

**`StdioSyncTransport`** (sync): Uses `sys.stdin.readline()` in a synchronous loop with `asyncio.run()` per message. Suitable for simpler integrations.

Both transports:

- Read newline-delimited JSON-RPC messages from stdin
- Write newline-delimited JSON-RPC responses to stdout
- Direct all logging to stderr
- Support bidirectional communication for server-to-client sampling requests

**STDIO Auto-Detection:**

STDIO mode is activated when any of these environment variables are set:
- `MCP_STDIO`
- `USE_STDIO`
- `MCP_TRANSPORT=stdio`

Or by passing `transport="stdio"` to the constructor, or `stdio=True` to `run()`.

---

## HTTP Endpoints

### Built-in Endpoints

| Path | Methods | Description |
|------|---------|-------------|
| `/mcp` | GET, POST, OPTIONS | Core MCP protocol endpoint (JSON-RPC over HTTP) |
| `/mcp/respond` | POST, OPTIONS | Client responses to server-initiated requests (bidirectional) |
| `/ping` | GET | Connectivity check; returns `{"status": "ok"}` |
| `/health` | GET | Liveness probe; returns `{"status": "healthy", "uptime": ...}` |
| `/health/ready` | GET | Readiness probe; returns 200 when tools are registered, 503 otherwise |
| `/health/detailed` | GET | Detailed health: tool/resource/prompt counts, sessions, in-flight requests |
| `/version` | GET | Server version information |
| `/` | GET | Server info (name, version, tools, resources, prompts) |
| `/info` | GET | Server info (alias for `/`) |
| `/docs` | GET | Documentation page |
| `/openapi.json` | GET | Auto-generated OpenAPI 3.1.0 specification |

### MCP Endpoint (`/mcp`)

- **GET**: Returns SSE stream for server-sent events
- **POST**: Accepts JSON-RPC requests; returns JSON-RPC responses
- **OPTIONS**: CORS preflight handling

The `Mcp-Session-Id` header is included in responses when a session is active.

### Custom Endpoints

Register custom HTTP endpoints using the decorator or manual method:

```python
@mcp.endpoint("/api/custom", methods=["GET", "POST"])
async def custom_handler(request: Request) -> Response:
    return Response('{"custom": true}', media_type="application/json")
```

---

## Type System

### ToolHandler

Wraps a Python function as an MCP tool with parameter validation, schema caching, annotations, and structured output.

```python
@dataclass
class ToolHandler:
    mcp_tool: MCPTool
    handler: Callable[..., Any]
    parameters: list[ToolParameter]
    requires_auth: bool = False
    auth_scopes: list[str] | None = None
    annotations: dict[str, bool] | None = None       # readOnlyHint, destructiveHint, etc.
    output_schema: dict[str, Any] | None = None       # JSON Schema for structured output
    icons: list[dict[str, Any]] | None = None          # Icons for UI rendering
```

**Key Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `from_function(func, name, description, read_only_hint, destructive_hint, idempotent_hint, open_world_hint, output_schema, icons)` | `ToolHandler` | Create from a Python function |
| `execute(arguments)` | `Any` | Validate arguments and execute the handler |
| `to_mcp_format()` | `dict` | Cached MCP tool schema as dict (includes annotations, outputSchema, icons) |
| `to_mcp_bytes()` | `bytes` | Cached orjson-serialized MCP schema |
| `invalidate_cache()` | `None` | Clear cached schema formats |

**Parameter Extraction:**

Parameters are extracted from the function signature. The `self` and `_external_access_token` parameters are excluded. Type annotations are mapped to JSON Schema types:

| Python Type | JSON Schema Type |
|-------------|-----------------|
| `str` | `string` |
| `int` | `integer` |
| `float` | `number` |
| `bool` | `boolean` |
| `list` / `list[T]` | `array` |
| `dict` | `object` |
| `Literal["a", "b"]` | `string` with `enum` |
| `Optional[T]` / `T \| None` | Type of `T`, not required |
| Pydantic `BaseModel` | `object` with full JSON Schema |
| `list[BaseModel]` | `array` with `items` schema |

### ResourceHandler

Wraps a Python function as an MCP resource with optional content caching and icons.

```python
@dataclass
class ResourceHandler:
    mcp_resource: MCPResource
    handler: Callable[..., Any]
    cache_ttl: int | None = None
    icons: list[dict[str, Any]] | None = None  # Icons for UI rendering
```

**Key Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `from_function(uri, func, name, description, mime_type, cache_ttl)` | `ResourceHandler` | Create from a function |
| `read()` | `str` | Read and format resource content |
| `to_mcp_format()` | `dict` | Cached MCP resource schema |
| `to_mcp_bytes()` | `bytes` | Cached orjson-serialized schema |
| `invalidate_cache()` | `None` | Clear cached content |
| `is_cached()` | `bool` | Check if content cache is valid |
| `get_cache_info()` | `dict` | Cache status details |

Content is formatted based on MIME type. Dicts and lists are serialized with orjson. Pydantic models are converted to dicts before serialization.

### PromptHandler

Wraps a Python function as an MCP prompt with argument validation.

```python
@dataclass
class PromptHandler:
    mcp_prompt: MCPPrompt
    handler: Callable[..., Any]
    parameters: list[ToolParameter]
```

**Key Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `from_function(func, name, description)` | `PromptHandler` | Create from a function |
| `get_prompt(arguments)` | `str \| dict` | Validate arguments and generate prompt |
| `to_mcp_format()` | `dict` | Cached MCP prompt schema |
| `to_mcp_bytes()` | `bytes` | Cached orjson-serialized schema |

### ToolParameter

Represents a single tool or prompt parameter with JSON Schema generation.

```python
@dataclass
class ToolParameter:
    name: str
    type: str                              # JSON Schema type
    description: str | None = None
    required: bool = True
    default: Any = None
    enum: list[Any] | None = None
    items_type: str | None = None          # For array: type of items
    pydantic_model: Any = None             # For object: Pydantic model class
    pydantic_items_model: Any = None       # For list[Model]: item model class
```

**Key Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `from_annotation(name, annotation, default)` | `ToolParameter` | Create from Python type annotation |
| `to_json_schema()` | `dict` | JSON Schema representation |
| `to_json_schema_bytes()` | `bytes` | Cached orjson-serialized schema |

---

## OAuth

OAuth support enables tools to access external APIs on behalf of authenticated users.

### Setup

```python
from chuk_mcp_server import ChukMCPServer, requires_auth
from chuk_mcp_server.oauth.helpers import setup_google_drive_oauth

mcp = ChukMCPServer("my-server")

@mcp.tool
@requires_auth(scopes=["drive.file"])
async def list_files(_external_access_token: str | None = None) -> dict:
    # Use _external_access_token to call Google Drive API
    ...

oauth_hook = setup_google_drive_oauth(mcp)
mcp.run(post_register_hook=oauth_hook)
```

### Flow

1. Tool is decorated with `@requires_auth(scopes=[...])`.
2. Client sends OAuth token in the request.
3. Protocol handler validates token via `oauth_provider_getter`.
4. External provider token is injected as `_external_access_token`.
5. User ID is injected as `_user_id` and set in context (`set_user_id()`).
6. Tool executes with access to the external API.

### Parameter Injection

When OAuth is configured and a tool requires auth, these parameters are automatically injected into the tool arguments:

| Parameter | Type | Description |
|-----------|------|-------------|
| `_external_access_token` | `str` | External provider's access token |
| `_user_id` | `str` | Authenticated user's identifier |

Both parameters must be declared in the tool function signature with defaults of `None`.

### OAuth Endpoints

When OAuth middleware is registered, the following additional endpoints become available:

| Path | Description |
|------|-------------|
| `/.well-known/oauth-authorization-server` | Authorization server metadata discovery |
| `/authorize` | Authorization endpoint |
| `/token` | Token endpoint |
| `/oauth/callback` | OAuth callback handler |

### Environment Variables

| Variable | Description |
|----------|-------------|
| `GOOGLE_CLIENT_ID` | Google OAuth client ID |
| `GOOGLE_CLIENT_SECRET` | Google OAuth client secret |
| `GOOGLE_REDIRECT_URI` | OAuth callback URL (default: `http://localhost:8000/oauth/callback`) |
| `OAUTH_SERVER_URL` | OAuth server base URL (default: `http://localhost:8000`) |

---

## Sampling

Sampling enables the server to request LLM completions from the connected MCP client. This is a server-to-client request flow.

### Usage in Tools

```python
from chuk_mcp_server import tool, create_message

@tool
async def smart_tool(question: str) -> str:
    response = await create_message(
        messages=[
            {"role": "user", "content": {"type": "text", "text": question}}
        ],
        max_tokens=500,
        system_prompt="You are a helpful assistant.",
        temperature=0.7,
    )
    return response["content"]["text"]
```

### `create_message()` Parameters

```python
async def create_message(
    messages: list[Any],
    max_tokens: int = 1000,
    system_prompt: str | None = None,
    temperature: float | None = None,
    model_preferences: Any = None,
    stop_sequences: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    include_context: str | None = None,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: dict[str, Any] | str | None = None,
) -> dict[str, Any]
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `messages` | `list[Any]` | **required** | List of sampling messages with `role` and `content` |
| `max_tokens` | `int` | `1000` | Maximum tokens for the LLM response |
| `system_prompt` | `str \| None` | `None` | System prompt for the LLM |
| `temperature` | `float \| None` | `None` | Sampling temperature |
| `model_preferences` | `Any` | `None` | Model selection preferences |
| `stop_sequences` | `list[str] \| None` | `None` | Stop sequences |
| `metadata` | `dict \| None` | `None` | Request metadata |
| `include_context` | `str \| None` | `None` | Context inclusion: `"none"`, `"thisServer"`, or `"allServers"` |
| `tools` | `list[dict] \| None` | `None` | Tools the client's LLM can call during sampling (MCP 2025-11-25) |
| `tool_choice` | `dict \| str \| None` | `None` | Tool selection policy: `"auto"`, `"none"`, `"required"`, or `{"type": "tool", "name": "..."}` |

**Returns:** Dict with `role`, `content`, `model`, and `stopReason` from the client's LLM.

**Raises:** `RuntimeError` if sampling is not available (client does not support it or the tool is not running within an MCP request context).

### Requirements

- The client must declare `sampling` in its capabilities during `initialize`.
- The transport must support bidirectional communication. Both transports support this:
  - **STDIO**: Uses stdin/stdout with pending request futures
  - **HTTP**: Uses SSE `server_request` events; client responds via `POST /mcp/respond`
- `create_message()` is only callable within a tool execution context.

### Timeout

Sampling requests have a 120-second timeout. If the client does not respond within this window, a `RuntimeError` is raised.

---

## Elicitation

Elicitation enables the server to request structured user input from the connected MCP client. Two modes are supported: **form mode** (structured input) and **URL mode** (external URL redirect).

### Form Mode

```python
from chuk_mcp_server import tool, create_elicitation

@tool
async def configure_analysis(dataset: str) -> str:
    result = await create_elicitation(
        message="How would you like to analyze this dataset?",
        schema={
            "type": "object",
            "properties": {
                "method": {"type": "string", "enum": ["regression", "classification", "clustering"],
                           "default": "regression"},
                "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0, "default": 0.95},
            },
            "required": ["method"],
        },
        title="Analysis Configuration",
    )
    return f"Running {result['method']} analysis on {dataset}"
```

### URL Mode (MCP 2025-11-25)

URL mode directs users to external URLs for sensitive interactions (OAuth flows, payment, credentials). The tool raises `URLElicitationRequiredError` which the protocol handler converts to a JSON-RPC error with code `-32042`.

```python
from chuk_mcp_server import tool
from chuk_mcp_server.types.errors import URLElicitationRequiredError

@tool
async def connect_account(service: str) -> str:
    raise URLElicitationRequiredError(
        url=f"https://auth.example.com/connect/{service}",
        description=f"Please authorize access to your {service} account",
    )
```

### `create_elicitation()` Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `message` | `str` | **required** | Message to display to the user |
| `schema` | `dict[str, Any]` | **required** | JSON Schema with optional `default` values on primitives |
| `title` | `str \| None` | `None` | Optional title for the elicitation dialog |
| `description` | `str \| None` | `None` | Optional description for the dialog |

**Returns:** Dict with the user's structured input matching the provided schema.

**Raises:** `RuntimeError` if elicitation is not available (client does not support it).

### Requirements

- The client must declare `elicitation` in its capabilities during `initialize`.
- The transport must support bidirectional communication.
- `create_elicitation()` is only callable within a tool execution context.
- Schema properties support `default` values on primitives (MCP 2025-11-25).

---

## Progress Notifications

Progress notifications allow long-running tools to report progress to the client. Progress is optional and fire-and-forget.

### Usage in Tools

```python
from chuk_mcp_server import tool, send_progress

@tool
async def process_large_dataset(path: str) -> str:
    total_steps = 100
    for step in range(total_steps):
        # Do work...
        await send_progress(
            progress=step + 1,
            total=total_steps,
            message=f"Processing chunk {step + 1}/{total_steps}",
        )
    return f"Processed {total_steps} chunks from {path}"
```

### `send_progress()` Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `progress` | `float` | **required** | Current progress value |
| `total` | `float \| None` | `None` | Total expected value (for percentage calculation) |
| `message` | `str \| None` | `None` | Human-readable progress description |

**Returns:** `None`. This is a fire-and-forget notification.

**Behavior:** `send_progress()` is a silent no-op if:
- No progress notification function is available (client doesn't support it)
- No progress token was set for the current request (client didn't request progress)

### Requirements

- The client must include a `progressToken` in the tool call's `_meta` field.
- The transport must support bidirectional communication.
- Progress notifications have no `id` field (they are JSON-RPC notifications, not requests).

---

## Logging Notifications

Logging notifications allow tools to emit structured log messages to the client. Logging is fire-and-forget, like progress.

### Usage in Tools

```python
from chuk_mcp_server import tool, send_log

@tool
async def process_data(input: str) -> str:
    await send_log("info", "Starting data processing", logger_name="processor")
    # Do work...
    await send_log("warning", {"message": "Skipped 3 invalid rows", "count": 3})
    return "Done"
```

### `send_log()` Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `level` | `str` | **required** | Log level: `debug`, `info`, `notice`, `warning`, `error`, `critical`, `alert`, `emergency` |
| `data` | `Any` | **required** | Log data (string or structured dict) |
| `logger_name` | `str \| None` | `None` | Optional logger name for categorization |

**Returns:** `None`. This is a fire-and-forget notification.

**Behavior:** `send_log()` is a silent no-op if no log function is available.

### Wire Format

```json
{
    "jsonrpc": "2.0",
    "method": "notifications/message",
    "params": {
        "level": "info",
        "data": "Starting data processing",
        "logger": "processor"
    }
}
```

---

## Roots

Roots support allows the server to discover the client's filesystem roots. This enables resource scoping and path resolution.

### Usage in Tools

```python
from chuk_mcp_server import tool, list_roots

@tool
async def scan_project() -> str:
    roots = await list_roots()
    paths = [root["uri"] for root in roots]
    return f"Client roots: {', '.join(paths)}"
```

### `list_roots()` Return Value

Returns a list of root objects, each with:

| Field | Type | Description |
|-------|------|-------------|
| `uri` | `str` | Root URI (e.g., `file:///home/user/project`) |
| `name` | `str \| None` | Optional human-readable name |

**Raises:** `RuntimeError` if roots is not available (client does not support it).

### Notifications

The server handles `notifications/roots/list_changed` from the client when the client's roots change.

### Requirements

- The client must declare `roots` in its capabilities during `initialize`.
- The transport must support bidirectional communication.
- `list_roots()` is only callable within a tool execution context.

---

## Resource Subscriptions

Resource subscriptions allow clients to subscribe to resource URIs and receive notifications when those resources are updated.

### Client Subscribes

Clients send `resources/subscribe` and `resources/unsubscribe` requests:

```json
{"jsonrpc": "2.0", "id": 1, "method": "resources/subscribe", "params": {"uri": "config://settings"}}
{"jsonrpc": "2.0", "id": 2, "method": "resources/unsubscribe", "params": {"uri": "config://settings"}}
```

Both return an empty result `{}` on success.

### Server Notifies

When a resource changes, the server sends a notification to all subscribed sessions:

```python
await handler.notify_resource_updated("config://settings")
```

This sends a JSON-RPC notification (no `id`):

```json
{"jsonrpc": "2.0", "method": "notifications/resources/updated", "params": {"uri": "config://settings"}}
```

### Subscription Tracking

Subscriptions are tracked per-session in `MCPProtocolHandler._resource_subscriptions` (a dict mapping session ID to a set of subscribed URIs).

- Subscribing to a non-existent resource is allowed (per MCP spec).
- Unsubscribing when not subscribed is a no-op (no error).
- The `resources` capability must include `subscribe: true` (this is the default).

---

## Completions

Completions provide argument auto-completion for resources and prompts.

### Server Setup

Register completion providers on the protocol handler:

```python
handler = MCPProtocolHandler(server_info, capabilities)

async def resource_completer(ref, argument):
    value = argument.get("value", "")
    all_uris = ["config://app", "config://db", "config://cache"]
    matches = [u for u in all_uris if u.startswith(value)]
    return {"values": matches, "hasMore": False}

handler.register_completion_provider("ref/resource", resource_completer)
```

### Client Requests

Clients send `completion/complete` requests:

```json
{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "completion/complete",
    "params": {
        "ref": {"type": "ref/resource", "uri": "config://app"},
        "argument": {"name": "uri", "value": "config://"}
    }
}
```

### Response Format

```json
{
    "jsonrpc": "2.0",
    "id": 1,
    "result": {
        "completion": {
            "values": ["config://app", "config://db", "config://cache"],
            "hasMore": false
        }
    }
}
```

### Completion Provider Interface

```python
async def provider(ref: dict, argument: dict) -> dict:
    """
    Args:
        ref: Reference object with "type" and "uri" or "name"
        argument: Argument with "name" and "value" (partial input)

    Returns:
        {"values": list[str], "hasMore": bool}
    """
```

### Requirements

- Enable completions: `ChukMCPServer(completions=True)` or `create_server_capabilities(completions=True)`.
- Register providers by ref type: `"ref/resource"` or `"ref/prompt"`.
- If no provider is registered for a ref type, an empty result is returned.

---

## Tasks

The tasks system (MCP 2025-11-25, experimental) provides durable state machines for long-running requests. Instead of keeping a connection open, `tools/call` can return a task ID that clients poll for progress and results.

### Task Lifecycle

```
tools/call (long-running)
  -> Returns task ID with status "working"
  -> Client polls tasks/get for status updates
  -> Server sends notifications/tasks/status on state changes
  -> Client calls tasks/result when status is "completed"
  -> Client can call tasks/cancel to abort
```

### Task Status Values

| Status | Description |
|--------|-------------|
| `working` | Task is in progress |
| `completed` | Task finished successfully; result available via `tasks/result` |
| `failed` | Task failed; error details in status notification |
| `cancelled` | Task was cancelled by the client |

### Protocol Methods

**`tasks/get`** -- Get task status

```json
{"jsonrpc": "2.0", "id": 1, "method": "tasks/get", "params": {"taskId": "abc-123"}}
```

**`tasks/result`** -- Get completed task result

```json
{"jsonrpc": "2.0", "id": 2, "method": "tasks/result", "params": {"taskId": "abc-123"}}
```

**`tasks/list`** -- List all tasks for current session

```json
{"jsonrpc": "2.0", "id": 3, "method": "tasks/list", "params": {}}
```

**`tasks/cancel`** -- Cancel a running task

```json
{"jsonrpc": "2.0", "id": 4, "method": "tasks/cancel", "params": {"taskId": "abc-123"}}
```

### Status Notifications

```json
{
    "jsonrpc": "2.0",
    "method": "notifications/tasks/status",
    "params": {
        "taskId": "abc-123",
        "status": "completed"
    }
}
```

---

## Artifacts

Artifact support provides blob and workspace storage backed by `chuk-artifacts`. This is an optional dependency.

### Installation

```bash
pip install 'chuk-mcp-server[artifacts]'
```

### Store Management

```python
from chuk_mcp_server import (
    get_artifact_store,
    set_artifact_store,
    set_global_artifact_store,
    has_artifact_store,
    clear_artifact_store,
)

# Set store for current context (request-scoped)
set_artifact_store(store)

# Set global fallback store
set_global_artifact_store(store)

# Check availability
if has_artifact_store():
    store = get_artifact_store()
```

Resolution order: context variable first, then global singleton. Raises `RuntimeError` if neither is set.

### Blob Namespace

```python
from chuk_mcp_server import (
    create_blob_namespace,
    write_blob,
    read_blob,
    StorageScope,
)

# Create namespace
ns = await create_blob_namespace(
    scope=StorageScope.SESSION,
    session_id="optional-session-id",
)

# Write and read
await write_blob(ns.namespace_id, b"Hello World", mime="text/plain")
data = await read_blob(ns.namespace_id)
```

### Workspace Namespace

```python
from chuk_mcp_server import (
    create_workspace_namespace,
    write_workspace_file,
    read_workspace_file,
    get_namespace_vfs,
)

# Create workspace
ws = await create_workspace_namespace(
    name="my-project",
    scope=StorageScope.USER,
    user_id="alice",
    provider_type="vfs-memory",
)

# File operations
await write_workspace_file(ws.namespace_id, "/main.py", b"print('hello')")
data = await read_workspace_file(ws.namespace_id, "/main.py")

# Direct VFS access
vfs = get_namespace_vfs(ws.namespace_id)
```

### `create_blob_namespace()` Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `scope` | `StorageScope \| None` | `SESSION` | Storage scope |
| `session_id` | `str \| None` | Auto-allocated | Session identifier |
| `user_id` | `str \| None` | `None` | User ID (required for `USER` scope) |

### `create_workspace_namespace()` Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | **required** | Workspace name |
| `scope` | `StorageScope \| None` | `SESSION` | Storage scope |
| `session_id` | `str \| None` | Auto-allocated | Session identifier |
| `user_id` | `str \| None` | `None` | User ID (required for `USER` scope) |
| `provider_type` | `str` | `"vfs-memory"` | VFS provider: `vfs-memory`, `vfs-filesystem`, `vfs-s3`, `vfs-sqlite` |

### Storage Scopes

| Scope | Description |
|-------|-------------|
| `StorageScope.SESSION` | Scoped to a single MCP session |
| `StorageScope.USER` | Scoped to an authenticated user |
| `StorageScope.SANDBOX` | Scoped to a sandbox environment |

### Namespace Types

| Type | Description |
|------|-------------|
| `NamespaceType.BLOB` | Single binary object storage |
| `NamespaceType.WORKSPACE` | VFS-backed file tree |

---

## Server Composition

Compose multiple MCP servers into a single unified server.

### Import (Static Composition)

Creates a one-time copy of components. Changes to the source server after import are not reflected.

```python
main_mcp.import_server(
    server: ChukMCPServer,
    prefix: str | None = None,
    components: list[str] | None = None,  # ["tools", "resources", "prompts"]
    tags: list[str] | None = None,
)
```

```python
# Import all components with a prefix
main_mcp.import_server(weather_server, prefix="weather")
# Result: tools named "weather.get_forecast", "weather.get_current", etc.

# Import only tools
main_mcp.import_server(api_server, prefix="api", components=["tools"])
```

### Mount (Dynamic Composition)

Creates a live link to another server. Changes to the mounted server are reflected immediately.

```python
main_mcp.mount(
    server: ChukMCPServer | dict[str, Any],
    prefix: str | None = None,
    as_proxy: bool = False,
)
```

```python
# Mount local server (live updates)
main_mcp.mount(api_server)

# Mount remote server as proxy
main_mcp.mount(remote_config, prefix="remote", as_proxy=True)
```

### Load Module

Load Python modules containing tools dynamically.

```python
mcp.load_module({
    "math": {
        "enabled": True,
        "location": "./modules",
        "module": "math_tools.tools",
        "namespace": "math"
    }
}) -> dict[str, list[str]]  # module name -> loaded tool names
```

### Composition Statistics

```python
mcp.get_composition_stats() -> dict[str, Any]
```

---

## Configuration

### Smart Configuration

`ChukMCPServer` uses a modular `SmartConfig` system that auto-detects the runtime environment and sets optimal defaults.

**Detected Parameters:**

| Parameter | Description | Example |
|-----------|-------------|---------|
| `project_name` | Inferred from directory or package | `"my-server"` |
| `host` | Bind address | `"0.0.0.0"` |
| `port` | Bind port | `8000` |
| `debug` | Debug mode | `False` |
| `workers` | Worker processes | `1` |
| `max_connections` | Connection limit | `2000` |
| `log_level` | Logging level | `"warning"` |
| `performance_mode` | Performance tier | `"standard"` |
| `containerized` | Running in container | `False` |
| `environment` | Detected environment | `"development"` |
| `transport_mode` | Transport type | `"http"` |

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MCP_LOG_LEVEL` | Override logging level | `"warning"` |
| `MCP_TRANSPORT` | Force transport mode | Auto-detect |
| `MCP_STDIO` | Force STDIO mode | Unset |
| `USE_STDIO` | Force STDIO mode (alias) | Unset |
| `MCP_SERVER_NAME` | Override server name | Auto-detect |
| `MCP_SERVER_VERSION` | Override server version | `"1.0.0"` |
| `PORT` | Override bind port | Auto-detect |

### Configuration Methods

```python
mcp.get_smart_config() -> dict[str, Any]          # Current configuration
mcp.get_smart_config_summary() -> dict[str, Any]  # Detection summary
mcp.refresh_smart_config()                          # Re-detect and update
```

### Cloud Detection

The framework auto-detects cloud environments and creates appropriate handlers:

| Environment | Detection | Handler Name |
|-------------|-----------|--------------|
| Google Cloud Functions | `K_SERVICE`, `FUNCTION_TARGET` | `mcp_gcf_handler` |
| AWS Lambda | `AWS_LAMBDA_FUNCTION_NAME` | `lambda_handler` |
| Azure Functions | `FUNCTIONS_WORKER_RUNTIME` | `main` |
| Vercel / Netlify / Cloudflare | Various env vars | `handler` |
| Docker / Kubernetes | `/.dockerenv`, `KUBERNETES_SERVICE_HOST` | N/A (HTTP mode) |

Cloud detection helpers:

```python
from chuk_mcp_server import is_cloud, is_gcf, is_lambda, is_azure, get_deployment_info
```

### Proxy Configuration

Configure multi-server proxy mode to aggregate tools from remote MCP servers:

```python
mcp = ChukMCPServer(proxy_config={
    "servers": {
        "time": {
            "url": "http://time-server:8001/mcp",
            "transport": "streamable-http"
        },
        "weather": {
            "url": "http://weather-server:8002/mcp",
            "transport": "streamable-http"
        }
    }
})
```

Proxied tools are namespaced as `proxy.<server>.<tool_name>`.

### Tool Modules Configuration

Load external tool modules at startup:

```python
mcp = ChukMCPServer(tool_modules_config={
    "math": {
        "enabled": True,
        "location": "./modules",
        "module": "math_tools.tools",
        "namespace": "math"
    }
})
```

---

## Testing

### ToolRunner

`ToolRunner` provides a lightweight test harness for invoking MCP tools without a transport layer. It is exported from `chuk_mcp_server`.

```python
from chuk_mcp_server import ToolRunner

runner = ToolRunner()

# Call a tool and get the raw JSON-RPC response
result = await runner.call_tool("add", {"a": 1, "b": 2})

# Call a tool and get the text content directly
text = await runner.call_tool_text("greet", {"name": "World"})

# List all registered tools
tools = await runner.list_tools()

# Get just tool names
names = await runner.list_tool_names()
```

#### Constructor

```python
ToolRunner(server: ChukMCPServer | None = None)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `server` | `ChukMCPServer \| None` | `None` | Server instance; creates a temporary one if `None` |

When no server is provided, `ToolRunner` creates a `ChukMCPServer` that adopts all globally registered tools.

#### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `call_tool(name, arguments)` | `dict` | Call a tool and return the raw JSON-RPC response |
| `call_tool_text(name, arguments)` | `str` | Call a tool and return the text content |
| `list_tools()` | `list[dict]` | List all registered tools with schemas |
| `list_tool_names()` | `list[str]` | List just the tool names |

---

## OpenAPI

An OpenAPI 3.1.0 specification is auto-generated from registered tool schemas and served at `/openapi.json`.

Each registered tool becomes a `POST /tools/{name}` operation with:
- `operationId` matching the tool name
- `summary` from the tool description
- `requestBody` schema derived from the tool's input schema
- Standard `200`, `400`, `500` response definitions

```python
from chuk_mcp_server.openapi import generate_openapi_spec

spec = generate_openapi_spec(protocol)
# Returns dict with "openapi", "info", "paths"
```

---

## CLI

### HTTP Subcommand Flags

```bash
chuk-mcp-server http [--host HOST] [--port PORT] [--debug] [--reload] [--inspect] [--log-level LEVEL]
```

| Flag | Description |
|------|-------------|
| `--host` | Bind address |
| `--port` | Bind port |
| `--debug` | Enable debug logging |
| `--reload` | Enable hot reload (auto-restart on file changes) |
| `--inspect` | Open MCP Inspector in browser on server start |
| `--log-level` | Logging level (debug, info, warning, error, critical) |

---

## Public Exports

All public symbols are available from the top-level `chuk_mcp_server` package:

```python
from chuk_mcp_server import (
    # Server
    ChukMCPServer,
    create_mcp_server,
    quick_server,
    get_mcp_server,
    run,

    # Decorators
    tool,
    resource,
    resource_template,
    prompt,
    requires_auth,

    # Context
    RequestContext,
    get_session_id, set_session_id, require_session_id,
    get_user_id, set_user_id, require_user_id,
    create_message, get_sampling_fn, set_sampling_fn,
    create_elicitation, get_elicitation_fn, set_elicitation_fn,
    send_progress, get_progress_notify_fn, set_progress_notify_fn,
    send_log,                                    # Log notifications (MCP 2025-06-18)
    add_resource_link,                           # Resource links in tool results (MCP 2025-06-18)
    list_roots, get_roots_fn, set_roots_fn,

    # Types
    Tool,         # alias for ToolHandler
    Resource,     # alias for ResourceHandler
    Prompt,       # alias for PromptHandler
    ToolParameter,
    MCPPrompt,
    ServerInfo,

    # Errors
    URLElicitationRequiredError,                 # URL mode elicitation (MCP 2025-11-25)

    # Cloud
    is_cloud, is_gcf, is_lambda, is_azure,
    get_deployment_info,

    # Artifacts (requires chuk-artifacts)
    get_artifact_store, set_artifact_store,
    set_global_artifact_store, has_artifact_store,
    create_blob_namespace, create_workspace_namespace,
    write_blob, read_blob,
    write_workspace_file, read_workspace_file,
    get_namespace_vfs,
    NamespaceType, StorageScope, NamespaceInfo,

    # Composition
    ProxyManager,
    ModuleLoader,

    # Testing
    ToolRunner,
)
```
