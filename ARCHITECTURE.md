# chuk-mcp-server Architecture

**Version**: 0.21.0 | **Python**: >=3.11 | **License**: MIT
**Tests**: 2330+ passed, 97% coverage | **Performance**: 36K+ RPS, <3ms overhead

## Overview

chuk-mcp-server is a zero-configuration MCP (Model Context Protocol) server
framework for Python. It provides decorator-based tool, resource, and prompt
registration; automatic cloud and environment detection; Streamable HTTP and
STDIO transports; server composition and proxy capabilities; and full
conformance with MCP specification 2025-11-25 (latest).

The framework is designed around three principles: no configuration required
for common cases, no unnecessary abstraction layers, and deploy-anywhere
without code changes.

```
+--------------------------------------------------------------------+
|                         chuk-mcp-server                            |
|                                                                    |
|  @tool / @resource / @prompt           run() / ChukMCPServer()    |
|  @resource_template                    (core.py)                  |
|  (decorators.py)                             |                     |
|         |                                    v                     |
|  +-----------------+     +--------------------------------------+ |
|  | Global Registry |---->| MCPProtocolHandler (JSON-RPC 2.0)    | |
|  | (mcp_registry)  |     | - Tool dispatch + structured output   | |
|  +-----------------+     | - Resource dispatch + templates        | |
|                          | - Prompt dispatch + pagination         | |
|                          | - Sampling / Elicitation / Roots       | |
|                          | - Progress / Subscriptions / Complete  | |
|                          | - Tasks / Cancellation / Logging       | |
|                          +--------------------------------------+ |
|                                    |              |                |
|                          +---------+----+  +-----+--------+       |
|                          | Streamable   |  | STDIO Transport|      |
|                          | HTTP Server  |  | (sync/async)   |      |
|                          | (Starlette)  |  |                |      |
|                          +--------------+  +--------------+       |
+--------------------------------------------------------------------+
```

## Directory Structure

```
src/chuk_mcp_server/
|-- __init__.py              # Exports, cloud auto-detection
|-- core.py                  # ChukMCPServer main class
|-- decorators.py            # @tool, @resource, @resource_template, @prompt, @requires_auth
|-- protocol.py              # MCPProtocolHandler (JSON-RPC 2.0, tasks, pagination)
|-- context.py               # Request context (contextvars, send_log, add_resource_link)
|-- constants.py             # Protocol constants, tool name regex, validation limits
|-- http_server.py           # Starlette/Uvicorn HTTP server (Streamable HTTP)
|-- stdio_transport.py       # STDIO transport (sync + async)
|-- rate_limiter.py          # Per-session token bucket rate limiter
|-- telemetry.py             # Thin OpenTelemetry wrapper (no-op without otel)
|-- artifacts_context.py     # Optional artifact/workspace support
|-- cli.py                   # CLI (scaffolding, --reload, --inspect)
|-- errors.py                # Structured error messages
|-- openapi.py               # OpenAPI 3.1.0 spec generation
|-- testing.py               # ToolRunner test harness
|-- mcp_registry.py          # Component registry
|-- endpoint_registry.py     # HTTP endpoint registry
|
|-- types/                   # Type system
|   |-- base.py              # Direct chuk_mcp types (no conversion)
|   |-- tools.py             # ToolHandler (orjson caching, annotations, structured output, icons)
|   |-- resources.py         # ResourceHandler, ResourceTemplateHandler (icons)
|   |-- prompts.py           # PromptHandler (icons)
|   |-- parameters.py        # ToolParameter + JSON Schema
|   |-- capabilities.py      # ServerCapabilities
|   |-- content.py           # Content types, annotations, resource links
|   |-- serialization.py     # Serialization utils
|   +-- errors.py            # Error types (URLElicitationRequiredError)
|
|-- config/                  # Smart configuration
|   |-- smart_config.py      # SmartConfig orchestrator
|   |-- cloud_detector.py    # GCP/AWS/Azure detection
|   |-- container_detector.py # Docker/K8s detection
|   |-- environment_detector.py # Dev/prod/serverless
|   |-- network_detector.py  # Host/port detection
|   |-- project_detector.py  # Project name detection
|   +-- system_detector.py   # Workers, performance
|
|-- cloud/                   # Cloud provider support
|   |-- providers/           # GCP, AWS, Azure, Edge
|   +-- adapters/            # Platform-specific adapters
|
|-- endpoints/               # HTTP endpoints
|   |-- mcp.py               # Core /mcp endpoint with SSE
|   |-- health.py            # /health, /health/ready, /health/detailed
|   |-- ping.py, info.py, version.py
|   +-- constants.py
|
|-- oauth/                   # OAuth 2.1 support
|   |-- base_provider.py     # Abstract OAuth provider
|   |-- middleware.py         # OAuth middleware
|   |-- token_store.py       # Token storage
|   +-- providers/google_drive.py
|
|-- proxy/                   # Multi-server proxy
|   |-- manager.py           # ProxyManager (chuk-tool-processor)
|   +-- mcp_tool_wrapper.py  # Tool wrapper
|
|-- composition/             # Server composition
|   |-- manager.py           # import_server / mount
|   +-- config_loader.py     # YAML config loader
|
|-- modules/                 # Module loading
|   +-- loader.py            # Dynamic tool loading
|
+-- middlewares/              # HTTP middleware
    +-- context_middleware.py # Context injection
```

## Request Flow

### HTTP Transport

```
HTTP POST /mcp
  |
  v
Starlette Router
  |
  v
ContextMiddleware
  - Extracts session ID from headers / query params
  - Extracts user context (OAuth token, user ID)
  - Sets contextvars for the request scope
  |
  v
MCPEndpoint.handle_request()
  - Parses JSON-RPC envelope
  - Routes to MCPProtocolHandler
  |
  v
MCPProtocolHandler.handle_request()
  - Validates JSON-RPC 2.0 structure
  - Dispatches by method:
      "initialize"              -> _handle_initialize()
      "tools/list"              -> _handle_tools_list()       (with pagination)
      "tools/call"              -> _handle_tools_call()       (structured output)
      "resources/list"          -> _handle_resources_list()   (with pagination)
      "resources/read"          -> _handle_resources_read()
      "resources/subscribe"     -> _handle_resources_subscribe()
      "resources/unsubscribe"   -> _handle_resources_unsubscribe()
      "resources/templates/list"-> _handle_resource_templates_list()
      "prompts/list"            -> _handle_prompts_list()     (with pagination)
      "prompts/get"             -> _handle_prompts_get()
      "completion/complete"     -> _handle_completion_complete()
      "tasks/get"               -> _handle_tasks_get()
      "tasks/result"            -> _handle_tasks_result()
      "tasks/list"              -> _handle_tasks_list()
      "tasks/cancel"            -> _handle_tasks_cancel()
      "notifications/cancelled" -> _handle_cancelled()
      "notifications/*"         -> _handle_notification()
  |
  v
Handler Execution
  - ToolHandler.execute(name, arguments)
  - ResourceHandler.read(uri)
  - PromptHandler.get_prompt(name, arguments)
  |
  v
format_content() -> orjson.dumps()
  |
  v
Response (JSON or SSE stream)
```

### STDIO Transport

```
stdin (one JSON-RPC line per message)
  |
  v
StdioSyncTransport.run() / StdioTransport.start()
  - Reads lines from stdin
  - Parses JSON-RPC
  |
  v
MCPProtocolHandler.handle_request()
  - Same dispatch as HTTP
  |
  v
stdout (one JSON-RPC response line per message)
```

The STDIO transport supports both synchronous (StdioSyncTransport) and
asynchronous (StdioTransport) modes. The synchronous mode uses readline
on stdin with the async protocol handler, making it compatible with
subprocess-based MCP clients.

## Type System

The type system in `types/` avoids a conversion layer between internal
representations and the MCP protocol. Instead, it uses chuk_mcp types
directly wherever possible.

```
types/
  base.py         Re-exports from chuk_mcp (ServerInfo, etc.)
                  No wrapper classes, no conversion overhead.

  tools.py        ToolHandler
                  - Wraps a callable with metadata (name, description, schema)
                  - Caches both dict representation and orjson bytes
                  - Schema is computed once from function signature
                  - Supports annotations (readOnlyHint, destructiveHint, etc.)
                  - Supports outputSchema for structured tool output
                  - Supports icons for UI rendering

  resources.py    ResourceHandler + ResourceTemplateHandler
                  - Wraps a callable with URI pattern
                  - Supports content caching with TTL
                  - ResourceTemplateHandler for RFC 6570 URI templates
                  - Both support icons for UI rendering

  prompts.py      PromptHandler
                  - Wraps a callable with prompt metadata
                  - Arguments extracted from function signature
                  - Supports icons for UI rendering

  parameters.py   ToolParameter + JSON Schema generation
                  - Converts Python type hints to JSON Schema
                  - Supports str, int, float, bool, list, dict, Optional
                  - Handles default values and descriptions

  content.py      Content types (TextContent, ImageContent, etc.)
                  - format_content() normalizes tool return values
                  - Content annotations (audience, priority)
                  - Resource link creation

  errors.py       Error types
                  - URLElicitationRequiredError for URL mode elicitation
                  - ParameterValidationError, ToolExecutionError

  serialization.py  orjson-based serialization utilities
```

### Handler Caching Strategy

```
Registration time:
  function + metadata -> ToolHandler
                         |-- .to_dict()  -> cached dict    (for tools/list)
                         +-- .to_bytes() -> cached orjson   (for wire format)

Call time:
  arguments -> function(**args)
            -> format_content(result)
            -> orjson.dumps(response)
```

Each ToolHandler computes its JSON schema and serialized form once at
registration. Subsequent `tools/list` calls return the pre-computed bytes
directly without re-serialization.

## Context System

Request context uses Python `contextvars` for async-safe, per-request state.
This avoids passing context objects through every function signature.

```
contextvars (context.py):
  _session_id        : ContextVar[str | None]
  _user_id           : ContextVar[str | None]
  _progress_token    : ContextVar[str | int | None]
  _metadata          : ContextVar[dict | None]
  _http_request      : ContextVar[Scope | None]
  _sampling_fn       : ContextVar[Callable | None]
  _elicitation_fn    : ContextVar[Callable | None]
  _progress_notify_fn: ContextVar[Callable | None]
  _roots_fn          : ContextVar[Callable | None]
  _log_fn            : ContextVar[Callable | None]       # send_log() notifications
  _resource_links    : ContextVar[list | None]            # add_resource_link() for tool results
```

The `RequestContext` async context manager sets these variables on entry
and resets them on exit:

```python
async with RequestContext(session_id="abc", user_id="user@example.com"):
    # All code here sees session_id="abc", user_id="user@example.com"
    result = await my_tool(args)
    # Tools can call context.get_session_id(), context.get_user_id()
```

For HTTP, `ContextMiddleware` wraps each request in a `RequestContext`
automatically. For STDIO, the transport sets context before dispatching
each message.

## Sampling

Sampling enables server-initiated LLM requests back to the client. This
is the reverse of the normal flow: the server asks the client to create
a message using the client's model.

As of MCP 2025-11-25, sampling supports **tool calling**: the server can
include `tools` and `toolChoice` in the sampling request, allowing the
client's LLM to invoke tools during the sampled conversation.

```
Tool code
  |
  v
context.create_message(messages, model_preferences, tools, tool_choice, ...)
  |
  v
_sampling_fn (set in contextvars)
  |
  v
protocol.send_sampling_request()
  - Builds JSON-RPC request:
      {"jsonrpc": "2.0",
       "id": <uuid>,
       "method": "sampling/createMessage",
       "params": {messages, modelPreferences, tools, toolChoice, ...}}
  |
  v
transport._send_to_client(request)
  - HTTP: emits as SSE `server_request` event on open stream
  - STDIO: writes JSON line to stdout
  |
  v
Client processes request, sends response
  - HTTP: client POSTs response to /mcp/respond
  - STDIO: client writes JSON line to stdin
  |
  v
transport reads response -> resolves Future -> CreateMessageResult
  |
  v
Return to tool code
```

Sampling is optional. If the client does not declare sampling support
during initialization, `create_message()` raises an error.

### HTTP Bidirectional Transport

HTTP bidirectional uses SSE streaming with a request-response pairing
mechanism. During tool execution, server-to-client requests are emitted
as SSE events. The client responds via `POST /mcp/respond`.

```
Client                                    Server
  |                                          |
  |-- POST /mcp (tools/call) ------------->  |
  |                                          |  [tool starts executing]
  |<--- SSE event: server_request --------   |  [tool calls create_message()]
  |                                          |  [server creates Future, waits]
  |-- POST /mcp/respond {id, result} ----->  |  [resolve Future]
  |                                          |  [tool continues with result]
  |<--- SSE event: final tool result -----   |  [tool done, stream ends]
```

Key components in `endpoints/mcp.py`:
- `_pending_requests` dict maps request IDs to asyncio Futures
- `_send_to_client_http()` enqueues server requests to an SSE queue
- `handle_respond()` resolves pending futures from client POSTs
- `_sse_stream_generator()` yields SSE events from the queue

Notifications (progress) are fire-and-forget: they are enqueued and
yielded as SSE events without creating a Future.

## Elicitation, Progress, Logging, and Roots

Additional server-to-client features follow the same pattern as sampling:
a context variable holds a function injected by the protocol handler
during tool execution.

```
Elicitation (structured user input -- form mode):
  context.create_elicitation(message, schema, ...)
    -> _elicitation_fn -> protocol.send_elicitation_request()
    -> transport._send_to_client() -> client responds

Elicitation (URL mode -- MCP 2025-11-25):
  Tool raises URLElicitationRequiredError(url, description)
    -> protocol catches error -> returns JSON-RPC error -32042
    -> client directs user to URL for out-of-band interaction

Progress (fire-and-forget notifications):
  context.send_progress(progress, total, message)
    -> _progress_notify_fn -> protocol.send_progress_notification()
    -> transport._send_to_client() (no response expected)

Logging (fire-and-forget notifications -- MCP 2025-06-18):
  context.send_log(level, data, logger_name)
    -> _log_fn -> protocol.send_log_notification()
    -> transport._send_to_client() (no response expected)

Resource Links (attached to tool results -- MCP 2025-06-18):
  context.add_resource_link(uri, name, description, mime_type)
    -> accumulated in _resource_links context var
    -> protocol attaches links to tool result _meta.links

Roots (filesystem root discovery):
  context.list_roots()
    -> _roots_fn -> protocol.send_roots_request()
    -> transport._send_to_client() -> client responds with root list
```

Key differences:
- `send_progress()` and `send_log()` are silent no-ops if unavailable
- `create_elicitation()` and `list_roots()` raise `RuntimeError` if unavailable
- Progress and logging notifications have no `id` field (JSON-RPC notifications)
- Elicitation and roots are request-response (have `id` field)
- Resource links are accumulated per-request and attached to the final tool result

## Resource Subscriptions and Completions

Two client-to-server features that work on all transports:

```
Resource Subscriptions:
  Client sends resources/subscribe   -> server tracks URI per session
  Client sends resources/unsubscribe -> server removes URI tracking
  Server calls notify_resource_updated(uri)
    -> sends notification to all sessions subscribed to that URI

Completions:
  Client sends completion/complete with ref type + partial value
    -> server looks up provider in completion_providers dict
    -> provider returns {values: [...], hasMore: bool}
    -> server wraps in {completion: result} and returns
```

Subscription tracking: `protocol._resource_subscriptions` maps
session_id to a set of subscribed URIs.

Completion providers: `protocol.completion_providers` maps ref type
(`"ref/resource"` or `"ref/prompt"`) to an async provider function.

## Configuration

The `config/` module provides automatic environment detection through
independent, composable detectors. Each detector probes one aspect of
the runtime environment.

```
SmartConfig (orchestrator)
  |
  +-- CloudDetector        GCP / AWS / Azure (env vars, metadata endpoints)
  +-- ContainerDetector    Docker / Kubernetes (cgroup, service account)
  +-- EnvironmentDetector  dev / staging / prod / serverless
  +-- NetworkDetector      host / port (cloud metadata, env vars, defaults)
  +-- ProjectDetector      project name (pyproject.toml, package.json, git)
  +-- SystemDetector       CPU count, memory, optimal worker count
```

Detection results are cached after first probe. The `SmartConfig`
orchestrator merges results into a single configuration:

```python
config = SmartConfig()
# config.host        -> "0.0.0.0" (cloud) or "127.0.0.1" (local)
# config.port        -> 8080 (Cloud Run) or 3000 (local)
# config.workers     -> based on CPU count and memory
# config.cloud       -> "gcp" | "aws" | "azure" | None
# config.environment -> "production" | "development" | "serverless"
# config.project     -> "my-project"
```

When `run()` is called with no arguments, SmartConfig determines all
server parameters automatically.

## Composition and Proxy

### Server Composition

The composition system allows building MCP servers from other MCP
servers. Two patterns are supported:

```
import_server("math_tools", "path/to/math_server.py")
  - Loads module, imports its registered tools/resources/prompts
  - Tools appear with optional prefix: "math_tools.add"

mount("/analysis", analysis_server)
  - Mounts a sub-server at a path prefix
  - Requests matching the prefix are routed to the sub-server
```

Composition can also be configured via YAML:

```yaml
servers:
  math:
    module: math_tools.server
    prefix: math
  data:
    module: data_tools.server
    prefix: data
```

### Proxy

The proxy system connects to remote MCP servers and exposes their tools
as local tools. It uses chuk-tool-processor for resilience:

```
Local Server
  |
  v
ProxyManager
  |-- connects to remote MCP server(s)
  |-- discovers remote tools via tools/list
  |-- wraps each remote tool as a local ToolHandler
  +-- calls route through chuk-tool-processor:
        - Timeout enforcement
        - Retry with backoff
        - Circuit breaker
        - Connection pooling
```

## Performance

### Design Decisions for Throughput

| Layer              | Technique                              | Impact           |
|--------------------|----------------------------------------|------------------|
| Serialization      | orjson (Rust-based JSON)               | ~5x vs json      |
| Schema caching     | Computed once at registration          | Zero per-request |
| Handler caching    | Pre-serialized bytes on ToolHandler    | Zero per-list    |
| Protocol dispatch  | Direct dict lookup, no reflection      | O(1) dispatch    |
| HTTP parser        | httptools (C-based HTTP parsing)       | ~3x vs pure-py   |
| Event loop         | uvloop (libuv-based event loop)        | ~2x vs asyncio   |
| Context            | contextvars (C implementation in CPython) | Zero-lock      |

### Benchmark Profile

```
36,000+ requests/second (tools/call, single worker)
< 3ms framework overhead per request
```

The overhead measurement isolates the framework's contribution: JSON-RPC
parsing, dispatch, context setup, response serialization. Tool execution
time is not included.

### Resource Caching

ResourceHandler supports TTL-based content caching. When a resource is
read, the result is cached and subsequent reads within the TTL window
return the cached value without invoking the handler function.

## Production Hardening (v0.21)

The server includes several production-readiness features added in v0.21:

### Session Lifecycle

When sessions are evicted at capacity, an `on_evict` callback cleans up
`_resource_subscriptions`, `_sse_event_buffers`, and `_sse_event_counters`
to prevent memory leaks. Sessions with active SSE streams are protected
from eviction. `_cleanup_session_state()` centralizes all per-session
cleanup and is also called during `terminate_session` and `shutdown`.

### Request Validation

Three layers of input validation protect against oversized or malformed
requests:

- HTTP body size: `MAX_REQUEST_BODY_BYTES` (10 MB) in `endpoints/mcp.py`
- STDIO message size: Same limit in `stdio_transport.py`
- Tool arguments: Type check (`dict` required), key count limit (`MAX_ARGUMENT_KEYS = 100`)

### Rate Limiting

`TokenBucketRateLimiter` in `rate_limiter.py` provides per-session
token bucket rate limiting. Enabled by passing `rate_limit_rps` to
`MCPProtocolHandler`. Disabled by default (no overhead when off).
Session buckets are cleaned up on session eviction.

### Thread Safety

Global mutable state is protected with locks:

- `_registry_lock` in `decorators.py` wraps all `_global_*.append()` calls
- `_server_lock` in `__init__.py` uses double-checked locking for the singleton
- `clear_global_registry()` uses `.clear()` instead of `= []` to preserve references

### Graceful Shutdown

`MCPProtocolHandler.shutdown(timeout=5.0)` drains in-flight requests
with a configurable timeout, then cancels remaining tasks and cleans
all session state. Called from `core.py` on `KeyboardInterrupt`.

### Health Endpoints

| Endpoint | Purpose |
|----------|---------|
| `/health` | Liveness probe (status + uptime) |
| `/health/ready` | Readiness probe (checks tools registered, 200 or 503) |
| `/health/detailed` | Full state: tools, resources, prompts, sessions, in-flight requests |

### Telemetry

`telemetry.py` provides a thin OpenTelemetry wrapper. `trace_tool_call()`
creates spans when otel is installed and falls back to a lightweight
timing dict when not. Zero overhead without the `opentelemetry` package.

## Dependencies

### Required

| Package               | Purpose                                      |
|-----------------------|----------------------------------------------|
| chuk-mcp              | MCP protocol types (ServerInfo, etc.)        |
| chuk-sessions         | Session lifecycle management                 |
| starlette             | ASGI framework for HTTP transport            |
| uvicorn               | ASGI server                                  |
| uvloop                | Fast event loop (libuv-based)                |
| httptools             | Fast HTTP parser (C-based)                   |
| orjson                | Fast JSON serialization (Rust-based)         |
| psutil                | System detection (CPU, memory)               |
| pyyaml                | YAML configuration loading                   |

### Optional

| Package               | Purpose                                      |
|-----------------------|----------------------------------------------|
| chuk-tool-processor   | Proxy resilience (timeouts, retries, circuit breakers) |
| chuk-artifacts        | Artifact and workspace storage               |
