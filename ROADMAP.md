# ChukMCPServer Development Roadmap

This document outlines the development roadmap for `chuk-mcp-server`, the Python MCP framework. It reflects current capabilities, recent completions, review findings, MCP specification gap analysis, and planned work across upcoming releases.

---

## Current State: v0.21.0

**2330+ tests | 97% coverage | 36K+ RPS**

ChukMCPServer provides a decorator-based framework for building production-ready Model Context Protocol servers in Python. Full conformance with MCP specification **2025-11-25** (latest). The current release includes:

| Area | Capabilities |
|------|-------------|
| **Core API** | `@tool`, `@resource`, `@resource_template`, `@prompt` decorators with automatic JSON schema generation |
| **Transports** | Streamable HTTP (Starlette/Uvicorn), STDIO (sync + async), bidirectional on both transports |
| **Protocol** | JSON-RPC 2.0, MCP specification 2025-11-25, full protocol surface |
| **Context** | Session management, user context, progress reporting, log notifications, resource links, metadata, sampling, elicitation, roots |
| **Types** | `ToolHandler`, `ResourceHandler`, `ResourceTemplateHandler`, `PromptHandler` with orjson caching |
| **Tool Annotations** | `readOnlyHint`, `destructiveHint`, `idempotentHint`, `openWorldHint` metadata on tools |
| **Structured Output** | `outputSchema` on tool definitions, `structuredContent` in tool results |
| **Icons** | Icons on tools, resources, resource templates, prompts, and server info |
| **Tasks** | Durable long-running request state machines with `tasks/get`, `tasks/result`, `tasks/list`, `tasks/cancel` |
| **Cancellation** | `notifications/cancelled` support for cancelling in-flight requests |
| **Pagination** | Cursor-based pagination for `tools/list`, `resources/list`, `prompts/list`, `resources/templates/list` |
| **OAuth** | Base provider, Google Drive provider, middleware integration |
| **Cloud** | Auto-detection and deployment for GCP, AWS, Azure, Vercel, Netlify, Cloudflare |
| **Composition** | `import_server` (static), `mount` (dynamic), module loader, proxy |
| **Sampling** | Server-to-client LLM requests via `context.create_message()` with tool calling support |
| **Elicitation** | Server-to-client structured user input via `context.create_elicitation()` (form mode + URL mode + defaults) |
| **Progress** | Server-to-client progress notifications via `context.send_progress()` |
| **Roots** | Client filesystem root discovery via `context.list_roots()` |
| **Subscriptions** | Client subscribes to resource URIs and receives update notifications |
| **Completions** | Argument auto-completion for resources and prompts |
| **Logging** | `logging/setLevel` handler + `notifications/message` log entries to clients |
| **Audio** | `AudioContent` type support via chuk_mcp |
| **Artifacts** | Optional `chuk-artifacts` integration for persistent storage |
| **Performance** | 36K+ requests per second with schema and orjson caching |
| **CLI** | Project scaffolding (`init`), Claude Desktop integration (`--claude`), `--reload`, `--inspect` |
| **Testing** | `ToolRunner` test harness for invoking tools without transport |
| **OpenAPI** | Auto-generated OpenAPI 3.1.0 spec from tool schemas at `/openapi.json` |
| **Errors** | Structured error messages with fix suggestions and fuzzy tool name matching |
| **Rate Limiting** | Per-session token bucket rate limiter, configurable via `rate_limit_rps`, disabled by default |
| **Validation** | Request body size limits (10 MB), argument key count limits (100), type checking |
| **Thread Safety** | Locked global registries, double-checked locking singleton, graceful shutdown with drain |
| **Health** | `/health` (liveness), `/health/ready` (readiness), `/health/detailed` (full state) |
| **Telemetry** | Thin OpenTelemetry wrapper, zero overhead without otel |

### Recently Completed (v0.21.0 -- Phase 5: Production Hardening)

- **Session lifecycle cleanup** -- `on_evict` callback cleans up `_resource_subscriptions`, `_sse_event_buffers`, `_sse_event_counters` on session eviction; protected sessions with active SSE streams skip eviction
- **Request validation** -- 10 MB body size limit on HTTP and STDIO transports, 100-key argument limit, type checking on tool arguments
- **Rate limiting** -- Per-session token bucket rate limiter (`TokenBucketRateLimiter`), configurable via `rate_limit_rps` parameter, disabled by default
- **Narrow exception handling** -- `asyncio.CancelledError` re-raised, `ValueError`/`TypeError`/`KeyError` return `INVALID_PARAMS`, generic exceptions return sanitized `"Internal server error"` message
- **Thread safety** -- `_registry_lock` on global decorator registries, double-checked locking on singleton in `__init__.py`, `clear_global_registry()` uses `.clear()` to preserve references
- **Graceful shutdown** -- `MCPProtocolHandler.shutdown(timeout=5.0)` drains in-flight requests with configurable timeout, then cancels remaining tasks and cleans all state
- **Health check enhancements** -- `/health/ready` readiness probe (checks tools registered), `/health/detailed` endpoint (sessions, tools, resources, prompts, in-flight requests)
- **Telemetry** -- Thin OpenTelemetry wrapper (`telemetry.py`), `trace_tool_call()` context manager, zero overhead when otel is not installed
- **Pending request limits** -- `MAX_PENDING_REQUESTS = 100` enforced in STDIO transport, `stop()` cancels all pending futures

### Previously Completed (v0.20.0 -- Phase 4: MCP 2025-11-25)

- **Streamable HTTP transport** -- Single MCP endpoint (POST+GET), `MCP-Session-Id` header, session DELETE, `MCP-Protocol-Version` header, SSE resumability with event IDs
- **Tasks system** -- `tasks/get`, `tasks/result`, `tasks/list`, `tasks/cancel`, `notifications/tasks/status` for durable long-running request state machines
- **URL mode elicitation** -- Direct users to external URLs for sensitive interactions, `URLElicitationRequiredError` (-32042)
- **Tool calling in sampling** -- `tools` array and `toolChoice` parameter in `sampling/createMessage`
- **Icons** -- `icons` field on tools, resources, resource templates, prompts, and server info
- **Enhanced ServerInfo** -- `title`, `description`, `icons`, `websiteUrl` in `serverInfo` during initialize
- **Elicitation defaults** -- `default` values on primitive types in elicitation schemas
- **Tool name validation** -- Enforce 1-128 character naming rules (alphanumeric + underscore/hyphen/dot)

### Previously Completed (v0.19.0 -- Phase 3: MCP 2025-06-18)

- **Structured tool output** -- `outputSchema` on tool definitions, `structuredContent` in tool results
- **Tool annotations** -- `readOnlyHint`, `destructiveHint`, `idempotentHint`, `openWorldHint` via `@tool` decorator
- **Pagination** -- Cursor-based pagination for `tools/list`, `resources/list`, `prompts/list`, `resources/templates/list`
- **Resource templates** -- `resources/templates/list` method for URI template-based resource discovery (RFC 6570)
- **Resource links** -- Return `ResourceLink` objects alongside tool content to reference server resources
- **Content annotations** -- `audience`, `priority` annotations on content types
- **Request cancellation** -- `notifications/cancelled` support for cancelling in-flight requests
- **Log message notifications** -- Emit `notifications/message` to clients for server-side log events

### Previously Completed (v0.18.0)

- **Elicitation support** -- `elicitation/create` via `context.create_elicitation()` enables server-initiated structured user input requests
- **Progress notifications** -- `notifications/progress` via `context.send_progress()` for long-running tool operations
- **Roots support** -- `roots/list` via `context.list_roots()` for client filesystem root discovery
- **Resource subscriptions** -- `resources/subscribe` and `resources/unsubscribe` with `notify_resource_updated()` for live resource updates
- **Completions support** -- `completion/complete` with pluggable completion providers for resources and prompts

### Previously Completed (v0.16.5)

- **MCP Sampling support** -- `sampling/createMessage` via `context.create_message()` enables server-initiated LLM requests through the client
- **Bidirectional STDIO transport** -- Full duplex communication allowing server-to-client requests over STDIO
- **Client capability detection** -- Server reads and respects client capabilities declared during `initialize` handshake

---

## Code Review Findings

A comprehensive code review identified the following issues organized by severity. Items marked **Fixed** were addressed in Phase 5 (v0.21).

### Critical

| Finding | Location | Description | Status |
|---------|----------|-------------|--------|
| Resource subscription memory leak | `protocol.py` | `_resource_subscriptions` dict grows unbounded on session eviction. | **Fixed** (WP1: `on_evict` callback) |
| Session eviction drops active sessions | `protocol.py` | Oldest session evicted with no check for in-flight requests. | **Fixed** (WP1: protected sessions) |
| Version drift | `pyproject.toml` vs docs | `pyproject.toml` says `0.16.5` while docs reference `0.18.0`. | Open |

### High

| Finding | Location | Description | Status |
|---------|----------|-------------|--------|
| Broad exception handling | `protocol.py` | OAuth validation catches bare `Exception`, masking real bugs. | **Fixed** (WP4: narrowed handlers) |
| No rate limiting or request size limits | Protocol layer | Any client can send unlimited requests or arbitrarily large payloads. | **Fixed** (WP2+WP3: size limits + rate limiter) |
| Thread safety of global state | `decorators.py`, `__init__.py` | Global mutable lists/singletons with no locking. | **Fixed** (WP5: `_registry_lock` + `_server_lock`) |
| Large files need splitting | `core.py`, `protocol.py` | Both files handle too many concerns. | Open (Phase 6) |

### Medium

| Finding | Location | Description | Status |
|---------|----------|-------------|--------|
| Pending request cleanup | `stdio_transport.py` | `_pending_requests` entries linger after timeout. | **Fixed** (WP4: `stop()` cancels, limit enforced) |
| Test suite duplication | `tests/` | Multiple `*_coverage.py` files alongside main counterparts. | Open (Phase 6) |
| Weak assertions in core tests | `test_core.py` | Tests use `try/except`+`pytest.skip()` patterns. | Open (Phase 6) |
| No concurrency tests | `tests/` | Zero tests for concurrent tool execution or race conditions. | **Partial** (WP5: thread safety tests) |
| No input validation at protocol boundary | `protocol.py` | JSON-RPC handler trusts incoming `params` dicts. | **Fixed** (WP2: argument validation) |

### Low

| Finding | Location | Description | Status |
|---------|----------|-------------|--------|
| Import-time side effects | `__init__.py` | `_auto_export_cloud_handlers()` runs at import time. | Open (Phase 6) |
| Magic numbers | `errors.py`, `protocol.py` | Undocumented constants like `max_sessions=100`. | **Partial** (WP2+WP3+WP4: named constants) |
| `dist/` in working tree | Repository root | Built distribution files should be gitignored. | Open |

---

## MCP Specification Gap Analysis

The server currently targets **MCP specification 2025-06-18**. The latest specification is **2025-11-25**. This section identifies all gaps against the latest spec, organized by specification version.

### Missing from 2025-03-26

| Feature | Spec Section | Gap | Priority |
|---------|-------------|-----|----------|
| **Tool annotations** | `server/tools` | No support for `readOnlyHint`, `destructiveHint`, `idempotentHint`, `openWorldHint` metadata on tools. Clients use these to decide execution policy. | High |
| **Streamable HTTP transport** | `basic/transports` | Server uses legacy HTTP+SSE (separate GET/POST endpoints). Spec replaced this with Streamable HTTP: single endpoint, POST returns JSON or SSE, GET opens SSE stream, `MCP-Session-Id` header, session termination via DELETE. | High |

### Missing from 2025-06-18

| Feature | Spec Section | Gap | Priority |
|---------|-------------|-----|----------|
| **Structured tool output** | `server/tools` | No `outputSchema` on tool definitions. No `structuredContent` in tool results. Tools can only return `content` (text/image/audio), not typed JSON objects. | High |
| **Resource links in tool results** | `server/tools` | No support for returning `ResourceLink` objects alongside tool content to reference server resources. | Medium |
| **Resource templates** | `server/resources` | No `resources/templates/list` method. URI template-based resources (RFC 6570) not discoverable. | Medium |
| **Content annotations** | Multiple | No `annotations` field on content types (`audience`, `priority`, `lastModified`). Used for filtering content by recipient (user vs assistant). | Medium |
| **Request cancellation** | `basic/utilities` | No `notifications/cancelled` support. Clients cannot cancel in-flight requests; servers cannot cancel sampling/elicitation. | Medium |
| **Pagination** | `server/utilities` | No cursor-based pagination for `tools/list`, `resources/list`, `prompts/list`, `resources/templates/list`. All responses return full lists. | Medium |
| **`notifications/message` for logging** | `server/utilities` | `logging/setLevel` is implemented but server does not emit `notifications/message` log entries to clients via the protocol. | Low |
| **`MCP-Protocol-Version` header** | `basic/transports` | Not sent on HTTP responses. Required by spec for Streamable HTTP transport. | Low (blocked by Streamable HTTP) |

### Missing from 2025-11-25 (Latest)

| Feature | Spec Section | Gap | Priority |
|---------|-------------|-----|----------|
| **Tasks system** (experimental) | `basic/utilities/tasks` | No support for `tasks/get`, `tasks/result`, `tasks/list`, `tasks/cancel`, `notifications/tasks/status`. Tasks are durable state machines for long-running requests with status lifecycle (`working` → `completed`/`failed`/`cancelled`). | High |
| **URL mode elicitation** | `client/elicitation` | Only form mode implemented. No URL mode for directing users to external URLs for sensitive interactions (OAuth flows, payment, credential entry). No `URLElicitationRequiredError` (-32042). | High |
| **Tool calling in sampling** | `client/sampling` | `sampling/createMessage` does not support `tools` array or `toolChoice` parameter. Clients cannot provide tools for the sampled LLM to call. | Medium |
| **Icons** | Multiple | No `icons` field on `Implementation`, `Tool`, `Prompt`, `Resource`, `ResourceTemplate`. Icons enable richer UI rendering. | Medium |
| **Enhanced Implementation info** | `basic/lifecycle` | Server `initialize` response does not include `title`, `description`, `icons`, or `websiteUrl` in `serverInfo`. | Low |
| **SSE resumability** | `basic/transports` | No SSE event IDs for reconnection. Clients cannot resume interrupted SSE streams with `Last-Event-ID`. | Low |
| **Enhanced OAuth** | `basic/authorization` | No Protected Resource Metadata (RFC 9728), no Client ID Metadata Documents, no Resource Indicators (RFC 8707). | Low |
| **Default values in elicitation schema** | `client/elicitation` | Elicitation `requestedSchema` does not support `default` on primitives. | Low |
| **Tool name validation** | `server/tools` | No enforcement of 1-128 character limit, case-sensitive, alphanumeric + underscore/hyphen/dot naming rules. | Low |

### Conformance Summary

| Spec Area | Status |
|-----------|--------|
| **Lifecycle** (initialize, ping) | Implemented |
| **Tools** (list, call) | Implemented (annotations, structured output, pagination, icons, name validation) |
| **Resources** (list, read, subscribe) | Implemented (templates, links, pagination, content annotations, icons) |
| **Prompts** (list, get) | Implemented (pagination, icons) |
| **Sampling** (createMessage) | Implemented (including tool calling) |
| **Elicitation** (create) | Implemented (form mode + URL mode + defaults) |
| **Roots** (list) | Implemented |
| **Progress** (notifications) | Implemented |
| **Completions** (complete) | Implemented |
| **Logging** (setLevel, notifications/message) | Implemented |
| **Cancellation** | Implemented |
| **Tasks** | Implemented |
| **Pagination** | Implemented |
| **Streamable HTTP** | Implemented |
| **Content annotations** | Implemented |
| **Tool annotations** | Implemented |
| **Structured output** | Implemented |
| **Icons** | Implemented |

---

## Phase 1: Core Hardening -- v0.17 (Complete)

Complete the MCP specification surface area. Every protocol capability that a conformant server can expose is now supported or gracefully declined.

| Feature | Status | Description |
|---------|--------|-------------|
| Elicitation support | **Done** | `elicitation/create` -- server-to-client structured user input via `create_elicitation()` |
| Progress notifications | **Done** | `notifications/progress` -- server-to-client progress updates via `send_progress()` |
| Roots support | **Done** | `roots/list` -- client filesystem root discovery via `list_roots()` |
| Resource subscriptions | **Done** | `resources/subscribe` / `resources/unsubscribe` with update notifications |
| Completions support | **Done** | `completion/complete` with pluggable providers for resources and prompts |

---

## Phase 2: Developer Experience -- v0.18 (Complete)

Reduce friction for developers building and debugging MCP servers. Focus on the inner development loop: write, test, iterate.

| Feature | Status | Description |
|---------|--------|-------------|
| HTTP bidirectional transport | **Done** | SSE-based server-to-client requests (sampling, elicitation, roots, progress) over HTTP |
| Tool testing utilities | **Done** | `ToolRunner` test harness for invoking tools without transport overhead |
| Improved error messages | **Done** | Structured errors with fix suggestions; fuzzy tool name matching |
| OpenAPI generation | **Done** | Auto-generate OpenAPI 3.1.0 spec from tool schemas at `/openapi.json` |
| Hot reload | **Done** | `--reload` CLI flag enables uvicorn hot reload during development |
| MCP Inspector integration | **Done** | `--inspect` CLI flag opens MCP Inspector in browser on server start |
| Type stub generation | Deferred | Generate `.pyi` stubs for downstream consumers of published tool packages |

---

## Phase 3: Spec Conformance (2025-06-18) -- v0.19

Close all gaps against MCP specification 2025-06-18. Implement the protocol features needed for full conformance with the spec the server already targets.

| Feature | Status | Description |
|---------|--------|-------------|
| Structured tool output | **Done** | `outputSchema` on tool definitions, `structuredContent` in tool results for typed JSON responses |
| Tool annotations | **Done** | `readOnlyHint`, `destructiveHint`, `idempotentHint`, `openWorldHint` metadata via `@tool` decorator |
| Pagination | **Done** | Cursor-based pagination for `tools/list`, `resources/list`, `prompts/list`, `resources/templates/list` |
| Resource templates | **Done** | `resources/templates/list` method for URI template-based resource discovery (RFC 6570) |
| Resource links | **Done** | Return `ResourceLink` objects alongside tool content to reference server resources |
| Content annotations | **Done** | `audience`, `priority`, `lastModified` annotations on content types |
| Request cancellation | **Done** | `notifications/cancelled` support for cancelling in-flight requests in both directions |
| Log message notifications | **Done** | Emit `notifications/message` to clients for server-side log events (completes logging support) |

---

## Phase 4: Spec Conformance (2025-11-25) -- v0.20

Upgrade to the latest MCP specification. Implement the new transport, tasks system, and enhanced features.

| Feature | Status | Description |
|---------|--------|-------------|
| Streamable HTTP transport | **Done** | Single MCP endpoint (POST+GET), `MCP-Session-Id` header, session DELETE, `MCP-Protocol-Version` header, SSE resumability with event IDs |
| Tasks system | **Done** | `tasks/get`, `tasks/result`, `tasks/list`, `tasks/cancel`, `notifications/tasks/status` for durable long-running request state machines |
| URL mode elicitation | **Done** | Direct users to external URLs for sensitive interactions (OAuth flows, payment, credentials), `URLElicitationRequiredError` (-32042) |
| Tool calling in sampling | **Done** | `tools` array and `toolChoice` parameter in `sampling/createMessage` for client-side tool loops |
| Icons | **Done** | `icons` field on `Implementation`, `Tool`, `Prompt`, `Resource`, `ResourceTemplate` for richer UI rendering |
| Enhanced Implementation info | **Done** | `title`, `description`, `icons`, `websiteUrl` in `serverInfo` during initialize |
| Elicitation defaults | **Done** | `default` values on primitive types in elicitation `requestedSchema` |
| Tool name validation | **Done** | Enforce 1-128 character naming rules (alphanumeric + underscore/hyphen/dot, case-sensitive) |

---

## Phase 5: Production Hardening -- v0.21 (Complete)

Address critical findings from the code review. Harden the server for sustained production workloads with reliability, observability, and security improvements.

| Feature | Status | Description |
|---------|--------|-------------|
| Resource subscription cleanup | **Done** | `on_evict` callback ties `_resource_subscriptions`, `_sse_event_buffers`, `_sse_event_counters` cleanup to session eviction and expiry |
| Session eviction improvements | **Done** | Protected sessions with active SSE streams skip eviction; `_cleanup_session_state()` centralized cleanup |
| Request size limits | **Done** | 10 MB body size limit on HTTP and STDIO transports; 100-key argument limit with type checking |
| Rate limiting | **Done** | Per-session token bucket rate limiter (`TokenBucketRateLimiter`), configurable via `rate_limit_rps`, disabled by default |
| Protocol boundary validation | **Done** | Argument type checking and key count validation before invoking tool handlers |
| Narrow exception handling | **Done** | `CancelledError` re-raised; `ValueError`/`TypeError`/`KeyError` → `INVALID_PARAMS`; generic exceptions sanitized |
| Pending request cleanup | **Done** | `MAX_PENDING_REQUESTS = 100` enforced in STDIO transport; `stop()` cancels all pending futures |
| Thread safety | **Done** | `_registry_lock` on global registries; double-checked locking on singleton; `.clear()` preserves references |
| OpenTelemetry integration | **Done** | Thin wrapper in `telemetry.py`; `trace_tool_call()` context manager; zero overhead without otel |
| Health check enhancements | **Done** | `/health/ready` readiness probe; `/health/detailed` with session/tool/resource/in-flight counts |
| Graceful shutdown | **Done** | `MCPProtocolHandler.shutdown(timeout=5.0)` drains in-flight requests then cancels remaining |
| Connection pooling for proxy | Deferred | Deferred to Phase 7 (Advanced Composition) |

---

## Phase 6: Codebase Quality -- v0.22

Refactoring and test improvements to keep the codebase maintainable as it grows.

| Feature | Status | Description |
|---------|--------|-------------|
| Split `core.py` | Planned | Extract composition, proxy, and server lifecycle into separate modules (currently 1031 lines) |
| Split `protocol.py` | Planned | Extract session management and subscription tracking into separate modules (currently 953 lines) |
| Consolidate test files | Planned | Merge `*_coverage.py` and `*_final_coverage.py` test variants into their main test files |
| Strengthen core tests | Planned | Replace `try/except`+`pytest.skip()` patterns in `test_core.py` with proper behavioral assertions |
| Add concurrency tests | Planned | Test concurrent tool execution, parallel sampling, race conditions in context system |
| Add integration tests | Planned | End-to-end HTTP and STDIO transport tests with real client connections |
| Remove import-time side effects | Planned | Defer `_auto_export_cloud_handlers()` from import time to first server creation |
| Document magic numbers | Planned | Replace hardcoded values (fuzzy cutoff `0.6`, `max_sessions=100`, `cleanup_interval=10`) with named constants |

---

## Phase 7: Advanced Composition -- v0.23

Enable large-scale MCP architectures where servers are composed from many sources and communicate across boundaries.

| Feature | Status | Description |
|---------|--------|-------------|
| Plugin system | Planned | Pip-installable tool packages discovered via entry points |
| Dynamic tool discovery | Planned | Discover and install tool packages from PyPI at runtime |
| Service mesh integration | Planned | First-class support for Istio, Linkerd, and similar service meshes |
| gRPC transport | Planned | Alternative transport for high-throughput, low-latency inter-service communication |
| WebSocket transport | Planned | Persistent bidirectional connections for sampling, elicitation, and streaming |
| Federation | Planned | Server-to-server tool delegation across trust boundaries |

---

## Phase 8: Security and Enterprise -- v0.24+

Capabilities required for enterprise deployments with strict compliance, governance, and multi-tenancy requirements.

| Feature | Status | Description |
|---------|--------|-------------|
| Enhanced OAuth (RFC 9728) | Planned | Protected Resource Metadata, Client ID Metadata Documents, Resource Indicators (RFC 8707) per latest MCP spec |
| Role-based access control | Planned | Per-tool RBAC with configurable roles and permissions |
| Audit logging | Planned | Immutable, structured audit trail for all tool invocations and administrative actions |
| Input sanitization framework | Planned | Pluggable validation and sanitization pipeline for tool inputs |
| Tool execution sandboxing | Planned | Isolate tool execution to limit blast radius of untrusted code |
| Certificate-based auth | Planned | Mutual TLS and certificate-based authentication for server-to-server communication |
| Multi-tenant support | Planned | Isolated tool registries, context, and storage per tenant |

---

## Version History

| Version | Milestone |
|---------|-----------|
| v0.21.0 | Production hardening: session lifecycle cleanup, request validation, rate limiting, exception handling, thread safety, graceful shutdown, health probes, telemetry |
| v0.20.0 | MCP 2025-11-25: Streamable HTTP, tasks, URL elicitation, tool calling in sampling, icons, enhanced ServerInfo |
| v0.19.0 | MCP 2025-06-18: Structured output, tool annotations, pagination, resource templates, resource links, content annotations, cancellation, log notifications |
| v0.18.0 | Developer experience: HTTP bidirectional, ToolRunner, improved errors, OpenAPI, hot reload, inspect |
| v0.17.0 | Full MCP protocol surface: elicitation, progress, roots, subscriptions, completions |
| v0.16.5 | MCP sampling, bidirectional STDIO, client capability detection |
| v0.16.0 | OAuth middleware, Google Drive provider |
| v0.15.0 | Composition: import_server, mount, proxy |
| v0.14.0 | Cloud auto-detection (GCP, AWS, Azure, Vercel, Netlify, Cloudflare) |
| v0.13.0 | orjson caching, 36K+ RPS performance milestone |
| v0.12.0 | CLI scaffolding, project init |

---

## Contributing

Contributions are welcome for any roadmap item. If you are interested in working on a planned feature, please open an issue to discuss the approach before submitting a pull request. See the [contributing guide](docs/contributing/setup.md) for development setup instructions.

---

*This roadmap is a living document. Priorities may shift based on community feedback, MCP specification changes, and production experience. Last updated: 2025-02.*
