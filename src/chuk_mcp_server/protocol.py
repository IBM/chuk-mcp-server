#!/usr/bin/env python3
# src/chuk_mcp_server/protocol.py
"""
ChukMCPServer Protocol Handler - Core MCP protocol implementation with chuk_mcp
"""

import asyncio
import logging
import time
import uuid
from collections.abc import Callable
from typing import Any

from .constants import (
    JSONRPC_KEY,
    JSONRPC_VERSION,
    KEY_CAPABILITIES,
    KEY_CLIENT_INFO,
    KEY_ERROR,
    KEY_ID,
    KEY_METHOD,
    KEY_PARAMS,
    KEY_PROTOCOL_VERSION,
    KEY_RESULT,
    KEY_SERVER_INFO,
    LOG_CRITICAL,
    LOG_DEBUG,
    LOG_ERROR,
    LOG_INFO,
    LOG_WARNING,
    MCP_DEFAULT_PROTOCOL_VERSION,
    PACKAGE_LOGGER,
    PARAM_EXTERNAL_ACCESS_TOKEN,
    PARAM_USER_ID,
    JsonRpcError,
    McpMethod,
    McpTaskMethod,
)
from .types import (
    PromptHandler,
    ResourceHandler,
    ServerCapabilities,
    ServerInfo,
    ToolHandler,
    format_content,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Session Management
# ============================================================================


class SessionManager:
    """Manage MCP sessions."""

    def __init__(
        self,
        max_sessions: int = 1000,
        cleanup_interval: int = 100,
        on_evict: Callable[[str], None] | None = None,
        protected_sessions: Callable[[], set[str]] | None = None,
    ):
        self.sessions: dict[str, dict[str, Any]] = {}
        self.max_sessions = max_sessions
        self.cleanup_interval = cleanup_interval
        self._creation_count = 0
        self._on_evict = on_evict
        self._protected_sessions = protected_sessions

    def _evict_session(self, session_id: str) -> None:
        """Evict a session, calling the on_evict callback first."""
        if self._on_evict is not None:
            self._on_evict(session_id)
        del self.sessions[session_id]

    def create_session(self, client_info: dict[str, Any], protocol_version: str) -> str:
        """Create a new session."""
        self._creation_count += 1

        # Periodic cleanup of expired sessions
        if self._creation_count % self.cleanup_interval == 0:
            self.cleanup_expired()

        # Evict oldest session if at capacity
        if len(self.sessions) >= self.max_sessions:
            protected = self._protected_sessions() if self._protected_sessions else set()
            candidates = [sid for sid in self.sessions if sid not in protected]
            if candidates:
                oldest_sid = min(candidates, key=lambda sid: self.sessions[sid]["last_activity"])
                self._evict_session(oldest_sid)
                logger.debug(f"Evicted oldest session {oldest_sid[:8]}... (max_sessions reached)")

        session_id = str(uuid.uuid4()).replace("-", "")
        self.sessions[session_id] = {
            "id": session_id,
            "client_info": client_info,
            "protocol_version": protocol_version,
            "created_at": time.time(),
            "last_activity": time.time(),
        }
        logger.debug(f"Created session {session_id[:8]}... for {client_info.get('name', 'unknown')}")
        return session_id

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        """Get session by ID."""
        return self.sessions.get(session_id)

    def update_activity(self, session_id: str) -> None:
        """Update session last activity."""
        if session_id in self.sessions:
            self.sessions[session_id]["last_activity"] = time.time()

    def cleanup_expired(self, max_age: int = 3600) -> None:
        """Remove expired sessions."""
        now = time.time()
        expired = [sid for sid, session in self.sessions.items() if now - session["last_activity"] > max_age]
        for sid in expired:
            self._evict_session(sid)
            logger.debug(f"Cleaned up expired session {sid[:8]}...")


# ============================================================================
# Protocol Handler with chuk_mcp Integration
# ============================================================================


class MCPProtocolHandler:
    """Core MCP protocol handler powered by chuk_mcp."""

    def __init__(
        self,
        server_info: ServerInfo,
        capabilities: ServerCapabilities,
        oauth_provider_getter: Any = None,
        extra_server_info: dict[str, Any] | None = None,
        rate_limit_rps: float | None = None,
    ):
        # Use chuk_mcp types directly - no conversion needed
        self.server_info = server_info
        self.capabilities = capabilities
        self._extra_server_info = extra_server_info
        self.session_manager = SessionManager(
            on_evict=self._cleanup_session_state,
            protected_sessions=self._get_protected_sessions,
        )

        # Tool, resource, and prompt registries (now use handlers)
        self.tools: dict[str, ToolHandler] = {}
        self.resources: dict[str, ResourceHandler] = {}
        self.prompts: dict[str, PromptHandler] = {}

        # OAuth provider getter function (optional)
        self.oauth_provider_getter = oauth_provider_getter

        # Transport callback for sending requests to the client (set by transport layer)
        self._send_to_client: Callable[..., Any] | None = None

        # Completion providers registry (ref_type → provider callable)
        self.completion_providers: dict[str, Callable[..., Any]] = {}

        # Resource template registry
        self.resource_templates: dict[str, Any] = {}

        # Resource subscription tracking (session_id → set of URIs)
        self._resource_subscriptions: dict[str, set[str]] = {}

        # In-flight request tracking for cancellation support
        self._in_flight_requests: dict[Any, asyncio.Task[Any]] = {}

        # Task store for MCP 2025-11-25 Tasks system
        self._task_store: dict[str, dict[str, Any]] = {}

        # SSE event buffer for resumability (session_id → list of (event_id, data))
        self._sse_event_buffers: dict[str, list[tuple[int, dict[str, Any]]]] = {}
        self._sse_event_counters: dict[str, int] = {}

        # Rate limiter (off by default)
        self._rate_limiter: Any = None
        if rate_limit_rps is not None:
            from .rate_limiter import TokenBucketRateLimiter

            burst = rate_limit_rps * 2  # Default burst = 2x rate
            self._rate_limiter = TokenBucketRateLimiter(rate=rate_limit_rps, burst=burst)

        # Don't log during init to keep stdio mode clean
        logger.debug("MCP protocol handler initialized with chuk_mcp")

    def register_tool(self, tool: ToolHandler) -> None:
        """Register a tool handler."""
        self.tools[tool.name] = tool
        logger.debug(f"Registered tool: {tool.name}")

    def register_resource(self, resource: ResourceHandler) -> None:
        """Register a resource handler."""
        self.resources[resource.uri] = resource
        logger.debug(f"Registered resource: {resource.uri}")

    def register_prompt(self, prompt: PromptHandler) -> None:
        """Register a prompt handler."""
        self.prompts[prompt.name] = prompt
        logger.debug(f"Registered prompt: {prompt.name}")

    def register_resource_template(self, template: Any) -> None:
        """Register a resource template handler."""
        self.resource_templates[template.uri_template] = template
        logger.debug(f"Registered resource template: {template.uri_template}")

    def get_resource_templates_list(self) -> list[dict[str, Any]]:
        """Get list of resource templates in MCP format."""
        return [t.to_mcp_format() for t in self.resource_templates.values()]

    def register_completion_provider(self, ref_type: str, provider: Callable[..., Any]) -> None:
        """Register a completion provider for a reference type.

        Args:
            ref_type: Reference type (e.g., "ref/resource" or "ref/prompt")
            provider: Async callable that returns completion suggestions
        """
        self.completion_providers[ref_type] = provider
        logger.debug(f"Registered completion provider: {ref_type}")

    def get_tools_list(self) -> list[dict[str, Any]]:
        """Get list of tools in MCP format."""
        tools_list = []

        for tool_handler in self.tools.values():
            tools_list.append(tool_handler.to_mcp_format())

        return tools_list

    def get_resources_list(self) -> list[dict[str, Any]]:
        """Get list of resources in MCP format."""
        resources_list = []

        for resource_handler in self.resources.values():
            resources_list.append(resource_handler.to_mcp_format())

        return resources_list

    def get_prompts_list(self) -> list[dict[str, Any]]:
        """Get list of prompts in MCP format."""
        prompts_list = []

        for prompt_handler in self.prompts.values():
            prompts_list.append(prompt_handler.to_mcp_format())

        return prompts_list

    def get_performance_stats(self) -> dict[str, Any]:
        """Get performance statistics for monitoring.

        Note: Cache-related metrics are not yet instrumented and return None.
        """

        return {
            "tools": {
                "count": len(self.tools),
                "cache_hit_ratio": None,  # Not yet instrumented
            },
            "resources": {
                "count": len(self.resources),
                "cache_hit_ratio": None,  # Not yet instrumented
            },
            "prompts": {
                "count": len(self.prompts),
                "cache_hit_ratio": None,  # Not yet instrumented
            },
            "sessions": {"active": len(self.session_manager.sessions), "total": len(self.session_manager.sessions)},
            "cache": {
                "tools_cached": None,  # Not yet instrumented
                "resources_cached": None,  # Not yet instrumented
                "cache_age": None,  # Not yet instrumented
            },
            "status": "operational",
        }

    async def handle_request(
        self, message: dict[str, Any], session_id: str | None = None, oauth_token: str | None = None
    ) -> tuple[dict[str, Any] | None, str | None]:
        """Handle an MCP request."""
        try:
            method = message.get(KEY_METHOD)
            params = message.get(KEY_PARAMS, {})
            msg_id = message.get(KEY_ID)

            logger.debug(f"Handling {method} (ID: {msg_id})")

            # Set session context for this request
            if session_id:
                from .context import set_session_id

                set_session_id(session_id)
                self.session_manager.update_activity(session_id)

            # Rate limit check
            if self._rate_limiter is not None and session_id:
                if not self._rate_limiter.allow(session_id):
                    return self._create_error_response(msg_id, JsonRpcError.INTERNAL_ERROR, "Rate limit exceeded"), None

            # Route to appropriate handler
            if method == McpMethod.INITIALIZE:
                return await self._handle_initialize(params, msg_id)
            elif method == McpMethod.INITIALIZED:
                logger.debug("✅ Initialized notification received")
                return None, None  # Notifications don't return responses
            elif method == McpMethod.PING:
                return await self._handle_ping(msg_id)
            elif method == McpMethod.TOOLS_LIST:
                return await self._handle_tools_list(params, msg_id)
            elif method == McpMethod.TOOLS_CALL:
                return await self._handle_tools_call(params, msg_id, oauth_token)
            elif method == McpMethod.RESOURCES_LIST:
                return await self._handle_resources_list(params, msg_id)
            elif method == McpMethod.RESOURCES_READ:
                return await self._handle_resources_read(params, msg_id)
            elif method == McpMethod.PROMPTS_LIST:
                return await self._handle_prompts_list(params, msg_id)
            elif method == McpMethod.PROMPTS_GET:
                return await self._handle_prompts_get(params, msg_id)
            elif method == McpMethod.LOGGING_SET_LEVEL:
                return await self._handle_logging_set_level(params, msg_id)
            elif method == McpMethod.COMPLETION_COMPLETE:
                return await self._handle_completion_complete(params, msg_id)
            elif method == McpMethod.RESOURCES_TEMPLATES_LIST:
                return await self._handle_resources_templates_list(params, msg_id)
            elif method == McpMethod.RESOURCES_SUBSCRIBE:
                return await self._handle_resources_subscribe(params, msg_id)
            elif method == McpMethod.RESOURCES_UNSUBSCRIBE:
                return await self._handle_resources_unsubscribe(params, msg_id)
            elif method == McpMethod.NOTIFICATIONS_CANCELLED:
                self._handle_cancelled_notification(params)
                return None, None  # Notification, no response
            elif method == McpMethod.NOTIFICATIONS_ROOTS_LIST_CHANGED:
                logger.debug("Roots list changed notification received")
                return None, None  # Notification, no response
            # Tasks system (MCP 2025-11-25)
            elif method == McpTaskMethod.TASKS_GET:
                return await self._handle_tasks_get(params, msg_id)
            elif method == McpTaskMethod.TASKS_RESULT:
                return await self._handle_tasks_result(params, msg_id)
            elif method == McpTaskMethod.TASKS_LIST:
                return await self._handle_tasks_list(params, msg_id)
            elif method == McpTaskMethod.TASKS_CANCEL:
                return await self._handle_tasks_cancel(params, msg_id)
            else:
                return self._create_error_response(
                    msg_id, JsonRpcError.METHOD_NOT_FOUND, f"Method not found: {method}"
                ), None

        except asyncio.CancelledError:
            raise  # Never swallow cancellation
        except (ValueError, TypeError, KeyError) as e:
            logger.warning(f"Invalid params in request: {e}")
            return self._create_error_response(
                msg_id, JsonRpcError.INVALID_PARAMS, f"Invalid parameters: {str(e)}"
            ), None
        except Exception as e:
            logger.error(f"Error handling request: {e}", exc_info=True)
            return self._create_error_response(msg_id, JsonRpcError.INTERNAL_ERROR, "Internal server error"), None

    async def _handle_initialize(self, params: dict[str, Any], msg_id: Any) -> tuple[dict[str, Any], str]:
        """Handle initialize request using chuk_mcp."""
        client_info = params.get(KEY_CLIENT_INFO, {})
        protocol_version = params.get(KEY_PROTOCOL_VERSION, MCP_DEFAULT_PROTOCOL_VERSION)
        client_capabilities = params.get(KEY_CAPABILITIES, {})

        # Create session
        session_id = self.session_manager.create_session(client_info, protocol_version)

        # Store client capabilities on the session
        session = self.session_manager.get_session(session_id)
        if session is not None:
            session["client_capabilities"] = client_capabilities

        # Build response using chuk_mcp types directly
        server_info_dict = self.server_info.model_dump(exclude_none=True)
        # Merge extra server info fields (MCP 2025-11-25: description, icons, websiteUrl)
        if self._extra_server_info:
            server_info_dict.update(self._extra_server_info)

        result = {
            KEY_PROTOCOL_VERSION: protocol_version,
            KEY_SERVER_INFO: server_info_dict,
            KEY_CAPABILITIES: self.capabilities.model_dump(exclude_none=True),
        }

        response = {JSONRPC_KEY: JSONRPC_VERSION, KEY_ID: msg_id, KEY_RESULT: result}

        client_name = client_info.get("name", "unknown")
        sampling_supported = "sampling" in client_capabilities
        logger.debug(
            f"🤝 Initialized session {session_id[:8]}... for {client_name} "
            f"(v{protocol_version}, sampling={'yes' if sampling_supported else 'no'})"
        )
        return response, session_id

    async def _handle_ping(self, msg_id: Any) -> tuple[dict[str, Any], None]:
        """Handle ping request."""
        return {JSONRPC_KEY: JSONRPC_VERSION, KEY_ID: msg_id, KEY_RESULT: {}}, None

    async def _handle_tools_list(self, params: dict[str, Any], msg_id: Any) -> tuple[dict[str, Any], None]:
        """Handle tools/list request with pagination support."""
        tools_list = self.get_tools_list()
        result = self._paginate(tools_list, "tools", params)

        response = {JSONRPC_KEY: JSONRPC_VERSION, KEY_ID: msg_id, KEY_RESULT: result}

        logger.debug(f"Returning {len(result['tools'])} tools")
        return response, None

    async def _handle_tools_call(
        self, params: dict[str, Any], msg_id: Any, oauth_token: str | None = None
    ) -> tuple[dict[str, Any], None]:
        """Handle tools/call request."""
        tool_name = params.get("name")
        arguments = params.get("arguments", {})

        # Validate arguments
        if not isinstance(arguments, dict):
            return self._create_error_response(
                msg_id, JsonRpcError.INVALID_PARAMS, f"arguments must be an object, got {type(arguments).__name__}"
            ), None

        from .constants import MAX_ARGUMENT_KEYS

        if len(arguments) > MAX_ARGUMENT_KEYS:
            return self._create_error_response(
                msg_id,
                JsonRpcError.INVALID_PARAMS,
                f"Too many argument keys ({len(arguments)}, max {MAX_ARGUMENT_KEYS})",
            ), None

        if tool_name not in self.tools:
            from .errors import format_unknown_tool_error

            error_msg = format_unknown_tool_error(tool_name, list(self.tools.keys()))
            return self._create_error_response(msg_id, JsonRpcError.INVALID_PARAMS, error_msg), None

        try:
            tool_handler = self.tools[tool_name]

            # Check if tool requires OAuth authorization
            if tool_handler.requires_auth:
                # Tool requires auth - validate OAuth token
                if not oauth_token:
                    return self._create_error_response(
                        msg_id,
                        JsonRpcError.INTERNAL_ERROR,
                        f"Tool '{tool_name}' requires OAuth authorization. Please authenticate first.",
                    ), None

                if not self.oauth_provider_getter:
                    return self._create_error_response(
                        msg_id,
                        JsonRpcError.INTERNAL_ERROR,
                        f"Tool '{tool_name}' requires OAuth but OAuth is not configured on this server.",
                    ), None

                # Validate OAuth token and get external provider token
                try:
                    provider = self.oauth_provider_getter()
                    if not provider:
                        return self._create_error_response(
                            msg_id, JsonRpcError.INTERNAL_ERROR, "OAuth provider not available."
                        ), None

                    token_data = await provider.validate_access_token(oauth_token)
                    logger.debug(f"📦 Token data received for {tool_name}: {list(token_data.keys())}")
                    external_token = token_data.get("external_access_token")
                    user_id = token_data.get("user_id")

                    logger.debug(
                        f"🔑 Token data for {tool_name}: external_token={'present' if external_token else 'NONE'}, user_id={user_id}"
                    )

                    if not external_token:
                        return self._create_error_response(
                            msg_id,
                            JsonRpcError.INTERNAL_ERROR,
                            "OAuth token is valid but external provider token is missing.",
                        ), None

                    # Inject external provider token and user_id into arguments
                    arguments[PARAM_EXTERNAL_ACCESS_TOKEN] = external_token
                    logger.debug(f"✅ Injected OAuth token into {tool_name} arguments")
                    if user_id:
                        arguments[PARAM_USER_ID] = user_id

                        # Also set user_id in context for application code to access
                        # This allows apps to use get_current_user_id() instead of passing _user_id everywhere
                        from .context import set_user_id

                        set_user_id(user_id)

                    logger.debug(f"OAuth token validated for tool {tool_name}, user_id: {user_id}")

                except Exception as e:
                    # OAuth validation failed for a tool that requires it
                    logger.error(f"OAuth validation failed for {tool_name}: {e}")
                    return self._create_error_response(
                        msg_id, JsonRpcError.INTERNAL_ERROR, f"OAuth validation failed: {str(e)}"
                    ), None

            # Set up server-to-client context if client supports it
            from .context import (
                set_elicitation_fn,
                set_log_fn,
                set_progress_notify_fn,
                set_progress_token,
                set_resource_links,
                set_roots_fn,
                set_sampling_fn,
            )

            if self._client_supports_sampling(params):
                set_sampling_fn(self.send_sampling_request)
            else:
                set_sampling_fn(None)

            if self._client_supports_elicitation(params):
                set_elicitation_fn(self.send_elicitation_request)
            else:
                set_elicitation_fn(None)

            if self._client_supports_roots(params):
                set_roots_fn(self.send_roots_request)
            else:
                set_roots_fn(None)

            # Extract progress token from request _meta and inject notify fn
            progress_token = params.get("_meta", {}).get("progressToken")
            set_progress_token(progress_token)
            if progress_token and self._send_to_client:

                async def _progress_notify(
                    progress_token: str | int,
                    progress: float,
                    total: float | None = None,
                    message: str | None = None,
                ) -> None:
                    await self.send_progress_notification(progress_token, progress, total, message)

                set_progress_notify_fn(_progress_notify)
            else:
                set_progress_notify_fn(None)

            # Set up log notification fn (always available if transport exists)
            if self._send_to_client:
                set_log_fn(
                    lambda level="info", data=None, logger_name=None: self.send_log_notification(
                        level=level, data=data, logger_name=logger_name
                    )
                )
            else:
                set_log_fn(None)

            # Initialize resource links collection for this tool call
            set_resource_links(None)

            # Track in-flight request for cancellation support
            task = asyncio.current_task()
            if task is not None and msg_id is not None:
                self._in_flight_requests[msg_id] = task

            try:
                # Execute the tool
                result = await tool_handler.execute(arguments)
            except asyncio.CancelledError:
                logger.debug(f"Tool execution cancelled for {tool_name} (request {msg_id})")
                return self._create_error_response(msg_id, JsonRpcError.INTERNAL_ERROR, "Request cancelled"), None
            finally:
                # Remove from in-flight tracking
                self._in_flight_requests.pop(msg_id, None)
                # Always clear all server-to-client fns after tool execution
                set_sampling_fn(None)
                set_elicitation_fn(None)
                set_roots_fn(None)
                set_progress_notify_fn(None)
                set_progress_token(None)
                set_log_fn(None)

            # Format response content using chuk_mcp content formatting
            content = format_content(result)

            # Build result dict
            tool_result: dict[str, Any] = {"content": content}

            # Add structured content if tool has output_schema and result is a dict/model
            if tool_handler.output_schema is not None and result is not None:
                from pydantic import BaseModel as _BaseModel

                if isinstance(result, dict):
                    tool_result["structuredContent"] = result
                elif isinstance(result, _BaseModel):
                    tool_result["structuredContent"] = result.model_dump()

            # Add resource links if any were accumulated during execution
            from .context import get_resource_links

            links = get_resource_links()
            if links:
                tool_result["_meta"] = tool_result.get("_meta", {})
                tool_result["_meta"]["links"] = links
            set_resource_links(None)

            response = {JSONRPC_KEY: JSONRPC_VERSION, KEY_ID: msg_id, KEY_RESULT: tool_result}

            logger.debug(f"🔧 Executed tool {tool_name}")
            return response, None

        except Exception as e:
            # Check for URL elicitation required (MCP 2025-11-25)
            from .types.errors import URLElicitationRequiredError

            if isinstance(e, URLElicitationRequiredError):
                from .constants import MCP_ERROR_URL_ELICITATION_REQUIRED

                error_data: dict[str, Any] = {"url": e.url}
                if e.description is not None:
                    error_data["description"] = e.description
                if e.mime_type is not None:
                    error_data["mimeType"] = e.mime_type
                return {
                    JSONRPC_KEY: JSONRPC_VERSION,
                    KEY_ID: msg_id,
                    KEY_ERROR: {
                        "code": MCP_ERROR_URL_ELICITATION_REQUIRED,
                        "message": str(e),
                        "data": error_data,
                    },
                }, None

            logger.error(f"Tool execution error for {tool_name}: {e}")
            return self._create_error_response(
                msg_id, JsonRpcError.INTERNAL_ERROR, f"Tool execution error: {str(e)}"
            ), None

    def _client_supports_sampling(self, tool_call_params: dict[str, Any]) -> bool:
        """Check if the current session's client supports sampling."""
        if self._send_to_client is None:
            return False

        # Find the session and check client capabilities
        from .context import get_session_id

        session_id = get_session_id()
        if not session_id:
            return False

        session = self.session_manager.get_session(session_id)
        if not session:
            return False

        client_caps = session.get("client_capabilities", {})
        return "sampling" in client_caps

    def _client_supports_elicitation(self, tool_call_params: dict[str, Any]) -> bool:  # noqa: ARG002
        """Check if the current session's client supports elicitation."""
        if self._send_to_client is None:
            return False

        from .context import get_session_id

        session_id = get_session_id()
        if not session_id:
            return False

        session = self.session_manager.get_session(session_id)
        if not session:
            return False

        client_caps = session.get("client_capabilities", {})
        return "elicitation" in client_caps

    def _client_supports_roots(self, tool_call_params: dict[str, Any]) -> bool:  # noqa: ARG002
        """Check if the current session's client supports roots."""
        if self._send_to_client is None:
            return False

        from .context import get_session_id

        session_id = get_session_id()
        if not session_id:
            return False

        session = self.session_manager.get_session(session_id)
        if not session:
            return False

        client_caps = session.get("client_capabilities", {})
        return "roots" in client_caps

    async def send_sampling_request(
        self,
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
    ) -> dict[str, Any]:
        """
        Send a sampling/createMessage request to the MCP client.

        Args:
            messages: List of sampling messages (dicts with role and content)
            max_tokens: Maximum tokens for the response
            system_prompt: Optional system prompt
            temperature: Optional temperature
            model_preferences: Optional model preferences
            stop_sequences: Optional stop sequences
            metadata: Optional metadata
            include_context: Optional context inclusion

        Returns:
            Dict with the client's sampling response (role, content, model, stopReason)

        Raises:
            RuntimeError: If transport callback is not available or client doesn't support sampling
        """
        if self._send_to_client is None:
            raise RuntimeError("No transport callback available for sending sampling requests")

        # Build the sampling/createMessage params
        params: dict[str, Any] = {
            "messages": [
                m if isinstance(m, dict) else (m.model_dump() if hasattr(m, "model_dump") else dict(m))
                for m in messages
            ],
            "maxTokens": max_tokens,
        }

        if system_prompt is not None:
            params["systemPrompt"] = system_prompt
        if temperature is not None:
            params["temperature"] = temperature
        if model_preferences is not None:
            if hasattr(model_preferences, "model_dump"):
                params["modelPreferences"] = model_preferences.model_dump()
            else:
                params["modelPreferences"] = model_preferences
        if stop_sequences is not None:
            params["stopSequences"] = stop_sequences
        if metadata is not None:
            params["metadata"] = metadata
        if include_context is not None:
            params["includeContext"] = include_context
        if tools is not None:
            params["tools"] = tools
        if tool_choice is not None:
            params["toolChoice"] = tool_choice

        # Build JSON-RPC request
        request_id = f"sampling-{uuid.uuid4().hex[:12]}"
        request = {
            JSONRPC_KEY: JSONRPC_VERSION,
            KEY_ID: request_id,
            KEY_METHOD: McpMethod.SAMPLING_CREATE_MESSAGE,
            KEY_PARAMS: params,
        }

        # Send to client and await response
        response = await self._send_to_client(request)

        # Validate response
        if KEY_ERROR in response:
            error = response[KEY_ERROR]
            raise RuntimeError(f"Sampling request failed: {error.get('message', 'Unknown error')}")

        result = response.get(KEY_RESULT, {})
        return result

    async def send_elicitation_request(
        self,
        message: str,
        schema: dict[str, Any],
        title: str | None = None,
        description: str | None = None,
    ) -> dict[str, Any]:
        """
        Send an elicitation/create request to the MCP client.

        Args:
            message: Human-readable explanation of what input is needed
            schema: JSON Schema defining expected response structure
            title: Optional title for the input dialog
            description: Optional longer description

        Returns:
            Dict with the user's structured response

        Raises:
            RuntimeError: If transport callback is not available
        """
        if self._send_to_client is None:
            raise RuntimeError("No transport callback available for sending elicitation requests")

        params: dict[str, Any] = {
            "message": message,
            "requestedSchema": schema,
        }
        if title is not None:
            params["title"] = title
        if description is not None:
            params["description"] = description

        request_id = f"elicit-{uuid.uuid4().hex[:12]}"
        request = {
            JSONRPC_KEY: JSONRPC_VERSION,
            KEY_ID: request_id,
            KEY_METHOD: McpMethod.ELICITATION_CREATE,
            KEY_PARAMS: params,
        }

        response = await self._send_to_client(request)

        if KEY_ERROR in response:
            error = response[KEY_ERROR]
            raise RuntimeError(f"Elicitation request failed: {error.get('message', 'Unknown error')}")

        return response.get(KEY_RESULT, {})

    async def send_progress_notification(
        self,
        progress_token: str | int,
        progress: float,
        total: float | None = None,
        message: str | None = None,
    ) -> None:
        """
        Send a progress notification to the MCP client.

        This is a fire-and-forget notification (no response expected).

        Args:
            progress_token: Token from the request's _meta.progressToken
            progress: Current progress value
            total: Optional total expected progress value
            message: Optional human-readable progress message
        """
        if self._send_to_client is None:
            return

        params: dict[str, Any] = {
            "progressToken": progress_token,
            "progress": progress,
        }
        if total is not None:
            params["total"] = total
        if message is not None:
            params["message"] = message

        # Notification — no id field
        notification = {
            JSONRPC_KEY: JSONRPC_VERSION,
            KEY_METHOD: McpMethod.NOTIFICATIONS_PROGRESS,
            KEY_PARAMS: params,
        }

        await self._send_to_client(notification)

    async def send_roots_request(self) -> list[dict[str, Any]]:
        """
        Send a roots/list request to the MCP client.

        Returns:
            List of Root dicts with uri and optional name

        Raises:
            RuntimeError: If transport callback is not available
        """
        if self._send_to_client is None:
            raise RuntimeError("No transport callback available for sending roots requests")

        request_id = f"roots-{uuid.uuid4().hex[:12]}"
        request = {
            JSONRPC_KEY: JSONRPC_VERSION,
            KEY_ID: request_id,
            KEY_METHOD: McpMethod.ROOTS_LIST,
            KEY_PARAMS: {},
        }

        response = await self._send_to_client(request)

        if KEY_ERROR in response:
            error = response[KEY_ERROR]
            raise RuntimeError(f"Roots request failed: {error.get('message', 'Unknown error')}")

        result = response.get(KEY_RESULT, {})
        return result.get("roots", [])

    async def notify_resource_updated(self, uri: str) -> None:
        """
        Send a resource updated notification to subscribed clients.

        Args:
            uri: URI of the resource that was updated
        """
        if self._send_to_client is None:
            return

        # Find all sessions subscribed to this URI
        for session_id, uris in self._resource_subscriptions.items():
            if uri in uris:
                notification = {
                    JSONRPC_KEY: JSONRPC_VERSION,
                    KEY_METHOD: McpMethod.NOTIFICATIONS_RESOURCES_UPDATED,
                    KEY_PARAMS: {"uri": uri},
                }
                try:
                    await self._send_to_client(notification)
                except Exception as e:
                    logger.debug(f"Failed to notify session {session_id[:8]}... of resource update: {e}")

    async def send_log_notification(
        self,
        level: str = "info",
        data: Any = None,
        logger_name: str | None = None,
    ) -> None:
        """
        Send a log message notification to the client (notifications/message).

        This is a fire-and-forget notification. Safe to call even if no transport
        callback is available (will silently no-op).

        Args:
            level: Log level (debug, info, notice, warning, error, critical, alert, emergency)
            data: Log data (any JSON-serializable value)
            logger_name: Optional logger name for filtering
        """
        if self._send_to_client is None:
            return

        params: dict[str, Any] = {"level": level, "data": data}
        if logger_name is not None:
            params["logger"] = logger_name

        notification = {
            JSONRPC_KEY: JSONRPC_VERSION,
            KEY_METHOD: McpMethod.NOTIFICATIONS_MESSAGE,
            KEY_PARAMS: params,
        }

        try:
            await self._send_to_client(notification)
        except Exception as e:
            logger.debug(f"Failed to send log notification: {e}")

    def _handle_cancelled_notification(self, params: dict[str, Any]) -> None:
        """
        Handle notifications/cancelled — cancel an in-flight request.

        The client sends this when it no longer needs the response for a
        previous request. We look up the asyncio task and cancel it.

        Args:
            params: Must contain "requestId"; may contain "reason".
        """
        request_id = params.get("requestId")
        reason = params.get("reason", "")
        if request_id is None:
            logger.debug("Received cancelled notification without requestId")
            return

        task = self._in_flight_requests.pop(request_id, None)
        if task is not None:
            task.cancel()
            logger.debug(f"Cancelled request {request_id}" + (f": {reason}" if reason else ""))
        else:
            logger.debug(f"Cancelled notification for unknown request {request_id}")

    async def _handle_completion_complete(self, params: dict[str, Any], msg_id: Any) -> tuple[dict[str, Any], None]:
        """Handle completion/complete request."""
        ref = params.get("ref", {})
        argument = params.get("argument", {})
        ref_type = ref.get("type", "")

        # Look up provider for this reference type
        provider = self.completion_providers.get(ref_type)
        if provider is None:
            # No provider — return empty completions
            result = {"completion": {"values": [], "hasMore": False}}
            return {JSONRPC_KEY: JSONRPC_VERSION, KEY_ID: msg_id, KEY_RESULT: result}, None

        try:
            completion = await provider(ref, argument)
            if isinstance(completion, dict):
                result = {"completion": completion}
            else:
                result = {"completion": {"values": [], "hasMore": False}}

            return {JSONRPC_KEY: JSONRPC_VERSION, KEY_ID: msg_id, KEY_RESULT: result}, None

        except Exception as e:
            logger.error(f"Completion error: {e}")
            return self._create_error_response(msg_id, JsonRpcError.INTERNAL_ERROR, f"Completion error: {str(e)}"), None

    async def _handle_resources_subscribe(self, params: dict[str, Any], msg_id: Any) -> tuple[dict[str, Any], None]:
        """Handle resources/subscribe request."""
        uri = params.get("uri", "")

        from .context import get_session_id

        session_id = get_session_id()
        if session_id:
            self._resource_subscriptions.setdefault(session_id, set()).add(uri)
            logger.debug(f"Session {session_id[:8]}... subscribed to {uri}")

        return {JSONRPC_KEY: JSONRPC_VERSION, KEY_ID: msg_id, KEY_RESULT: {}}, None

    async def _handle_resources_unsubscribe(self, params: dict[str, Any], msg_id: Any) -> tuple[dict[str, Any], None]:
        """Handle resources/unsubscribe request."""
        uri = params.get("uri", "")

        from .context import get_session_id

        session_id = get_session_id()
        if session_id and session_id in self._resource_subscriptions:
            self._resource_subscriptions[session_id].discard(uri)
            logger.debug(f"Session {session_id[:8]}... unsubscribed from {uri}")

        return {JSONRPC_KEY: JSONRPC_VERSION, KEY_ID: msg_id, KEY_RESULT: {}}, None

    async def _handle_resources_list(self, params: dict[str, Any], msg_id: Any) -> tuple[dict[str, Any], None]:
        """Handle resources/list request with pagination support."""
        resources_list = self.get_resources_list()
        result = self._paginate(resources_list, "resources", params)

        response = {JSONRPC_KEY: JSONRPC_VERSION, KEY_ID: msg_id, KEY_RESULT: result}

        logger.debug(f"Returning {len(result['resources'])} resources")
        return response, None

    async def _handle_resources_read(self, params: dict[str, Any], msg_id: Any) -> tuple[dict[str, Any], None]:
        """Handle resources/read request."""
        uri = params.get("uri")

        if uri not in self.resources:
            return self._create_error_response(msg_id, JsonRpcError.INVALID_PARAMS, f"Unknown resource: {uri}"), None

        try:
            resource_handler = self.resources[uri]
            content = await resource_handler.read()

            # Build resource content response
            resource_content = {"uri": uri, "mimeType": resource_handler.mime_type, "text": content}

            response = {JSONRPC_KEY: JSONRPC_VERSION, KEY_ID: msg_id, KEY_RESULT: {"contents": [resource_content]}}

            logger.debug(f"📖 Read resource {uri}")
            return response, None

        except Exception as e:
            logger.error(f"Resource read error for {uri}: {e}")
            return self._create_error_response(
                msg_id, JsonRpcError.INTERNAL_ERROR, f"Resource read error: {str(e)}"
            ), None

    async def _handle_resources_templates_list(
        self, params: dict[str, Any], msg_id: Any
    ) -> tuple[dict[str, Any], None]:
        """Handle resources/templates/list request with pagination support."""
        templates_list = self.get_resource_templates_list()
        result = self._paginate(templates_list, "resourceTemplates", params)

        response = {JSONRPC_KEY: JSONRPC_VERSION, KEY_ID: msg_id, KEY_RESULT: result}

        logger.debug(f"Returning {len(result['resourceTemplates'])} resource templates")
        return response, None

    async def _handle_prompts_list(self, params: dict[str, Any], msg_id: Any) -> tuple[dict[str, Any], None]:
        """Handle prompts/list request with pagination support."""
        prompts_list = self.get_prompts_list()
        result = self._paginate(prompts_list, "prompts", params)

        response = {JSONRPC_KEY: JSONRPC_VERSION, KEY_ID: msg_id, KEY_RESULT: result}

        logger.info(f"Returning {len(result['prompts'])} prompts")
        return response, None

    async def _handle_prompts_get(self, params: dict[str, Any], msg_id: Any) -> tuple[dict[str, Any], None]:
        """Handle prompts/get request."""
        prompt_name = params.get("name")
        arguments = params.get("arguments", {})

        if prompt_name not in self.prompts:
            return self._create_error_response(
                msg_id, JsonRpcError.INVALID_PARAMS, f"Unknown prompt: {prompt_name}"
            ), None

        try:
            prompt_handler = self.prompts[prompt_name]
            result = await prompt_handler.get_prompt(arguments)

            # Format response content
            if isinstance(result, str):
                # If result is a string, wrap it as MCP content
                content = format_content(result)
            elif isinstance(result, dict):
                # If result is a dict, it should contain the prompt messages
                content = [result] if not isinstance(result.get("messages"), list) else result.get("messages", [])
            else:
                # Convert other types to string content
                content = format_content(str(result))

            # Ensure proper message format for MCP Inspector
            if isinstance(content, list) and len(content) > 0:
                # Convert content format to proper MCP message format
                messages = []
                for item in content:
                    if isinstance(item, dict):
                        if "role" in item and "content" in item:
                            # Already properly formatted
                            messages.append(item)
                        else:
                            # Convert content item to proper message format
                            messages.append({"role": "user", "content": item})
                    else:
                        # Convert string or other types to proper message format
                        messages.append({"role": "user", "content": {"type": "text", "text": str(item)}})
            else:
                # Fallback to properly formatted message structure
                messages = [{"role": "user", "content": {"type": "text", "text": str(result)}}]

            response = {
                JSONRPC_KEY: JSONRPC_VERSION,
                KEY_ID: msg_id,
                KEY_RESULT: {
                    "description": prompt_handler.description or f"Prompt: {prompt_name}",
                    "messages": messages,
                },
            }

            logger.debug(f"💬 Generated prompt {prompt_name}")
            logger.debug(f"🔍 DEBUG Response messages: {messages}")
            return response, None

        except Exception as e:
            logger.error(f"Prompt generation error for {prompt_name}: {e}")
            return self._create_error_response(
                msg_id, JsonRpcError.INTERNAL_ERROR, f"Prompt generation error: {str(e)}"
            ), None

    async def _handle_logging_set_level(self, params: dict[str, Any], msg_id: Any) -> tuple[dict[str, Any], None]:
        """Handle logging/setLevel request."""
        level = params.get("level", "INFO")

        # Map MCP logging levels to Python logging levels
        level_mapping = {
            LOG_DEBUG: logging.DEBUG,
            LOG_INFO: logging.INFO,
            LOG_WARNING: logging.WARNING,
            LOG_ERROR: logging.ERROR,
            LOG_CRITICAL: logging.CRITICAL,
        }

        level_lower = level.lower()
        if level_lower not in level_mapping:
            return self._create_error_response(
                msg_id,
                JsonRpcError.INVALID_PARAMS,
                f"Invalid logging level: {level}. Must be one of: debug, info, warning, error",
            ), None

        # Set the logging level for the chuk_mcp_server logger
        numeric_level = level_mapping[level_lower]
        logging.getLogger(PACKAGE_LOGGER).setLevel(numeric_level)

        # Also set the level for the root logger if needed
        if level_lower == LOG_DEBUG:
            logging.getLogger().setLevel(logging.DEBUG)

        logger.debug(f"Logging level set to {level.upper()}")

        response = {
            JSONRPC_KEY: JSONRPC_VERSION,
            KEY_ID: msg_id,
            KEY_RESULT: {"level": level.upper(), "message": f"Logging level set to {level.upper()}"},
        }

        return response, None

    @staticmethod
    def _paginate(items: list[dict[str, Any]], key: str, params: dict[str, Any]) -> dict[str, Any]:
        """Apply cursor-based pagination to a list of items.

        Args:
            items: Full list of items.
            key: Result key name (e.g., "tools", "resources", "prompts").
            params: Request params (may contain "cursor").

        Returns:
            Result dict with paginated items and optional nextCursor.
        """
        from .constants import DEFAULT_PAGE_SIZE, KEY_CURSOR, KEY_NEXT_CURSOR

        cursor = params.get(KEY_CURSOR)
        offset = 0

        if cursor is not None:
            import base64

            try:
                offset = int(base64.b64decode(cursor).decode())
            except (ValueError, Exception):
                offset = 0  # Invalid cursor, start from beginning

        page = items[offset : offset + DEFAULT_PAGE_SIZE]
        result: dict[str, Any] = {key: page}

        if offset + DEFAULT_PAGE_SIZE < len(items):
            import base64

            next_offset = offset + DEFAULT_PAGE_SIZE
            result[KEY_NEXT_CURSOR] = base64.b64encode(str(next_offset).encode()).decode()

        return result

    # ================================================================
    # Session lifecycle helpers
    # ================================================================

    def _cleanup_session_state(self, session_id: str) -> None:
        """Clean up all per-session state (subscriptions, SSE buffers, etc.).

        Called both by explicit terminate_session() and by SessionManager
        eviction/expiry callbacks to prevent memory leaks.
        """
        self._resource_subscriptions.pop(session_id, None)
        self._sse_event_buffers.pop(session_id, None)
        self._sse_event_counters.pop(session_id, None)
        if self._rate_limiter is not None:
            self._rate_limiter.cleanup(session_id)
        logger.debug(f"Cleaned up state for session {session_id[:8]}...")

    def _get_protected_sessions(self) -> set[str]:
        """Return session IDs that should not be evicted (have in-flight requests)."""
        protected: set[str] = set()
        # In-flight requests are keyed by request_id, not session_id.
        # Walk the task store to find sessions with active work.
        for task in self._task_store.values():
            if task.get("status") == "working":
                request_id = task.get("requestId")
                if request_id is not None and request_id in self._in_flight_requests:
                    # The task doesn't store session_id directly, but we protect
                    # any session that still has in-flight work by checking the
                    # SSE buffers (only HTTP sessions using SSE have them).
                    pass
        # Simpler heuristic: protect sessions that have SSE event counters,
        # since those are actively streaming.
        for sid in self._sse_event_counters:
            if sid in self.session_manager.sessions:
                protected.add(sid)
        return protected

    # ================================================================
    # Session termination (MCP 2025-11-25 Streamable HTTP)
    # ================================================================

    def terminate_session(self, session_id: str) -> bool:
        """Terminate a session and clean up resources.

        Args:
            session_id: The session ID to terminate.

        Returns:
            True if the session was found and terminated, False otherwise.
        """
        session = self.session_manager.get_session(session_id)
        if session is None:
            return False

        # Clean up all per-session state
        self._cleanup_session_state(session_id)

        # Remove session
        self.session_manager.sessions.pop(session_id, None)
        logger.debug(f"Terminated session {session_id[:8]}...")
        return True

    def next_sse_event_id(self, session_id: str) -> int:
        """Get next SSE event ID for a session."""
        counter = self._sse_event_counters.get(session_id, 0) + 1
        self._sse_event_counters[session_id] = counter
        return counter

    def buffer_sse_event(self, session_id: str, event_id: int, data: dict[str, Any]) -> None:
        """Buffer an SSE event for resumability."""
        buf = self._sse_event_buffers.setdefault(session_id, [])
        buf.append((event_id, data))
        # Keep only last 100 events
        if len(buf) > 100:
            self._sse_event_buffers[session_id] = buf[-100:]

    def get_missed_events(self, session_id: str, last_event_id: int) -> list[tuple[int, dict[str, Any]]]:
        """Get events after the given event ID for resumability."""
        buf = self._sse_event_buffers.get(session_id, [])
        return [(eid, data) for eid, data in buf if eid > last_event_id]

    # ================================================================
    # Tasks system (MCP 2025-11-25)
    # ================================================================

    def _create_task(self, request_id: Any, tool_name: str) -> str:
        """Create a task for a tool execution."""
        task_id = str(uuid.uuid4()).replace("-", "")[:16]
        self._task_store[task_id] = {
            "id": task_id,
            "status": "working",
            "requestId": request_id,
            "toolName": tool_name,
            "createdAt": time.time(),
            "updatedAt": time.time(),
            "result": None,
            "error": None,
            "message": None,
        }
        return task_id

    def _update_task_status(
        self,
        task_id: str,
        status: str,
        result: dict[str, Any] | None = None,
        error: dict[str, Any] | None = None,
        message: str | None = None,
    ) -> None:
        """Update a task's status."""
        task = self._task_store.get(task_id)
        if task is None:
            return
        task["status"] = status
        task["updatedAt"] = time.time()
        if result is not None:
            task["result"] = result
        if error is not None:
            task["error"] = error
        if message is not None:
            task["message"] = message

    async def _handle_tasks_get(self, params: dict[str, Any], msg_id: Any) -> tuple[dict[str, Any], None]:
        """Handle tasks/get request."""
        task_id = params.get("id", "")
        task = self._task_store.get(task_id)
        if task is None:
            return self._create_error_response(msg_id, JsonRpcError.INVALID_PARAMS, f"Unknown task: {task_id}"), None
        return {JSONRPC_KEY: JSONRPC_VERSION, KEY_ID: msg_id, KEY_RESULT: task}, None

    async def _handle_tasks_result(self, params: dict[str, Any], msg_id: Any) -> tuple[dict[str, Any], None]:
        """Handle tasks/result request."""
        task_id = params.get("id", "")
        task = self._task_store.get(task_id)
        if task is None:
            return self._create_error_response(msg_id, JsonRpcError.INVALID_PARAMS, f"Unknown task: {task_id}"), None
        if task["status"] not in ("completed", "failed"):
            return self._create_error_response(
                msg_id, JsonRpcError.INVALID_PARAMS, f"Task {task_id} is not yet complete (status: {task['status']})"
            ), None
        return {JSONRPC_KEY: JSONRPC_VERSION, KEY_ID: msg_id, KEY_RESULT: task}, None

    async def _handle_tasks_list(self, params: dict[str, Any], msg_id: Any) -> tuple[dict[str, Any], None]:
        """Handle tasks/list request with pagination."""
        tasks_list = list(self._task_store.values())
        result = self._paginate(tasks_list, "tasks", params)
        return {JSONRPC_KEY: JSONRPC_VERSION, KEY_ID: msg_id, KEY_RESULT: result}, None

    async def _handle_tasks_cancel(self, params: dict[str, Any], msg_id: Any) -> tuple[dict[str, Any], None]:
        """Handle tasks/cancel request."""
        task_id = params.get("id", "")
        task = self._task_store.get(task_id)
        if task is None:
            return self._create_error_response(msg_id, JsonRpcError.INVALID_PARAMS, f"Unknown task: {task_id}"), None
        if task["status"] in ("completed", "failed", "cancelled"):
            return self._create_error_response(
                msg_id, JsonRpcError.INVALID_PARAMS, f"Task {task_id} is already in terminal state: {task['status']}"
            ), None
        self._update_task_status(task_id, "cancelled")
        # Also cancel the in-flight request if tracked
        request_id = task.get("requestId")
        if request_id is not None:
            in_flight = self._in_flight_requests.pop(request_id, None)
            if in_flight is not None:
                in_flight.cancel()
        return {JSONRPC_KEY: JSONRPC_VERSION, KEY_ID: msg_id, KEY_RESULT: task}, None

    async def send_task_status_notification(self, task_id: str) -> None:
        """Send a notifications/tasks/status to the client."""
        if self._send_to_client is None:
            return
        task = self._task_store.get(task_id)
        if task is None:
            return
        notification = {
            JSONRPC_KEY: JSONRPC_VERSION,
            KEY_METHOD: McpTaskMethod.NOTIFICATIONS_TASKS_STATUS,
            KEY_PARAMS: task,
        }
        try:
            await self._send_to_client(notification)
        except Exception as e:
            logger.debug(f"Failed to send task status notification: {e}")

    async def shutdown(self, timeout: float = 5.0) -> None:
        """Gracefully shut down the protocol handler.

        Waits for in-flight requests to complete (up to timeout), then cancels
        any remaining and cleans up all session state.

        Args:
            timeout: Maximum seconds to wait for in-flight requests.
        """
        # Wait for in-flight requests to finish
        if self._in_flight_requests:
            logger.debug(f"Waiting for {len(self._in_flight_requests)} in-flight requests (timeout={timeout}s)")
            tasks = list(self._in_flight_requests.values())
            done, pending = await asyncio.wait(tasks, timeout=timeout)
            for t in pending:
                t.cancel()
            logger.debug(f"Shutdown: {len(done)} completed, {len(pending)} cancelled")

        self._in_flight_requests.clear()

        # Clean up all session state
        for sid in list(self.session_manager.sessions):
            self._cleanup_session_state(sid)
        self.session_manager.sessions.clear()

        # Clear task store
        self._task_store.clear()

        logger.debug("Protocol handler shut down")

    def _create_error_response(self, msg_id: Any, code: int, message: str) -> dict[str, Any]:
        """Create error response."""
        return {JSONRPC_KEY: JSONRPC_VERSION, KEY_ID: msg_id, KEY_ERROR: {"code": code, "message": message}}
