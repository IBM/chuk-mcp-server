#!/usr/bin/env python3
# src/chuk_mcp_server/core.py
"""
Core - Main ChukMCP Server class with modular smart configuration
"""

import logging
import sys
from collections.abc import Callable
from typing import Any

# Import the modular smart configuration system
from .config import SmartConfig
from .decorators import (
    clear_global_registry,
    get_global_prompts,
    get_global_resource_templates,
    get_global_resources,
    get_global_tools,
)
from .endpoint_registry import http_endpoint_registry
from .http_server import create_server
from .mcp_registry import mcp_registry
from .protocol import MCPProtocolHandler
from .proxy import ProxyManager
from .stdio_transport import StdioSyncTransport

# Updated imports for clean types API
from .types import (
    PromptHandler,
    ResourceHandler,
    # Direct chuk_mcp types
    ServerInfo,
    # Framework handlers
    ToolHandler,
    create_server_capabilities,
)
from .types.resources import ResourceTemplateHandler

logger = logging.getLogger(__name__)


# ============================================================================
# Main ChukMCPServer Class with Modular Smart Configuration
# ============================================================================


class ChukMCPServer:
    """
    ChukMCPServer - Zero configuration MCP framework with modular smart configuration.

    Usage:
        # Zero config - everything auto-detected using modular system
        mcp = ChukMCPServer()

        @mcp.tool
        def hello(name: str) -> str:
            return f"Hello, {name}!"

        mcp.run()  # Auto-detects host, port, performance settings using SmartConfig
    """

    def __init__(
        self,
        name: str | None = None,
        version: str = "1.0.0",
        title: str | None = None,
        description: str | None = None,
        icons: list[dict[str, Any]] | None = None,
        website_url: str | None = None,
        capabilities=None,
        tools: bool = True,
        resources: bool = True,
        prompts: bool = False,
        logging: bool = False,
        completions: bool = False,
        experimental: dict[str, Any] | None = None,
        # Smart defaults (all optional - will use SmartConfig)
        host: str | None = None,
        port: int | None = None,
        debug: bool | None = None,
        transport: str | None = None,  # Added transport parameter
        # Proxy configuration
        proxy_config: dict[str, Any] | None = None,
        # Tool modules configuration
        tool_modules_config: dict[str, Any] | None = None,
        **kwargs,  # noqa: ARG002
    ):
        """
        Initialize ChukMCP Server with modular smart configuration.

        Args:
            name: Server name (auto-detected if None)
            version: Server version
            title: Optional server title
            description: Optional server description
            capabilities: ServerCapabilities object (auto-configured if None)
            tools: Enable tools capability
            resources: Enable resources capability
            prompts: Enable prompts capability
            logging: Enable logging capability
            experimental: Experimental capabilities
            host: Host to bind to (auto-detected if None)
            port: Port to bind to (auto-detected if None)
            debug: Debug mode (auto-detected if None)
            transport: Transport mode ('http' or 'stdio') (auto-detected if None)
            proxy_config: Configuration for multi-server proxy
            tool_modules_config: Configuration for loading tool modules
            **kwargs: Additional keyword arguments
        """
        # Initialize the modular smart configuration system
        self.smart_config = SmartConfig()

        # Get all smart defaults in one efficient call
        smart_defaults = self.smart_config.get_all_defaults()

        # Auto-detect name if not provided
        if name is None:
            name = smart_defaults["project_name"]

        # Use chuk_mcp ServerInfo directly
        self.server_info = ServerInfo(name=name, version=version, title=title)

        # Handle capabilities flexibly
        if capabilities is not None:
            self.capabilities = capabilities
        else:
            self.capabilities = create_server_capabilities(
                tools=tools,
                resources=resources,
                prompts=prompts,
                logging=logging,
                completions=completions,
                experimental=experimental,
            )

        # Store smart defaults for run() - using modular config
        self.smart_host = host or smart_defaults["host"]
        self.smart_port = port or smart_defaults["port"]
        self.smart_debug = debug if debug is not None else smart_defaults["debug"]
        self.smart_transport = transport  # Store transport preference

        # Store additional smart configuration
        self.smart_environment = smart_defaults["environment"]
        self.smart_workers = smart_defaults["workers"]
        self.smart_max_connections = smart_defaults["max_connections"]
        self.smart_log_level = smart_defaults["log_level"]
        self.smart_performance_mode = smart_defaults["performance_mode"]
        self.smart_containerized = smart_defaults["containerized"]
        self.smart_transport_mode = smart_defaults.get("transport_mode", "http")

        # Build extra server info fields for MCP 2025-11-25
        extra_server_info: dict[str, Any] = {}
        if description is not None:
            extra_server_info["description"] = description
        if icons is not None:
            extra_server_info["icons"] = icons
        if website_url is not None:
            extra_server_info["websiteUrl"] = website_url

        # Create protocol handler with direct chuk_mcp types
        self.protocol = MCPProtocolHandler(
            self.server_info, self.capabilities, extra_server_info=extra_server_info or None
        )

        # Register any globally decorated functions
        self._register_global_functions()

        # HTTP server will be created when needed
        self._server = None

        # Proxy manager for multi-server support
        self.proxy_manager: ProxyManager | None = None
        if proxy_config:
            self.proxy_manager = ProxyManager(proxy_config, self.protocol)

        # Module loader for hosting multiple tool collections
        self.module_loader: Any = None
        if tool_modules_config:
            from .modules import ModuleLoader

            # Wrap config in proper structure (ModuleLoader expects {"tool_modules": {...}})
            config = {"tool_modules": tool_modules_config}
            self.module_loader = ModuleLoader(config, self)

        # Composition manager for unified server composition
        from .composition import CompositionManager

        self.composition = CompositionManager(self)

        # Don't print banner during init - will be done in run() if needed
        # This avoids cluttering stdio mode
        self._should_print_config = self.smart_debug

        logger.debug(f"Initialized ChukMCP Server: {name} v{version}")

    def _print_smart_config(self, actual_log_level: str | None = None):
        """Print smart configuration summary using modular config."""
        import sys

        # Always use stderr for debug output to keep stdout clean
        output = sys.stderr
        log_level_display = actual_log_level or self.smart_log_level
        print("🧠 ChukMCPServer - Modular Zero Configuration Mode", file=output)
        print("=" * 60, file=output)
        print(f"📊 Environment: {self.smart_environment}", file=output)
        print(f"🌐 Network: {self.smart_host}:{self.smart_port}", file=output)
        print(f"🔧 Workers: {self.smart_workers}", file=output)
        print(f"🔗 Max Connections: {self.smart_max_connections}", file=output)
        print(f"🐳 Container: {self.smart_containerized}", file=output)
        print(f"⚡ Performance Mode: {self.smart_performance_mode}", file=output)
        print(f"📝 Log Level: {log_level_display}", file=output)
        print("=" * 60, file=output)

    def _register_global_functions(self):
        """Register globally decorated functions in both protocol and registries."""
        # Register global tools
        for tool_handler in get_global_tools():
            self.protocol.register_tool(tool_handler)
            mcp_registry.register_tool(tool_handler.name, tool_handler)

        # Register global resources
        for resource_handler in get_global_resources():
            self.protocol.register_resource(resource_handler)
            mcp_registry.register_resource(resource_handler.uri, resource_handler)

        # Register global prompts
        for prompt_handler in get_global_prompts():
            self.protocol.register_prompt(prompt_handler)
            mcp_registry.register_prompt(prompt_handler.name, prompt_handler)

        # Register global resource templates
        for template_handler in get_global_resource_templates():
            self.protocol.register_resource_template(template_handler)

        # Clear global registry to avoid duplicate registrations
        clear_global_registry()

    # ============================================================================
    # Tool Registration
    # ============================================================================

    def tool(self, name: str | None = None, description: str | None = None, **kwargs):
        """
        Tool decorator with simple registration.

        Usage:
            @mcp.tool
            def hello(name: str) -> str:
                return f"Hello, {name}!"

            @mcp.tool(tags=["custom"])
            def advanced_tool(data: dict) -> dict:
                return {"processed": data}

            @mcp.tool(read_only_hint=True, idempotent_hint=True)
            def safe_lookup(key: str) -> str:
                return db[key]
        """

        def decorator(func: Callable) -> Callable:
            # Simple tool creation
            tool_name = name or func.__name__
            tool_description = description or func.__doc__ or f"Execute {tool_name}"

            # Extract annotation, output_schema, and icons kwargs
            annotation_kwargs = {}
            for key in (
                "read_only_hint",
                "destructive_hint",
                "idempotent_hint",
                "open_world_hint",
                "output_schema",
                "icons",
            ):
                if key in kwargs:
                    annotation_kwargs[key] = kwargs.pop(key)

            # Create tool handler from function
            tool_handler = ToolHandler.from_function(
                func, name=tool_name, description=tool_description, **annotation_kwargs
            )

            # Register in protocol handler (for MCP functionality)
            self.protocol.register_tool(tool_handler)

            # Simple metadata for registry
            metadata = {"function_name": func.__name__, "parameter_count": len(tool_handler.parameters)}

            # Simple tags
            tags = ["tool"]
            if "tags" in kwargs:
                tags.extend(kwargs.pop("tags"))

            # Register in MCP registry
            mcp_registry.register_tool(tool_handler.name, tool_handler, metadata=metadata, tags=tags, **kwargs)

            # Add tool metadata to function
            func._mcp_tool = tool_handler

            logger.debug(f"Registered tool: {tool_handler.name}")
            return func

        # Handle both @mcp.tool and @mcp.tool() usage
        if callable(name):
            func = name
            name = None
            return decorator(func)
        else:
            return decorator

    def resource(
        self, uri: str, name: str | None = None, description: str | None = None, mime_type: str | None = None, **kwargs
    ):
        """
        Resource decorator with simple registration.

        Usage:
            @mcp.resource("config://settings")
            def get_settings() -> dict:
                return {"app": "my_app"}
        """

        def decorator(func: Callable) -> Callable:
            # Simple resource creation
            resource_name = name or func.__name__.replace("_", " ").title()
            resource_description = description or func.__doc__ or f"Resource: {uri}"
            resource_mime_type = mime_type or "application/json"  # Simple default

            # Extract icons kwarg for resource handler
            handler_kwargs = {}
            if "icons" in kwargs:
                handler_kwargs["icons"] = kwargs.pop("icons")

            # Create resource handler from function
            resource_handler = ResourceHandler.from_function(
                uri=uri,
                func=func,
                name=resource_name,
                description=resource_description,
                mime_type=resource_mime_type,
                **handler_kwargs,
            )

            # Register in protocol handler (for MCP functionality)
            self.protocol.register_resource(resource_handler)

            # Simple metadata for registry
            metadata = {
                "function_name": func.__name__,
                "mime_type": resource_mime_type,
                "uri_scheme": uri.split("://")[0] if "://" in uri else "unknown",
            }

            # Simple tags
            tags = ["resource"]
            if "tags" in kwargs:
                tags.extend(kwargs.pop("tags"))

            # Register in MCP registry
            mcp_registry.register_resource(
                resource_handler.uri, resource_handler, metadata=metadata, tags=tags, **kwargs
            )

            # Add resource metadata to function
            func._mcp_resource = resource_handler

            logger.debug(f"Registered resource: {resource_handler.uri}")
            return func

        return decorator

    def resource_template(
        self,
        uri_template: str,
        name: str | None = None,
        description: str | None = None,
        mime_type: str | None = None,
        **kwargs,
    ):
        """
        Resource template decorator for parameterized resources (RFC 6570).

        Usage:
            @mcp.resource_template("users://{user_id}/profile")
            def get_user_profile(user_id: str) -> dict:
                return {"id": user_id, "name": "Alice"}
        """

        def decorator(func: Callable) -> Callable:
            # Extract icons kwarg for template handler
            handler_kwargs = {}
            if "icons" in kwargs:
                handler_kwargs["icons"] = kwargs.pop("icons")

            template_handler = ResourceTemplateHandler.from_function(
                uri_template=uri_template,
                func=func,
                name=name,
                description=description,
                mime_type=mime_type,
                **handler_kwargs,
            )

            # Register in protocol handler
            self.protocol.register_resource_template(template_handler)

            # Add metadata to function
            func._mcp_resource_template = template_handler

            logger.debug(f"Registered resource template: {uri_template}")
            return func

        return decorator

    def prompt(self, name: str | None = None, description: str | None = None, **kwargs):
        """
        Prompt decorator with simple registration.

        Usage:
            @mcp.prompt
            def code_review(code: str, language: str = "python") -> str:
                return f"Please review this {language} code:\\n\\n{code}"

            @mcp.prompt(name="custom_prompt", description="Custom prompt template")
            def my_prompt(topic: str, style: str = "formal") -> str:
                return f"Write about {topic} in a {style} style"
        """

        def decorator(func: Callable) -> Callable:
            # Simple prompt creation
            prompt_name = name or func.__name__
            prompt_description = description or func.__doc__ or f"Prompt: {prompt_name}"

            # Extract icons kwarg for prompt handler
            handler_kwargs = {}
            if "icons" in kwargs:
                handler_kwargs["icons"] = kwargs.pop("icons")

            # Create prompt handler from function
            prompt_handler = PromptHandler.from_function(
                func, name=prompt_name, description=prompt_description, **handler_kwargs
            )

            # Register in protocol handler (for MCP functionality)
            self.protocol.register_prompt(prompt_handler)

            # Simple metadata for registry
            metadata = {"function_name": func.__name__, "parameter_count": len(prompt_handler.parameters)}

            # Simple tags
            tags = ["prompt"]
            if "tags" in kwargs:
                tags.extend(kwargs.pop("tags"))

            # Register in MCP registry
            mcp_registry.register_prompt(prompt_handler.name, prompt_handler, metadata=metadata, tags=tags, **kwargs)

            # Add prompt metadata to function
            func._mcp_prompt = prompt_handler

            logger.debug(f"Registered prompt: {prompt_handler.name}")
            return func

        # Handle both @mcp.prompt and @mcp.prompt() usage
        if callable(name):
            func = name
            name = None
            return decorator(func)
        else:
            return decorator

    # ============================================================================
    # Server Composition Methods
    # ============================================================================

    def import_server(
        self,
        server: "ChukMCPServer",
        prefix: str | None = None,
        components: list[str] | None = None,
        tags: list[str] | None = None,
    ) -> None:
        """
        Import (copy) components from another MCP server.

        Static composition: creates a one-time copy of tools, resources, and prompts.
        Changes to the original server after import will NOT be reflected.

        Args:
            server: ChukMCPServer instance to import from
            prefix: Optional prefix for namespacing (e.g., "weather" -> "weather.get_forecast")
            components: List of component types ["tools", "resources", "prompts"]
                       If None, imports all components
            tags: Optional tags to filter components

        Example:
            # Import weather server with prefix
            main_mcp.import_server(weather_server, prefix="weather")

            # Import only tools
            main_mcp.import_server(api_server, prefix="api", components=["tools"])

            # Import filtered by tags
            main_mcp.import_server(data_server, tags=["public"])
        """
        self.composition.import_server(server, prefix, components, tags)

    def mount(
        self,
        server: "ChukMCPServer | dict[str, Any]",
        prefix: str | None = None,
        as_proxy: bool = False,
    ) -> None:
        """
        Mount a server for live delegation.

        Dynamic composition: creates a live link to another server.
        Changes to the mounted server are reflected immediately.

        Args:
            server: ChukMCPServer instance or proxy config to mount
            prefix: Optional prefix for namespacing
            as_proxy: If True, mount as a proxy (for remote servers)

        Example:
            # Mount local server (live updates)
            main_mcp.mount(api_server)

            # Mount remote server as proxy
            main_mcp.mount(remote_config, prefix="remote", as_proxy=True)

            # Mount with prefix
            main_mcp.mount(data_server, prefix="data")
        """
        self.composition.mount(server, prefix, as_proxy)

    def load_module(self, module_config: dict[str, Any]) -> dict[str, list[str]]:
        """
        Load Python modules with tools.

        Convenience method for loading Python modules through the composition layer.

        Args:
            module_config: Module configuration dictionary

        Returns:
            Dictionary mapping module names to loaded tool names

        Example:
            mcp.load_module({
                "math": {
                    "enabled": True,
                    "location": "./modules",
                    "module": "math_tools.tools",
                    "namespace": "math"
                }
            })
        """
        return self.composition.load_module(module_config)

    def get_composition_stats(self) -> dict[str, Any]:
        """
        Get statistics about composed/mounted/proxied servers.

        Returns:
            Dictionary with composition statistics
        """
        return self.composition.get_composition_stats()

    # ============================================================================
    # HTTP Endpoint Registration
    # ============================================================================

    def endpoint(
        self, path: str, methods: list[str] | None = None, **kwargs: Any
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """
        Decorator to register a custom HTTP endpoint.

        Usage:
            @mcp.endpoint("/api/data", methods=["GET", "POST"])
            async def data_handler(request):
                return Response('{"data": "example"}')
        """

        def decorator(handler: Callable[..., Any]) -> Callable[..., Any]:
            http_endpoint_registry.register_endpoint(path, handler, methods=methods, **kwargs)
            logger.debug(f"Registered endpoint: {path}")
            return handler

        return decorator

    # ============================================================================
    # Manual Registration Methods
    # ============================================================================

    def add_tool(self, tool_handler: ToolHandler, **kwargs):
        """Manually add an MCP tool handler."""
        self.protocol.register_tool(tool_handler)
        mcp_registry.register_tool(tool_handler.name, tool_handler, **kwargs)
        logger.debug(f"Added tool: {tool_handler.name}")

    def add_resource(self, resource_handler: ResourceHandler, **kwargs):
        """Manually add an MCP resource handler."""
        self.protocol.register_resource(resource_handler)
        mcp_registry.register_resource(resource_handler.uri, resource_handler, **kwargs)
        logger.debug(f"Added resource: {resource_handler.uri}")

    def add_prompt(self, prompt_handler: PromptHandler, **kwargs):
        """Manually add an MCP prompt handler."""
        self.protocol.register_prompt(prompt_handler)
        mcp_registry.register_prompt(prompt_handler.name, prompt_handler, **kwargs)
        logger.debug(f"Added prompt: {prompt_handler.name}")

    def add_endpoint(self, path: str, handler: Callable, methods: list[str] | None = None, **kwargs):
        """Manually add a custom HTTP endpoint."""
        http_endpoint_registry.register_endpoint(path, handler, methods=methods, **kwargs)
        logger.debug(f"Added endpoint: {path}")

    def register_function_as_tool(
        self, func: Callable, name: str | None = None, description: str | None = None, **kwargs
    ):
        """Register an existing function as an MCP tool."""
        tool_handler = ToolHandler.from_function(func, name=name, description=description)
        self.add_tool(tool_handler, **kwargs)
        return tool_handler

    def register_function_as_resource(
        self,
        func: Callable,
        uri: str,
        name: str | None = None,
        description: str | None = None,
        mime_type: str = "text/plain",
        **kwargs,
    ):
        """Register an existing function as an MCP resource."""
        resource_handler = ResourceHandler.from_function(
            uri=uri, func=func, name=name, description=description, mime_type=mime_type
        )
        self.add_resource(resource_handler, **kwargs)
        return resource_handler

    def register_function_as_prompt(
        self, func: Callable, name: str | None = None, description: str | None = None, **kwargs
    ):
        """Register an existing function as an MCP prompt."""
        prompt_handler = PromptHandler.from_function(func, name=name, description=description)
        self.add_prompt(prompt_handler, **kwargs)
        return prompt_handler

    # ============================================================================
    # Component Search and Discovery
    # ============================================================================

    def search_tools_by_tag(self, tag: str) -> list[ToolHandler]:
        """Search tools by tag."""
        configs = mcp_registry.search_by_tag(tag)
        return [config.component for config in configs if config.component_type.value == "tool"]

    def search_resources_by_tag(self, tag: str) -> list[ResourceHandler]:
        """Search resources by tag."""
        configs = mcp_registry.search_by_tag(tag)
        return [config.component for config in configs if config.component_type.value == "resource"]

    def search_prompts_by_tag(self, tag: str) -> list[PromptHandler]:
        """Search prompts by tag."""
        configs = mcp_registry.search_by_tag(tag)
        return [config.component for config in configs if config.component_type.value == "prompt"]

    def search_components_by_tags(self, tags: list[str], match_all: bool = False):
        """Search components by multiple tags."""
        return mcp_registry.search_by_tags(tags, match_all=match_all)

    # ============================================================================
    # Information and Introspection
    # ============================================================================

    def get_tools(self) -> list[ToolHandler]:
        """Get all registered MCP tool handlers."""
        return list(self.protocol.tools.values())

    def get_resources(self) -> list[ResourceHandler]:
        """Get all registered MCP resource handlers."""
        return list(self.protocol.resources.values())

    def get_prompts(self) -> list[PromptHandler]:
        """Get all registered MCP prompt handlers."""
        return list(self.protocol.prompts.values())

    def get_endpoints(self) -> list[dict[str, Any]]:
        """Get all registered custom HTTP endpoints."""
        return [
            {
                "path": config.path,
                "name": config.name,
                "methods": config.methods,
                "description": config.description,
                "registered_at": config.registered_at,
            }
            for config in http_endpoint_registry.list_endpoints()
        ]

    def get_component_info(self, name: str) -> dict[str, Any] | None:
        """Get detailed information about an MCP component."""
        return mcp_registry.get_component_info(name)

    def info(self) -> dict[str, Any]:
        """Get comprehensive server information using modular smart config."""
        # Get comprehensive smart config summary
        smart_summary = self.smart_config.get_summary()

        return {
            "server": self.server_info.model_dump(exclude_none=True),
            "capabilities": self.capabilities.model_dump(exclude_none=True),
            "smart_config": smart_summary["full_config"],
            "smart_detection_summary": smart_summary["detection_summary"],
            "mcp_components": {
                "tools": {"count": len(self.protocol.tools), "names": list(self.protocol.tools.keys())},
                "resources": {"count": len(self.protocol.resources), "uris": list(self.protocol.resources.keys())},
                "prompts": {"count": len(self.protocol.prompts), "names": list(self.protocol.prompts.keys())},
                "stats": mcp_registry.get_stats(),
            },
            "http_endpoints": {
                "count": len(http_endpoint_registry.list_endpoints()),
                "custom": self.get_endpoints(),
                "stats": http_endpoint_registry.get_stats(),
            },
        }

    # ============================================================================
    # Registry Management
    # ============================================================================

    def clear_tools(self):
        """Clear all registered tools."""
        self.protocol.tools.clear()
        mcp_registry.clear_type(mcp_registry.MCPComponentType.TOOL)
        logger.info("Cleared all tools")

    def clear_resources(self):
        """Clear all registered resources."""
        self.protocol.resources.clear()
        mcp_registry.clear_type(mcp_registry.MCPComponentType.RESOURCE)
        logger.info("Cleared all resources")

    def clear_prompts(self):
        """Clear all registered prompts."""
        self.protocol.prompts.clear()
        mcp_registry.clear_type(mcp_registry.MCPComponentType.PROMPT)
        logger.info("Cleared all prompts")

    def clear_endpoints(self):
        """Clear all custom HTTP endpoints."""
        http_endpoint_registry.clear_endpoints()
        logger.info("Cleared all custom endpoints")

    def clear_all(self):
        """Clear all registered components and endpoints."""
        self.clear_tools()
        self.clear_resources()
        self.clear_prompts()
        self.clear_endpoints()
        logger.info("Cleared all components and endpoints")

    # ============================================================================
    # Smart Server Management with Modular Configuration
    # ============================================================================

    async def _start_proxy_if_enabled(self):
        """Start proxy manager if configured."""
        if self.proxy_manager:
            await self.proxy_manager.start_servers()
            logger.info(f"Proxy manager started: {self.proxy_manager.get_stats()}")

    async def _stop_proxy_if_enabled(self):
        """Stop proxy manager if running."""
        if self.proxy_manager:
            await self.proxy_manager.stop_servers()
            logger.info("Proxy manager stopped")

    def run(
        self,
        host: str | None = None,
        port: int | None = None,
        debug: bool | None = None,
        stdio: bool | None = None,
        log_level: str = "warning",
        post_register_hook=None,
        reload: bool = False,
        inspect: bool = False,
    ):
        """
        Run the MCP server with modular smart defaults.

        Args:
            host: Host to bind to (uses smart default if None)
            port: Port to bind to (uses smart default if None)
            debug: Enable debug logging (uses smart default if None)
            stdio: Run in stdio mode instead of HTTP server (auto-detects if None)
            log_level: Logging level for application (debug, info, warning, error, critical)
            post_register_hook: Optional callback to register additional endpoints after default endpoints
            reload: Enable hot reload (auto-restart on file changes, HTTP mode only)
            inspect: Open MCP Inspector in browser after starting (HTTP mode only)
        """
        # Set logging level FIRST, before any other operations
        import os

        # If debug is explicitly set to True, override log_level
        if debug is True:
            log_level = "debug"

        # Handle log_level - ensure it's a string
        default_level = log_level if isinstance(log_level, str) else "warning"
        app_log_level = os.getenv("MCP_LOG_LEVEL", default_level).upper()

        # Configure all loggers early
        logging.basicConfig(level=getattr(logging, app_log_level, logging.WARNING))
        logging.getLogger("chuk_mcp_server").setLevel(getattr(logging, app_log_level, logging.WARNING))

        # Check if STDIO transport was requested
        if self.smart_transport == "stdio":
            return self.run_stdio(debug, log_level=log_level)

        # Use smart defaults if not overridden
        final_host = host or self.smart_host
        final_port = port or self.smart_port
        final_debug = debug if debug is not None else self.smart_debug

        # Auto-detect stdio mode if not explicitly set
        if stdio is None:
            # Only use stdio if explicitly requested via environment
            stdio = (
                self.smart_config.environment_detector.get_env_var("MCP_STDIO")
                or self.smart_config.environment_detector.get_env_var("USE_STDIO")
                or self.smart_config.environment_detector.get_env_var("MCP_TRANSPORT") == "stdio"
            )
            stdio = bool(stdio)

        # Check if stdio mode is requested
        if stdio:
            # In stdio mode, suppress all logging to keep stdout clean
            # Set logging to CRITICAL to suppress most messages
            import sys

            logging.basicConfig(
                level=logging.CRITICAL,
                stream=sys.stderr,
                format="%(message)s",  # Minimal format
                force=True,  # Override earlier basicConfig
            )
            # Suppress specific noisy loggers
            logging.getLogger("chuk_mcp_server").setLevel(logging.CRITICAL)
            logging.getLogger("chuk_mcp_server.protocol").setLevel(logging.CRITICAL)
            logging.getLogger("chuk_mcp_server.core").setLevel(logging.CRITICAL)
            logging.getLogger("chuk_mcp_server.stdio_transport").setLevel(logging.CRITICAL)

            # Run in stdio mode
            from .stdio_transport import run_stdio_server

            run_stdio_server(self.protocol)
        else:
            # HTTP mode - logging already set at start of run()

            # Show the banner (always, regardless of log level)
            if self.smart_debug:
                self._print_smart_config(actual_log_level=app_log_level)

            # Start proxy manager if configured (BEFORE showing startup info)
            import asyncio

            if self.proxy_manager:
                try:
                    logger.info("Starting proxy manager...")
                    asyncio.run(self._start_proxy_if_enabled())
                    stats = self.proxy_manager.get_stats()
                    logger.info(f"Proxy manager started successfully: {stats}")

                    # Log registered tools
                    tool_count = len(self.protocol.tools)
                    logger.info(f"Total tools after proxy: {tool_count}")
                    logger.debug(f"Tools: {list(self.protocol.tools.keys())}")
                except Exception as e:
                    logger.error(f"Failed to start proxy manager: {e}", exc_info=True)

            # Load tool modules if configured
            if self.module_loader:
                try:
                    logger.info("Loading tool modules...")
                    loaded_modules = self.module_loader.load_modules()
                    module_info = self.module_loader.get_module_info()
                    logger.info(
                        f"Loaded {module_info['total_modules']} modules with {module_info['total_tools']} tools"
                    )
                    logger.debug(f"Modules: {list(loaded_modules.keys())}")
                except Exception as e:
                    logger.error(f"Failed to load tool modules: {e}", exc_info=True)

            # Show startup information AFTER proxy initialization
            if getattr(self, "_should_print_config", True):
                self._print_startup_info(final_host, final_port, final_debug, actual_log_level=app_log_level)

            # Create HTTP server
            if self._server is None:
                self._server = create_server(self.protocol, post_register_hook=post_register_hook)

            # Open MCP Inspector if requested
            if inspect:
                import threading
                import webbrowser

                inspect_url = f"http://{final_host}:{final_port}/docs"

                def _open_browser():
                    import time

                    time.sleep(1.5)  # Wait for server to start
                    webbrowser.open(inspect_url)

                threading.Thread(target=_open_browser, daemon=True).start()
                logger.info(f"Opening MCP Inspector: {inspect_url}")

            # Run the server
            try:
                self._server.run(
                    host=final_host,
                    port=final_port,
                    debug=final_debug,
                    log_level=log_level,
                    reload=reload,
                )
            except KeyboardInterrupt:
                logger.info("\n👋 Server shutting down gracefully...")
                # Gracefully shut down protocol handler (drain in-flight requests)
                asyncio.run(self.protocol.shutdown())
                # Stop proxy servers on shutdown
                if self.proxy_manager:
                    asyncio.run(self._stop_proxy_if_enabled())
            except Exception as e:
                logger.error(f"❌ Server error: {e}")
                # Stop proxy servers on error
                if self.proxy_manager:
                    asyncio.run(self._stop_proxy_if_enabled())
                raise

    def run_stdio(self, debug: bool | None = None, log_level: str = "warning"):
        """
        Run the MCP server over STDIO transport.

        Args:
            debug: Enable debug logging (uses smart default if None)
            log_level: Logging level for application (debug, info, warning, error, critical)
        """
        final_debug = debug if debug is not None else self.smart_debug

        # Configure logging to stderr for stdio transport
        if final_debug:
            logging.basicConfig(level=logging.DEBUG, stream=sys.stderr)
        else:
            # Use provided log_level or smart log level from modular config, always to stderr
            import os

            default_level = log_level if isinstance(log_level, str) else "warning"
            app_log_level = os.getenv("MCP_LOG_LEVEL", default_level).upper()
            log_level_int = getattr(logging, app_log_level, logging.WARNING)
            logging.basicConfig(level=log_level_int, stream=sys.stderr)

        # Suppress smart config output for stdio transport
        self._should_print_config = False

        # Create and run STDIO transport
        transport = StdioSyncTransport(self.protocol)

        logger.info(f"🚀 Starting {self.server_info.name} over STDIO transport")

        try:
            transport.run()
        except KeyboardInterrupt:
            logger.info("👋 STDIO transport shutting down gracefully...")
        except Exception as e:
            logger.error(f"❌ STDIO transport error: {e}")
            raise

    def _print_startup_info(self, host: str, port: int, debug: bool, actual_log_level: str | None = None):
        """Print comprehensive startup information using modular config."""
        import sys

        # Use stderr if in stdio mode to keep stdout clean
        output = sys.stderr if self.smart_transport_mode == "stdio" else sys.stdout
        print("🚀 ChukMCPServer - Modular Smart Configuration", file=output)
        print("=" * 60, file=output)

        # Server information
        info = self.info()
        print(f"Server: {info['server']['name']}", file=output)
        print(f"Version: {info['server']['version']}", file=output)
        print("Framework: ChukMCPServer with Modular Zero Configuration", file=output)
        print(file=output)

        # Smart configuration summary from modular system (with actual log level override)
        detection_summary = info["smart_detection_summary"].copy()
        if actual_log_level:
            detection_summary["logging"] = f"{actual_log_level} level, debug={debug}"
        print("🧠 Smart Detection Summary:", file=output)
        for key, value in detection_summary.items():
            print(f"   {key.replace('_', ' ').title()}: {value}", file=output)
        print(file=output)

        # MCP Components
        mcp_info = info["mcp_components"]
        print(f"🔧 MCP Tools: {mcp_info['tools']['count']}", file=output)
        for tool_name in mcp_info["tools"]["names"]:
            print(f"   - {tool_name}", file=output)
        print(file=output)

        print(f"📂 MCP Resources: {mcp_info['resources']['count']}", file=output)
        for resource_uri in mcp_info["resources"]["uris"]:
            print(f"   - {resource_uri}", file=output)
        print(file=output)

        print(f"💬 MCP Prompts: {mcp_info['prompts']['count']}", file=output)
        for prompt_name in mcp_info["prompts"]["names"]:
            print(f"   - {prompt_name}", file=output)
        print(file=output)

        # Connection information
        print("🌐 Server Information:", file=output)
        print(f"   URL: http://{host}:{port}", file=output)
        print(f"   MCP Endpoint: http://{host}:{port}/mcp", file=output)
        print(f"   Debug: {debug}", file=output)
        print(file=output)

        # Performance mode information
        print("⚡ Performance Configuration:", file=output)
        print(f"   Mode: {self.smart_performance_mode}", file=output)
        print(f"   Workers: {self.smart_workers}", file=output)
        print(f"   Max Connections: {self.smart_max_connections}", file=output)
        print(file=output)

        # Inspector compatibility
        print("🔍 MCP Inspector:", file=output)
        print(f"   URL: http://{host}:{port}/mcp", file=output)
        print("   Transport: Streamable HTTP", file=output)
        print("=" * 60, file=output)

    # ============================================================================
    # Configuration Management
    # ============================================================================

    def get_smart_config(self) -> dict[str, Any]:
        """Get the current smart configuration."""
        return self.smart_config.get_all_defaults()

    def get_smart_config_summary(self) -> dict[str, Any]:
        """Get a summary of smart configuration detection."""
        return self.smart_config.get_summary()

    def refresh_smart_config(self):
        """Refresh the smart configuration (clear cache and re-detect)."""
        self.smart_config.clear_cache()
        smart_defaults = self.smart_config.get_all_defaults()

        # Update stored values
        self.smart_environment = smart_defaults["environment"]
        self.smart_workers = smart_defaults["workers"]
        self.smart_max_connections = smart_defaults["max_connections"]
        self.smart_log_level = smart_defaults["log_level"]
        self.smart_performance_mode = smart_defaults["performance_mode"]
        self.smart_containerized = smart_defaults["containerized"]

        logger.info("🔄 Smart configuration refreshed")

    # ============================================================================
    # Proxy Management
    # ============================================================================

    def enable_proxy(self, config: dict[str, Any]) -> None:
        """
        Enable proxy mode with the given configuration.

        Args:
            config: Proxy configuration dictionary
        """
        self.proxy_manager = ProxyManager(config, self.protocol)
        logger.info("Proxy mode enabled")

    def get_proxy_stats(self) -> dict[str, Any] | None:
        """Get proxy manager statistics."""
        if self.proxy_manager:
            return self.proxy_manager.get_stats()
        return None

    async def call_proxied_tool(self, name: str, **kwargs: Any) -> Any:
        """
        Call a proxied tool.

        Args:
            name: Full tool name (e.g., "proxy.time.get_current_time")
            **kwargs: Tool arguments

        Returns:
            Tool result
        """
        if not self.proxy_manager:
            raise RuntimeError("Proxy mode not enabled")
        return await self.proxy_manager.call_tool(name, **kwargs)

    # ============================================================================
    # Context Manager Support
    # ============================================================================

    def __enter__(self):
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager."""
        pass


# ============================================================================
# Factory Functions
# ============================================================================


def create_mcp_server(name: str | None = None, **kwargs) -> ChukMCPServer:
    """Factory function to create a ChukMCP Server with modular zero config."""
    return ChukMCPServer(name=name, **kwargs)


def quick_server(name: str | None = None) -> ChukMCPServer:
    """Create a server with minimal configuration for quick prototyping."""
    return ChukMCPServer(name=name or "Quick Server", version="0.1.0")
