#!/usr/bin/env python3
# src/chuk_mcp_server/__init__.py (Enhanced with Modular Cloud Support)
"""
ChukMCPServer - Zero Configuration MCP Framework with Modular Cloud Support

The world's smartest MCP framework with zero configuration and automatic cloud detection.

ULTIMATE ZERO CONFIG (Works everywhere):
    from chuk_mcp_server import tool, resource, run

    @tool
    def hello(name: str) -> str:
        return f"Hello, {name}!"

    @resource("config://settings")
    def get_settings() -> dict:
        return {"app": "my_app", "magic": True}

    if __name__ == "__main__":
        run()  # Auto-detects EVERYTHING!

CLOUD EXAMPLES:

Google Cloud Functions:
    # main.py
    from chuk_mcp_server import ChukMCPServer, tool

    mcp = ChukMCPServer()  # Auto-detects GCF!

    @mcp.tool
    def hello_gcf(name: str) -> str:
        return f"Hello from GCF, {name}!"

    # Handler auto-created as 'mcp_gcf_handler'
    # Deploy: gcloud functions deploy my-server --entry-point mcp_gcf_handler

AWS Lambda:
    # lambda_function.py
    from chuk_mcp_server import ChukMCPServer, tool

    mcp = ChukMCPServer()  # Auto-detects Lambda!

    @mcp.tool
    def hello_lambda(name: str) -> str:
        return f"Hello from Lambda, {name}!"

    # Handler auto-created as 'lambda_handler'

Azure Functions:
    # function_app.py
    from chuk_mcp_server import ChukMCPServer, tool

    mcp = ChukMCPServer()  # Auto-detects Azure!

    @mcp.tool
    def hello_azure(name: str) -> str:
        return f"Hello from Azure, {name}!"

    # Handler auto-created as 'main'

All platforms work with ZERO configuration! 🚀
"""

import logging
import sys
from typing import Any

# Import cloud functionality
from .cloud import detect_cloud_provider, is_cloud_environment

# Import context management
from .context import (
    RequestContext,
    add_resource_link,
    create_elicitation,
    create_message,
    get_elicitation_fn,
    get_progress_notify_fn,
    get_roots_fn,
    get_sampling_fn,
    get_session_id,
    get_user_id,
    list_roots,
    require_user_id,
    send_log,
    send_progress,
    set_elicitation_fn,
    set_progress_notify_fn,
    set_roots_fn,
    set_sampling_fn,
    set_session_id,
    set_user_id,
)

# Import artifact/workspace context (optional - requires chuk-artifacts)
try:
    from chuk_artifacts import (
        NamespaceInfo,
        NamespaceType,
        StorageScope,
    )

    from .artifacts_context import (
        clear_artifact_store,
        create_blob_namespace,
        create_workspace_namespace,
        get_artifact_store,
        get_namespace_vfs,
        has_artifact_store,
        read_blob,
        read_workspace_file,
        set_artifact_store,
        set_global_artifact_store,
        write_blob,
        write_workspace_file,
    )

    _ARTIFACTS_AVAILABLE = True
    _ARTIFACTS_TYPES_AVAILABLE = True
except ImportError:
    _ARTIFACTS_AVAILABLE = False
    _ARTIFACTS_TYPES_AVAILABLE = False

    # Create stub functions that provide helpful error messages
    from typing import NoReturn

    def _artifact_not_available(*args: Any, **kwargs: Any) -> NoReturn:
        raise RuntimeError(
            "Artifact/workspace functionality requires chuk-artifacts. "
            "Install with: pip install 'chuk-mcp-server[artifacts]'"
        )

    get_artifact_store = _artifact_not_available
    set_artifact_store = _artifact_not_available
    set_global_artifact_store = _artifact_not_available
    clear_artifact_store = _artifact_not_available

    def has_artifact_store() -> bool:
        return False

    create_blob_namespace = _artifact_not_available
    create_workspace_namespace = _artifact_not_available
    write_blob = _artifact_not_available
    read_blob = _artifact_not_available
    write_workspace_file = _artifact_not_available
    read_workspace_file = _artifact_not_available
    get_namespace_vfs = _artifact_not_available

    # Type stubs
    NamespaceType = None
    StorageScope = None
    NamespaceInfo = None
from .core import ChukMCPServer, create_mcp_server, quick_server

# Import traditional decorators for global usage
from .decorators import prompt, requires_auth, resource, resource_template, tool
from .modules import ModuleLoader

# Import proxy functionality
from .proxy import ProxyManager
from .testing import ToolRunner
from .types import (
    MCPPrompt,
    ServerInfo,
    ToolParameter,
    create_server_capabilities,
)
from .types import (
    PromptHandler as Prompt,
)
from .types import (
    ResourceHandler as Resource,
)
from .types import (
    ToolHandler as Tool,
)
from .types.errors import URLElicitationRequiredError


# Create backward compatibility
def Capabilities(**kwargs: Any) -> dict[str, Any]:
    """Legacy capabilities function for backward compatibility."""
    return create_server_capabilities(**kwargs)  # type: ignore[no-any-return]


__version__ = "0.18"

# ============================================================================
# Global Magic with Cloud Support
# ============================================================================

_global_server: ChukMCPServer | None = None
_server_lock = __import__("threading").Lock()


def get_or_create_global_server() -> ChukMCPServer:
    """Get or create the global server instance with cloud detection."""
    global _global_server
    if _global_server is None:
        with _server_lock:
            if _global_server is None:
                _global_server = ChukMCPServer()  # Auto-detects cloud environment
    return _global_server


def get_mcp_server() -> ChukMCPServer:
    """Get the global MCP server instance (alias for get_or_create_global_server).

    Useful for accessing the server instance in OAuth setup and other contexts.

    Returns:
        The global ChukMCPServer instance

    Example:
        from chuk_mcp_server import get_mcp_server
        from chuk_mcp_server.oauth.helpers import setup_google_drive_oauth

        oauth_hook = setup_google_drive_oauth(get_mcp_server())
    """
    return get_or_create_global_server()


def run(transport: str = "http", **kwargs: Any) -> None:
    """
    Run the global smart server with cloud detection and transport selection.

    Args:
        transport: Transport type ("http" or "stdio")
        **kwargs: Additional arguments passed to the transport
    """
    server = get_or_create_global_server()

    if transport.lower() == "stdio":
        server.run_stdio(**kwargs)
    else:
        server.run(**kwargs)


# ============================================================================
# Cloud Magic Functions
# ============================================================================


def get_cloud_handler() -> object:
    """Magic function to get cloud-specific handler."""
    server = get_or_create_global_server()
    handler = server.get_cloud_handler()  # type: ignore

    if handler is None:
        cloud_provider = detect_cloud_provider()
        if cloud_provider:
            raise RuntimeError(
                f"Detected {cloud_provider.display_name} but no handler available. "
                f"Install with: pip install 'chuk-mcp-server[{cloud_provider.name}]'"
            )
        else:
            raise RuntimeError("Not in a cloud environment or no cloud provider detected.")

    return handler


def get_gcf_handler() -> object:
    """Get Google Cloud Functions handler."""
    server = get_or_create_global_server()
    adapter = server.get_cloud_adapter()  # type: ignore

    if adapter and hasattr(adapter, "get_handler"):
        from .cloud.providers.gcp import GCPProvider

        cloud_provider = detect_cloud_provider()
        if cloud_provider and isinstance(cloud_provider, GCPProvider):
            return adapter.get_handler()

    raise RuntimeError(
        "Not in Google Cloud Functions environment or functions-framework not installed. "
        "Install with: pip install 'chuk-mcp-server[gcf]'"
    )


def get_lambda_handler() -> object:
    """Get AWS Lambda handler."""
    server = get_or_create_global_server()
    adapter = server.get_cloud_adapter()  # type: ignore

    if adapter and hasattr(adapter, "get_handler"):
        from .cloud.providers.aws import AWSProvider

        cloud_provider = detect_cloud_provider()
        if cloud_provider and isinstance(cloud_provider, AWSProvider):
            return adapter.get_handler()

    raise RuntimeError("Not in AWS Lambda environment.")


def get_azure_handler() -> object:
    """Get Azure Functions handler."""
    server = get_or_create_global_server()
    adapter = server.get_cloud_adapter()  # type: ignore

    if adapter and hasattr(adapter, "get_handler"):
        from .cloud.providers.azure import AzureProvider

        cloud_provider = detect_cloud_provider()
        if cloud_provider and isinstance(cloud_provider, AzureProvider):
            return adapter.get_handler()

    raise RuntimeError("Not in Azure Functions environment.")


def is_cloud() -> bool:
    """Check if running in any cloud environment."""
    return is_cloud_environment()


def is_gcf() -> bool:
    """Check if running in Google Cloud Functions."""
    cloud_provider = detect_cloud_provider()
    return bool(cloud_provider and cloud_provider.name == "gcp")


def is_lambda() -> bool:
    """Check if running in AWS Lambda."""
    cloud_provider = detect_cloud_provider()
    return bool(cloud_provider and cloud_provider.name == "aws")


def is_azure() -> bool:
    """Check if running in Azure Functions."""
    cloud_provider = detect_cloud_provider()
    return bool(cloud_provider and cloud_provider.name == "azure")


def get_deployment_info() -> dict[str, Any]:
    """Get deployment information for current environment."""
    server = get_or_create_global_server()
    return server.get_cloud_deployment_info()  # type: ignore


# ============================================================================
# Auto-Cloud Handler Export
# ============================================================================


def _auto_export_cloud_handlers() -> None:
    """Automatically export cloud handlers based on environment detection."""
    import sys

    current_module = sys.modules[__name__]

    try:
        cloud_provider = detect_cloud_provider()
        if not cloud_provider:
            return

        # Get the global server and its cloud adapter
        server = get_or_create_global_server()
        adapter = server.get_cloud_adapter()  # type: ignore

        if not adapter:
            return

        handler = adapter.get_handler()
        if not handler:
            return

        # Export handler with standard names for each platform
        if cloud_provider.name == "gcp":
            # GCF expects 'mcp_gcf_handler' or custom entry point
            current_module.mcp_gcf_handler = handler  # type: ignore

        elif cloud_provider.name == "aws":
            # Lambda expects 'lambda_handler' by default
            current_module.lambda_handler = handler  # type: ignore
            current_module.handler = handler  # type: ignore

        elif cloud_provider.name == "azure":
            # Azure Functions expects 'main' by default
            current_module.main = handler  # type: ignore
            current_module.azure_handler = handler  # type: ignore

        elif cloud_provider.name in ["vercel", "netlify", "cloudflare"]:
            # Edge functions often expect 'handler' or 'main'
            current_module.handler = handler  # type: ignore
            current_module.main = handler  # type: ignore

        # Always export generic names
        current_module.cloud_handler = handler  # type: ignore
        current_module.mcp_handler = handler  # type: ignore

    except Exception as e:
        logging.getLogger(__name__).debug(f"Cloud handler auto-export skipped: {e}")


# Auto-export handlers when module is imported
_auto_export_cloud_handlers()

# ============================================================================
# Enhanced Exports
# ============================================================================

__all__ = [
    # 🧠 PRIMARY INTERFACE (Zero Config)
    "ChukMCPServer",
    "get_mcp_server",
    # 🪄 MAGIC DECORATORS
    "tool",
    "resource",
    "resource_template",
    "prompt",
    "requires_auth",
    "run",
    # 🏭 FACTORY FUNCTIONS
    "create_mcp_server",
    "quick_server",
    # ☁️ CLOUD MAGIC
    "get_cloud_handler",  # Generic cloud handler
    "get_gcf_handler",  # Google Cloud Functions
    "get_lambda_handler",  # AWS Lambda
    "get_azure_handler",  # Azure Functions
    # 🔍 CLOUD DETECTION
    "is_cloud",  # Any cloud environment
    "is_gcf",  # Google Cloud Functions
    "is_lambda",  # AWS Lambda
    "is_azure",  # Azure Functions
    "get_deployment_info",  # Deployment information
    # 📚 TYPES & UTILITIES
    "Tool",
    "Resource",
    "Prompt",
    "MCPPrompt",
    "ToolParameter",
    "ServerInfo",
    "Capabilities",
    # 🔐 CONTEXT MANAGEMENT
    "RequestContext",  # Context manager
    "get_session_id",  # Get current session
    "get_user_id",  # Get current user
    "require_user_id",  # Require authenticated user
    "set_session_id",  # Set session context
    "set_user_id",  # Set user context
    # 🤖 SAMPLING (server → client LLM requests)
    "create_message",  # Request client LLM sampling
    "get_sampling_fn",  # Get sampling function
    "set_sampling_fn",  # Set sampling function
    # 💬 ELICITATION (server → client user input)
    "create_elicitation",  # Request structured user input
    "get_elicitation_fn",  # Get elicitation function
    "set_elicitation_fn",  # Set elicitation function
    # 📊 PROGRESS (server → client notifications)
    "send_progress",  # Send progress update
    "get_progress_notify_fn",  # Get progress notify function
    "set_progress_notify_fn",  # Set progress notify function
    # 📁 ROOTS (server → client filesystem roots)
    "list_roots",  # Request client filesystem roots
    "get_roots_fn",  # Get roots function
    "set_roots_fn",  # Set roots function
    # 📝 LOGGING (server → client log notifications)
    "send_log",  # Send log notification to client
    # 🔗 RESOURCE LINKS (tool → resource references)
    "add_resource_link",  # Add resource link from tool execution
    # 📦 ARTIFACT/WORKSPACE CONTEXT (Optional - requires chuk-artifacts)
    "get_artifact_store",  # Get artifact store from context
    "set_artifact_store",  # Set artifact store in context
    "set_global_artifact_store",  # Set global artifact store
    "has_artifact_store",  # Check if artifact store available
    "create_blob_namespace",  # Create blob namespace
    "create_workspace_namespace",  # Create workspace namespace
    "write_blob",  # Write to blob namespace
    "read_blob",  # Read from blob namespace
    "write_workspace_file",  # Write file to workspace
    "read_workspace_file",  # Read file from workspace
    "get_namespace_vfs",  # Get VFS for namespace
    # 📦 ARTIFACT/WORKSPACE TYPES (Optional - from chuk-artifacts)
    "NamespaceType",  # BLOB or WORKSPACE
    "StorageScope",  # SESSION, USER, or SANDBOX
    "NamespaceInfo",  # Namespace information model
    # 🌐 PROXY FUNCTIONALITY
    "ProxyManager",  # Multi-server proxy manager
    # 📦 MODULE LOADING
    "ModuleLoader",  # Dynamic tool module loader
    # 🧪 TESTING
    "ToolRunner",  # Test harness for invoking tools without transport
    # ⚠️ ERRORS
    "URLElicitationRequiredError",  # URL mode elicitation (MCP 2025-11-25)
]

# ============================================================================
# Enhanced Examples Documentation
# ============================================================================


def show_cloud_examples() -> None:
    """Show cloud-specific zero configuration examples."""
    examples = """
☁️ ChukMCPServer - Cloud Zero Configuration Examples

1️⃣ GOOGLE CLOUD FUNCTIONS:

   # main.py
   from chuk_mcp_server import ChukMCPServer, tool

   mcp = ChukMCPServer()  # 🧠 Auto-detects GCF!

   @mcp.tool
   def hello_gcf(name: str) -> str:
       return f"Hello from GCF, {name}!"

   # ✨ Handler auto-created as 'mcp_gcf_handler'
   # Deploy: gcloud functions deploy my-server --entry-point mcp_gcf_handler

2️⃣ AWS LAMBDA:

   # lambda_function.py
   from chuk_mcp_server import ChukMCPServer, tool

   mcp = ChukMCPServer()  # 🧠 Auto-detects Lambda!

   @mcp.tool
   def hello_lambda(name: str) -> str:
       return f"Hello from Lambda, {name}!"

   # ✨ Handler auto-created as 'lambda_handler'
   # Deploy: AWS CLI or SAM

3️⃣ AZURE FUNCTIONS:

   # function_app.py
   from chuk_mcp_server import ChukMCPServer, tool

   mcp = ChukMCPServer()  # 🧠 Auto-detects Azure!

   @mcp.tool
   def hello_azure(name: str) -> str:
       return f"Hello from Azure, {name}!"

   # ✨ Handler auto-created as 'main'
   # Deploy: Azure CLI or VS Code

4️⃣ VERCEL EDGE:

   # api/mcp.py
   from chuk_mcp_server import tool, get_cloud_handler

   @tool
   def hello_edge(name: str) -> str:
       return f"Hello from Vercel Edge, {name}!"

   # ✨ Handler auto-exported
   handler = get_cloud_handler()

5️⃣ MULTI-CLOUD (Works everywhere):

   # server.py
   from chuk_mcp_server import ChukMCPServer, tool, is_cloud

   mcp = ChukMCPServer()  # 🧠 Auto-detects ANY cloud!

   @mcp.tool
   def universal_tool(data: str) -> dict:
       cloud_info = "cloud" if is_cloud() else "local"
       return {"data": data, "environment": cloud_info}

   if __name__ == "__main__":
       if is_cloud():
           print("🌟 Cloud environment detected - handler auto-created!")
       else:
           mcp.run()  # Local development

🚀 ALL PLATFORMS SUPPORTED WITH ZERO CONFIG:
   ✅ Google Cloud Functions (Gen 1 & 2)
   ✅ AWS Lambda (x86 & ARM64)
   ✅ Azure Functions (Python)
   ✅ Vercel Edge Functions
   ✅ Netlify Edge Functions
   ✅ Cloudflare Workers
   ✅ Local Development
   ✅ Docker Containers
   ✅ Kubernetes
"""
    print(examples)


# Show enhanced examples in interactive environments

if hasattr(sys, "ps1"):  # Interactive Python
    print("🌟 ChukMCPServer v0.16 - Enhanced Cloud Support")
    print("Type show_cloud_examples() to see cloud deployment examples!")
