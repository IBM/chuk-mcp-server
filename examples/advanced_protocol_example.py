#!/usr/bin/env python3
"""
Demo: Advanced MCP Protocol Features (MCP 2025-11-25)

Demonstrates the most advanced protocol features from the latest MCP spec:
1. Sampling with tool calling -- create_message() with tools and tool_choice
2. URL mode elicitation -- URLElicitationRequiredError for external URLs
3. Elicitation with default values -- schemas with defaults in properties
4. Tasks system -- full lifecycle: create, get, list, result, cancel

This script exercises each feature directly through the protocol handler,
showing the JSON-RPC messages exchanged. No external MCP client is needed.

Run:
    python examples/advanced_protocol_example.py
"""

import asyncio
import json
import sys

from chuk_mcp_server import URLElicitationRequiredError
from chuk_mcp_server.context import (
    clear_all,
    create_elicitation,
    create_message,
    set_elicitation_fn,
    set_sampling_fn,
    set_session_id,
)
from chuk_mcp_server.protocol import MCPProtocolHandler
from chuk_mcp_server.types import ServerInfo, ToolHandler, create_server_capabilities


def banner(title: str) -> None:
    print(f"\n{'=' * 60}", file=sys.stderr)
    print(f"  {title}", file=sys.stderr)
    print(f"{'=' * 60}", file=sys.stderr)


def show_json(label: str, data: dict) -> None:
    print(f"\n  {label}:", file=sys.stderr)
    print(f"  {json.dumps(data, indent=2, default=str)}", file=sys.stderr)


# ============================================================================
# 1. Sampling with Tool Calling
# ============================================================================


async def demo_sampling_with_tools(handler: MCPProtocolHandler) -> None:
    """Sampling requests can now include tools for the LLM to call."""
    banner("1. Sampling with Tool Calling")
    print("\n  create_message() now supports tools and tool_choice params.", file=sys.stderr)
    print("  The client's LLM can call the provided tools.", file=sys.stderr)

    captured_request: dict = {}

    async def mock_sampling_fn(**kwargs):
        captured_request.update(kwargs)
        return {
            "role": "assistant",
            "content": {"type": "text", "text": "The weather in Tokyo is 22C and sunny."},
            "model": "claude-sonnet-4-5-20250929",
            "stopReason": "endTurn",
        }

    set_sampling_fn(mock_sampling_fn)

    # Call create_message with tools
    result = await create_message(
        messages=[{"role": "user", "content": {"type": "text", "text": "What's the weather in Tokyo?"}}],
        max_tokens=500,
        tools=[
            {
                "name": "get_weather",
                "description": "Get current weather for a city",
                "inputSchema": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            }
        ],
        tool_choice={"type": "auto"},
    )

    show_json(
        "Sampling request sent to client (tools included)",
        {
            "tools": captured_request.get("tools"),
            "tool_choice": captured_request.get("tool_choice"),
        },
    )
    show_json("Client LLM response", result)

    set_sampling_fn(None)


# ============================================================================
# 2. URL Mode Elicitation
# ============================================================================


async def demo_url_elicitation(handler: MCPProtocolHandler, session_id: str) -> None:
    """Tools can redirect users to external URLs via URLElicitationRequiredError."""
    banner("2. URL Mode Elicitation")
    print("\n  Tools raise URLElicitationRequiredError to send users to a URL.", file=sys.stderr)
    print("  Protocol returns JSON-RPC error -32042 with the URL.", file=sys.stderr)

    async def mock_send(request: dict) -> dict:
        return {"jsonrpc": "2.0", "id": request.get("id", 1), "result": {}}

    handler._send_to_client = mock_send

    # A tool that requires OAuth consent
    def connect_github(org: str) -> str:
        """Connect to a GitHub organization. Requires OAuth consent."""
        raise URLElicitationRequiredError(
            url=f"https://github.com/login/oauth/authorize?scope=repo&org={org}",
            description="Please authorize access to your GitHub organization",
            mime_type="text/html",
        )

    tool = ToolHandler.from_function(connect_github, name="connect_github", description="Connect to GitHub")
    handler.register_tool(tool)

    response, _ = await handler._handle_tools_call({"name": "connect_github", "arguments": {"org": "acme-corp"}}, 20)

    show_json("Error response (-32042)", response)

    error = response["error"]
    print(f"\n  Error code: {error['code']} (URL elicitation required)", file=sys.stderr)
    print(f"  URL: {error['data']['url']}", file=sys.stderr)
    print(f"  Description: {error['data']['description']}", file=sys.stderr)

    handler._send_to_client = None


# ============================================================================
# 3. Elicitation with Default Values
# ============================================================================


async def demo_elicitation_defaults() -> None:
    """Elicitation schemas can include default values on properties."""
    banner("3. Elicitation with Default Values")
    print("\n  Schema properties can have 'default' values for pre-filled forms.", file=sys.stderr)

    captured_request: dict = {}

    async def mock_elicitation_fn(**kwargs):
        captured_request.update(kwargs)
        # Simulate user accepting defaults for name and changing priority
        return {"action": "confirm", "content": {"name": "Anonymous", "priority": "high", "notify": True}}

    set_elicitation_fn(mock_elicitation_fn)

    result = await create_elicitation(
        message="Configure your notification preferences",
        schema={
            "type": "object",
            "properties": {
                "name": {"type": "string", "default": "Anonymous"},
                "priority": {
                    "type": "string",
                    "enum": ["low", "medium", "high"],
                    "default": "medium",
                },
                "notify": {"type": "boolean", "default": True},
            },
        },
        title="Notification Settings",
    )

    show_json("Schema with defaults sent to client", captured_request.get("schema", {}))
    show_json("Client response", result)

    defaults = captured_request["schema"]["properties"]
    print("\n  Default values pre-filled in form:", file=sys.stderr)
    for key, prop in defaults.items():
        if "default" in prop:
            print(f"    {key}: {prop['default']}", file=sys.stderr)

    set_elicitation_fn(None)


# ============================================================================
# 4. Tasks System
# ============================================================================


async def demo_tasks(handler: MCPProtocolHandler, session_id: str) -> None:
    """Tasks track the lifecycle of long-running tool executions."""
    banner("4. Tasks System")
    print("\n  Tasks provide durable state for long-running requests.", file=sys.stderr)
    print("  Lifecycle: working -> completed/failed/cancelled", file=sys.stderr)

    async def mock_send(request: dict) -> dict:
        return {"jsonrpc": "2.0", "id": request.get("id", 1), "result": {}}

    handler._send_to_client = mock_send

    # Create a task manually (normally done automatically during tools/call)
    task_id = handler._create_task(request_id="req-42", tool_name="long_analysis")
    print(f"\n  Created task: {task_id}", file=sys.stderr)

    # tasks/get -- query task status
    response, _ = await handler._handle_tasks_get({"id": task_id}, 100)
    task = response["result"]
    print(f"  Status: {task['status']} (working)", file=sys.stderr)

    # tasks/list -- list all tasks
    response, _ = await handler._handle_tasks_list({}, 101)
    tasks_list = response["result"]["tasks"]
    print(f"  Total tasks: {len(tasks_list)}", file=sys.stderr)

    # Update task to completed
    handler._update_task_status(
        task_id,
        "completed",
        result={"content": [{"type": "text", "text": "Analysis complete: 150 anomalies found"}]},
    )
    print("  Updated to: completed", file=sys.stderr)

    # tasks/result -- get the result (only works for completed/failed tasks)
    response, _ = await handler._handle_tasks_result({"id": task_id}, 102)
    result_task = response["result"]
    show_json("Completed task with result", result_task)

    # Demonstrate cancel on a new working task
    cancel_task_id = handler._create_task(request_id="req-99", tool_name="slow_export")
    response, _ = await handler._handle_tasks_cancel({"id": cancel_task_id}, 103)
    cancelled = response["result"]
    print(f"\n  Cancelled task {cancel_task_id}: status={cancelled['status']}", file=sys.stderr)

    # Trying to cancel again fails
    response, _ = await handler._handle_tasks_cancel({"id": cancel_task_id}, 104)
    print(f"  Cancel again: error={response['error']['message']}", file=sys.stderr)

    handler._send_to_client = None


# ============================================================================
# Main
# ============================================================================


async def main():
    print("\nAdvanced MCP Protocol Features Demo (MCP 2025-11-25)", file=sys.stderr)
    print("Demonstrates sampling+tools, URL elicitation, and tasks", file=sys.stderr)

    server_info = ServerInfo(name="advanced-demo", version="0.20.0")
    capabilities = create_server_capabilities()
    handler = MCPProtocolHandler(server_info, capabilities)

    # Initialize a session with all client capabilities
    init_msg = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-11-25",
            "capabilities": {"sampling": {}, "elicitation": {}, "roots": {"listChanged": True}},
            "clientInfo": {"name": "demo-client", "version": "2.0.0"},
        },
    }
    _, session_id = await handler.handle_request(init_msg)
    set_session_id(session_id)
    print(f"\n  Session: {session_id[:12]}...", file=sys.stderr)

    try:
        await demo_sampling_with_tools(handler)
        await demo_url_elicitation(handler, session_id)
        await demo_elicitation_defaults()
        await demo_tasks(handler, session_id)

        banner("Summary")
        print(
            """
  Feature                | Error Code / Method       | Key API
  -----------------------|---------------------------|----------------------------------
  Sampling + tools       | sampling/createMessage    | create_message(tools=..., tool_choice=...)
  URL elicitation        | -32042                    | raise URLElicitationRequiredError(url=...)
  Elicitation defaults   | elicitation/create        | schema properties with "default" key
  Tasks system           | tasks/get,list,result,cancel | handler._create_task(), tasks/* methods
""",
            file=sys.stderr,
        )
        print("  All 4 features demonstrated successfully!", file=sys.stderr)

    finally:
        clear_all()


if __name__ == "__main__":
    asyncio.run(main())
