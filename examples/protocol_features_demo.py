#!/usr/bin/env python3
"""
Demo: MCP Protocol Features (v0.17)

Demonstrates the five protocol features added in v0.17:
1. Completions - Argument auto-completion for resources and prompts
2. Resource Subscriptions - Subscribe to resource updates
3. Elicitation - Server-to-client structured user input (context API)
4. Progress Notifications - Server-to-client progress updates (context API)
5. Roots - Client filesystem root discovery (context API)

This script exercises each feature directly through the protocol handler,
showing the JSON-RPC messages exchanged. No external MCP client is needed.
"""

import asyncio
import json
import sys

from chuk_mcp_server.context import (
    clear_all,
    create_elicitation,
    get_elicitation_fn,
    get_roots_fn,
    list_roots,
    send_progress,
    set_elicitation_fn,
    set_progress_notify_fn,
    set_progress_token,
    set_roots_fn,
    set_session_id,
)
from chuk_mcp_server.protocol import MCPProtocolHandler
from chuk_mcp_server.types import ServerInfo, create_server_capabilities


def banner(title: str) -> None:
    """Print a section banner."""
    print(f"\n{'=' * 60}", file=sys.stderr)
    print(f"  {title}", file=sys.stderr)
    print(f"{'=' * 60}", file=sys.stderr)


def show_json(label: str, data: dict) -> None:
    """Print a labeled JSON message."""
    print(f"\n  {label}:", file=sys.stderr)
    print(f"  {json.dumps(data, indent=2, default=str)}", file=sys.stderr)


async def demo_completions(handler: MCPProtocolHandler) -> None:
    """Demo 1: Completion providers for resources and prompts."""
    banner("1. Completions")
    print("\n  Register completion providers, then handle client requests.", file=sys.stderr)

    # Register a resource completion provider
    async def resource_completer(ref, argument):
        value = argument.get("value", "")
        all_uris = [
            "config://app",
            "config://database",
            "config://cache",
            "config://auth",
        ]
        matches = [u for u in all_uris if u.startswith(value)]
        return {"values": matches, "hasMore": False}

    handler.register_completion_provider("ref/resource", resource_completer)
    print("\n  Registered resource completion provider.", file=sys.stderr)

    # Send a completion request
    msg = {
        "jsonrpc": "2.0",
        "id": 10,
        "method": "completion/complete",
        "params": {
            "ref": {"type": "ref/resource", "uri": "config://app"},
            "argument": {"name": "uri", "value": "config://"},
        },
    }
    show_json("Request", msg)

    response, _ = await handler.handle_request(msg, session_id="demo-session")
    completion = response["result"]["completion"]
    show_json("Response (completion)", completion)
    print(f"\n  Matched {len(completion['values'])} URIs.", file=sys.stderr)


async def demo_subscriptions(handler: MCPProtocolHandler, session_id: str) -> None:
    """Demo 2: Resource subscriptions and update notifications."""
    banner("2. Resource Subscriptions")
    print("\n  Subscribe to resources and receive update notifications.", file=sys.stderr)

    # Subscribe to a resource
    subscribe_msg = {
        "jsonrpc": "2.0",
        "id": 20,
        "method": "resources/subscribe",
        "params": {"uri": "config://settings"},
    }
    show_json("Subscribe request", subscribe_msg)

    response, _ = await handler.handle_request(subscribe_msg, session_id)
    print(f"\n  Subscribe result: {response['result']}", file=sys.stderr)

    # Check subscription tracking
    subs = handler._resource_subscriptions.get(session_id, set())
    print(f"  Active subscriptions for session: {subs}", file=sys.stderr)

    # Show what a notification looks like
    notification = {
        "jsonrpc": "2.0",
        "method": "notifications/resources/updated",
        "params": {"uri": "config://settings"},
    }
    show_json("Update notification (sent to subscribers)", notification)
    print("\n  Note: notify_resource_updated() sends this to all subscribed sessions.", file=sys.stderr)

    # Unsubscribe
    unsub_msg = {
        "jsonrpc": "2.0",
        "id": 21,
        "method": "resources/unsubscribe",
        "params": {"uri": "config://settings"},
    }
    response, _ = await handler.handle_request(unsub_msg, session_id)
    subs = handler._resource_subscriptions.get(session_id, set())
    print(f"\n  After unsubscribe, active subscriptions: {subs}", file=sys.stderr)


async def demo_elicitation() -> None:
    """Demo 3: Elicitation - server requests structured user input."""
    banner("3. Elicitation (Context API)")
    print("\n  Server tools can request structured input from the client.", file=sys.stderr)

    # Simulate what happens inside a tool execution
    captured_request = {}

    async def mock_elicitation_fn(**kwargs):
        captured_request.update(kwargs)
        # In a real scenario, the client would show a dialog and return user input
        return {"action": "confirm", "content": {"method": "regression", "confidence": 0.95}}

    set_elicitation_fn(mock_elicitation_fn)
    print(f"\n  Elicitation available: {get_elicitation_fn() is not None}", file=sys.stderr)

    # Call create_elicitation (as a tool would)
    result = await create_elicitation(
        message="How would you like to analyze this dataset?",
        schema={
            "type": "object",
            "properties": {
                "method": {"type": "string", "enum": ["regression", "classification"]},
                "confidence": {"type": "number"},
            },
            "required": ["method"],
        },
        title="Analysis Configuration",
    )

    show_json("Elicitation request sent to client", captured_request)
    show_json("Client response", result)
    print(
        f"\n  User chose: {result['content']['method']} with confidence {result['content']['confidence']}",
        file=sys.stderr,
    )


async def demo_progress() -> None:
    """Demo 4: Progress notifications during long-running operations."""
    banner("4. Progress Notifications (Context API)")
    print("\n  Tools can report progress back to the client.", file=sys.stderr)

    # Track sent notifications
    notifications = []

    async def mock_progress_fn(progress_token, progress, total=None, message=None):
        notifications.append(
            {
                "progressToken": progress_token,
                "progress": progress,
                "total": total,
                "message": message,
            }
        )

    set_progress_token("task-42")
    set_progress_notify_fn(mock_progress_fn)

    # Simulate a tool sending progress updates
    total_steps = 5
    for step in range(total_steps):
        await send_progress(
            progress=step + 1,
            total=total_steps,
            message=f"Processing step {step + 1}/{total_steps}",
        )

    print(f"\n  Sent {len(notifications)} progress notifications:", file=sys.stderr)
    for n in notifications:
        pct = int(n["progress"] / n["total"] * 100) if n["total"] else "?"
        print(f"    [{pct:3d}%] {n['message']}", file=sys.stderr)

    # Show what the JSON-RPC notification looks like
    sample_notification = {
        "jsonrpc": "2.0",
        "method": "notifications/progress",
        "params": notifications[-1],
    }
    show_json("Example notification (no id = fire-and-forget)", sample_notification)

    # Demonstrate no-op behavior
    set_progress_notify_fn(None)
    set_progress_token(None)
    await send_progress(progress=1, total=1)  # Silent no-op
    print("\n  send_progress() is a silent no-op when unavailable.", file=sys.stderr)


async def demo_roots() -> None:
    """Demo 5: Roots - discover client filesystem roots."""
    banner("5. Roots (Context API)")
    print("\n  Server tools can discover the client's filesystem roots.", file=sys.stderr)

    # Simulate the roots function that the protocol handler would inject
    async def mock_roots_fn():
        return [
            {"uri": "file:///home/user/project", "name": "Project Root"},
            {"uri": "file:///home/user/data", "name": "Data Directory"},
        ]

    set_roots_fn(mock_roots_fn)
    print(f"\n  Roots available: {get_roots_fn() is not None}", file=sys.stderr)

    # Call list_roots (as a tool would)
    roots = await list_roots()

    print(f"\n  Client has {len(roots)} roots:", file=sys.stderr)
    for root in roots:
        name = root.get("name", "(unnamed)")
        print(f"    {name}: {root['uri']}", file=sys.stderr)

    # Show what the JSON-RPC request/response looks like
    request = {
        "jsonrpc": "2.0",
        "id": "roots-abc123",
        "method": "roots/list",
    }
    response = {
        "jsonrpc": "2.0",
        "id": "roots-abc123",
        "result": {"roots": roots},
    }
    show_json("Request (server -> client)", request)
    show_json("Response (client -> server)", response)


async def main():
    """Run all protocol feature demos."""
    print("\nMCP Protocol Features Demo (v0.17)", file=sys.stderr)
    print("Demonstrates all 5 new protocol features", file=sys.stderr)

    # Create handler with completions enabled
    server_info = ServerInfo(name="protocol-demo", version="0.17.0")
    capabilities = create_server_capabilities(completions=True)
    handler = MCPProtocolHandler(server_info, capabilities)

    # Initialize a session
    init_msg = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-06-18",
            "capabilities": {
                "sampling": {},
                "elicitation": {},
                "roots": {"listChanged": True},
            },
            "clientInfo": {"name": "demo-client", "version": "1.0.0"},
        },
    }
    response, session_id = await handler.handle_request(init_msg)
    set_session_id(session_id)
    print(f"\n  Session initialized: {session_id[:12]}...", file=sys.stderr)

    try:
        # Run all demos
        await demo_completions(handler)
        await demo_subscriptions(handler, session_id)
        await demo_elicitation()
        await demo_progress()
        await demo_roots()

        # Summary
        banner("Summary")
        print(
            """
  Feature             | Direction       | Transport  | Method
  --------------------|-----------------|------------|-----------------------------
  Completions         | client->server  | All        | completion/complete
  Subscriptions       | client->server  | All        | resources/subscribe
  Elicitation         | server->client  | STDIO      | elicitation/create
  Progress            | server->client  | STDIO      | notifications/progress
  Roots               | server->client  | STDIO      | roots/list
""",
            file=sys.stderr,
        )
        print("  All 5 features demonstrated successfully!", file=sys.stderr)
        print("", file=sys.stderr)

    finally:
        clear_all()


if __name__ == "__main__":
    asyncio.run(main())
