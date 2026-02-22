#!/usr/bin/env python3
"""
Demo: Context API Features (MCP 2025-06-18)

Demonstrates server-to-client features available inside tool functions:
1. Log notifications -- send_log() sends notifications/message to the client
2. Progress notifications -- send_progress() for long-running operations
3. Resource links -- add_resource_link() attaches resource refs to tool output
4. Content annotations -- audience/priority on content types

This script exercises each feature directly through the protocol handler,
showing the JSON-RPC messages exchanged. No external MCP client is needed.

Run:
    python examples/context_features_example.py
"""

import asyncio
import json
import sys

from chuk_mcp_server.context import (
    add_resource_link,
    clear_all,
    send_log,
    send_progress,
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
# 1. Log Notifications
# ============================================================================


async def demo_log_notifications(handler: MCPProtocolHandler, session_id: str) -> None:
    """Tools can send log messages to the client via send_log()."""
    banner("1. Log Notifications (send_log)")
    print("\n  Tools call send_log() to push notifications/message to the client.", file=sys.stderr)

    # Track notifications sent to the client
    sent_notifications: list[dict] = []

    async def mock_send(request: dict) -> dict:
        sent_notifications.append(request)
        return {"jsonrpc": "2.0", "id": request.get("id", 1), "result": {}}

    handler._send_to_client = mock_send

    # Define a tool that uses send_log()
    async def analyze_data(input_data: str) -> str:
        """Analyze data and log progress to the client."""
        await send_log("info", f"Starting analysis of: {input_data}", logger_name="analyzer")
        await send_log("warning", "Dataset has missing values", logger_name="analyzer")
        await send_log("info", {"step": "complete", "records_processed": 42})
        return "Analysis complete: 42 records processed"

    tool = ToolHandler.from_function(analyze_data, name="analyze_data", description="Analyze data")
    handler.register_tool(tool)

    # Execute the tool (protocol handler auto-injects send_log)
    response, _ = await handler._handle_tools_call(
        {"name": "analyze_data", "arguments": {"input_data": "sales_q4.csv"}}, 10
    )

    # Show the notifications that were sent
    log_notifications = [n for n in sent_notifications if n.get("method") == "notifications/message"]
    print(f"\n  Tool sent {len(log_notifications)} log notifications:", file=sys.stderr)
    for notif in log_notifications:
        params = notif["params"]
        print(f"    [{params['level']:>7s}] {params['data']}", file=sys.stderr)

    show_json("Example notification shape", log_notifications[0] if log_notifications else {})
    handler._send_to_client = None


# ============================================================================
# 2. Progress Notifications
# ============================================================================


async def demo_progress(handler: MCPProtocolHandler, session_id: str) -> None:
    """Tools can report progress via send_progress()."""
    banner("2. Progress Notifications (send_progress)")
    print("\n  Tools call send_progress() to update clients on long-running work.", file=sys.stderr)

    sent_notifications: list[dict] = []

    async def mock_send(request: dict) -> dict:
        sent_notifications.append(request)
        return {"jsonrpc": "2.0", "id": request.get("id", 1), "result": {}}

    handler._send_to_client = mock_send

    async def process_batch(count: int) -> str:
        """Process items in a batch with progress reporting."""
        for i in range(count):
            await send_progress(
                progress=i + 1,
                total=count,
                message=f"Processing item {i + 1}/{count}",
            )
        return f"Processed {count} items"

    tool = ToolHandler.from_function(process_batch, name="process_batch", description="Batch processing")
    handler.register_tool(tool)

    # Execute with a progress token (client provides this)
    response, _ = await handler._handle_tools_call(
        {"name": "process_batch", "arguments": {"count": 5}, "_meta": {"progressToken": "batch-42"}}, 20
    )

    progress_notifications = [n for n in sent_notifications if n.get("method") == "notifications/progress"]
    print(f"\n  Tool sent {len(progress_notifications)} progress notifications:", file=sys.stderr)
    for notif in progress_notifications:
        p = notif["params"]
        pct = int(p["progress"] / p["total"] * 100) if p.get("total") else "?"
        print(f"    [{pct:3d}%] {p.get('message', '')}", file=sys.stderr)

    handler._send_to_client = None


# ============================================================================
# 3. Resource Links
# ============================================================================


async def demo_resource_links(handler: MCPProtocolHandler, session_id: str) -> None:
    """Tools can attach resource links to their output."""
    banner("3. Resource Links (add_resource_link)")
    print("\n  Tools call add_resource_link() to reference server resources.", file=sys.stderr)

    async def mock_send(request: dict) -> dict:
        return {"jsonrpc": "2.0", "id": request.get("id", 1), "result": {}}

    handler._send_to_client = mock_send

    def create_report(title: str) -> str:
        """Create a report and link to downloadable resources."""
        add_resource_link(
            uri=f"reports://{title.lower().replace(' ', '-')}",
            name=f"Report: {title}",
            description="Full report in PDF format",
            mime_type="application/pdf",
        )
        add_resource_link(
            uri=f"data://{title.lower().replace(' ', '-')}/raw",
            name="Raw Data",
            mime_type="text/csv",
        )
        return f"Report '{title}' generated with 2 linked resources"

    tool = ToolHandler.from_function(create_report, name="create_report", description="Create report")
    handler.register_tool(tool)

    response, _ = await handler._handle_tools_call({"name": "create_report", "arguments": {"title": "Q4 Sales"}}, 30)

    result = response["result"]
    print(f"\n  Tool result text: {result['content'][0]['text']}", file=sys.stderr)

    if "_meta" in result and "links" in result["_meta"]:
        print(f"  Attached {len(result['_meta']['links'])} resource links:", file=sys.stderr)
        for link in result["_meta"]["links"]:
            print(f"    - {link['name']}: {link['uri']} ({link.get('mimeType', 'n/a')})", file=sys.stderr)

    show_json("Tool result with resource links", result)
    handler._send_to_client = None


# ============================================================================
# 4. Content Annotations
# ============================================================================


async def demo_content_annotations(handler: MCPProtocolHandler) -> None:
    """Content can be annotated with audience and priority."""
    banner("4. Content Annotations")
    print("\n  Content types support audience and priority annotations.", file=sys.stderr)
    print("  audience: ['user'] or ['assistant'] -- controls who sees the content", file=sys.stderr)
    print("  priority: 0.0 to 1.0 -- hints at content importance\n", file=sys.stderr)

    # Show the annotation structure
    annotated_content = {
        "type": "text",
        "text": "The analysis found 3 anomalies in the dataset.",
        "annotations": {
            "audience": ["user"],
            "priority": 0.8,
        },
    }
    show_json("User-facing content (high priority)", annotated_content)

    assistant_content = {
        "type": "text",
        "text": "Internal note: anomaly detection used z-score > 3.0",
        "annotations": {
            "audience": ["assistant"],
            "priority": 0.3,
        },
    }
    show_json("Assistant-only content (low priority)", assistant_content)

    print("\n  Clients use annotations to filter what users see.", file=sys.stderr)


# ============================================================================
# Main
# ============================================================================


async def main():
    print("\nContext API Features Demo (MCP 2025-06-18)", file=sys.stderr)
    print("Demonstrates server-to-client features inside tool functions", file=sys.stderr)

    server_info = ServerInfo(name="context-demo", version="0.19.0")
    capabilities = create_server_capabilities(logging=True)
    handler = MCPProtocolHandler(server_info, capabilities)

    # Initialize a session
    init_msg = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-06-18",
            "capabilities": {"sampling": {}, "elicitation": {}, "roots": {"listChanged": True}},
            "clientInfo": {"name": "demo-client", "version": "1.0.0"},
        },
    }
    _, session_id = await handler.handle_request(init_msg)
    set_session_id(session_id)
    print(f"\n  Session: {session_id[:12]}...", file=sys.stderr)

    try:
        await demo_log_notifications(handler, session_id)
        await demo_progress(handler, session_id)
        await demo_resource_links(handler, session_id)
        await demo_content_annotations(handler)

        banner("Summary")
        print(
            """
  Feature             | Context API          | Protocol Method
  --------------------|----------------------|-----------------------------
  Log notifications   | send_log()           | notifications/message
  Progress            | send_progress()      | notifications/progress
  Resource links      | add_resource_link()  | _meta.links in tool result
  Content annotations | (in content dicts)   | annotations.audience/priority
""",
            file=sys.stderr,
        )
        print("  All 4 features demonstrated successfully!", file=sys.stderr)

    finally:
        clear_all()


if __name__ == "__main__":
    asyncio.run(main())
