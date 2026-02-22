#!/usr/bin/env python3
"""
Example: Tool & Resource Features (MCP 2025-06-18 / 2025-11-25)

Demonstrates the decorator-level features added in Phase 3 and Phase 4:
- Enhanced ServerInfo (description, icons, website_url)
- Tool annotations (read_only_hint, destructive_hint, idempotent_hint)
- Structured tool output (output_schema + structuredContent)
- Icons on tools, resources, prompts, and resource templates
- Tool name validation (dots and hyphens in names)
- Resource templates (RFC 6570 URI templates)
- Resource links from tool results

Run:
    python examples/tool_features_example.py
"""

from chuk_mcp_server import ChukMCPServer
from chuk_mcp_server.context import add_resource_link

# ============================================================================
# Server with Enhanced ServerInfo (MCP 2025-11-25)
# ============================================================================

mcp = ChukMCPServer(
    name="tool-features-demo",
    title="Tool Features Demo",
    description="Demonstrates tool annotations, structured output, and icons",
    icons=[{"uri": "https://example.com/icons/server.png", "mimeType": "image/png"}],
    website_url="https://github.com/chuk-ai/chuk-mcp-server",
)

# ============================================================================
# Tool Annotations + Structured Output + Icons
# ============================================================================


@mcp.tool(
    name="db.lookup",
    description="Look up a user by ID with typed JSON output",
    read_only_hint=True,
    idempotent_hint=True,
    destructive_hint=False,
    output_schema={
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "name": {"type": "string"},
            "email": {"type": "string"},
        },
        "required": ["id", "name"],
    },
    icons=[{"uri": "https://example.com/icons/search.svg", "mimeType": "image/svg+xml"}],
)
def lookup_user(user_id: str) -> dict:
    """Look up a user record by their ID."""
    users = {
        "u1": {"id": "u1", "name": "Alice", "email": "alice@example.com"},
        "u2": {"id": "u2", "name": "Bob", "email": "bob@example.com"},
    }
    return users.get(user_id, {"id": user_id, "name": "Unknown", "email": ""})


@mcp.tool(
    name="db.delete-record",
    description="Delete a record (destructive operation)",
    destructive_hint=True,
    read_only_hint=False,
    idempotent_hint=False,
)
def delete_record(record_id: str) -> str:
    """Delete a record by ID. This is a destructive, non-idempotent operation."""
    return f"Record {record_id} deleted"


# ============================================================================
# Tool with Resource Links
# ============================================================================


@mcp.tool(
    name="report.generate",
    description="Generate a report and link to its resource URI",
)
def generate_report(topic: str) -> str:
    """Generate a report and attach a resource link for downloading it."""
    report_uri = f"reports://{topic.lower().replace(' ', '-')}"

    # Attach a resource link to the tool result
    add_resource_link(
        uri=report_uri,
        name=f"Report: {topic}",
        description=f"Generated report on {topic}",
        mime_type="application/pdf",
    )

    return f"Report on '{topic}' generated. See attached resource link."


# ============================================================================
# Resource Template (RFC 6570 URI Templates)
# ============================================================================


@mcp.resource_template(
    "users://{user_id}/profile",
    name="User Profile",
    description="Retrieve a user profile by ID",
    icons=[{"uri": "https://example.com/icons/user.svg", "mimeType": "image/svg+xml"}],
)
def get_user_profile(user_id: str) -> dict:
    """Get a user's profile. The {user_id} is extracted from the URI."""
    return {"id": user_id, "name": f"User {user_id}", "role": "member"}


# ============================================================================
# Resource with Icons
# ============================================================================


@mcp.resource(
    "system://status",
    name="System Status",
    description="Current system health and status",
    icons=[{"uri": "https://example.com/icons/health.svg", "mimeType": "image/svg+xml"}],
)
def get_system_status() -> dict:
    """Return current system health information."""
    return {"status": "healthy", "uptime": "72h", "version": "1.0.0"}


# ============================================================================
# Prompt with Icons
# ============================================================================


@mcp.prompt(
    name="code-review",
    description="Generate a code review prompt",
    icons=[{"uri": "https://example.com/icons/review.svg", "mimeType": "image/svg+xml"}],
)
def code_review(code: str, language: str = "python") -> str:
    """Create a code review prompt for the given code."""
    return f"Please review this {language} code:\n\n```{language}\n{code}\n```"


# ============================================================================
# Run
# ============================================================================

if __name__ == "__main__":
    mcp.run()
