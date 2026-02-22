# ChukMCPServer Examples

Complete collection of working examples demonstrating ChukMCPServer features.

## Quick Start

```bash
# Run any example
uv run python examples/zero_config_example.py

# Or with regular Python
python examples/zero_config_example.py
```

## 📁 Examples by Category

### 🚀 Basic (Getting Started)

Perfect for beginners - start here!

| Example | Description | Key Features |
|---------|-------------|--------------|
| **[zero_config_example.py](zero_config_example.py)** | Zero-config async server with auto-detection | Async tools, auto-config, performance modes |
| **[stdio_example.py](stdio_example.py)** | STDIO transport for Claude Desktop | Standard MCP protocol, subprocess communication |
| **[example.py](example.py)** | Basic server with tools and resources | Foundation concepts, simple examples |

**Start with:** `zero_config_example.py` - Shows the simplest way to create an MCP server

### ⚡ Async & Performance

High-performance async examples for production use.

| Example | Description | Key Features |
|---------|-------------|--------------|
| **[async_example.py](async_example.py)** | Comprehensive async patterns | Async tools, httpx integration, error handling |
| **[standalone_async_e2e_demo.py](standalone_async_e2e_demo.py)** | End-to-end async demo | Full async flow, real-world patterns |

**Best for:** Production servers, high-throughput applications

### 🔄 Context Management

Session tracking, user authentication, and context isolation.

| Example | Description | Key Features |
|---------|-------------|--------------|
| **[context_basics_example.py](context_basics_example.py)** | Basic context usage | `get_session_id()`, `get_user_id()`, context access |
| **[context_session_isolation_example.py](context_session_isolation_example.py)** | Session-scoped state | Session isolation, concurrent sessions |
| **[context_user_persistence_example.py](context_user_persistence_example.py)** | User-scoped persistence | User authentication, persistent data |

**Documentation:** See [docs/guides/context-management.md](../docs/guides/context-management.md)

### 🧩 Composition & Modules

Building modular servers with reusable tool modules.

| Example | Description | Key Features |
|---------|-------------|--------------|
| **[composition_demo.py](composition_demo.py)** | Server composition patterns | Import servers, mount tools, composition |
| **[multi_module_hosting_demo.py](multi_module_hosting_demo.py)** | Host multiple tool modules | Module loading, namespace management |
| **[tool_modules/](tool_modules/)** | Reusable tool modules | `math_tools`, `text_tools` examples |

**Best for:** Large projects, team development, reusable components

### 🔗 Proxy & Multi-Server

Aggregate multiple MCP servers into one endpoint.

| Example | Description | Key Features |
|---------|-------------|--------------|
| **[proxy_demo.py](proxy_demo.py)** | Basic proxy setup | Proxy manager, server aggregation |
| **[proxy_config_example.yaml](proxy_config_example.yaml)** | Proxy configuration file | YAML config, declarative setup |
| **[proxy_example.py](proxy_example.py)** | Simple proxy | Quick proxy demonstration |
| **[proxy_multi_server_example.py](proxy_multi_server_example.py)** | Multi-server proxy | Multiple backends, routing |
| **[proxy_time_example.py](proxy_time_example.py)** | Proxy with time tools | Proxy + custom tools |
| **[simple_backend_server.py](simple_backend_server.py)** | Backend server for proxy | Example backend |
| **[simple_proxy_test.py](simple_proxy_test.py)** | Proxy testing | Test client |
| **[test_proxy_client.py](test_proxy_client.py)** | Proxy client tests | Client testing |

**Documentation:** See README section on Multi-Server Proxy

### ⚙️ Configuration

Smart configuration, logging, and transport options.

| Example | Description | Key Features |
|---------|-------------|--------------|
| **[demo_smart_config.py](demo_smart_config.py)** | Smart auto-configuration | Environment detection, auto-config |
| **[demo_logging_setlevel.py](demo_logging_setlevel.py)** | Logging configuration | Log levels, custom logging |
| **[mcp_logging_example.py](mcp_logging_example.py)** | MCP protocol logging | Request/response logging |
| **[constructor_transport_example.py](constructor_transport_example.py)** | Transport selection | STDIO vs HTTP transport |

**Best for:** Production configuration, debugging, deployment

### 🔌 Protocol Features

Demonstrations of MCP protocol capabilities.

| Example | Description | Key Features |
|---------|-------------|--------------|
| **[protocol_features_demo.py](protocol_features_demo.py)** | Protocol features demo | Completions, subscriptions, elicitation, progress, roots |
| **[tool_features_example.py](tool_features_example.py)** | Tool annotations, structured output, icons, resource templates | Phase 3 & 4 decorator features |
| **[context_features_example.py](context_features_example.py)** | Log notifications, progress, resource links, content annotations | Phase 3 context API features |
| **[advanced_protocol_example.py](advanced_protocol_example.py)** | Sampling with tools, URL elicitation, tasks system | Phase 4 protocol features |
| **[demo_logging_setlevel.py](demo_logging_setlevel.py)** | Dynamic log level changes | MCP logging/setLevel protocol |

**Best for:** Understanding MCP protocol, building advanced integrations

### 🎯 Advanced

Advanced patterns and architectures.

| Example | Description | Key Features |
|---------|-------------|--------------|
| **[clean_archictecture_demo.py](clean_archictecture_demo.py)** | Clean architecture pattern | Layered design, separation of concerns |
| **[prompt_example.py](prompt_example.py)** | Prompt templates | Reusable prompts, template system |

**Best for:** Enterprise applications, complex architectures

## 🧪 Testing Examples

Most examples can be tested with the MCP Inspector or direct HTTP calls:

### Using MCP Inspector (STDIO)

```bash
# Run example in STDIO mode
python examples/stdio_example.py

# In another terminal, use MCP Inspector
npx @modelcontextprotocol/inspector python examples/stdio_example.py
```

### Using HTTP Endpoints

```bash
# Run example in HTTP mode
python examples/zero_config_example.py --port 8000

# Test with curl
curl http://localhost:8000/health
curl http://localhost:8000/tools

# Call a tool
curl -X POST http://localhost:8000/mcp \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "tools/call",
    "params": {
      "name": "hello",
      "arguments": {"name": "World"}
    }
  }'
```

### Using Claude Desktop

Add to your Claude Desktop config (`~/Library/Application Support/Claude/claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "my-server": {
      "command": "uv",
      "args": [
        "run",
        "python",
        "/path/to/examples/stdio_example.py"
      ]
    }
  }
}
```

## 📝 Example Template

Use this template to create new examples:

```python
#!/usr/bin/env python3
"""
Example: [Brief Description]

Demonstrates:
- Feature 1
- Feature 2
- Feature 3
"""

from chuk_mcp_server import tool, run

@tool
def my_tool(param: str) -> str:
    """Tool description for Claude."""
    return f"Result: {param}"

if __name__ == "__main__":
    # STDIO mode (default - for Claude Desktop)
    run()

    # Or HTTP mode
    # run(transport="http", port=8000)
```

## 🔍 Finding Examples

**By Feature:**
- **Decorators:** `zero_config_example.py`, `prompt_example.py`
- **Tool Annotations:** `tool_features_example.py`
- **Structured Output:** `tool_features_example.py`
- **Icons:** `tool_features_example.py`
- **Resource Templates:** `tool_features_example.py`
- **Log Notifications:** `context_features_example.py`
- **Resource Links:** `context_features_example.py`
- **Sampling + Tools:** `advanced_protocol_example.py`
- **URL Elicitation:** `advanced_protocol_example.py`
- **Tasks System:** `advanced_protocol_example.py`
- **Async/Await:** `async_example.py`, `zero_config_example.py`
- **Session Management:** `context_session_isolation_example.py`
- **User Auth:** `context_user_persistence_example.py`
- **Proxy/Composition:** `proxy_demo.py`, `composition_demo.py`
- **Configuration:** `demo_smart_config.py`, `constructor_transport_example.py`
- **Transport Modes:** `stdio_example.py`, `constructor_transport_example.py`

**By Use Case:**
- **Claude Desktop:** `stdio_example.py`
- **Web API:** `zero_config_example.py` (HTTP mode)
- **Multi-server:** `proxy_multi_server_example.py`
- **Production:** `async_example.py`, `demo_smart_config.py`
- **Testing:** `simple_proxy_test.py`, `test_proxy_client.py`

## 🐛 Troubleshooting

### Example won't start

```bash
# Ensure dependencies are installed
uv sync

# Or with pip
pip install chuk-mcp-server
```

### Import errors

```bash
# Run from project root
cd /path/to/chuk-mcp-server
uv run python examples/example_name.py
```

### Port already in use

```bash
# Use a different port
python examples/zero_config_example.py --port 8001
```

## 📚 Related Documentation

- [Main README](../README.md) - Project overview
- [Getting Started Guide](../docs/getting-started/) - Step-by-step guides
- [API Reference](../docs/api-reference/README.md) - Complete API docs
- [Context Management](../docs/guides/context-management.md) - Context guide
- [Artifacts](../docs/guides/artifacts.md) - Artifact storage guide
- [OAuth Integration](../docs/OAUTH.md) - OAuth setup
- [Transport Modes](../docs/TRANSPORT_MODES.md) - STDIO vs HTTP

## 💡 Contributing Examples

Have a cool example? Add it!

1. Create your example file
2. Add it to the appropriate category in this README
3. Include clear comments and docstrings
4. Test it works with both STDIO and HTTP modes (if applicable)
5. Submit a PR

**Example Criteria:**
- ✅ Self-contained (runs independently)
- ✅ Well-commented
- ✅ Demonstrates one clear concept
- ✅ Includes docstrings
- ✅ Works with current version

## 📊 Example Statistics

- **Total Examples:** 29 Python files
- **Categories:** 8 (Basic, Async, Context, Composition, Proxy, Config, Protocol, Advanced)
- **Test Coverage:** Examples are tested in CI
- **Documentation:** All major features have examples

---

**Questions?** See the [main documentation](../README.md) or [open an issue](https://github.com/chuk-ai/chuk-mcp-server/issues).
