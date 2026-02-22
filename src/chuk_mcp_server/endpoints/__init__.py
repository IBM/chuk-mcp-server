#!/usr/bin/env python3
"""
Endpoints - Clean imports for optimized endpoint handlers

Simple, focused imports without redundant metadata.
Each endpoint file contains its own documentation and performance info.
"""

# Class-based endpoints
# Add the ultra-fast health endpoint function
from .health import HealthEndpoint, handle_health_detailed, handle_health_ready, handle_health_ultra_fast
from .info import InfoEndpoint
from .mcp import MCPEndpoint

# Function-based endpoints (optimized)
from .ping import handle_request as handle_ping
from .utils import error_response_fast as error_response
from .utils import internal_error_response, method_not_allowed_response, not_found_response

# Utilities
from .utils import json_response_fast as json_response
from .version import handle_request as handle_version

__all__ = [  # FIXED: Double underscores, not asterisks
    # Endpoints
    "MCPEndpoint",
    "HealthEndpoint",
    "InfoEndpoint",
    "handle_ping",
    "handle_version",
    "handle_health_ultra_fast",
    "handle_health_ready",
    "handle_health_detailed",
    # Utilities
    "json_response",
    "error_response",
    "not_found_response",
    "method_not_allowed_response",
    "internal_error_response",
]
