#!/usr/bin/env python3
# src/chuk_mcp_server/types/parameters.py
"""
Parameters - Tool parameter definitions and JSON Schema generation

This module handles tool parameter type inference, JSON Schema generation,
and orjson optimization for maximum performance in schema operations.
"""

import inspect
from dataclasses import dataclass
from types import UnionType
from typing import Any, Union, cast

import orjson

# ============================================================================
# Pre-computed orjson Schema Fragments for Maximum Performance
# ============================================================================

# Pre-serialize common JSON schema fragments with orjson for maximum speed
_SCHEMA_FRAGMENTS = {
    "string": orjson.dumps({"type": "string"}),
    "integer": orjson.dumps({"type": "integer"}),
    "number": orjson.dumps({"type": "number"}),
    "boolean": orjson.dumps({"type": "boolean"}),
    "array": orjson.dumps({"type": "array"}),
    "object": orjson.dumps({"type": "object"}),
}

# Pre-computed base schemas for common parameter patterns
_BASE_SCHEMAS = {
    ("string", True, None): orjson.dumps({"type": "string"}),
    ("string", False, None): orjson.dumps({"type": "string"}),
    ("integer", True, None): orjson.dumps({"type": "integer"}),
    ("integer", False, None): orjson.dumps({"type": "integer"}),
    ("number", True, None): orjson.dumps({"type": "number"}),
    ("number", False, None): orjson.dumps({"type": "number"}),
    ("boolean", True, None): orjson.dumps({"type": "boolean"}),
    ("boolean", False, None): orjson.dumps({"type": "boolean"}),
}

# ============================================================================
# Pydantic Model Detection
# ============================================================================


def _is_pydantic_model(type_annotation: Any) -> bool:
    """Check if a type annotation is a Pydantic BaseModel subclass."""
    try:
        # Import here to avoid hard dependency
        from pydantic import BaseModel

        # Check if it's a class and subclass of BaseModel (but not BaseModel itself)
        return (
            inspect.isclass(type_annotation)
            and issubclass(type_annotation, BaseModel)
            and type_annotation is not BaseModel
        )
    except ImportError:  # pragma: no cover
        # Pydantic not available - hard to test without uninstalling pydantic
        return False
    except TypeError:
        # Not a class or can't check subclass
        return False


# ============================================================================
# Tool Parameter with orjson Optimization
# ============================================================================


@dataclass
class ToolParameter:
    """Tool parameter definition with orjson-optimized schema generation."""

    name: str
    type: str
    description: str | None = None
    required: bool = True
    default: Any = None
    enum: list[Any] | None = None
    items_type: str | None = None  # For array types: the type of items in the array
    pydantic_model: Any = None  # For Pydantic models: the model class itself
    pydantic_items_model: Any = None  # For list[PydanticModel]: the model class for array items
    _cached_schema: bytes | None = None  # 🚀 Cache orjson-serialized schema

    @classmethod
    def from_annotation(cls, name: str, annotation: Any, default: Any = inspect.Parameter.empty) -> "ToolParameter":
        """Create parameter from function annotation with modern typing support."""
        import typing
        from typing import Union  # Add explicit imports

        # Enhanced type mapping for modern Python
        type_map = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
            list: "array",
            dict: "object",
        }

        param_type = "string"  # default
        enum_values = None
        items_type = None  # Track array item type
        pydantic_model = None  # Track Pydantic model for direct types
        pydantic_items_model = None  # Track Pydantic model for array items

        # First check for direct basic types (most common case)
        if annotation in type_map:
            param_type = type_map[annotation]
        # Handle modern typing features
        elif hasattr(typing, "get_origin") and hasattr(typing, "get_args"):
            origin = typing.get_origin(annotation)
            args = typing.get_args(annotation)

            # Handle both typing.Union and types.UnionType (Python 3.10+ X | Y syntax)
            if origin is Union or (UnionType is not None and origin is UnionType):
                # Handle Optional[T] and Union types (including T | None syntax)
                if len(args) == 2 and type(None) in args:
                    # Optional[T] case - recursively infer type of non-None arg
                    non_none_type = next(arg for arg in args if arg is not type(None))
                    # Check if it's a basic type first
                    if non_none_type in type_map:
                        param_type = type_map[non_none_type]
                    else:
                        # Recursively process complex types like list[dict]
                        non_none_origin = typing.get_origin(non_none_type)
                        non_none_args = typing.get_args(non_none_type)

                        if non_none_origin in (list, list):
                            param_type = "array"
                            # Extract item type from List[T] or list[T]
                            if non_none_args:  # pragma: no branch
                                item_arg = non_none_args[0]
                                # Check if item is a Pydantic model
                                if _is_pydantic_model(item_arg):
                                    items_type = "object"
                                    pydantic_items_model = item_arg
                                else:
                                    items_type = type_map.get(item_arg, "string")
                        elif non_none_origin in (dict, dict):
                            param_type = "object"
                        else:  # pragma: no cover - defensive fallback
                            param_type = type_map.get(non_none_type, "string")
                else:
                    # Multiple union types - try to find common JSON Schema type
                    non_none_args = tuple(arg for arg in args if arg is not type(None))

                    # Check if all types are numeric (int, float)
                    if all(arg in (int, float) for arg in non_none_args):
                        param_type = "number"  # JSON Schema "number" accepts both int and float
                    # Check if all types map to the same JSON Schema type
                    elif len({type_map.get(arg) for arg in non_none_args}) == 1:
                        param_type = type_map.get(non_none_args[0], "string")
                    else:
                        # Mixed types - default to string
                        param_type = "string"

            # Handle Literal types for enums
            elif hasattr(typing, "Literal") and origin is typing.Literal:
                param_type = "string"
                enum_values = list(args)

            # Handle generic containers
            elif origin in (list, list):
                param_type = "array"
                # Extract item type from List[T] or list[T]
                if args:  # pragma: no branch
                    item_arg = args[0]
                    # Check if item is a Pydantic model
                    if _is_pydantic_model(item_arg):
                        items_type = "object"
                        pydantic_items_model = item_arg
                    else:
                        items_type = type_map.get(item_arg, "string")
            elif origin in (dict, dict):
                param_type = "object"
            else:
                # origin is None or unknown type
                # Check if the original annotation is a Pydantic model
                if _is_pydantic_model(annotation):
                    param_type = "object"
                    pydantic_model = annotation
                else:
                    param_type = type_map.get(origin, "string")

        # Fallback for older typing or direct types (Python 3.7-3.8 compatibility)
        elif hasattr(annotation, "__origin__"):  # pragma: no cover - Python 3.7-3.8 compatibility
            origin = annotation.__origin__
            if origin is Union:
                args = annotation.__args__
                if len(args) == 2 and type(None) in args:
                    # Optional[T] case - recursively infer type of non-None arg
                    non_none_type = next(arg for arg in args if arg is not type(None))
                    # Check if it's a basic type first
                    if non_none_type in type_map:
                        param_type = type_map[non_none_type]
                    else:
                        # Recursively process complex types like List[dict]
                        if hasattr(non_none_type, "__origin__"):
                            non_none_origin = non_none_type.__origin__
                            if non_none_origin in (list, list):
                                param_type = "array"
                                # Extract item type from List[T] or list[T]
                                if hasattr(non_none_type, "__args__") and non_none_type.__args__:
                                    item_arg = non_none_type.__args__[0]
                                    # Check if item is a Pydantic model
                                    if _is_pydantic_model(item_arg):
                                        items_type = "object"
                                        pydantic_items_model = item_arg
                                    else:
                                        items_type = type_map.get(item_arg, "string")
                            elif non_none_origin in (dict, dict):
                                param_type = "object"
                            else:
                                param_type = type_map.get(non_none_origin, "string")
                        else:
                            param_type = type_map.get(non_none_type, "string")
                else:
                    # Multiple union types - try to find common JSON Schema type
                    non_none_args = tuple(arg for arg in args if arg is not type(None))

                    # Check if all types are numeric (int, float)
                    if all(arg in (int, float) for arg in non_none_args):
                        param_type = "number"  # JSON Schema "number" accepts both int and float
                    # Check if all types map to the same JSON Schema type
                    elif len({type_map.get(arg) for arg in non_none_args}) == 1:
                        param_type = type_map.get(non_none_args[0], "string")
                    else:
                        # Mixed types - default to string
                        param_type = "string"
            elif origin in (list, list):
                param_type = "array"
                # Extract item type from List[T] or list[T]
                if hasattr(annotation, "__args__") and annotation.__args__:
                    item_arg = annotation.__args__[0]
                    # Check if item is a Pydantic model
                    if _is_pydantic_model(item_arg):
                        items_type = "object"
                        pydantic_items_model = item_arg
                    else:
                        items_type = type_map.get(item_arg, "string")
            elif origin in (dict, dict):
                param_type = "object"
            else:
                # origin is None or unknown type
                # Check if the original annotation is a Pydantic model
                if _is_pydantic_model(annotation):
                    param_type = "object"
                    pydantic_model = annotation
                else:
                    param_type = type_map.get(origin, "string")
        else:
            # Handle direct type annotations (int, str, bool, etc.)
            # Check if it's a Pydantic model
            if _is_pydantic_model(annotation):
                param_type = "object"
                pydantic_model = annotation
            else:
                param_type = type_map.get(annotation, "string")

        # Check if it has a default value
        required = default is inspect.Parameter.empty
        actual_default = None if default is inspect.Parameter.empty else default

        return cls(
            name=name,
            type=param_type,
            description=None,
            required=required,
            default=actual_default,
            enum=enum_values,
            items_type=items_type,
            pydantic_model=pydantic_model,
            pydantic_items_model=pydantic_items_model,
            _cached_schema=None,  # Will be computed on first access
        )

    def to_json_schema(self) -> dict[str, Any]:
        """Convert to JSON Schema format with orjson optimization."""
        # Check if we can use a pre-computed base schema
        cache_key = (self.type, self.required, self.default)
        if (
            cache_key in _BASE_SCHEMAS
            and not self.description
            and not self.enum
            and not self.items_type
            and not self.pydantic_model
            and not self.pydantic_items_model
        ):
            # Return pre-computed schema for maximum speed
            return cast(dict[str, Any], orjson.loads(_BASE_SCHEMAS[cache_key]))
        # Build custom schema
        schema: dict[str, Any] = {"type": self.type}

        # Handle Pydantic model for direct object types
        if self.type == "object" and self.pydantic_model:
            # Extract full Pydantic JSON schema
            try:
                pydantic_schema = self.pydantic_model.model_json_schema(mode="serialization")
                # Keep $defs if present - they contain referenced types (enums, nested models)
                if "$defs" in pydantic_schema:
                    defs = pydantic_schema.pop("$defs")
                    schema["$defs"] = defs
                schema.update(pydantic_schema)
            except Exception:
                # If Pydantic schema extraction fails, keep generic object
                pass

        # Add items field for array types
        if self.type == "array":
            if self.pydantic_items_model:
                # Extract Pydantic schema for array items
                try:
                    items_schema = self.pydantic_items_model.model_json_schema(mode="serialization")
                    # Remove top-level "$defs" if present
                    if "$defs" in items_schema:
                        # Keep $defs at the top level if needed for nested references
                        defs = items_schema.pop("$defs")
                        schema["$defs"] = defs
                    schema["items"] = items_schema
                except Exception:
                    # Fallback to generic object
                    schema["items"] = {"type": "object"}
            elif self.items_type:
                schema["items"] = {"type": self.items_type}

        if self.description:
            schema["description"] = self.description
        if self.enum:
            schema["enum"] = self.enum
        if self.default is not None:
            schema["default"] = self.default

        return schema

    def to_json_schema_bytes(self) -> bytes:
        """Get orjson-serialized schema bytes for maximum performance."""
        if self._cached_schema is None:
            schema = self.to_json_schema()
            self._cached_schema = orjson.dumps(schema)
        return self._cached_schema

    def invalidate_cache(self) -> None:
        """Invalidate the cached schema."""
        self._cached_schema = None


# ============================================================================
# Schema Generation Utilities
# ============================================================================


def build_input_schema(parameters: list[ToolParameter]) -> dict[str, Any]:
    """Build JSON Schema input schema from parameters."""
    properties = {}
    required = []
    all_defs = {}

    for param in parameters:
        param_schema = param.to_json_schema()
        # Extract and merge $defs if present
        if "$defs" in param_schema:
            defs = param_schema.pop("$defs")
            all_defs.update(defs)
        properties[param.name] = param_schema
        if param.required:
            required.append(param.name)

    schema = {"type": "object", "properties": properties, "required": required if required else None}
    # Add merged $defs if any were found
    if all_defs:
        schema["$defs"] = all_defs

    return schema


def build_input_schema_bytes(parameters: list[ToolParameter]) -> bytes:
    """Build orjson-serialized input schema for maximum performance."""
    schema = build_input_schema(parameters)
    result: bytes = orjson.dumps(schema)
    return result


# ============================================================================
# Type Inference Utilities
# ============================================================================


def infer_type_from_annotation(annotation: Any) -> str:
    """Infer JSON Schema type from Python type annotation."""
    import typing

    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
    }

    # First check for direct basic types (most common case)
    if annotation in type_map:
        return type_map[annotation]

    # Handle modern typing features
    if hasattr(typing, "get_origin") and hasattr(typing, "get_args"):
        origin = typing.get_origin(annotation)

        # Handle both typing.Union and types.UnionType (Python 3.10+ X | Y syntax)
        if origin is Union or (UnionType is not None and origin is UnionType):
            args = typing.get_args(annotation)
            if len(args) == 2 and type(None) in args:
                # Optional[T] case (Union[T, None] or T | None)
                non_none_type = next(arg for arg in args if arg is not type(None))
                return type_map.get(non_none_type, "string")
            else:
                # Multiple union types - try to find common JSON Schema type
                non_none_args = [arg for arg in args if arg is not type(None)]

                # Check if all types are numeric (int, float)
                if all(arg in (int, float) for arg in non_none_args):
                    return "number"  # JSON Schema "number" accepts both int and float
                # Check if all types map to the same JSON Schema type
                elif len({type_map.get(arg) for arg in non_none_args}) == 1:
                    return type_map.get(non_none_args[0], "string")
                else:
                    # Mixed types - default to string
                    return "string"

        # Handle generic containers
        elif origin in (list, list):
            return "array"
        elif origin in (dict, dict):
            return "object"
        else:
            return type_map.get(origin, "string")

    # Fallback for older typing or direct types (Python 3.7-3.8 compatibility)
    elif hasattr(annotation, "__origin__"):  # pragma: no cover - Python 3.7-3.8 compatibility
        origin = annotation.__origin__
        if origin is Union:
            args = annotation.__args__
            if len(args) == 2 and type(None) in args:
                non_none_type = next(arg for arg in args if arg is not type(None))
                return type_map.get(non_none_type, "string")
            else:
                # Multiple union types - try to find common JSON Schema type
                non_none_args = [arg for arg in args if arg is not type(None)]

                # Check if all types are numeric (int, float)
                if all(arg in (int, float) for arg in non_none_args):
                    return "number"  # JSON Schema "number" accepts both int and float
                # Check if all types map to the same JSON Schema type
                elif len({type_map.get(arg) for arg in non_none_args}) == 1:
                    return type_map.get(non_none_args[0], "string")
                else:
                    # Mixed types - default to string
                    return "string"
        elif origin in (list, list):
            return "array"
        elif origin in (dict, dict):
            return "object"
        else:
            return type_map.get(origin, "string")

    # Handle direct types (int, str, bool, etc.)
    return type_map.get(annotation, "string")


def extract_parameters_from_function(func: Any) -> list[ToolParameter]:
    """Extract parameters from a function signature."""
    sig = inspect.signature(func)
    parameters = []

    for param_name, param in sig.parameters.items():
        if param_name == "self":  # Skip self parameter for methods
            continue

        tool_param = ToolParameter.from_annotation(
            name=param_name,
            annotation=param.annotation if param.annotation != inspect.Parameter.empty else str,
            default=param.default,
        )
        parameters.append(tool_param)

    return parameters


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    "ToolParameter",
    "build_input_schema",
    "build_input_schema_bytes",
    "infer_type_from_annotation",
    "extract_parameters_from_function",
]
