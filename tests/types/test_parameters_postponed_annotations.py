"""Regression test: `extract_parameters_from_function` must resolve real types for
functions compiled under `from __future__ import annotations` (PEP 563).

Without this, `inspect.signature(func).parameters[name].annotation` is the literal
source text (e.g. the string `"list[int] | None"`), not a type object. Every
parameter then falls through `ToolParameter.from_annotation`'s type_map lookups and
silently becomes `{"type": "string"}` in the generated JSON Schema — a client that
sends a real array/int/bool per the *declared* Python signature gets a tool crash
instead, because the tool's own schema told it to send a string.

This bit a real MCP server (cell80-mcp, which uses `from __future__ import
annotations`): calling its `cell_run(id, args: list[int] | None)` tool over MCP
raised `TypeError: argument 'args': 'str' object cannot be interpreted as an
integer`, even though the identical Python call worked fine directly.
"""

from _postponed_annotations_fixtures import import_facts, route_by_example, run_cell, search

from chuk_mcp_server.types.parameters import extract_parameters_from_function


def _by_name(params, name):
    return next(p for p in params if p.name == name)


def test_postponed_str_and_optional_list_params_resolve_correctly():
    params = extract_parameters_from_function(run_cell)

    assert _by_name(params, "id").type == "string"

    args_param = _by_name(params, "args")
    assert args_param.type == "array"
    assert args_param.items_type == "integer"
    assert args_param.required is False

    fields_param = _by_name(params, "fields")
    assert fields_param.type == "object"
    assert fields_param.required is False


def test_postponed_int_default_is_not_string():
    params = extract_parameters_from_function(search)

    assert _by_name(params, "query").type == "string"

    limit_param = _by_name(params, "limit")
    assert limit_param.type == "integer"
    assert limit_param.default == 10


def test_postponed_bare_generic_list_param():
    params = extract_parameters_from_function(route_by_example)

    examples_param = _by_name(params, "examples")
    assert examples_param.type == "array"
    assert examples_param.items_type == "object"


def test_postponed_float_and_bool_defaults_are_not_string():
    params = extract_parameters_from_function(import_facts)

    assert _by_name(params, "facts").type == "string"

    verify_fraction = _by_name(params, "verify_fraction")
    assert verify_fraction.type == "number"
    assert verify_fraction.default == 0.01

    quarantine = _by_name(params, "quarantine")
    assert quarantine.type == "boolean"
    assert quarantine.default is False
