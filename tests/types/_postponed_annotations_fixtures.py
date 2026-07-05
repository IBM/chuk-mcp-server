"""Fixture functions compiled under PEP 563 (postponed evaluation of annotations).

Kept in a separate module on purpose: `from __future__ import annotations` is a
per-module compiler directive, so a repro for "annotations arrive as source-text
strings, not real type objects" has to live in its own file — a function merely
*defined inside* a test that lacks this import still gets real type objects.
"""

from __future__ import annotations


def run_cell(id: str, args: list[int] | None = None, fields: dict | None = None) -> dict:
    """Mirrors a real MCP tool signature (cell80-mcp's `cell_run`)."""


def search(query: str, limit: int = 10) -> dict:
    """A plain `int` default alongside `str` — both should resolve, not just str."""


def route_by_example(examples: list[dict]) -> dict:
    """A bare (non-Optional) parameterized generic."""


def import_facts(facts: str, verify_fraction: float = 0.01, quarantine: bool = False) -> dict:
    """`float` and `bool` defaults — neither is `str`, so the old bug always caught these."""
