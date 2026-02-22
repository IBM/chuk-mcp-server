#!/usr/bin/env python3
"""Tests for MCP content annotations (audience, priority)."""

from chuk_mcp_server.types.base import Annotations
from chuk_mcp_server.types.content import (
    create_annotated_content,
    create_resource_link,
    format_content,
)


class TestContentAnnotationsViaFormatContent:
    """Test annotations parameter on format_content()."""

    def test_no_annotations_by_default(self):
        """Content without annotations should not include the field."""
        result = format_content("hello")
        assert len(result) == 1
        assert "annotations" not in result[0]

    def test_annotations_as_dict(self):
        """Annotations can be passed as a plain dict."""
        ann = {"audience": ["user"], "priority": 0.8}
        result = format_content("hello", annotations=ann)
        assert result[0]["annotations"] == ann

    def test_annotations_as_pydantic(self):
        """Annotations can be passed as an Annotations Pydantic model."""
        ann = Annotations(audience=["user", "assistant"], priority=0.5)
        result = format_content("important data", annotations=ann)
        assert result[0]["annotations"]["audience"] == ["user", "assistant"]
        assert result[0]["annotations"]["priority"] == 0.5

    def test_annotations_on_dict_content(self):
        """Annotations apply to dict content."""
        ann = {"audience": ["assistant"]}
        result = format_content({"key": "value"}, annotations=ann)
        assert result[0]["annotations"]["audience"] == ["assistant"]

    def test_annotations_on_list_content(self):
        """Annotations apply to each item in a list."""
        ann = {"priority": 1.0}
        result = format_content(["first", "second"], annotations=ann)
        assert len(result) == 2
        assert result[0]["annotations"]["priority"] == 1.0
        assert result[1]["annotations"]["priority"] == 1.0

    def test_empty_annotations_not_applied(self):
        """Empty annotations dict should not add the field."""
        result = format_content("hello", annotations={})
        assert "annotations" not in result[0]

    def test_none_annotations_not_applied(self):
        """None annotations should not add the field."""
        result = format_content("hello", annotations=None)
        assert "annotations" not in result[0]

    def test_audience_only(self):
        """Audience-only annotations work."""
        result = format_content("data", annotations={"audience": ["user"]})
        assert result[0]["annotations"] == {"audience": ["user"]}

    def test_priority_only(self):
        """Priority-only annotations work."""
        result = format_content("data", annotations={"priority": 0.3})
        assert result[0]["annotations"] == {"priority": 0.3}


class TestCreateAnnotatedContent:
    """Test the create_annotated_content() convenience function."""

    def test_with_audience(self):
        result = create_annotated_content("hello", audience=["user"])
        assert result[0]["annotations"]["audience"] == ["user"]

    def test_with_priority(self):
        result = create_annotated_content("hello", priority=0.9)
        assert result[0]["annotations"]["priority"] == 0.9

    def test_with_both(self):
        result = create_annotated_content("hello", audience=["assistant"], priority=0.5)
        assert result[0]["annotations"]["audience"] == ["assistant"]
        assert result[0]["annotations"]["priority"] == 0.5

    def test_no_annotations(self):
        """Without audience or priority, no annotations are added."""
        result = create_annotated_content("hello")
        assert "annotations" not in result[0]

    def test_dict_content(self):
        result = create_annotated_content({"data": 42}, audience=["user"])
        assert result[0]["annotations"]["audience"] == ["user"]
        assert result[0]["type"] == "text"  # dict becomes JSON text


class TestCreateResourceLink:
    """Test the create_resource_link() helper."""

    def test_minimal_link(self):
        link = create_resource_link("file:///data.txt")
        assert link == {"uri": "file:///data.txt"}

    def test_full_link(self):
        link = create_resource_link(
            uri="file:///report.pdf",
            name="Monthly Report",
            description="The monthly sales report",
            mime_type="application/pdf",
        )
        assert link == {
            "uri": "file:///report.pdf",
            "name": "Monthly Report",
            "description": "The monthly sales report",
            "mimeType": "application/pdf",
        }

    def test_partial_link(self):
        link = create_resource_link("config://app", name="App Config")
        assert link == {"uri": "config://app", "name": "App Config"}
        assert "description" not in link
        assert "mimeType" not in link
