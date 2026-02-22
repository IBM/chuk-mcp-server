#!/usr/bin/env python3
"""
Comprehensive tests for request context management.

Tests cover all context functions and the RequestContext async context manager.
"""

import pytest

from chuk_mcp_server.context import (
    RequestContext,
    clear_all,
    clear_metadata,
    create_message,
    get_current_context,
    get_metadata,
    get_progress_token,
    get_sampling_fn,
    get_session_id,
    get_user_id,
    require_session_id,
    require_user_id,
    set_metadata,
    set_progress_token,
    set_sampling_fn,
    set_session_id,
    set_user_id,
    update_metadata,
)


@pytest.fixture(autouse=True)
def cleanup_context():
    """Ensure context is clean before and after each test."""
    clear_all()
    yield
    clear_all()


class TestSessionContext:
    """Test session ID context management."""

    def test_get_session_id_default(self):
        """Test get_session_id returns None by default."""
        assert get_session_id() is None

    def test_set_and_get_session_id(self):
        """Test setting and getting session ID."""
        set_session_id("session123")
        assert get_session_id() == "session123"

    def test_set_session_id_none(self):
        """Test setting session ID to None."""
        set_session_id("session123")
        set_session_id(None)
        assert get_session_id() is None

    def test_require_session_id_with_session(self):
        """Test require_session_id returns session when set."""
        set_session_id("session456")
        assert require_session_id() == "session456"

    def test_require_session_id_without_session(self):
        """Test require_session_id raises when not set."""
        with pytest.raises(RuntimeError) as exc_info:
            require_session_id()
        assert "Session context required" in str(exc_info.value)

    def test_require_session_id_empty_string(self):
        """Test require_session_id raises with empty string."""
        set_session_id("")
        with pytest.raises(RuntimeError):
            require_session_id()


class TestUserContext:
    """Test user ID context management."""

    def test_get_user_id_default(self):
        """Test get_user_id returns None by default."""
        assert get_user_id() is None

    def test_set_and_get_user_id(self):
        """Test setting and getting user ID."""
        set_user_id("user789")
        assert get_user_id() == "user789"

    def test_set_user_id_none(self):
        """Test setting user ID to None."""
        set_user_id("user789")
        set_user_id(None)
        assert get_user_id() is None

    def test_require_user_id_with_user(self):
        """Test require_user_id returns user when set."""
        set_user_id("user123")
        assert require_user_id() == "user123"

    def test_require_user_id_without_user(self):
        """Test require_user_id raises PermissionError when not set."""
        with pytest.raises(PermissionError) as exc_info:
            require_user_id()
        assert "User authentication required" in str(exc_info.value)
        assert "OAuth authentication" in str(exc_info.value)

    def test_require_user_id_empty_string(self):
        """Test require_user_id raises with empty string."""
        set_user_id("")
        with pytest.raises(PermissionError):
            require_user_id()


class TestProgressToken:
    """Test progress token context management."""

    def test_get_progress_token_default(self):
        """Test get_progress_token returns None by default."""
        assert get_progress_token() is None

    def test_set_and_get_progress_token_string(self):
        """Test setting and getting string progress token."""
        set_progress_token("token123")
        assert get_progress_token() == "token123"

    def test_set_and_get_progress_token_int(self):
        """Test setting and getting integer progress token."""
        set_progress_token(42)
        assert get_progress_token() == 42

    def test_set_progress_token_none(self):
        """Test setting progress token to None."""
        set_progress_token("token123")
        set_progress_token(None)
        assert get_progress_token() is None


class TestMetadata:
    """Test metadata context management."""

    def test_get_metadata_default(self):
        """Test get_metadata returns empty dict by default."""
        assert get_metadata() == {}

    def test_set_and_get_metadata(self):
        """Test setting and getting metadata."""
        metadata = {"key1": "value1", "key2": 42}
        set_metadata(metadata)
        result = get_metadata()
        assert result == metadata
        # Verify it's a copy
        assert result is not metadata

    def test_get_metadata_returns_copy(self):
        """Test get_metadata returns a copy to prevent mutation."""
        set_metadata({"key": "value"})
        meta1 = get_metadata()
        meta1["key"] = "modified"
        meta2 = get_metadata()
        assert meta2["key"] == "value"  # Original unchanged

    def test_update_metadata_new_key(self):
        """Test update_metadata adds new key."""
        update_metadata("new_key", "new_value")
        assert get_metadata() == {"new_key": "new_value"}

    def test_update_metadata_existing_key(self):
        """Test update_metadata updates existing key."""
        set_metadata({"key": "old"})
        update_metadata("key", "new")
        assert get_metadata() == {"key": "new"}

    def test_update_metadata_preserves_other_keys(self):
        """Test update_metadata preserves other keys."""
        set_metadata({"key1": "value1", "key2": "value2"})
        update_metadata("key1", "updated")
        result = get_metadata()
        assert result == {"key1": "updated", "key2": "value2"}

    def test_update_metadata_when_none(self):
        """Test update_metadata works when metadata is None."""
        clear_metadata()
        update_metadata("key", "value")
        assert get_metadata() == {"key": "value"}

    def test_clear_metadata(self):
        """Test clear_metadata removes all metadata."""
        set_metadata({"key": "value"})
        clear_metadata()
        assert get_metadata() == {}


class TestClearAll:
    """Test clearing all context."""

    def test_clear_all(self):
        """Test clear_all removes all context values."""
        set_session_id("session123")
        set_user_id("user456")
        set_progress_token("token789")
        set_metadata({"key": "value"})

        clear_all()

        assert get_session_id() is None
        assert get_user_id() is None
        assert get_progress_token() is None
        assert get_metadata() == {}


class TestGetCurrentContext:
    """Test getting all context at once."""

    def test_get_current_context_empty(self):
        """Test get_current_context with no values set."""
        context = get_current_context()
        assert context == {
            "session_id": None,
            "user_id": None,
            "progress_token": None,
            "metadata": {},
            "sampling_available": False,
            "elicitation_available": False,
            "roots_available": False,
        }

    def test_get_current_context_all_set(self):
        """Test get_current_context with all values set."""
        set_session_id("session123")
        set_user_id("user456")
        set_progress_token(42)
        set_metadata({"key": "value"})

        context = get_current_context()
        assert context == {
            "session_id": "session123",
            "user_id": "user456",
            "progress_token": 42,
            "metadata": {"key": "value"},
            "sampling_available": False,
            "elicitation_available": False,
            "roots_available": False,
        }

    def test_get_current_context_returns_copy_of_metadata(self):
        """Test get_current_context returns copy of metadata."""
        set_metadata({"key": "value"})
        context = get_current_context()
        context["metadata"]["key"] = "modified"
        assert get_metadata()["key"] == "value"


@pytest.mark.asyncio
class TestRequestContextManager:
    """Test RequestContext async context manager."""

    async def test_request_context_basic(self):
        """Test basic RequestContext usage."""
        async with RequestContext(session_id="session123", user_id="user456"):
            assert get_session_id() == "session123"
            assert get_user_id() == "user456"

        # Context should be cleared after exit
        assert get_session_id() is None
        assert get_user_id() is None

    async def test_request_context_all_parameters(self):
        """Test RequestContext with all parameters."""
        metadata = {"key": "value"}
        async with RequestContext(
            session_id="session123",
            user_id="user456",
            progress_token=42,
            metadata=metadata,
        ):
            assert get_session_id() == "session123"
            assert get_user_id() == "user456"
            assert get_progress_token() == 42
            assert get_metadata() == metadata

    async def test_request_context_none_values(self):
        """Test RequestContext with None values doesn't override."""
        set_session_id("existing_session")
        set_user_id("existing_user")

        async with RequestContext(session_id=None, user_id=None):
            # Should preserve existing values when passed None
            assert get_session_id() == "existing_session"
            assert get_user_id() == "existing_user"

    async def test_request_context_restores_previous(self):
        """Test RequestContext restores previous values on exit."""
        set_session_id("original_session")
        set_user_id("original_user")
        set_progress_token("original_token")
        set_metadata({"original": "data"})

        async with RequestContext(
            session_id="new_session",
            user_id="new_user",
            progress_token="new_token",
            metadata={"new": "data"},
        ):
            assert get_session_id() == "new_session"
            assert get_user_id() == "new_user"

        # Should restore original values
        assert get_session_id() == "original_session"
        assert get_user_id() == "original_user"
        assert get_progress_token() == "original_token"
        assert get_metadata() == {"original": "data"}

    async def test_request_context_nested(self):
        """Test nested RequestContext managers."""
        async with RequestContext(session_id="outer_session", user_id="outer_user"):
            assert get_session_id() == "outer_session"
            assert get_user_id() == "outer_user"

            async with RequestContext(session_id="inner_session", user_id="inner_user"):
                assert get_session_id() == "inner_session"
                assert get_user_id() == "inner_user"

            # Outer context restored
            assert get_session_id() == "outer_session"
            assert get_user_id() == "outer_user"

        # All cleared
        assert get_session_id() is None
        assert get_user_id() is None

    async def test_request_context_empty_metadata_dict(self):
        """Test RequestContext with empty metadata dict."""
        async with RequestContext(metadata={}):
            # Empty dict should not set metadata
            assert get_metadata() == {}

    async def test_request_context_metadata_copy(self):
        """Test RequestContext does not copy metadata dict on init (performance)."""
        original = {"key": "value"}
        async with RequestContext(metadata=original):
            # Note: RequestContext stores reference for performance
            # Modifying original affects context
            original["key"] = "modified"
            assert get_metadata() == {"key": "modified"}

    async def test_request_context_exception_handling(self):
        """Test RequestContext restores context even with exceptions."""
        set_session_id("original")

        try:
            async with RequestContext(session_id="exception_session"):
                assert get_session_id() == "exception_session"
                raise ValueError("Test exception")
        except ValueError:
            pass

        # Should restore original even after exception
        assert get_session_id() == "original"

    async def test_request_context_returns_self(self):
        """Test RequestContext __aenter__ returns self."""
        ctx = RequestContext(session_id="test")
        result = await ctx.__aenter__()
        assert result is ctx
        await ctx.__aexit__(None, None, None)

    async def test_request_context_aexit_returns_false(self):
        """Test RequestContext __aexit__ returns False (doesn't suppress exceptions)."""
        ctx = RequestContext(session_id="test")
        await ctx.__aenter__()
        result = await ctx.__aexit__(None, None, None)
        assert result is False

    async def test_request_context_preserves_metadata_copy_on_exit(self):
        """Test RequestContext copies previous metadata on entry."""
        original_meta = {"key": "original"}
        set_metadata(original_meta)

        async with RequestContext(session_id="test", metadata={"key": "new"}):
            assert get_metadata() == {"key": "new"}

        # Should restore copy of original
        restored = get_metadata()
        assert restored == {"key": "original"}
        # Verify it's a copy
        original_meta["key"] = "modified"
        assert get_metadata() == {"key": "original"}


class TestConvenienceAliases:
    """Test backward compatibility aliases."""

    def test_set_current_user_alias(self):
        """Test set_current_user is alias for set_user_id."""
        from chuk_mcp_server.context import set_current_user

        set_current_user("user123")
        assert get_user_id() == "user123"

    def test_get_current_user_id_alias(self):
        """Test get_current_user_id is alias for require_user_id."""
        from chuk_mcp_server.context import get_current_user_id

        set_user_id("user456")
        assert get_current_user_id() == "user456"

        # Should also raise when not set (like require_user_id)
        clear_all()
        with pytest.raises(PermissionError):
            get_current_user_id()


class TestSamplingContext:
    """Test sampling context management."""

    def test_get_sampling_fn_default(self):
        """get_sampling_fn returns None by default."""
        assert get_sampling_fn() is None

    def test_set_and_get_sampling_fn(self):
        """set_sampling_fn and get_sampling_fn round-trip."""

        async def mock_fn(**kwargs):
            return {"role": "assistant", "content": {"type": "text", "text": "ok"}}

        set_sampling_fn(mock_fn)
        assert get_sampling_fn() is mock_fn

    def test_set_sampling_fn_none(self):
        """Setting sampling fn to None clears it."""

        async def mock_fn(**kwargs):
            return {}

        set_sampling_fn(mock_fn)
        set_sampling_fn(None)
        assert get_sampling_fn() is None

    @pytest.mark.asyncio
    async def test_create_message_raises_when_no_fn(self):
        """create_message raises RuntimeError when no sampling fn set."""
        with pytest.raises(RuntimeError, match="MCP sampling is not available"):
            await create_message(messages=[{"role": "user", "content": {"type": "text", "text": "hello"}}])

    @pytest.mark.asyncio
    async def test_create_message_delegates_to_fn(self):
        """create_message delegates to the set sampling fn."""
        captured_kwargs = {}

        async def mock_fn(**kwargs):
            captured_kwargs.update(kwargs)
            return {
                "role": "assistant",
                "content": {"type": "text", "text": "response"},
                "model": "test-model",
            }

        set_sampling_fn(mock_fn)
        result = await create_message(
            messages=[{"role": "user", "content": {"type": "text", "text": "hello"}}],
            max_tokens=500,
            system_prompt="You are helpful.",
        )
        assert result["model"] == "test-model"
        assert captured_kwargs["max_tokens"] == 500
        assert captured_kwargs["system_prompt"] == "You are helpful."

    def test_clear_all_resets_sampling_fn(self):
        """clear_all() resets the sampling fn."""

        async def mock_fn(**kwargs):
            return {}

        set_sampling_fn(mock_fn)
        clear_all()
        assert get_sampling_fn() is None

    def test_get_current_context_includes_sampling(self):
        """get_current_context includes sampling_available."""
        ctx = get_current_context()
        assert ctx["sampling_available"] is False

        async def mock_fn(**kwargs):
            return {}

        set_sampling_fn(mock_fn)
        ctx = get_current_context()
        assert ctx["sampling_available"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
