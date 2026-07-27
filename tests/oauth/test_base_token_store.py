"""Tests for BaseTokenStore abstract base class."""

import pytest

from chuk_mcp_server.oauth.base_token_store import BaseTokenStore


class TestBaseTokenStore:
    """Test BaseTokenStore abstract base class."""

    def test_cannot_instantiate_directly(self):
        """Test that BaseTokenStore cannot be instantiated directly."""
        with pytest.raises(TypeError) as exc_info:
            BaseTokenStore()
        assert "abstract" in str(exc_info.value).lower()

    def test_complete_subclass_can_be_instantiated(self):
        """Test that a complete subclass can be instantiated."""

        class CompleteTokenStore(BaseTokenStore):
            async def create_authorization_code(
                self, user_id, client_id, redirect_uri, scope=None, code_challenge=None, code_challenge_method=None
            ):
                return "auth_code_123"

            async def validate_authorization_code(self, code, client_id, redirect_uri, code_verifier=None):
                return {"user_id": "user123", "client_id": client_id}

            async def create_access_token(self, user_id, client_id, scope=None):
                return ("access_token_123", "refresh_token_123")

            async def validate_access_token(self, token):
                return {"user_id": "user123", "client_id": "client456"}

            async def refresh_access_token(self, refresh_token):
                return ("new_access_token", "new_refresh_token")

            async def link_external_token(
                self, user_id, access_token, refresh_token=None, expires_in=None, provider="external"
            ):
                pass

            async def get_external_token(self, user_id, provider="external"):
                return {"access_token": "external_token"}

            async def update_external_token(
                self, user_id, access_token, refresh_token=None, expires_in=None, provider="external"
            ):
                pass

            async def is_external_token_expired(self, user_id, provider="external"):
                return False

            async def register_client(self, client_name, redirect_uris):
                return {"client_id": "client123", "client_secret": "secret123"}

            async def validate_client(self, client_id, client_secret=None, redirect_uri=None):
                return True

        store = CompleteTokenStore()
        assert isinstance(store, BaseTokenStore)

    @pytest.mark.asyncio
    async def test_complete_subclass_methods_work(self):
        """Test that complete subclass methods are callable."""

        class CompleteTokenStore(BaseTokenStore):
            async def create_authorization_code(
                self, user_id, client_id, redirect_uri, scope=None, code_challenge=None, code_challenge_method=None
            ):
                return "auth_code_123"

            async def validate_authorization_code(self, code, client_id, redirect_uri, code_verifier=None):
                return {"user_id": "user123", "client_id": client_id, "scope": "read write"}

            async def create_access_token(self, user_id, client_id, scope=None):
                return ("access_token_123", "refresh_token_123")

            async def validate_access_token(self, token):
                return {"user_id": "user123", "client_id": "client456"}

            async def refresh_access_token(self, refresh_token):
                return ("new_access_token", "new_refresh_token")

            async def link_external_token(
                self, user_id, access_token, refresh_token=None, expires_in=None, provider="external"
            ):
                pass

            async def get_external_token(self, user_id, provider="external"):
                return {"access_token": "external_token", "provider": provider}

            async def update_external_token(
                self, user_id, access_token, refresh_token=None, expires_in=None, provider="external"
            ):
                pass

            async def is_external_token_expired(self, user_id, provider="external"):
                return False

            async def register_client(self, client_name, redirect_uris):
                return {"client_id": "client_new", "client_secret": "secret_new"}

            async def validate_client(self, client_id, client_secret=None, redirect_uri=None):
                return client_id == "valid_client"

        store = CompleteTokenStore()

        # Test create_authorization_code
        code = await store.create_authorization_code(
            user_id="user123",
            client_id="client456",
            redirect_uri="http://localhost/callback",
        )
        assert code == "auth_code_123"

        # Test validate_authorization_code
        data = await store.validate_authorization_code(
            code="auth_code",
            client_id="client456",
            redirect_uri="http://localhost/callback",
        )
        assert data["user_id"] == "user123"
        assert data["scope"] == "read write"

        # Test create_access_token
        access, refresh = await store.create_access_token(
            user_id="user123",
            client_id="client456",
        )
        assert access == "access_token_123"
        assert refresh == "refresh_token_123"

        # Test validate_access_token
        token_data = await store.validate_access_token("access_token")
        assert token_data["user_id"] == "user123"

        # Test refresh_access_token
        new_access, new_refresh = await store.refresh_access_token("refresh_token")
        assert new_access == "new_access_token"
        assert new_refresh == "new_refresh_token"

        # Test link_external_token
        await store.link_external_token(
            user_id="user123",
            access_token="external_token",
            refresh_token="external_refresh",
            expires_in=3600,
            provider="linkedin",
        )

        # Test get_external_token
        external = await store.get_external_token("user123", "linkedin")
        assert external["access_token"] == "external_token"
        assert external["provider"] == "linkedin"

        # Test update_external_token
        await store.update_external_token(
            user_id="user123",
            access_token="new_external_token",
            provider="linkedin",
        )

        # Test is_external_token_expired
        expired = await store.is_external_token_expired("user123", "linkedin")
        assert expired is False

        # Test register_client
        client = await store.register_client(
            client_name="New Client",
            redirect_uris=["http://localhost/callback"],
        )
        assert client["client_id"] == "client_new"
        assert client["client_secret"] == "secret_new"

        # Test validate_client
        valid = await store.validate_client("valid_client", "secret")
        assert valid is True
        invalid = await store.validate_client("invalid_client")
        assert invalid is False


class LegacyTokenStore(BaseTokenStore):
    """A token store written before authenticate_client existed."""

    def __init__(self):
        self.validate_calls = []

    async def create_authorization_code(
        self, user_id, client_id, redirect_uri, scope=None, code_challenge=None, code_challenge_method=None
    ):  # pragma: no cover - not exercised here
        return "auth_code"

    async def validate_authorization_code(
        self, code, client_id, redirect_uri, code_verifier=None
    ):  # pragma: no cover - not exercised here
        return None

    async def create_access_token(self, user_id, client_id, scope=None):  # pragma: no cover - not exercised here
        return ("access", "refresh")

    async def validate_access_token(self, token):  # pragma: no cover - not exercised here
        return None

    async def refresh_access_token(self, refresh_token):  # pragma: no cover - not exercised here
        return None

    async def link_external_token(
        self, user_id, access_token, refresh_token=None, expires_in=None, provider="external"
    ):  # pragma: no cover - not exercised here
        pass

    async def get_external_token(self, user_id, provider="external"):  # pragma: no cover - not exercised here
        return None

    async def update_external_token(
        self, user_id, access_token, refresh_token=None, expires_in=None, provider="external"
    ):  # pragma: no cover - not exercised here
        pass

    async def is_external_token_expired(self, user_id, provider="external"):  # pragma: no cover - not exercised here
        return False

    async def register_client(self, client_name, redirect_uris):  # pragma: no cover - not exercised here
        return {"client_id": "client123", "client_secret": "secret123"}

    async def validate_client(self, client_id, client_secret=None, redirect_uri=None):
        self.validate_calls.append((client_id, client_secret, redirect_uri))
        return client_secret != "wrong-secret"


class TestDefaultAuthenticateClient:
    """The default authenticate_client keeps pre-existing stores working."""

    @pytest.mark.asyncio
    async def test_delegates_to_validate_client(self):
        store = LegacyTokenStore()

        assert await store.authenticate_client("client-a", "s3cret", "http://localhost/cb")
        assert store.validate_calls == [("client-a", "s3cret", "http://localhost/cb")]

    @pytest.mark.asyncio
    async def test_passes_through_rejection(self):
        store = LegacyTokenStore()

        assert not await store.authenticate_client("client-a", "wrong-secret")

    @pytest.mark.asyncio
    async def test_defaults_to_no_credentials(self):
        store = LegacyTokenStore()

        assert await store.authenticate_client("client-a")
        assert store.validate_calls == [("client-a", None, None)]
