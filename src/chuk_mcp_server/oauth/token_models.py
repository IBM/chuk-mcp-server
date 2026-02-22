#!/usr/bin/env python3
"""
OAuth token data models using dataclasses with orjson optimization.

Provides type-safe token data structures with validation, following the same
pattern as the types module for consistency and performance.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import orjson

# ============================================================================
# Authorization Code Data
# ============================================================================


@dataclass
class AuthorizationCodeData:
    """Authorization code data stored in token store."""

    user_id: str
    client_id: str
    redirect_uri: str
    scope: str | None = None
    code_challenge: str | None = None
    code_challenge_method: str | None = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "user_id": self.user_id,
            "client_id": self.client_id,
            "redirect_uri": self.redirect_uri,
            "scope": self.scope,
            "code_challenge": self.code_challenge,
            "code_challenge_method": self.code_challenge_method,
            "created_at": self.created_at,
        }

    def to_json_bytes(self) -> bytes:
        """🚀 Get orjson-serialized bytes for maximum performance."""
        result: bytes = orjson.dumps(self.to_dict())
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AuthorizationCodeData":
        """Create from dictionary."""
        return cls(
            user_id=data["user_id"],
            client_id=data["client_id"],
            redirect_uri=data["redirect_uri"],
            scope=data.get("scope"),
            code_challenge=data.get("code_challenge"),
            code_challenge_method=data.get("code_challenge_method"),
            created_at=data.get("created_at", datetime.now().isoformat()),
        )


# ============================================================================
# Access Token Data
# ============================================================================


@dataclass
class AccessTokenData:
    """MCP access token data."""

    user_id: str
    client_id: str
    scope: str | None = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "user_id": self.user_id,
            "client_id": self.client_id,
            "scope": self.scope,
            "created_at": self.created_at,
        }

    def to_json_bytes(self) -> bytes:
        """🚀 Get orjson-serialized bytes for maximum performance."""
        result: bytes = orjson.dumps(self.to_dict())
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AccessTokenData":
        """Create from dictionary."""
        return cls(
            user_id=data["user_id"],
            client_id=data["client_id"],
            scope=data.get("scope"),
            created_at=data.get("created_at", datetime.now().isoformat()),
        )


# ============================================================================
# Refresh Token Data
# ============================================================================


@dataclass
class RefreshTokenData:
    """MCP refresh token data."""

    user_id: str
    client_id: str
    access_token: str
    scope: str | None = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "user_id": self.user_id,
            "client_id": self.client_id,
            "scope": self.scope,
            "access_token": self.access_token,
            "created_at": self.created_at,
        }

    def to_json_bytes(self) -> bytes:
        """🚀 Get orjson-serialized bytes for maximum performance."""
        result: bytes = orjson.dumps(self.to_dict())
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RefreshTokenData":
        """Create from dictionary."""
        return cls(
            user_id=data["user_id"],
            client_id=data["client_id"],
            access_token=data["access_token"],
            scope=data.get("scope"),
            created_at=data.get("created_at", datetime.now().isoformat()),
        )


# ============================================================================
# External Token Data
# ============================================================================


@dataclass
class ExternalTokenData:
    """External OAuth provider token data (e.g., LinkedIn, GitHub, Google)."""

    access_token: str
    expires_at: str
    expires_in: int
    refresh_token: str | None = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def __post_init__(self) -> None:
        """Validate data after initialization."""
        # Validate expires_in is positive
        if self.expires_in <= 0:
            raise ValueError(f"expires_in must be positive, got {self.expires_in}")

        # Validate expires_at is valid ISO timestamp
        try:
            datetime.fromisoformat(self.expires_at)
        except ValueError as e:
            raise ValueError(f"expires_at must be valid ISO format timestamp: {e}") from e

    def is_expired(self, buffer_minutes: int = 5) -> bool:
        """
        Check if token is expired or will expire soon.

        Args:
            buffer_minutes: Refresh proactively N minutes before expiration

        Returns:
            True if expired or expiring soon
        """
        expires_at = datetime.fromisoformat(self.expires_at)
        buffer_time = timedelta(minutes=buffer_minutes)
        return datetime.now() >= (expires_at - buffer_time)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "access_token": self.access_token,
            "refresh_token": self.refresh_token,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "expires_in": self.expires_in,
        }

    def to_json_bytes(self) -> bytes:
        """🚀 Get orjson-serialized bytes for maximum performance."""
        result: bytes = orjson.dumps(self.to_dict())
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExternalTokenData":
        """Create from dictionary."""
        return cls(
            access_token=data["access_token"],
            expires_at=data["expires_at"],
            expires_in=data["expires_in"],
            refresh_token=data.get("refresh_token"),
            created_at=data.get("created_at", datetime.now().isoformat()),
        )


# ============================================================================
# Client Data
# ============================================================================


@dataclass
class ClientData:
    """Registered MCP client data."""

    client_name: str
    client_secret: str
    redirect_uris: list[str]
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def __post_init__(self) -> None:
        """Validate data after initialization."""
        # Validate client_name is not empty
        if not self.client_name or not self.client_name.strip():
            raise ValueError("client_name must not be empty")

        # Validate redirect_uris
        if not self.redirect_uris:
            raise ValueError("At least one redirect URI is required")

        for uri in self.redirect_uris:
            if not uri or not uri.strip():
                raise ValueError("Redirect URIs cannot be empty")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "client_name": self.client_name,
            "client_secret": self.client_secret,
            "redirect_uris": self.redirect_uris,
            "created_at": self.created_at,
        }

    def to_json_bytes(self) -> bytes:
        """🚀 Get orjson-serialized bytes for maximum performance."""
        result: bytes = orjson.dumps(self.to_dict())
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ClientData":
        """Create from dictionary."""
        return cls(
            client_name=data["client_name"],
            client_secret=data["client_secret"],
            redirect_uris=data["redirect_uris"],
            created_at=data.get("created_at", datetime.now().isoformat()),
        )


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    "AuthorizationCodeData",
    "AccessTokenData",
    "RefreshTokenData",
    "ExternalTokenData",
    "ClientData",
]
