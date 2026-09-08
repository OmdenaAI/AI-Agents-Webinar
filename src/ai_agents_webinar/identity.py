"""
Agent identity.
The agent has its own credential and a service identity, never a human's and
every tool server verifies it independently rather than trusting the caller's
claim about who it is.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Protocol

import jwt

ALGORITHM = "HS256"
ISSUER = "project-ops-orchestrator"
DEFAULT_TTL_SECONDS = 3600


class IdentityError(Exception):
    """Raised when a credential is missing, malformed, expired, or out of scope."""


@dataclass(frozen=True)
class Identity:
    """A validated agent identity. This is what lands in an audit record's actor."""
    subject: str
    projects: frozenset[str]
    expires_at: int

    def permits(self, project: str | None) -> bool:
        """Scope check. A call with no project touches no project data."""
        return project is None or project in self.projects


class IdentityProvider(Protocol):
    """The contract. A real IdP implementation satisfies this same shape."""

    def mint(self) -> str: ...

    def validate(self, token: str | None) -> Identity: ...


class JWTIdentityProvider:
    """Self-signed, scoped JWT (§11 recommendation)."""

    def __init__(self, secret: str | None = None, *, subject: str = "agent:project-ops",
                 projects: frozenset[str] | set[str] = frozenset({"APOLLO"}),
                 ttl_seconds: int = DEFAULT_TTL_SECONDS):
        secret = secret or os.environ.get("AGENT_JWT_SECRET")
        if not secret:
            raise IdentityError("AGENT_JWT_SECRET is not set — no identity can be minted")
        # RFC 7518: an HMAC key shorter than the hash output weakens the
        # signature. Refuse rather than warn as this is the whole security story.
        if len(secret.encode()) < 32:
            raise IdentityError(
                f"AGENT_JWT_SECRET must be at least 32 bytes, got {len(secret.encode())}")
        self._secret = secret
        self.subject = subject
        self.projects = frozenset(projects)
        self.ttl = ttl_seconds

    def mint(self) -> str:
        """Called by the orchestrator at startup (FR-17)."""
        now = int(time.time())
        return jwt.encode(
            {"iss": ISSUER, "sub": self.subject, "projects": sorted(self.projects),
             "iat": now, "exp": now + self.ttl},
            self._secret, algorithm=ALGORITHM,
        )

    def validate(self, token: str | None) -> Identity:
        """
        Called by each tool server, on every call.
        Signature and expiry are checked by the library.
        """
        if not token:
            raise IdentityError("no credential presented")
        token = token.removeprefix("Bearer ").strip()
        try:
            claims = jwt.decode(token, self._secret, algorithms=[ALGORITHM],
                                issuer=ISSUER, options={"require": ["exp", "sub", "iss"]})
        except jwt.ExpiredSignatureError as e:
            raise IdentityError("credential expired") from e
        except jwt.InvalidTokenError as e:
            raise IdentityError(f"invalid credential: {e}") from e

        return Identity(subject=claims["sub"],
                        projects=frozenset(claims.get("projects", ())),
                        expires_at=claims["exp"])
