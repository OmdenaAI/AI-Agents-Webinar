"""
The four MCP tool servers over Streamable HTTP.
One module serves any of the four; which one is chosen at startup.
Policy is deliberately NOT enforced here as it runs in the orchestrator so every decision is made in one place. 
These servers expose tools and the orchestrator decides whether they may be called.
"""

from __future__ import annotations

import argparse
import os

from mcp.server import MCPServer
from mcp.server.extension import Extension
from mcp.shared.exceptions import MCPError
from mcp_types import INVALID_REQUEST

from .identity import IdentityError, IdentityProvider, JWTIdentityProvider
from .tools import SERVERS, TOOLS

DEFAULT_PORTS = {"sprint": 8101, "warehouse": 8102, "http": 8103, "fs": 8104}


class IdentityGate(Extension):
    """
    Validates the caller's credential before every tool call.
    The tool server does this itself. It does not trust the orchestrator's claim about who it is,
    that is the difference between an identity that is enforced and one that is merely asserted.
    """

    identifier = "projectops.demo/identity-gate"

    def __init__(self, provider: IdentityProvider):
        super().__init__()
        self._provider = provider

    async def intercept_tool_call(self, params, ctx, call_next):
        request = getattr(ctx, "request", None)
        headers = getattr(request, "headers", None) or {}
        token = headers.get("authorization") or headers.get("Authorization")
        try:
            identity = self._provider.validate(token)
        except IdentityError as e:
            raise MCPError(INVALID_REQUEST, f"identity rejected: {e}") from e

        project = (params.arguments or {}).get("project_key")
        if not identity.permits(project):
            raise MCPError(
                INVALID_REQUEST,
                f"identity rejected: {identity.subject} is not scoped to {project!r}")

        return await call_next(ctx)


def build(server_name: str, provider: IdentityProvider | None = None) -> MCPServer:
    if server_name not in SERVERS:
        raise SystemExit(f"unknown server {server_name!r}; expected one of {SERVERS}")

    provider = provider or JWTIdentityProvider()
    server = MCPServer(name=f"{server_name}-tools", extensions=[IdentityGate(provider)])
    for spec in TOOLS.values():
        if spec.server == server_name:
            server.add_tool(spec.handler, name=spec.name, description=spec.description)
    return server


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("server", choices=SERVERS)
    args = parser.parse_args(argv)

    host = os.environ.get("MCP_HOST", "127.0.0.1")
    port = int(os.environ.get("MCP_PORT", DEFAULT_PORTS[args.server]))
    build(args.server).run(transport="streamable-http", host=host, port=port)


if __name__ == "__main__":
    main()
