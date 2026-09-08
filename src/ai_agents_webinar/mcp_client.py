"""
Calling tools over Streamable HTTP.
The orchestrator does not import tool handlers and call them it crosses the
network to a separate process, presenting a scoped credential that the tool
server validates for itself.
"""

from __future__ import annotations

import os
import time
from typing import Any, Callable

from .events import publish
from .identity import JWTIdentityProvider

from .servers import DEFAULT_PORTS
from .tools import ToolSpec

_provider: JWTIdentityProvider | None = None


def server_url(server: str) -> str:
    explicit = os.environ.get(f"MCP_URL_{server.upper()}")
    if explicit:
        return explicit
    host = os.environ.get("MCP_CLIENT_HOST", "127.0.0.1")
    return f"http://{host}:{DEFAULT_PORTS[server]}/mcp"


def auth_headers(server: str) -> dict[str, str]:
    """Each server gets the credential it actually verifies.
    Ours validate an agent JWT we mint."""
    return {"Authorization": f"Bearer {_token()}"}

def _token() -> str:
    """The orchestrator holds its own credential, not a human's."""
    global _provider
    if _provider is None:
        _provider = JWTIdentityProvider()
    return _provider.mint()


def _unwrap(result: Any) -> Any:
    if result.is_error:
        detail = result.content[0].text if result.content else "unknown tool error"
        raise RuntimeError(detail)
    if result.structured_content is not None:
        return result.structured_content
    return "\n".join(getattr(c, "text", "") for c in (result.content or []))


async def call_async(spec: ToolSpec, args: dict) -> Any:
    import httpx2
    from mcp import Client
    from mcp.client.streamable_http import streamable_http_client
    from mcp.shared.exceptions import MCPError

    async with httpx2.AsyncClient(

    ) as http:
        transport = streamable_http_client(server_url(spec.server), http_client=http)
        async with Client(transport) as client:
            try:
                return _unwrap(await client.call_tool(spec.wire_name, args))
            except MCPError as e:
                raise RuntimeError(e.message) from e


TRANSPORT_RETRIES = 2

TRANSIENT_TYPES = ("ConnectError", "ConnectTimeout", "ReadTimeout", "ReadError",
                   "WriteError", "RemoteProtocolError", "PoolTimeout")
TRANSIENT_TEXT = ("SSE stream", "All connection attempts failed",
                  "Server disconnected", "peer closed connection")


def transient(error: BaseException) -> bool:
    """
    Whether a failure is worth trying again.
    Deliberately an allowlist. A tool that refused the call, or a credential the
    server rejected, must fail on the first attempt.
    """
    return (type(error).__name__ in TRANSIENT_TYPES
            or any(text in str(error) for text in TRANSIENT_TEXT))


def with_retry(fn: Callable[[], Any], *, retries: int = TRANSPORT_RETRIES, tool: str | None = None) -> Any:
    
    for attempt in range(1 + retries):
        try:
            return fn()
        except BaseException as e:  # noqa: BLE001 - re-raised below
            cause = _only(e) if isinstance(e, BaseExceptionGroup) else e
            if attempt == retries or not transient(cause):
                raise cause from None
            publish({"event": "tool_retry", "tool": tool, "attempt": attempt + 1,
                     "of": retries, "error": f"{type(cause).__name__}: {cause}"[:120]})
            time.sleep(1)


def over_mcp(spec: ToolSpec, args: dict) -> Any:
    """Sync entry point, because the graph nodes are sync."""
    return with_retry(lambda: _call_once(spec, args), tool=spec.name,
                      retries=0 if spec.write else TRANSPORT_RETRIES)


def _call_once(spec: ToolSpec, args: dict) -> Any:
    import anyio

    try:
        return anyio.run(call_async, spec, args)
    except BaseExceptionGroup as group:
        raise _only(group) from None


def _only(group: BaseException) -> BaseException:
    while isinstance(group, BaseExceptionGroup) and len(group.exceptions) == 1:
        group = group.exceptions[0]
    return group


async def _ping_async(server: str, timeout: float) -> int:
    import httpx2
    from mcp import Client
    from mcp.client.streamable_http import streamable_http_client

    async with httpx2.AsyncClient(headers=auth_headers(server),
                                  timeout=timeout) as http:
        async with Client(streamable_http_client(server_url(server),
                                                 http_client=http)) as client:
            return len((await client.list_tools()).tools)


def ping(server: str, timeout: float = 10.0) -> int:
    """Tools a server is offering. A real MCP handshake, not an open port."""
    import anyio

    return anyio.run(_ping_async, server, timeout)


def in_process(spec: ToolSpec, args: dict) -> Any:
    """
    Stand-in for tests and for running with no containers up.
    """
    return spec.handler(**args)


def invoker() -> Callable[[ToolSpec, dict], Any]:
    return in_process if os.environ.get("TOOL_TRANSPORT") == "inprocess" else over_mcp
