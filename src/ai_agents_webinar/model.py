"""
The model gateway.
This is the entry point for all model calls.
Every model call goes through LiteLLM Proxy. The orchestrator names "reasoning"
or "routing" -- never a provider model id -- so the routing story is a
config change in "litellm/config.yaml", not a code change here.

Budget caps live in the proxy. When the proxy refuses a call for budget,
that refusal arrives here as "BudgetExceeded" and terminates the run: the cap is
enforced by the gateway and merely *observed* by the application, which is the
distinction the webinar exists to show.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from typing import Any

import httpx2

DEFAULT_BASE_URL = "http://127.0.0.1:4000"

REASONING = "reasoning"
ROUTING = "routing"
LOCAL = "local"


class ModelError(RuntimeError):
    """The gateway could not serve the call."""


class BudgetExceeded(ModelError):
    """The proxy refused on budget. Not retryable — the cap is the point."""


class GatewayUnavailable(ModelError):
    """The proxy is unreachable. Triggers the degraded mode."""


@dataclass(frozen=True)
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    calls: int = 0

    @property
    def tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(self.prompt_tokens + other.prompt_tokens,
                     self.completion_tokens + other.completion_tokens,
                     self.cost_usd + other.cost_usd,
                     self.calls + other.calls)


@dataclass(frozen=True)
class Completion:
    text: str
    tool_calls: list[dict]
    usage: Usage
    model: str
    # Why the model stopped. "content_filter" matters here: the provider returns
    # HTTP 200 with null content, so without this an injection payload in the
    # evidence produces a silently blank answer that still costs money.
    finish_reason: str = "stop"


@dataclass(frozen=True)
class ModelProfile:
    """The model half of the naive/hardened pair.

    The policy half — step and token caps — lives in "PolicyConfig". Both halves
    are needed for the comparison to be honest: the delta must come from
    step count and context handling, not from quietly picking a cheaper model
    for the hardened run.
    """
    name: str
    planning_model: str
    tool_selection_model: str
    cache_prompt: bool
    cache_tool_results: bool
    history_turns: int | None      # None = resend everything, every turn


# One large model for every step, nothing cached, the whole transcript resent.
NAIVE = ModelProfile("naive", REASONING, REASONING, False, False, None)

# Routed, cached, trimmed. Same models are available to both — the saving comes
# from using the small one where a small one suffices, and from not resending
# what the provider already has.
HARDENED = ModelProfile("hardened", REASONING, ROUTING, True, True, 6)

# The local path (guidelines stack table; the answer to "our data cannot leave
# our infrastructure"). Everything else is unchanged — same policy, same tools,
# same guardrails, same audit — because only the *model* moves. Prompt caching
# is off: it is an Anthropic mechanism, not something Ollama serves.
#
# Context trimming matters more here, not less: a local 8B model has a smaller
# window than the hosted one, so the hardened context discipline is what makes
# the run fit at all.
LOCAL_PROFILE = ModelProfile("local", LOCAL, LOCAL, False, True, 6)


def _content(text: str, *, cache: bool) -> Any:
    """Anthropic prompt caching, expressed through the OpenAI-shaped API.

    LiteLLM passes `cache_control` through to the provider; without it the same
    system prompt is re-read at full price on every turn of the loop.
    """
    if not cache:
        return text
    return [{"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}]


def tool_schema(spec) -> dict:
    """The wire schema the model sees is generated from the same
    `ToolSpec` that validates the call and builds the permission matrix."""
    return {"type": "function",
            "function": {"name": spec.name.replace(".", "__"),
                         "description": spec.description,
                         "parameters": spec.args.model_json_schema()}}


class Gateway:
    """Thin OpenAI-shaped client. 
    We use httpx (already present via mcp) instead of an SDK — this is one POST and three fields off the response.
    """

    def __init__(self, base_url: str | None = None, api_key: str | None = None,
                 timeout: float = 60.0):
        self.base_url = (base_url or os.environ.get("LITELLM_BASE_URL")
                         or DEFAULT_BASE_URL).rstrip("/")
        # Rehearsal traffic uses its own key, so it cannot spend the live
        # session's budget. Falls back to the master key for local bring-up.
        self.api_key = (api_key or os.environ.get("LITELLM_API_KEY")
                        or os.environ.get("LITELLM_MASTER_KEY") or "")
        self.timeout = timeout

    def complete(self, model: str, messages: list[dict], *,
                 tools: list[dict] | None = None,
                 max_tokens: int = 2048) -> Completion:
        body: dict[str, Any] = {"model": model, "messages": messages,
                                "max_tokens": max_tokens}
        if tools:
            body["tools"] = tools
        try:
            r = httpx2.post(f"{self.base_url}/v1/chat/completions", json=body,
                            headers={"Authorization": f"Bearer {self.api_key}"},
                            timeout=self.timeout)
        except Exception as e:  # DNS, connection refused, timeout
            raise GatewayUnavailable(f"{type(e).__name__}: {e}") from e

        if r.status_code >= 400:
            detail = r.text[:1200]
            # Only the proxy's own budget refusal counts. A provider-side
            # billing failure "credit balance is too low" or a plain 429 is an
            # outage, and calling it BudgetExceeded would claim that the cap fired when it did not.
            if "budget" in detail.lower():
                raise BudgetExceeded(detail)
            raise ModelError(f"HTTP {r.status_code}: {detail}")
        return _parse(r.json(), r.headers.get("x-litellm-response-cost"))


def _parse(payload: dict, cost_header: str | None) -> Completion:
    message = payload["choices"][0]["message"]
    raw_usage = payload.get("usage") or {}
    try:
        cost = float(cost_header) if cost_header else 0.0
    except ValueError:
        cost = 0.0
    return Completion(
        text=message.get("content") or "",
        tool_calls=list(message.get("tool_calls") or []),
        usage=Usage(prompt_tokens=raw_usage.get("prompt_tokens", 0),
                    completion_tokens=raw_usage.get("completion_tokens", 0),
                    cost_usd=cost, calls=1),
        model=payload.get("model", ""),
        finish_reason=payload["choices"][0].get("finish_reason") or "stop",
    )


# Per-key budgets and the rehearsal/live split. These live in the
# proxy's database, which shares the Postgres volume — so `docker compose down -v`
# destroys them along with the seed data. Provisioning has to be a command, not a
# thing someone did once by hand.
KEY_BUDGETS = {
    "demo-live": 10.0,        # the live session, its own envelope
    "demo-rehearsal": 10.0,   # rehearsal traffic, so it cannot spend the live session's budget
    "scenario3-cap": 0.05,    # small on purpose: Scenario 3 trips it on stage
}


def provision_keys(base_url: str | None = None, master_key: str | None = None,
                   budget_duration: str = "1d") -> dict[str, str]:
    """Create the virtual keys, skipping any alias that already exists."""
    base = (base_url or os.environ.get("LITELLM_BASE_URL")
            or DEFAULT_BASE_URL).rstrip("/")
    key = master_key or os.environ["LITELLM_MASTER_KEY"]
    auth = {"Authorization": f"Bearer {key}"}

    created: dict[str, str] = {}
    for alias, budget in KEY_BUDGETS.items():
        r = httpx2.post(f"{base}/key/generate", headers=auth, timeout=30,
                        json={"key_alias": alias, "max_budget": budget,
                              "budget_duration": budget_duration,
                              # LOCAL belongs here or the failover cannot fire:
                              # an unreachable gateway would be met with a 403
                              # rather than a degraded run, which is worse than
                              # having no fallback at all.
                              "models": [REASONING, ROUTING, LOCAL]})
        if r.status_code >= 400:
            # The proxy enforces unique aliases, so its own refusal is the
            # idempotency signal. "/key/list" returns hashed keys, not aliases,
            # so checking first would need an extra flag and still race.
            if "already exists" in r.text:
                continue
            raise ModelError(f"could not create {alias}: {r.text[:200]}")
        created[alias] = r.json()["key"]
    return created
