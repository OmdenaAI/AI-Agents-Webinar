"""
Deterministic policy middleware.
Enforced, not advisory. Every check here runs before a tool executes and has
zero dependency on a model — given the same call and config it always returns
the same decision, which is what makes it testable exhaustively and honest to
show on stage.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from urllib.parse import urlsplit

from pydantic import ValidationError

from .tools import TOOLS, ToolSpec


class Outcome(str, Enum):
    ALLOW = "allowed"
    DENY = "denied"
    APPROVAL = "approval_required"


@dataclass(frozen=True)
class Decision:
    outcome: Outcome
    rule: str          
    reason: str
    tool: str

    @property
    def allowed(self) -> bool:
        return self.outcome is Outcome.ALLOW


@dataclass(frozen=True)
class PolicyConfig:
    allowed_tools: frozenset[str]
    project_scope: frozenset[str]        
    egress_allowlist: frozenset[str]    
    value_threshold: int | None = None  
    max_steps: int = 12
    max_tokens: int = 120_000


@dataclass(frozen=True)
class RunState:
    """Per-run counters backing the step and token caps."""
    steps: int = 0
    tokens: int = 0

    def step(self, tokens: int = 0) -> "RunState":
        return replace(self, steps=self.steps + 1, tokens=self.tokens + tokens)


def _deny(rule: str, reason: str, tool: str) -> Decision:
    return Decision(Outcome.DENY, rule, reason, tool)


def check(
    tool_name: str,
    raw_args: dict,
    config: PolicyConfig,
    run: RunState = RunState(),
    *,
    value: int | None = None,
    untrusted_input: bool = False,
) -> Decision:
    """Decide whether a tool call may proceed. Never executes anything."""

    # 1. Tool allowlist — unknown tools are denied before anything else.
    spec: ToolSpec | None = TOOLS.get(tool_name)
    if spec is None:
        return _deny("allowlist", f"unknown tool {tool_name!r}", tool_name)
    if tool_name not in config.allowed_tools:
        return _deny("allowlist", f"{tool_name} is not in this agent's allowlist", tool_name)

    # 2. Argument schema validation, against the one schema in tools.py.
    try:
        args = spec.args(**raw_args).model_dump()
    except ValidationError as e:
        first = e.errors()[0]
        loc = ".".join(str(p) for p in first["loc"]) or "<args>"
        return _deny("schema", f"invalid argument {loc}: {first['msg']}", tool_name)
    except TypeError as e:
        return _deny("schema", f"invalid arguments: {e}", tool_name)

    # 3. Per-run caps. A run that has spent its budget stops, whatever it asks for.
    if run.steps >= config.max_steps:
        return _deny("step_cap", f"run exceeded {config.max_steps} steps", tool_name)
    if run.tokens >= config.max_tokens:
        return _deny("token_cap", f"run exceeded {config.max_tokens} tokens", tool_name)

    # 4. Project / data scope — the Scenario 2a boundary.
    if spec.project_arg:
        project = args.get(spec.project_arg)
        if project not in config.project_scope:
            return _deny("scope",
                         f"project {project!r} is outside this agent's scope", tool_name)

    # 5. Egress allowlist — the Scenario 2b block.
    if spec.reach == "external":
        url = str(args.get("url", ""))
        parts = urlsplit(url)
        if parts.scheme not in ("http", "https"):
            return _deny("egress", f"scheme {parts.scheme or '<none>'!r} is not permitted",
                         tool_name)
        if parts.hostname not in config.egress_allowlist:
            return _deny("egress", f"host {parts.hostname!r} is not on the egress allowlist",
                         tool_name)

    # 6. Consequential-action rule. Any one limb qualifies.
    if spec.approval_required:
        why = "irreversible" if spec.irreversible else "externally visible"
        return Decision(Outcome.APPROVAL, "consequential",
                        f"{tool_name} is {why}", tool_name)
    if config.value_threshold is not None and value is not None and value > config.value_threshold:
        return Decision(Outcome.APPROVAL, "value_threshold",
                        f"value {value} exceeds threshold {config.value_threshold}",
                        tool_name)
    if untrusted_input:
        return Decision(Outcome.APPROVAL, "untrusted_input",
                        "call was influenced by untrusted content", tool_name)

    return Decision(Outcome.ALLOW, "ok", "passed all policy checks", tool_name)


# The configuration the demo agent actually runs under. "naive" vs "hardened"
# differ in caps and allowlist breadth, not in whether policy runs.
HARDENED = PolicyConfig(
    allowed_tools=frozenset(TOOLS) - {"fs.write_file"},
    project_scope=frozenset({"APOLLO"}),
    egress_allowlist=frozenset({"status.example.com"}),
    value_threshold=8,
    max_steps=12,
    max_tokens=120_000,
)

NAIVE = PolicyConfig(
    allowed_tools=frozenset(TOOLS),
    project_scope=frozenset({"APOLLO"}),
    egress_allowlist=frozenset({"status.example.com"}),
    value_threshold=None,
    max_steps=40,
    max_tokens=500_000,
)
