"""
LangGraph orchestration.

"propose" → "policy check" → "approval gate" → "execute", as an explicit state machine.
The pause is a real LangGraph "interrupt()", resumed with "Command(resume=...)",
which is what makes the approval gate a first-class visible mechanism rather
than an ad hoc blocking call: the graph's state is checkpointed at the pause, so
a decision can arrive from anywhere (Slack, a CLI, a test) minutes later.

Policy runs here, in the orchestrator the single place decisions are made never inside the tool servers.
"""

from __future__ import annotations

import os
import time
import uuid
from typing import Any, Literal, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from . import db
from .approval import ApprovalChannel, from_env
from .audit import AuditRecord, new_trace_id, record
from .events import publish
from .guardrails import scan_result
from .mcp_client import invoker
from .policy import HARDENED, Decision, Outcome, PolicyConfig, RunState, check
from .telemetry import run_span, tool_span
from .tools import TOOLS


class RunStateDict(TypedDict, total=False):
    """What flows through the graph. Every field is inspectable in a trace."""
    tool: str
    args: dict
    config: PolicyConfig
    run: RunState
    value: int | None
    untrusted_input: bool
    decision: dict          # the policy decision, as plain data for the event stream
    identity: str | None    # the validated agent identity
    requested_by: str | None
    trace_id: str           # correlates the audit row to the trace
    findings: list[dict]    # guardrail flags raised on this run
    approved: bool | None
    approver: str | None
    result: Any
    error: str | None
    events: list[dict]      # ordered record of what happened, for audit + UI


def _emit(state: RunStateDict, kind: str, **fields) -> list[dict]:
    """Append one event. the same decision reaches trace, audit and UI."""
    return [*state.get("events", []), publish({"event": kind, **fields})]


def policy_node(state: RunStateDict) -> RunStateDict:
    """Deterministic gate. Runs before anything executes, on every path."""
    decision: Decision = check(
        state["tool"],
        state.get("args", {}),
        state.get("config") or HARDENED,
        state.get("run") or RunState(),
        value=state.get("value"),
        untrusted_input=state.get("untrusted_input", False),
    )
    as_data = {
        "outcome": decision.outcome.value,
        "rule": decision.rule,
        "reason": decision.reason,
        "tool": decision.tool,
    }
    trace_id = state.get("trace_id") or new_trace_id()
    record(AuditRecord(
        actor=state.get("identity") or "agent:project-ops",
        action=decision.tool,
        arguments=state.get("args", {}),
        policy_result=decision.outcome.value,
        policy_rule=decision.rule,
        trace_id=trace_id,
    ))
    return {
        "decision": as_data,
        "trace_id": trace_id,
        "events": _emit(state, "policy_decision", trace_id=trace_id, **as_data),
    }


def route_after_policy(state: RunStateDict) -> Literal["approval", "execute", "denied"]:
    outcome = state["decision"]["outcome"]
    if outcome == Outcome.DENY.value:
        return "denied"
    if outcome == Outcome.APPROVAL.value:
        return "approval"
    return "execute"


def approval_node(state: RunStateDict) -> RunStateDict:
    """
    The gate. Execution cannot proceed past here without a human decision.

    "interrupt()" suspends the graph and persists its state; the value passed in
    is what the approver sees the exact proposed action and why it stopped.
    """
    decision = interrupt({
        "question": "Approve this action?",
        "tool": state["tool"],
        "args": state.get("args", {}),
        "rule": state["decision"]["rule"],
        "reason": state["decision"]["reason"],
        "requested_by": state.get("requested_by"),
    })

    if isinstance(decision, dict):
        approved = bool(decision.get("approved"))
        approver = decision.get("approver")
    else:  # a bare True/False, for a CLI or a test
        approved, approver = bool(decision), None

    record(AuditRecord(
        actor=state.get("identity") or "agent:project-ops",
        action=state["tool"],
        arguments=state.get("args", {}),
        policy_result="approved" if approved else "rejected",
        policy_rule="human_decision",
        approver=approver,
        trace_id=state["trace_id"],
    ))
    return {
        "approved": approved,
        "approver": approver,
        "events": _emit(
            state, "approval_decision", approved=approved, approver=approver
        ),
    }


def route_after_approval(state: RunStateDict) -> Literal["execute", "rejected"]:
    return "execute" if state.get("approved") else "rejected"


def execute_node(state: RunStateDict) -> RunStateDict:
    """Run the tool, then scan what it returned before anything else sees it."""
    spec = TOOLS[state["tool"]]
    with tool_span(spec.name, **{"projectops.trace_id": state.get("trace_id"),
                                 "projectops.policy_rule": state["decision"]["rule"]}) as span:
        try:
            result = invoker()(spec, state.get("args", {}))
        except Exception as e:
            span.set_attribute("error.type", type(e).__name__)
            return {"error": f"{type(e).__name__}: {e}",
                    "events": _emit(state, "tool_error", tool=spec.name,
                                    error=type(e).__name__)}

        scanned = scan_result(result)
        span.set_attribute("projectops.guardrail_flagged", scanned.flagged)
    events = state.get("events", [])
    for finding in scanned.findings:
        events = [*events, publish(finding.as_event())]
    events = [*events, publish({"event": "tool_result", "tool": spec.name,
                                "ok": True, "flagged": scanned.flagged})]

    return {
        # The redacted copy is what downstream consumers get, never the raw text.
        "result": scanned.redacted,
        "findings": [f.as_event() for f in scanned.findings],
        "events": events,
    }


def denied_node(state: RunStateDict) -> RunStateDict:
    return {"error": f"denied by policy: {state['decision']['reason']}",
            "events": _emit(state, "run_denied", rule=state["decision"]["rule"])}


def rejected_node(state: RunStateDict) -> RunStateDict:
    return {"error": "rejected by approver",
            "events": _emit(state, "run_rejected", approver=state.get("approver"))}


_SERDE = JsonPlusSerializer(allowed_msgpack_modules=[
    ("ai_agents_webinar.policy", "PolicyConfig"),
    ("ai_agents_webinar.policy", "RunState"),
])


_SAVER = None


def checkpointer_from_env():
    """
    Postgres when it is there, memory when it is not.
    An approval can sit pending for minutes while a human decides. With an
    in-memory checkpointer a restart in that window loses the run entirely. 
    Checkpoints go to their own "agent_state" schema so the audit-log grants stay untouched.
    """
    dsn = os.environ.get("DATABASE_URL")
    if not dsn:
        return InMemorySaver(serde=_SERDE)

    global _SAVER
    if _SAVER is None:
        from langgraph.checkpoint.postgres import PostgresSaver
        from psycopg_pool import ConnectionPool

        pool = ConnectionPool(dsn, min_size=1, max_size=8, open=True,
                              kwargs={"autocommit": True,
                                      "options": "-csearch_path=agent_state"})
        _SAVER = PostgresSaver(pool, serde=_SERDE)
        _SAVER.setup()
    return _SAVER


def build_graph(checkpointer=None):
    """
    The write-action state machine.
    A checkpointer is required for "interrupt()" — without persisted state there
    is nothing to resume into.
    """
    graph = StateGraph(RunStateDict)
    graph.add_node("policy", policy_node)
    graph.add_node("approval", approval_node)
    graph.add_node("execute", execute_node)
    graph.add_node("denied", denied_node)
    graph.add_node("rejected", rejected_node)

    graph.add_edge(START, "policy")
    graph.add_conditional_edges("policy", route_after_policy,
                            {"approval": "approval", "execute": "execute",
                             "denied": "denied"})
    graph.add_conditional_edges("approval", route_after_approval,
                            {"execute": "execute", "rejected": "rejected"})
    for terminal in ("execute", "denied", "rejected"):
        graph.add_edge(terminal, END)

    return graph.compile(checkpointer=checkpointer or checkpointer_from_env())


def propose(
    graph: Any,
    state: RunStateDict,
    channel: ApprovalChannel | None = None,
    thread_id: str | None = None,
) -> tuple[RunStateDict, str]:
    """
    Start a run if policy stopped it for approval, put the request to a human.

    The ask lives here rather than inside "approval_node" on purpose: LangGraph
    re-runs a node from the top when it resumes, so posting from inside the node
    would post the request a second time on every decision.
    """
    thread_id = thread_id or uuid.uuid4().hex
    with run_span("agent.write_action",
                  **{"projectops.tool": state.get("tool")}) as span:
        state = {**state, "trace_id": state.get("trace_id") or new_trace_id()}
        span.set_attribute("projectops.trace_id", state["trace_id"])
        out = graph.invoke(state, {"configurable": {"thread_id": thread_id}})
    if out.get("__interrupt__"):
        (channel or from_env()).request(thread_id, out["__interrupt__"][0].value)
    return out, thread_id


def await_decision(graph, thread_id: str, timeout: float = 300.0) -> RunStateDict:
    """Block until whatever channel holds the request answers it.

    The graph itself is not blocked it is checkpointed at the pause. This
    only blocks the *caller*, which is what a presenter driving a live demo
    wants: run the scenario, wait for the click, show the result.
    """
    cfg = {"configurable": {"thread_id": thread_id}}
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        snapshot = graph.get_state(cfg)
        if not snapshot.next:
            return snapshot.values
        time.sleep(0.5)
    raise TimeoutError(f"no approval decision within {timeout:.0f}s")


def pending_threads(graph) -> list[str]:
    """
    Approvals that were never decided their Slack buttons are still live.

    The thread list comes from SQL rather than "checkpointer.list()": that walks
    every checkpoint, of which each run writes several, so the scan is O(steps)
    when only the distinct threads matter. It ran for minutes before this.
    """
    with db.writable() as conn:
        rows = db.rows(conn, "SELECT DISTINCT thread_id FROM agent_state.checkpoints")
    pending = []
    for row in rows:
        state = graph.get_state({"configurable": {"thread_id": row["thread_id"]}})
        if "approval" in (state.next or ()):
            pending.append(row["thread_id"])
    return pending


def cancel_pending(graph, approver: str = "system:pre-demo-reset") -> list[str]:
    """
    Reject every undecided approval, so old buttons become no-ops.

    A timed-out run stays checkpointed on purpose that is what makes "--resume"
    possible — but it also leaves a live Approve button in Slack pointing at a
    real write. Clicking one mid-session executes an action nobody asked for.
    Rejecting is the safe resolution: a decided thread ignores later clicks.
    """
    cancelled = []
    for thread_id in pending_threads(graph):
        try:
            graph.invoke(Command(resume={"approved": False, "approver": approver}),
                         {"configurable": {"thread_id": thread_id}})
            cancelled.append(thread_id)
        except Exception:
            continue     
    return cancelled

