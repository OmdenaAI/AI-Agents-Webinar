"""Project Operations Agent"""


from __future__ import annotations

import argparse
import hashlib
import os
import sys
import threading
import time
import uuid
from pathlib import Path
from dotenv import load_dotenv


def build_id() -> str:
    """A short fingerprint of the source this process is actually running"""

    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.read_bytes())
    return digest.hexdigest()[:12]


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    load_dotenv()

    parser = argparse.ArgumentParser(prog="ai-agents-webinar")
    parser.add_argument("--seed", action="store_true",
                        help="rebuild the demo database from seed.sql")
    parser.add_argument("--scenario-4", action="store_true",
                        help="propose a write action and wait for a human decision")
    parser.add_argument("--ask", metavar="QUESTION",
                        help="run the read-only planning loop (Scenarios 1-2)")
    parser.add_argument("--profile", choices=("naive", "hardened", "local"),
                        default="hardened",
                        help="config pair to run under. `local` routes every "
                             "model call to Ollama - nothing leaves the host")
    parser.add_argument("--tools", choices=("local", "umaku"), default="local",
                        help="kanban tool layer: our MCP servers, or Umaku's "
                             "(guidelines stack table). Falls back to local if "
                             "Umaku is unreachable.")
    parser.add_argument("--compare", metavar="QUESTION",
                        help="run the same question under both profiles (Scenario 3)")
    parser.add_argument("--ui", action="store_true",
                            help="serve the audience display alongside the run")
    parser.add_argument("--provision-keys", action="store_true",
                        help="create the LiteLLM virtual keys after a "
                             "volume wipe - they live in the proxy's database")
    parser.add_argument("--eval", action="store_true",
                        help="run the offline policy regression gate")
    parser.add_argument("--resume", metavar="THREAD_ID",
                        help="wait again on an approval left pending by a timeout")
    parser.add_argument("--cancel-pending", action="store_true",
                        help="reject every undecided approval, so stale Slack "
                             "buttons cannot fire a real write mid-session")
    parser.add_argument("--prove-append-only", action="store_true",
                        help="try to tamper with the audit table as each role the "
                             "agent holds, and print the refusals")
    parser.add_argument("--build-id", action="store_true",
                        help="print the fingerprint of the source this process runs")
    args = parser.parse_args()

    from . import db
    from .tools import matrix_markdown

    if args.build_id:
        print(build_id())
        return

    if args.prove_append_only:
        from .audit import append_only_proof, append_only_report

        results = append_only_proof()
        print(append_only_report(results))
        raise SystemExit(0 if all(r["refused"] for r in results) else 1)

    if args.cancel_pending:
        from .orchestrator import build_graph, cancel_pending

        cancelled = cancel_pending(build_graph())
        print(f"cancelled {len(cancelled)} pending approval(s)")
        for thread_id in cancelled:
            print(f"  {thread_id}")
        return

    if args.seed:
        print(f"seeded {db.build()}")
        return

    if args.provision_keys:
        from .model import KEY_BUDGETS, provision_keys

        created = provision_keys()
        if not created:
            print(f"all {len(KEY_BUDGETS)} keys already exist - nothing to do")
        for alias, key in created.items():
            print(f"# {alias} (${KEY_BUDGETS[alias]}/day)\n"
                  f"{'LITELLM_API_KEY' if alias == 'demo-rehearsal' else
                     'LITELLM_KEY_LIVE' if alias == 'demo-live'
                     else 'LITELLM_KEY_SCENARIO3'}={key}")
        return

    if args.eval:
        from .evals import report, run_all

        results = run_all()
        print(report(results))
        raise SystemExit(0 if all(r.passed for r in results) else 1)

    print(f"build {build_id()}")
    _start_slack(args.tools)
    server = _serve_ui() if args.ui else None

    ran_a_scenario = bool(args.scenario_4 or args.ask or args.compare or args.resume)
    try:
        _run(args, db, matrix_markdown)
    finally:
        if server is not None:
            print("scenario finished - servers still up, Ctrl-C to stop"
                  if ran_a_scenario else "ready - waiting for a scenario")
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                pass


def _run(args, db, matrix_markdown) -> None:
    if args.resume:
        _resume(args.resume)
        return

    if args.scenario_4:
        _scenario_4()
        return

    if args.ask:
        _print_run(_ask(args.ask, args.profile, args.tools))
        return

    if args.compare:
        _compare(args.compare, args.tools)
        return

    if args.ui:
        return  
    print(matrix_markdown())


def _scenario_4() -> None:
    """
    Human in the loop gate simulation.
    """
    from .approval import from_env, start_listener
    from .orchestrator import await_decision, build_graph, propose
    from .policy import HARDENED, RunState

    graph = build_graph()
    channel = from_env()
    if os.getenv("SLACK_APP_TOKEN") and os.getenv("SLACK_APPROVAL_CHANNEL"):
        start_listener(graph)
        print(f"approvals via Slack channel {os.environ['SLACK_APPROVAL_CHANNEL']}")
    else:
        print("approvals via the console fallback (FR-33) - no Slack configured")

    out, thread_id = propose(graph, {
        "tool": "sprint.post_status_update",
        "args": {"project_key": "APOLLO",
                 "body": "Two items are at risk of missing the sprint."},
        "config": HARDENED,
        "run": RunState(),
        "identity": "agent:project-ops",
    }, channel=channel)

    if not out.get("__interrupt__"):
        print(f"no approval needed: {out.get('decision')}")
        return

    print(f"awaiting decision (thread {thread_id}, trace {out['trace_id']})")
    try:
        final = await_decision(graph, thread_id)
    except TimeoutError as e:
        # The run is checkpointed, so nobody has to start over.
        print(f"{e}. Still pending: approve in Slack and resume with "
              f"--resume {thread_id}")
        raise SystemExit(2)
    verdict = "approved" if final.get("approved") else "rejected"
    print(f"{verdict} by {final.get('approver')}; result={final.get('result')}")


def _profiles(name: str, tools: str = "local"):
    from .model import HARDENED as HARDENED_MODEL
    from .model import LOCAL_PROFILE
    from .model import NAIVE as NAIVE_MODEL
    from .policy import HARDENED, NAIVE

    config = NAIVE if name == "naive" else HARDENED
    model_profile = {"naive": NAIVE_MODEL, "local": LOCAL_PROFILE}.get(
        name, HARDENED_MODEL)
    return config, model_profile


def _ask(question: str, profile_name: str, tools: str = "local",
         session_id: str | None = None):

    from .planner import ask

    config, profile = _profiles(profile_name, tools)
    started = time.monotonic()
    out = ask(question, config=config, profile=profile, session_id=session_id,
              tags=(f"tools:{tools}",))
    out["latency_s"] = time.monotonic() - started
    return out


def _print_run(out) -> None:
    usage = out.get("usage")
    for event in out.get("events", []):
        print(f"  · {event}")
    print(f"\nanswer: {out.get('answer') or out.get('error')}")
    if usage:
        print(f"steps={out['run'].steps} calls={usage.calls} "
              f"tokens={usage.tokens} cost=${usage.cost_usd:.4f} "
              f"trace={out.get('trace_id')}")


def _compare(question: str, tools: str = "local") -> None:
    """
    Compare two profiles. The numbers here populate the webinar's cost table, so they
    are measured from the same question against the same gateway - never
    estimated, and never taken from a price list.
    """

    # One session for both runs, so the trace store can put them side by side.
    session_id = f"scenario3-{uuid.uuid4().hex[:8]}"
    print(f"session: {session_id}")
    rows = []
    for name in ("naive", "hardened"):
        out = _ask(question, name, tools, session_id)
        usage, run = out.get("usage"), out.get("run")
        rows.append((name, run.steps, usage.calls, usage.tokens,
                     out.get("latency_s", 0.0), usage.cost_usd, out.get("trace_id")))
        print(f"[{name}] {out.get('answer') or out.get('error')}\n")

    print(f"{'profile':<10}{'steps':>7}{'calls':>7}{'tokens':>9}"
          f"{'latency s':>11}{'cost USD':>11}  trace")
    for name, steps, calls, tokens, latency, cost, trace in rows:
        print(f"{name:<10}{steps:>7}{calls:>7}{tokens:>9}"
              f"{latency:>11.1f}{cost:>11.4f}  {trace}")
    (_, _, _, n_tok, _, n_cost, _), (_, _, _, h_tok, _, h_cost, _) = rows
    if not (n_cost and h_cost and h_tok):
        return
    # cost = tokens x price-per-token.
    fewer = n_tok / h_tok
    cheaper = (n_cost / n_tok) / (h_cost / h_tok)
    print(f"\nhardened is {n_cost / h_cost:.1f}x cheaper on this question:")
    print(f"  {fewer:>4.1f}x  fewer tokens    (step count + context handling)")
    print(f"  {cheaper:>4.1f}x  cheaper tokens  (model routing)")


def _serve_ui():
    """The display shares a process with the run, because the event bus does."""

    from .ui import serve

    server = serve()
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"audience UI on http://localhost:{server.server_address[1]}")
    return server


def _resume(thread_id: str) -> None:
    """Pick a pending approval back up. Possible only because the pause is
    checkpointed in Postgres rather than held in a process (FR-5)."""

    from .approval import start_listener
    from .orchestrator import await_decision, build_graph

    graph = build_graph()
    if os.getenv("SLACK_APP_TOKEN"):
        start_listener(graph)
    print(f"waiting on thread {thread_id}")
    final = await_decision(graph, thread_id)
    verdict = "approved" if final.get("approved") else "rejected"
    print(f"{verdict} by {final.get('approver')}; result={final.get('result')}")


def _start_slack(tools: str = "local"):
    """One always-on Slack listener: questions in, approvals out.

    Started at boot rather than per scenario, so there is exactly one Socket Mode
    connection. Approvals resume through the shared Postgres checkpointer, so the
    graph this listener holds does not have to be the one that paused.
    """

    if not (os.getenv("SLACK_BOT_TOKEN") and os.getenv("SLACK_APP_TOKEN")):
        return None

    from .approval import start_listener
    from .orchestrator import build_graph

    def _summary(out: dict) -> dict:
        usage = out["usage"]
        return {"steps": out["run"].steps, "calls": usage.calls,
                "tokens": usage.tokens, "cost_usd": usage.cost_usd,
                "latency_s": out.get("latency_s", 0.0),
                "trace_id": out.get("trace_id")}

    def on_question(question: str, requester: str | None,
                    directive: str | None = None) -> dict:
        if directive == "help":
            return {"help": True}

        if directive == "compare":

            # One session so the trace store can show both runs together -
            # Scenario 3 is the comparison, not either run on its own.
            session = f"scenario3-{uuid.uuid4().hex[:8]}"
            naive = _ask(question, "naive", tools, session)
            hardened = _ask(question, "hardened", tools, session)
            return {"comparison": {"session": session,
                                   "naive": _summary(naive),
                                   "hardened": _summary(hardened)},
                    "requester": requester}

        if directive == "audit":
            return {"audit": _audit_reconstruction(question)}

        if directive == "propose":
            return _propose_from_slack(question, requester)

        out = _ask(question, directive or "hardened", tools)
        return {"answer": out.get("answer"), "trace_id": out.get("trace_id"),
                "cost_usd": out["usage"].cost_usd, "tokens": out["usage"].tokens,
                "steps": out["run"].steps, "requester": requester}

    handler = start_listener(build_graph(), on_question=on_question)
    channel = os.getenv("SLACK_ASK_CHANNEL")
    print(f"slack listener up - ask by mentioning the bot"
          f"{f' in {channel}' if channel else ' in any channel it is in'}")
    return handler


def _propose_from_slack(body: str, requester: str | None) -> dict:
    """
    Human in the loop gate

    The request is asked in one channel, the approval lands in another, and the
    outcome comes back to the requester. Three identities, visibly distinct: the
    person who asked, the agent that acts and the person who decides.
    """
    from .approval import from_env
    from .orchestrator import await_decision, build_graph, propose
    from .policy import HARDENED, RunState

    graph = build_graph()
    out, thread_id = propose(graph, {
        "tool": "sprint.post_status_update",
        "args": {"project_key": "APOLLO", "body": body},
        "config": HARDENED, "run": RunState(),
        "identity": "agent:project-ops", "requested_by": requester,
    }, channel=from_env())

    if not out.get("__interrupt__"):
        decision = out.get("decision") or {}
        return {"decision": decision.get("outcome", "unknown"),
                "rule": decision.get("rule"), "reason": decision.get("reason"),
                "tool": decision.get("tool"), "trace_id": out.get("trace_id")}

    try:
        final = await_decision(graph, thread_id)
    except TimeoutError:
        return {"decision": "still pending",
                "tool": "sprint.post_status_update",
                "trace_id": out.get("trace_id")}
    return {"decision": "approved" if final.get("approved") else "rejected",
            "approver": final.get("approver"),
            "tool": "sprint.post_status_update",
            "trace_id": final.get("trace_id")}


_TRACE_ID = __import__("re").compile(r"^[0-9a-f]{8,32}$", __import__("re").I)


def _audit_reconstruction(argument: str = "") -> dict:
    """Audit reconstruction and Accountability.

    Someone who has a trace id is reading a system; someone asking in words is
    asking a question about their team. Both get the same rows out of the same
    append-only table - the difference is only in how they are rendered, which
    is the honest split: nothing is summarised away for the manager that the
    engineer would have been shown.

    A trace id is hex, so "what did it do on the last run?" falls through to the
    most recent run and reads as narrative; "audit: 0055f530716c" reads as the
    full record.
    """
    from .audit import by_trace, by_trace_prefix, recent
    from .telemetry import trace_url

    argument = (argument or "").strip()
    asked_for = argument if _TRACE_ID.match(argument) else ""

    trace_id = asked_for
    if not trace_id:
        latest = recent(1)
        if not latest:
            return {"rows": [], "recent": []}
        trace_id = latest[0]["trace_id"]

    rows = by_trace(trace_id) or (
        by_trace_prefix(trace_id) if asked_for else [])
    if rows:
        trace_id = rows[0]["trace_id"]      # resolve a prefix to the real id
        return {"rows": rows, "trace_id": trace_id, "trace_url": trace_url(trace_id),
                "narrative": not asked_for}

    seen, options = set(), []
    for row in recent(40):
        if row["trace_id"] not in seen:
            seen.add(row["trace_id"])
            options.append({"trace_id": row["trace_id"], "action": row["action"],
                            "when": str(row["created_at"])[11:19]})
    return {"rows": [], "asked_for": asked_for, "recent": options[:5]}
