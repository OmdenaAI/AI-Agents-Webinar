"""
The approval channel: the orchestrator hands a pending decision to an ApprovalChannel
and never names Slack. 
This module is the only place that knows Slack exists.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import threading
from typing import Any, Callable, Protocol

from .telemetry import continue_trace, flush, session_url
from .tools import TOOLS

APPROVE = "projectops_approve"
REJECT = "projectops_reject"

# Slack configuration.
MAX_ANSWER_CHARS = 2600
_LISTENER = None
_LISTENER_STATE: dict = {"listening": False, "answers_questions": False}


def listener_state() -> dict:
    """
    Whether this process has a live Slack connection, and what it serves.
    """

    return {**_LISTENER_STATE,
            "ask_channel": os.environ.get("SLACK_ASK_CHANNEL") or "(any)",
            "approval_channel": os.environ.get("SLACK_APPROVAL_CHANNEL")}


_BOLD = re.compile(r"\*\*(.+?)\*\*", re.S)
_UNDERSCORE_BOLD = re.compile(r"__(.+?)__", re.S)
_HEADING = re.compile(r"^[ \t]{0,3}#{1,6}[ \t]+(.+?)[ \t]*$", re.M)
_LINK = re.compile(r"\[([^\]]+)\]\((https?://[^)\s]+)\)")

_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$")
_TABLE_RULE = re.compile(r"^\s*\|[\s:|-]+\|\s*$")
_EMPHASIS = re.compile(r"\*{1,2}|__")


def _table_to_monospace(rows: list[str]) -> str:
    """
    One markdown table as an aligned code block.
    """
    cells = [[_EMPHASIS.sub("", cell).strip()
              for cell in row.strip().strip("|").split("|")]
             for row in rows if not _TABLE_RULE.match(row)]
    if not cells:
        return "\n".join(rows)
    columns = max(len(row) for row in cells)
    cells = [row + [""] * (columns - len(row)) for row in cells]
    widths = [max(len(row[i]) for row in cells) for i in range(columns)]
    lines = ["  ".join(c.ljust(w) for c, w in zip(row, widths)).rstrip()
             for row in cells]
    if len(lines) > 1:
        lines.insert(1, "  ".join("-" * w for w in widths))
    return "```" + "\n".join(lines) + "```"


def _tables_to_code(text: str) -> str:
    """Replace every markdown table in "text"; leave everything else alone."""
    out: list[str] = []
    block: list[str] = []

    def flush() -> None:
        out.append(_table_to_monospace(block)) if len(block) >= 2 else out.extend(block)
        block.clear()

    for line in text.splitlines():
        if _TABLE_ROW.match(line):
            block.append(line)
        else:
            flush()
            out.append(line)
    flush()
    return "\n".join(out)


def to_slack_mrkdwn(text: str) -> str:
    """Markdown as models write it -> mrkdwn as Slack renders it."""
    if not text:
        return text
    text = _tables_to_code(text)
    text = _HEADING.sub(r"**\1**", text)
    text = _LINK.sub(r"<\2|\1>", text)
    text = _UNDERSCORE_BOLD.sub(r"**\1**", text)
    return _BOLD.sub(r"*\1*", text)


# A leading "word:" selects what to run.
DIRECTIVES = ("compare", "naive", "hardened", "local", "propose", "audit", "help")

HELP = ("*What I can do*\n"
        " `@me <question>` - answer it (hardened profile)\n"
        " `@me naive: <question>` / `local: ..` - run one profile\n"
        " `@me compare: <question>` - run naive *and* hardened, side by side\n"
        " `@me propose: <status update>` - a write action; it pauses for "
        "approval before anything happens\n"
        " `@me audit:` - what the last run did, in plain English\n"
        " `@me audit: <trace id>` - the same run as the full record: every "
        "row, the rule that decided it, and the matching trace\n"
        "\nEvery call still passes the same policy checks, whichever you pick.")


def parse_directive(text: str) -> tuple[str | None, str]:
    """Split a leading `word:` directive off the question."""
    head, sep, rest = text.partition(":")
    word = head.strip().lower()
    if sep and word in DIRECTIVES:
        return word, rest.strip()
    if word in ("help", "?"):
        return "help", ""
    return None, text.strip()


def _session_link(session: str | None) -> str:
    """
    A link straight to the two runs, side by side
    """
    if not session:
        return ""
    url = session_url(session)
    return f"<{url}|open both traces>" if url else f"session `{session}`"


def comparison_blocks(comparison: dict) -> list[dict]:
    """Scenario 3 as a Slack message.

    A code block, because Slack does not render tables and the columns only line
    up in monospace. The factorisation is shown because it is the claim that
    survives run-to-run variance - the multiple itself moves a lot.
    """
    naive, hardened = comparison["naive"], comparison["hardened"]
    rows = [f"{'profile':<10}{'steps':>6}{'calls':>6}{'tokens':>8}"
            f"{'latency':>9}{'cost USD':>10}"]
    for label, r in (("naive", naive), ("hardened", hardened)):
        rows.append(f"{label:<10}{r['steps']:>6}{r['calls']:>6}{r['tokens']:>8}"
                    f"{r['latency_s']:>8.1f}s{r['cost_usd']:>10.4f}")
    body = "```" + "\n".join(rows) + "```"

    lines = [body]
    if hardened["cost_usd"] and hardened["tokens"]:
        fewer = naive["tokens"] / hardened["tokens"]
        cheaper = ((naive["cost_usd"] / max(naive["tokens"], 1))
                   / (hardened["cost_usd"] / hardened["tokens"]))
        lines.append(
            f"*{naive['cost_usd'] / hardened['cost_usd']:.1f}x cheaper* - "
            f"{fewer:.1f}x fewer tokens (steps + context) "
            f"x {cheaper:.1f}x cheaper tokens (routing)")
    link = _session_link(comparison.get("session"))
    if link:
        lines.append(link)
    return [{"type": "section", "text": {"type": "mrkdwn", "text": "\n".join(lines)}}]


# Plain-English glosses for the audit renderers. 
RULE_IN_WORDS = {
    "allowlist": "that tool is not one this agent is allowed to use at all",
    "scope": "that project is outside what this agent is authorised to touch",
    "egress": "that address is not on the list of places this agent may reach",
    "schema": "the request did not match what the tool accepts",
    "step_cap": "the run had already spent its step budget",
    "token_cap": "the run had already spent its token budget",
    "consequential": "it changes something outside the system",
    "value_threshold": "it was above the value a human has to sign off",
    "untrusted_input": "it was shaped by text the agent did not write itself",
}


def action_in_words(action: str) -> str:
    """
    The tool's own description, reused rather than a second list to maintain.
    """
    spec = TOOLS.get(action)
    sentence = (spec.description or "").strip().split(". ")[0].rstrip(".") if spec else ""
    if sentence:
        return sentence[0].lower() + sentence[1:]
    return action.split(".")[-1].replace("_", " ")


def _row_in_words(row: dict) -> str:
    when = str(row["created_at"])[11:19]
    what = action_in_words(row["action"])
    result = row["policy_result"]
    approver = row.get("approver")
    if result == "denied":
        why = RULE_IN_WORDS.get(row["policy_rule"], row["policy_rule"])
        return f"`{when}`  tried to {what} - *refused* because {why}. Nothing ran."
    if result == "approval_required":
        why = RULE_IN_WORDS.get(row["policy_rule"], row["policy_rule"])
        return (f"`{when}`  asked to {what} - *held for a person* because {why}. "
                "It could not proceed on its own.")
    if result == "approved":
        return f"`{when}`  {what} - *approved* by <@{approver}>, then carried out."
    if result == "rejected":
        return f"`{when}`  {what} - *rejected* by <@{approver}>. Nothing was written."
    return f"`{when}`  {what} - allowed, and it ran."


def narrative_blocks(audit: dict) -> list[dict]:
    
    rows = audit["rows"]
    span = f"{str(rows[0]['created_at'])[11:19]} to {str(rows[-1]['created_at'])[11:19]}"
    lines = [f"*What the agent did* - most recent run, {span}", ""]
    lines += [f" {_row_in_words(r)}" for r in rows]

    refused = [r for r in rows if r["policy_result"] == "denied"]
    decided = [r for r in rows if r.get("approver")]
    actors = sorted({r["actor"] for r in rows})

    lines += ["", "*And what makes this a record rather than a report*"]
    lines.append(f" It acted as itself - `{', '.join(actors)}` - never under a "
                 "person's account, so nothing it did is attributed to a human.")
    if decided:
        who = ", ".join(f"<@{r['approver']}>" for r in decided)
        lines.append(f" Where a person was required, a person decided: {who}. "
                     "The agent waited; it had no way to continue without that.")
    else:
        lines.append(" Nothing on this run needed a human decision - "
                     "everything it did was already permitted in advance.")
    if refused:
        lines.append(f" {len(refused)} request"
                     f"{'s were' if len(refused) != 1 else ' was'} refused before "
                     "anything happened, by a rule rather than by the agent's "
                     "own judgement.")
    lines.append(" This log is written by the system, not by the agent, and the "
                 "database refuses to change or delete a row once it exists.")
    blocks = [{"type": "section", "text": {"type": "mrkdwn", "text": "\n".join(lines)}}]
    trace = audit.get("trace_id")
    if trace:
        # Small print on purpose: the manager does not need it, and the engineer
        # sitting next to them can now ask for the same run as a record.
        blocks.append({"type": "context", "elements": [
            {"type": "mrkdwn", "text": f"the full record: `audit: {trace}`"}]})
    return blocks


def audit_blocks(audit: dict) -> list[dict]:
    """
    Scenario 5: what did it do, who authorised it, can we prove it.
    """
    rows = audit.get("rows") or []
    if rows and audit.get("narrative"):
        return narrative_blocks(audit)
    if not rows:
        asked = audit.get("asked_for")
        lead = (f"No audit rows for trace `{asked}`." if asked
                else "No audit rows recorded yet - run a scenario first.")
        options = audit.get("recent") or []
        if options:
            listing = "\n".join(f" `{o['trace_id']}` - {o['action']} at {o['when']}"
                                for o in options)
            lead += f"\n\n*Recent runs you can reconstruct:*\n{listing}"
        return [{"type": "section", "text": {"type": "mrkdwn", "text": lead}}]

    lines = [f"{'time':<9}{'action':<28}{'result':<18}approver"]
    for r in rows:
        lines.append(f"{str(r['created_at'])[11:19]:<9}{r['action']:<28}"
                     f"{r['policy_result']:<18}{r['approver'] or '-'}")
    actors = {r["actor"] for r in rows}
    approvers = {r["approver"] for r in rows if r["approver"]}

    body = ["```" + "\n".join(lines) + "```",
            f"*What it did* - {len(rows)} recorded decision"
            f"{'s' if len(rows) != 1 else ''} on this run",
            f"*Under which identity* - {', '.join(sorted(actors))}"
            f"{' (never a human)' if actors == {'agent:project-ops'} else ''}",
            f"*Who authorised it* - "
            + (", ".join(f"<@{a}>" for a in sorted(approvers)) if approvers
               else "no human decision was needed on this run"),
            "*Can we prove it* - the audit table refuses UPDATE, DELETE and "
            "TRUNCATE at the database role level, and is queried independently "
            "of the trace store."]
    url = audit.get("trace_url")
    trace = audit.get("trace_id") or ""
    body.append(f"<{url}|the same run in the trace store>" if url
                else f"trace `{trace}`")
    return [{"type": "section", "text": {"type": "mrkdwn", "text": "\n".join(body)}}]


def decision_blocks(result: dict) -> list[dict]:
    """What came back from a write action put to a human."""
    if result["decision"] == "denied":
        text = (f"*Denied by policy* - `{result.get('rule')}`\n"
                f"{result.get('reason', '')}\nNo approval was ever requested.")
    elif result["decision"] == "approved":
        text = (f"*Approved* by <@{result.get('approver')}> - the action ran.\n"
                f"`{result.get('tool')}`")
    elif result["decision"] == "rejected":
        text = (f"*Rejected* by <@{result.get('approver')}> - nothing was written.\n"
                f"`{result.get('tool')}`")
    else:
        text = f"*{result['decision']}* - `{result.get('tool')}`"
    trace = result.get("trace_id") or ""
    blocks = [{"type": "section", "text": {"type": "mrkdwn", "text": text}}]
    if trace:
        blocks.append({"type": "context", "elements": [
            {"type": "mrkdwn", "text": f"trace `{trace}`"}]})
    return blocks


def answer_blocks(question: str, result: dict) -> list[dict]:
    """The agent's reply, as Block Kit. Pure - testable without Slack."""
    if result.get("comparison"):
        return comparison_blocks(result["comparison"])
    if result.get("audit") is not None:
        return audit_blocks(result["audit"])
    if result.get("decision") is not None:
        return decision_blocks(result)
    if result.get("help"):
        return [{"type": "section", "text": {"type": "mrkdwn", "text": HELP}}]
    answer = to_slack_mrkdwn(
        result.get("answer") or "(no answer produced)")[:MAX_ANSWER_CHARS]
    tokens = result.get("tokens")
    footer = " · ".join(part for part in (
        f"{result.get('steps', '?')} steps",
        f"{tokens:,} tokens" if tokens is not None else None,
        f"${result.get('cost_usd', 0):.4f}",
        f"trace `{result.get('trace_id') or ''}`",
    ) if part)
    return [
        {"type": "section", "text": {"type": "mrkdwn", "text": answer}},
        {"type": "context", "elements": [{"type": "mrkdwn", "text": footer}]},
    ]


class ApprovalChannel(Protocol):
    """
    Create a pending approval. The decision comes back separately,
    as "Command(resume=...)" into the paused graph -- so a channel only has to
    know how to *ask*, never how to block.
    """

    def request(self, thread_id: str, payload: dict) -> None: ...


def _summary(payload: dict) -> str:
    return f"Approval needed: {payload.get('tool', '?')}"


def blocks(thread_id: str, payload: dict) -> list[dict]:
    args = json.dumps(payload.get("args", {}), indent=2, default=str)
    return [
        {"type": "header",
         "text": {"type": "plain_text", "text": "Action awaiting approval"}},
        {"type": "section", "fields": [
            {"type": "mrkdwn", "text": f"*Tool*\n`{payload.get('tool', '?')}`"},
            {"type": "mrkdwn", "text": f"*Stopped by*\n`{payload.get('rule', '?')}`"},
        ] + ([{"type": "mrkdwn",
               "text": f"*Requested by*\n<@{payload['requested_by']}>"}]
             if payload.get("requested_by") else [])},
        {"type": "section",
         "text": {"type": "mrkdwn", "text": f"*Why*\n{payload.get('reason', '')}"}},
        {"type": "section",
         "text": {"type": "mrkdwn", "text": f"*Arguments*\n```{args}```"}},
        {"type": "actions", "elements": [
            {"type": "button", "action_id": APPROVE, "style": "primary",
             "text": {"type": "plain_text", "text": "Approve"},
             "value": thread_id},
            {"type": "button", "action_id": REJECT, "style": "danger",
             "text": {"type": "plain_text", "text": "Reject"},
             "value": thread_id},
        ]},
    ]


def settled_blocks(
    detail: dict,
    *,
    approved: bool,
    approver: str,
) -> list[dict]:
    verdict = "Approved" if approved else "Rejected"
    return [
        {"type": "section",
         "text": {"type": "mrkdwn", "text": f"*{verdict}* by <@{approver}>"}},
        detail,
    ]


class ConsoleApprovalChannel:

    def __init__(self, stream: Any = None) -> None:
        self._stream = stream if stream is not None else sys.stdout

    def request(self, thread_id: str, payload: dict) -> None:
        print(
            f"\n=== APPROVAL REQUIRED ({thread_id}) ===\n"
            f"tool:   {payload.get('tool')}\n"
            f"args:   {json.dumps(payload.get('args', {}), default=str)}\n"
            f"rule:   {payload.get('rule')}\n"
            f"reason: {payload.get('reason')}\n"
            f"resume: Command(resume={{'approved': True, 'approver': '<you>'}})"
            f" with thread_id={thread_id!r}\n",
            file=self._stream,
        )


class SlackApprovalChannel:
    """Posts the request with real Approve/Reject controls."""

    def __init__(self, client: Any, channel: str) -> None:
        self._client = client
        self._channel = channel

    def request(self, thread_id: str, payload: dict) -> None:
        self._client.chat_postMessage(
            channel=self._channel,
            text=_summary(payload),  # notification fallback; blocks carry the detail
            blocks=blocks(thread_id, payload),
        )


def from_env(stream: Any = None) -> ApprovalChannel:
    """
    Slack when configured, console otherwise.
    """
    token = os.getenv("SLACK_BOT_TOKEN")
    channel = os.getenv("SLACK_APPROVAL_CHANNEL")
    if token and channel:
        from slack_sdk import WebClient

        return SlackApprovalChannel(WebClient(token=token), channel)
    return ConsoleApprovalChannel(stream)


def resume(graph: Any, thread_id: str, *, approved: bool, approver: str) -> Any:
    """Deliver a human decision into the paused graph."""
    from langgraph.types import Command

    config = {"configurable": {"thread_id": thread_id}}
    trace_id = (graph.get_state(config).values or {}).get("trace_id")
    with continue_trace(trace_id or ""):
        final = graph.invoke(
            Command(resume={"approved": approved, "approver": approver}), config)
    flush()          # the trace is opened moments later - see planner.ask
    return final


def _guarded(fn: Callable[[], None]) -> Callable[[], None]:
    """
    A decision that fails must say so.
    """
    def run() -> None:
        try:
            fn()
        except Exception:
            logging.getLogger(__name__).exception("approval decision failed")
    return run


def start_listener(
    graph: Any,
    on_question: Any = None,
    bot_token: str | None = None,
    app_token: str | None = None,
) -> Any:
    """
    Resume the paused graph when someone clicks a button.
    """
    global _LISTENER
    if _LISTENER is not None:
        return _LISTENER

    from slack_bolt import App
    from slack_bolt.adapter.socket_mode import SocketModeHandler

    app = App(token=bot_token or os.environ["SLACK_BOT_TOKEN"])
    ask_channel = os.environ.get("SLACK_ASK_CHANNEL")

    @app.event("app_mention")
    def _on_mention(event, client) -> None:  # pragma: no cover - needs Slack
        """
        Slack as the human interface: ask in a channel, answer in-thread.
        """
        if on_question is None:
            return
        if ask_channel and event.get("channel") != ask_channel:
            return
        raw = re.sub(r"<@[^>]+>", "", event.get("text") or "").strip()
        directive, question = parse_directive(raw)
        if not question and directive != "help":
            return
        channel = event["channel"]
        thread_ts = event.get("thread_ts") or event["ts"]
        client.chat_postMessage(channel=channel, thread_ts=thread_ts,
                                text="working on it..")

        def answer() -> None:
            result = on_question(question, event.get("user"), directive)
            client.chat_postMessage(channel=channel, thread_ts=thread_ts,
                                    text=(result.get("answer") or "")[:200],
                                    blocks=answer_blocks(question, result))

        threading.Thread(target=_guarded(answer), daemon=True).start()

    @app.action(re.compile(f"{APPROVE}|{REJECT}"))
    def _decide(ack, body, action, client) -> None:  # pragma: no cover - needs Slack
        ack()
        approved = action["action_id"] == APPROVE
        thread_id = action["value"]
        approver = body["user"]["id"]
        detail = body["message"]["blocks"][1]
        channel, ts = body["channel"]["id"], body["message"]["ts"]

        def settle() -> None:
            resume(graph, thread_id, approved=approved, approver=approver)
            client.chat_update(
                channel=channel, ts=ts,
                text=f"{'Approved' if approved else 'Rejected'} by {approver}",
                blocks=settled_blocks(detail, approved=approved, approver=approver),
            )
        threading.Thread(target=_guarded(settle), daemon=True).start()

    handler = SocketModeHandler(app, app_token or os.environ["SLACK_APP_TOKEN"])
    threading.Thread(target=handler.start, daemon=True).start()
    _LISTENER = handler
    _LISTENER_STATE.update(listening=True, answers_questions=on_question is not None)
    return handler
