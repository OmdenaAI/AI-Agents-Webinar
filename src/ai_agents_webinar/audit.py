"""
Append-only audit trail.
Append-only is enforced by Postgres. The application role holds INSERT
and SELECT on this table and nothing else, so "UPDATE"/"DELETE" fail at the
database regardless of what any code above tries to do.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime

import psycopg

from . import db


def new_trace_id() -> str:
    """
    Correlation id for one run.
    Uses the active OpenTelemetry span when there is one, so an audit row and a
    trace can be joined; falls back to a random id so audit never depends on the
    observability stack being up.
    """
    try:
        from opentelemetry import trace

        span = trace.get_current_span()
        ctx = span.get_span_context()
        if ctx.is_valid:
            return format(ctx.trace_id, "032x")
    except Exception:
        pass
    return uuid.uuid4().hex


@dataclass(frozen=True)
class AuditRecord:
    actor: str
    action: str
    arguments: dict
    policy_result: str
    policy_rule: str
    trace_id: str
    approver: str | None = None
    created_at: datetime | None = None


def record(entry: AuditRecord) -> int:
    """Write one audit row. Returns its id."""
    with db.writable() as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO audit_log"
            " (actor, action, arguments, policy_result, policy_rule, approver, trace_id)"
            " VALUES (%s,%s,%s,%s,%s,%s,%s) RETURNING id",
            (entry.actor, entry.action, json.dumps(entry.arguments, default=str),
             entry.policy_result, entry.policy_rule, entry.approver, entry.trace_id),
        )
        return cur.fetchone()["id"]


def by_trace(trace_id: str) -> list[dict]:
    """Reconstruct one run — the audit query."""
    with db.readonly() as conn:
        return db.rows(
            conn,
            "SELECT * FROM audit_log WHERE trace_id=%s ORDER BY id",
            trace_id,
        )


def by_trace_prefix(prefix: str) -> list[dict]:
    """
    Reconstruct a run from the start of its trace id.
    """
    with db.readonly() as conn:
        return db.rows(
            conn,
            "SELECT * FROM audit_log WHERE trace_id LIKE %s ORDER BY id",
            prefix + "%",
        )


def recent(limit: int = 20) -> list[dict]:
    with db.readonly() as conn:
        return db.rows(conn, "SELECT * FROM audit_log ORDER BY id DESC LIMIT %s", limit)


TAMPER_ATTEMPTS = (
    "UPDATE audit_log SET actor='tampered' WHERE id=1",
    "DELETE FROM audit_log WHERE id=1",
    "TRUNCATE audit_log",
)

SUPERUSER_CAVEAT = (
    """Note: a Postgres superuser bypasses every grant, so "psql -U postgres"\n"
    proves nothing here — it succeeds, which reads as the opposite of the\n"
    claim."""
)


def append_only_proof() -> list[dict]:
    """
    Attempt to tamper with the audit table as each role the agent holds.
    """
    results: list[dict] = []
    for role, connect in (("agent_app", db.writable),
                          ("agent_replica", db.readonly)):
        for statement in TAMPER_ATTEMPTS:
            try:
                with connect() as conn, conn.cursor() as cur:
                    cur.execute(statement)
                    conn.rollback()
                results.append({"role": role, "statement": statement,
                                "refused": False, "detail": "SUCCEEDED"})
            except psycopg.errors.InsufficientPrivilege as e:
                results.append({"role": role, "statement": statement,
                                "refused": True,
                                "detail": str(e).splitlines()[0]})
    return results


def append_only_report(results: list[dict]) -> str:
    lines = []
    for role in dict.fromkeys(r["role"] for r in results):
        lines.append(f"as {role}"
                     + (" — the role the agent connects with:" if role == "agent_app"
                        else ":"))
        for r in (x for x in results if x["role"] == role):
            verb = "refused  " if r["refused"] else "SUCCEEDED"
            lines.append(f"  {r['statement'][:38]:<40}{verb}  {r['detail']}")
    return "\n".join(lines) + "\n\n" + SUPERUSER_CAVEAT
