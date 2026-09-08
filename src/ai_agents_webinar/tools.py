"""
Tool definitions, the single source of truth.
Every tool's argument schema is declared exactly once, here. The policy
middleware validates against these models and the permission matrix is
generated from these specs. Nothing downstream re-declares a schema, and
the matrix is never hand-maintained.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Literal

from pydantic import BaseModel, Field

from . import db

SANDBOX = Path(__file__).with_name("sandbox")


# --------------------------------------------------------------------------
# Argument schemas
# --------------------------------------------------------------------------

class ListItems(BaseModel):
    project_key: str
    status: Literal["todo", "in_progress", "blocked", "done"] | None = None


class GetItem(BaseModel):
    item_id: int


class ListTeamMembers(BaseModel):
    project_key: str


class ReassignItem(BaseModel):
    item_id: int
    assignee_id: int


class PostStatusUpdate(BaseModel):
    project_key: str
    body: str = Field(min_length=1, max_length=2000)


class VelocityHistory(BaseModel):
    project_key: str


class FetchUrl(BaseModel):
    url: str


class ReadFile(BaseModel):
    path: str


class WriteFile(BaseModel):
    path: str
    content: str


# --------------------------------------------------------------------------
# Tool specifications
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ToolSpec:
    name: str
    server: str
    args: type[BaseModel]
    description: str
    write: bool = False
    reach: Literal["internal", "external"] = "internal"
    scope: Literal["project", "path", "domain"] = "project"
    irreversible: bool = False
    externally_visible: bool = False
    project_arg: str | None = "project_key"
    remote_name: str | None = None
    handler: Callable[..., object] | None = field(default=None, compare=False)

    @property
    def wire_name(self) -> str:
        return self.remote_name or self.name

    @property
    def approval_required(self) -> bool:
        """Consequential = irreversible OR externally visible.

        The other two limbs of the rule — above a value threshold, or triggered
        by untrusted input — depend on the call, not the tool, so they are
        evaluated per-call in policy.py.
        """
        return self.irreversible or self.externally_visible


# --------------------------------------------------------------------------
# Handlers
# --------------------------------------------------------------------------

def _list_items(project_key: str, status: str | None = None) -> list[dict]:
    with db.readonly() as c:
        if status:
            return db.rows(c, "SELECT * FROM sprint_items WHERE project_key=%s AND status=%s",
                           project_key, status)
        return db.rows(c, "SELECT * FROM sprint_items WHERE project_key=%s", project_key)


def _get_item(item_id: int) -> dict | None:
    with db.readonly() as c:
        found = db.rows(c, "SELECT * FROM sprint_items WHERE id=%s", item_id)
    return found[0] if found else None


def _list_team_members(project_key: str) -> list[dict]:
    with db.readonly() as c:
        return db.rows(c, "SELECT * FROM team_members WHERE project_key=%s", project_key)


def _reassign_item(item_id: int, assignee_id: int) -> dict:
    with db.writable() as c, c.cursor() as cur:
        cur.execute("UPDATE sprint_items SET assignee_id=%s WHERE id=%s", (assignee_id, item_id))
        if cur.rowcount == 0:
            raise LookupError(f"no sprint item {item_id}")
    return {"item_id": item_id, "assignee_id": assignee_id}


def _post_status_update(project_key: str, body: str) -> dict:
    now = datetime.now(timezone.utc).isoformat()
    with db.writable() as c, c.cursor() as cur:
        cur.execute(
            "INSERT INTO status_updates (project_key, body, posted_at)"
            " VALUES (%s,%s,%s) RETURNING id",
            (project_key, body, now),
        )
        new_id = cur.fetchone()["id"]
    return {"id": new_id, "project_key": project_key, "posted_at": now}


def _velocity_history(project_key: str) -> list[dict]:
    # Uses the replica role: this connection cannot write, by permission.
    with db.readonly() as c:
        return db.rows(c, "SELECT * FROM velocity_history WHERE project_key=%s ORDER BY sprint",
                       project_key)


def _fetch_url(url: str) -> dict:
    return {"url": url, "status": "not_fetched", "note": "egress allowed by policy"}


def _safe_path(path: str) -> Path:
    SANDBOX.mkdir(exist_ok=True)
    resolved = (SANDBOX / path).resolve()
    if not resolved.is_relative_to(SANDBOX.resolve()):
        raise PermissionError(f"path escapes sandbox: {path}")
    return resolved


def _read_file(path: str) -> str:
    return _safe_path(path).read_text()


def _write_file(path: str, content: str) -> dict:
    target = _safe_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content)
    return {"path": path, "bytes": len(content)}


# --------------------------------------------------------------------------
# Tool registry
# --------------------------------------------------------------------------

TOOLS: dict[str, ToolSpec] = {
    t.name: t for t in [
        ToolSpec("sprint.list_items", "sprint", ListItems,
                 "List sprint items for a project.", handler=_list_items),
        ToolSpec("sprint.get_item", "sprint", GetItem,
                 "Get one sprint item by id.", project_arg=None, handler=_get_item),
        ToolSpec("sprint.list_team_members", "sprint", ListTeamMembers,
                 "List team members and their load.", handler=_list_team_members),
        ToolSpec("sprint.reassign_item", "sprint", ReassignItem,
                 "Reassign a sprint item to another team member.",
                 write=True, externally_visible=True, project_arg=None,
                 handler=_reassign_item),
        ToolSpec("sprint.post_status_update", "sprint", PostStatusUpdate,
                 "Post a delivery status update to the project channel.",
                 write=True, irreversible=True, externally_visible=True,
                 handler=_post_status_update),
        ToolSpec("warehouse.velocity_history", "warehouse", VelocityHistory,
                 "Read committed-vs-completed points per sprint (read replica).",
                 handler=_velocity_history),
        ToolSpec("http.fetch_url", "http", FetchUrl,
                 "Fetch a URL from the egress allowlist.",
                 reach="external", scope="domain", project_arg=None,
                 handler=_fetch_url),
        ToolSpec("fs.read_file", "fs", ReadFile,
                 "Read a file inside the sandbox.",
                 scope="path", project_arg=None, handler=_read_file),
        ToolSpec("fs.write_file", "fs", WriteFile,
                 "Write a file inside the sandbox.",
                 write=True, irreversible=True, scope="path", project_arg=None,
                 handler=_write_file),
    ]
}

SERVERS = sorted({t.server for t in TOOLS.values()})


def permission_matrix() -> list[dict]:
    """Generated from the same specs used at runtime, never hand-written."""
    return [
        {
            "tool": t.name,
            "server": t.server,
            "access": "write" if t.write else "read",
            "reach": t.reach,
            "scope": t.scope,
            "approval": "yes" if t.approval_required else "no",
        }
        for t in sorted(TOOLS.values(), key=lambda t: (t.server, t.name))
    ]


def matrix_markdown() -> str:
    rows = permission_matrix()
    cols = ["tool", "server", "access", "reach", "scope", "approval"]
    out = ["| " + " | ".join(c.capitalize() for c in cols) + " |",
           "|" + "|".join("---" for _ in cols) + "|"]
    out += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
    return "\n".join(out)
