"""
Backend access, through two distinct database roles.
Connection details come entirely from the environment: no host, port, or
credential is baked into the image or this module.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path

import psycopg
from psycopg import sql
from psycopg.rows import dict_row


def _dsn(role: str) -> str:
    """Build a DSN for one role. Raises if the environment is incomplete."""
    var = {"app": "DATABASE_URL", "replica": "DATABASE_URL_RO",
           "admin": "DATABASE_ADMIN_URL"}[role]
    dsn = os.environ.get(var)
    if not dsn:
        raise RuntimeError(
            f"{var} is not set. start Postgres and export it "
            f"(see .env.example). No default is assumed."
        )
    return dsn


@contextmanager
def writable():
    """The application role: may INSERT/UPDATE sprint_items and status_updates."""
    with psycopg.connect(_dsn("app"), row_factory=dict_row) as conn:
        yield conn


@contextmanager
def readonly():
    """The replica role. Writes raise psycopg.errors.InsufficientPrivilege."""
    with psycopg.connect(_dsn("replica"), row_factory=dict_row) as conn:
        yield conn


def rows(conn, sql: str, *params) -> list[dict]:
    with conn.cursor() as cur:
        cur.execute(sql, params or None)
        return cur.fetchall()


def build() -> str:
    """
    Recreate schema, seed data and grants.

    Runs as the admin role: agent_app deliberately cannot create or drop tables. 
    Recreating the tables drops their grants with
    them, so the roles file is re-applied every time, not only on first init.

    Under Compose, Postgres runs these same two files on first init.
    """

    sql_dir = Path(__file__).parent.parent.parent / "sql"
    with psycopg.connect(_dsn("admin"), autocommit=True) as conn:
        conn.execute((sql_dir / "01-schema-and-seed.sql").read_text())
        roles = (sql_dir / "02-roles.sql").read_text()
        roles = "\n".join(
            line for line in roles.splitlines() if not line.startswith("\\set"))
        for placeholder, var in ((":'app_pw'", "AGENT_APP_PASSWORD"),
                                 (":'ro_pw'", "AGENT_REPLICA_PASSWORD")):
            literal = sql.Literal(os.environ[var]).as_string(conn)
            roles = roles.replace(placeholder, literal)
        conn.execute(roles)
    return _dsn("admin").rsplit("@", 1)[-1]  # host/db only, never the credential
