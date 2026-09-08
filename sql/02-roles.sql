-- Two genuinely separate database roles.
--
-- The "read replica" in the guidelines stack table must be a real permission
-- boundary, not a naming convention: agent_replica is refused writes by
-- Postgres itself, so the warehouse tool cannot mutate project data even if
-- every layer above it is compromised or buggy.
--
-- Passwords come from the environment - psql substitutes :'var' from
-- the variables the entrypoint passes in.

\set app_pw `echo "$AGENT_APP_PASSWORD"`
\set ro_pw  `echo "$AGENT_REPLICA_PASSWORD"`

-- Idempotent: this file also runs on every `build()` reset.
DO $$ BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'agent_app') THEN
        CREATE ROLE agent_app LOGIN;
    END IF;
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'agent_replica') THEN
        CREATE ROLE agent_replica LOGIN;
    END IF;
END $$;

ALTER ROLE agent_app     PASSWORD :'app_pw';
ALTER ROLE agent_replica PASSWORD :'ro_pw';

GRANT CONNECT ON DATABASE projectops TO agent_app, agent_replica;
GRANT USAGE   ON SCHEMA public       TO agent_app, agent_replica;

-- The application role: reads everything, writes only what the agent may change.
GRANT SELECT ON ALL TABLES IN SCHEMA public TO agent_app;
GRANT INSERT, UPDATE ON sprint_items   TO agent_app;
GRANT INSERT         ON status_updates TO agent_app;
GRANT USAGE, SELECT  ON ALL SEQUENCES IN SCHEMA public TO agent_app;

-- append-only is enforced by the database, not by application code.
-- The application role may add audit records and read them back, and is
-- structurally incapable of altering or deleting one.
GRANT INSERT, SELECT ON audit_log TO agent_app;
REVOKE UPDATE, DELETE, TRUNCATE ON audit_log FROM agent_app;

-- Graph checkpoints live in their own schema, owned by the app role.
-- Not in `public`: agent run state is not project data, and keeping it out
-- leaves the audit-log grants above exactly as they are.
CREATE SCHEMA IF NOT EXISTS agent_state AUTHORIZATION agent_app;
REVOKE ALL ON SCHEMA agent_state FROM agent_replica;

-- The replica role: SELECT and nothing else, on every table, forever.
GRANT SELECT ON ALL TABLES IN SCHEMA public TO agent_replica;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO agent_replica;

-- Belt and braces: even a future GRANT cannot hand the replica a write.
REVOKE INSERT, UPDATE, DELETE, TRUNCATE ON ALL TABLES IN SCHEMA public FROM agent_replica;
