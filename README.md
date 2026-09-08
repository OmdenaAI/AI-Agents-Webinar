# AI-Agents-Webinar

## Project Operations Agent

A live-demo system for the webinar *Deploying AI Agents with Real-World Access:
What You Need to Know About Security, Cost, and Control*.

An agent reads real sprint and kanban data, answers delivery-risk questions, and
proposes write actions. Every scenario runs against real services — a real model
behind a real gateway, real MCP tool servers over HTTP, a real Postgres, a real
Slack approval, a real trace store. The point of the demo is to show the
mechanisms working, so nothing here is simulated.

## Quickstart

```bash
uv sync
cp .env.example .env          # then fill it in — see "Credentials" below
docker compose up -d --build
uv run ai-agents-webinar --provision-keys   # LiteLLM budget keys; paste into .env
docker compose up -d --force-recreate orchestrator
```

Then open the **audience display** at <http://localhost:8080>. The control surface is
published to loopback only, deliberately: it can start real actions, so it must
not be reachable from the room.

Everything is also drivable from the CLI:

```bash
uv run ai-agents-webinar                       # print the tool-permission matrix
uv run ai-agents-webinar --seed                # rebuild the demo database
uv run ai-agents-webinar --ask "what is at risk?"
uv run ai-agents-webinar --compare "..."       # naive vs hardened, with costs
uv run ai-agents-webinar --scenario-4          # write action + human approval
uv run ai-agents-webinar --eval                # offline policy regression gate
```

## The six scenarios

| # | Scenario | What it proves |
|---|---|---|
| 0 | Architecture walkthrough | the services are actually running |
| 1 | Baseline read-only run | real tool access, with a live trace and cost |
| 2a | Out-of-scope request | the boundary holds without changing the prompt |
| 2b | Prompt injection via seeded data | the model was persuaded; the system was not |
| 3 | `naive` vs `hardened` | where agent cost actually comes from |
| 4 | Write action | a human decides, and the reject path really stops it |
| 5 | Audit reconstruction | what it did, who authorised it, can we prove it |

## How it fits together

![Architecture: Slack to the orchestrator, through deterministic policy, out to
five MCP tool servers and the systems behind them, with OpenTelemetry observing
every step](docs/architecture.svg)

A question arrives in Slack. The orchestrator plans against a model it can only
reach through the gateway, and **every tool call the model chooses passes the
same deterministic policy check** before anything executes. Consequential
actions stop there and wait for a human in a second Slack channel. Each tool
server is a separate process reached over HTTP, and validates the agent's
credential itself rather than trusting the caller. One trace per run carries the
gateway's own cost figure and joins to the append-only audit log by trace id.

The design constraints that matter:

- **Policy is deterministic and model-free.** Allowlist, schema validation,
  scope, egress, value thresholds, per-run caps, approval rule. It is the
  cheapest layer to test exhaustively and the one an audience will scrutinise.
- **Tool schemas are defined once.** Runtime validation, the permission matrix,
  and the schema the model sees all come from the same `ToolSpec`. Third-party
  tools are *discovered* from their published schema, never transcribed.
- **Identity is verified at the tool server**, not asserted by the caller.
- **Approval is a real LangGraph `interrupt()`**, checkpointed in Postgres, so a
  pending decision survives a restart.
- **Audit is append-only by database grant**, not by application code.
- **Instrumentation names no backend.** Standard OTLP with GenAI semantic
  conventions; where it goes is an environment variable.

## Credentials

Nothing is committed. `.env.example` lists what is needed: a model provider key
(behind the gateway), Slack bot and app tokens, trace-store keys, database
passwords, an agent signing secret, and a token for the third-party kanban tool.

## Repository

| Path | Role |
|---|---|
| `src/ai_agents_webinar/` | the agent, policy, tools, servers, UI |
| `sql/` | schema, seed data, and the role grants that enforce append-only audit |
| `litellm/config.yaml` | model routing and budget caps |
| `docker-compose.yml` | the whole topology |
