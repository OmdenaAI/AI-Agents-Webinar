"""
The audience-facing display.
Display-only, one-way, and deliberately small. 
It renders events that have already been through the guardrail scan: 
the stream carries decisions, tool names, flags and costs, 
never raw tool output, so there is nothing here to redact a second time.
"""

from __future__ import annotations

import html
import json
import os
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from psycopg import sql as pgsql

from . import db
from .events import stream
from .guardrails import scan

PAGE = """<!doctype html>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Project Operations Agent</title>
<style>
 :root{--bg:#0b0d10;--fg:#e8ecf1;--dim:#7b8794;--ok:#25c56a;--no:#ff4d4d;--wait:#ffb020;--line:#1d2229}
 *{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--fg);
   font:600 20px/1.45 ui-monospace,SFMono-Regular,Menlo,monospace}
 header{display:flex;justify-content:space-between;align-items:baseline;gap:16px;
   padding:18px 28px;border-bottom:2px solid var(--line)}
 h1{font-size:22px;margin:0;letter-spacing:.06em;text-transform:uppercase}
 #cost{font-size:30px} #cost small{color:var(--dim);font-size:15px}
 /* FR-35: what the agent is doing *now*, in words, above the detail feed. */
 #now{padding:20px 28px;border-bottom:2px solid var(--line);background:#0e1116}
 #scenario{font-size:14px;letter-spacing:.09em;text-transform:uppercase;color:var(--dim)}
 #step{font-size:clamp(22px,3.2vw,38px);margin-top:6px}
 #step.allowed{color:var(--ok)} #step.denied{color:var(--no)} #step.flag{color:var(--wait)}
 /* The architecture, drawn once and coloured live. Inline SVG on purpose,
    nothing fetched at runtime, and a diagram that fails to render on venue wifi during
    the opening scenario is the worst possible failure. */
 #arch{padding:14px 28px 4px;border-bottom:2px solid var(--line)}
 #arch svg{width:100%;height:auto;max-height:38vh;display:block}
 .n rect{fill:#11151a;stroke:var(--line);stroke-width:2;rx:7}
 .n .lbl{fill:var(--fg);font-size:15px;font-weight:600}
 .n .sub{fill:var(--dim);font-size:12px;font-weight:400}
 .n .ms{fill:var(--dim);font-size:12px}
 .n.up rect{stroke:var(--ok)} .n.up .ms{fill:var(--ok)}
 .n.down rect{stroke:var(--no)} .n.down .ms{fill:var(--no)}
 .wire{stroke:var(--line);stroke-width:2;fill:none;marker-end:url(#a)}
 .tap{stroke:#2c3a44;stroke-width:2;fill:none;stroke-dasharray:4 5}
 .zone{fill:var(--dim);font-size:11px;letter-spacing:.13em;text-transform:uppercase}
 /* The result, not just the trail that led to it. */
 #answer{display:none;margin:0 28px 18px;padding:20px 24px;border-left:8px solid var(--ok);
   background:#101a14;font-size:clamp(16px,1.5vw,21px);font-weight:400;white-space:pre-wrap}
 #answer.filtered{border-color:var(--wait);background:#1a1408}
 #answer h2{font-size:13px;letter-spacing:.09em;text-transform:uppercase;
   color:var(--dim);margin:0 0 10px;font-weight:600}
 #feed{padding:16px 28px;display:flex;flex-direction:column-reverse;gap:10px}
 .row{display:grid;grid-template-columns:150px 1fr;gap:18px;align-items:baseline;
   padding:12px 16px;border-left:8px solid var(--line);background:#11151a}
 .tag{font-size:15px;letter-spacing:.08em;text-transform:uppercase;color:var(--dim)}
 .allowed{border-color:var(--ok)} .allowed .tag{color:var(--ok)}
 .denied{border-color:var(--no)} .denied .tag{color:var(--no)}
 .flag{border-color:var(--wait)} .flag .tag{color:var(--wait)}
 .detail{font-size:24px;word-break:break-word}
 .why{display:block;font-weight:400;font-size:17px;color:var(--dim);margin-top:4px}
 /* Unmistakable, not a spinner. */
 #gate{position:fixed;inset:0;background:#1a1206;color:var(--wait);display:none;
   flex-direction:column;align-items:center;justify-content:center;text-align:center;
   gap:24px;padding:40px;z-index:9}
 #gate.on{display:flex} #gate b{font-size:min(9vw,86px);letter-spacing:.04em}
 #gate span{font-size:min(3.4vw,30px);color:#f5e2c0;font-weight:400}
</style>
<header>
  <h1>Project Operations Agent</h1>
  <div id="cost">$0.0000 <small id="meta">0 steps</small></div>
</header>
<div id="gate"><b>AWAITING HUMAN APPROVAL</b><span id="gatewhat"></span></div>
<div id="arch"><svg viewBox="0 0 940 404" role="img" aria-label="System architecture"><defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0 L10 5 L0 10 z" fill="#2a323d"/></marker></defs><text class="zone" x="16" y="14">ask</text><text class="zone" x="364" y="14">decide</text><text class="zone" x="556" y="14">tools</text><text class="zone" x="760" y="14">systems</text><g class="n" id="n-ask"><rect x="16" y="150" width="158" height="54"/><text class="lbl" x="28" y="173">Slack</text><text class="sub" x="28" y="191">#agent-ask</text><text class="ms" x="162" y="191" text-anchor="end"></text></g><g class="n" id="n-litellm"><rect x="190" y="60" width="158" height="54"/><text class="lbl" x="202" y="83">LiteLLM</text><text class="sub" x="202" y="101">model gateway</text><text class="ms" x="336" y="101" text-anchor="end"></text></g><g class="n" id="n-orchestrator"><rect x="190" y="150" width="158" height="54"/><text class="lbl" x="202" y="173">Orchestrator</text><text class="sub" x="202" y="191">LangGraph</text><text class="ms" x="336" y="191" text-anchor="end"></text></g><g class="n" id="n-policy"><rect x="364" y="150" width="158" height="54"/><text class="lbl" x="376" y="173">Policy</text><text class="sub" x="376" y="191">deterministic</text><text class="ms" x="510" y="191" text-anchor="end"></text></g><g class="n" id="n-approvals"><rect x="364" y="258" width="158" height="54"/><text class="lbl" x="376" y="281">Slack</text><text class="sub" x="376" y="299">#agent-approvals</text><text class="ms" x="510" y="299" text-anchor="end"></text></g><g class="n" id="n-sprint"><rect x="556" y="40" width="150" height="46"/><text class="lbl" x="568" y="63">sprint</text><text class="sub" x="568" y="81">kanban</text><text class="ms" x="694" y="81" text-anchor="end"></text></g><g class="n" id="n-warehouse"><rect x="556" y="100" width="150" height="46"/><text class="lbl" x="568" y="123">warehouse</text><text class="sub" x="568" y="141">read replica</text><text class="ms" x="694" y="141" text-anchor="end"></text></g><g class="n" id="n-http"><rect x="556" y="160" width="150" height="46"/><text class="lbl" x="568" y="183">http</text><text class="sub" x="568" y="201">egress allowlist</text><text class="ms" x="694" y="201" text-anchor="end"></text></g><g class="n" id="n-fs"><rect x="556" y="220" width="150" height="46"/><text class="lbl" x="568" y="243">fs</text><text class="sub" x="568" y="261">sandbox</text><text class="ms" x="694" y="261" text-anchor="end"></text></g><g class="n" id="n-umaku"><rect x="556" y="280" width="150" height="46"/><text class="lbl" x="568" y="303">umaku</text><text class="sub" x="568" y="321">third-party</text><text class="ms" x="694" y="321" text-anchor="end"></text></g><g class="n" id="n-postgres"><rect x="760" y="70" width="164" height="46"/><text class="lbl" x="772" y="93">Postgres</text><text class="sub" x="772" y="111">seed + audit</text><text class="ms" x="912" y="111" text-anchor="end"></text></g><g class="n" id="n-internet"><rect x="760" y="160" width="164" height="46"/><text class="lbl" x="772" y="183">Internet</text><text class="sub" x="772" y="201">allowlisted</text><text class="ms" x="912" y="201" text-anchor="end"></text></g><g class="n" id="n-sandbox"><rect x="760" y="220" width="164" height="46"/><text class="lbl" x="772" y="243">Filesystem</text><text class="sub" x="772" y="261">path-restricted</text><text class="ms" x="912" y="261" text-anchor="end"></text></g><g class="n" id="n-umaku-saas"><rect x="760" y="280" width="164" height="46"/><text class="lbl" x="772" y="303">Umaku SaaS</text><text class="sub" x="772" y="321">live board</text><text class="ms" x="912" y="321" text-anchor="end"></text></g><path class="wire" d="M174 177 H186"/><path class="wire" d="M348 177 H360"/><path class="wire" d="M269 148 V116"/><path class="wire" d="M443 204 V254"/><path class="tap" d="M522 63 V303"/><path class="wire" d="M522 177 H360" transform="rotate(180 441 177)"/><path class="wire" d="M522 63 H552"/><path class="wire" d="M522 123 H552"/><path class="wire" d="M522 183 H552"/><path class="wire" d="M522 243 H552"/><path class="wire" d="M522 303 H552"/><path class="wire" d="M706 63 H730 V93 H756"/><path class="wire" d="M706 123 H730 V93 H756"/><path class="wire" d="M706 183 H730 V183 H756"/><path class="wire" d="M706 243 H730 V243 H756"/><path class="wire" d="M706 303 H730 V303 H756"/><path class="tap" d="M269 210 V352 H520"/><path class="tap" d="M443 210 V352"/><path class="tap" d="M631 326 V352"/><g class="n" id="n-otel"><rect x="520" y="330" width="404" height="46"/><text class="lbl" x="532" y="353">OpenTelemetry</text><text class="sub" x="532" y="371">one trace per run, joined to audit</text><text class="ms" x="912" y="371" text-anchor="end"></text></g></svg></div>
<div id="now">
  <div id="scenario">idle</div>
  <div id="step">waiting for a scenario</div>
</div>
<div id="answer"><h2>Agent answer</h2><div id="answertext"></div></div>
<div id="feed"></div>
<script>
const feed=document.getElementById('feed'),gate=document.getElementById('gate');
let cost=0,steps=0;
const TAG={policy_decision:'policy',tool_result:'tool',tool_error:'tool',
  tool_retry:'retry',
  model_call:'model',guardrail_flag:'guardrail',approval_decision:'approval',
  budget_exceeded:'budget',model_degraded:'degraded',model_fallback:'failover',run_stopped:'stopped',answer:'answer',answer_unavailable:'answer',
  service_status:'service',tool_permission:'permissions',build:'build',
  audit_row:'audit'};
function cls(e){
  if(e.event==='policy_decision')return e.outcome==='denied'?'denied':
    (e.outcome==='approval_required'?'flag':'allowed');
  if(e.event==='guardrail_flag'||e.event==='model_degraded'||
     e.event==='model_fallback'||e.event==='tool_retry')return 'flag';
  if(e.event==='tool_error'||e.event==='budget_exceeded'||e.event==='run_denied')return 'denied';
  if(e.event==='service_status')return e.ok?'allowed':'denied';
  if(e.event==='approval_decision')return e.approved?'allowed':'denied';
  return '';
}
function line(e){
  if(e.event==='policy_decision')return (e.outcome||'').toUpperCase()+' · '+e.tool;
  if(e.event==='tool_result')return e.tool+(e.flagged?' · FLAGGED':' · ok');
  if(e.event==='tool_retry')return e.tool+' · attempt '+e.attempt+'/'+e.of+
    ' · '+(e.error||'');
  if(e.event==='tool_error')return e.tool+' · '+(e.error||'failed')+' · '+
    (e.detail||'');
  if(e.event==='model_call')return (e.model||'model')+' · '+(e.tokens||0)+' tokens';
  if(e.event==='guardrail_flag')return (e.kind||'guardrail')+' · '+(e.category||'')+
    ' · '+(e.excerpt||'');
  if(e.event==='approval_decision')return (e.approved?'APPROVED':'REJECTED')+
    ' by '+(e.approver||'—');
  if(e.event==='answer')return 'answer ready';
  if(e.event==='answer_unavailable')return 'no model summary — '+(e.reason||'');
  return e.event;
}
// A step description, not a log line: what is happening, in
// words a room can read from the back.
function describe(e){
  switch(e.event){
    case 'scenario_started': return 'starting…';
    case 'scenario_finished': return 'done';
    case 'scenario_failed':   return 'scenario failed: '+(e.error||'');
    case 'model_call': return e.tool_calls&&e.tool_calls.length
        ? 'choosing tools: '+e.tool_calls.join(', ')
        : 'thinking on '+(e.model||'the model');
    case 'policy_decision':
      if(e.outcome==='denied') return 'DENIED · '+e.tool+' — '+(e.reason||'');
      if(e.outcome==='approval_required') return 'needs approval · '+e.tool;
      return 'calling '+e.tool;
    case 'tool_result': return e.tool+(e.flagged?' returned flagged content':' returned');
    case 'tool_retry':
      return e.tool+' unreachable — retrying ('+e.attempt+' of '+e.of+')';
    case 'tool_error':
      return e.tool+' failed: '+(e.error||'')+' — continuing without it';
    case 'guardrail_flag': return 'guardrail: '+(e.kind||'')+' '+(e.category||'')+' detected';
    case 'approval_decision': return (e.approved?'approved':'rejected')+' by '+(e.approver||'a human');
    case 'budget_exceeded': return 'stopped by the gateway budget';
    case 'model_degraded': return 'model gateway unavailable — degraded mode';
    case 'model_fallback':
      return 'hosted model unavailable — continuing on '+e.to;
    case 'run_stopped': return 'run stopped: '+(e.reason||'');
    case 'answer': return 'answer ready';
    case 'audit_row': return 'audit · '+e.action+' · '+e.outcome;
    case 'build': return 'running build '+e.build;
    case 'service_status':
      return e.service+' · '+(e.ok?'up':'DOWN')+' · '+e.ms+' ms'+
             (e.detail?' · '+e.detail:'');
    case 'tool_permission':
      return e.server+' · '+e.tools+' tools · '+e.writes+' write · '+
             e.approval+' need approval · '+e.reach;
    default: return e.event;
  }
}
function markService(e){
  const g=document.getElementById('n-'+e.service);
  if(!g)return;
  g.classList.remove('up','down');
  g.classList.add(e.ok?'up':'down');
  const t=g.querySelector('.ms');
  if(t)t.textContent=e.ok?(e.ms+' ms'):'down';
}
function render(e){
  if(e.event==='service_status')markService(e);
  if(e.event==='model_call'||e.event==='answer'){cost+=e.cost_usd||0;}
  if(e.event==='policy_decision'||e.event==='tool_result')steps++;
  document.getElementById('cost').firstChild.textContent='$'+cost.toFixed(4)+' ';
  document.getElementById('meta').textContent=steps+' steps';
  if(e.event==='scenario_started'){
    document.getElementById('scenario').textContent=e.scenario||'running';
    document.getElementById('answer').style.display='none';
  }
  if((e.event==='answer'||e.event==='answer_unavailable')&&e.text){
    const box=document.getElementById('answer');
    document.getElementById('answertext').textContent=e.text;
    box.className=e.event==='answer_unavailable'?'filtered':'';
    box.style.display='block';
  }
  const step=document.getElementById('step');
  step.textContent=describe(e);
  step.className=cls(e);
  const pending=e.event==='policy_decision'&&e.outcome==='approval_required';
  if(pending){document.getElementById('gatewhat').textContent=e.tool+' — '+(e.reason||'');}
  if(pending)gate.classList.add('on');
  if(e.event==='approval_decision'||e.event==='run_stopped')gate.classList.remove('on');
  const row=document.createElement('div');row.className='row '+cls(e);
  row.innerHTML='<div class="tag"></div><div class="detail"></div>';
  row.children[0].textContent=TAG[e.event]||e.event;
  row.children[1].textContent=line(e);
  if(e.reason){const w=document.createElement('span');w.className='why';
    w.textContent=e.reason;row.children[1].appendChild(w);}
  feed.appendChild(row);
  while(feed.children.length>60)feed.removeChild(feed.firstChild);
}
new EventSource('/events').onmessage=m=>{if(m.data)render(JSON.parse(m.data));};
</script>
"""


# The data behind the demo, shown rather than described (the scenarios run against real services). 
# A fixed list, not a query box: this page is reachable from the venue network, 
# and the honest way to keep a display display-only is to give it nothing to drive.
DATA_TABLES = ("sprint_items", "team_members", "projects", "velocity_history",
               "status_updates", "audit_log")

DATA_PAGE = """<!doctype html>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="refresh" content="5">
<title>Project Operations Agent — data</title>
<style>
 :root{{--bg:#0b0d10;--fg:#e8ecf1;--dim:#7b8794;--no:#ff4d4d;--line:#1d2229}}
 *{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--fg);
   font:500 15px/1.45 ui-monospace,SFMono-Regular,Menlo,monospace}}
 header{{padding:16px 24px;border-bottom:2px solid var(--line)}}
 h1{{font-size:18px;margin:0 0 10px;letter-spacing:.06em;text-transform:uppercase}}
 nav a{{display:inline-block;margin:0 8px 6px 0;padding:7px 13px;color:var(--dim);
   text-decoration:none;border:1px solid var(--line);border-radius:5px}}
 nav a.on{{color:var(--fg);border-color:#3a4653;background:#161b22}}
 .wrap{{overflow-x:auto;padding:18px 24px}}
 table{{border-collapse:collapse;width:100%}}
 th,td{{text-align:left;padding:7px 14px 7px 0;border-bottom:1px solid var(--line);
   vertical-align:top;white-space:pre-wrap}}
 th{{color:var(--dim);font-weight:600;text-transform:uppercase;font-size:12px;
   letter-spacing:.07em}}
 td.r{{color:var(--no)}}
 footer{{padding:0 24px 26px;color:var(--dim);font-size:13px;max-width:78ch}}
</style>
<header>
  <h1>What the agent can actually read</h1>
  <nav>{nav}</nav>
</header>
<div class="wrap">{table}</div>
<footer>{note}</footer>
"""

REDACTION_NOTE = ("{n} field{s} on this page {were} redacted before rendering, by "
                  "the same scan that runs on every tool result before it reaches "
                  "the model. The database holds the real value; this display "
                  "never receives it, and neither did the agent.")

READ_ONLY_NOTE = ("Read through the replica role, which Postgres refuses writes "
                  "to — this page could not change a row if it tried. Refreshes "
                  "every 5 seconds.")


def _cell(value) -> tuple[str, bool]:
    """
    One cell, scanned. Returns the text to render and whether it was redacted.
    This page goes on a projector and may be photographed. The seed data
    carries a government id on purpose, so rendering it raw would break
    the constraint the demo states out loud two scenarios earlier.
    """
    text = "" if value is None else str(value)
    result = scan(text)
    return result.redacted, result.redacted != text


def data_page(table: str) -> str:
    """Render one table. "table" must already be one of DATA_TABLES."""
    with db.readonly() as conn:
        # Identifier, not interpolation, the membership check above is the real
        # control, and this is the one that still holds if that check is edited.
        rows = db.rows(conn, pgsql.SQL("SELECT * FROM {} ORDER BY 1")
                       .format(pgsql.Identifier(table)))

    nav = "".join(
        f'<a href="/data?table={name}" class="{"on" if name == table else ""}">'
        f'{name}</a>' for name in DATA_TABLES)

    if not rows:
        body, redacted = f"<p>{table} is empty.</p>", 0
    else:
        redacted = 0
        head = "".join(f"<th>{html.escape(c)}</th>" for c in rows[0])
        lines = []
        for row in rows:
            cells = []
            for value in row.values():
                text, was = _cell(value)
                redacted += was
                cells.append(f'<td class="{"r" if was else ""}">'
                             f'{html.escape(text)}</td>')
            lines.append("<tr>" + "".join(cells) + "</tr>")
        body = (f"<table><tr>{head}</tr>" + "".join(lines) + "</table>")

    note = READ_ONLY_NOTE
    if redacted:
        note = REDACTION_NOTE.format(n=redacted, s="" if redacted == 1 else "s",
                                     were="was" if redacted == 1 else "were") \
            + " " + note
    return DATA_PAGE.format(nav=nav, table=body, note=note)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:  # noqa: N802 - stdlib naming
        if self.path.startswith("/events"):
            return self._sse()
        if self.path.startswith("/data"):
            return self._data()
        if self.path in ("/", "/index.html"):
            return self._page()
        self.send_error(404)

    def _data(self) -> None:

        query = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        table = (query.get("table") or [DATA_TABLES[0]])[0]
        if table not in DATA_TABLES:
            return self.send_error(404)
        try:
            body = data_page(table).encode()
        except Exception as e:
            # A database that is down must not take the display with it.
            body = (f"<p>data unavailable: {type(e).__name__}</p>").encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _page(self) -> None:
        body = PAGE.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _sse(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        # An HTTP/1.1 response with neither Content-Length nor chunked encoding
        # has no framing, so a browser cannot tell where the body ends and may
        # hold the whole stream in a buffer — the page connects and then shows
        # nothing. Closing the connection at the end is the framing. (curl -N
        # tolerates the ambiguity, which is why this passed a curl check.)
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        try:
            for event in stream():
                # A bare comment is the SSE keepalive; it keeps an idle
                # connection open through a proxy without faking an event.
                chunk = (f"data: {json.dumps(event)}\n\n" if event
                         else ": keepalive\n\n")
                self.wfile.write(chunk.encode())
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass  # the projector closed the tab; the run continues

    def log_message(self, *args) -> None:
        pass  # a live demo does not need an access log on stdout


def serve(host: str = "0.0.0.0", port: int | None = None) -> ThreadingHTTPServer:
    # `port or ...` would turn an explicit 0 — "pick a free port" — into 8080.
    if port is None:
        port = int(os.environ.get("UI_PORT", 8080))
    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    return server


def main() -> None:
    server = serve()
    host, port = server.server_address[:2]
    print(f"audience UI on http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
