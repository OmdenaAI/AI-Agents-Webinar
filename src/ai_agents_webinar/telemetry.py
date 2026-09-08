"""
OpenTelemetry instrumentation.
Instrumentation emits standard OTLP using the GenAI
semantic conventions, and where it goes is decided by environment variables. 
A different backend, or a second one, is a configuration change, no span is
re-instrumented.
"""

from __future__ import annotations

import base64
import os
from contextlib import contextmanager

from opentelemetry import context, trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

SERVICE_NAME = "project-ops-orchestrator"

# GenAI semantic convention attribute names (stable subset).
GEN_AI_SYSTEM = "gen_ai.system"
GEN_AI_OPERATION = "gen_ai.operation.name"
GEN_AI_REQUEST_MODEL = "gen_ai.request.model"
GEN_AI_USAGE_INPUT = "gen_ai.usage.input_tokens"
GEN_AI_USAGE_OUTPUT = "gen_ai.usage.output_tokens"
GEN_AI_TOOL_NAME = "gen_ai.tool.name"
GEN_AI_RESPONSE_MODEL = "gen_ai.response.model"
GEN_AI_USAGE_COST = "gen_ai.usage.cost"
SESSION_ID = "session.id"
TRACE_TAGS = "langfuse.trace.tags"

_configured = False


def _otlp_settings() -> tuple[str, dict] | None:
    """
    Resolve an OTLP endpoint from the environment, or None to run dark.
    Standard OTEL_* variables win. The Langfuse variables are a convenience for
    this project only, they are translated into plain OTLP here so that nothing
    downstream knows which backend is in use.
    """
    endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    if endpoint:
        return endpoint.rstrip("/") + "/v1/traces", {}

    base = os.environ.get("LANGFUSE_BASE_URL")
    public = os.environ.get("LANGFUSE_PUBLIC_KEY")
    secret = os.environ.get("LANGFUSE_SECRET_KEY")
    if base and public and secret:
        token = base64.b64encode(f"{public}:{secret}".encode()).decode()
        return base.rstrip("/") + "/api/public/otel/v1/traces", {
            "Authorization": f"Basic {token}"}
    return None


def setup(*, force: bool = False) -> bool:
    """Install the tracer provider. Returns whether an exporter was attached."""
    global _configured
    if _configured and not force:
        return True

    provider = TracerProvider(resource=Resource.create({"service.name": SERVICE_NAME}))
    settings = _otlp_settings()
    attached = False

    if settings:
        endpoint, headers = settings
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )

        exporter = OTLPSpanExporter(endpoint=endpoint, headers=headers)
        provider.add_span_processor(BatchSpanProcessor(exporter))
        attached = True

    trace.set_tracer_provider(provider)
    _configured = True
    return attached


def session_url(session_id: str) -> str | None:
    """
    A link to every trace in one session, if the backend can show one.
    Lives here because makes this the only module permitted to know which
    backend is in use, a link is backend-specific by nature, so building it
    anywhere else would put a vendor name in a module that must not have one.
    Returns None when unconfigured; callers fall back to naming the session.
    """
    project = os.environ.get("LANGFUSE_PROJECT_ID")
    base = (os.environ.get("LANGFUSE_BASE_URL") or "").strip('"').rstrip("/")
    if not (session_id and project and base):
        return None
    return f"{base}/project/{project}/sessions/{session_id}"


def trace_url(trace_id: str) -> str | None:
    """
    Link to one trace. Here for the same reason as "session_url": a URL is
    backend-specific, and only this module may know which backend is in use.
    """
    project = os.environ.get("LANGFUSE_PROJECT_ID")
    base = (os.environ.get("LANGFUSE_BASE_URL") or "").strip('"').rstrip("/")
    if not (trace_id and project and base):
        return None
    return f"{base}/project/{project}/traces/{trace_id}"


def tracer():
    if not _configured:
        setup()
    return trace.get_tracer(SERVICE_NAME)


@contextmanager
def run_span(name: str = "agent.run", *, session_id: str | None = None,
             tags: tuple[str, ...] | list[str] = (), **attributes):
    """
    One span for a whole run, parenting every tool and model call under it.
    """
    with tracer().start_as_current_span(name) as span:
        if session_id:
            span.set_attribute(SESSION_ID, session_id)
        if tags:
            span.set_attribute(TRACE_TAGS, list(tags))
        for key, value in attributes.items():
            if value is not None:
                span.set_attribute(key, value)
        yield span


def record_usage(span, *, input_tokens: int = 0, output_tokens: int = 0,
                 cost_usd: float = 0.0, response_model: str | None = None) -> None:
    """
    Attach what a call actually consumed.
    Kept here rather than at the call site so the attribute names and the
    knowledge of what a backend reads — stay in this module.
    """
    span.set_attribute(GEN_AI_USAGE_INPUT, input_tokens)
    span.set_attribute(GEN_AI_USAGE_OUTPUT, output_tokens)
    span.set_attribute(GEN_AI_USAGE_COST, cost_usd)
    if response_model:
        span.set_attribute(GEN_AI_RESPONSE_MODEL, response_model)


@contextmanager
def continue_trace(trace_id: str):
    """
    Re-enter an existing trace by id, for work that resumes later.
    An approval can be decided minutes after the run paused, in another thread.
    Without this the executed action lands in an unrelated trace and has to join two of them by hand.
    """
    from opentelemetry.trace import (
        NonRecordingSpan,
        SpanContext,
        TraceFlags,
        set_span_in_context,
    )

    try:
        parent = SpanContext(trace_id=int(trace_id, 16), span_id=1, is_remote=True,
                             trace_flags=TraceFlags(TraceFlags.SAMPLED))
    except (TypeError, ValueError):
        yield None          # not a hex trace id, carry on untethered
        return
    token = context.attach(set_span_in_context(NonRecordingSpan(parent)))
    try:
        yield parent
    finally:
        context.detach(token)


@contextmanager
def tool_span(tool_name: str, **attributes):
    """Span for one tool call."""
    with tracer().start_as_current_span(f"tool.{tool_name}") as span:
        span.set_attribute(GEN_AI_OPERATION, "execute_tool")
        span.set_attribute(GEN_AI_TOOL_NAME, tool_name)
        for key, value in attributes.items():
            if value is not None:
                span.set_attribute(key, value)
        yield span


@contextmanager
def model_span(model: str, **attributes):
    """Span for one model call."""
    with tracer().start_as_current_span(f"chat {model}") as span:
        span.set_attribute(GEN_AI_SYSTEM, "anthropic")
        span.set_attribute(GEN_AI_OPERATION, "chat")
        span.set_attribute(GEN_AI_REQUEST_MODEL, model)
        for key, value in attributes.items():
            if value is not None:
                span.set_attribute(key, value)
        yield span


def flush(timeout_millis: int = 5000) -> None:
    """Force export — call before a short-lived process exits."""
    provider = trace.get_tracer_provider()
    if hasattr(provider, "force_flush"):
        provider.force_flush(timeout_millis)
