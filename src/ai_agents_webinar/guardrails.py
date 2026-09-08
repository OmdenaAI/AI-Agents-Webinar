"""
Guardrails on untrusted content.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Literal

# prompt injection heuristics
# Each pattern matches a phrase that has no legitimate reason to appear in
# project data but is characteristic of an instruction aimed at a model.
INJECTION_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("override", re.compile(r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions", re.I)),
    ("override", re.compile(r"disregard\s+(all\s+)?(previous|prior|your)\s+", re.I)),
    ("role_change", re.compile(r"you\s+are\s+now\s+(in\s+)?\w+\s*mode", re.I)),
    ("role_change", re.compile(r"act\s+as\s+(a\s+)?(system|admin|developer)", re.I)),
    ("exfiltration", re.compile(r"\b(post|send|export|upload|exfiltrate)\b[^.]{0,60}\bhttps?://", re.I)),
    ("remote_fetch", re.compile(r"\b(retrieve|fetch|download|pull|load)\b[^.]{0,60}\bhttps?://", re.I)),
    ("concealment", re.compile(r"do\s+not\s+(mention|tell|reveal|report)", re.I)),
    ("secret_seeking", re.compile(r"\b(api[_\s-]?key|password|secret|token|credential)s?\b[^.]{0,30}\b(show|print|reveal|send)", re.I)),
]

# PII patterns
PII_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("email", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    ("phone", re.compile(r"(?<!\d)(?:\+\d{1,3}[\s.-]?)?(?:\(\d{3}\)|\d{3})[\s.-]?\d{3}[\s.-]?\d{4}(?!\d)")),
    ("government_id", re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)")),
]

# When deployed in production, we use encryption modules such as Fernet for redaction
REDACTION = {"email": "[REDACTED:email]", "phone": "[REDACTED:phone]",
             "government_id": "[REDACTED:gov-id]"}

@dataclass(frozen=True)
class Finding:
    kind: Literal["injection", "pii"]
    category: str
    excerpt: str

    def as_event(self) -> dict:
        return {"event": "guardrail_flag", "kind": self.kind,
                "category": self.category, "excerpt": self.excerpt}


@dataclass(frozen=True)
class ScanResult:
    findings: tuple[Finding, ...]
    redacted: str

    @property
    def flagged(self) -> bool:
        return bool(self.findings)

    @property
    def has_injection(self) -> bool:
        return any(f.kind == "injection" for f in self.findings)


def _window(text: str, match: re.Match, size: int = 30) -> str:
    """A short surrounding excerpt, for explaining the flag without dumping data."""
    start = max(0, match.start() - size)
    end = min(len(text), match.end() + size)
    return ("…" if start else "") + text[start:end].strip() + ("…" if end < len(text) else "")


def scan(text: str) -> ScanResult:
    """
    Scan tool-returned content before it reaches the model, trace or UI.

    Returns findings plus a redacted copy. 
    PII is replaced and injection text is left intact because the flag is what matters.
    """
    if not text:
        return ScanResult((), text)

    findings: list[Finding] = []

    for category, pattern in INJECTION_PATTERNS:
        for match in pattern.finditer(text):
            findings.append(Finding("injection", category, _window(text, match)))

    redacted = text
    for category, pattern in PII_PATTERNS:
        for match in pattern.finditer(text):
            findings.append(Finding("pii", category, f"<{category} in tool result>"))
        redacted = pattern.sub(REDACTION[category], redacted)

    return ScanResult(tuple(findings), redacted)


def scan_result(payload) -> ScanResult:
    """Scan an arbitrary tool result by flattening it to text first."""
    text = payload if isinstance(payload, str) else json.dumps(payload, default=str)
    return scan(text)
