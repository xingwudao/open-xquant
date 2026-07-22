"""Tracer — component-level execution tracing for the engine pipeline."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone; UTC = timezone.utc  # py3.9 compat
from typing import Any


@dataclass(frozen=True)
class TraceSpan:
    """A single component execution record.

    Captures inputs, output summary, timing, and status for one
    component (indicator, signal, or rule) during an engine run.
    """

    trace_id: str
    span_id: str
    parent_id: str | None
    component: str
    inputs: dict[str, Any]
    output_summary: dict[str, Any]
    started_at: str
    ended_at: str
    duration_ms: float
    status: str  # "ok" | "error" | "skipped"
    error: str | None


class DefaultTracer:
    """Default tracer implementation — collects TraceSpans in memory."""

    def __init__(self) -> None:
        self._trace_id: str = ""
        self._config: dict[str, Any] = {}
        self._spans: list[TraceSpan] = []
        self._run_status: str = ""

    @property
    def spans(self) -> list[TraceSpan]:
        return list(self._spans)

    @property
    def trace_id(self) -> str:
        return self._trace_id

    @property
    def run_status(self) -> str:
        return self._run_status

    def on_run_start(self, strategy_name: str, config: dict[str, Any]) -> str:
        self._trace_id = uuid.uuid4().hex[:12]
        self._config = config
        self._spans = []
        self._run_status = ""
        return self._trace_id

    def on_indicator(
        self,
        name: str,
        params: dict[str, Any],
        output_summary: dict[str, Any],
        duration_ms: float,
        status: str = "ok",
        error: str | None = None,
    ) -> None:
        now = datetime.now(tz=UTC).isoformat()
        self._spans.append(
            TraceSpan(
                trace_id=self._trace_id,
                span_id=uuid.uuid4().hex[:12],
                parent_id=None,
                component=f"indicator:{name}",
                inputs=params,
                output_summary=output_summary,
                started_at=now,
                ended_at=now,
                duration_ms=duration_ms,
                status=status,
                error=error,
            ),
        )

    def on_signal(
        self,
        name: str,
        inputs: dict[str, Any],
        output_summary: dict[str, Any],
        duration_ms: float,
        status: str = "ok",
        error: str | None = None,
    ) -> None:
        now = datetime.now(tz=UTC).isoformat()
        self._spans.append(
            TraceSpan(
                trace_id=self._trace_id,
                span_id=uuid.uuid4().hex[:12],
                parent_id=None,
                component=f"signal:{name}",
                inputs=inputs,
                output_summary=output_summary,
                started_at=now,
                ended_at=now,
                duration_ms=duration_ms,
                status=status,
                error=error,
            ),
        )

    def on_rule(
        self,
        name: str,
        rule_type: str,
        output_summary: dict[str, Any],
        duration_ms: float,
        status: str = "ok",
        error: str | None = None,
    ) -> None:
        now = datetime.now(tz=UTC).isoformat()
        self._spans.append(
            TraceSpan(
                trace_id=self._trace_id,
                span_id=uuid.uuid4().hex[:12],
                parent_id=None,
                component=f"rule:{name}",
                inputs={"rule_type": rule_type},
                output_summary=output_summary,
                started_at=now,
                ended_at=now,
                duration_ms=duration_ms,
                status=status,
                error=error,
            ),
        )

    def on_run_end(self, status: str) -> None:
        self._run_status = status
