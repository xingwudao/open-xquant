"""AuditRecord — complete run snapshot for reproducibility verification."""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone; UTC = timezone.utc  # py3.9 compat
from pathlib import Path
from typing import Any

from oxq.observe.hashing import (
    combined_hash,
    hash_equity,
    hash_mktdata,
    hash_trades,
)
from oxq.observe.tracer import DefaultTracer, TraceSpan
from oxq.portfolio.analytics import RunResult


@dataclass(frozen=True)
class AuditRecord:
    """A complete run snapshot for reproducibility and audit.

    Contains strategy configuration, trace spans, and layered hashes
    for determinism verification.
    """

    run_id: str
    strategy_name: str
    strategy_config: dict[str, Any]
    start_date: str
    end_date: str
    initial_cash: float
    trace_spans: tuple[TraceSpan, ...]
    mktdata_hash: str
    trades_hash: str
    equity_hash: str
    result_hash: str
    created_at: str

    @classmethod
    def build(
        cls,
        tracer: DefaultTracer,
        result: RunResult,
        strategy_name: str,
        strategy_config: dict[str, Any],
        start_date: str,
        end_date: str,
        initial_cash: float,
    ) -> AuditRecord:
        """Build an AuditRecord from a tracer and run result."""
        mh = hash_mktdata(result.mktdata)
        th = hash_trades(result.trades)
        eh = hash_equity(result.equity_curve)
        rh = combined_hash(mh, th, eh)
        return cls(
            run_id=f"run_{uuid.uuid4().hex[:12]}",
            strategy_name=strategy_name,
            strategy_config=strategy_config,
            start_date=start_date,
            end_date=end_date,
            initial_cash=initial_cash,
            trace_spans=tuple(tracer.spans),
            mktdata_hash=mh,
            trades_hash=th,
            equity_hash=eh,
            result_hash=rh,
            created_at=datetime.now(tz=UTC).isoformat(),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict suitable for JSON."""
        d = asdict(self)
        d["trace_spans"] = [asdict(s) for s in self.trace_spans]
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> AuditRecord:
        """Restore an AuditRecord from a dict."""
        spans = tuple(TraceSpan(**s) for s in d.pop("trace_spans", []))
        return cls(trace_spans=spans, **d)

    def to_json(self, path: str) -> None:
        """Write to a JSON file."""
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, default=str))

    @classmethod
    def from_json(cls, path: str) -> AuditRecord:
        """Read from a JSON file."""
        data = json.loads(Path(path).read_text())
        return cls.from_dict(data)
