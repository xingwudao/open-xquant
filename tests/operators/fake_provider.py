"""Provider fixture implementing only the published operator boundary."""

from __future__ import annotations

from oxq.operators.types import OperatorDiagnostics, OperatorProvenance, OperatorRequest, OperatorResult

IMPLEMENTATION_DIGEST = "sha256:" + "c" * 64


def sma(request: OperatorRequest) -> OperatorResult:
    period = int(request.parameters["period"])
    panel = request.input_panel.sort_values(["code", "date"], kind="stable").copy(deep=True)
    output_name = f"sma_{period}"
    panel[output_name] = panel.groupby("code", sort=False)["close"].transform(
        lambda values: values.rolling(period).mean()
    )
    output = panel[["date", "code", output_name]].sort_values(["date", "code"], kind="stable", ignore_index=True)
    diagnostics = OperatorDiagnostics(
        input_rows=len(request.input_panel),
        output_rows=len(output),
        warmup_rows=max(period - 1, 0) * request.input_panel["code"].nunique(),
    )
    provenance = OperatorProvenance(
        operator_id=request.operator_id,
        operator_version="1.0.0",
        implementation_digest=IMPLEMENTATION_DIGEST,
    )
    return OperatorResult.for_request(
        request,
        data=output,
        diagnostics=diagnostics,
        provenance=provenance,
    )
