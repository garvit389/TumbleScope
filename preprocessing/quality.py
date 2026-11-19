"""Phase 0 quality placeholder."""

from core.schemas import PreparedObservation, QualityReport


def assess_stage1(prep: PreparedObservation, cfg: dict) -> QualityReport:
    return QualityReport(
        status="suitable",
        metrics={},
        warnings=["Synthetic fixture. Stage 1 scoring is a placeholder."],
        reason_codes=[],
        is_stub=True,
    )
