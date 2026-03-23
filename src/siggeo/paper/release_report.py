from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl

from siggeo.io.atomic import write_json_atomic, write_text_atomic
from siggeo.paper.claims import (
    ARTIFACT_RUN_NAME,
    INTERPRETABLE_RUN_NAME,
    NO_CENTER_RUN_NAME,
    OFFICIAL_RUN_NAME,
    build_claim_evidence_matrix,
    summarize_claim_matrix,
)


def _required_run_artifacts(root: Path) -> dict[str, Path]:
    return {
        OFFICIAL_RUN_NAME: root / OFFICIAL_RUN_NAME / "tables" / "same_gene_signed_geometry.parquet",
        INTERPRETABLE_RUN_NAME: root / INTERPRETABLE_RUN_NAME / "tables" / "same_gene_signed_geometry.parquet",
        NO_CENTER_RUN_NAME: root / NO_CENTER_RUN_NAME / "tables" / "same_gene_signed_geometry.parquet",
        ARTIFACT_RUN_NAME: root / ARTIFACT_RUN_NAME / "tables" / "same_gene_signed_geometry.parquet",
    }


def determine_release_decision(
    run_dir: Path,
    paper_dir: Path,
    *,
    claim_matrix: pl.DataFrame | None = None,
) -> dict[str, Any]:
    claim_frame = claim_matrix
    claim_matrix_path = paper_dir / "claim_evidence_matrix.csv"
    if claim_frame is None:
        if claim_matrix_path.exists():
            claim_frame = pl.read_csv(claim_matrix_path)
        else:
            claim_frame = build_claim_evidence_matrix(run_dir)
    claim_summary = summarize_claim_matrix(claim_frame)

    required_runs = _required_run_artifacts(run_dir.parent)
    missing_runs = [name for name, artifact in required_runs.items() if not artifact.exists()]
    official_confound_ready = (run_dir.parent / OFFICIAL_RUN_NAME / "tables" / "confound_summary.csv").exists()
    if run_dir.name == OFFICIAL_RUN_NAME and not (run_dir / "tables" / "confound_summary.csv").exists():
        official_confound_ready = False

    if run_dir.name.startswith("smoke__"):
        decision = "continue"
        paper_framing = "none"
        rationale = "Smoke runs are deterministic validation artifacts and do not authorize paper claims."
    elif missing_runs or not official_confound_ready:
        decision = "continue"
        paper_framing = "none"
        rationale = "Mandatory robustness runs or official confound outputs are still missing."
    elif claim_summary["positive_ready"]:
        decision = "continue"
        paper_framing = "positive"
        rationale = "C1, C2-positive, and C3 are supported without confound defeat."
    elif claim_summary["fragility_ready"]:
        decision = "pivot"
        paper_framing = "negative_result"
        rationale = "C1 is supported, but the locked runs quantify a fragility story more clearly than a positive one."
    else:
        decision = "kill"
        paper_framing = "none"
        rationale = "Neither the positive story nor the quantified fragility fallback is defensible."

    return {
        "decision": decision,
        "paper_framing": paper_framing,
        "rationale": rationale,
        "missing_runs": missing_runs,
        "official_confound_ready": official_confound_ready,
        "supported_claim_ids": claim_summary["supported_claim_ids"],
        "claim_support": {
            claim_id: payload["support_type"] for claim_id, payload in claim_summary["claims"].items()
        },
        "retrieval_supported": claim_summary["retrieval_supported"],
    }


def write_release_report(run_dir: Path, paper_dir: Path) -> dict[str, object]:
    figures = sorted(path.name for path in (run_dir / "figures").glob("figure*.png"))
    tables = sorted(path.name for path in (run_dir / "tables").glob("*"))
    claim_matrix = (
        pl.read_csv(paper_dir / "claim_evidence_matrix.csv")
        if (paper_dir / "claim_evidence_matrix.csv").exists()
        else build_claim_evidence_matrix(run_dir)
    )
    decision = determine_release_decision(run_dir, paper_dir, claim_matrix=claim_matrix)
    report = {
        "run_dir": str(run_dir),
        "paper_dir": str(paper_dir),
        "figure_count": len(figures),
        "table_count": len(tables),
        "manuscript_exists": (paper_dir / "manuscript.md").exists(),
        "claim_matrix_exists": (paper_dir / "claim_evidence_matrix.csv").exists(),
        **decision,
    }
    write_json_atomic(run_dir / "release_report.json", report)
    if report["decision"] == "kill":
        write_text_atomic(
            paper_dir / "kill_decision.md",
            "\n".join(
                [
                    "# Kill Decision",
                    "",
                    f"Decision: `{report['decision']}`",
                    "",
                    f"Rationale: {report['rationale']}",
                    "",
                    "## Missing runs",
                    ", ".join(report["missing_runs"]) if report["missing_runs"] else "none",
                    "",
                    "## Supported claims",
                    ", ".join(report["supported_claim_ids"]) if report["supported_claim_ids"] else "none",
                ]
            )
            + "\n",
        )
    return report
