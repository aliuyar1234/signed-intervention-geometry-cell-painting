from __future__ import annotations

from pathlib import Path

import polars as pl

from siggeo.io.atomic import write_csv_atomic, write_text_atomic
from siggeo.paper.release_report import determine_release_decision


def _supported_claim_lines(claim_matrix: pl.DataFrame) -> list[str]:
    supported = claim_matrix.filter(pl.col("supported"))
    if supported.height == 0:
        return ["No main-text claim is yet supported by the locked evidence rules."]
    return [f"- {row['claim_id']}: {row['claim_text']}" for row in supported.to_dicts()]


def write_manuscript_bundle(run_dir: Path, paper_dir: Path, claim_matrix: pl.DataFrame) -> None:
    paper_dir.mkdir(parents=True, exist_ok=True)
    decision = determine_release_decision(run_dir, paper_dir, claim_matrix=claim_matrix)
    decision_mode = str(decision["paper_framing"])
    if decision_mode == "positive":
        abstract_lines = [
            "We define and quantify a signed intervention geometry for same-gene ORF and CRISPR perturbations within cpg0016.",
            "The current locked evidence bundle supports a restrained positive framing centered on a statistical regime map.",
        ]
    elif decision_mode == "negative_result":
        abstract_lines = [
            "We define and quantify a signed intervention geometry for same-gene ORF and CRISPR perturbations within cpg0016.",
            "The current locked evidence bundle supports a negative-result framing in which preprocessing fragility is itself the scientific finding.",
        ]
    else:
        abstract_lines = [
            "We define and quantify a signed intervention geometry for same-gene ORF and CRISPR perturbations within cpg0016.",
            "The current bundle is incomplete or not yet strong enough to support a paper-level claim.",
        ]
    manuscript = paper_dir / "manuscript.md"
    write_text_atomic(
        manuscript,
        "\n".join(
            [
                "# Signed Intervention Geometry of Gain- and Loss-of-Function Perturbations in Genome-Scale Cell Painting",
                "",
                f"Decision mode: `{decision_mode}`",
                "",
                "## Abstract",
                *abstract_lines,
                "Claims in this draft are intentionally constrained to the locked evidence bundle.",
                "",
                "## Evidence status",
                "See `claim_evidence_matrix.csv` for claim-by-claim support.",
                "",
                "### Supported claims",
                *_supported_claim_lines(claim_matrix),
                "",
                "### Decision rationale",
                str(decision["rationale"]),
                "",
                "## Limitations",
                "- cpg0016-specific scope",
                "- residual modality/source confounding risk",
                "- no wet-lab validation",
                "- dependence on public preprocessing choices",
            ]
        )
        + "\n",
    )
    write_text_atomic(
        paper_dir / "limitations_checklist.md",
        "\n".join(
            [
                "# Limitations Checklist",
                "",
                "- [x] cpg0016-specific scope",
                "- [x] residual modality/source confounding risk",
                "- [x] no wet-lab validation",
                "- [x] no mechanistic causality claims",
                "- [x] preprocessing dependence acknowledged",
            ]
        )
        + "\n",
    )
    write_text_atomic(
        paper_dir / "decision_summary.md",
        "\n".join(
            [
                "# Decision Summary",
                "",
                f"Decision: `{decision['decision']}`",
                f"Paper framing: `{decision_mode}`",
                "",
                f"Rationale: {decision['rationale']}",
                "",
                "Supported claims: " + (", ".join(decision["supported_claim_ids"]) if decision["supported_claim_ids"] else "none"),
            ]
        )
        + "\n",
    )
    figure_manifest = pl.DataFrame(
        {
            "figure_name": sorted(path.name for path in (run_dir / "figures").glob("figure*.png")),
            "source_data_name": sorted(path.name for path in (run_dir / "source_data").glob("figure*.csv")),
        }
    )
    write_csv_atomic(figure_manifest, paper_dir / "figure_manifest.csv")
