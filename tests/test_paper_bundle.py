from __future__ import annotations

import shutil

import polars as pl

from siggeo.config import repo_root
from siggeo.paper.claims import build_claim_evidence_matrix
from siggeo.paper.manuscript import write_manuscript_bundle
from siggeo.paper.release_report import determine_release_decision, write_release_report


def test_paper_bundle_helpers_write_outputs() -> None:
    run_dir = repo_root() / "results" / "test_paper_bundle"
    shutil.rmtree(run_dir, ignore_errors=True)
    (run_dir / "figures").mkdir(parents=True, exist_ok=True)
    (run_dir / "source_data").mkdir(parents=True, exist_ok=True)
    (run_dir / "tables").mkdir(parents=True, exist_ok=True)
    try:
        for name in [
            "figure1_dataset.png",
            "figure2_regime_map.png",
            "figure3_stability.png",
            "figure4_retrieval.png",
            "figure5_enrichment.png",
            "figure6_confounds.png",
        ]:
            (run_dir / "figures" / name).write_bytes(b"png")
        for name in [
            "figure1_dataset.csv",
            "figure2_regime_map.csv",
            "figure3_stability.csv",
            "figure4_retrieval.csv",
            "figure5_enrichment.csv",
            "figure6_confounds.csv",
        ]:
            (run_dir / "source_data" / name).write_text("x,y\n1,2\n", encoding="utf-8")

        claim_matrix = build_claim_evidence_matrix(run_dir)
        paper_dir = run_dir / "paper"
        paper_dir.mkdir(parents=True, exist_ok=True)
        claim_matrix.write_csv(paper_dir / "claim_evidence_matrix.csv")
        write_manuscript_bundle(run_dir, paper_dir, claim_matrix)
        report = write_release_report(run_dir, paper_dir)

        assert (paper_dir / "manuscript.md").exists()
        assert (paper_dir / "figure_manifest.csv").exists()
        assert report["claim_matrix_exists"] is True
        assert report["decision"] == "continue"
        assert claim_matrix["supported"].sum() == 0
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)


def test_negative_result_pivot_requires_c1_support() -> None:
    base_root = repo_root() / "results" / "test_paper_decision"
    shutil.rmtree(base_root, ignore_errors=True)
    try:
        official = base_root / "offv1__featint__platectrl__thrA2_thrR2__z1p96__beta1p0"
        interpretable = base_root / "intv1__featint__platectrl__thrA2_thrR2__z1p96__beta1p0"
        no_center = base_root / "offv1__featint__no_platectrl__thrA2_thrR2__z1p96__beta1p0"
        artifact = base_root / "offv1__featint__platectrl__artifact_sens__thrA2_thrR2__z1p96__beta1p0"
        for run_dir in [official, interpretable, no_center, artifact]:
            (run_dir / "tables").mkdir(parents=True, exist_ok=True)
            pl.DataFrame({"gene_id": ["1"], "regime_label": ["aligned"]}).write_parquet(
                run_dir / "tables" / "same_gene_signed_geometry.parquet"
            )
        pl.DataFrame(
            {
                "pipeline": ["official"],
                "confound_name": ["artifact_sensitivity"],
                "test_name": ["agreement"],
                "statistic_name": ["agreement_rate"],
                "statistic_value": [0.2],
                "p_value": [None],
                "supports_positive_story": [False],
                "notes": ["fragile"],
            }
        ).write_csv(official / "tables" / "confound_summary.csv")

        claim_matrix = pl.DataFrame(
            {
                "claim_id": ["C1", "C2", "C3", "C4"],
                "claim_text": ["", "", "", ""],
                "support_type": ["unsupported", "fragility", "unsupported", "unsupported"],
                "supported": [False, True, False, False],
                "required_artifacts": ["", "", "", ""],
                "evidence_detail": ["", "", "", ""],
            }
        )
        paper_dir = official / "paper"
        paper_dir.mkdir(parents=True, exist_ok=True)

        report = determine_release_decision(official, paper_dir, claim_matrix=claim_matrix)

        assert report["decision"] == "kill"
        assert report["paper_framing"] == "none"
    finally:
        shutil.rmtree(base_root, ignore_errors=True)
