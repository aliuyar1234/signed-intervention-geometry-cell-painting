from __future__ import annotations

import shutil

import polars as pl

from siggeo.config import repo_root
from siggeo.evaluation.enrichment import EnrichmentConfig, run_enrichment


def test_run_enrichment_returns_resource_tables() -> None:
    base_dir = repo_root() / "results" / "test_enrichment"
    shutil.rmtree(base_dir, ignore_errors=True)
    base_dir.mkdir(parents=True, exist_ok=True)
    try:
        geometry = pl.DataFrame(
            {
                "gene_id": ["1", "2", "3", "4"],
                "display_symbol": ["G1", "G2", "G3", "G4"],
                "canonical_symbol": ["G1", "G2", "G3", "G4"],
                "regime_label": ["aligned", "aligned", "inverse", "orthogonal"],
            }
        )
        reactome_path = base_dir / "reactome.gmt"
        reactome_path.write_text(
            "Pathway A\tR-HSA-1\tG1\tG2\tG9\nPathway B\tR-HSA-2\tG3\tG8\n",
            encoding="utf-8",
        )
        corum_path = base_dir / "corum.tsv"
        corum_path.write_text(
            "ComplexID\tComplexName\tsubunits(Gene name)\nC1\tComplex A\tG1;G2\n",
            encoding="utf-8",
        )

        reactome, corum = run_enrichment(
            geometry,
            reactome_path,
            corum_path,
            config=EnrichmentConfig(
                reactome_min_set_size=2,
                corum_min_set_size=2,
                primary_min_regime_size=2,
                fdr_method="fdr_bh",
            ),
        )

        assert reactome.height > 0
        assert corum.height > 0
        assert "q_value" in reactome.columns
    finally:
        shutil.rmtree(base_dir, ignore_errors=True)


def test_run_enrichment_accepts_bh_alias() -> None:
    base_dir = repo_root() / "results" / "test_enrichment_bh_alias"
    shutil.rmtree(base_dir, ignore_errors=True)
    base_dir.mkdir(parents=True, exist_ok=True)
    try:
        geometry = pl.DataFrame(
            {
                "gene_id": ["1", "2", "3", "4"],
                "display_symbol": ["G1", "G2", "G3", "G4"],
                "canonical_symbol": ["G1", "G2", "G3", "G4"],
                "regime_label": ["aligned", "aligned", "inverse", "orthogonal"],
            }
        )
        reactome_path = base_dir / "reactome.gmt"
        reactome_path.write_text(
            "Pathway A\tR-HSA-1\tG1\tG2\tG9\nPathway B\tR-HSA-2\tG3\tG8\n",
            encoding="utf-8",
        )

        reactome, corum = run_enrichment(
            geometry,
            reactome_path,
            None,
            config=EnrichmentConfig(
                reactome_min_set_size=2,
                corum_min_set_size=2,
                primary_min_regime_size=2,
                fdr_method="bh",
            ),
        )

        assert reactome.height > 0
        assert corum.height == 0
        assert "q_value" in reactome.columns
    finally:
        shutil.rmtree(base_dir, ignore_errors=True)


def test_run_enrichment_requires_canonical_symbols_by_default() -> None:
    base_dir = repo_root() / "results" / "test_enrichment_requires_canonical"
    shutil.rmtree(base_dir, ignore_errors=True)
    base_dir.mkdir(parents=True, exist_ok=True)
    try:
        geometry = pl.DataFrame(
            {
                "gene_id": ["1", "2"],
                "display_symbol": ["G1", "G2"],
                "regime_label": ["aligned", "aligned"],
            }
        )
        reactome_path = base_dir / "reactome.gmt"
        reactome_path.write_text("Pathway A\tR-HSA-1\tG1\tG2\n", encoding="utf-8")

        try:
            run_enrichment(
                geometry,
                reactome_path,
                None,
                config=EnrichmentConfig(
                    reactome_min_set_size=2,
                    corum_min_set_size=2,
                    primary_min_regime_size=2,
                    fdr_method="fdr_bh",
                ),
            )
        except KeyError as exc:
            assert "canonical_symbol" in str(exc)
        else:
            raise AssertionError("Expected run_enrichment to reject geometry without canonical_symbol.")
    finally:
        shutil.rmtree(base_dir, ignore_errors=True)
