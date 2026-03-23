from __future__ import annotations

from pathlib import Path

import polars as pl

from siggeo.compute import ComputeConfig
from siggeo.gene_stats.activity import ActivityConfig, compute_gene_modality_stats, recompute_postfilter_eligibility


def test_compute_gene_modality_stats_returns_locked_columns() -> None:
    centered = pl.DataFrame(
        {
            "Metadata_Plate": ["P1", "P1", "P1", "P1", "P2", "P2"],
            "Metadata_Well": ["A01", "A02", "A03", "A04", "A01", "A02"],
            "Metadata_Batch": ["B1", "B1", "B1", "B1", "B2", "B2"],
            "Metadata_NCBI_Gene_ID": ["101", "101", "202", "202", None, None],
            "is_negative_control": [False, False, False, False, True, True],
            "feat1": [3.0, 3.5, 0.1, 0.2, 0.0, 0.0],
            "feat2": [2.0, 2.5, 0.1, 0.2, 0.0, 0.0],
        }
    )
    overlap = pl.DataFrame(
        {
            "gene_id": ["101", "202"],
            "display_symbol": ["GENEA", "GENEB"],
            "low_replicate_support": [True, True],
        }
    )

    stats = compute_gene_modality_stats(
        centered,
        overlap,
        ["feat1", "feat2"],
        pipeline="official",
        modality="orf",
        config=ActivityConfig(
            n_strength_null_samples=8,
            n_rep_null_samples=8,
            strength_z_threshold=2.0,
            rep_z_threshold=2.0,
            random_seed=1337,
        ),
    )

    assert stats.columns[:5] == ["gene_id", "display_symbol", "canonical_symbol", "modality", "pipeline"]
    assert set(stats["gene_id"].to_list()) == {"101", "202"}
    assert "canonical_symbol" in stats.columns
    assert "strength_z" in stats.columns
    assert "rep_z" in stats.columns
    assert "active_bool" in stats.columns


def test_compute_gene_modality_stats_respects_keep_for_analysis() -> None:
    centered = pl.DataFrame(
        {
            "Metadata_Plate": ["P1", "P1", "P1", "P1", "P2", "P2"],
            "Metadata_Well": ["A01", "A02", "A03", "A04", "A01", "A02"],
            "Metadata_Batch": ["B1", "B1", "B1", "B1", "B2", "B2"],
            "Metadata_NCBI_Gene_ID": ["101", "101", "202", "202", None, None],
            "is_negative_control": [False, False, False, False, True, True],
            "feat1": [3.0, 3.5, 0.1, 0.2, 0.0, 0.0],
            "feat2": [2.0, 2.5, 0.1, 0.2, 0.0, 0.0],
        }
    )
    overlap = pl.DataFrame(
        {
            "gene_id": ["101", "202"],
            "display_symbol": ["GENEA", "GENEB"],
            "low_replicate_support": [True, True],
            "keep_for_analysis": [True, False],
        }
    )

    stats = compute_gene_modality_stats(
        centered,
        overlap,
        ["feat1", "feat2"],
        pipeline="official",
        modality="orf",
        config=ActivityConfig(
            n_strength_null_samples=8,
            n_rep_null_samples=8,
            strength_z_threshold=2.0,
            rep_z_threshold=2.0,
            random_seed=1337,
        ),
    )

    assert stats["gene_id"].to_list() == ["101"]


def test_compute_gene_modality_stats_is_stable_under_row_shuffle() -> None:
    centered = pl.DataFrame(
        {
            "Metadata_Plate": ["P1", "P1", "P2", "P2", "P1", "P1", "P2", "P2"],
            "Metadata_Well": ["A01", "A02", "A03", "A04", "B01", "B02", "B03", "B04"],
            "Metadata_Batch": ["B1"] * 8,
            "Metadata_NCBI_Gene_ID": ["101", "101", "101", "101", None, None, None, None],
            "is_negative_control": [False, False, False, False, True, True, True, True],
            "feat1": [1.0, 0.9, 1.1, 1.0, 0.0, 0.0, 0.0, 0.0],
            "feat2": [0.0, 0.1, -0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
        }
    )
    overlap = pl.DataFrame(
        {
            "gene_id": ["101"],
            "display_symbol": ["GENEA"],
            "canonical_symbol": ["GENEA"],
            "low_replicate_support": [False],
            "keep_for_analysis": [True],
        }
    )
    config = ActivityConfig(
        n_strength_null_samples=20,
        n_rep_null_samples=20,
        strength_z_threshold=2.0,
        rep_z_threshold=2.0,
        random_seed=1337,
    )

    first = compute_gene_modality_stats(centered, overlap, ["feat1", "feat2"], pipeline="official", modality="orf", config=config)
    second = compute_gene_modality_stats(
        centered.sample(fraction=1.0, shuffle=True, seed=42),
        overlap,
        ["feat1", "feat2"],
        pipeline="official",
        modality="orf",
        config=config,
    )

    assert first.to_dicts() == second.to_dicts()


def test_recompute_postfilter_eligibility_drops_genes_below_retained_well_threshold() -> None:
    overlap = pl.DataFrame(
        {
            "gene_id": ["1", "2"],
            "display_symbol": ["GENEA", "GENEB"],
            "canonical_symbol": ["GENEA", "GENEB"],
            "low_replicate_support": [False, False],
            "keep_for_analysis": [True, True],
        }
    )
    centered_orf = pl.DataFrame(
        {
            "Metadata_NCBI_Gene_ID": ["1", "1", "2"],
            "Metadata_Plate": ["P1", "P2", "P1"],
            "Metadata_Well": ["A01", "A02", "A03"],
            "Metadata_Batch": ["B1", "B1", "B1"],
            "is_negative_control": [False, False, False],
            "f1": [1.0, 1.1, 0.5],
            "f2": [0.0, 0.1, 0.5],
        }
    )
    centered_crispr = pl.DataFrame(
        {
            "Metadata_NCBI_Gene_ID": ["1", "1", "2", "2"],
            "Metadata_Plate": ["Q1", "Q2", "Q1", "Q2"],
            "Metadata_Well": ["A01", "A02", "A03", "A04"],
            "Metadata_Batch": ["B2", "B2", "B2", "B2"],
            "is_negative_control": [False, False, False, False],
            "f1": [1.0, 1.0, 0.5, 0.6],
            "f2": [0.0, 0.0, 0.5, 0.6],
        }
    )

    updated, qc = recompute_postfilter_eligibility(
        overlap,
        centered_orf,
        centered_crispr,
        min_wells_per_modality=2,
    )

    kept = {row["gene_id"]: row["postfilter_keep_for_analysis"] for row in qc.to_dicts()}
    assert kept == {"1": True, "2": False}
    assert updated.filter(pl.col("keep_for_analysis"))["gene_id"].to_list() == ["1"]


def test_compute_gene_modality_stats_parallel_matches_serial(tmp_path: Path) -> None:
    centered = pl.DataFrame(
        {
            "Metadata_Plate": ["P1", "P1", "P2", "P2", "P1", "P1", "P2", "P2", "P3", "P3"],
            "Metadata_Well": ["A01", "A02", "A03", "A04", "B01", "B02", "B03", "B04", "C01", "C02"],
            "Metadata_Batch": ["B1", "B1", "B1", "B1", "B2", "B2", "B2", "B2", "B3", "B3"],
            "Metadata_NCBI_Gene_ID": ["101", "101", "202", "202", "303", "303", None, None, None, None],
            "is_negative_control": [False, False, False, False, False, False, True, True, True, True],
            "feat1": [3.0, 3.5, 0.1, 0.2, 1.1, 1.0, 0.0, 0.0, 0.0, 0.0],
            "feat2": [2.0, 2.5, 0.1, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
        }
    )
    overlap = pl.DataFrame(
        {
            "gene_id": ["101", "202", "303"],
            "display_symbol": ["GENEA", "GENEB", "GENEC"],
            "canonical_symbol": ["GENEA", "GENEB", "GENEC"],
            "low_replicate_support": [False, False, False],
            "keep_for_analysis": [True, True, True],
        }
    )
    activity_config = ActivityConfig(
        n_strength_null_samples=8,
        n_rep_null_samples=8,
        strength_z_threshold=2.0,
        rep_z_threshold=2.0,
        random_seed=1337,
    )
    serial = compute_gene_modality_stats(
        centered,
        overlap,
        ["feat1", "feat2"],
        pipeline="official",
        modality="orf",
        config=activity_config,
    )
    parallel = compute_gene_modality_stats(
        centered,
        overlap,
        ["feat1", "feat2"],
        pipeline="official",
        modality="orf",
        config=activity_config,
        compute_config=ComputeConfig(
            parallel_backend="process_pool",
            max_workers=2,
            phase4_workers=2,
            phase5_workers=1,
            phase8_workers=1,
            phase4_gene_chunk_size=2,
            phase5_gene_chunk_size=1,
            phase8_gene_chunk_size=1,
            numpy_threads_per_worker=1,
            parallelize_modalities=False,
            mmap_root=tmp_path / "memmaps",
            temp_root=tmp_path / "tmp",
            max_concurrent_runs=2,
        ),
        run_dir=tmp_path,
    )

    assert parallel.to_dicts() == serial.to_dicts()
