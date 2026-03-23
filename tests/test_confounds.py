from __future__ import annotations

import polars as pl

from siggeo.compute import load_compute_config
from siggeo.evaluation.confounds import ConfoundConfig, compare_geometry_tables, run_confounds


def test_run_confounds_returns_expected_tables() -> None:
    geometry = pl.DataFrame(
        {
            "gene_id": ["1", "2", "3"],
            "display_symbol": ["G1", "G2", "G3"],
            "pipeline": ["official", "official", "official"],
            "orf_strength_z": [3.0, 2.5, 2.1],
            "crispr_strength_z": [3.0, 2.0, 2.2],
            "orf_rep_z": [3.0, 3.0, 2.5],
            "crispr_rep_z": [3.0, 3.0, 2.5],
            "orf_active": [True, True, True],
            "crispr_active": [True, True, True],
            "delta_activity": [0.0, 0.5, -0.1],
            "raw_signed_cosine": [0.9, -0.8, 0.1],
            "sign_null_orf_bin": [1, 1, 1],
            "sign_null_crispr_bin": [1, 1, 1],
            "sign_null_fallback_level": ["global", "global", "global"],
            "sign_null_mean": [0.0, 0.0, 0.0],
            "sign_null_std": [1.0, 1.0, 1.0],
            "sign_z": [0.9, -0.8, 0.1],
            "regime_label_prebootstrap": ["orthogonal", "orthogonal", "orthogonal"],
            "bootstrap_prob_aligned": [0.2, 0.2, 0.2],
            "bootstrap_prob_inverse": [0.2, 0.2, 0.2],
            "bootstrap_prob_orthogonal": [0.6, 0.6, 0.6],
            "bootstrap_prob_asym_orf": [0.0, 0.0, 0.0],
            "bootstrap_prob_asym_crispr": [0.0, 0.0, 0.0],
            "bootstrap_prob_inactive": [0.0, 0.0, 0.0],
            "bootstrap_consensus": [0.6, 0.6, 0.6],
            "bootstrap_consensus_label": ["orthogonal", "orthogonal", "orthogonal"],
            "prebootstrap_matches_consensus": [True, True, True],
            "regime_label": ["orthogonal", "orthogonal", "orthogonal"],
        }
    )
    gene_stats = pl.DataFrame(
        {
            "gene_id": ["1", "2", "3", "1", "2", "3"],
            "display_symbol": ["G1", "G2", "G3", "G1", "G2", "G3"],
            "modality": ["orf", "orf", "orf", "crispr", "crispr", "crispr"],
            "pipeline": ["official"] * 6,
            "n_wells": [2] * 6,
            "n_plates": [2, 2, 2, 2, 2, 2],
            "centroid_l2": [1.0] * 6,
            "centroid_vector": [[1.0, 0.0]] * 6,
            "strength_z": [3.0, 2.5, 2.1, 3.0, 2.0, 2.2],
            "rep_similarity": [1.0] * 6,
            "rep_z": [3.0] * 6,
            "active_bool": [True] * 6,
            "null_strength_median": [0.0] * 6,
            "null_strength_mad": [1.0] * 6,
            "null_rep_median": [0.0] * 6,
            "null_rep_mad": [1.0] * 6,
            "low_replicate_support": [True] * 6,
        }
    )
    centered_orf = pl.DataFrame(
        {
            "Metadata_NCBI_Gene_ID": ["1", "1", "2", "2", "3", "3"],
            "Metadata_Plate": ["P1", "P2", "P1", "P2", "P1", "P2"],
            "Metadata_Well": ["A01", "A02", "A03", "A04", "A05", "A06"],
            "Metadata_Batch": ["B1"] * 6,
            "is_negative_control": [False] * 6,
            "f1": [1.0, 1.1, 0.0, 0.0, 1.0, 1.1],
            "f2": [0.0, 0.1, 1.0, 1.1, 1.0, 1.1],
        }
    )
    centered_crispr = pl.DataFrame(
        {
            "Metadata_NCBI_Gene_ID": ["1", "1", "2", "2", "3", "3"],
            "Metadata_Plate": ["Q1", "Q2", "Q1", "Q2", "Q1", "Q2"],
            "Metadata_Well": ["A01", "A02", "A03", "A04", "A05", "A06"],
            "Metadata_Batch": ["B2"] * 6,
            "is_negative_control": [False] * 6,
            "f1": [1.0, 1.1, 0.0, 0.0, 1.0, 1.1],
            "f2": [0.0, 0.1, -1.0, -1.1, 0.1, 0.2],
        }
    )

    confounds, prevalence = run_confounds(
        geometry,
        gene_stats,
        centered_orf,
        centered_crispr,
        ["f1", "f2"],
        config=ConfoundConfig(
            strength_threshold=2.0,
            rep_threshold=2.0,
            sign_z_threshold=1.96,
            asymmetry_threshold=1.0,
            null_std_floor=1.0e-6,
            random_seed=1337,
            n_different_gene_samples=20,
        ),
    )

    assert confounds.height >= 4
    assert prevalence.height >= 1
    assert "confound_name" in confounds.columns
    assert "regime_label" in prevalence.columns


def test_compare_geometry_tables_outputs_schema() -> None:
    base = pl.DataFrame({"gene_id": ["1", "2"], "regime_label": ["aligned", "inverse"]})
    other = pl.DataFrame({"gene_id": ["1", "2"], "regime_label": ["aligned", "orthogonal"]})
    agreement = compare_geometry_tables(base, other, "baseline_vs_other")
    assert agreement.height == 1
    assert "overall_agreement" in agreement.columns
    assert "overall_agreement_primary_only" in agreement.columns
    assert "has_real_comparison" in agreement.columns
    assert "asymmetric_orf_agreement" in agreement.columns
    assert "asymmetric_crispr_agreement" in agreement.columns


def test_run_confounds_handles_distinct_cell_count_column_names() -> None:
    geometry = pl.DataFrame(
        {
            "gene_id": ["1", "2"],
            "display_symbol": ["G1", "G2"],
            "canonical_symbol": ["G1", "G2"],
            "pipeline": ["official", "official"],
            "orf_strength_z": [3.0, 3.0],
            "crispr_strength_z": [3.0, 3.0],
            "orf_rep_z": [3.0, 3.0],
            "crispr_rep_z": [3.0, 3.0],
            "orf_active": [True, True],
            "crispr_active": [True, True],
            "delta_activity": [0.0, 0.0],
            "raw_signed_cosine": [0.9, 0.8],
            "sign_null_orf_bin": [1, 1],
            "sign_null_crispr_bin": [1, 1],
            "sign_null_fallback_level": ["global", "global"],
            "sign_null_mean": [0.0, 0.0],
            "sign_null_std": [1.0, 1.0],
            "sign_z": [2.0, 2.0],
            "regime_label_prebootstrap": ["aligned", "aligned"],
            "bootstrap_prob_aligned": [1.0, 1.0],
            "bootstrap_prob_inverse": [0.0, 0.0],
            "bootstrap_prob_orthogonal": [0.0, 0.0],
            "bootstrap_prob_asym_orf": [0.0, 0.0],
            "bootstrap_prob_asym_crispr": [0.0, 0.0],
            "bootstrap_prob_inactive": [0.0, 0.0],
            "bootstrap_consensus": [1.0, 1.0],
            "bootstrap_consensus_label": ["aligned", "aligned"],
            "prebootstrap_matches_consensus": [True, True],
            "regime_label": ["aligned", "aligned"],
        }
    )
    gene_stats = pl.DataFrame(
        {
            "gene_id": ["1", "2", "1", "2"],
            "display_symbol": ["G1", "G2", "G1", "G2"],
            "canonical_symbol": ["G1", "G2", "G1", "G2"],
            "modality": ["orf", "orf", "crispr", "crispr"],
            "pipeline": ["official"] * 4,
            "n_wells": [2] * 4,
            "n_plates": [1] * 4,
            "centroid_l2": [1.0] * 4,
            "centroid_vector": [[1.0, 0.0]] * 4,
            "strength_z": [3.0] * 4,
            "rep_similarity": [1.0] * 4,
            "rep_z": [3.0] * 4,
            "active_bool": [True] * 4,
            "null_strength_median": [0.0] * 4,
            "null_strength_mad": [1.0] * 4,
            "null_rep_median": [0.0] * 4,
            "null_rep_mad": [1.0] * 4,
            "low_replicate_support": [False] * 4,
        }
    )
    centered_orf = pl.DataFrame(
        {
            "Metadata_NCBI_Gene_ID": ["1", "1", "2", "2"],
            "Metadata_Plate": ["P1", "P1", "P1", "P1"],
            "Metadata_Well": ["A01", "A02", "A03", "A04"],
            "Metadata_Batch": ["B1"] * 4,
            "is_negative_control": [False] * 4,
            "Image_Count_Cells": [10, 11, 20, 21],
            "f1": [1.0, 1.0, 1.0, 1.0],
            "f2": [0.0, 0.0, 0.0, 0.0],
        }
    )
    centered_crispr = pl.DataFrame(
        {
            "Metadata_NCBI_Gene_ID": ["1", "1", "2", "2"],
            "Metadata_Plate": ["Q1", "Q1", "Q1", "Q1"],
            "Metadata_Well": ["A01", "A02", "A03", "A04"],
            "Metadata_Batch": ["B2"] * 4,
            "is_negative_control": [False] * 4,
            "Metadata_Count_Cells": [12, 13, 22, 23],
            "f1": [1.0, 1.0, 1.0, 1.0],
            "f2": [0.0, 0.0, 0.0, 0.0],
        }
    )

    confounds, _ = run_confounds(
        geometry,
        gene_stats,
        centered_orf,
        centered_crispr,
        ["f1", "f2"],
        config=ConfoundConfig(
            strength_threshold=2.0,
            rep_threshold=2.0,
            sign_z_threshold=1.96,
            asymmetry_threshold=1.0,
            null_std_floor=1.0e-6,
            random_seed=1337,
            n_different_gene_samples=10,
        ),
    )

    notes = confounds.filter(pl.col("confound_name") == "cell_count_availability")["notes"][0]
    assert "orf:Image_Count_Cells" in notes
    assert "crispr:Metadata_Count_Cells" in notes


def test_run_confounds_parallel_matches_serial(tmp_path) -> None:
    geometry = pl.DataFrame(
        {
            "gene_id": ["1", "2", "3"],
            "display_symbol": ["G1", "G2", "G3"],
            "canonical_symbol": ["G1", "G2", "G3"],
            "pipeline": ["official", "official", "official"],
            "orf_strength_z": [3.0, 2.5, 2.1],
            "crispr_strength_z": [3.0, 2.0, 2.2],
            "orf_rep_z": [3.0, 3.0, 2.5],
            "crispr_rep_z": [3.0, 3.0, 2.5],
            "orf_active": [True, True, True],
            "crispr_active": [True, True, True],
            "delta_activity": [0.0, 0.5, -0.1],
            "raw_signed_cosine": [0.9, -0.8, 0.1],
            "sign_null_orf_bin": [1, 1, 1],
            "sign_null_crispr_bin": [1, 1, 1],
            "sign_null_fallback_level": ["global", "global", "global"],
            "sign_null_mean": [0.0, 0.0, 0.0],
            "sign_null_std": [1.0, 1.0, 1.0],
            "sign_z": [0.9, -0.8, 0.1],
            "regime_label_prebootstrap": ["orthogonal", "orthogonal", "orthogonal"],
            "bootstrap_prob_aligned": [0.2, 0.2, 0.2],
            "bootstrap_prob_inverse": [0.2, 0.2, 0.2],
            "bootstrap_prob_orthogonal": [0.6, 0.6, 0.6],
            "bootstrap_prob_asym_orf": [0.0, 0.0, 0.0],
            "bootstrap_prob_asym_crispr": [0.0, 0.0, 0.0],
            "bootstrap_prob_inactive": [0.0, 0.0, 0.0],
            "bootstrap_consensus": [0.6, 0.6, 0.6],
            "bootstrap_consensus_label": ["orthogonal", "orthogonal", "orthogonal"],
            "prebootstrap_matches_consensus": [True, True, True],
            "regime_label": ["orthogonal", "orthogonal", "orthogonal"],
        }
    )
    gene_stats = pl.DataFrame(
        {
            "gene_id": ["1", "2", "3", "1", "2", "3"],
            "display_symbol": ["G1", "G2", "G3", "G1", "G2", "G3"],
            "canonical_symbol": ["G1", "G2", "G3", "G1", "G2", "G3"],
            "modality": ["orf", "orf", "orf", "crispr", "crispr", "crispr"],
            "pipeline": ["official"] * 6,
            "n_wells": [2] * 6,
            "n_plates": [2, 2, 2, 2, 2, 2],
            "centroid_l2": [1.0] * 6,
            "centroid_vector": [[1.0, 0.0]] * 6,
            "strength_z": [3.0, 2.5, 2.1, 3.0, 2.0, 2.2],
            "rep_similarity": [1.0] * 6,
            "rep_z": [3.0] * 6,
            "active_bool": [True] * 6,
            "null_strength_median": [0.0] * 6,
            "null_strength_mad": [1.0] * 6,
            "null_rep_median": [0.0] * 6,
            "null_rep_mad": [1.0] * 6,
            "low_replicate_support": [False] * 6,
        }
    )
    centered_orf = pl.DataFrame(
        {
            "Metadata_NCBI_Gene_ID": ["1", "1", "2", "2", "3", "3"],
            "Metadata_Plate": ["P1", "P2", "P1", "P2", "P1", "P2"],
            "Metadata_Well": ["A01", "A02", "A03", "A04", "A05", "A06"],
            "Metadata_Batch": ["B1"] * 6,
            "is_negative_control": [False] * 6,
            "f1": [1.0, 1.1, 0.0, 0.0, 1.0, 1.1],
            "f2": [0.0, 0.1, 1.0, 1.1, 1.0, 1.1],
        }
    )
    centered_crispr = pl.DataFrame(
        {
            "Metadata_NCBI_Gene_ID": ["1", "1", "2", "2", "3", "3"],
            "Metadata_Plate": ["Q1", "Q2", "Q1", "Q2", "Q1", "Q2"],
            "Metadata_Well": ["A01", "A02", "A03", "A04", "A05", "A06"],
            "Metadata_Batch": ["B2"] * 6,
            "is_negative_control": [False] * 6,
            "f1": [1.0, 1.1, 0.0, 0.0, 1.0, 1.1],
            "f2": [0.0, 0.1, -1.0, -1.1, 0.1, 0.2],
        }
    )
    confound_config = ConfoundConfig(
        strength_threshold=2.0,
        rep_threshold=2.0,
        sign_z_threshold=1.96,
        asymmetry_threshold=1.0,
        null_std_floor=1.0e-6,
        random_seed=1337,
        n_different_gene_samples=20,
        n_bootstrap=5,
        consensus_threshold=0.7,
    )

    serial_confounds, serial_prevalence = run_confounds(
        geometry,
        gene_stats,
        centered_orf,
        centered_crispr,
        ["f1", "f2"],
        config=confound_config,
    )

    compute_config = load_compute_config(
        {
            "compute": {
                "parallel_backend": "process_pool",
                "phase8_workers": 2,
                "phase8_gene_chunk_size": 1,
                "numpy_threads_per_worker": 1,
            }
        },
        run_dir=tmp_path,
    )
    parallel_confounds, parallel_prevalence = run_confounds(
        geometry,
        gene_stats,
        centered_orf,
        centered_crispr,
        ["f1", "f2"],
        config=confound_config,
        compute_config=compute_config,
        run_dir=tmp_path,
        progress_path=tmp_path / "08_progress.json",
    )

    assert serial_confounds.sort(["pipeline", "confound_name"]).to_dicts() == (
        parallel_confounds.sort(["pipeline", "confound_name"]).to_dicts()
    )
    assert serial_prevalence.sort(["pipeline", "regime_label"]).to_dicts() == (
        parallel_prevalence.sort(["pipeline", "regime_label"]).to_dicts()
    )
    assert (tmp_path / "08_progress.json").exists()
