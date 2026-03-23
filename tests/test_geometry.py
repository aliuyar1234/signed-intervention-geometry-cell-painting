from __future__ import annotations

from pathlib import Path

import polars as pl

from siggeo.compute import ComputeConfig
import siggeo.geometry.signed_scores as signed_scores_module
from siggeo.geometry.regimes import assign_regime
from siggeo.geometry.signed_scores import GeometryConfig, compute_signed_geometry


def test_assign_regime_matches_locked_order() -> None:
    assert assign_regime(
        orf_active=False,
        crispr_active=False,
        sign_z=10.0,
        delta_activity=0.0,
        sign_z_threshold=1.96,
        asymmetry_threshold=1.0,
    ) == "inactive"
    assert assign_regime(
        orf_active=True,
        crispr_active=False,
        sign_z=0.0,
        delta_activity=3.0,
        sign_z_threshold=1.96,
        asymmetry_threshold=1.0,
    ) == "asymmetric_ORF"
    assert assign_regime(
        orf_active=True,
        crispr_active=True,
        sign_z=-3.0,
        delta_activity=0.0,
        sign_z_threshold=1.96,
        asymmetry_threshold=1.0,
    ) == "inverse"


def test_compute_signed_geometry_emits_expected_rows() -> None:
    orf_stats = pl.DataFrame(
        {
            "gene_id": ["1", "2", "3"],
            "display_symbol": ["G1", "G2", "G3"],
            "modality": ["orf", "orf", "orf"],
            "pipeline": ["official", "official", "official"],
            "n_wells": [2, 2, 2],
            "n_plates": [1, 1, 1],
            "centroid_l2": [1.0, 1.0, 1.0],
            "centroid_vector": [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
            "strength_z": [3.0, 3.0, 3.0],
            "rep_similarity": [1.0, 1.0, 1.0],
            "rep_z": [3.0, 3.0, 3.0],
            "active_bool": [True, True, True],
            "null_strength_median": [0.0, 0.0, 0.0],
            "null_strength_mad": [1.0, 1.0, 1.0],
            "null_rep_median": [0.0, 0.0, 0.0],
            "null_rep_mad": [1.0, 1.0, 1.0],
            "low_replicate_support": [True, True, True],
        }
    )
    crispr_stats = pl.DataFrame(
        {
            "gene_id": ["1", "2", "3"],
            "display_symbol": ["G1", "G2", "G3"],
            "modality": ["crispr", "crispr", "crispr"],
            "pipeline": ["official", "official", "official"],
            "n_wells": [2, 2, 2],
            "n_plates": [1, 1, 1],
            "centroid_l2": [1.0, 1.0, 1.0],
            "centroid_vector": [[1.0, 0.0], [0.0, -1.0], [1.0, -1.0]],
            "strength_z": [3.0, 3.0, 3.0],
            "rep_similarity": [1.0, 1.0, 1.0],
            "rep_z": [3.0, 3.0, 3.0],
            "active_bool": [True, True, True],
            "null_strength_median": [0.0, 0.0, 0.0],
            "null_strength_mad": [1.0, 1.0, 1.0],
            "null_rep_median": [0.0, 0.0, 0.0],
            "null_rep_mad": [1.0, 1.0, 1.0],
            "low_replicate_support": [True, True, True],
        }
    )
    centered_orf = pl.DataFrame(
        {
            "Metadata_NCBI_Gene_ID": ["1", "1", "2", "2", "3", "3"],
            "Metadata_Plate": ["P1", "P1", "P1", "P1", "P1", "P1"],
            "Metadata_Well": ["A01", "A02", "A03", "A04", "A05", "A06"],
            "is_negative_control": [False, False, False, False, False, False],
            "f1": [1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
            "f2": [0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
        }
    )
    centered_crispr = pl.DataFrame(
        {
            "Metadata_NCBI_Gene_ID": ["1", "1", "2", "2", "3", "3"],
            "Metadata_Plate": ["P2", "P2", "P2", "P2", "P2", "P2"],
            "Metadata_Well": ["A01", "A02", "A03", "A04", "A05", "A06"],
            "is_negative_control": [False, False, False, False, False, False],
            "f1": [1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
            "f2": [0.0, 0.0, -1.0, -1.0, -1.0, -1.0],
        }
    )

    geometry = compute_signed_geometry(
        orf_stats,
        crispr_stats,
        centered_orf,
        centered_crispr,
        ["f1", "f2"],
        strength_threshold=2.0,
        rep_threshold=2.0,
        config=GeometryConfig(
            activity_deciles=10,
            n_cross_null_samples_per_cell=100,
            min_pairs_per_null_cell=1,
            sign_z_threshold=1.96,
            asymmetry_threshold=1.0,
            null_std_floor=1.0e-6,
            n_bootstrap=5,
            consensus_threshold=0.0,
            random_seed=1337,
        ),
    )

    rows = {row["gene_id"]: row for row in geometry.to_dicts()}
    assert geometry.height == 3
    assert rows["1"]["raw_signed_cosine"] > 0.99
    assert rows["2"]["raw_signed_cosine"] < -0.99
    assert "sign_z" in geometry.columns


def test_compute_signed_geometry_is_deterministic_under_fixed_seed() -> None:
    orf_stats = pl.DataFrame(
        {
            "gene_id": ["1", "2", "3"],
            "display_symbol": ["G1", "G2", "G3"],
            "modality": ["orf", "orf", "orf"],
            "pipeline": ["official", "official", "official"],
            "n_wells": [2, 2, 2],
            "n_plates": [1, 1, 1],
            "centroid_l2": [1.0, 1.0, 1.0],
            "centroid_vector": [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
            "strength_z": [3.0, 2.0, 1.0],
            "rep_similarity": [1.0, 1.0, 1.0],
            "rep_z": [3.0, 3.0, 3.0],
            "active_bool": [True, True, False],
            "null_strength_median": [0.0, 0.0, 0.0],
            "null_strength_mad": [1.0, 1.0, 1.0],
            "null_rep_median": [0.0, 0.0, 0.0],
            "null_rep_mad": [1.0, 1.0, 1.0],
            "low_replicate_support": [True, True, True],
        }
    )
    crispr_stats = pl.DataFrame(
        {
            "gene_id": ["1", "2", "3"],
            "display_symbol": ["G1", "G2", "G3"],
            "modality": ["crispr", "crispr", "crispr"],
            "pipeline": ["official", "official", "official"],
            "n_wells": [2, 2, 2],
            "n_plates": [1, 1, 1],
            "centroid_l2": [1.0, 1.0, 1.0],
            "centroid_vector": [[1.0, 0.0], [0.0, -1.0], [1.0, -1.0]],
            "strength_z": [1.0, 2.0, 3.0],
            "rep_similarity": [1.0, 1.0, 1.0],
            "rep_z": [3.0, 3.0, 3.0],
            "active_bool": [False, True, True],
            "null_strength_median": [0.0, 0.0, 0.0],
            "null_strength_mad": [1.0, 1.0, 1.0],
            "null_rep_median": [0.0, 0.0, 0.0],
            "null_rep_mad": [1.0, 1.0, 1.0],
            "low_replicate_support": [True, True, True],
        }
    )
    centered_orf = pl.DataFrame(
        {
            "Metadata_NCBI_Gene_ID": ["1", "1", "2", "2", "3", "3"],
            "Metadata_Plate": ["P1", "P1", "P1", "P1", "P1", "P1"],
            "Metadata_Well": ["A01", "A02", "A03", "A04", "A05", "A06"],
            "is_negative_control": [False, False, False, False, False, False],
            "f1": [1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
            "f2": [0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
        }
    )
    centered_crispr = pl.DataFrame(
        {
            "Metadata_NCBI_Gene_ID": ["1", "1", "2", "2", "3", "3"],
            "Metadata_Plate": ["P2", "P2", "P2", "P2", "P2", "P2"],
            "Metadata_Well": ["A01", "A02", "A03", "A04", "A05", "A06"],
            "is_negative_control": [False, False, False, False, False, False],
            "f1": [1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
            "f2": [0.0, 0.0, -1.0, -1.0, -1.0, -1.0],
        }
    )

    config = GeometryConfig(
        activity_deciles=10,
        n_cross_null_samples_per_cell=100,
        min_pairs_per_null_cell=1,
        sign_z_threshold=1.96,
        asymmetry_threshold=1.0,
        null_std_floor=1.0e-6,
        n_bootstrap=10,
        consensus_threshold=0.0,
        random_seed=1337,
    )

    first = compute_signed_geometry(
        orf_stats,
        crispr_stats,
        centered_orf,
        centered_crispr,
        ["f1", "f2"],
        strength_threshold=2.0,
        rep_threshold=2.0,
        config=config,
    )
    second = compute_signed_geometry(
        orf_stats,
        crispr_stats,
        centered_orf,
        centered_crispr,
        ["f1", "f2"],
        strength_threshold=2.0,
        rep_threshold=2.0,
        config=config,
    )

    assert first.to_dicts() == second.to_dicts()


def test_compute_signed_geometry_uses_bootstrap_consensus_label(monkeypatch) -> None:
    def fake_bootstrap(*args, **kwargs):
        return {
            "aligned": 0.1,
            "inverse": 0.8,
            "orthogonal": 0.1,
            "asymmetric_ORF": 0.0,
            "asymmetric_CRISPR": 0.0,
            "inactive": 0.0,
        }

    monkeypatch.setattr(signed_scores_module, "_bootstrap_probabilities", fake_bootstrap)

    orf_stats = pl.DataFrame(
        {
            "gene_id": ["1", "2"],
            "display_symbol": ["G1", "G2"],
            "modality": ["orf", "orf"],
            "pipeline": ["official", "official"],
            "n_wells": [2, 2],
            "n_plates": [1, 1],
            "centroid_l2": [1.0, 1.0],
            "centroid_vector": [[1.0, 0.0], [0.0, 1.0]],
            "strength_z": [3.0, 3.0],
            "rep_similarity": [1.0, 1.0],
            "rep_z": [3.0, 3.0],
            "active_bool": [True, True],
            "null_strength_median": [0.0, 0.0],
            "null_strength_mad": [1.0, 1.0],
            "null_rep_median": [0.0, 0.0],
            "null_rep_mad": [1.0, 1.0],
            "low_replicate_support": [True, True],
        }
    )
    crispr_stats = pl.DataFrame(
        {
            "gene_id": ["1", "2"],
            "display_symbol": ["G1", "G2"],
            "modality": ["crispr", "crispr"],
            "pipeline": ["official", "official"],
            "n_wells": [2, 2],
            "n_plates": [1, 1],
            "centroid_l2": [1.0, 1.0],
            "centroid_vector": [[1.0, 0.0], [0.0, -1.0]],
            "strength_z": [3.0, 3.0],
            "rep_similarity": [1.0, 1.0],
            "rep_z": [3.0, 3.0],
            "active_bool": [True, True],
            "null_strength_median": [0.0, 0.0],
            "null_strength_mad": [1.0, 1.0],
            "null_rep_median": [0.0, 0.0],
            "null_rep_mad": [1.0, 1.0],
            "low_replicate_support": [True, True],
        }
    )
    centered_orf = pl.DataFrame(
        {
            "Metadata_NCBI_Gene_ID": ["1", "1", "2", "2"],
            "Metadata_Plate": ["P1", "P1", "P1", "P1"],
            "Metadata_Well": ["A01", "A02", "A03", "A04"],
            "is_negative_control": [False, False, False, False],
            "f1": [1.0, 1.0, 0.0, 0.0],
            "f2": [0.0, 0.0, 1.0, 1.0],
        }
    )
    centered_crispr = pl.DataFrame(
        {
            "Metadata_NCBI_Gene_ID": ["1", "1", "2", "2"],
            "Metadata_Plate": ["P2", "P2", "P2", "P2"],
            "Metadata_Well": ["A01", "A02", "A03", "A04"],
            "is_negative_control": [False, False, False, False],
            "f1": [1.0, 1.0, 0.0, 0.0],
            "f2": [0.0, 0.0, -1.0, -1.0],
        }
    )

    geometry = compute_signed_geometry(
        orf_stats,
        crispr_stats,
        centered_orf,
        centered_crispr,
        ["f1", "f2"],
        strength_threshold=2.0,
        rep_threshold=2.0,
        config=GeometryConfig(
            activity_deciles=10,
            n_cross_null_samples_per_cell=100,
            min_pairs_per_null_cell=1,
            sign_z_threshold=1.96,
            asymmetry_threshold=1.0,
            null_std_floor=1.0e-6,
            n_bootstrap=5,
            consensus_threshold=0.70,
            random_seed=1337,
        ),
    )

    assert set(geometry["bootstrap_consensus_label"].to_list()) == {"inverse"}
    assert set(geometry["regime_label"].to_list()) == {"inverse"}


def test_compute_signed_geometry_parallel_matches_serial(tmp_path: Path) -> None:
    orf_stats = pl.DataFrame(
        {
            "gene_id": ["1", "2", "3"],
            "display_symbol": ["G1", "G2", "G3"],
            "canonical_symbol": ["G1", "G2", "G3"],
            "modality": ["orf", "orf", "orf"],
            "pipeline": ["official", "official", "official"],
            "n_wells": [2, 2, 2],
            "n_plates": [1, 1, 1],
            "centroid_l2": [1.0, 1.0, 1.0],
            "centroid_vector": [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
            "strength_z": [3.0, 2.0, 1.0],
            "rep_similarity": [1.0, 1.0, 1.0],
            "rep_z": [3.0, 3.0, 3.0],
            "active_bool": [True, True, False],
            "null_strength_median": [0.0, 0.0, 0.0],
            "null_strength_mad": [1.0, 1.0, 1.0],
            "null_rep_median": [0.0, 0.0, 0.0],
            "null_rep_mad": [1.0, 1.0, 1.0],
            "low_replicate_support": [True, True, True],
        }
    )
    crispr_stats = pl.DataFrame(
        {
            "gene_id": ["1", "2", "3"],
            "display_symbol": ["G1", "G2", "G3"],
            "canonical_symbol": ["G1", "G2", "G3"],
            "modality": ["crispr", "crispr", "crispr"],
            "pipeline": ["official", "official", "official"],
            "n_wells": [2, 2, 2],
            "n_plates": [1, 1, 1],
            "centroid_l2": [1.0, 1.0, 1.0],
            "centroid_vector": [[1.0, 0.0], [0.0, -1.0], [1.0, -1.0]],
            "strength_z": [1.0, 2.0, 3.0],
            "rep_similarity": [1.0, 1.0, 1.0],
            "rep_z": [3.0, 3.0, 3.0],
            "active_bool": [False, True, True],
            "null_strength_median": [0.0, 0.0, 0.0],
            "null_strength_mad": [1.0, 1.0, 1.0],
            "null_rep_median": [0.0, 0.0, 0.0],
            "null_rep_mad": [1.0, 1.0, 1.0],
            "low_replicate_support": [True, True, True],
        }
    )
    centered_orf = pl.DataFrame(
        {
            "Metadata_NCBI_Gene_ID": ["1", "1", "2", "2", "3", "3"],
            "Metadata_Plate": ["P1", "P1", "P1", "P1", "P1", "P1"],
            "Metadata_Well": ["A01", "A02", "A03", "A04", "A05", "A06"],
            "is_negative_control": [False, False, False, False, False, False],
            "f1": [1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
            "f2": [0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
        }
    )
    centered_crispr = pl.DataFrame(
        {
            "Metadata_NCBI_Gene_ID": ["1", "1", "2", "2", "3", "3"],
            "Metadata_Plate": ["P2", "P2", "P2", "P2", "P2", "P2"],
            "Metadata_Well": ["A01", "A02", "A03", "A04", "A05", "A06"],
            "is_negative_control": [False, False, False, False, False, False],
            "f1": [1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
            "f2": [0.0, 0.0, -1.0, -1.0, -1.0, -1.0],
        }
    )
    geometry_config = GeometryConfig(
        activity_deciles=10,
        n_cross_null_samples_per_cell=100,
        min_pairs_per_null_cell=1,
        sign_z_threshold=1.96,
        asymmetry_threshold=1.0,
        null_std_floor=1.0e-6,
        n_bootstrap=10,
        consensus_threshold=0.0,
        random_seed=1337,
    )

    serial = compute_signed_geometry(
        orf_stats,
        crispr_stats,
        centered_orf,
        centered_crispr,
        ["f1", "f2"],
        strength_threshold=2.0,
        rep_threshold=2.0,
        config=geometry_config,
    )
    parallel = compute_signed_geometry(
        orf_stats,
        crispr_stats,
        centered_orf,
        centered_crispr,
        ["f1", "f2"],
        strength_threshold=2.0,
        rep_threshold=2.0,
        config=geometry_config,
        compute_config=ComputeConfig(
            parallel_backend="process_pool",
            max_workers=2,
            phase4_workers=1,
            phase5_workers=2,
            phase8_workers=1,
            phase4_gene_chunk_size=1,
            phase5_gene_chunk_size=2,
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
