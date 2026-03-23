from __future__ import annotations

import polars as pl

from siggeo.evaluation.retrieval import RetrievalConfig, evaluate_retrieval


def test_evaluate_retrieval_outputs_ranks_and_summary() -> None:
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
            "strength_z": [3.0, 2.5],
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
            "centroid_vector": [[1.0, 0.0], [0.0, 1.0]],
            "strength_z": [3.0, 2.5],
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

    ranks, summary = evaluate_retrieval(
        orf_stats,
        crispr_stats,
        config=RetrievalConfig(
            lambda_values=[0.5],
            primary_lambda=0.5,
            activity_deciles=10,
            n_cross_null_samples_per_cell=100,
            min_pairs_per_null_cell=1,
            null_std_floor=1.0e-6,
            random_seed=1337,
        ),
    )

    assert ranks.height > 0
    assert summary.height > 0
    assert set(summary["direction"].to_list()) == {"orf_to_crispr", "crispr_to_orf"}
    assert "regime_score_lambda_0.5" in summary["score_name"].to_list()


def test_evaluate_retrieval_perfect_matches_rank_first() -> None:
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
            "centroid_vector": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "strength_z": [3.0, 3.0, 3.0],
            "rep_similarity": [1.0, 1.0, 1.0],
            "rep_z": [3.0, 3.0, 3.0],
            "active_bool": [True, True, True],
            "null_strength_median": [0.0, 0.0, 0.0],
            "null_strength_mad": [1.0, 1.0, 1.0],
            "null_rep_median": [0.0, 0.0, 0.0],
            "null_rep_mad": [1.0, 1.0, 1.0],
            "low_replicate_support": [False, False, False],
        }
    )
    crispr_stats = orf_stats.with_columns(pl.lit("crispr").alias("modality"))

    ranks, summary = evaluate_retrieval(
        orf_stats,
        crispr_stats,
        config=RetrievalConfig(
            lambda_values=[0.5],
            primary_lambda=0.5,
            activity_deciles=10,
            n_cross_null_samples_per_cell=100,
            min_pairs_per_null_cell=1,
            null_std_floor=1.0e-6,
            random_seed=1337,
        ),
    )

    assert set(ranks["rank"].to_list()) == {1}
    assert set(summary["mrr"].to_list()) == {1.0}
    assert set(summary["hits_at_1"].to_list()) == {1.0}
