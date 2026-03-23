from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from siggeo.io.atomic import write_csv_atomic, write_parquet_atomic
from siggeo.geometry.nulls import NullSummary, assign_strength_bins, precompute_null_summaries
from siggeo.geometry.signed_scores import _null_choice


@dataclass(frozen=True)
class RetrievalConfig:
    lambda_values: list[float]
    primary_lambda: float
    activity_deciles: int
    n_cross_null_samples_per_cell: int
    min_pairs_per_null_cell: int
    null_std_floor: float
    random_seed: int


def _prepare_binned_stats(stats: pl.DataFrame, prefix: str, n_bins: int) -> pl.DataFrame:
    return assign_strength_bins(stats, "strength_z", n_bins, f"{prefix}_bin").sort("gene_id")


def _normalized_centroids(stats: pl.DataFrame) -> np.ndarray:
    centroids = np.array(stats["centroid_vector"].to_list(), dtype=float)
    if centroids.size == 0:
        return np.zeros((0, 0), dtype=float)
    norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    safe_norms = np.where(norms <= 0, 1.0, norms)
    return centroids / safe_norms


def _candidate_null_arrays(
    query_bin: int,
    candidate_bins: np.ndarray,
    *,
    decile_summaries: dict[tuple[int | None, int | None], NullSummary],
    quintile_summaries: dict[tuple[int | None, int | None], NullSummary],
    config: RetrievalConfig,
) -> tuple[np.ndarray, np.ndarray]:
    means = np.zeros(candidate_bins.shape[0], dtype=float)
    stds = np.full(candidate_bins.shape[0], config.null_std_floor, dtype=float)
    for candidate_bin in np.unique(candidate_bins):
        _, null_summary = _null_choice(
            int(query_bin),
            int(candidate_bin),
            decile_summaries,
            quintile_summaries,
            config.min_pairs_per_null_cell,
        )
        mask = candidate_bins == candidate_bin
        means[mask] = null_summary.mean
        stds[mask] = max(null_summary.std, config.null_std_floor)
    return means, stds


def _true_match_rank(scores: np.ndarray, true_index: int, candidate_gene_ids: np.ndarray, true_gene_id: str) -> int:
    true_score = float(scores[true_index])
    higher = int(np.count_nonzero(scores > true_score))
    ties_before = int(np.count_nonzero((scores == true_score) & (candidate_gene_ids < true_gene_id)))
    return higher + ties_before + 1


def _score_direction(
    query_stats: pl.DataFrame,
    candidate_stats: pl.DataFrame,
    *,
    query_label: str,
    candidate_label: str,
    decile_summaries: dict[tuple[int | None, int | None], NullSummary],
    quintile_summaries: dict[tuple[int | None, int | None], NullSummary],
    config: RetrievalConfig,
    batch_size: int = 256,
) -> pl.DataFrame:
    if query_stats.is_empty() or candidate_stats.is_empty():
        return pl.DataFrame(
            schema={
                "query_modality": pl.String,
                "query_gene_id": pl.String,
                "candidate_modality": pl.String,
                "candidate_gene_id": pl.String,
                "score_name": pl.String,
                "true_match_score": pl.Float64,
                "rank": pl.Int64,
                "n_candidates": pl.Int64,
                "is_true_match": pl.Boolean,
            }
        )

    query_gene_ids = np.array(query_stats["gene_id"].to_list(), dtype=object)
    candidate_gene_ids = np.array(candidate_stats["gene_id"].to_list(), dtype=object)
    query_bins = np.array(query_stats["query_bin"].to_list(), dtype=int)
    candidate_bins = np.array(candidate_stats["candidate_bin"].to_list(), dtype=int)
    query_strength = np.array(query_stats["strength_z"].to_list(), dtype=float)
    candidate_strength = np.array(candidate_stats["strength_z"].to_list(), dtype=float)
    query_vectors = _normalized_centroids(query_stats)
    candidate_vectors = _normalized_centroids(candidate_stats)
    candidate_lookup = {str(gene_id): idx for idx, gene_id in enumerate(candidate_gene_ids.tolist())}
    null_arrays = {
        int(query_bin): _candidate_null_arrays(
            int(query_bin),
            candidate_bins,
            decile_summaries=decile_summaries,
            quintile_summaries=quintile_summaries,
            config=config,
        )
        for query_bin in np.unique(query_bins)
    }

    rows: list[dict[str, object]] = []
    for start in range(0, query_vectors.shape[0], batch_size):
        stop = min(start + batch_size, query_vectors.shape[0])
        raw_block = query_vectors[start:stop] @ candidate_vectors.T
        for offset, query_idx in enumerate(range(start, stop)):
            query_gene_id = str(query_gene_ids[query_idx])
            true_index = candidate_lookup.get(query_gene_id)
            if true_index is None:
                continue

            raw_scores = raw_block[offset]
            mean_array, std_array = null_arrays[int(query_bins[query_idx])]
            sign_z_scores = (raw_scores - mean_array) / std_array
            penalty = np.abs(query_strength[query_idx] - candidate_strength)
            score_vectors = {
                "signed_cosine": raw_scores,
                "abs_cosine": np.abs(raw_scores),
                "sign_z": sign_z_scores,
            }
            for lambda_value in config.lambda_values:
                score_vectors[f"regime_score_lambda_{lambda_value:g}"] = sign_z_scores - lambda_value * penalty

            for score_name, score_values in score_vectors.items():
                rows.append(
                    {
                        "query_modality": query_label,
                        "query_gene_id": query_gene_id,
                        "candidate_modality": candidate_label,
                        "candidate_gene_id": query_gene_id,
                        "score_name": score_name,
                        "true_match_score": float(score_values[true_index]),
                        "rank": _true_match_rank(score_values, true_index, candidate_gene_ids, query_gene_id),
                        "n_candidates": int(candidate_gene_ids.shape[0]),
                        "is_true_match": True,
                    }
                )
    return pl.DataFrame(rows).sort(["score_name", "query_gene_id"])


def _summary_from_ranks(ranks: pl.DataFrame) -> pl.DataFrame:
    if ranks.is_empty():
        return pl.DataFrame(
            schema={
                "score_name": pl.String,
                "direction": pl.String,
                "mrr": pl.Float64,
                "hits_at_1": pl.Float64,
                "n_queries": pl.Int64,
            }
        )
    summaries: list[dict[str, object]] = []
    combinations = ranks.select(["score_name", "direction"]).unique(maintain_order=True).to_dicts()
    for combo in combinations:
        score_name = combo["score_name"]
        direction = combo["direction"]
        frame = ranks.filter((pl.col("score_name") == score_name) & (pl.col("direction") == direction))
        ranks_list = frame["rank"].to_list()
        summaries.append(
            {
                "score_name": score_name,
                "direction": direction,
                "mrr": float(sum(1.0 / rank for rank in ranks_list) / len(ranks_list)),
                "hits_at_1": float(sum(1 for rank in ranks_list if rank == 1) / len(ranks_list)),
                "n_queries": len(ranks_list),
            }
        )
    return pl.DataFrame(summaries).sort(["score_name", "direction"])


def evaluate_retrieval(
    orf_stats: pl.DataFrame,
    crispr_stats: pl.DataFrame,
    *,
    config: RetrievalConfig,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    orf_query = _prepare_binned_stats(orf_stats, "query", config.activity_deciles)
    crispr_candidate = _prepare_binned_stats(crispr_stats, "candidate", config.activity_deciles)
    orf_query_quint = _prepare_binned_stats(orf_stats, "query", 5)
    crispr_candidate_quint = _prepare_binned_stats(crispr_stats, "candidate", 5)

    forward_decile_summaries = precompute_null_summaries(
        orf_query,
        crispr_candidate,
        orf_bin_column="query_bin",
        crispr_bin_column="candidate_bin",
        n_samples=config.n_cross_null_samples_per_cell,
        random_seed=config.random_seed,
    )
    forward_quintile_summaries = precompute_null_summaries(
        orf_query_quint,
        crispr_candidate_quint,
        orf_bin_column="query_bin",
        crispr_bin_column="candidate_bin",
        n_samples=config.n_cross_null_samples_per_cell,
        random_seed=config.random_seed,
    )
    crispr_query = _prepare_binned_stats(crispr_stats, "query", config.activity_deciles)
    orf_candidate = _prepare_binned_stats(orf_stats, "candidate", config.activity_deciles)
    crispr_query_quint = _prepare_binned_stats(crispr_stats, "query", 5)
    orf_candidate_quint = _prepare_binned_stats(orf_stats, "candidate", 5)
    reverse_decile_summaries = precompute_null_summaries(
        crispr_query,
        orf_candidate,
        orf_bin_column="query_bin",
        crispr_bin_column="candidate_bin",
        n_samples=config.n_cross_null_samples_per_cell,
        random_seed=config.random_seed,
    )
    reverse_quintile_summaries = precompute_null_summaries(
        crispr_query_quint,
        orf_candidate_quint,
        orf_bin_column="query_bin",
        crispr_bin_column="candidate_bin",
        n_samples=config.n_cross_null_samples_per_cell,
        random_seed=config.random_seed,
    )

    orf_to_crispr = _score_direction(
        orf_query,
        crispr_candidate,
        query_label="orf",
        candidate_label="crispr",
        decile_summaries=forward_decile_summaries,
        quintile_summaries=forward_quintile_summaries,
        config=config,
    ).with_columns(pl.lit("orf_to_crispr").alias("direction"))
    crispr_to_orf = _score_direction(
        crispr_query,
        orf_candidate,
        query_label="crispr",
        candidate_label="orf",
        decile_summaries=reverse_decile_summaries,
        quintile_summaries=reverse_quintile_summaries,
        config=config,
    ).with_columns(pl.lit("crispr_to_orf").alias("direction"))

    ranks = pl.concat([orf_to_crispr, crispr_to_orf]).select(
        "query_modality",
        "query_gene_id",
        "candidate_modality",
        "candidate_gene_id",
        "score_name",
        "true_match_score",
        "rank",
        "n_candidates",
        "is_true_match",
        "direction",
    )
    summary = _summary_from_ranks(ranks)
    return ranks, summary


def write_retrieval_outputs(
    ranks: pl.DataFrame,
    summary: pl.DataFrame,
    ranks_path: Path,
    summary_path: Path,
) -> None:
    write_parquet_atomic(ranks, ranks_path)
    write_csv_atomic(summary, summary_path)
