from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl


@dataclass(frozen=True)
class NullSummary:
    count: int
    mean: float
    std: float


def compute_bin_edges(frame: pl.DataFrame, column: str, n_bins: int) -> list[float]:
    if frame.is_empty():
        return []
    values = np.array(frame[column].to_list(), dtype=float)
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
    return np.quantile(values, quantiles, method="linear").tolist()


def assign_value_to_bin(value: float, edges: list[float], n_bins: int) -> int:
    if not edges:
        return 1
    return int(min(n_bins, np.searchsorted(np.array(edges, dtype=float), value, side="right") + 1))


def assign_strength_bins(frame: pl.DataFrame, column: str, n_bins: int, output_column: str) -> pl.DataFrame:
    if frame.is_empty():
        return frame.with_columns(pl.lit(None).alias(output_column))
    edges = compute_bin_edges(frame, column, n_bins)
    bins = [assign_value_to_bin(float(value), edges, n_bins) for value in frame[column].to_list()]
    return frame.with_columns(pl.Series(output_column, bins)).sort("gene_id")


def _normalize_centroids(frame: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    centroids = np.array(frame["centroid_vector"].to_list(), dtype=float)
    norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    safe_norms = np.where(norms <= 0, 1.0, norms)
    return centroids / safe_norms, norms.squeeze(-1)


def precompute_null_summaries(
    orf_stats: pl.DataFrame,
    crispr_stats: pl.DataFrame,
    *,
    orf_bin_column: str,
    crispr_bin_column: str,
    n_samples: int,
    random_seed: int,
) -> dict[tuple[int | None, int | None], NullSummary]:
    if orf_stats.is_empty() or crispr_stats.is_empty():
        return {(None, None): NullSummary(0, 0.0, 0.0)}
    rng = np.random.default_rng(random_seed)
    orf_normed, _ = _normalize_centroids(orf_stats)
    crispr_normed, _ = _normalize_centroids(crispr_stats)
    orf_gene_ids = np.array(orf_stats["gene_id"].to_list(), dtype=object)
    crispr_gene_ids = np.array(crispr_stats["gene_id"].to_list(), dtype=object)
    orf_bins = np.array(orf_stats[orf_bin_column].to_list(), dtype=object)
    crispr_bins = np.array(crispr_stats[crispr_bin_column].to_list(), dtype=object)

    summaries: dict[tuple[int | None, int | None], NullSummary] = {}
    unique_orf_bins = sorted({value for value in orf_bins.tolist() if value is not None})
    unique_crispr_bins = sorted({value for value in crispr_bins.tolist() if value is not None})
    for orf_bin in unique_orf_bins:
        for crispr_bin in unique_crispr_bins:
            orf_idx = np.where(orf_bins == orf_bin)[0]
            crispr_idx = np.where(crispr_bins == crispr_bin)[0]
            if orf_idx.size == 0 or crispr_idx.size == 0:
                summaries[(orf_bin, crispr_bin)] = NullSummary(0, 0.0, 0.0)
                continue
            cosines = orf_normed[orf_idx] @ crispr_normed[crispr_idx].T
            gene_mask = orf_gene_ids[orf_idx][:, None] != crispr_gene_ids[crispr_idx][None, :]
            values = cosines[gene_mask]
            if values.size == 0:
                summaries[(orf_bin, crispr_bin)] = NullSummary(0, 0.0, 0.0)
                continue
            if values.size > n_samples:
                values = rng.choice(values, size=n_samples, replace=False)
            summaries[(orf_bin, crispr_bin)] = NullSummary(
                count=int(values.size),
                mean=float(values.mean()),
                std=float(values.std(ddof=0)),
            )
    all_values = (orf_normed @ crispr_normed.T)
    global_mask = orf_gene_ids[:, None] != crispr_gene_ids[None, :]
    global_values = all_values[global_mask]
    if global_values.size > n_samples:
        global_values = rng.choice(global_values, size=n_samples, replace=False)
    summaries[(None, None)] = NullSummary(
        count=int(global_values.size),
        mean=float(global_values.mean()) if global_values.size else 0.0,
        std=float(global_values.std(ddof=0)) if global_values.size else 0.0,
    )
    return summaries
