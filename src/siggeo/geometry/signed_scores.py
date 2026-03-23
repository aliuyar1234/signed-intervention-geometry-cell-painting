from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from siggeo.compute import ComputeConfig, set_numpy_worker_env
from siggeo.io.atomic import write_json_atomic, write_parquet_atomic
from siggeo.gene_stats.centroids import centroid_l2, coordinate_median
from siggeo.gene_stats.reproducibility import pairwise_cosine_median
from siggeo.geometry.nulls import (
    NullSummary,
    assign_strength_bins,
    assign_value_to_bin,
    compute_bin_edges,
    precompute_null_summaries,
)
from siggeo.geometry.regimes import REGIME_ORDER, assign_regime


_PHASE5_SHARED: dict[str, Any] | None = None


@dataclass(frozen=True)
class GeometryConfig:
    activity_deciles: int
    n_cross_null_samples_per_cell: int
    min_pairs_per_null_cell: int
    sign_z_threshold: float
    asymmetry_threshold: float
    null_std_floor: float
    n_bootstrap: int
    consensus_threshold: float
    random_seed: int


def _cosine(left: list[float], right: list[float]) -> float:
    left_vec = np.array(left, dtype=float)
    right_vec = np.array(right, dtype=float)
    denom = np.linalg.norm(left_vec) * np.linalg.norm(right_vec)
    if denom <= 0:
        return 0.0
    return float(np.dot(left_vec, right_vec) / denom)


def _null_choice(
    orf_bin: int,
    crispr_bin: int,
    decile_summaries: dict[tuple[int | None, int | None], NullSummary],
    quintile_summaries: dict[tuple[int | None, int | None], NullSummary],
    min_pairs: int,
) -> tuple[str, NullSummary]:
    decile = decile_summaries.get((orf_bin, crispr_bin), NullSummary(0, 0.0, 0.0))
    if decile.count >= min_pairs:
        return "decile", decile
    quint_orf = min(5, (int(orf_bin) - 1) // 2 + 1)
    quint_crispr = min(5, (int(crispr_bin) - 1) // 2 + 1)
    quintile = quintile_summaries.get((quint_orf, quint_crispr), NullSummary(0, 0.0, 0.0))
    if quintile.count >= min_pairs:
        return "quintile", quintile
    return "global", decile_summaries[(None, None)]


def _bootstrap_probabilities(
    row: dict[str, Any],
    orf_values: np.ndarray,
    crispr_values: np.ndarray,
    config: GeometryConfig,
    *,
    orf_decile_edges: list[float],
    crispr_decile_edges: list[float],
    decile_summaries: dict[tuple[int | None, int | None], NullSummary],
    quintile_summaries: dict[tuple[int | None, int | None], NullSummary],
) -> dict[str, float]:
    if orf_values.shape[0] == 0 or crispr_values.shape[0] == 0 or config.n_bootstrap <= 0:
        return {label: 0.0 for label in REGIME_ORDER}
    stable = hashlib.sha256(f"{config.random_seed}:{row['gene_id']}".encode("utf-8")).digest()
    rng = np.random.default_rng(int.from_bytes(stable[:8], "big", signed=False))
    counts = {label: 0 for label in REGIME_ORDER}
    for _ in range(config.n_bootstrap):
        orf_sample = orf_values[rng.choice(orf_values.shape[0], size=orf_values.shape[0], replace=True)]
        crispr_sample = crispr_values[rng.choice(crispr_values.shape[0], size=crispr_values.shape[0], replace=True)]
        orf_centroid = coordinate_median(orf_sample)
        crispr_centroid = coordinate_median(crispr_sample)
        orf_strength_z = (
            centroid_l2(orf_centroid) - row["orf_null_strength_median"]
        ) / max(row["orf_null_strength_mad"], config.null_std_floor)
        crispr_strength_z = (
            centroid_l2(crispr_centroid) - row["crispr_null_strength_median"]
        ) / max(row["crispr_null_strength_mad"], config.null_std_floor)
        orf_rep_z = (
            pairwise_cosine_median(orf_sample) - row["orf_null_rep_median"]
        ) / max(row["orf_null_rep_mad"], config.null_std_floor)
        crispr_rep_z = (
            pairwise_cosine_median(crispr_sample) - row["crispr_null_rep_median"]
        ) / max(row["crispr_null_rep_mad"], config.null_std_floor)
        raw = _cosine(orf_centroid.tolist(), crispr_centroid.tolist())
        orf_bin = assign_value_to_bin(orf_strength_z, orf_decile_edges, config.activity_deciles)
        crispr_bin = assign_value_to_bin(crispr_strength_z, crispr_decile_edges, config.activity_deciles)
        _, null_summary = _null_choice(
            orf_bin,
            crispr_bin,
            decile_summaries,
            quintile_summaries,
            config.min_pairs_per_null_cell,
        )
        sign_z = (raw - null_summary.mean) / max(null_summary.std, config.null_std_floor)
        label = assign_regime(
            orf_active=orf_strength_z >= row["strength_threshold"] and orf_rep_z >= row["rep_threshold"],
            crispr_active=crispr_strength_z >= row["strength_threshold"] and crispr_rep_z >= row["rep_threshold"],
            sign_z=sign_z,
            delta_activity=orf_strength_z - crispr_strength_z,
            sign_z_threshold=config.sign_z_threshold,
            asymmetry_threshold=config.asymmetry_threshold,
        )
        counts[label] += 1
    return {label: counts[label] / config.n_bootstrap for label in REGIME_ORDER}


def _build_replicate_manifest(
    centered_df: pl.DataFrame,
    feature_columns: list[str],
) -> tuple[np.ndarray, dict[str, tuple[int, int]]]:
    treated = (
        centered_df.filter(~pl.col("is_negative_control") & pl.col("Metadata_NCBI_Gene_ID").is_not_null())
        .sort(["Metadata_NCBI_Gene_ID", "Metadata_Plate", "Metadata_Well"])
    )
    manifest: dict[str, tuple[int, int]] = {}
    offset = 0
    for group in treated.partition_by("Metadata_NCBI_Gene_ID", maintain_order=True):
        gene_id = str(group["Metadata_NCBI_Gene_ID"][0])
        manifest[gene_id] = (offset, offset + group.height)
        offset += group.height
    return treated.select(feature_columns).to_numpy(), manifest


def _write_phase5_array(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array)


def _save_phase5_shared_state(
    *,
    work_dir: Path,
    orf_centroids: np.ndarray,
    crispr_centroids: np.ndarray,
    orf_treated_matrix: np.ndarray,
    crispr_treated_matrix: np.ndarray,
) -> Path:
    work_dir.mkdir(parents=True, exist_ok=True)
    array_paths = {
        "orf_centroids": str(work_dir / "orf_centroids.npy"),
        "crispr_centroids": str(work_dir / "crispr_centroids.npy"),
        "orf_treated_matrix": str(work_dir / "orf_treated_matrix.npy"),
        "crispr_treated_matrix": str(work_dir / "crispr_treated_matrix.npy"),
    }
    _write_phase5_array(Path(array_paths["orf_centroids"]), orf_centroids)
    _write_phase5_array(Path(array_paths["crispr_centroids"]), crispr_centroids)
    _write_phase5_array(Path(array_paths["orf_treated_matrix"]), orf_treated_matrix)
    _write_phase5_array(Path(array_paths["crispr_treated_matrix"]), crispr_treated_matrix)
    manifest_path = work_dir / "manifest.pkl"
    with manifest_path.open("wb") as handle:
        pickle.dump(array_paths, handle)
    return manifest_path


def _phase5_worker_init(
    manifest_path: str,
    decile_summaries: dict[tuple[int | None, int | None], NullSummary],
    quintile_summaries: dict[tuple[int | None, int | None], NullSummary],
    orf_decile_edges: list[float],
    crispr_decile_edges: list[float],
    numpy_threads_per_worker: int,
) -> None:
    global _PHASE5_SHARED
    set_numpy_worker_env(numpy_threads_per_worker)
    with Path(manifest_path).open("rb") as handle:
        array_paths = pickle.load(handle)
    _PHASE5_SHARED = {
        "orf_centroids": np.load(array_paths["orf_centroids"], mmap_mode="r"),
        "crispr_centroids": np.load(array_paths["crispr_centroids"], mmap_mode="r"),
        "orf_treated_matrix": np.load(array_paths["orf_treated_matrix"], mmap_mode="r"),
        "crispr_treated_matrix": np.load(array_paths["crispr_treated_matrix"], mmap_mode="r"),
        "decile_summaries": decile_summaries,
        "quintile_summaries": quintile_summaries,
        "orf_decile_edges": orf_decile_edges,
        "crispr_decile_edges": crispr_decile_edges,
    }


def _compute_geometry_row_parallel(
    record: dict[str, Any],
    *,
    config: GeometryConfig,
    strength_threshold: float,
    rep_threshold: float,
) -> dict[str, Any]:
    if _PHASE5_SHARED is None:  # pragma: no cover - defensive worker setup guard
        raise RuntimeError("Phase 05 worker shared state was not initialized.")
    orf_idx = int(record["orf_idx"])
    crispr_idx = int(record["crispr_idx"])
    raw = _cosine(
        _PHASE5_SHARED["orf_centroids"][orf_idx].tolist(),
        _PHASE5_SHARED["crispr_centroids"][crispr_idx].tolist(),
    )
    fallback_level, null_summary = _null_choice(
        int(record["orf_decile_bin"]),
        int(record["crispr_decile_bin"]),
        _PHASE5_SHARED["decile_summaries"],
        _PHASE5_SHARED["quintile_summaries"],
        config.min_pairs_per_null_cell,
    )
    sign_z = (raw - null_summary.mean) / max(null_summary.std, config.null_std_floor)
    delta_activity = float(record["orf_strength_z"]) - float(record["crispr_strength_z"])
    regime = assign_regime(
        orf_active=bool(record["orf_active"]),
        crispr_active=bool(record["crispr_active"]),
        sign_z=sign_z,
        delta_activity=delta_activity,
        sign_z_threshold=config.sign_z_threshold,
        asymmetry_threshold=config.asymmetry_threshold,
    )
    orf_values = (
        np.asarray(
            _PHASE5_SHARED["orf_treated_matrix"][int(record["orf_rep_start"]) : int(record["orf_rep_stop"])],
            dtype=float,
        )
        if int(record["orf_rep_stop"]) > int(record["orf_rep_start"])
        else np.zeros((0, _PHASE5_SHARED["orf_treated_matrix"].shape[1]), dtype=float)
    )
    crispr_values = (
        np.asarray(
            _PHASE5_SHARED["crispr_treated_matrix"][int(record["crispr_rep_start"]) : int(record["crispr_rep_stop"])],
            dtype=float,
        )
        if int(record["crispr_rep_stop"]) > int(record["crispr_rep_start"])
        else np.zeros((0, _PHASE5_SHARED["crispr_treated_matrix"].shape[1]), dtype=float)
    )
    boot = _bootstrap_probabilities(
        {
            "gene_id": record["gene_id"],
            "orf_null_strength_median": float(record["orf_null_strength_median"]),
            "orf_null_strength_mad": float(record["orf_null_strength_mad"]),
            "crispr_null_strength_median": float(record["crispr_null_strength_median"]),
            "crispr_null_strength_mad": float(record["crispr_null_strength_mad"]),
            "orf_null_rep_median": float(record["orf_null_rep_median"]),
            "orf_null_rep_mad": float(record["orf_null_rep_mad"]),
            "crispr_null_rep_median": float(record["crispr_null_rep_median"]),
            "crispr_null_rep_mad": float(record["crispr_null_rep_mad"]),
            "strength_threshold": strength_threshold,
            "rep_threshold": rep_threshold,
        },
        orf_values,
        crispr_values,
        config,
        orf_decile_edges=_PHASE5_SHARED["orf_decile_edges"],
        crispr_decile_edges=_PHASE5_SHARED["crispr_decile_edges"],
        decile_summaries=_PHASE5_SHARED["decile_summaries"],
        quintile_summaries=_PHASE5_SHARED["quintile_summaries"],
    )
    consensus_label = max(boot, key=boot.get) if boot else "ambiguous"
    bootstrap_consensus = boot.get(consensus_label, 0.0) if boot else 0.0
    final_label = consensus_label if bootstrap_consensus >= config.consensus_threshold else "ambiguous"
    return {
        "gene_id": record["gene_id"],
        "display_symbol": record["display_symbol"],
        "canonical_symbol": record["canonical_symbol"],
        "pipeline": record["pipeline"],
        "orf_strength_z": float(record["orf_strength_z"]),
        "crispr_strength_z": float(record["crispr_strength_z"]),
        "orf_rep_z": float(record["orf_rep_z"]),
        "crispr_rep_z": float(record["crispr_rep_z"]),
        "orf_active": bool(record["orf_active"]),
        "crispr_active": bool(record["crispr_active"]),
        "delta_activity": delta_activity,
        "raw_signed_cosine": raw,
        "sign_null_orf_bin": int(record["orf_decile_bin"]),
        "sign_null_crispr_bin": int(record["crispr_decile_bin"]),
        "sign_null_fallback_level": fallback_level,
        "sign_null_mean": null_summary.mean,
        "sign_null_std": max(null_summary.std, config.null_std_floor),
        "sign_z": sign_z,
        "regime_label_prebootstrap": regime,
        "bootstrap_prob_aligned": boot["aligned"],
        "bootstrap_prob_inverse": boot["inverse"],
        "bootstrap_prob_orthogonal": boot["orthogonal"],
        "bootstrap_prob_asym_orf": boot["asymmetric_ORF"],
        "bootstrap_prob_asym_crispr": boot["asymmetric_CRISPR"],
        "bootstrap_prob_inactive": boot["inactive"],
        "bootstrap_consensus_label": consensus_label,
        "bootstrap_consensus": bootstrap_consensus,
        "prebootstrap_matches_consensus": regime == consensus_label,
        "regime_label": final_label,
    }


def _phase5_worker_chunk(
    chunk_records: list[dict[str, Any]],
    *,
    config: GeometryConfig,
    strength_threshold: float,
    rep_threshold: float,
) -> list[dict[str, Any]]:
    return [
        _compute_geometry_row_parallel(
            record,
            config=config,
            strength_threshold=strength_threshold,
            rep_threshold=rep_threshold,
        )
        for record in chunk_records
    ]


def _chunk_phase5_records(records: list[dict[str, Any]], chunk_size: int) -> list[list[dict[str, Any]]]:
    ordered = sorted(records, key=lambda record: str(record["gene_id"]))
    return [ordered[start : start + chunk_size] for start in range(0, len(ordered), chunk_size)]


def compute_signed_geometry(
    orf_stats: pl.DataFrame,
    crispr_stats: pl.DataFrame,
    centered_orf: pl.DataFrame,
    centered_crispr: pl.DataFrame,
    feature_columns: list[str],
    *,
    strength_threshold: float,
    rep_threshold: float,
    config: GeometryConfig,
    compute_config: ComputeConfig | None = None,
    run_dir: Path | None = None,
) -> pl.DataFrame:
    schema = {
        "gene_id": pl.String,
        "display_symbol": pl.String,
        "canonical_symbol": pl.String,
        "pipeline": pl.String,
        "orf_strength_z": pl.Float64,
        "crispr_strength_z": pl.Float64,
        "orf_rep_z": pl.Float64,
        "crispr_rep_z": pl.Float64,
        "orf_active": pl.Boolean,
        "crispr_active": pl.Boolean,
        "delta_activity": pl.Float64,
        "raw_signed_cosine": pl.Float64,
        "sign_null_orf_bin": pl.Int64,
        "sign_null_crispr_bin": pl.Int64,
        "sign_null_fallback_level": pl.String,
        "sign_null_mean": pl.Float64,
        "sign_null_std": pl.Float64,
        "sign_z": pl.Float64,
        "regime_label_prebootstrap": pl.String,
        "bootstrap_prob_aligned": pl.Float64,
        "bootstrap_prob_inverse": pl.Float64,
        "bootstrap_prob_orthogonal": pl.Float64,
        "bootstrap_prob_asym_orf": pl.Float64,
        "bootstrap_prob_asym_crispr": pl.Float64,
        "bootstrap_prob_inactive": pl.Float64,
        "bootstrap_consensus_label": pl.String,
        "bootstrap_consensus": pl.Float64,
        "prebootstrap_matches_consensus": pl.Boolean,
        "regime_label": pl.String,
    }
    if orf_stats.is_empty() or crispr_stats.is_empty():
        return pl.DataFrame(schema=schema)

    orf_deciles = assign_strength_bins(orf_stats, "strength_z", config.activity_deciles, "orf_decile_bin")
    crispr_deciles = assign_strength_bins(crispr_stats, "strength_z", config.activity_deciles, "crispr_decile_bin")
    orf_quintiles = assign_strength_bins(orf_stats, "strength_z", 5, "orf_quintile_bin")
    crispr_quintiles = assign_strength_bins(crispr_stats, "strength_z", 5, "crispr_quintile_bin")
    orf_decile_edges = compute_bin_edges(orf_stats, "strength_z", config.activity_deciles)
    crispr_decile_edges = compute_bin_edges(crispr_stats, "strength_z", config.activity_deciles)

    decile_summaries = precompute_null_summaries(
        orf_deciles,
        crispr_deciles,
        orf_bin_column="orf_decile_bin",
        crispr_bin_column="crispr_decile_bin",
        n_samples=config.n_cross_null_samples_per_cell,
        random_seed=config.random_seed,
    )
    quintile_summaries = precompute_null_summaries(
        orf_quintiles,
        crispr_quintiles,
        orf_bin_column="orf_quintile_bin",
        crispr_bin_column="crispr_quintile_bin",
        n_samples=config.n_cross_null_samples_per_cell,
        random_seed=config.random_seed,
    )

    paired = (
        orf_deciles.join(orf_quintiles.select(["gene_id", "orf_quintile_bin"]), on="gene_id", how="left")
        .join(
            crispr_deciles.join(
                crispr_quintiles.select(["gene_id", "crispr_quintile_bin"]), on="gene_id", how="left"
            ),
            on="gene_id",
            how="inner",
            suffix="_crispr",
        )
    )

    if (
        compute_config is not None
        and compute_config.parallel_backend == "process_pool"
        and compute_config.phase5_workers > 1
        and run_dir is not None
        and paired.height > 1
    ):
        orf_centroids = np.asarray(orf_deciles["centroid_vector"].to_list(), dtype=float)
        crispr_centroids = np.asarray(crispr_deciles["centroid_vector"].to_list(), dtype=float)
        orf_treated_matrix, orf_replicates = _build_replicate_manifest(centered_orf, feature_columns)
        crispr_treated_matrix, crispr_replicates = _build_replicate_manifest(centered_crispr, feature_columns)
        orf_index = {str(gene_id): idx for idx, gene_id in enumerate(orf_deciles["gene_id"].to_list())}
        crispr_index = {str(gene_id): idx for idx, gene_id in enumerate(crispr_deciles["gene_id"].to_list())}
        pair_records = []
        for row in paired.to_dicts():
            gene_id = str(row["gene_id"])
            orf_rep_start, orf_rep_stop = orf_replicates.get(gene_id, (0, 0))
            crispr_rep_start, crispr_rep_stop = crispr_replicates.get(gene_id, (0, 0))
            pair_records.append(
                {
                    "gene_id": gene_id,
                    "display_symbol": row["display_symbol"],
                    "canonical_symbol": row.get("canonical_symbol") or row.get("canonical_symbol_crispr"),
                    "pipeline": row["pipeline"],
                    "orf_idx": orf_index[gene_id],
                    "crispr_idx": crispr_index[gene_id],
                    "orf_strength_z": row["strength_z"],
                    "crispr_strength_z": row["strength_z_crispr"],
                    "orf_rep_z": row["rep_z"],
                    "crispr_rep_z": row["rep_z_crispr"],
                    "orf_active": row["active_bool"],
                    "crispr_active": row["active_bool_crispr"],
                    "orf_null_strength_median": row["null_strength_median"],
                    "orf_null_strength_mad": row["null_strength_mad"],
                    "crispr_null_strength_median": row["null_strength_median_crispr"],
                    "crispr_null_strength_mad": row["null_strength_mad_crispr"],
                    "orf_null_rep_median": row["null_rep_median"],
                    "orf_null_rep_mad": row["null_rep_mad"],
                    "crispr_null_rep_median": row["null_rep_median_crispr"],
                    "crispr_null_rep_mad": row["null_rep_mad_crispr"],
                    "orf_decile_bin": row["orf_decile_bin"],
                    "crispr_decile_bin": row["crispr_decile_bin"],
                    "orf_rep_start": orf_rep_start,
                    "orf_rep_stop": orf_rep_stop,
                    "crispr_rep_start": crispr_rep_start,
                    "crispr_rep_stop": crispr_rep_stop,
                }
            )
        phase_work_dir = compute_config.mmap_root / "phase05"
        manifest_path = _save_phase5_shared_state(
            work_dir=phase_work_dir,
            orf_centroids=orf_centroids,
            crispr_centroids=crispr_centroids,
            orf_treated_matrix=orf_treated_matrix,
            crispr_treated_matrix=crispr_treated_matrix,
        )
        set_numpy_worker_env(compute_config.numpy_threads_per_worker)
        rows: list[dict[str, Any]] = []
        with ProcessPoolExecutor(
            max_workers=compute_config.phase5_workers,
            mp_context=mp.get_context("spawn"),
            initializer=_phase5_worker_init,
            initargs=(
                str(manifest_path),
                decile_summaries,
                quintile_summaries,
                orf_decile_edges,
                crispr_decile_edges,
                compute_config.numpy_threads_per_worker,
            ),
        ) as executor:
            futures = [
                executor.submit(
                    _phase5_worker_chunk,
                    chunk,
                    config=config,
                    strength_threshold=strength_threshold,
                    rep_threshold=rep_threshold,
                )
                for chunk in _chunk_phase5_records(pair_records, compute_config.phase5_gene_chunk_size)
            ]
            for future in as_completed(futures):
                rows.extend(future.result())
    else:
        centered_orf_groups = {
            str(group["Metadata_NCBI_Gene_ID"][0]): group.select(feature_columns).to_numpy()
            for group in centered_orf
            .filter(~pl.col("is_negative_control") & pl.col("Metadata_NCBI_Gene_ID").is_not_null())
            .sort(["Metadata_NCBI_Gene_ID", "Metadata_Plate", "Metadata_Well"])
            .partition_by("Metadata_NCBI_Gene_ID", maintain_order=True)
        }
        centered_crispr_groups = {
            str(group["Metadata_NCBI_Gene_ID"][0]): group.select(feature_columns).to_numpy()
            for group in centered_crispr
            .filter(~pl.col("is_negative_control") & pl.col("Metadata_NCBI_Gene_ID").is_not_null())
            .sort(["Metadata_NCBI_Gene_ID", "Metadata_Plate", "Metadata_Well"])
            .partition_by("Metadata_NCBI_Gene_ID", maintain_order=True)
        }

        rows = []
        for row in paired.to_dicts():
            raw = _cosine(row["centroid_vector"], row["centroid_vector_crispr"])
            fallback_level, null_summary = _null_choice(
                row["orf_decile_bin"],
                row["crispr_decile_bin"],
                decile_summaries,
                quintile_summaries,
                config.min_pairs_per_null_cell,
            )
            sign_z = (raw - null_summary.mean) / max(null_summary.std, config.null_std_floor)
            delta_activity = row["strength_z"] - row["strength_z_crispr"]
            regime = assign_regime(
                orf_active=row["active_bool"],
                crispr_active=row["active_bool_crispr"],
                sign_z=sign_z,
                delta_activity=delta_activity,
                sign_z_threshold=config.sign_z_threshold,
                asymmetry_threshold=config.asymmetry_threshold,
            )
            orf_values = centered_orf_groups.get(str(row["gene_id"]))
            crispr_values = centered_crispr_groups.get(str(row["gene_id"]))
            boot = _bootstrap_probabilities(
                {
                    "gene_id": row["gene_id"],
                    "orf_null_strength_median": row["null_strength_median"],
                    "orf_null_strength_mad": row["null_strength_mad"],
                    "crispr_null_strength_median": row["null_strength_median_crispr"],
                    "crispr_null_strength_mad": row["null_strength_mad_crispr"],
                    "orf_null_rep_median": row["null_rep_median"],
                    "orf_null_rep_mad": row["null_rep_mad"],
                    "crispr_null_rep_median": row["null_rep_median_crispr"],
                    "crispr_null_rep_mad": row["null_rep_mad_crispr"],
                    "strength_threshold": strength_threshold,
                    "rep_threshold": rep_threshold,
                },
                orf_values if orf_values is not None else np.zeros((0, len(feature_columns)), dtype=float),
                crispr_values if crispr_values is not None else np.zeros((0, len(feature_columns)), dtype=float),
                config,
                orf_decile_edges=orf_decile_edges,
                crispr_decile_edges=crispr_decile_edges,
                decile_summaries=decile_summaries,
                quintile_summaries=quintile_summaries,
            )
            consensus_label = max(boot, key=boot.get) if boot else "ambiguous"
            bootstrap_consensus = boot.get(consensus_label, 0.0) if boot else 0.0
            final_label = consensus_label if bootstrap_consensus >= config.consensus_threshold else "ambiguous"
            rows.append(
                {
                    "gene_id": row["gene_id"],
                    "display_symbol": row["display_symbol"],
                    "canonical_symbol": row.get("canonical_symbol") or row.get("canonical_symbol_crispr"),
                    "pipeline": row["pipeline"],
                    "orf_strength_z": row["strength_z"],
                    "crispr_strength_z": row["strength_z_crispr"],
                    "orf_rep_z": row["rep_z"],
                    "crispr_rep_z": row["rep_z_crispr"],
                    "orf_active": row["active_bool"],
                    "crispr_active": row["active_bool_crispr"],
                    "delta_activity": delta_activity,
                    "raw_signed_cosine": raw,
                    "sign_null_orf_bin": row["orf_decile_bin"],
                    "sign_null_crispr_bin": row["crispr_decile_bin"],
                    "sign_null_fallback_level": fallback_level,
                    "sign_null_mean": null_summary.mean,
                    "sign_null_std": max(null_summary.std, config.null_std_floor),
                    "sign_z": sign_z,
                    "regime_label_prebootstrap": regime,
                    "bootstrap_prob_aligned": boot["aligned"],
                    "bootstrap_prob_inverse": boot["inverse"],
                    "bootstrap_prob_orthogonal": boot["orthogonal"],
                    "bootstrap_prob_asym_orf": boot["asymmetric_ORF"],
                    "bootstrap_prob_asym_crispr": boot["asymmetric_CRISPR"],
                    "bootstrap_prob_inactive": boot["inactive"],
                    "bootstrap_consensus_label": consensus_label,
                    "bootstrap_consensus": bootstrap_consensus,
                    "prebootstrap_matches_consensus": regime == consensus_label,
                    "regime_label": final_label,
                }
            )
    return pl.DataFrame(rows, schema=schema).sort("gene_id") if rows else pl.DataFrame(schema=schema)


def write_geometry_outputs(frame: pl.DataFrame, output_path: Path, summary_path: Path) -> dict[str, Any]:
    write_parquet_atomic(frame, output_path)
    fallback_counts = (
        {
            str(row["sign_null_fallback_level"]): int(row["count"])
            for row in frame.group_by("sign_null_fallback_level").len(name="count").to_dicts()
        }
        if frame.height
        else {}
    )
    summary = {
        "rows": frame.height,
        "confident_noninactive_count": int(
            frame.filter((pl.col("regime_label") != "ambiguous") & (pl.col("regime_label") != "inactive")).height
        )
        if frame.height
        else 0,
        "null_fallback_counts": fallback_counts,
    }
    write_json_atomic(summary_path, summary)
    return summary
