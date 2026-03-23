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


_PHASE4_SHARED: dict[str, Any] | None = None


def _robust_z(value: float, null_values: np.ndarray, floor: float = 1.0e-6) -> tuple[float, float, float]:
    median = float(np.median(null_values))
    mad = float(np.median(np.abs(null_values - median)))
    z_score = (value - median) / max(mad, floor)
    return z_score, median, mad


@dataclass(frozen=True)
class ActivityConfig:
    n_strength_null_samples: int
    n_rep_null_samples: int
    strength_z_threshold: float
    rep_z_threshold: float
    random_seed: int


def recompute_postfilter_eligibility(
    overlap_df: pl.DataFrame,
    centered_orf: pl.DataFrame,
    centered_crispr: pl.DataFrame,
    *,
    min_wells_per_modality: int,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    def _counts(frame: pl.DataFrame, prefix: str) -> pl.DataFrame:
        if frame.is_empty():
            return pl.DataFrame(
                schema={
                    "gene_id": pl.String,
                    f"n_{prefix}_retained_wells": pl.Int64,
                    f"n_{prefix}_retained_plates": pl.Int64,
                }
            )
        return (
            frame.filter(~pl.col("is_negative_control") & pl.col("Metadata_NCBI_Gene_ID").is_not_null())
            .group_by("Metadata_NCBI_Gene_ID")
            .agg(
                pl.len().alias(f"n_{prefix}_retained_wells"),
                pl.col("Metadata_Plate").n_unique().alias(f"n_{prefix}_retained_plates"),
            )
            .rename({"Metadata_NCBI_Gene_ID": "gene_id"})
        )

    orf_counts = _counts(centered_orf, "orf")
    crispr_counts = _counts(centered_crispr, "crispr")
    updated = (
        overlap_df.join(orf_counts, on="gene_id", how="left")
        .join(crispr_counts, on="gene_id", how="left")
        .with_columns(
            pl.col("n_orf_retained_wells").fill_null(0).cast(pl.Int64),
            pl.col("n_crispr_retained_wells").fill_null(0).cast(pl.Int64),
            pl.col("n_orf_retained_plates").fill_null(0).cast(pl.Int64),
            pl.col("n_crispr_retained_plates").fill_null(0).cast(pl.Int64),
        )
        .with_columns(
            (
                (pl.col("n_orf_retained_wells") >= min_wells_per_modality)
                & (pl.col("n_crispr_retained_wells") >= min_wells_per_modality)
            ).alias("postfilter_keep_for_analysis"),
            (
                (pl.col("n_orf_retained_wells") < max(4, min_wells_per_modality))
                | (pl.col("n_crispr_retained_wells") < max(4, min_wells_per_modality))
            ).alias("postfilter_low_replicate_support"),
        )
        .with_columns(
            pl.when(pl.col("postfilter_keep_for_analysis"))
            .then(pl.lit(None, dtype=pl.String))
            .when(
                (pl.col("n_orf_retained_wells") < min_wells_per_modality)
                & (pl.col("n_crispr_retained_wells") < min_wells_per_modality)
            )
            .then(pl.lit("postfilter_insufficient_wells_both_modalities"))
            .when(pl.col("n_orf_retained_wells") < min_wells_per_modality)
            .then(pl.lit("postfilter_insufficient_orf_wells"))
            .otherwise(pl.lit("postfilter_insufficient_crispr_wells"))
            .alias("postfilter_drop_reason")
        )
        .with_columns(
            pl.col("postfilter_keep_for_analysis").alias("keep_for_analysis"),
            pl.col("postfilter_low_replicate_support").alias("low_replicate_support"),
        )
    )
    qc = (
        updated.select(
            [
                "gene_id",
                "display_symbol",
                "canonical_symbol",
                "n_orf_retained_wells",
                "n_crispr_retained_wells",
                "n_orf_retained_plates",
                "n_crispr_retained_plates",
                "postfilter_keep_for_analysis",
                "postfilter_low_replicate_support",
                "postfilter_drop_reason",
            ]
        )
        .sort("gene_id")
    )
    return updated, qc


def _sample_strength_null(
    plate_counts: list[dict[str, Any]],
    global_controls: np.ndarray,
    control_idx_by_plate: dict[str, np.ndarray],
    control_idx_by_batch: dict[str, np.ndarray],
    rng: np.random.Generator,
    n_samples: int,
) -> np.ndarray:
    if global_controls.shape[0] == 0:
        return np.zeros(n_samples, dtype=float)
    results = np.zeros(n_samples, dtype=float)
    all_control_idx = np.arange(global_controls.shape[0], dtype=np.int64)
    control_indices: list[tuple[np.ndarray, int]] = []
    for item in plate_counts:
        plate = item["Metadata_Plate"]
        batch = item["Metadata_Batch"]
        count = int(item["len"])
        candidate_idx = control_idx_by_plate.get(plate)
        if candidate_idx is None or candidate_idx.size == 0:
            candidate_idx = control_idx_by_batch.get(batch, all_control_idx)
        control_indices.append((candidate_idx, count))
    for idx in range(n_samples):
        sampled_parts: list[np.ndarray] = []
        for candidate_idx, count in control_indices:
            indices = rng.choice(candidate_idx, size=count, replace=True)
            sampled_parts.append(global_controls[indices])
        sampled = np.vstack(sampled_parts)
        results[idx] = centroid_l2(coordinate_median(sampled))
    return results


def _sample_rep_null(
    gene_id: str,
    plate_counts: list[dict[str, Any]],
    treated_matrix: np.ndarray,
    treated_gene_ids: np.ndarray,
    treated_indices_by_plate: dict[str, np.ndarray],
    rng: np.random.Generator,
    n_samples: int,
) -> np.ndarray:
    n_wells = sum(int(item["len"]) for item in plate_counts)
    global_candidate_idx = np.flatnonzero(treated_gene_ids != gene_id)
    if global_candidate_idx.size == 0:
        return np.zeros(n_samples, dtype=float)

    results = np.zeros(n_samples, dtype=float)
    plate_indices: list[tuple[np.ndarray, int]] = []
    for item in plate_counts:
        plate = item["Metadata_Plate"]
        count = int(item["len"])
        pool = treated_indices_by_plate.get(plate)
        if pool is None or pool.size == 0:
            candidate_idx = global_candidate_idx
        else:
            candidate_idx = pool[treated_gene_ids[pool] != gene_id]
            if candidate_idx.size < count:
                candidate_idx = global_candidate_idx
        plate_indices.append((candidate_idx, count))
    for idx in range(n_samples):
        sampled_parts: list[np.ndarray] = []
        for candidate_idx, count in plate_indices:
            indices = rng.choice(candidate_idx, size=count, replace=True)
            sampled_parts.append(treated_matrix[indices])
        sampled = np.vstack(sampled_parts)
        if sampled.shape[0] != n_wells:
            extra = rng.choice(global_candidate_idx, size=n_wells - sampled.shape[0], replace=True)
            sampled = np.vstack([sampled, treated_matrix[extra]])
        results[idx] = pairwise_cosine_median(sampled)
    return results


def _seeded_rng(random_seed: int, modality: str, gene_id: str) -> np.random.Generator:
    digest = hashlib.sha256(f"{random_seed}:{modality}:{gene_id}".encode("utf-8")).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big", signed=False))


def _unicode_array(values: list[Any]) -> np.ndarray:
    return np.asarray(["" if value is None else str(value) for value in values], dtype=np.str_)


def _build_index_lookup(values: np.ndarray) -> dict[str, np.ndarray]:
    if values.size == 0:
        return {}
    return {str(value): np.flatnonzero(values == value) for value in np.unique(values)}


def _build_phase4_shared_state(
    treated: pl.DataFrame,
    negative_controls: pl.DataFrame,
    feature_columns: list[str],
) -> dict[str, Any]:
    treated_matrix = treated.select(feature_columns).to_numpy()
    treated_gene_ids = _unicode_array(treated["Metadata_NCBI_Gene_ID"].to_list())
    treated_plate_values = _unicode_array(treated["Metadata_Plate"].to_list())
    global_controls = negative_controls.select(feature_columns).to_numpy()
    control_plate_values = _unicode_array(negative_controls["Metadata_Plate"].to_list())
    control_batch_values = _unicode_array(negative_controls["Metadata_Batch"].to_list())
    return {
        "treated_matrix": treated_matrix,
        "treated_gene_ids": treated_gene_ids,
        "treated_indices_by_plate": _build_index_lookup(treated_plate_values),
        "global_controls": global_controls,
        "control_idx_by_plate": _build_index_lookup(control_plate_values),
        "control_idx_by_batch": _build_index_lookup(control_batch_values),
    }


def _build_gene_records(
    treated: pl.DataFrame,
    overlap_meta: pl.DataFrame,
) -> list[dict[str, Any]]:
    overlap_meta_map = {str(row["gene_id"]): row for row in overlap_meta.to_dicts()}
    records: list[dict[str, Any]] = []
    offset = 0
    for group in treated.partition_by("Metadata_NCBI_Gene_ID", maintain_order=True):
        gene_id = str(group["Metadata_NCBI_Gene_ID"][0])
        group_height = group.height
        if gene_id in overlap_meta_map:
            meta = overlap_meta_map[gene_id]
            plate_counts = tuple(
                {
                    "Metadata_Plate": str(row["Metadata_Plate"]),
                    "Metadata_Batch": str(row["Metadata_Batch"]),
                    "len": int(row["len"]),
                }
                for row in group.group_by(["Metadata_Plate", "Metadata_Batch"]).len().sort(["Metadata_Plate", "Metadata_Batch"]).to_dicts()
            )
            records.append(
                {
                    "gene_id": gene_id,
                    "display_symbol": meta["display_symbol"],
                    "canonical_symbol": meta["canonical_symbol"],
                    "low_replicate_support": bool(meta["low_replicate_support"]),
                    "start_idx": offset,
                    "stop_idx": offset + group_height,
                    "n_wells": group_height,
                    "n_plates": int(group["Metadata_Plate"].n_unique()),
                    "plate_counts": plate_counts,
                }
            )
        offset += group_height
    return records


def _compute_gene_stats_row(
    record: dict[str, Any],
    shared_state: dict[str, Any],
    *,
    pipeline: str,
    modality: str,
    config: ActivityConfig,
) -> dict[str, Any]:
    gene_id = str(record["gene_id"])
    rng = _seeded_rng(config.random_seed, modality, gene_id)
    start_idx = int(record["start_idx"])
    stop_idx = int(record["stop_idx"])
    values = np.asarray(shared_state["treated_matrix"][start_idx:stop_idx], dtype=float)
    centroid = coordinate_median(values)
    centroid_norm = centroid_l2(centroid)
    strength_null = _sample_strength_null(
        list(record["plate_counts"]),
        shared_state["global_controls"],
        shared_state["control_idx_by_plate"],
        shared_state["control_idx_by_batch"],
        rng,
        config.n_strength_null_samples,
    )
    strength_z, null_strength_median, null_strength_mad = _robust_z(centroid_norm, strength_null)

    rep_similarity = pairwise_cosine_median(values)
    rep_null = _sample_rep_null(
        gene_id,
        list(record["plate_counts"]),
        shared_state["treated_matrix"],
        shared_state["treated_gene_ids"],
        shared_state["treated_indices_by_plate"],
        rng,
        config.n_rep_null_samples,
    )
    rep_z, null_rep_median, null_rep_mad = _robust_z(rep_similarity, rep_null)
    return {
        "gene_id": gene_id,
        "display_symbol": record["display_symbol"],
        "canonical_symbol": record["canonical_symbol"],
        "modality": modality,
        "pipeline": pipeline,
        "n_wells": int(record["n_wells"]),
        "n_plates": int(record["n_plates"]),
        "centroid_l2": centroid_norm,
        "centroid_vector": centroid.tolist(),
        "strength_z": strength_z,
        "rep_similarity": rep_similarity,
        "rep_z": rep_z,
        "active_bool": bool(
            strength_z >= config.strength_z_threshold and rep_z >= config.rep_z_threshold
        ),
        "null_strength_median": null_strength_median,
        "null_strength_mad": null_strength_mad,
        "null_rep_median": null_rep_median,
        "null_rep_mad": null_rep_mad,
        "low_replicate_support": bool(record["low_replicate_support"]),
    }


def _write_memmap_array(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array)


def _save_phase4_shared_state(shared_state: dict[str, Any], work_dir: Path) -> Path:
    work_dir.mkdir(parents=True, exist_ok=True)
    array_paths: dict[str, str] = {}
    for name in (
        "treated_matrix",
        "treated_gene_ids",
        "treated_plate_values",
        "global_controls",
        "control_plate_values",
        "control_batch_values",
    ):
        path = work_dir / f"{name}.npy"
        _write_memmap_array(path, np.asarray(shared_state[name]))
        array_paths[name] = str(path)
    manifest_path = work_dir / "manifest.pkl"
    with manifest_path.open("wb") as handle:
        pickle.dump(array_paths, handle)
    return manifest_path


def _phase4_worker_init(manifest_path: str, numpy_threads_per_worker: int) -> None:
    global _PHASE4_SHARED
    set_numpy_worker_env(numpy_threads_per_worker)
    with Path(manifest_path).open("rb") as handle:
        array_paths = pickle.load(handle)
    treated_gene_ids = np.load(array_paths["treated_gene_ids"], mmap_mode="r")
    treated_plate_values = np.load(array_paths["treated_plate_values"], mmap_mode="r")
    control_plate_values = np.load(array_paths["control_plate_values"], mmap_mode="r")
    control_batch_values = np.load(array_paths["control_batch_values"], mmap_mode="r")
    _PHASE4_SHARED = {
        "treated_matrix": np.load(array_paths["treated_matrix"], mmap_mode="r"),
        "treated_gene_ids": treated_gene_ids,
        "treated_indices_by_plate": _build_index_lookup(treated_plate_values),
        "global_controls": np.load(array_paths["global_controls"], mmap_mode="r"),
        "control_idx_by_plate": _build_index_lookup(control_plate_values),
        "control_idx_by_batch": _build_index_lookup(control_batch_values),
    }


def _phase4_worker_chunk(
    chunk_records: list[dict[str, Any]],
    *,
    pipeline: str,
    modality: str,
    config: ActivityConfig,
) -> list[dict[str, Any]]:
    if _PHASE4_SHARED is None:  # pragma: no cover - defensive worker setup guard
        raise RuntimeError("Phase 04 worker shared state was not initialized.")
    return [
        _compute_gene_stats_row(
            record,
            _PHASE4_SHARED,
            pipeline=pipeline,
            modality=modality,
            config=config,
        )
        for record in chunk_records
    ]


def _chunk_records(records: list[dict[str, Any]], chunk_size: int) -> list[list[dict[str, Any]]]:
    ordered = sorted(records, key=lambda record: (-int(record["n_wells"]), str(record["gene_id"])))
    return [ordered[start : start + chunk_size] for start in range(0, len(ordered), chunk_size)]


def _compute_gene_modality_stats_parallel(
    records: list[dict[str, Any]],
    shared_state: dict[str, Any],
    *,
    pipeline: str,
    modality: str,
    config: ActivityConfig,
    compute_config: ComputeConfig,
    run_dir: Path,
) -> list[dict[str, Any]]:
    if not records:
        return []
    phase_work_dir = compute_config.mmap_root / "phase04" / modality
    treated_plate_values_list = [""] * int(shared_state["treated_gene_ids"].shape[0])
    for plate, indices in shared_state["treated_indices_by_plate"].items():
        for index in indices.tolist():
            treated_plate_values_list[int(index)] = str(plate)
    control_plate_values_list = [""] * int(shared_state["global_controls"].shape[0])
    for plate, indices in shared_state["control_idx_by_plate"].items():
        for index in indices.tolist():
            control_plate_values_list[int(index)] = str(plate)
    control_batch_values_list = [""] * int(shared_state["global_controls"].shape[0])
    for batch, indices in shared_state["control_idx_by_batch"].items():
        for index in indices.tolist():
            control_batch_values_list[int(index)] = str(batch)
    shared_for_disk = {
        "treated_matrix": shared_state["treated_matrix"],
        "treated_gene_ids": shared_state["treated_gene_ids"],
        "treated_plate_values": _unicode_array(treated_plate_values_list),
        "global_controls": shared_state["global_controls"],
        "control_plate_values": _unicode_array(control_plate_values_list),
        "control_batch_values": _unicode_array(control_batch_values_list),
    }
    manifest_path = _save_phase4_shared_state(shared_for_disk, phase_work_dir)
    set_numpy_worker_env(compute_config.numpy_threads_per_worker)
    chunks = _chunk_records(records, compute_config.phase4_gene_chunk_size)
    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(
        max_workers=compute_config.phase4_workers,
        mp_context=mp.get_context("spawn"),
        initializer=_phase4_worker_init,
        initargs=(str(manifest_path), compute_config.numpy_threads_per_worker),
    ) as executor:
        futures = [
            executor.submit(
                _phase4_worker_chunk,
                chunk,
                pipeline=pipeline,
                modality=modality,
                config=config,
            )
            for chunk in chunks
        ]
        for future in as_completed(futures):
            rows.extend(future.result())
    return rows


def compute_gene_modality_stats(
    centered_df: pl.DataFrame,
    overlap_df: pl.DataFrame,
    feature_columns: list[str],
    *,
    pipeline: str,
    modality: str,
    config: ActivityConfig,
    compute_config: ComputeConfig | None = None,
    run_dir: Path | None = None,
) -> pl.DataFrame:
    centered_sorted = centered_df.sort(
        ["Metadata_NCBI_Gene_ID", "Metadata_Plate", "Metadata_Well"],
        nulls_last=True,
    )
    treated = centered_sorted.filter(~pl.col("is_negative_control") & pl.col("Metadata_NCBI_Gene_ID").is_not_null())
    negative_controls = centered_sorted.filter(pl.col("is_negative_control"))
    eligible_overlap = (
        overlap_df.filter(pl.col("postfilter_keep_for_analysis"))
        if "postfilter_keep_for_analysis" in overlap_df.columns
        else overlap_df.filter(pl.col("keep_for_analysis"))
        if "keep_for_analysis" in overlap_df.columns
        else overlap_df
    )
    if "canonical_symbol" not in eligible_overlap.columns:
        eligible_overlap = eligible_overlap.with_columns(pl.lit(None).alias("canonical_symbol"))
    overlap_meta = eligible_overlap.select(["gene_id", "display_symbol", "canonical_symbol", "low_replicate_support"])
    records = _build_gene_records(treated, overlap_meta)
    shared_state = _build_phase4_shared_state(treated, negative_controls, feature_columns)
    if (
        compute_config is not None
        and compute_config.parallel_backend == "process_pool"
        and compute_config.phase4_workers > 1
        and run_dir is not None
        and len(records) > 1
    ):
        rows = _compute_gene_modality_stats_parallel(
            records,
            shared_state,
            pipeline=pipeline,
            modality=modality,
            config=config,
            compute_config=compute_config,
            run_dir=run_dir,
        )
    else:
        rows = [
            _compute_gene_stats_row(
                record,
                shared_state,
                pipeline=pipeline,
                modality=modality,
                config=config,
            )
            for record in records
        ]

    if not rows:
        return pl.DataFrame(
            schema={
                "gene_id": pl.String,
                "display_symbol": pl.String,
                "canonical_symbol": pl.String,
                "low_replicate_support": pl.Boolean,
                "modality": pl.String,
                "pipeline": pl.String,
                "n_wells": pl.Int64,
                "n_plates": pl.Int64,
                "centroid_l2": pl.Float64,
                "centroid_vector": pl.List(pl.Float64),
                "strength_z": pl.Float64,
                "rep_similarity": pl.Float64,
                "rep_z": pl.Float64,
                "active_bool": pl.Boolean,
                "null_strength_median": pl.Float64,
                "null_strength_mad": pl.Float64,
                "null_rep_median": pl.Float64,
                "null_rep_mad": pl.Float64,
            }
        )

    result = pl.DataFrame(rows).join(overlap_meta, on="gene_id", how="left")
    return result.select(
        "gene_id",
        "display_symbol",
        "canonical_symbol",
        "modality",
        "pipeline",
        "n_wells",
        "n_plates",
        "centroid_l2",
        "centroid_vector",
        "strength_z",
        "rep_similarity",
        "rep_z",
        "active_bool",
        "null_strength_median",
        "null_strength_mad",
        "null_rep_median",
        "null_rep_mad",
        "low_replicate_support",
    ).sort("gene_id")


def write_gene_stats_outputs(frame: pl.DataFrame, output_path: Path, summary_path: Path) -> dict[str, Any]:
    write_parquet_atomic(frame, output_path)
    summary = {
        "rows": frame.height,
        "active_count": int(frame["active_bool"].sum()) if frame.height else 0,
    }
    write_json_atomic(summary_path, summary)
    return summary
