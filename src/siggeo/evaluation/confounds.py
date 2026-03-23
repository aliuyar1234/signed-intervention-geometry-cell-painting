from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing
from dataclasses import dataclass
from pathlib import Path
import pickle
from typing import Any

import numpy as np
import polars as pl
import statsmodels.api as sm
from scipy.stats import kruskal, ttest_ind

from siggeo.io.atomic import write_csv_atomic
from siggeo.compute import ComputeConfig, set_numpy_worker_env
from siggeo.gene_stats.centroids import centroid_l2, coordinate_median
from siggeo.gene_stats.reproducibility import pairwise_cosine_median
from siggeo.geometry.nulls import assign_strength_bins
from siggeo.preprocess.artifact_filter import ArtifactFilterConfig, mark_artifact_outliers
from siggeo.geometry.regimes import assign_regime
from siggeo.preprocess.feature_space import find_cell_count_columns


@dataclass(frozen=True)
class ConfoundConfig:
    strength_threshold: float
    rep_threshold: float
    sign_z_threshold: float
    asymmetry_threshold: float
    null_std_floor: float
    random_seed: int
    n_different_gene_samples: int = 5000
    n_bootstrap: int = 0
    consensus_threshold: float = 0.70


_PHASE8_SHARED: dict[str, Any] | None = None


def _cosine(left: list[float], right: list[float]) -> float:
    left_vec = np.array(left, dtype=float)
    right_vec = np.array(right, dtype=float)
    denom = np.linalg.norm(left_vec) * np.linalg.norm(right_vec)
    if denom <= 0:
        return 0.0
    return float(np.dot(left_vec, right_vec) / denom)


def _seeded_rng(random_seed: int, gene_id: str, label: str) -> np.random.Generator:
    digest = hashlib.sha256(f"{random_seed}:{gene_id}:{label}".encode("utf-8")).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big", signed=False))


def _group_gene_frames(frame: pl.DataFrame) -> dict[str, pl.DataFrame]:
    if frame.is_empty():
        return {}
    prepared = (
        frame.filter(pl.col("Metadata_NCBI_Gene_ID").is_not_null())
        .sort(["Metadata_NCBI_Gene_ID", "Metadata_Plate", "Metadata_Well"])
    )
    return {
        str(group["Metadata_NCBI_Gene_ID"][0]): group
        for group in prepared.partition_by("Metadata_NCBI_Gene_ID", maintain_order=True)
    }


def _unicode_array(values: list[Any]) -> np.ndarray:
    width = max((len(str(value)) for value in values), default=1)
    return np.array([str(value) for value in values], dtype=f"<U{max(1, width)}")


def _write_memmap_array(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array)


def _write_progress_state(progress_path: Path | None, payload: dict[str, Any]) -> None:
    if progress_path is None:
        return
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {key: value for key, value in payload.items()}
    serializable["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    temp_path = progress_path.with_suffix(progress_path.suffix + ".tmp")
    temp_path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
    temp_path.replace(progress_path)
    print(json.dumps(serializable), flush=True)


def _chunk_records(records: list[dict[str, Any]], chunk_size: int, *, cost_key: str) -> list[list[dict[str, Any]]]:
    ordered = sorted(records, key=lambda record: (-int(record.get(cost_key, 1)), str(record["gene_id"])))
    return [ordered[start : start + chunk_size] for start in range(0, len(ordered), chunk_size)]


def _build_gene_array_state(
    frame: pl.DataFrame,
    feature_columns: list[str],
    *,
    exclude_artifact_outliers: bool = False,
) -> dict[str, Any]:
    mask = ~pl.col("is_negative_control") & pl.col("Metadata_NCBI_Gene_ID").is_not_null()
    if exclude_artifact_outliers:
        mask = mask & ~pl.col("artifact_outlier")
    prepared = frame.filter(mask).sort(["Metadata_NCBI_Gene_ID", "Metadata_Plate", "Metadata_Well"])
    manifest: dict[str, tuple[int, int]] = {}
    offset = 0
    for group in prepared.partition_by("Metadata_NCBI_Gene_ID", maintain_order=True):
        gene_id = str(group["Metadata_NCBI_Gene_ID"][0])
        manifest[gene_id] = (offset, offset + group.height)
        offset += group.height
    return {
        "matrix": prepared.select(feature_columns).to_numpy(),
        "plate_values": _unicode_array(prepared["Metadata_Plate"].to_list()),
        "manifest": manifest,
    }


def _save_phase8_shared_state(
    *,
    work_dir: Path,
    arrays: dict[str, np.ndarray],
    metadata: dict[str, Any],
) -> Path:
    work_dir.mkdir(parents=True, exist_ok=True)
    array_paths: dict[str, str] = {}
    for name, array in arrays.items():
        path = work_dir / f"{name}.npy"
        _write_memmap_array(path, np.asarray(array))
        array_paths[name] = str(path)
    manifest_path = work_dir / "manifest.pkl"
    with manifest_path.open("wb") as handle:
        pickle.dump({"array_paths": array_paths, "metadata": metadata}, handle)
    return manifest_path


def _phase8_worker_init(manifest_path: str, numpy_threads_per_worker: int) -> None:
    global _PHASE8_SHARED
    set_numpy_worker_env(numpy_threads_per_worker)
    with Path(manifest_path).open("rb") as handle:
        payload = pickle.load(handle)
    state = dict(payload["metadata"])
    for name, path_str in payload["array_paths"].items():
        state[name] = np.load(path_str, mmap_mode="r")
    _PHASE8_SHARED = state


def _sample_different_gene_pairs(
    orf_stats: pl.DataFrame,
    crispr_stats: pl.DataFrame,
    *,
    n_samples: int,
    random_seed: int,
) -> pl.DataFrame:
    rng = np.random.default_rng(random_seed)
    orf_rows = orf_stats.to_dicts()
    crispr_rows = crispr_stats.to_dicts()
    rows: list[dict[str, float | str]] = []
    attempts = 0
    max_attempts = max(n_samples * 10, 1000)
    while len(rows) < n_samples and attempts < max_attempts:
        attempts += 1
        orf_row = orf_rows[int(rng.integers(0, len(orf_rows)))]
        crispr_row = crispr_rows[int(rng.integers(0, len(crispr_rows)))]
        if orf_row["gene_id"] == crispr_row["gene_id"]:
            continue
        mean_strength = (float(orf_row["strength_z"]) + float(crispr_row["strength_z"])) / 2.0
        abs_delta_activity = abs(float(orf_row["strength_z"]) - float(crispr_row["strength_z"]))
        rows.append(
            {
                "orf_gene_id": str(orf_row["gene_id"]),
                "crispr_gene_id": str(crispr_row["gene_id"]),
                "raw_signed_cosine": _cosine(orf_row["centroid_vector"], crispr_row["centroid_vector"]),
                "mean_strength": mean_strength,
                "abs_delta_activity": abs_delta_activity,
            }
        )
    return pl.DataFrame(rows)


def _fit_magnitude_model(
    geometry: pl.DataFrame,
    orf_stats: pl.DataFrame,
    crispr_stats: pl.DataFrame,
    config: ConfoundConfig,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    different_gene = _sample_different_gene_pairs(
        orf_stats,
        crispr_stats,
        n_samples=config.n_different_gene_samples,
        random_seed=config.random_seed,
    )
    design = sm.add_constant(different_gene.select(["mean_strength", "abs_delta_activity"]).to_numpy(), has_constant="add")
    model = sm.OLS(different_gene["raw_signed_cosine"].to_numpy(), design).fit()

    same_gene = geometry.with_columns(
        ((pl.col("orf_strength_z") + pl.col("crispr_strength_z")) / 2.0).alias("mean_strength"),
        pl.col("delta_activity").abs().alias("abs_delta_activity"),
    )
    same_design = sm.add_constant(same_gene.select(["mean_strength", "abs_delta_activity"]).to_numpy(), has_constant="add")
    same_residual = same_gene["raw_signed_cosine"].to_numpy() - model.predict(same_design)

    diff_design = sm.add_constant(different_gene.select(["mean_strength", "abs_delta_activity"]).to_numpy(), has_constant="add")
    diff_residual = different_gene["raw_signed_cosine"].to_numpy() - model.predict(diff_design)
    statistic = float(np.mean(same_residual) - np.mean(diff_residual))
    p_value = float(ttest_ind(same_residual, diff_residual, equal_var=False).pvalue)
    row = {
        "pipeline": str(same_gene["pipeline"][0]) if same_gene.height else "unknown",
        "confound_name": "magnitude_residualization",
        "test_name": "ols_residual_mean_shift_descriptive",
        "statistic_name": "same_minus_null_mean",
        "statistic_value": statistic,
        "p_value": p_value,
        "supports_positive_story": bool(statistic > 0),
        "notes": "Residualized raw_signed_cosine against mean_strength and abs_delta_activity; p-value reported descriptively only.",
    }
    residual_frame = same_gene.with_columns(pl.Series("residual_signed_cosine", same_residual))
    return residual_frame, row


def _regime_prevalence(geometry: pl.DataFrame) -> pl.DataFrame:
    total = geometry.height if geometry.height else 1
    return (
        geometry.group_by(["pipeline", "regime_label"])
        .agg(pl.len().alias("n_genes"))
        .with_columns(
            (pl.col("n_genes") / total).alias("fraction_of_analyzed"),
            pl.col("n_genes").alias("n_regime_members"),
        )
        .sort(["pipeline", "regime_label"])
    )


def _strength_decile_prevalence_summary(geometry: pl.DataFrame) -> dict[str, Any]:
    with_strength = geometry.with_columns(
        ((pl.col("orf_strength_z") + pl.col("crispr_strength_z")) / 2.0).alias("mean_strength")
    )
    binned = assign_strength_bins(with_strength, "mean_strength", 10, "strength_decile")
    overall = {
        row["regime_label"]: row["count"] / binned.height
        for row in binned.group_by("regime_label").len(name="count").to_dicts()
    }
    max_shift = 0.0
    for row in binned.group_by(["strength_decile", "regime_label"]).len(name="count").to_dicts():
        decile_total = binned.filter(pl.col("strength_decile") == row["strength_decile"]).height
        frac = row["count"] / decile_total if decile_total else 0.0
        max_shift = max(max_shift, abs(frac - overall.get(row["regime_label"], 0.0)))
    return {
        "confound_name": "strength_decile_prevalence",
        "test_name": "max_shift_from_overall",
        "statistic_name": "max_absolute_fraction_shift",
        "statistic_value": max_shift,
        "p_value": None,
        "supports_positive_story": max_shift < 0.35,
        "notes": "Regime prevalence summarized across mean-strength deciles.",
    }


def _relabel_from_arrays(
    row: dict[str, Any],
    orf_values: np.ndarray,
    crispr_values: np.ndarray,
    config: ConfoundConfig,
) -> str | None:
    if orf_values.shape[0] < 2 or crispr_values.shape[0] < 2:
        return None
    orf_centroid = coordinate_median(orf_values)
    crispr_centroid = coordinate_median(crispr_values)
    orf_strength_z = (centroid_l2(orf_centroid) - row["orf_null_strength_median"]) / max(
        row["orf_null_strength_mad"], config.null_std_floor
    )
    crispr_strength_z = (centroid_l2(crispr_centroid) - row["crispr_null_strength_median"]) / max(
        row["crispr_null_strength_mad"], config.null_std_floor
    )
    orf_rep_z = (pairwise_cosine_median(orf_values) - row["orf_null_rep_median"]) / max(
        row["orf_null_rep_mad"], config.null_std_floor
    )
    crispr_rep_z = (pairwise_cosine_median(crispr_values) - row["crispr_null_rep_median"]) / max(
        row["crispr_null_rep_mad"], config.null_std_floor
    )
    raw = _cosine(orf_centroid.tolist(), crispr_centroid.tolist())
    sign_z = (raw - row["sign_null_mean"]) / max(row["sign_null_std"], config.null_std_floor)
    prebootstrap = assign_regime(
        orf_active=orf_strength_z >= config.strength_threshold and orf_rep_z >= config.rep_threshold,
        crispr_active=crispr_strength_z >= config.strength_threshold and crispr_rep_z >= config.rep_threshold,
        sign_z=sign_z,
        delta_activity=orf_strength_z - crispr_strength_z,
        sign_z_threshold=config.sign_z_threshold,
        asymmetry_threshold=config.asymmetry_threshold,
    )
    if config.n_bootstrap <= 0:
        return prebootstrap

    counts: dict[str, int] = {}
    rng = _seeded_rng(config.random_seed, str(row["gene_id"]), "confound")
    for _ in range(config.n_bootstrap):
        boot_orf = orf_values[rng.choice(orf_values.shape[0], size=orf_values.shape[0], replace=True)]
        boot_crispr = crispr_values[rng.choice(crispr_values.shape[0], size=crispr_values.shape[0], replace=True)]
        boot_orf_centroid = coordinate_median(boot_orf)
        boot_crispr_centroid = coordinate_median(boot_crispr)
        boot_orf_strength_z = (centroid_l2(boot_orf_centroid) - row["orf_null_strength_median"]) / max(
            row["orf_null_strength_mad"], config.null_std_floor
        )
        boot_crispr_strength_z = (centroid_l2(boot_crispr_centroid) - row["crispr_null_strength_median"]) / max(
            row["crispr_null_strength_mad"], config.null_std_floor
        )
        boot_orf_rep_z = (pairwise_cosine_median(boot_orf) - row["orf_null_rep_median"]) / max(
            row["orf_null_rep_mad"], config.null_std_floor
        )
        boot_crispr_rep_z = (pairwise_cosine_median(boot_crispr) - row["crispr_null_rep_median"]) / max(
            row["crispr_null_rep_mad"], config.null_std_floor
        )
        boot_raw = _cosine(boot_orf_centroid.tolist(), boot_crispr_centroid.tolist())
        boot_sign_z = (boot_raw - row["sign_null_mean"]) / max(row["sign_null_std"], config.null_std_floor)
        label = assign_regime(
            orf_active=boot_orf_strength_z >= config.strength_threshold and boot_orf_rep_z >= config.rep_threshold,
            crispr_active=boot_crispr_strength_z >= config.strength_threshold and boot_crispr_rep_z >= config.rep_threshold,
            sign_z=boot_sign_z,
            delta_activity=boot_orf_strength_z - boot_crispr_strength_z,
            sign_z_threshold=config.sign_z_threshold,
            asymmetry_threshold=config.asymmetry_threshold,
        )
        counts[label] = counts.get(label, 0) + 1

    consensus_label = max(counts, key=counts.get)
    consensus_probability = counts[consensus_label] / config.n_bootstrap
    return consensus_label if consensus_probability >= config.consensus_threshold else "ambiguous"


def _relabel_from_groups(
    row: dict[str, Any],
    orf_group: pl.DataFrame,
    crispr_group: pl.DataFrame,
    feature_columns: list[str],
    config: ConfoundConfig,
) -> str | None:
    if orf_group.height < 2 or crispr_group.height < 2:
        return None
    return _relabel_from_arrays(
        row,
        orf_group.select(feature_columns).to_numpy(),
        crispr_group.select(feature_columns).to_numpy(),
        config,
    )


def _build_plate_holdout_records(
    merged: pl.DataFrame,
    orf_state: dict[str, Any],
    crispr_state: dict[str, Any],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in merged.to_dicts():
        gene_id = str(row["gene_id"])
        orf_range = orf_state["manifest"].get(gene_id)
        crispr_range = crispr_state["manifest"].get(gene_id)
        if orf_range is None or crispr_range is None:
            continue
        orf_plates = set(orf_state["plate_values"][orf_range[0] : orf_range[1]].tolist())
        crispr_plates = set(crispr_state["plate_values"][crispr_range[0] : crispr_range[1]].tolist())
        if len(orf_plates) < 2 or len(crispr_plates) < 2:
            continue
        record = dict(row)
        record["gene_id"] = gene_id
        record["estimated_cost"] = len(orf_plates.union(crispr_plates))
        records.append(record)
    return records


def _build_artifact_records(
    merged: pl.DataFrame,
    orf_state: dict[str, Any],
    crispr_state: dict[str, Any],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in merged.to_dicts():
        gene_id = str(row["gene_id"])
        orf_range = orf_state["manifest"].get(gene_id)
        crispr_range = crispr_state["manifest"].get(gene_id)
        if orf_range is None or crispr_range is None:
            continue
        orf_n = int(orf_range[1] - orf_range[0])
        crispr_n = int(crispr_range[1] - crispr_range[0])
        if orf_n < 2 or crispr_n < 2:
            continue
        record = dict(row)
        record["gene_id"] = gene_id
        record["estimated_cost"] = orf_n + crispr_n
        records.append(record)
    return records


def _phase8_plate_worker_chunk(
    chunk_records: list[dict[str, Any]],
    *,
    config: ConfoundConfig,
) -> dict[str, int]:
    if _PHASE8_SHARED is None:  # pragma: no cover - defensive worker setup guard
        raise RuntimeError("Phase 08 worker shared state was not initialized.")
    agreements = 0
    evaluated = 0
    for record in chunk_records:
        gene_id = str(record["gene_id"])
        orf_start, orf_stop = _PHASE8_SHARED["orf_manifest"][gene_id]
        crispr_start, crispr_stop = _PHASE8_SHARED["crispr_manifest"][gene_id]
        orf_values = np.asarray(_PHASE8_SHARED["orf_matrix"][orf_start:orf_stop], dtype=float)
        crispr_values = np.asarray(_PHASE8_SHARED["crispr_matrix"][crispr_start:crispr_stop], dtype=float)
        orf_plates = np.asarray(_PHASE8_SHARED["orf_plate_values"][orf_start:orf_stop])
        crispr_plates = np.asarray(_PHASE8_SHARED["crispr_plate_values"][crispr_start:crispr_stop])
        holdout_plates = sorted(set(orf_plates.tolist()) | set(crispr_plates.tolist()))
        for plate in holdout_plates:
            relabel = _relabel_from_arrays(
                record,
                orf_values[orf_plates != plate],
                crispr_values[crispr_plates != plate],
                config,
            )
            if relabel is None:
                continue
            evaluated += 1
            if relabel == record["regime_label"]:
                agreements += 1
    return {"agreements": agreements, "evaluated": evaluated}


def _phase8_artifact_worker_chunk(
    chunk_records: list[dict[str, Any]],
    *,
    config: ConfoundConfig,
) -> dict[str, int]:
    if _PHASE8_SHARED is None:  # pragma: no cover - defensive worker setup guard
        raise RuntimeError("Phase 08 worker shared state was not initialized.")
    agreements = 0
    evaluated = 0
    for record in chunk_records:
        gene_id = str(record["gene_id"])
        orf_start, orf_stop = _PHASE8_SHARED["orf_manifest"][gene_id]
        crispr_start, crispr_stop = _PHASE8_SHARED["crispr_manifest"][gene_id]
        relabel = _relabel_from_arrays(
            record,
            np.asarray(_PHASE8_SHARED["orf_matrix"][orf_start:orf_stop], dtype=float),
            np.asarray(_PHASE8_SHARED["crispr_matrix"][crispr_start:crispr_stop], dtype=float),
            config,
        )
        if relabel is None:
            continue
        evaluated += 1
        if relabel == record["regime_label"]:
            agreements += 1
    return {"agreements": agreements, "evaluated": evaluated}


def _run_parallel_reduction(
    records: list[dict[str, Any]],
    *,
    chunk_size: int,
    worker_count: int,
    cost_key: str,
    progress_path: Path | None,
    stage_name: str,
    manifest_path: Path,
    config: ConfoundConfig,
    numpy_threads_per_worker: int,
    worker_fn: Any,
) -> tuple[int, int]:
    chunks = _chunk_records(records, chunk_size, cost_key=cost_key)
    total_chunks = len(chunks)
    agreements = 0
    evaluated = 0
    _write_progress_state(
        progress_path,
        {
            "phase": "08_run_confounds_and_ablations",
            "stage": stage_name,
            "status": "running",
            "completed_chunks": 0,
            "total_chunks": total_chunks,
            "agreements": agreements,
            "evaluated": evaluated,
        },
    )
    if total_chunks == 0:
        _write_progress_state(
            progress_path,
            {
                "phase": "08_run_confounds_and_ablations",
                "stage": stage_name,
                "status": "completed",
                "completed_chunks": 0,
                "total_chunks": 0,
                "agreements": 0,
                "evaluated": 0,
            },
        )
        return 0, 0

    log_every = max(1, min(32, total_chunks // 10 or 1))
    with ProcessPoolExecutor(
        max_workers=worker_count,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_phase8_worker_init,
        initargs=(str(manifest_path), numpy_threads_per_worker),
    ) as executor:
        futures = [executor.submit(worker_fn, chunk, config=config) for chunk in chunks]
        for completed_chunks, future in enumerate(as_completed(futures), start=1):
            try:
                result = future.result()
            except Exception:
                _write_progress_state(
                    progress_path,
                    {
                        "phase": "08_run_confounds_and_ablations",
                        "stage": stage_name,
                        "status": "failed",
                        "completed_chunks": completed_chunks - 1,
                        "total_chunks": total_chunks,
                        "agreements": agreements,
                        "evaluated": evaluated,
                    },
                )
                raise
            agreements += int(result["agreements"])
            evaluated += int(result["evaluated"])
            if completed_chunks == total_chunks or completed_chunks % log_every == 0:
                _write_progress_state(
                    progress_path,
                    {
                        "phase": "08_run_confounds_and_ablations",
                        "stage": stage_name,
                        "status": "running" if completed_chunks < total_chunks else "completed",
                        "completed_chunks": completed_chunks,
                        "total_chunks": total_chunks,
                        "agreements": agreements,
                        "evaluated": evaluated,
                    },
                )
    return agreements, evaluated


def _plate_holdout_summary(
    geometry: pl.DataFrame,
    gene_stats: pl.DataFrame,
    centered_orf: pl.DataFrame,
    centered_crispr: pl.DataFrame,
    feature_columns: list[str],
    config: ConfoundConfig,
) -> dict[str, Any]:
    merged = geometry.join(
        gene_stats.filter(pl.col("modality") == "orf").select(
            ["gene_id", "null_strength_median", "null_strength_mad", "null_rep_median", "null_rep_mad"]
        ),
        on="gene_id",
        how="left",
    ).rename(
        {
            "null_strength_median": "orf_null_strength_median",
            "null_strength_mad": "orf_null_strength_mad",
            "null_rep_median": "orf_null_rep_median",
            "null_rep_mad": "orf_null_rep_mad",
        }
    ).join(
        gene_stats.filter(pl.col("modality") == "crispr").select(
            ["gene_id", "null_strength_median", "null_strength_mad", "null_rep_median", "null_rep_mad"]
        ),
        on="gene_id",
        how="left",
    ).rename(
        {
            "null_strength_median": "crispr_null_strength_median",
            "null_strength_mad": "crispr_null_strength_mad",
            "null_rep_median": "crispr_null_rep_median",
            "null_rep_mad": "crispr_null_rep_mad",
        }
    )
    orf_groups = _group_gene_frames(centered_orf.filter(~pl.col("is_negative_control")))
    crispr_groups = _group_gene_frames(centered_crispr.filter(~pl.col("is_negative_control")))
    agreements = 0
    evaluated = 0
    for row in merged.to_dicts():
        orf_group = orf_groups.get(str(row["gene_id"]))
        crispr_group = crispr_groups.get(str(row["gene_id"]))
        if orf_group is None or crispr_group is None:
            continue
        if orf_group["Metadata_Plate"].n_unique() < 2 or crispr_group["Metadata_Plate"].n_unique() < 2:
            continue
        holdout_plates = sorted(set(orf_group["Metadata_Plate"].to_list()) | set(crispr_group["Metadata_Plate"].to_list()))
        for plate in holdout_plates:
            relabel = _relabel_from_groups(
                row,
                orf_group.filter(pl.col("Metadata_Plate") != plate),
                crispr_group.filter(pl.col("Metadata_Plate") != plate),
                feature_columns,
                config,
            )
            if relabel is None:
                continue
            evaluated += 1
            if relabel == row["regime_label"]:
                agreements += 1
    agreement = agreements / evaluated if evaluated else 0.0
    return {
        "confound_name": "plate_holdout",
        "test_name": "leave_one_plate_out_final_label_agreement",
        "statistic_name": "agreement_rate",
        "statistic_value": agreement,
        "p_value": None,
        "supports_positive_story": agreement >= 0.7 if evaluated else False,
        "notes": f"Evaluated {evaluated} leave-one-plate-out folds.",
    }


def _cell_count_summary(
    geometry: pl.DataFrame,
    centered_orf: pl.DataFrame,
    centered_crispr: pl.DataFrame,
) -> dict[str, Any]:
    orf_cols = [column for column in find_cell_count_columns(centered_orf.columns) if column in centered_orf.columns]
    crispr_cols = [column for column in find_cell_count_columns(centered_crispr.columns) if column in centered_crispr.columns]
    if not orf_cols and not crispr_cols:
        return {
            "confound_name": "cell_count_availability",
            "test_name": "availability_scan",
            "statistic_name": "cell_count_available",
            "statistic_value": 0.0,
            "p_value": None,
            "supports_positive_story": True,
            "notes": "No cell-count-like metadata columns were available after the locked scan.",
        }

    def _median_by_gene(frame: pl.DataFrame, column: str, alias: str) -> pl.DataFrame:
        if column not in frame.columns:
            return pl.DataFrame(schema={"gene_id": pl.String, alias: pl.Float64})
        return (
            frame.filter(pl.col("Metadata_NCBI_Gene_ID").is_not_null())
            .group_by("Metadata_NCBI_Gene_ID")
            .agg(pl.col(column).median().alias(alias))
            .rename({"Metadata_NCBI_Gene_ID": "gene_id"})
        )

    def _regime_p_value(frame: pl.DataFrame, value_column: str) -> float | None:
        groups: list[list[float]] = []
        for regime in sorted(set(frame["regime_label"].to_list())):
            values = frame.filter(pl.col("regime_label") == regime)[value_column].drop_nulls().to_list()
            if len(values) >= 2:
                groups.append(values)
        return float(kruskal(*groups).pvalue) if len(groups) >= 2 else None

    p_values: list[float] = []
    measured: list[str] = []
    geometry_labels = geometry.select(["gene_id", "regime_label"])
    for column in orf_cols:
        joined = geometry_labels.join(_median_by_gene(centered_orf, column, "metric_value"), on="gene_id", how="inner")
        p_value = _regime_p_value(joined, "metric_value")
        if p_value is not None:
            p_values.append(p_value)
        measured.append(f"orf:{column}")
    for column in crispr_cols:
        joined = geometry_labels.join(_median_by_gene(centered_crispr, column, "metric_value"), on="gene_id", how="inner")
        p_value = _regime_p_value(joined, "metric_value")
        if p_value is not None:
            p_values.append(p_value)
        measured.append(f"crispr:{column}")
    for column in sorted(set(orf_cols).intersection(crispr_cols)):
        orf_by_gene = _median_by_gene(centered_orf, column, "orf_metric")
        crispr_by_gene = _median_by_gene(centered_crispr, column, "crispr_metric")
        joined = (
            geometry_labels.join(orf_by_gene, on="gene_id", how="inner")
            .join(crispr_by_gene, on="gene_id", how="inner")
            .with_columns((pl.col("crispr_metric") - pl.col("orf_metric")).alias("metric_delta"))
        )
        p_value = _regime_p_value(joined, "metric_delta")
        if p_value is not None:
            p_values.append(p_value)
        measured.append(f"delta:{column}")
    best_p = min(p_values) if p_values else None
    return {
        "confound_name": "cell_count_availability",
        "test_name": "kruskal_gene_medians_by_regime" if best_p is not None else "availability_only",
        "statistic_name": "min_p_value_across_gene_level_metrics" if best_p is not None else "cell_count_available",
        "statistic_value": best_p if best_p is not None else 1.0,
        "p_value": best_p,
        "supports_positive_story": best_p is None or best_p >= 0.05,
        "notes": "Cell-count-like metrics examined: " + ", ".join(measured),
    }


def _artifact_sensitivity_summary(
    geometry: pl.DataFrame,
    gene_stats: pl.DataFrame,
    centered_orf: pl.DataFrame,
    centered_crispr: pl.DataFrame,
    feature_columns: list[str],
    config: ConfoundConfig,
) -> dict[str, Any]:
    marked_orf, orf_artifact = mark_artifact_outliers(
        centered_orf,
        feature_columns,
        config=ArtifactFilterConfig(null_std_floor=config.null_std_floor),
    )
    marked_crispr, crispr_artifact = mark_artifact_outliers(
        centered_crispr,
        feature_columns,
        config=ArtifactFilterConfig(null_std_floor=config.null_std_floor),
    )
    merged = geometry.join(
        gene_stats.filter(pl.col("modality") == "orf").select(
            ["gene_id", "null_strength_median", "null_strength_mad", "null_rep_median", "null_rep_mad"]
        ),
        on="gene_id",
        how="left",
    ).rename(
        {
            "null_strength_median": "orf_null_strength_median",
            "null_strength_mad": "orf_null_strength_mad",
            "null_rep_median": "orf_null_rep_median",
            "null_rep_mad": "orf_null_rep_mad",
        }
    ).join(
        gene_stats.filter(pl.col("modality") == "crispr").select(
            ["gene_id", "null_strength_median", "null_strength_mad", "null_rep_median", "null_rep_mad"]
        ),
        on="gene_id",
        how="left",
    ).rename(
        {
            "null_strength_median": "crispr_null_strength_median",
            "null_strength_mad": "crispr_null_strength_mad",
            "null_rep_median": "crispr_null_rep_median",
            "null_rep_mad": "crispr_null_rep_mad",
        }
    )
    filtered_orf_groups = _group_gene_frames(
        marked_orf.filter(~pl.col("is_negative_control") & ~pl.col("artifact_outlier"))
    )
    filtered_crispr_groups = _group_gene_frames(
        marked_crispr.filter(~pl.col("is_negative_control") & ~pl.col("artifact_outlier"))
    )
    agreements = 0
    evaluated = 0
    for row in merged.to_dicts():
        orf_group = filtered_orf_groups.get(str(row["gene_id"]))
        crispr_group = filtered_crispr_groups.get(str(row["gene_id"]))
        if orf_group is None or crispr_group is None:
            continue
        relabel = _relabel_from_groups(row, orf_group, crispr_group, feature_columns, config)
        if relabel is None:
            continue
        evaluated += 1
        if relabel == row["regime_label"]:
            agreements += 1
    agreement = agreements / evaluated if evaluated else 0.0
    removed = int(marked_orf["artifact_outlier"].sum() + marked_crispr["artifact_outlier"].sum())
    norm_removed = int(orf_artifact["artifact_norm_outlier_count"]) + int(crispr_artifact["artifact_norm_outlier_count"])
    plate_removed = int(orf_artifact["artifact_plate_outlier_count"]) + int(
        crispr_artifact["artifact_plate_outlier_count"]
    )
    return {
        "confound_name": "artifact_sensitivity",
        "test_name": "outlier_removed_final_label_agreement",
        "statistic_name": "agreement_rate",
        "statistic_value": agreement,
        "p_value": None,
        "supports_positive_story": agreement >= 0.7 if evaluated else False,
        "notes": (
            f"Removed {removed} wells by transparent outlier rule "
            f"({norm_removed} norm outliers, {plate_removed} plate-flagged wells)."
        ),
    }


def _plate_holdout_summary_parallel(
    geometry: pl.DataFrame,
    gene_stats: pl.DataFrame,
    centered_orf: pl.DataFrame,
    centered_crispr: pl.DataFrame,
    feature_columns: list[str],
    config: ConfoundConfig,
    *,
    compute_config: ComputeConfig,
    run_dir: Path,
    progress_path: Path | None,
) -> dict[str, Any]:
    merged = geometry.join(
        gene_stats.filter(pl.col("modality") == "orf").select(
            ["gene_id", "null_strength_median", "null_strength_mad", "null_rep_median", "null_rep_mad"]
        ),
        on="gene_id",
        how="left",
    ).rename(
        {
            "null_strength_median": "orf_null_strength_median",
            "null_strength_mad": "orf_null_strength_mad",
            "null_rep_median": "orf_null_rep_median",
            "null_rep_mad": "orf_null_rep_mad",
        }
    ).join(
        gene_stats.filter(pl.col("modality") == "crispr").select(
            ["gene_id", "null_strength_median", "null_strength_mad", "null_rep_median", "null_rep_mad"]
        ),
        on="gene_id",
        how="left",
    ).rename(
        {
            "null_strength_median": "crispr_null_strength_median",
            "null_strength_mad": "crispr_null_strength_mad",
            "null_rep_median": "crispr_null_rep_median",
            "null_rep_mad": "crispr_null_rep_mad",
        }
    )
    orf_state = _build_gene_array_state(centered_orf, feature_columns)
    crispr_state = _build_gene_array_state(centered_crispr, feature_columns)
    records = _build_plate_holdout_records(merged, orf_state, crispr_state)
    manifest_path = _save_phase8_shared_state(
        work_dir=compute_config.mmap_root / "phase08" / "plate_holdout",
        arrays={
            "orf_matrix": orf_state["matrix"],
            "orf_plate_values": orf_state["plate_values"],
            "crispr_matrix": crispr_state["matrix"],
            "crispr_plate_values": crispr_state["plate_values"],
        },
        metadata={
            "orf_manifest": orf_state["manifest"],
            "crispr_manifest": crispr_state["manifest"],
        },
    )
    agreements, evaluated = _run_parallel_reduction(
        records,
        chunk_size=compute_config.phase8_gene_chunk_size,
        worker_count=compute_config.phase8_workers,
        cost_key="estimated_cost",
        progress_path=progress_path,
        stage_name="plate_holdout",
        manifest_path=manifest_path,
        config=config,
        numpy_threads_per_worker=compute_config.numpy_threads_per_worker,
        worker_fn=_phase8_plate_worker_chunk,
    )
    agreement = agreements / evaluated if evaluated else 0.0
    return {
        "confound_name": "plate_holdout",
        "test_name": "leave_one_plate_out_final_label_agreement",
        "statistic_name": "agreement_rate",
        "statistic_value": agreement,
        "p_value": None,
        "supports_positive_story": agreement >= 0.7 if evaluated else False,
        "notes": f"Evaluated {evaluated} leave-one-plate-out folds.",
    }


def _artifact_sensitivity_summary_parallel(
    geometry: pl.DataFrame,
    gene_stats: pl.DataFrame,
    centered_orf: pl.DataFrame,
    centered_crispr: pl.DataFrame,
    feature_columns: list[str],
    config: ConfoundConfig,
    *,
    compute_config: ComputeConfig,
    run_dir: Path,
    progress_path: Path | None,
) -> dict[str, Any]:
    _write_progress_state(
        progress_path,
        {
            "phase": "08_run_confounds_and_ablations",
            "stage": "artifact_marking",
            "status": "running",
        },
    )
    marked_orf, orf_artifact = mark_artifact_outliers(
        centered_orf,
        feature_columns,
        config=ArtifactFilterConfig(null_std_floor=config.null_std_floor),
    )
    marked_crispr, crispr_artifact = mark_artifact_outliers(
        centered_crispr,
        feature_columns,
        config=ArtifactFilterConfig(null_std_floor=config.null_std_floor),
    )
    _write_progress_state(
        progress_path,
        {
            "phase": "08_run_confounds_and_ablations",
            "stage": "artifact_marking",
            "status": "completed",
            "orf_removed": int(marked_orf["artifact_outlier"].sum()),
            "crispr_removed": int(marked_crispr["artifact_outlier"].sum()),
        },
    )
    merged = geometry.join(
        gene_stats.filter(pl.col("modality") == "orf").select(
            ["gene_id", "null_strength_median", "null_strength_mad", "null_rep_median", "null_rep_mad"]
        ),
        on="gene_id",
        how="left",
    ).rename(
        {
            "null_strength_median": "orf_null_strength_median",
            "null_strength_mad": "orf_null_strength_mad",
            "null_rep_median": "orf_null_rep_median",
            "null_rep_mad": "orf_null_rep_mad",
        }
    ).join(
        gene_stats.filter(pl.col("modality") == "crispr").select(
            ["gene_id", "null_strength_median", "null_strength_mad", "null_rep_median", "null_rep_mad"]
        ),
        on="gene_id",
        how="left",
    ).rename(
        {
            "null_strength_median": "crispr_null_strength_median",
            "null_strength_mad": "crispr_null_strength_mad",
            "null_rep_median": "crispr_null_rep_median",
            "null_rep_mad": "crispr_null_rep_mad",
        }
    )
    filtered_orf_state = _build_gene_array_state(marked_orf, feature_columns, exclude_artifact_outliers=True)
    filtered_crispr_state = _build_gene_array_state(marked_crispr, feature_columns, exclude_artifact_outliers=True)
    records = _build_artifact_records(merged, filtered_orf_state, filtered_crispr_state)
    manifest_path = _save_phase8_shared_state(
        work_dir=compute_config.mmap_root / "phase08" / "artifact_sensitivity",
        arrays={
            "orf_matrix": filtered_orf_state["matrix"],
            "crispr_matrix": filtered_crispr_state["matrix"],
        },
        metadata={
            "orf_manifest": filtered_orf_state["manifest"],
            "crispr_manifest": filtered_crispr_state["manifest"],
        },
    )
    agreements, evaluated = _run_parallel_reduction(
        records,
        chunk_size=compute_config.phase8_gene_chunk_size,
        worker_count=compute_config.phase8_workers,
        cost_key="estimated_cost",
        progress_path=progress_path,
        stage_name="artifact_sensitivity",
        manifest_path=manifest_path,
        config=config,
        numpy_threads_per_worker=compute_config.numpy_threads_per_worker,
        worker_fn=_phase8_artifact_worker_chunk,
    )
    agreement = agreements / evaluated if evaluated else 0.0
    removed = int(marked_orf["artifact_outlier"].sum() + marked_crispr["artifact_outlier"].sum())
    norm_removed = int(orf_artifact["artifact_norm_outlier_count"]) + int(crispr_artifact["artifact_norm_outlier_count"])
    plate_removed = int(orf_artifact["artifact_plate_outlier_count"]) + int(
        crispr_artifact["artifact_plate_outlier_count"]
    )
    return {
        "confound_name": "artifact_sensitivity",
        "test_name": "outlier_removed_final_label_agreement",
        "statistic_name": "agreement_rate",
        "statistic_value": agreement,
        "p_value": None,
        "supports_positive_story": agreement >= 0.7 if evaluated else False,
        "notes": (
            f"Removed {removed} wells by transparent outlier rule "
            f"({norm_removed} norm outliers, {plate_removed} plate-flagged wells)."
        ),
    }


def compare_geometry_tables(base: pl.DataFrame, other: pl.DataFrame, comparison_name: str) -> pl.DataFrame:
    confident_base = base.filter(pl.col("regime_label") != "ambiguous").select(["gene_id", "regime_label"])
    confident_other = other.filter(pl.col("regime_label") != "ambiguous").select(["gene_id", "regime_label"])
    joined = confident_base.join(confident_other, on="gene_id", how="inner", suffix="_other")
    if joined.height == 0:
        return pl.DataFrame(
            {
                "comparison_name": [comparison_name],
                "has_real_comparison": [False],
                "n_genes_both_confident": [0],
                "n_genes_both_primary_confident": [0],
                "overall_agreement": [0.0],
                "overall_agreement_primary_only": [0.0],
                "agreement_collapsed_asymmetric": [0.0],
                "aligned_agreement": [0.0],
                "inverse_agreement": [0.0],
                "orthogonal_agreement": [0.0],
                "asymmetric_orf_agreement": [0.0],
                "asymmetric_crispr_agreement": [0.0],
                "notes": ["No overlapping confident genes."],
            }
        )
    primary_joined = joined.filter(
        (pl.col("regime_label") != "inactive") & (pl.col("regime_label_other") != "inactive")
    )
    collapse = lambda label: "asymmetric" if label in {"asymmetric_ORF", "asymmetric_CRISPR"} else label
    collapsed = joined.with_columns(
        pl.col("regime_label").map_elements(collapse, return_dtype=pl.String).alias("collapsed_label"),
        pl.col("regime_label_other").map_elements(collapse, return_dtype=pl.String).alias("collapsed_label_other"),
    )
    def agreement_for(label: str) -> float:
        subset = joined.filter(pl.col("regime_label") == label)
        return float((subset["regime_label"] == subset["regime_label_other"]).mean()) if subset.height else 0.0
    return pl.DataFrame(
        {
            "comparison_name": [comparison_name],
            "has_real_comparison": [True],
            "n_genes_both_confident": [joined.height],
            "n_genes_both_primary_confident": [primary_joined.height],
            "overall_agreement": [float((joined["regime_label"] == joined["regime_label_other"]).mean())],
            "overall_agreement_primary_only": [
                float((primary_joined["regime_label"] == primary_joined["regime_label_other"]).mean())
                if primary_joined.height
                else 0.0
            ],
            "agreement_collapsed_asymmetric": [
                float((collapsed["collapsed_label"] == collapsed["collapsed_label_other"]).mean())
            ],
            "aligned_agreement": [agreement_for("aligned")],
            "inverse_agreement": [agreement_for("inverse")],
            "orthogonal_agreement": [agreement_for("orthogonal")],
            "asymmetric_orf_agreement": [agreement_for("asymmetric_ORF")],
            "asymmetric_crispr_agreement": [agreement_for("asymmetric_CRISPR")],
            "notes": ["Agreement computed on genes confident in both runs."],
        }
    )


def compare_regime_prevalence(base: pl.DataFrame, other: pl.DataFrame, comparison_name: str) -> pl.DataFrame:
    base_named = base.rename(
        {
            "n_genes": "base_n_genes",
            "fraction_of_analyzed": "base_fraction_of_analyzed",
            "n_regime_members": "base_n_regime_members",
        }
    )
    other_named = other.rename(
        {
            "n_genes": "other_n_genes",
            "fraction_of_analyzed": "other_fraction_of_analyzed",
            "n_regime_members": "other_n_regime_members",
        }
    )
    joined = base_named.join(other_named, on="regime_label", how="full", coalesce=True).with_columns(
        pl.lit(comparison_name).alias("comparison_name"),
        pl.col("base_n_genes").fill_null(0),
        pl.col("base_fraction_of_analyzed").fill_null(0.0),
        pl.col("base_n_regime_members").fill_null(0),
        pl.col("other_n_genes").fill_null(0),
        pl.col("other_fraction_of_analyzed").fill_null(0.0),
        pl.col("other_n_regime_members").fill_null(0),
    ).with_columns(
        (pl.col("other_fraction_of_analyzed") - pl.col("base_fraction_of_analyzed")).alias("fraction_delta")
    )
    return joined.select(
        "comparison_name",
        "regime_label",
        "base_n_genes",
        "base_fraction_of_analyzed",
        "base_n_regime_members",
        "other_n_genes",
        "other_fraction_of_analyzed",
        "other_n_regime_members",
        "fraction_delta",
    ).sort("regime_label")


def run_confounds(
    geometry: pl.DataFrame,
    gene_stats: pl.DataFrame,
    centered_orf: pl.DataFrame,
    centered_crispr: pl.DataFrame,
    feature_columns: list[str],
    *,
    config: ConfoundConfig,
    compute_config: ComputeConfig | None = None,
    run_dir: Path | None = None,
    progress_path: Path | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    if geometry.is_empty() or gene_stats.is_empty():
        return (
            pl.DataFrame(
                schema={
                    "pipeline": pl.String,
                    "confound_name": pl.String,
                    "test_name": pl.String,
                    "statistic_name": pl.String,
                    "statistic_value": pl.Float64,
                    "p_value": pl.Float64,
                    "supports_positive_story": pl.Boolean,
                    "notes": pl.String,
                }
            ),
            pl.DataFrame(
                schema={
                    "pipeline": pl.String,
                    "regime_label": pl.String,
                    "n_genes": pl.Int64,
                    "fraction_of_analyzed": pl.Float64,
                    "n_regime_members": pl.Int64,
                }
            ),
        )
    use_parallel = (
        compute_config is not None
        and compute_config.parallel_backend == "process_pool"
        and compute_config.phase8_workers > 1
        and run_dir is not None
    )
    orf_stats = gene_stats.filter(pl.col("modality") == "orf")
    crispr_stats = gene_stats.filter(pl.col("modality") == "crispr")
    _write_progress_state(
        progress_path,
        {
            "phase": "08_run_confounds_and_ablations",
            "stage": "magnitude_residualization",
            "status": "running",
        },
    )
    _, magnitude_row = _fit_magnitude_model(geometry, orf_stats, crispr_stats, config)
    _write_progress_state(
        progress_path,
        {
            "phase": "08_run_confounds_and_ablations",
            "stage": "magnitude_residualization",
            "status": "completed",
        },
    )
    if use_parallel:
        plate_row = _plate_holdout_summary_parallel(
            geometry,
            gene_stats,
            centered_orf,
            centered_crispr,
            feature_columns,
            config,
            compute_config=compute_config,
            run_dir=run_dir,
            progress_path=progress_path,
        )
    else:
        _write_progress_state(
            progress_path,
            {
                "phase": "08_run_confounds_and_ablations",
                "stage": "plate_holdout",
                "status": "running",
            },
        )
        plate_row = _plate_holdout_summary(
            geometry, gene_stats, centered_orf, centered_crispr, feature_columns, config
        )
        _write_progress_state(
            progress_path,
            {
                "phase": "08_run_confounds_and_ablations",
                "stage": "plate_holdout",
                "status": "completed",
                "notes": plate_row["notes"],
            },
        )
    _write_progress_state(
        progress_path,
        {
            "phase": "08_run_confounds_and_ablations",
            "stage": "cell_count_availability",
            "status": "running",
        },
    )
    cell_count_row = _cell_count_summary(geometry, centered_orf, centered_crispr)
    _write_progress_state(
        progress_path,
        {
            "phase": "08_run_confounds_and_ablations",
            "stage": "cell_count_availability",
            "status": "completed",
        },
    )
    _write_progress_state(
        progress_path,
        {
            "phase": "08_run_confounds_and_ablations",
            "stage": "strength_decile_prevalence",
            "status": "running",
        },
    )
    strength_row = _strength_decile_prevalence_summary(geometry)
    _write_progress_state(
        progress_path,
        {
            "phase": "08_run_confounds_and_ablations",
            "stage": "strength_decile_prevalence",
            "status": "completed",
        },
    )
    if use_parallel:
        artifact_row = _artifact_sensitivity_summary_parallel(
            geometry,
            gene_stats,
            centered_orf,
            centered_crispr,
            feature_columns,
            config,
            compute_config=compute_config,
            run_dir=run_dir,
            progress_path=progress_path,
        )
    else:
        artifact_row = _artifact_sensitivity_summary(
            geometry, gene_stats, centered_orf, centered_crispr, feature_columns, config
        )
    confounds = pl.DataFrame(
        [
            {**magnitude_row},
            {"pipeline": str(geometry["pipeline"][0]), **plate_row},
            {"pipeline": str(geometry["pipeline"][0]), **cell_count_row},
            {"pipeline": str(geometry["pipeline"][0]), **strength_row},
            {"pipeline": str(geometry["pipeline"][0]), **artifact_row},
        ]
    ).select(
        "pipeline",
        "confound_name",
        "test_name",
        "statistic_name",
        "statistic_value",
        "p_value",
        "supports_positive_story",
        "notes",
    )
    regime_prevalence = _regime_prevalence(geometry)
    _write_progress_state(
        progress_path,
        {
            "phase": "08_run_confounds_and_ablations",
            "stage": "completed",
            "status": "completed",
            "confound_rows": confounds.height,
            "regime_prevalence_rows": regime_prevalence.height,
        },
    )
    return confounds, regime_prevalence


def write_confound_outputs(
    confounds: pl.DataFrame,
    regime_prevalence: pl.DataFrame,
    confound_path: Path,
    prevalence_path: Path,
) -> None:
    write_csv_atomic(confounds, confound_path)
    write_csv_atomic(regime_prevalence, prevalence_path)
