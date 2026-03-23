from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import polars as pl

from siggeo.io.atomic import write_json_atomic, write_parquet_atomic
from siggeo.preprocess.artifact_filter import ArtifactFilterConfig, remove_artifact_outliers
from siggeo.preprocess.control_centering import CenteringConfig, center_by_controls
from siggeo.preprocess.feature_space import find_cell_count_columns, schema_columns


def _load_join_metadata(db_path: Path, modality: str) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    with duckdb.connect(str(db_path), read_only=True) as connection:
        well = pl.from_arrow(
            connection.execute("SELECT Metadata_Plate, Metadata_Well, Metadata_JCP2022 FROM well").arrow()
        )
        plate = pl.from_arrow(
            connection.execute(
                "SELECT Metadata_Plate, Metadata_Source, Metadata_Batch, Metadata_PlateType FROM plate"
            ).arrow()
        )
        modality_df = pl.from_arrow(
            connection.execute(
                f"SELECT Metadata_JCP2022, Metadata_Symbol, CAST(Metadata_NCBI_Gene_ID AS VARCHAR) AS Metadata_NCBI_Gene_ID "
                f"FROM {modality}"
            ).arrow()
        )
        controls = pl.from_arrow(
            connection.execute(
                "SELECT Metadata_JCP2022, Metadata_pert_type, Metadata_Name FROM perturbation_control"
            ).arrow()
        )
    return well, plate, modality_df, controls


def _select_smoke_plates(profile_path: Path, plate_limit: int) -> list[str]:
    return (
        pl.scan_parquet(profile_path)
        .select("Metadata_Plate")
        .unique()
        .sort("Metadata_Plate")
        .limit(plate_limit)
        .collect()["Metadata_Plate"]
        .to_list()
    )


def _control_norm(frame: pl.DataFrame, feature_columns: list[str]) -> float | None:
    controls = frame.filter(pl.col("is_negative_control"))
    if controls.height == 0:
        return None
    values = controls.select(feature_columns).to_numpy()
    if values.size == 0:
        return None
    return float(np.median(np.linalg.norm(values, axis=1) / np.sqrt(values.shape[1])))


def _assert_unique_keys(frame: pl.DataFrame, keys: list[str], label: str) -> None:
    duplicates = frame.group_by(keys).len().filter(pl.col("len") > 1)
    if duplicates.height:
        raise ValueError(f"Join metadata '{label}' is not unique on {keys}: {duplicates.height} duplicate key rows.")


def _join_audit_rows(
    joined: pl.DataFrame,
    *,
    control_ids: set[str],
) -> dict[str, Any]:
    with_flags = joined.with_columns(
        pl.col("Metadata_JCP2022").is_in(control_ids).fill_null(False).alias("is_known_control")
    )
    missing_gene_for_treated = with_flags.filter(~pl.col("is_known_control") & pl.col("Metadata_NCBI_Gene_ID").is_null()).height
    missing_control_annotation = with_flags.filter(pl.col("is_known_control") & pl.col("Metadata_pert_type").is_null()).height
    return {
        "n_profile_rows": joined.height,
        "missing_well_mapping_count": int(joined.filter(pl.col("Metadata_JCP2022").is_null()).height),
        "missing_batch_count": int(joined.filter(pl.col("Metadata_Batch").is_null()).height),
        "missing_source_count": int(joined.filter(pl.col("Metadata_Source").is_null()).height),
        "missing_plate_type_count": int(joined.filter(pl.col("Metadata_PlateType").is_null()).height),
        "missing_gene_id_for_treated_count": int(missing_gene_for_treated),
        "missing_control_annotation_count": int(missing_control_annotation),
    }


def _validate_joined_matrix(profile_df: pl.DataFrame, joined: pl.DataFrame, *, control_ids: set[str]) -> dict[str, Any]:
    if joined.height != profile_df.height:
        raise ValueError(
            "Metadata joins changed the row count. "
            f"Expected {profile_df.height} rows, found {joined.height}. This suggests duplicate join expansion."
        )

    audit = _join_audit_rows(joined, control_ids=control_ids)
    blocking_issues = {
        key: value
        for key, value in audit.items()
        if key
        in {
            "missing_well_mapping_count",
            "missing_batch_count",
            "missing_source_count",
            "missing_gene_id_for_treated_count",
            "missing_control_annotation_count",
        }
        and int(value) > 0
    }
    if blocking_issues:
        issue_text = ", ".join(f"{key}={value}" for key, value in sorted(blocking_issues.items()))
        raise ValueError(f"Join audit failed for centered matrix construction: {issue_text}")
    return audit


def _pass_through_without_centering(
    frame: pl.DataFrame,
    feature_columns: list[str],
) -> tuple[pl.DataFrame, dict[str, Any]]:
    centered = frame.with_columns(pl.lit("none").alias("centering_source")).sort(["Metadata_Plate", "Metadata_Well"])
    plate_qc = [
        {
            "Metadata_Plate": plate,
            "Metadata_Batch": subset["Metadata_Batch"][0],
            "centering_source": "none",
            "n_negative_controls": int(subset.filter(pl.col("is_negative_control")).height),
            "post_centering_control_norm": _control_norm(subset, feature_columns),
        }
        for plate in centered["Metadata_Plate"].unique().sort().to_list()
        for subset in [centered.filter(pl.col("Metadata_Plate") == plate)]
    ]
    qc = {
        "wells_kept": centered.height,
        "wells_dropped": 0,
        "plates_direct": 0,
        "plates_batch_fallback": 0,
        "plates_global_fallback": 0,
        "centering_disabled": True,
        "plate_qc": plate_qc,
    }
    return centered, qc


def build_centered_matrix(
    profile_path: Path,
    db_path: Path,
    modality: str,
    feature_list: list[str],
    preprocess_config: dict[str, Any],
    *,
    mode: str,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    columns = schema_columns(profile_path)
    cell_count_columns = [column for column in find_cell_count_columns(columns) if column in columns]
    selected_columns = ["Metadata_Plate", "Metadata_Well", *cell_count_columns, *feature_list]

    scan = pl.scan_parquet(profile_path).select(selected_columns)
    if mode == "smoke" and "smoke_plate_limit_per_modality" in preprocess_config:
        plates = _select_smoke_plates(profile_path, int(preprocess_config["smoke_plate_limit_per_modality"]))
        scan = scan.filter(pl.col("Metadata_Plate").is_in(plates))

    profile_df = scan.collect()
    well, plate, modality_df, controls = _load_join_metadata(db_path, modality)
    _assert_unique_keys(well, ["Metadata_Plate", "Metadata_Well"], "well")
    _assert_unique_keys(plate, ["Metadata_Plate"], "plate")
    _assert_unique_keys(modality_df, ["Metadata_JCP2022"], modality)
    _assert_unique_keys(controls, ["Metadata_JCP2022"], "perturbation_control")
    control_ids = set(controls["Metadata_JCP2022"].drop_nulls().cast(pl.String).to_list())

    joined = (
        profile_df.join(well, on=["Metadata_Plate", "Metadata_Well"], how="left")
        .join(modality_df, on="Metadata_JCP2022", how="left")
        .join(controls, on="Metadata_JCP2022", how="left")
        .join(plate, on="Metadata_Plate", how="left")
        .with_columns(
            (pl.col("Metadata_pert_type") == "negcon").fill_null(False).alias("is_negative_control"),
            pl.lit(modality).alias("modality"),
        )
    )
    join_audit = _validate_joined_matrix(profile_df, joined, control_ids=control_ids)

    if bool(preprocess_config.get("final_plate_negcon_centering", True)):
        centered_df, qc = center_by_controls(
            joined,
            feature_columns=feature_list,
            config=CenteringConfig(min_negcon_per_plate=int(preprocess_config.get("min_negcon_per_plate", 8))),
        )
        qc["centering_disabled"] = False
    else:
        centered_df, qc = _pass_through_without_centering(joined, feature_list)

    if bool(preprocess_config.get("artifact_sensitivity_filter", False)):
        centered_df, artifact_qc = remove_artifact_outliers(
            centered_df,
            feature_list,
            config=ArtifactFilterConfig(),
        )
        qc.update(artifact_qc)
        qc["wells_kept"] = centered_df.height
        qc["wells_dropped"] = int(artifact_qc["artifact_removed_count"])
    else:
        qc.update(
            {
                "artifact_filter_enabled": False,
                "artifact_norm_outlier_count": 0,
                "artifact_plate_outlier_count": 0,
                "artifact_removed_count": 0,
                "artifact_flagged_plates": [],
                "artifact_flagged_plate_count": 0,
            }
        )
    qc.update(
        {
            "cell_count_columns": cell_count_columns,
            "cell_count_available": bool(cell_count_columns),
            "feature_count": len(feature_list),
            "modality": modality,
            "join_audit": join_audit,
        }
    )
    return centered_df, qc


def write_centered_outputs(
    centered_df: pl.DataFrame,
    matrix_path: Path,
    qc_path: Path,
    qc_payload: dict[str, Any],
) -> None:
    write_parquet_atomic(centered_df, matrix_path)
    write_json_atomic(qc_path, qc_payload)
