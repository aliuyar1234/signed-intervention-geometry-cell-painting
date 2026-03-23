from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl


@dataclass(frozen=True)
class ArtifactFilterConfig:
    null_std_floor: float = 1.0e-6


def _row_norms(frame: pl.DataFrame, feature_columns: list[str]) -> np.ndarray:
    values = frame.select(feature_columns).to_numpy()
    if values.size == 0:
        return np.zeros(frame.height, dtype=float)
    return np.linalg.norm(values, axis=1) / np.sqrt(values.shape[1])


def _control_plate_flags(
    frame: pl.DataFrame,
    feature_columns: list[str],
    *,
    null_std_floor: float,
) -> tuple[set[str], float]:
    controls = frame.filter(pl.col("is_negative_control"))
    if controls.height == 0:
        return set(), 0.0

    per_plate: list[tuple[str, float]] = []
    for plate in controls["Metadata_Plate"].unique().sort().to_list():
        subset = controls.filter(pl.col("Metadata_Plate") == plate)
        norms = _row_norms(subset, feature_columns)
        norm = float(np.median(norms)) if norms.size else 0.0
        per_plate.append((plate, norm))

    values = np.array([norm for _, norm in per_plate], dtype=float)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    threshold = median + 5 * max(mad, null_std_floor)
    flagged = {plate for plate, norm in per_plate if norm > threshold}
    return flagged, threshold


def mark_artifact_outliers(
    frame: pl.DataFrame,
    feature_columns: list[str],
    *,
    config: ArtifactFilterConfig | None = None,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    settings = config or ArtifactFilterConfig()
    if frame.is_empty():
        marked = frame.with_columns(
            pl.lit(False).alias("artifact_norm_outlier"),
            pl.lit(False).alias("artifact_plate_outlier"),
            pl.lit(False).alias("artifact_outlier"),
        )
        return marked, {
            "artifact_filter_enabled": True,
            "artifact_norm_threshold": 0.0,
            "artifact_plate_threshold": 0.0,
            "artifact_norm_outlier_count": 0,
            "artifact_plate_outlier_count": 0,
            "artifact_removed_count": 0,
            "artifact_flagged_plates": [],
            "artifact_flagged_plate_count": 0,
        }

    norms = _row_norms(frame, feature_columns)
    median = float(np.median(norms))
    mad = float(np.median(np.abs(norms - median)))
    norm_threshold = median + 5 * max(mad, settings.null_std_floor)
    flagged_plates, plate_threshold = _control_plate_flags(
        frame,
        feature_columns,
        null_std_floor=settings.null_std_floor,
    )

    marked = frame.with_columns(
        pl.Series("artifact_norm_outlier", norms > norm_threshold),
        pl.col("Metadata_Plate").is_in(sorted(flagged_plates)).alias("artifact_plate_outlier"),
    ).with_columns((pl.col("artifact_norm_outlier") | pl.col("artifact_plate_outlier")).alias("artifact_outlier"))

    summary = {
        "artifact_filter_enabled": True,
        "artifact_norm_threshold": norm_threshold,
        "artifact_plate_threshold": plate_threshold,
        "artifact_norm_outlier_count": int(marked["artifact_norm_outlier"].sum()),
        "artifact_plate_outlier_count": int(marked["artifact_plate_outlier"].sum()),
        "artifact_removed_count": int(marked["artifact_outlier"].sum()),
        "artifact_flagged_plates": sorted(flagged_plates),
        "artifact_flagged_plate_count": len(flagged_plates),
    }
    return marked, summary


def remove_artifact_outliers(
    frame: pl.DataFrame,
    feature_columns: list[str],
    *,
    config: ArtifactFilterConfig | None = None,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    marked, summary = mark_artifact_outliers(frame, feature_columns, config=config)
    filtered = marked.filter(~pl.col("artifact_outlier")).sort(["Metadata_Plate", "Metadata_Well"])
    summary = {
        **summary,
        "artifact_retained_count": filtered.height,
    }
    return filtered, summary
