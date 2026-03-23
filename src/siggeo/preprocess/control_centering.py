from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl


@dataclass(frozen=True)
class CenteringConfig:
    min_negcon_per_plate: int = 8


def _median_vector(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        raise ValueError("Cannot compute centering vector from an empty matrix.")
    return np.median(matrix, axis=0)


def center_by_controls(
    frame: pl.DataFrame,
    feature_columns: list[str],
    config: CenteringConfig,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    if frame.is_empty():
        return frame, {
            "wells_kept": 0,
            "wells_dropped": 0,
            "plates_direct": 0,
            "plates_batch_fallback": 0,
            "plates_global_fallback": 0,
            "plate_qc": [],
        }

    if "Metadata_Batch" not in frame.columns:
        raise KeyError("Expected Metadata_Batch column before centering.")
    if "is_negative_control" not in frame.columns:
        raise KeyError("Expected is_negative_control column before centering.")

    global_controls = frame.filter(pl.col("is_negative_control"))
    global_vector = _median_vector(global_controls.select(feature_columns).to_numpy())

    centered_parts: list[pl.DataFrame] = []
    plate_qc: list[dict[str, Any]] = []
    direct = 0
    batch_fallback = 0
    global_fallback = 0

    for plate in frame["Metadata_Plate"].unique().sort().to_list():
        plate_df = frame.filter(pl.col("Metadata_Plate") == plate)
        batch = plate_df["Metadata_Batch"][0]
        plate_controls = plate_df.filter(pl.col("is_negative_control"))

        if plate_controls.height >= config.min_negcon_per_plate:
            center_vector = _median_vector(plate_controls.select(feature_columns).to_numpy())
            source = "plate"
            direct += 1
        else:
            batch_controls = global_controls.filter(pl.col("Metadata_Batch") == batch)
            if batch_controls.height >= config.min_negcon_per_plate:
                center_vector = _median_vector(batch_controls.select(feature_columns).to_numpy())
                source = "batch"
                batch_fallback += 1
            else:
                center_vector = global_vector
                source = "global"
                global_fallback += 1

        centered_matrix = plate_df.select(feature_columns).to_numpy() - center_vector
        centered_plate = plate_df.with_columns(
            [pl.Series(name=column, values=centered_matrix[:, idx]) for idx, column in enumerate(feature_columns)]
        ).with_columns(pl.lit(source).alias("centering_source"))
        centered_parts.append(centered_plate)

        centered_controls = centered_plate.filter(pl.col("is_negative_control"))
        control_norm = None
        if centered_controls.height > 0:
            values = centered_controls.select(feature_columns).to_numpy()
            control_norm = float(np.median(np.linalg.norm(values, axis=1) / np.sqrt(values.shape[1])))
        plate_qc.append(
            {
                "Metadata_Plate": plate,
                "Metadata_Batch": batch,
                "centering_source": source,
                "n_negative_controls": int(plate_controls.height),
                "post_centering_control_norm": control_norm,
            }
        )

    centered = pl.concat(centered_parts).sort(["Metadata_Plate", "Metadata_Well"])
    qc = {
        "wells_kept": centered.height,
        "wells_dropped": 0,
        "plates_direct": direct,
        "plates_batch_fallback": batch_fallback,
        "plates_global_fallback": global_fallback,
        "plate_qc": plate_qc,
    }
    return centered, qc
