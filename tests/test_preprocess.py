from __future__ import annotations

import shutil
from pathlib import Path

import duckdb
import polars as pl

from siggeo.config import repo_root
from siggeo.preprocess.artifact_filter import remove_artifact_outliers
from siggeo.preprocess.control_centering import CenteringConfig, center_by_controls
from siggeo.preprocess.feature_space import exact_feature_intersection, find_cell_count_columns
from siggeo.preprocess.matrix_builder import build_centered_matrix


def test_exact_feature_intersection_uses_non_metadata_columns_only() -> None:
    base_dir = repo_root() / "results" / "test_preprocess_features"
    shutil.rmtree(base_dir, ignore_errors=True)
    base_dir.mkdir(parents=True, exist_ok=True)
    try:
        orf = pl.DataFrame(
            {
                "Metadata_Plate": ["P1"],
                "Metadata_Well": ["A01"],
                "Metadata_Count_Cells": [10],
                "Image_Count_Cells": [12],
                "Cells_AreaShape_Area": [1.0],
                "Cytoplasm_Intensity": [2.0],
            }
        )
        crispr = pl.DataFrame(
            {
                "Metadata_Plate": ["P2"],
                "Metadata_Well": ["A01"],
                "Image_Count_Cells": [11],
                "Cells_AreaShape_Area": [1.0],
                "Nuclei_Texture": [3.0],
            }
        )
        orf_path = base_dir / "orf.parquet"
        crispr_path = base_dir / "crispr.parquet"
        orf.write_parquet(orf_path)
        crispr.write_parquet(crispr_path)

        assert exact_feature_intersection(orf_path, crispr_path) == ["Cells_AreaShape_Area"]
        assert find_cell_count_columns(orf.columns) == ["Image_Count_Cells", "Metadata_Count_Cells"]
    finally:
        shutil.rmtree(base_dir, ignore_errors=True)


def test_control_centering_uses_plate_then_global_fallback() -> None:
    frame = pl.DataFrame(
        {
            "Metadata_Plate": ["P1", "P1", "P1", "P2", "P2"],
            "Metadata_Well": ["A01", "A02", "A03", "A01", "A02"],
            "Metadata_Batch": ["B1", "B1", "B1", "B2", "B2"],
            "is_negative_control": [True, True, False, True, False],
            "feat1": [1.0, 3.0, 5.0, 2.0, 10.0],
            "feat2": [1.0, 3.0, 7.0, 2.0, 12.0],
        }
    )

    centered, qc = center_by_controls(frame, ["feat1", "feat2"], CenteringConfig(min_negcon_per_plate=2))
    rows = {(row["Metadata_Plate"], row["Metadata_Well"]): row for row in centered.to_dicts()}

    assert rows[("P1", "A03")]["feat1"] == 3.0
    assert rows[("P1", "A03")]["feat2"] == 5.0
    assert rows[("P2", "A02")]["feat1"] == 8.0
    assert rows[("P2", "A02")]["feat2"] == 10.0
    assert qc["plates_direct"] == 1
    assert qc["plates_global_fallback"] == 1


def test_artifact_filter_removes_norm_and_plate_outliers() -> None:
    frame = pl.DataFrame(
        {
            "Metadata_Plate": ["P1", "P1", "P2", "P2", "P3", "P3"],
            "Metadata_Well": ["A01", "A02", "A01", "A02", "A01", "A02"],
            "Metadata_Batch": ["B1", "B1", "B1", "B1", "B1", "B1"],
            "is_negative_control": [True, False, True, False, True, False],
            "feat1": [0.0, 0.1, 0.0, 50.0, 100.0, 0.2],
            "feat2": [0.0, 0.1, 0.0, 50.0, 100.0, 0.2],
        }
    )

    filtered, qc = remove_artifact_outliers(frame, ["feat1", "feat2"])

    assert filtered.height == 3
    assert qc["artifact_norm_outlier_count"] >= 1
    assert qc["artifact_flagged_plate_count"] == 1
    assert qc["artifact_flagged_plates"] == ["P3"]


def _make_test_join_db(db_path: Path, *, duplicate_well: bool = False, missing_batch: bool = False) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(db_path)) as connection:
        connection.execute(
            "CREATE TABLE well (Metadata_Plate VARCHAR, Metadata_Well VARCHAR, Metadata_JCP2022 VARCHAR)"
        )
        connection.execute("INSERT INTO well VALUES ('P1', 'A01', 'JCP1')")
        if duplicate_well:
            connection.execute("INSERT INTO well VALUES ('P1', 'A01', 'JCP1_DUP')")
        connection.execute(
            "CREATE TABLE plate (Metadata_Plate VARCHAR, Metadata_Source VARCHAR, Metadata_Batch VARCHAR, Metadata_PlateType VARCHAR)"
        )
        batch_value = "NULL" if missing_batch else "'B1'"
        connection.execute(
            f"INSERT INTO plate VALUES ('P1', 'SRC1', {batch_value}, 'ORF')"
        )
        connection.execute(
            "CREATE TABLE orf (Metadata_JCP2022 VARCHAR, Metadata_Symbol VARCHAR, Metadata_NCBI_Gene_ID VARCHAR)"
        )
        connection.execute("INSERT INTO orf VALUES ('JCP1', 'GENEA', '101')")
        if duplicate_well:
            connection.execute("INSERT INTO orf VALUES ('JCP1_DUP', 'GENEA', '101')")
        connection.execute(
            "CREATE TABLE perturbation_control (Metadata_JCP2022 VARCHAR, Metadata_pert_type VARCHAR, Metadata_Name VARCHAR)"
        )
        connection.execute("INSERT INTO perturbation_control VALUES ('CTRL', 'negcon', 'NEGATIVE_CONTROL')")


def test_build_centered_matrix_rejects_duplicate_join_keys() -> None:
    base_dir = repo_root() / "results" / "test_preprocess_join_duplicates"
    shutil.rmtree(base_dir, ignore_errors=True)
    base_dir.mkdir(parents=True, exist_ok=True)
    try:
        profile = pl.DataFrame(
            {
                "Metadata_Plate": ["P1"],
                "Metadata_Well": ["A01"],
                "Cells_AreaShape_Area": [1.0],
            }
        )
        profile_path = base_dir / "orf.parquet"
        db_path = base_dir / "metadata.duckdb"
        profile.write_parquet(profile_path)
        _make_test_join_db(db_path, duplicate_well=True)

        try:
            build_centered_matrix(
                profile_path=profile_path,
                db_path=db_path,
                modality="orf",
                feature_list=["Cells_AreaShape_Area"],
                preprocess_config={"final_plate_negcon_centering": False},
                mode="full",
            )
        except ValueError as exc:
            assert "not unique" in str(exc)
        else:
            raise AssertionError("Expected duplicate join keys to fail build_centered_matrix.")
    finally:
        shutil.rmtree(base_dir, ignore_errors=True)


def test_build_centered_matrix_rejects_missing_batch_metadata() -> None:
    base_dir = repo_root() / "results" / "test_preprocess_join_missing_batch"
    shutil.rmtree(base_dir, ignore_errors=True)
    base_dir.mkdir(parents=True, exist_ok=True)
    try:
        profile = pl.DataFrame(
            {
                "Metadata_Plate": ["P1"],
                "Metadata_Well": ["A01"],
                "Cells_AreaShape_Area": [1.0],
            }
        )
        profile_path = base_dir / "orf.parquet"
        db_path = base_dir / "metadata.duckdb"
        profile.write_parquet(profile_path)
        _make_test_join_db(db_path, missing_batch=True)

        try:
            build_centered_matrix(
                profile_path=profile_path,
                db_path=db_path,
                modality="orf",
                feature_list=["Cells_AreaShape_Area"],
                preprocess_config={"final_plate_negcon_centering": False},
                mode="full",
            )
        except ValueError as exc:
            assert "missing_batch_count=1" in str(exc)
        else:
            raise AssertionError("Expected missing batch metadata to fail build_centered_matrix.")
    finally:
        shutil.rmtree(base_dir, ignore_errors=True)
