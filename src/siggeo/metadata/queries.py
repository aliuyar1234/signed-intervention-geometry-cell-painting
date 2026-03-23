from __future__ import annotations

from pathlib import Path

import duckdb
import polars as pl


def _connect(db_path: Path, *, read_only: bool = True) -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(db_path), read_only=read_only)


def load_treated_gene_metadata(db_path: Path, modality: str) -> pl.DataFrame:
    if modality not in {"orf", "crispr"}:
        raise ValueError(f"Unsupported modality: {modality}")

    query = f"""
        SELECT
            CAST(m.Metadata_NCBI_Gene_ID AS VARCHAR) AS gene_id,
            NULLIF(TRIM(m.Metadata_Symbol), '') AS symbol
        FROM {modality} AS m
        LEFT JOIN perturbation_control AS pc
            ON m.Metadata_JCP2022 = pc.Metadata_JCP2022
        WHERE pc.Metadata_JCP2022 IS NULL
          AND NULLIF(TRIM(CAST(m.Metadata_NCBI_Gene_ID AS VARCHAR)), '') IS NOT NULL
    """
    with _connect(db_path) as connection:
        return pl.from_arrow(connection.execute(query).arrow())


def load_profile_gene_wells(db_path: Path, profile_path: Path, modality: str) -> pl.DataFrame:
    if modality not in {"orf", "crispr"}:
        raise ValueError(f"Unsupported modality: {modality}")

    query = f"""
        SELECT
            CAST(m.Metadata_NCBI_Gene_ID AS VARCHAR) AS gene_id,
            p.Metadata_Plate AS metadata_plate,
            p.Metadata_Well AS metadata_well
        FROM read_parquet(?) AS p
        JOIN well AS w
            ON p.Metadata_Plate = w.Metadata_Plate
           AND p.Metadata_Well = w.Metadata_Well
        JOIN {modality} AS m
            ON w.Metadata_JCP2022 = m.Metadata_JCP2022
        LEFT JOIN perturbation_control AS pc
            ON w.Metadata_JCP2022 = pc.Metadata_JCP2022
        WHERE pc.Metadata_JCP2022 IS NULL
          AND NULLIF(TRIM(CAST(m.Metadata_NCBI_Gene_ID AS VARCHAR)), '') IS NOT NULL
    """
    with _connect(db_path) as connection:
        return pl.from_arrow(connection.execute(query, [str(profile_path)]).arrow())
