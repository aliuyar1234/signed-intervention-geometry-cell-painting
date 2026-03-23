from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl

from siggeo.io.atomic import write_json_atomic, write_parquet_atomic
from siggeo.metadata.queries import load_profile_gene_wells, load_treated_gene_metadata


def _collapse_symbol(values: list[str | None]) -> str | None:
    unique = sorted({value.strip() for value in values if value and value.strip()})
    if not unique:
        return None
    return "|".join(unique)


def _display_symbol(orf_symbol: str | None, crispr_symbol: str | None, gene_id: str) -> str:
    if orf_symbol and crispr_symbol:
        if orf_symbol.casefold() == crispr_symbol.casefold():
            return orf_symbol
        return f"{orf_symbol} / {crispr_symbol}"
    if orf_symbol:
        return orf_symbol
    if crispr_symbol:
        return crispr_symbol
    return gene_id


def _single_clean_symbol(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    if not stripped or "|" in stripped:
        return None
    return stripped


def _canonical_symbol(orf_symbol: str | None, crispr_symbol: str | None) -> str | None:
    clean_orf = _single_clean_symbol(orf_symbol)
    clean_crispr = _single_clean_symbol(crispr_symbol)
    if clean_orf and clean_crispr:
        return clean_orf if clean_orf.casefold() == clean_crispr.casefold() else None
    if clean_orf and crispr_symbol is None:
        return clean_orf
    if clean_crispr and orf_symbol is None:
        return clean_crispr
    return None


def _aggregate_symbols(frame: pl.DataFrame, symbol_name: str) -> pl.DataFrame:
    if frame.is_empty():
        return pl.DataFrame({"gene_id": [], symbol_name: []})
    return (
        frame.group_by("gene_id")
        .agg(pl.col("symbol").drop_nulls().sort().unique())
        .with_columns(pl.col("symbol").map_elements(_collapse_symbol, return_dtype=pl.String).alias(symbol_name))
        .select("gene_id", symbol_name)
    )


def _aggregate_counts(frame: pl.DataFrame, prefix: str) -> pl.DataFrame:
    if frame.is_empty():
        return pl.DataFrame(
            {
                "gene_id": [],
                f"n_{prefix}_wells": [],
                f"n_{prefix}_plates": [],
            }
        )
    return (
        frame.group_by("gene_id")
        .agg(
            pl.len().alias(f"n_{prefix}_wells"),
            pl.col("metadata_plate").n_unique().alias(f"n_{prefix}_plates"),
        )
        .sort("gene_id")
    )


def _drop_reason(n_orf_wells: int, n_crispr_wells: int, min_wells_per_modality: int) -> str | None:
    if n_orf_wells >= min_wells_per_modality and n_crispr_wells >= min_wells_per_modality:
        return None
    if n_orf_wells < min_wells_per_modality and n_crispr_wells < min_wells_per_modality:
        return "insufficient_wells_both_modalities"
    if n_orf_wells < min_wells_per_modality:
        return "insufficient_orf_wells"
    return "insufficient_crispr_wells"


def make_overlap_gene_table(
    db_path: Path,
    orf_profile_path: Path,
    crispr_profile_path: Path,
    *,
    min_wells_per_modality: int = 2,
) -> pl.DataFrame:
    orf_meta = _aggregate_symbols(load_treated_gene_metadata(db_path, "orf"), "orf_symbol")
    crispr_meta = _aggregate_symbols(load_treated_gene_metadata(db_path, "crispr"), "crispr_symbol")
    orf_counts = _aggregate_counts(load_profile_gene_wells(db_path, orf_profile_path, "orf"), "orf")
    crispr_counts = _aggregate_counts(load_profile_gene_wells(db_path, crispr_profile_path, "crispr"), "crispr")

    return (
        orf_meta.join(crispr_meta, on="gene_id", how="inner")
        .join(orf_counts, on="gene_id", how="left")
        .join(crispr_counts, on="gene_id", how="left")
        .with_columns(
            pl.col("n_orf_wells").fill_null(0).cast(pl.Int64),
            pl.col("n_crispr_wells").fill_null(0).cast(pl.Int64),
            pl.col("n_orf_plates").fill_null(0).cast(pl.Int64),
            pl.col("n_crispr_plates").fill_null(0).cast(pl.Int64),
        )
        .with_columns(
            pl.struct(["orf_symbol", "crispr_symbol", "gene_id"])
            .map_elements(
                lambda row: _display_symbol(
                    row["orf_symbol"],
                    row["crispr_symbol"],
                    row["gene_id"],
                ),
                return_dtype=pl.String,
            )
            .alias("display_symbol"),
            pl.struct(["orf_symbol", "crispr_symbol"])
            .map_elements(
                lambda row: _canonical_symbol(row["orf_symbol"], row["crispr_symbol"]),
                return_dtype=pl.String,
            )
            .alias("canonical_symbol"),
        )
        .with_columns(
            ((pl.col("n_orf_wells") >= min_wells_per_modality) & (pl.col("n_crispr_wells") >= min_wells_per_modality)).alias("keep_for_analysis"),
            ((pl.col("n_orf_wells") < max(4, min_wells_per_modality)) | (pl.col("n_crispr_wells") < max(4, min_wells_per_modality))).alias("low_replicate_support"),
        )
        .with_columns(
            pl.struct(["n_orf_wells", "n_crispr_wells"])
            .map_elements(
                lambda row: _drop_reason(
                    row["n_orf_wells"],
                    row["n_crispr_wells"],
                    min_wells_per_modality,
                ),
                return_dtype=pl.String,
            )
            .alias("drop_reason")
        )
        .select(
            "gene_id",
            "orf_symbol",
            "crispr_symbol",
            "canonical_symbol",
            "display_symbol",
            "n_orf_wells",
            "n_crispr_wells",
            "n_orf_plates",
            "n_crispr_plates",
            "low_replicate_support",
            "keep_for_analysis",
            "drop_reason",
        )
        .sort("gene_id")
    )


def write_overlap_outputs(
    overlap: pl.DataFrame,
    output_path: Path,
    metrics_path: Path | None = None,
) -> dict[str, Any]:
    write_parquet_atomic(overlap, output_path)
    metrics = {
        "overlap_gene_count": overlap.height,
        "keep_for_analysis_count": int(overlap["keep_for_analysis"].sum()),
    }
    if metrics_path is not None:
        write_json_atomic(metrics_path, metrics)
    return metrics
