from __future__ import annotations

import re
from pathlib import Path

import polars as pl


CELL_COUNT_PATTERNS = (
    re.compile(r"count.*cells", re.IGNORECASE),
    re.compile(r"cells.*count", re.IGNORECASE),
    re.compile(r"image.*count", re.IGNORECASE),
    re.compile(r"metadata.*count.*cells", re.IGNORECASE),
)


def schema_columns(profile_path: Path) -> list[str]:
    return list(pl.scan_parquet(profile_path).collect_schema().names())


def split_metadata_and_features(columns: list[str]) -> tuple[list[str], list[str]]:
    metadata = [column for column in columns if column.startswith("Metadata_")]
    features = [column for column in columns if not column.startswith("Metadata_")]
    return metadata, features


def exact_feature_intersection(orf_profile_path: Path, crispr_profile_path: Path) -> list[str]:
    orf_columns = schema_columns(orf_profile_path)
    crispr_columns = schema_columns(crispr_profile_path)
    _, orf_features = split_metadata_and_features(orf_columns)
    _, crispr_features = split_metadata_and_features(crispr_columns)
    orf_feature_set = set(orf_features).difference(find_cell_count_columns(orf_columns))
    crispr_feature_set = set(crispr_features).difference(find_cell_count_columns(crispr_columns))
    return sorted(orf_feature_set.intersection(crispr_feature_set))


def find_cell_count_columns(columns: list[str]) -> list[str]:
    matched: list[str] = []
    for column in columns:
        if any(pattern.search(column) for pattern in CELL_COUNT_PATTERNS):
            matched.append(column)
    return sorted(set(matched))
