from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import duckdb


REQUIRED_METADATA_FILES = (
    "compound.csv.gz",
    "orf.csv.gz",
    "crispr.csv.gz",
    "perturbation_control.csv",
    "plate.csv.gz",
    "well.csv.gz",
    "compound_source.csv.gz",
    "microscope_config.csv",
    "microscope_filter.csv",
    "cellprofiler_version.csv",
    "db/setup.sql",
)

VALIDATION_TABLES = ("orf", "crispr", "well", "plate", "perturbation_control")


def _metadata_file_map(metadata_root: Path) -> dict[str, Path]:
    return {relative: metadata_root / relative for relative in REQUIRED_METADATA_FILES}


def validate_metadata_inputs(metadata_root: Path) -> dict[str, Path]:
    file_map = _metadata_file_map(metadata_root)
    missing = [str(path) for path in file_map.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing metadata inputs:\n" + "\n".join(sorted(missing)))
    return file_map


def _rewrite_setup_sql(setup_sql: str, metadata_root: Path) -> str:
    rewritten = setup_sql
    for relative, path in _metadata_file_map(metadata_root).items():
        if relative == "db/setup.sql":
            continue
        filename = Path(relative).name
        absolute = path.resolve().as_posix()
        rewritten = rewritten.replace(f"'{filename}'", f"'{absolute}'")
    return rewritten


def _fallback_build(connection: duckdb.DuckDBPyConnection, metadata_root: Path) -> None:
    file_map = _metadata_file_map(metadata_root)
    connection.execute(
        "CREATE TABLE microscope_filter AS SELECT * FROM read_csv_auto(?);",
        [str(file_map["microscope_filter.csv"])],
    )
    connection.execute(
        "CREATE TABLE microscope_config AS SELECT * FROM read_csv_auto(?);",
        [str(file_map["microscope_config.csv"])],
    )
    connection.execute(
        "CREATE TABLE cellprofiler_version AS SELECT * FROM read_csv_auto(?);",
        [str(file_map["cellprofiler_version.csv"])],
    )
    connection.execute(
        "CREATE TABLE compound AS SELECT * FROM read_csv_auto(?);",
        [str(file_map["compound.csv.gz"])],
    )
    connection.execute(
        "CREATE TABLE orf AS SELECT * FROM read_csv_auto(?);",
        [str(file_map["orf.csv.gz"])],
    )
    connection.execute(
        "CREATE TABLE crispr AS SELECT * FROM read_csv_auto(?);",
        [str(file_map["crispr.csv.gz"])],
    )
    connection.execute(
        "CREATE TABLE perturbation AS "
        "SELECT Metadata_JCP2022, 'compound' AS Metadata_perturbation_modality FROM compound "
        "UNION ALL "
        "SELECT Metadata_JCP2022, 'orf' AS Metadata_perturbation_modality FROM orf "
        "UNION ALL "
        "SELECT Metadata_JCP2022, 'crispr' AS Metadata_perturbation_modality FROM crispr"
    )
    connection.execute("INSERT INTO perturbation VALUES ('JCP2022_UNKNOWN', 'unknown');")
    connection.execute(
        "CREATE TABLE perturbation_control AS SELECT * FROM read_csv_auto(?);",
        [str(file_map["perturbation_control.csv"])],
    )
    connection.execute(
        "CREATE TABLE plate AS SELECT * FROM read_csv_auto(?);",
        [str(file_map["plate.csv.gz"])],
    )
    connection.execute(
        "CREATE TABLE compound_source AS SELECT DISTINCT * FROM read_csv_auto(?);",
        [str(file_map["compound_source.csv.gz"])],
    )
    connection.execute(
        "CREATE TABLE well AS SELECT * FROM read_csv_auto(?);",
        [str(file_map["well.csv.gz"])],
    )
    connection.execute("CREATE INDEX idx_well_jcp ON well(Metadata_JCP2022);")


def build_metadata_db(metadata_root: Path, db_path: Path) -> dict[str, Any]:
    validate_metadata_inputs(metadata_root)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db_path.unlink(missing_ok=True)

    setup_sql_path = metadata_root / "db" / "setup.sql"
    setup_sql = setup_sql_path.read_text(encoding="utf-8")
    rewritten_sql = _rewrite_setup_sql(setup_sql, metadata_root)

    build_mode = "official_setup_sql"
    with duckdb.connect(str(db_path)) as connection:
        try:
            connection.execute(rewritten_sql)
        except duckdb.Error:
            build_mode = "fallback_import"
            connection.close()
            db_path.unlink(missing_ok=True)
            with duckdb.connect(str(db_path)) as fallback_connection:
                _fallback_build(fallback_connection, metadata_root)

    with duckdb.connect(str(db_path), read_only=True) as connection:
        table_counts = {
            table: connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in VALIDATION_TABLES
        }

    report = {
        "db_path": str(db_path),
        "metadata_root": str(metadata_root),
        "build_mode": build_mode,
        "row_counts": table_counts,
    }
    report_path = db_path.with_suffix(".build_report.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
