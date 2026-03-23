from __future__ import annotations

import gzip
import shutil
from pathlib import Path

import polars as pl

from siggeo.config import repo_root
from siggeo.metadata.db import build_metadata_db
from siggeo.metadata.overlap import make_overlap_gene_table


def _write_metadata_fixture(metadata_root: Path) -> None:
    (metadata_root / "db").mkdir(parents=True, exist_ok=True)
    (metadata_root / "microscope_filter.csv").write_text(
        "Metadata_Filter_Configuration\nF1\n",
        encoding="utf-8",
    )
    (metadata_root / "microscope_config.csv").write_text(
        "Metadata_Source,Metadata_Filter_Configuration\nSRC1,F1\n",
        encoding="utf-8",
    )
    (metadata_root / "cellprofiler_version.csv").write_text(
        "Metadata_Source,Metadata_CellProfiler_Version\nSRC1,4.2.0\n",
        encoding="utf-8",
    )
    with gzip.open(metadata_root / "compound.csv.gz", "wt", encoding="utf-8") as handle:
        handle.write("Metadata_JCP2022,Metadata_InChIKey,Metadata_InChI,Metadata_SMILES\n")
    with gzip.open(metadata_root / "orf.csv.gz", "wt", encoding="utf-8") as handle:
        handle.write(
            "Metadata_JCP2022,Metadata_broad_sample,Metadata_Name,Metadata_Vector,"
            "Metadata_Transcript,Metadata_Symbol,Metadata_NCBI_Gene_ID,Metadata_Taxon_ID,"
            "Metadata_Gene_Description,Metadata_Prot_Match,Metadata_Insert_Length,Metadata_pert_type\n"
            "ORF_A,,A,,,GENEA,101,,,0,0,trt\n"
            "ORF_A_ALIAS,,A2,,,ALIASA,101,,,0,0,trt\n"
            "ORF_B,,B,,,GENEB,202,,,0,0,trt\n"
            "ORF_CTRL,,CTRL,,,CTRL,,,,0,0,negcon\n"
        )
    with gzip.open(metadata_root / "crispr.csv.gz", "wt", encoding="utf-8") as handle:
        handle.write(
            "Metadata_JCP2022,Metadata_NCBI_Gene_ID,Metadata_Symbol\n"
            "CRISPR_A,101,GENEA\n"
            "CRISPR_B,202,DIFFB\n"
            "CRISPR_CTRL,,CTRL\n"
        )
    (metadata_root / "perturbation_control.csv").write_text(
        "Metadata_JCP2022,Metadata_pert_type,Metadata_Name\n"
        "ORF_CTRL,negcon,NEGCTRL\n"
        "CRISPR_CTRL,negcon,NEGCTRL\n",
        encoding="utf-8",
    )
    with gzip.open(metadata_root / "plate.csv.gz", "wt", encoding="utf-8") as handle:
        handle.write(
            "Metadata_Source,Metadata_Batch,Metadata_Plate,Metadata_PlateType\n"
            "SRC1,B1,P1,ORF\n"
            "SRC1,B1,P2,CRISPR\n"
            "SRC1,B2,P3,ORF\n"
            "SRC1,B2,P4,CRISPR\n"
        )
    with gzip.open(metadata_root / "well.csv.gz", "wt", encoding="utf-8") as handle:
        handle.write(
            "Metadata_Source,Metadata_Plate,Metadata_Well,Metadata_JCP2022\n"
            "SRC1,P1,A01,ORF_A\n"
            "SRC1,P1,A02,ORF_A_ALIAS\n"
            "SRC1,P3,A01,ORF_B\n"
            "SRC1,P3,A02,ORF_CTRL\n"
            "SRC1,P2,A01,CRISPR_A\n"
            "SRC1,P2,A02,CRISPR_A\n"
            "SRC1,P4,A01,CRISPR_B\n"
            "SRC1,P4,A02,CRISPR_B\n"
            "SRC1,P4,A03,CRISPR_CTRL\n"
        )
    with gzip.open(metadata_root / "compound_source.csv.gz", "wt", encoding="utf-8") as handle:
        handle.write("Metadata_Compound_Source,Metadata_JCP2022\n")
    (metadata_root / "db" / "setup.sql").write_text(
        "\n".join(
            [
                "CREATE TABLE microscope_filter AS SELECT * FROM read_csv_auto('microscope_filter.csv');",
                "CREATE TABLE microscope_config AS SELECT * FROM read_csv_auto('microscope_config.csv');",
                "CREATE TABLE cellprofiler_version AS SELECT * FROM read_csv_auto('cellprofiler_version.csv');",
                "CREATE TABLE compound AS SELECT * FROM read_csv_auto('compound.csv.gz');",
                "CREATE TABLE orf AS SELECT * FROM read_csv_auto('orf.csv.gz');",
                "CREATE TABLE crispr AS SELECT * FROM read_csv_auto('crispr.csv.gz');",
                "CREATE TABLE perturbation AS "
                "SELECT Metadata_JCP2022, 'compound' AS Metadata_perturbation_modality FROM compound "
                "UNION ALL SELECT Metadata_JCP2022, 'orf' AS Metadata_perturbation_modality FROM orf "
                "UNION ALL SELECT Metadata_JCP2022, 'crispr' AS Metadata_perturbation_modality FROM crispr;",
                "INSERT INTO perturbation VALUES ('JCP2022_UNKNOWN', 'unknown');",
                "CREATE TABLE perturbation_control AS SELECT * FROM read_csv_auto('perturbation_control.csv');",
                "CREATE TABLE plate AS SELECT * FROM read_csv_auto('plate.csv.gz');",
                "CREATE TABLE compound_source AS SELECT * FROM read_csv_auto('compound_source.csv.gz');",
                "CREATE TABLE well AS SELECT * FROM read_csv_auto('well.csv.gz');",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_build_metadata_db_and_overlap_table() -> None:
    base_dir = repo_root() / "results" / "test_metadata_overlap"
    shutil.rmtree(base_dir, ignore_errors=True)
    base_dir.mkdir(parents=True, exist_ok=True)
    try:
        metadata_root = base_dir / "metadata"
        _write_metadata_fixture(metadata_root)
        db_path = metadata_root / "db" / "jump_metadata.duckdb"
        report = build_metadata_db(metadata_root, db_path)

        assert report["row_counts"]["orf"] == 4
        assert report["row_counts"]["crispr"] == 3

        orf_profile = pl.DataFrame(
            {
                "Metadata_Plate": ["P1", "P1", "P3", "P3"],
                "Metadata_Well": ["A01", "A02", "A01", "A02"],
                "Cells_AreaShape_Area": [1.0, 2.0, 3.0, 4.0],
            }
        )
        crispr_profile = pl.DataFrame(
            {
                "Metadata_Plate": ["P2", "P2", "P4", "P4"],
                "Metadata_Well": ["A01", "A02", "A01", "A02"],
                "Cells_AreaShape_Area": [1.0, 2.0, 3.0, 4.0],
            }
        )
        orf_path = base_dir / "orf.parquet"
        crispr_path = base_dir / "crispr.parquet"
        orf_profile.write_parquet(orf_path)
        crispr_profile.write_parquet(crispr_path)

        overlap = make_overlap_gene_table(db_path, orf_path, crispr_path)
        rows = {row["gene_id"]: row for row in overlap.to_dicts()}

        assert set(rows) == {"101", "202"}
        assert rows["101"]["keep_for_analysis"] is True
        assert rows["101"]["low_replicate_support"] is True
        assert rows["101"]["n_orf_wells"] == 2
        assert rows["101"]["n_crispr_wells"] == 2
        assert rows["202"]["keep_for_analysis"] is False
        assert rows["202"]["drop_reason"] == "insufficient_orf_wells"
        assert rows["202"]["orf_symbol"] == "GENEB"
        assert rows["202"]["crispr_symbol"] == "DIFFB"
        assert rows["101"]["canonical_symbol"] is None
        assert rows["202"]["canonical_symbol"] is None
    finally:
        shutil.rmtree(base_dir, ignore_errors=True)
