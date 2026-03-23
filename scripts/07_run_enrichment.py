from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SRC_DIR = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import polars as pl

from siggeo.config import load_analysis_config, resolve_repo_path
from siggeo.evaluation.enrichment import EnrichmentConfig, run_enrichment, write_enrichment_outputs
from siggeo.runs import tracked_phase, update_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Reactome and CORUM enrichment analyses.")
    parser.add_argument("--config", required=True, type=Path, help="Path to an analysis config YAML.")
    args = parser.parse_args()

    with tracked_phase(args.config, "07_run_enrichment") as context:
        config = load_analysis_config(args.config)
        geometry = pl.read_parquet(context.run_dir / "tables" / "same_gene_signed_geometry.parquet")
        primary_geometry = geometry.filter(
            pl.col("regime_label").is_in(["aligned", "inverse", "orthogonal", "asymmetric_ORF", "asymmetric_CRISPR"])
        )
        canonical_non_null = (
            int(primary_geometry.filter(pl.col("canonical_symbol").is_not_null()).height)
            if "canonical_symbol" in primary_geometry.columns
            else 0
        )
        canonical_coverage = (
            float(canonical_non_null / primary_geometry.height) if primary_geometry.height else None
        )
        duplicate_canonical_symbols = (
            int(
                primary_geometry.filter(pl.col("canonical_symbol").is_not_null())
                .group_by("canonical_symbol")
                .len()
                .filter(pl.col("len") > 1)
                .height
            )
            if "canonical_symbol" in primary_geometry.columns
            else 0
        )
        reactome, corum = run_enrichment(
            geometry,
            resolve_repo_path(config.raw["data"]["reactome_path"]),
            resolve_repo_path(config.raw["data"]["corum_path"]),
            config=EnrichmentConfig(
                reactome_min_set_size=int(config.raw["enrichment"]["reactome_min_set_size"]),
                corum_min_set_size=int(config.raw["enrichment"]["corum_min_set_size"]),
                primary_min_regime_size=int(config.raw["enrichment"]["primary_min_regime_size"]),
                fdr_method=str(config.raw["enrichment"]["fdr_method"]),
                allow_legacy_display_symbol_fallback=bool(
                    config.raw.get("enrichment", {}).get("allow_legacy_display_symbol_fallback", False)
                ),
            ),
        )
        write_enrichment_outputs(
            reactome,
            corum,
            context.run_dir / "tables" / "enrichment_reactome.csv",
            context.run_dir / "tables" / "enrichment_corum.csv",
        )
        update_metrics(
            context.run_dir,
            {
                "significant_enrichment_counts": {
                    "reactome": int(reactome.filter(pl.col("q_value") < 0.05).height) if reactome.height else 0,
                    "corum": int(corum.filter(pl.col("q_value") < 0.05).height) if corum.height else 0,
                },
                "enrichment_symbol_coverage": canonical_coverage,
                "duplicate_canonical_symbols": duplicate_canonical_symbols,
            },
        )
        print(
            json.dumps(
                {
                    "reactome_rows": reactome.height,
                    "corum_rows": corum.height,
                    "canonical_coverage": canonical_coverage,
                    "duplicate_canonical_symbols": duplicate_canonical_symbols,
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
