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

from siggeo.config import load_analysis_config
from siggeo.figures.generate import (
    make_figure1_dataset,
    make_figure2_regime_map,
    make_figure3_stability,
    make_figure4_retrieval,
    make_figure5_enrichment,
    make_figure6_confounds,
)
from siggeo.runs import load_metrics, tracked_phase


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate figures and matching source-data artifacts.")
    parser.add_argument("--config", required=True, type=Path, help="Path to an analysis config YAML.")
    args = parser.parse_args()

    with tracked_phase(args.config, "09_make_figures") as context:
        config = load_analysis_config(args.config)
        run_dir = context.run_dir
        metrics = load_metrics(run_dir)

        overlap = pl.read_parquet(run_dir / "tables" / "overlap_gene_table.parquet")
        gene_stats = pl.read_parquet(run_dir / "tables" / "gene_modality_stats.parquet")
        geometry = pl.read_parquet(run_dir / "tables" / "same_gene_signed_geometry.parquet")
        retrieval_summary = pl.read_csv(run_dir / "tables" / "retrieval_summary.csv")
        reactome = pl.read_csv(run_dir / "tables" / "enrichment_reactome.csv")
        corum = pl.read_csv(run_dir / "tables" / "enrichment_corum.csv")
        confounds = pl.read_csv(run_dir / "tables" / "confound_summary.csv")
        pipeline_agreement = pl.read_csv(run_dir / "tables" / "pipeline_agreement.csv")

        make_figure1_dataset(overlap, gene_stats, metrics, run_dir / "figures" / "figure1_dataset.png", run_dir / "source_data" / "figure1_dataset.csv")
        make_figure2_regime_map(
            geometry,
            run_dir / "figures" / "figure2_regime_map.png",
            run_dir / "source_data" / "figure2_regime_map.csv",
            sign_z_threshold=float(config.raw["geometry"]["sign_z_threshold"]),
            asymmetry_threshold=float(config.raw["geometry"]["asymmetry_threshold"]),
        )
        make_figure3_stability(pipeline_agreement, run_dir / "figures" / "figure3_stability.png", run_dir / "source_data" / "figure3_stability.csv")
        make_figure4_retrieval(retrieval_summary, run_dir / "figures" / "figure4_retrieval.png", run_dir / "source_data" / "figure4_retrieval.csv")
        make_figure5_enrichment(reactome, corum, run_dir / "figures" / "figure5_enrichment.png", run_dir / "source_data" / "figure5_enrichment.csv")
        make_figure6_confounds(confounds, run_dir / "figures" / "figure6_confounds.png", run_dir / "source_data" / "figure6_confounds.csv")

        print(json.dumps({"figure_count": 6, "run_dir": str(run_dir)}, indent=2))


if __name__ == "__main__":
    main()
