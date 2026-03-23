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

from siggeo.compute import load_compute_config
from siggeo.config import load_analysis_config
from siggeo.geometry.signed_scores import GeometryConfig, compute_signed_geometry, write_geometry_outputs
from siggeo.runs import tracked_phase, update_metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute signed geometry, matched nulls, and regime assignments."
    )
    parser.add_argument("--config", required=True, type=Path, help="Path to an analysis config YAML.")
    args = parser.parse_args()

    with tracked_phase(args.config, "05_compute_signed_geometry") as context:
        config = load_analysis_config(args.config)
        compute_cfg = load_compute_config(config.raw, run_dir=context.run_dir)
        geometry_cfg = config.raw["geometry"]
        bootstrap_cfg = config.raw["bootstrap"]
        activity_cfg = config.raw["activity"]

        gene_stats = pl.read_parquet(context.run_dir / "tables" / "gene_modality_stats.parquet")
        feature_list = (context.run_dir / "tables" / "feature_intersection.txt").read_text(encoding="utf-8").splitlines()
        centered_orf = pl.read_parquet(context.run_dir / "tables" / f"centered_wells_{config.pipeline}_orf.parquet")
        centered_crispr = pl.read_parquet(context.run_dir / "tables" / f"centered_wells_{config.pipeline}_crispr.parquet")

        geometry = compute_signed_geometry(
            gene_stats.filter(pl.col("modality") == "orf"),
            gene_stats.filter(pl.col("modality") == "crispr"),
            centered_orf,
            centered_crispr,
            feature_list,
            strength_threshold=float(activity_cfg["strength_z_threshold"]),
            rep_threshold=float(activity_cfg["rep_z_threshold"]),
            config=GeometryConfig(
                activity_deciles=int(geometry_cfg["activity_deciles"]),
                n_cross_null_samples_per_cell=int(geometry_cfg["n_cross_null_samples_per_cell"]),
                min_pairs_per_null_cell=int(geometry_cfg["min_pairs_per_null_cell"]),
                sign_z_threshold=float(geometry_cfg["sign_z_threshold"]),
                asymmetry_threshold=float(geometry_cfg["asymmetry_threshold"]),
                null_std_floor=float(geometry_cfg["null_std_floor"]),
                n_bootstrap=int(bootstrap_cfg["n_bootstrap"]),
                consensus_threshold=float(bootstrap_cfg["consensus_threshold"]),
                random_seed=config.seed,
            ),
            compute_config=compute_cfg,
            run_dir=context.run_dir,
        )

        summary = write_geometry_outputs(
            geometry,
            context.run_dir / "tables" / "same_gene_signed_geometry.parquet",
            context.run_dir / "logs" / "05_compute_signed_geometry.json",
        )
        update_metrics(
            context.run_dir,
            {
                "confident_noninactive_count": summary["confident_noninactive_count"],
                "null_fallback_counts": summary.get("null_fallback_counts", {}),
            },
        )
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
