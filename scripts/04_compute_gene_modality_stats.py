from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
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
from siggeo.gene_stats.activity import (
    ActivityConfig,
    compute_gene_modality_stats,
    recompute_postfilter_eligibility,
    write_gene_stats_outputs,
)
from siggeo.io.atomic import write_csv_atomic
from siggeo.runs import tracked_phase, update_metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute gene-level centroids, activity, and reproducibility statistics."
    )
    parser.add_argument("--config", required=True, type=Path, help="Path to an analysis config YAML.")
    args = parser.parse_args()

    with tracked_phase(args.config, "04_compute_gene_modality_stats") as context:
        config = load_analysis_config(args.config)
        compute_cfg = load_compute_config(config.raw, run_dir=context.run_dir)
        activity_cfg = config.raw["activity"]

        overlap_df = pl.read_parquet(context.run_dir / "tables" / "overlap_gene_table.parquet")
        feature_list = (context.run_dir / "tables" / "feature_intersection.txt").read_text(encoding="utf-8").splitlines()
        centered_orf = pl.read_parquet(context.run_dir / "tables" / f"centered_wells_{config.pipeline}_orf.parquet")
        centered_crispr = pl.read_parquet(context.run_dir / "tables" / f"centered_wells_{config.pipeline}_crispr.parquet")
        overlap_df, postfilter_qc = recompute_postfilter_eligibility(
            overlap_df,
            centered_orf,
            centered_crispr,
            min_wells_per_modality=int(activity_cfg.get("min_treated_wells_per_gene_modality", 2)),
        )
        write_csv_atomic(postfilter_qc, context.run_dir / "tables" / "postfilter_gene_eligibility.csv")
        summary_payload: dict[str, object] = {}

        def _compute_for_modality(modality: str) -> tuple[str, pl.DataFrame]:
            centered_df = centered_orf if modality == "orf" else centered_crispr
            stats = compute_gene_modality_stats(
                centered_df,
                overlap_df,
                feature_list,
                pipeline=config.pipeline,
                modality=modality,
                config=ActivityConfig(
                    n_strength_null_samples=int(activity_cfg["n_strength_null_samples"]),
                    n_rep_null_samples=int(activity_cfg["n_rep_null_samples"]),
                    strength_z_threshold=float(activity_cfg["strength_z_threshold"]),
                    rep_z_threshold=float(activity_cfg["rep_z_threshold"]),
                    random_seed=config.seed,
                ),
                compute_config=compute_cfg,
                run_dir=context.run_dir,
            )
            return modality, stats

        if compute_cfg.parallelize_modalities and compute_cfg.parallel_backend == "serial":
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = [executor.submit(_compute_for_modality, modality) for modality in ("orf", "crispr")]
                outputs = [future.result() for future in futures]
        else:
            outputs = [_compute_for_modality(modality) for modality in ("orf", "crispr")]

        output_frames: list[pl.DataFrame] = []
        for modality, stats in sorted(outputs, key=lambda item: item[0]):
            output_frames.append(stats)
            summary_payload[f"active_{modality}_count"] = int(stats["active_bool"].sum()) if stats.height else 0

        combined = pl.concat(output_frames).sort(["modality", "gene_id"])
        summary = write_gene_stats_outputs(
            combined,
            context.run_dir / "tables" / "gene_modality_stats.parquet",
            context.run_dir / "logs" / "04_compute_gene_modality_stats.json",
        )
        active_both = (
            combined.pivot(on="modality", index="gene_id", values="active_bool")
            .with_columns(
                pl.col("orf").fill_null(False),
                pl.col("crispr").fill_null(False),
            )
            .filter(pl.col("orf") & pl.col("crispr"))
            .height
            if combined.height
            else 0
        )
        update_metrics(
            context.run_dir,
            {
                "active_orf_count": summary_payload.get("active_orf_count", 0),
                "active_crispr_count": summary_payload.get("active_crispr_count", 0),
                "active_both_count": active_both,
                "postfilter_keep_for_analysis_count": int(postfilter_qc["postfilter_keep_for_analysis"].sum()),
                "postfilter_dropped_gene_count": int(
                    postfilter_qc.filter(~pl.col("postfilter_keep_for_analysis")).height
                ),
            },
        )
        print(json.dumps({**summary, **summary_payload, "active_both_count": active_both}, indent=2))


if __name__ == "__main__":
    main()
