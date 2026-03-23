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
from siggeo.evaluation.retrieval import RetrievalConfig, evaluate_retrieval, write_retrieval_outputs
from siggeo.runs import tracked_phase, update_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run same-gene ORF/CRISPR retrieval evaluation.")
    parser.add_argument("--config", required=True, type=Path, help="Path to an analysis config YAML.")
    args = parser.parse_args()

    with tracked_phase(args.config, "06_run_retrieval") as context:
        config = load_analysis_config(args.config)
        geometry_cfg = config.raw["geometry"]
        retrieval_cfg = config.raw["retrieval"]

        gene_stats = pl.read_parquet(context.run_dir / "tables" / "gene_modality_stats.parquet")
        ranks, summary = evaluate_retrieval(
            gene_stats.filter(pl.col("modality") == "orf"),
            gene_stats.filter(pl.col("modality") == "crispr"),
            config=RetrievalConfig(
                lambda_values=[float(value) for value in retrieval_cfg["lambda_values"]],
                primary_lambda=float(retrieval_cfg["primary_lambda"]),
                activity_deciles=int(geometry_cfg["activity_deciles"]),
                n_cross_null_samples_per_cell=int(geometry_cfg["n_cross_null_samples_per_cell"]),
                min_pairs_per_null_cell=int(geometry_cfg["min_pairs_per_null_cell"]),
                null_std_floor=float(geometry_cfg["null_std_floor"]),
                random_seed=config.seed,
            ),
        )
        write_retrieval_outputs(
            ranks,
            summary,
            context.run_dir / "tables" / "retrieval_ranks.parquet",
            context.run_dir / "tables" / "retrieval_summary.csv",
        )

        primary_score = f"regime_score_lambda_{float(retrieval_cfg['primary_lambda']):g}"
        primary_row = summary.filter(pl.col("score_name") == primary_score).sort("direction")
        retrieval_metrics_by_score: dict[str, dict[str, dict[str, float | int]]] = {}
        for row in summary.to_dicts():
            retrieval_metrics_by_score.setdefault(str(row["score_name"]), {})[str(row["direction"])] = {
                "mrr": float(row["mrr"]),
                "hits_at_1": float(row["hits_at_1"]),
                "n_queries": int(row["n_queries"]),
            }
        update_metrics(
            context.run_dir,
            {
                "retrieval_primary_score": primary_score,
                "retrieval_primary_mrr_mean": float(primary_row["mrr"].mean()) if primary_row.height else 0.0,
                "retrieval_metrics_by_score": retrieval_metrics_by_score,
            },
        )
        print(json.dumps({"ranks_rows": ranks.height, "summary_rows": summary.height}, indent=2))


if __name__ == "__main__":
    main()
