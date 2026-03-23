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

from siggeo.config import load_analysis_config, resolve_repo_path
from siggeo.io.atomic import write_json_atomic, write_parquet_atomic
from siggeo.metadata.overlap import make_overlap_gene_table
from siggeo.runs import tracked_phase, update_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Construct the overlap gene table keyed by NCBI Gene ID.")
    parser.add_argument("--config", required=True, type=Path, help="Path to an analysis config YAML.")
    args = parser.parse_args()

    with tracked_phase(args.config, "02_make_overlap_gene_table") as context:
        config = load_analysis_config(args.config)
        overlap = make_overlap_gene_table(
            db_path=resolve_repo_path(config.raw["data"]["metadata_db"]),
            orf_profile_path=resolve_repo_path(config.raw["data"]["orf_profile"]),
            crispr_profile_path=resolve_repo_path(config.raw["data"]["crispr_profile"]),
            min_wells_per_modality=int(config.raw["activity"]["min_treated_wells_per_gene_modality"]),
        )

        output_path = context.run_dir / "tables" / "overlap_gene_table.parquet"
        write_parquet_atomic(overlap, output_path)

        metrics = {
            "overlap_gene_count": overlap.height,
            "keep_for_analysis_count": int(overlap["keep_for_analysis"].sum()),
        }
        update_metrics(context.run_dir, metrics)
        write_json_atomic(
            context.run_dir / "logs" / "02_make_overlap_gene_table.json",
            {
                "status": "completed",
                "run_name": context.config.run_name,
                "output_path": str(output_path),
                **metrics,
            },
        )
        print(json.dumps({"output_path": str(output_path), **metrics}, indent=2))


if __name__ == "__main__":
    main()
