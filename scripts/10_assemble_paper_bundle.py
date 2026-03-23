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

from siggeo.config import load_analysis_config
from siggeo.io.atomic import write_csv_atomic
from siggeo.paper.claims import build_claim_evidence_matrix
from siggeo.paper.manuscript import write_manuscript_bundle
from siggeo.paper.source_data import bundle_source_data
from siggeo.runs import tracked_phase


def main() -> None:
    parser = argparse.ArgumentParser(description="Assemble the manuscript, source data, and claim bundle.")
    parser.add_argument("--config", required=True, type=Path, help="Path to an analysis config YAML.")
    args = parser.parse_args()

    with tracked_phase(args.config, "10_assemble_paper_bundle") as context:
        config = load_analysis_config(args.config)
        run_dir = config.result_dir / config.run_name
        paper_dir = run_dir / "paper"
        paper_dir.mkdir(parents=True, exist_ok=True)
        claim_matrix = build_claim_evidence_matrix(run_dir)
        write_csv_atomic(claim_matrix, paper_dir / "claim_evidence_matrix.csv")
        write_manuscript_bundle(run_dir, paper_dir, claim_matrix)
        bundle_source_data(run_dir, paper_dir)
        print(json.dumps({"paper_dir": str(paper_dir), "claim_rows": claim_matrix.height}, indent=2))


if __name__ == "__main__":
    main()
