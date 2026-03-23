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
from siggeo.paper.release_report import write_release_report
from siggeo.runs import tracked_phase


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the release report and final status bundle.")
    parser.add_argument("--config", required=True, type=Path, help="Path to an analysis config YAML.")
    args = parser.parse_args()

    with tracked_phase(args.config, "11_make_release_report") as context:
        config = load_analysis_config(args.config)
        run_dir = config.result_dir / config.run_name
        report = write_release_report(run_dir, run_dir / "paper")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
