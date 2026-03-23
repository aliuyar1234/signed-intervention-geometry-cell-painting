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

from siggeo.config import load_paths_config
from siggeo.io.downloads import download_required_assets


def main() -> None:
    parser = argparse.ArgumentParser(description="Download locked public assets and record provenance.")
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to the paths config, for example configs/paths.example.yaml.",
    )
    parser.add_argument("--refresh", action="store_true", help="Redownload assets even when local copies exist.")
    parser.add_argument("--timeout-seconds", type=int, default=120, help="Network timeout in seconds.")
    args = parser.parse_args()

    report = download_required_assets(
        load_paths_config(args.config),
        refresh=args.refresh,
        timeout_seconds=args.timeout_seconds,
    )
    print(json.dumps({"report_path": "data/raw/download_report.json", "n_assets": len(report["results"])}, indent=2))


if __name__ == "__main__":
    main()
