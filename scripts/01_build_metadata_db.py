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
from siggeo.metadata.db import build_metadata_db


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the canonical metadata DuckDB database.")
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to the paths config, for example configs/paths.example.yaml.",
    )
    args = parser.parse_args()

    paths = load_paths_config(args.config).paths
    metadata_root = paths["metadata_root"]
    db_path = metadata_root / "db" / "jump_metadata.duckdb"
    report = build_metadata_db(metadata_root, db_path)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
