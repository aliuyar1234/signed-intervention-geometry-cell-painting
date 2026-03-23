from __future__ import annotations

import shutil
from pathlib import Path


def bundle_source_data(run_dir: Path, paper_dir: Path) -> None:
    target = paper_dir / "source_data_bundle"
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(run_dir / "source_data", target)
