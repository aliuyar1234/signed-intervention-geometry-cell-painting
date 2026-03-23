from __future__ import annotations

import sys
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parent / "src"
if SRC_DIR.exists():
    src = str(SRC_DIR)
    if src not in sys.path:
        sys.path.insert(0, src)
