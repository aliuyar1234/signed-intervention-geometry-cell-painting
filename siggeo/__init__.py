from __future__ import annotations

from pathlib import Path
from pkgutil import extend_path


__path__ = extend_path(__path__, __name__)
__version__ = "0.1.0"

SRC_PACKAGE_DIR = Path(__file__).resolve().parent.parent / "src" / "siggeo"
if SRC_PACKAGE_DIR.exists():
    src_path = str(SRC_PACKAGE_DIR)
    if src_path not in __path__:
        __path__.append(src_path)
