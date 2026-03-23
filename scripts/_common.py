from __future__ import annotations

import argparse
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SRC_DIR = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from siggeo.runs import register_phase_stub


def run_stub_phase(phase_name: str, description: str, extra: dict[str, object] | None = None) -> None:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", required=True, type=Path, help="Path to the YAML config file.")
    parser.add_argument("--mode", required=False, help="Optional execution mode for scripts that need it.")
    args = parser.parse_args()

    payload = dict(extra or {})
    if args.mode:
        payload["mode_override"] = args.mode

    context = register_phase_stub(args.config, phase_name=phase_name, extra=payload)
    print(f"[stub] {phase_name} wired for run {context.config.run_name}")
    print(f"[stub] Run directory: {context.run_dir}")
