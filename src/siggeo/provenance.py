from __future__ import annotations

import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from siggeo.config import repo_root
from siggeo.io.atomic import write_json_atomic


def _git_output(args: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            args,
            cwd=repo_root(),
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip() or None


def collect_provenance(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(repo_root()),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "git_commit": _git_output(["git", "rev-parse", "HEAD"]),
        "git_branch": _git_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
    }
    if extra:
        payload.update(extra)
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    return write_json_atomic(path, payload, sort_keys=False)
