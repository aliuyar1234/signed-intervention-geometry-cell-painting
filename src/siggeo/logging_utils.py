from __future__ import annotations

import logging
import logging.config
from pathlib import Path

from siggeo.config import load_yaml, resolve_repo_path


def configure_logging(config_path: str | Path | None = None) -> None:
    if config_path is None:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        )
        return

    config = load_yaml(resolve_repo_path(config_path))
    logging.config.dictConfig(config)
