from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Callable

import polars as pl


def _temp_path(path: Path, suffix: str | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw_path = tempfile.mkstemp(
        prefix=f".{path.stem}.",
        suffix=suffix or f"{path.suffix}.tmp",
        dir=str(path.parent),
    )
    os.close(fd)
    return Path(raw_path)


def _replace_atomic(temp_path: Path, final_path: Path) -> Path:
    temp_path.replace(final_path)
    return final_path


def write_text_atomic(path: Path, content: str, *, encoding: str = "utf-8") -> Path:
    temp_path = _temp_path(path)
    try:
        temp_path.write_text(content, encoding=encoding)
        return _replace_atomic(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def write_json_atomic(path: Path, payload: Any, *, sort_keys: bool = False) -> Path:
    return write_text_atomic(path, json.dumps(payload, indent=2, sort_keys=sort_keys))


def write_csv_atomic(frame: pl.DataFrame, path: Path) -> Path:
    temp_path = _temp_path(path, ".csv.tmp")
    try:
        frame.write_csv(temp_path)
        return _replace_atomic(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def write_parquet_atomic(frame: pl.DataFrame, path: Path) -> Path:
    temp_path = _temp_path(path, ".parquet.tmp")
    try:
        frame.write_parquet(temp_path)
        return _replace_atomic(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def save_figure_atomic(path: Path, save_func: Callable[[Path], None]) -> Path:
    temp_path = _temp_path(path, path.suffix)
    try:
        save_func(temp_path)
        return _replace_atomic(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)
