from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]


def repo_root() -> Path:
    return REPO_ROOT


def resolve_repo_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return repo_root() / path


def load_yaml(path_like: str | Path) -> dict[str, Any]:
    path = resolve_repo_path(path_like)
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Expected mapping at {path}, got {type(data).__name__}")
    return data


def write_yaml(path_like: str | Path, payload: dict[str, Any]) -> Path:
    path = resolve_repo_path(path_like)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)
    return path


@dataclass(frozen=True)
class AnalysisConfig:
    path: Path
    raw: dict[str, Any]

    @property
    def run_name(self) -> str:
        return str(self.raw["run_name"])

    @property
    def project_name(self) -> str:
        return str(self.raw.get("project_name", "signed_intervention_geometry"))

    @property
    def pipeline(self) -> str:
        return str(self.raw.get("pipeline", "unknown"))

    @property
    def mode(self) -> str:
        return str(self.raw.get("mode", "full"))

    @property
    def seed(self) -> int:
        return int(self.raw.get("seed", 1337))

    @property
    def result_dir(self) -> Path:
        outputs = self.raw.get("outputs", {})
        result_dir = outputs.get("result_dir", "results/runs")
        return resolve_repo_path(result_dir)

    def snapshot(self) -> dict[str, Any]:
        return dict(self.raw)


@dataclass(frozen=True)
class PathsConfig:
    path: Path
    raw: dict[str, Any]

    @property
    def paths(self) -> dict[str, Path]:
        values = self.raw.get("paths", {})
        return {name: resolve_repo_path(value) for name, value in values.items()}


def load_analysis_config(path_like: str | Path) -> AnalysisConfig:
    path = resolve_repo_path(path_like)
    data = load_yaml(path)
    if "run_name" not in data:
        raise KeyError(f"Analysis config missing 'run_name': {path}")
    return AnalysisConfig(path=path, raw=data)


def load_paths_config(path_like: str | Path) -> PathsConfig:
    path = resolve_repo_path(path_like)
    data = load_yaml(path)
    if "paths" not in data:
        raise KeyError(f"Paths config missing 'paths': {path}")
    return PathsConfig(path=path, raw=data)
