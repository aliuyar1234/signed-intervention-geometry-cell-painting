from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ComputeConfig:
    parallel_backend: str
    max_workers: int
    phase4_workers: int
    phase5_workers: int
    phase8_workers: int
    phase4_gene_chunk_size: int
    phase5_gene_chunk_size: int
    phase8_gene_chunk_size: int
    numpy_threads_per_worker: int
    parallelize_modalities: bool
    mmap_root: Path
    temp_root: Path
    max_concurrent_runs: int


def _auto_worker_count() -> int:
    cpu_count = os.cpu_count() or 4
    return min(max(cpu_count - 2, 4), 24)


def _resolve_worker_count(raw_value: Any, default: int) -> int:
    if raw_value in {None, "auto"}:
        return default
    value = int(raw_value)
    if value < 1:
        raise ValueError(f"Worker count must be >= 1, got {value}.")
    return value


def _resolve_positive_int(raw_value: Any, default: int, label: str) -> int:
    if raw_value in {None, "auto"}:
        return default
    value = int(raw_value)
    if value < 1:
        raise ValueError(f"{label} must be >= 1, got {value}.")
    return value


def load_compute_config(raw_config: dict[str, Any], *, run_dir: Path) -> ComputeConfig:
    compute_raw = dict(raw_config.get("compute", {}))
    default_workers = _auto_worker_count()
    max_workers = _resolve_worker_count(compute_raw.get("max_workers"), default_workers)
    phase4_workers = _resolve_worker_count(compute_raw.get("phase4_workers"), max_workers)
    phase5_workers = _resolve_worker_count(compute_raw.get("phase5_workers"), max_workers)
    phase8_workers = _resolve_worker_count(compute_raw.get("phase8_workers"), max_workers)
    phase4_chunk_size = _resolve_positive_int(
        compute_raw.get("phase4_gene_chunk_size"),
        64,
        "compute.phase4_gene_chunk_size",
    )
    phase5_chunk_size = _resolve_positive_int(
        compute_raw.get("phase5_gene_chunk_size"),
        32,
        "compute.phase5_gene_chunk_size",
    )
    phase8_chunk_size = _resolve_positive_int(
        compute_raw.get("phase8_gene_chunk_size"),
        32,
        "compute.phase8_gene_chunk_size",
    )
    numpy_threads = _resolve_positive_int(
        compute_raw.get("numpy_threads_per_worker"),
        1,
        "compute.numpy_threads_per_worker",
    )
    parallel_backend = str(compute_raw.get("parallel_backend", "process_pool"))
    if parallel_backend not in {"serial", "process_pool"}:
        raise ValueError(
            "compute.parallel_backend must be one of {'serial', 'process_pool'}, "
            f"got {parallel_backend!r}."
        )
    mmap_root_raw = compute_raw.get("mmap_root", "auto")
    temp_root_raw = compute_raw.get("temp_root", "auto")
    mmap_root = run_dir / "work" / "memmaps" if mmap_root_raw in {None, "auto"} else Path(mmap_root_raw)
    temp_root = run_dir / "work" / "tmp" if temp_root_raw in {None, "auto"} else Path(temp_root_raw)
    return ComputeConfig(
        parallel_backend=parallel_backend,
        max_workers=max_workers,
        phase4_workers=phase4_workers,
        phase5_workers=phase5_workers,
        phase8_workers=phase8_workers,
        phase4_gene_chunk_size=phase4_chunk_size,
        phase5_gene_chunk_size=phase5_chunk_size,
        phase8_gene_chunk_size=phase8_chunk_size,
        numpy_threads_per_worker=numpy_threads,
        parallelize_modalities=bool(compute_raw.get("parallelize_modalities", False)),
        mmap_root=mmap_root,
        temp_root=temp_root,
        max_concurrent_runs=_resolve_positive_int(
            compute_raw.get("max_concurrent_runs"),
            2,
            "compute.max_concurrent_runs",
        ),
    )


def set_numpy_worker_env(n_threads: int) -> None:
    value = str(max(1, int(n_threads)))
    for env_name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[env_name] = value
