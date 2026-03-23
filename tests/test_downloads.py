from __future__ import annotations

import shutil
from pathlib import Path
from zipfile import ZipFile

from siggeo.config import load_paths_config, repo_root
from siggeo.io.downloads import AssetSpec, fetch_asset


def _write_paths_config(base_dir: Path) -> Path:
    path = base_dir / "paths.yaml"
    path.write_text(
        "\n".join(
            [
                "paths:",
                f"  raw_root: {base_dir.as_posix()}/raw",
                f"  interim_root: {base_dir.as_posix()}/interim",
                f"  processed_root: {base_dir.as_posix()}/processed",
                f"  results_root: {base_dir.as_posix()}/results",
                f"  metadata_root: {base_dir.as_posix()}/metadata",
                f"  cpg0016_root: {base_dir.as_posix()}/raw/cpg0016",
                f"  annotations_root: {base_dir.as_posix()}/annotations",
                f"  release_root: {base_dir.as_posix()}/results/releases",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def test_fetch_asset_from_file_url() -> None:
    base_dir = repo_root() / "results" / "test_downloads_file"
    shutil.rmtree(base_dir, ignore_errors=True)
    base_dir.mkdir(parents=True, exist_ok=True)
    try:
        source = base_dir / "source.txt"
        source.write_text("hello world\n", encoding="utf-8")

        paths = load_paths_config(_write_paths_config(base_dir))
        asset = AssetSpec(
            name="test_asset",
            category="annotation",
            target_relative_path="payload.txt",
            url=source.resolve().as_uri(),
        )
        result = fetch_asset(asset, paths, refresh=True, timeout_seconds=5)

        target = base_dir / "annotations" / "payload.txt"
        assert result.status == "downloaded"
        assert target.read_text(encoding="utf-8") == "hello world\n"
        assert result.size_bytes == target.stat().st_size
    finally:
        shutil.rmtree(base_dir, ignore_errors=True)


def test_fetch_zipped_asset_extracts_member() -> None:
    base_dir = repo_root() / "results" / "test_downloads_zip"
    shutil.rmtree(base_dir, ignore_errors=True)
    base_dir.mkdir(parents=True, exist_ok=True)
    try:
        zip_path = base_dir / "bundle.zip"
        with ZipFile(zip_path, "w") as archive:
            archive.writestr("nested/data.tsv", "a\tb\n1\t2\n")

        paths = load_paths_config(_write_paths_config(base_dir))
        asset = AssetSpec(
            name="zip_asset",
            category="annotation",
            target_relative_path="data.tsv",
            url=zip_path.resolve().as_uri(),
            archive_member="data.tsv",
        )

        result = fetch_asset(asset, paths, refresh=True, timeout_seconds=5)
        assert result.status == "downloaded"
        assert (base_dir / "annotations" / "data.tsv").read_text(encoding="utf-8") == "a\tb\n1\t2\n"
    finally:
        shutil.rmtree(base_dir, ignore_errors=True)
