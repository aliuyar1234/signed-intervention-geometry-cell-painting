from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen
from zipfile import ZipFile

from siggeo.config import PathsConfig


DEFAULT_TIMEOUT_SECONDS = 120
RELEASE_ID = "siggeo-ssot-r1.1-patched-2026-03-21"


@dataclass(frozen=True)
class AssetSpec:
    name: str
    category: str
    target_relative_path: str
    url: str
    frozen_etag: str | None = None
    optional: bool = False
    fallback_url: str | None = None
    archive_member: str | None = None


@dataclass
class AssetResult:
    name: str
    category: str
    target_path: str
    url: str
    fallback_url: str | None
    status: str
    observed_etag: str | None
    frozen_etag: str | None
    sha256: str | None
    size_bytes: int | None
    downloaded_at_utc: str
    used_fallback_url: bool
    error: str | None = None


PROFILE_ASSETS: tuple[AssetSpec, ...] = (
    AssetSpec(
        name="orf_official",
        category="profile",
        target_relative_path="cpg0016/orf_official.parquet",
        url=(
            "https://cellpainting-gallery.s3.amazonaws.com/"
            "cpg0016-jump-assembled/source_all/workspace/profiles_assembled/ORF/v1.0a/"
            "profiles_wellpos_cc_var_mad_outlier_featselect_sphering_harmony.parquet"
        ),
        frozen_etag="064759b3a850dc351b357116b3e7b32d",
    ),
    AssetSpec(
        name="crispr_official",
        category="profile",
        target_relative_path="cpg0016/crispr_official.parquet",
        url=(
            "https://cellpainting-gallery.s3.amazonaws.com/"
            "cpg0016-jump-assembled/source_all/workspace/profiles_assembled/CRISPR/v1.0a/"
            "profiles_wellpos_cc_var_mad_outlier_featselect_sphering_harmony_PCA_corrected.parquet"
        ),
        frozen_etag="5903af59605b2037190ff64c1f87c530",
    ),
    AssetSpec(
        name="orf_interpretable",
        category="profile",
        target_relative_path="cpg0016/orf_interpretable.parquet",
        url=(
            "https://cellpainting-gallery.s3.amazonaws.com/"
            "cpg0016-jump-assembled/source_all/workspace/profiles_assembled/ORF/v1.0a/"
            "profiles_wellpos_cc_var_mad_outlier.parquet"
        ),
        frozen_etag="a2ca4063bbfcee09ac303cf48eb8bafd",
    ),
    AssetSpec(
        name="crispr_interpretable",
        category="profile",
        target_relative_path="cpg0016/crispr_interpretable.parquet",
        url=(
            "https://cellpainting-gallery.s3.amazonaws.com/"
            "cpg0016-jump-assembled/source_all/workspace/profiles_assembled/CRISPR/v1.0a/"
            "profiles_wellpos_cc_var_mad_outlier.parquet"
        ),
        frozen_etag="0009a142e5d132d007696ed8f71f2da8",
    ),
)


METADATA_ASSETS: tuple[AssetSpec, ...] = tuple(
    AssetSpec(
        name=filename.replace(".", "_").replace("/", "_"),
        category="metadata",
        target_relative_path=filename,
        url=f"https://raw.githubusercontent.com/jump-cellpainting/datasets/main/metadata/{filename}",
    )
    for filename in (
        "compound.csv.gz",
        "orf.csv.gz",
        "crispr.csv.gz",
        "perturbation_control.csv",
        "plate.csv.gz",
        "well.csv.gz",
        "compound_source.csv.gz",
        "microscope_config.csv",
        "microscope_filter.csv",
        "cellprofiler_version.csv",
        "db/setup.sql",
    )
)


ANNOTATION_ASSETS: tuple[AssetSpec, ...] = (
    AssetSpec(
        name="reactome",
        category="annotation",
        target_relative_path="reactome.gmt",
        url="https://reactome.org/download/current/ReactomePathways.gmt.zip",
        archive_member="ReactomePathways.gmt",
        optional=True,
    ),
    AssetSpec(
        name="corum",
        category="annotation",
        target_relative_path="corum.tsv",
        url="https://mips.helmholtz-muenchen.de/corum/download/allComplexes.txt.zip",
        fallback_url="https://mips.helmholtz-muenchen.de/corum/download/current/allComplexes.txt.zip",
        archive_member="allComplexes.txt",
        optional=True,
    ),
)


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _strip_etag(value: str | None) -> str | None:
    if value is None:
        return None
    return value.strip().strip('"')


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _extract_header(headers: Any, key: str) -> str | None:
    if hasattr(headers, "get"):
        return headers.get(key) or headers.get(key.lower())
    return None


def _probe_url(url: str, timeout_seconds: int) -> dict[str, str | None]:
    request = Request(url, method="HEAD")
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            return {
                "etag": _strip_etag(_extract_header(response.headers, "ETag")),
                "content_length": _extract_header(response.headers, "Content-Length"),
            }
    except Exception:
        return {"etag": None, "content_length": None}


def _download_to_temp(url: str, timeout_seconds: int, temp_path: Path) -> tuple[Path, str | None]:
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path.unlink(missing_ok=True)
    request = Request(url, method="GET")
    with urlopen(request, timeout=timeout_seconds) as response, temp_path.open("wb") as handle:
        shutil.copyfileobj(response, handle)
        etag = _strip_etag(_extract_header(response.headers, "ETag"))
    if temp_path.stat().st_size <= 0:
        raise ValueError(f"Downloaded empty payload from {url}")
    return temp_path, etag


def _materialize_download(temp_path: Path, target_path: Path, asset: AssetSpec) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if asset.archive_member is None:
        shutil.move(str(temp_path), str(target_path))
        return

    with ZipFile(temp_path) as archive:
        members = {member.filename: member for member in archive.infolist()}
        if asset.archive_member in members:
            member = members[asset.archive_member]
        else:
            matching = [member for member in archive.infolist() if member.filename.endswith(asset.archive_member)]
            if not matching:
                raise FileNotFoundError(f"Archive member not found: {asset.archive_member}")
            member = matching[0]
        with archive.open(member) as source, target_path.open("wb") as destination:
            shutil.copyfileobj(source, destination)
    if target_path.stat().st_size <= 0:
        raise ValueError(f"Extracted empty payload for {asset.name}")


def _target_base_dir(paths: PathsConfig, asset: AssetSpec) -> Path:
    base_paths = paths.paths
    if asset.category == "profile":
        return base_paths["raw_root"]
    if asset.category == "metadata":
        return base_paths["metadata_root"]
    if asset.category == "annotation":
        return base_paths["annotations_root"]
    raise KeyError(f"Unknown asset category: {asset.category}")


def _target_path(paths: PathsConfig, asset: AssetSpec) -> Path:
    return _target_base_dir(paths, asset) / Path(asset.target_relative_path)


def _result_from_existing(asset: AssetSpec, target_path: Path, probe: dict[str, str | None]) -> AssetResult:
    return AssetResult(
        name=asset.name,
        category=asset.category,
        target_path=str(target_path),
        url=asset.url,
        fallback_url=asset.fallback_url,
        status="present",
        observed_etag=probe["etag"],
        frozen_etag=asset.frozen_etag,
        sha256=_sha256(target_path),
        size_bytes=target_path.stat().st_size,
        downloaded_at_utc=_now_utc(),
        used_fallback_url=False,
    )


def _download_with_urls(asset: AssetSpec, urls: list[str], target_path: Path, timeout_seconds: int) -> AssetResult:
    last_error: Exception | None = None
    temp_path = target_path.with_name(f"{target_path.name}.partial")
    for index, url in enumerate(urls):
        try:
            temp_file, observed_etag = _download_to_temp(url, timeout_seconds, temp_path)
            _materialize_download(temp_file, target_path, asset)
            return AssetResult(
                name=asset.name,
                category=asset.category,
                target_path=str(target_path),
                url=url,
                fallback_url=asset.fallback_url,
                status="downloaded" if index == 0 else "downloaded_via_fallback",
                observed_etag=observed_etag,
                frozen_etag=asset.frozen_etag,
                sha256=_sha256(target_path),
                size_bytes=target_path.stat().st_size,
                downloaded_at_utc=_now_utc(),
                used_fallback_url=index > 0,
            )
        except Exception as exc:
            last_error = exc
            if target_path.exists():
                target_path.unlink(missing_ok=True)
            temp_path.unlink(missing_ok=True)
        finally:
            temp_path.unlink(missing_ok=True)
    raise RuntimeError(str(last_error) if last_error else f"Download failed for {asset.name}")


def fetch_asset(
    asset: AssetSpec,
    paths: PathsConfig,
    *,
    refresh: bool = False,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> AssetResult:
    target_path = _target_path(paths, asset)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    probe = _probe_url(asset.url, timeout_seconds=timeout_seconds)

    if target_path.exists() and not refresh:
        return _result_from_existing(asset, target_path, probe)

    urls = [asset.url]
    if asset.fallback_url:
        urls.append(asset.fallback_url)

    try:
        return _download_with_urls(asset, urls, target_path, timeout_seconds)
    except Exception as exc:
        if asset.optional:
            return AssetResult(
                name=asset.name,
                category=asset.category,
                target_path=str(target_path),
                url=asset.url,
                fallback_url=asset.fallback_url,
                status="missing_optional",
                observed_etag=None,
                frozen_etag=asset.frozen_etag,
                sha256=None,
                size_bytes=None,
                downloaded_at_utc=_now_utc(),
                used_fallback_url=False,
                error=str(exc),
            )
        raise


def all_asset_specs() -> tuple[AssetSpec, ...]:
    return PROFILE_ASSETS + METADATA_ASSETS + ANNOTATION_ASSETS


def download_required_assets(
    paths: PathsConfig,
    *,
    refresh: bool = False,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    results: list[AssetResult] = []
    for asset in all_asset_specs():
        results.append(fetch_asset(asset, paths, refresh=refresh, timeout_seconds=timeout_seconds))

    report = {
        "release_id": RELEASE_ID,
        "created_at_utc": _now_utc(),
        "refresh": refresh,
        "results": [asdict(result) for result in results],
    }
    report_path = paths.paths["raw_root"] / "download_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
