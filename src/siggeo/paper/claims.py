from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl


OFFICIAL_RUN_NAME = "offv1__featint__platectrl__thrA2_thrR2__z1p96__beta1p0"
INTERPRETABLE_RUN_NAME = "intv1__featint__platectrl__thrA2_thrR2__z1p96__beta1p0"
NO_CENTER_RUN_NAME = "offv1__featint__no_platectrl__thrA2_thrR2__z1p96__beta1p0"
ARTIFACT_RUN_NAME = "offv1__featint__platectrl__artifact_sens__thrA2_thrR2__z1p96__beta1p0"

MIN_ACTIVE_BOTH_FOR_C1 = 25
MIN_CONFIDENT_NONINACTIVE_FOR_C1 = 10
MIN_SIGNIFICANT_TERMS_FOR_C3 = 2


def _read_csv_if_exists(path: Path) -> pl.DataFrame | None:
    if not path.exists():
        return None
    return pl.read_csv(path)


def _read_parquet_if_exists(path: Path) -> pl.DataFrame | None:
    if not path.exists():
        return None
    return pl.read_parquet(path)


def _has_figure_with_source(run_dir: Path, stem: str) -> bool:
    return (run_dir / "figures" / f"{stem}.png").exists() and (run_dir / "source_data" / f"{stem}.csv").exists()


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, float):
        return value
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _geometry_summary(run_dir: Path) -> dict[str, Any]:
    geometry = _read_parquet_if_exists(run_dir / "tables" / "same_gene_signed_geometry.parquet")
    if geometry is None or geometry.height == 0:
        return {
            "available": False,
            "analyzed": 0,
            "active_both": 0,
            "confident_noninactive": 0,
            "n_primary_regimes": 0,
        }
    confident = geometry.filter((pl.col("regime_label") != "ambiguous") & (pl.col("regime_label") != "inactive"))
    primary_regimes = confident["regime_label"].n_unique() if confident.height else 0
    return {
        "available": True,
        "analyzed": geometry.height,
        "active_both": int(geometry.filter(pl.col("orf_active") & pl.col("crispr_active")).height),
        "confident_noninactive": int(confident.height),
        "n_primary_regimes": int(primary_regimes),
    }


def _confound_summary(run_dir: Path) -> dict[str, Any]:
    confounds = _read_csv_if_exists(run_dir / "tables" / "confound_summary.csv")
    if confounds is None or confounds.height == 0:
        return {
            "available": False,
            "all_positive": False,
            "artifact_agreement": None,
            "plate_supported": False,
            "failed_confounds": [],
        }
    failed = sorted(set(confounds.filter(~pl.col("supports_positive_story"))["confound_name"].to_list()))
    artifact_row = confounds.filter(pl.col("confound_name") == "artifact_sensitivity")
    plate_row = confounds.filter(pl.col("confound_name") == "plate_holdout")
    artifact_agreement = (
        _safe_float(artifact_row["statistic_value"][0]) if artifact_row.height and "statistic_value" in artifact_row.columns else None
    )
    plate_supported = bool(plate_row["supports_positive_story"][0]) if plate_row.height else False
    return {
        "available": True,
        "all_positive": len(failed) == 0,
        "artifact_agreement": artifact_agreement,
        "plate_supported": plate_supported,
        "failed_confounds": failed,
    }


def _agreement_summary(run_dir: Path) -> dict[str, Any]:
    frame = _read_csv_if_exists(run_dir / "tables" / "pipeline_agreement.csv")
    if frame is None or frame.height == 0:
        return {"available": False, "overall_agreement": None, "overall_agreement_primary_only": None}
    if "has_real_comparison" in frame.columns:
        frame = frame.filter(pl.col("has_real_comparison"))
    if frame.height == 0 or "overall_agreement_primary_only" not in frame.columns:
        return {"available": False, "overall_agreement": None, "overall_agreement_primary_only": None}
    return {
        "available": True,
        "overall_agreement": _safe_float(frame["overall_agreement"][0]) if "overall_agreement" in frame.columns else None,
        "overall_agreement_primary_only": _safe_float(frame["overall_agreement_primary_only"][0]),
    }


def _retrieval_summary(run_dir: Path, primary_lambda: float = 0.5) -> dict[str, Any]:
    summary = _read_csv_if_exists(run_dir / "tables" / "retrieval_summary.csv")
    if summary is None or summary.height == 0:
        return {"available": False, "primary": None, "signed_cosine": None, "abs_cosine": None}

    def mean_mrr(score_name: str) -> float | None:
        rows = summary.filter(pl.col("score_name") == score_name)
        return float(rows["mrr"].mean()) if rows.height else None

    return {
        "available": True,
        "primary": mean_mrr(f"regime_score_lambda_{primary_lambda:g}"),
        "signed_cosine": mean_mrr("signed_cosine"),
        "abs_cosine": mean_mrr("abs_cosine"),
    }


def _enrichment_summary(run_dir: Path) -> dict[str, Any]:
    geometry = _read_parquet_if_exists(run_dir / "tables" / "same_gene_signed_geometry.parquet")

    def significant_count(name: str) -> int:
        frame = _read_csv_if_exists(run_dir / "tables" / name)
        if frame is None or frame.height == 0 or "q_value" not in frame.columns:
            return 0
        return int(frame.filter(pl.col("q_value") < 0.05).height)

    canonical_coverage = None
    duplicate_canonical_symbols = 0
    if geometry is not None and geometry.height and "canonical_symbol" in geometry.columns:
        primary = geometry.filter(pl.col("regime_label").is_in(["aligned", "inverse", "orthogonal", "asymmetric_ORF", "asymmetric_CRISPR"]))
        if primary.height:
            non_null = primary.filter(pl.col("canonical_symbol").is_not_null())
            canonical_coverage = float(non_null.height / primary.height)
            duplicate_canonical_symbols = int(
                non_null.group_by("canonical_symbol").len().filter(pl.col("len") > 1).height
            )
    return {
        "reactome_sig_terms": significant_count("enrichment_reactome.csv"),
        "corum_sig_terms": significant_count("enrichment_corum.csv"),
        "canonical_coverage": canonical_coverage,
        "duplicate_canonical_symbols": duplicate_canonical_symbols,
    }


def _claim_row(
    claim_id: str,
    claim_text: str,
    *,
    support_type: str,
    required_artifacts: list[str],
    evidence_detail: str,
) -> dict[str, object]:
    return {
        "claim_id": claim_id,
        "claim_text": claim_text,
        "support_type": support_type,
        "supported": support_type != "unsupported",
        "required_artifacts": "|".join(required_artifacts),
        "evidence_detail": evidence_detail,
    }


def build_claim_evidence_matrix(run_dir: Path) -> pl.DataFrame:
    if run_dir.name.startswith("smoke__"):
        rows = [
            _claim_row(
                "C1",
                "Same-gene ORF/CRISPR pairs occupy a structured regime space within cpg0016.",
                support_type="unsupported",
                required_artifacts=["tables/same_gene_signed_geometry.parquet", "tables/regime_prevalence.csv"],
                evidence_detail="Smoke runs are deterministic checks only and do not bear paper claims.",
            ),
            _claim_row(
                "C2",
                "The regime structure is partially stable or explicitly fragile under robustness checks.",
                support_type="unsupported",
                required_artifacts=["tables/confound_summary.csv", "tables/pipeline_agreement.csv"],
                evidence_detail="Smoke runs do not determine robustness or fragility claims.",
            ),
            _claim_row(
                "C3",
                "The regime map carries some biological structure under curated resources.",
                support_type="unsupported",
                required_artifacts=["tables/enrichment_reactome.csv", "tables/enrichment_corum.csv"],
                evidence_detail="Smoke runs are not eligible for biological claims.",
            ),
            _claim_row(
                "C4",
                "A regime-aware score is modestly informative for retrieval support.",
                support_type="unsupported",
                required_artifacts=["tables/retrieval_summary.csv"],
                evidence_detail="Smoke runs are not eligible for retrieval claims.",
            ),
        ]
        return pl.DataFrame(rows)

    related_root = run_dir.parent
    official_run_dir = related_root / OFFICIAL_RUN_NAME
    interpretable_run_dir = related_root / INTERPRETABLE_RUN_NAME
    no_center_run_dir = related_root / NO_CENTER_RUN_NAME
    base_run_dir = official_run_dir if official_run_dir.exists() else run_dir

    geometry = _geometry_summary(run_dir)
    confounds = _confound_summary(base_run_dir)
    interpretable_agreement = _agreement_summary(interpretable_run_dir)
    no_center_agreement = _agreement_summary(no_center_run_dir)
    retrieval = _retrieval_summary(run_dir)
    interpretable_retrieval = _retrieval_summary(interpretable_run_dir)
    enrichment = _enrichment_summary(run_dir)

    c1_supported = (
        geometry["available"]
        and (run_dir / "tables" / "regime_prevalence.csv").exists()
        and _has_figure_with_source(run_dir, "figure2_regime_map")
        and geometry["active_both"] >= MIN_ACTIVE_BOTH_FOR_C1
        and geometry["confident_noninactive"] >= MIN_CONFIDENT_NONINACTIVE_FOR_C1
        and geometry["n_primary_regimes"] >= 2
    )

    c2_ready = confounds["available"] and interpretable_agreement["available"] and no_center_agreement["available"]
    c2_positive = (
        c2_ready
        and confounds["all_positive"]
        and confounds["artifact_agreement"] is not None
        and confounds["artifact_agreement"] >= 0.70
        and interpretable_agreement["overall_agreement_primary_only"] is not None
        and interpretable_agreement["overall_agreement_primary_only"] >= 0.55
        and no_center_agreement["overall_agreement_primary_only"] is not None
        and no_center_agreement["overall_agreement_primary_only"] >= 0.35
    )
    c2_fragility = (
        c2_ready
        and not c2_positive
        and (
            not confounds["all_positive"]
            or (confounds["artifact_agreement"] is not None and confounds["artifact_agreement"] < 0.70)
            or (
                interpretable_agreement["overall_agreement_primary_only"] is not None
                and interpretable_agreement["overall_agreement_primary_only"] < 0.55
            )
            or (
                no_center_agreement["overall_agreement_primary_only"] is not None
                and no_center_agreement["overall_agreement_primary_only"] < 0.35
            )
        )
    )

    c3_supported = (
        _has_figure_with_source(run_dir, "figure5_enrichment")
        and confounds["available"]
        and (enrichment["reactome_sig_terms"] + enrichment["corum_sig_terms"] >= MIN_SIGNIFICANT_TERMS_FOR_C3)
        and enrichment["canonical_coverage"] is not None
    )

    primary_mrr = retrieval["primary"]
    signed_mrr = retrieval["signed_cosine"]
    abs_mrr = retrieval["abs_cosine"]
    interpretable_primary = interpretable_retrieval["primary"]
    interpretable_signed = interpretable_retrieval["signed_cosine"]
    interpretable_abs = interpretable_retrieval["abs_cosine"]
    c4_supported = (
        _has_figure_with_source(run_dir, "figure4_retrieval")
        and retrieval["available"]
        and interpretable_retrieval["available"]
        and primary_mrr is not None
        and signed_mrr is not None
        and abs_mrr is not None
        and interpretable_primary is not None
        and interpretable_signed is not None
        and interpretable_abs is not None
        and primary_mrr >= max(signed_mrr, abs_mrr)
        and interpretable_primary >= max(interpretable_signed, interpretable_abs)
    )

    rows = [
        _claim_row(
            "C1",
            "Same-gene ORF/CRISPR pairs occupy a structured regime space within cpg0016.",
            support_type="positive" if c1_supported else "unsupported",
            required_artifacts=[
                "tables/same_gene_signed_geometry.parquet",
                "tables/regime_prevalence.csv",
                "figures/figure2_regime_map.png",
                "source_data/figure2_regime_map.csv",
            ],
            evidence_detail=(
                f"analyzed={geometry['analyzed']}, active_both={geometry['active_both']}, "
                f"confident_noninactive={geometry['confident_noninactive']}, "
                f"primary_regimes={geometry['n_primary_regimes']}, "
                f"thresholds=(active_both>={MIN_ACTIVE_BOTH_FOR_C1}, confident_noninactive>={MIN_CONFIDENT_NONINACTIVE_FOR_C1})"
            ),
        ),
        _claim_row(
            "C2",
            "The regime structure is partially stable or explicitly fragile under robustness checks.",
            support_type="positive" if c2_positive else "fragility" if c2_fragility else "unsupported",
            required_artifacts=[
                "tables/confound_summary.csv",
                "tables/pipeline_agreement.csv",
                f"../{INTERPRETABLE_RUN_NAME}/tables/pipeline_agreement.csv",
                f"../{NO_CENTER_RUN_NAME}/tables/pipeline_agreement.csv",
            ],
            evidence_detail=(
                f"interpretable_primary_only_agreement={interpretable_agreement['overall_agreement_primary_only']}, "
                f"no_center_primary_only_agreement={no_center_agreement['overall_agreement_primary_only']}, "
                f"artifact_agreement={confounds['artifact_agreement']}, "
                f"failed_confounds={','.join(confounds['failed_confounds']) or 'none'}"
            ),
        ),
        _claim_row(
            "C3",
            "The regime map carries some biological structure under curated resources.",
            support_type="positive" if c3_supported else "unsupported",
            required_artifacts=[
                "tables/enrichment_reactome.csv",
                "tables/enrichment_corum.csv",
                "figures/figure5_enrichment.png",
                "source_data/figure5_enrichment.csv",
                "tables/confound_summary.csv",
            ],
            evidence_detail=(
                f"reactome_sig_terms={enrichment['reactome_sig_terms']}, "
                f"corum_sig_terms={enrichment['corum_sig_terms']}, "
                f"canonical_coverage={enrichment['canonical_coverage']}, "
                f"duplicate_canonical_symbols={enrichment['duplicate_canonical_symbols']}"
            ),
        ),
        _claim_row(
            "C4",
            "A regime-aware score is modestly informative for retrieval support.",
            support_type="positive" if c4_supported else "unsupported",
            required_artifacts=[
                "tables/retrieval_summary.csv",
                "figures/figure4_retrieval.png",
                "source_data/figure4_retrieval.csv",
                f"../{INTERPRETABLE_RUN_NAME}/tables/retrieval_summary.csv",
            ],
            evidence_detail=(
                f"official_primary_mrr={primary_mrr}, official_signed_mrr={signed_mrr}, "
                f"official_abs_mrr={abs_mrr}, interpretable_primary_mrr={interpretable_primary}, "
                f"interpretable_signed_mrr={interpretable_signed}, interpretable_abs_mrr={interpretable_abs}"
            ),
        ),
    ]
    return pl.DataFrame(rows)


def summarize_claim_matrix(claim_matrix: pl.DataFrame) -> dict[str, Any]:
    support_lookup = {
        row["claim_id"]: {
            "supported": bool(row["supported"]),
            "support_type": str(row["support_type"]),
            "claim_text": str(row["claim_text"]),
        }
        for row in claim_matrix.to_dicts()
    }
    c1 = support_lookup.get("C1", {"supported": False, "support_type": "unsupported"})
    c2 = support_lookup.get("C2", {"supported": False, "support_type": "unsupported"})
    c3 = support_lookup.get("C3", {"supported": False, "support_type": "unsupported"})
    c4 = support_lookup.get("C4", {"supported": False, "support_type": "unsupported"})
    return {
        "claims": support_lookup,
        "supported_claim_ids": sorted(
            claim_id for claim_id, payload in support_lookup.items() if bool(payload.get("supported", False))
        ),
        "positive_ready": bool(c1["supported"] and c2["support_type"] == "positive" and c3["supported"]),
        "fragility_ready": bool(c1["supported"] and c2["support_type"] == "fragility"),
        "retrieval_supported": bool(c4["supported"]),
    }
