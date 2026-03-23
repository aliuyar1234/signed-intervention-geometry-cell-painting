from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SRC_DIR = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import polars as pl

from siggeo.compute import load_compute_config
from siggeo.config import load_analysis_config
from siggeo.evaluation.confounds import (
    ConfoundConfig,
    compare_geometry_tables,
    compare_regime_prevalence,
    run_confounds,
    write_confound_outputs,
)
from siggeo.io.atomic import write_csv_atomic
from siggeo.runs import tracked_phase, update_metrics


def _read_csv_if_exists(path: Path) -> pl.DataFrame | None:
    if not path.exists():
        return None
    return pl.read_csv(path)


def _empty_pipeline_agreement(comparison_name: str) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "comparison_name": [comparison_name],
            "has_real_comparison": [False],
            "n_genes_both_confident": [0],
            "n_genes_both_primary_confident": [0],
            "overall_agreement": [0.0],
            "overall_agreement_primary_only": [0.0],
            "agreement_collapsed_asymmetric": [0.0],
            "aligned_agreement": [0.0],
            "inverse_agreement": [0.0],
            "orthogonal_agreement": [0.0],
            "asymmetric_orf_agreement": [0.0],
            "asymmetric_crispr_agreement": [0.0],
            "notes": ["Comparison run not available yet."],
        }
    )


def _baseline_regime_prevalence_comparison(comparison_name: str, prevalence: pl.DataFrame) -> pl.DataFrame:
    return compare_regime_prevalence(prevalence, prevalence, comparison_name)


def _comparison_name(config_pipeline: str, mode: str, plate_centering: bool) -> str:
    if mode == "artifact_sensitivity":
        return "official_vs_artifact_sensitivity"
    if not plate_centering:
        return "official_vs_no_final_centering"
    if config_pipeline == "interpretable":
        return "official_vs_interpretable"
    return "official_primary_baseline"


def _comparison_targets(result_dir: Path) -> list[tuple[str, Path]]:
    return [
        ("official_vs_interpretable", result_dir / "intv1__featint__platectrl__thrA2_thrR2__z1p96__beta1p0"),
        ("official_vs_no_final_centering", result_dir / "offv1__featint__no_platectrl__thrA2_thrR2__z1p96__beta1p0"),
        ("official_vs_artifact_sensitivity", result_dir / "offv1__featint__platectrl__artifact_sens__thrA2_thrR2__z1p96__beta1p0"),
    ]


def _primary_retrieval_mrr(run_dir: Path, primary_lambda: float) -> float | None:
    retrieval_summary = _read_csv_if_exists(run_dir / "tables" / "retrieval_summary.csv")
    if retrieval_summary is None or retrieval_summary.height == 0:
        return None
    primary_score = f"regime_score_lambda_{primary_lambda:g}"
    primary_rows = retrieval_summary.filter(pl.col("score_name") == primary_score)
    if primary_rows.height == 0:
        return None
    return float(primary_rows["mrr"].mean())


def _significant_term_count(path: Path) -> int:
    frame = _read_csv_if_exists(path)
    if frame is None or frame.height == 0 or "q_value" not in frame.columns:
        return 0
    return int(frame.filter(pl.col("q_value") < 0.05).height)


def _decision_note(
    confounds: pl.DataFrame,
    *,
    agreement_vs_official: float | None,
    is_official_baseline: bool,
    mode: str,
) -> str:
    failures = confounds.filter(~pl.col("supports_positive_story"))["confound_name"].to_list()
    if failures:
        return "Threatened by confound checks: " + ", ".join(sorted(set(str(item) for item in failures)))
    if is_official_baseline and agreement_vs_official is None:
        return "Official run completed, but cross-run comparison rows are not available yet."
    if not is_official_baseline and agreement_vs_official is not None and agreement_vs_official < 0.35:
        return f"{mode} materially diverges from the official baseline."
    if not is_official_baseline and agreement_vs_official is not None and agreement_vs_official < 0.55:
        return f"{mode} shows partial fragility relative to the official baseline."
    if is_official_baseline:
        return "Official run summarized against the locked comparison runs."
    return "Locked confound and ablation outputs completed without immediate threat flags."


def main() -> None:
    parser = argparse.ArgumentParser(description="Run confound checks and locked ablations.")
    parser.add_argument("--config", required=True, type=Path, help="Path to an analysis config YAML.")
    parser.add_argument("--mode", default="official_confounds", help="Execution mode, e.g. official_confounds.")
    args = parser.parse_args()

    with tracked_phase(args.config, "08_run_confounds_and_ablations") as context:
        config = load_analysis_config(args.config)
        compute_cfg = load_compute_config(config.raw, run_dir=context.run_dir)
        run_dir = context.run_dir
        plate_centering = bool(config.raw["preprocess"].get("final_plate_negcon_centering", True))
        progress_path = run_dir / "logs" / "08_run_confounds_progress.json"
        if progress_path.exists():
            progress_path.unlink()
        geometry = pl.read_parquet(run_dir / "tables" / "same_gene_signed_geometry.parquet")
        gene_stats = pl.read_parquet(run_dir / "tables" / "gene_modality_stats.parquet")
        centered_orf = pl.read_parquet(run_dir / "tables" / f"centered_wells_{config.pipeline}_orf.parquet")
        centered_crispr = pl.read_parquet(run_dir / "tables" / f"centered_wells_{config.pipeline}_crispr.parquet")
        feature_columns = (run_dir / "tables" / "feature_intersection.txt").read_text(encoding="utf-8").splitlines()

        confounds, regime_prevalence = run_confounds(
            geometry,
            gene_stats,
            centered_orf,
            centered_crispr,
            feature_columns,
            config=ConfoundConfig(
                strength_threshold=float(config.raw["activity"]["strength_z_threshold"]),
                rep_threshold=float(config.raw["activity"]["rep_z_threshold"]),
                sign_z_threshold=float(config.raw["geometry"]["sign_z_threshold"]),
                asymmetry_threshold=float(config.raw["geometry"]["asymmetry_threshold"]),
                null_std_floor=float(config.raw["geometry"]["null_std_floor"]),
                random_seed=config.seed,
                n_bootstrap=int(config.raw["bootstrap"]["n_bootstrap"]),
                consensus_threshold=float(config.raw["bootstrap"]["consensus_threshold"]),
            ),
            compute_config=compute_cfg,
            run_dir=run_dir,
            progress_path=progress_path,
        )
        write_confound_outputs(
            confounds,
            regime_prevalence,
            run_dir / "tables" / "confound_summary.csv",
            run_dir / "tables" / "regime_prevalence.csv",
        )

        official_run_dir = config.result_dir / "offv1__featint__platectrl__thrA2_thrR2__z1p96__beta1p0"
        comparison_name = _comparison_name(config.pipeline, args.mode, plate_centering)
        if run_dir == official_run_dir:
            pipeline_frames: list[pl.DataFrame] = []
            prevalence_frames: list[pl.DataFrame] = []
            comparison_scores: list[float] = []
            for target_name, target_run_dir in _comparison_targets(config.result_dir):
                target_geometry_path = target_run_dir / "tables" / "same_gene_signed_geometry.parquet"
                if target_geometry_path.exists():
                    target_geometry = pl.read_parquet(target_geometry_path)
                    agreement = compare_geometry_tables(geometry, target_geometry, target_name)
                    pipeline_frames.append(agreement)
                    if bool(agreement["has_real_comparison"][0]):
                        comparison_scores.append(float(agreement["overall_agreement_primary_only"][0]))
                    prevalence_path = target_run_dir / "tables" / "regime_prevalence.csv"
                    if prevalence_path.exists():
                        prevalence_frames.append(
                            compare_regime_prevalence(
                                regime_prevalence,
                                pl.read_csv(prevalence_path),
                                target_name,
                            )
                        )
                else:
                    pipeline_frames.append(_empty_pipeline_agreement(target_name))
            pipeline_agreement = pl.concat(pipeline_frames, how="diagonal_relaxed") if pipeline_frames else pl.DataFrame()
            regime_prevalence_comparison = (
                pl.concat(prevalence_frames, how="diagonal_relaxed")
                if prevalence_frames
                else pl.DataFrame(
                    schema={
                        "comparison_name": pl.String,
                        "regime_label": pl.String,
                        "base_n_genes": pl.Int64,
                        "base_fraction_of_analyzed": pl.Float64,
                        "base_n_regime_members": pl.Int64,
                        "other_n_genes": pl.Int64,
                        "other_fraction_of_analyzed": pl.Float64,
                        "other_n_regime_members": pl.Int64,
                        "fraction_delta": pl.Float64,
                    }
                )
            )
            agreement_vs_official = (
                float(sum(comparison_scores) / len(comparison_scores)) if comparison_scores else None
            )
        elif (official_run_dir / "tables" / "same_gene_signed_geometry.parquet").exists():
            official_geometry = pl.read_parquet(official_run_dir / "tables" / "same_gene_signed_geometry.parquet")
            pipeline_agreement = compare_geometry_tables(official_geometry, geometry, comparison_name)
            agreement_vs_official = float(
                pipeline_agreement["overall_agreement_primary_only"][0]
                if "overall_agreement_primary_only" in pipeline_agreement.columns
                else pipeline_agreement["overall_agreement"][0]
            )
            official_prevalence_path = official_run_dir / "tables" / "regime_prevalence.csv"
            if official_prevalence_path.exists():
                official_prevalence = pl.read_csv(official_prevalence_path)
                regime_prevalence_comparison = compare_regime_prevalence(
                    official_prevalence, regime_prevalence, comparison_name
                )
            else:
                regime_prevalence_comparison = _baseline_regime_prevalence_comparison(comparison_name, regime_prevalence)
        else:
            pipeline_agreement = _empty_pipeline_agreement(comparison_name)
            agreement_vs_official = None
            regime_prevalence_comparison = _baseline_regime_prevalence_comparison(comparison_name, regime_prevalence)
        write_csv_atomic(pipeline_agreement, run_dir / "tables" / "pipeline_agreement.csv")
        write_csv_atomic(regime_prevalence_comparison, run_dir / "tables" / "regime_prevalence_comparison.csv")

        n_confident_noninactive = int(
            geometry.filter((pl.col("regime_label") != "ambiguous") & (pl.col("regime_label") != "inactive")).height
        )
        mrr_regime_score = _primary_retrieval_mrr(run_dir, float(config.raw["retrieval"]["primary_lambda"]))
        reactome_sig_terms = _significant_term_count(run_dir / "tables" / "enrichment_reactome.csv")
        corum_sig_terms = _significant_term_count(run_dir / "tables" / "enrichment_corum.csv")
        ablation_summary = pl.DataFrame(
            {
                "run_name": [config.run_name],
                "pipeline": [config.pipeline],
                "plate_centering": [plate_centering],
                "artifact_sensitivity": [args.mode == "artifact_sensitivity"],
                "n_confident_noninactive": [n_confident_noninactive],
                "agreement_vs_official": [agreement_vs_official],
                "mrr_regime_score": [mrr_regime_score],
                "reactome_sig_terms": [reactome_sig_terms],
                "corum_sig_terms": [corum_sig_terms],
                "decision_note": [
                    _decision_note(
                        confounds,
                        agreement_vs_official=agreement_vs_official,
                        is_official_baseline=run_dir == official_run_dir,
                        mode=args.mode,
                    )
                ],
            }
        )
        write_csv_atomic(ablation_summary, run_dir / "tables" / "ablation_summary.csv")
        update_metrics(
            run_dir,
            {
                "confound_support": {
                    str(row["confound_name"]): bool(row["supports_positive_story"]) for row in confounds.to_dicts()
                },
                "agreement_vs_official": agreement_vs_official,
                "mrr_regime_score": mrr_regime_score,
                "reactome_sig_terms": reactome_sig_terms,
                "corum_sig_terms": corum_sig_terms,
            },
        )
        print(
            json.dumps(
                {
                    "confound_rows": confounds.height,
                    "regime_prevalence_rows": regime_prevalence.height,
                    "pipeline_agreement_rows": pipeline_agreement.height,
                    "regime_prevalence_comparison_rows": regime_prevalence_comparison.height,
                    "ablation_summary_rows": ablation_summary.height,
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
