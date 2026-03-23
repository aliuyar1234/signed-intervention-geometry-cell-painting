import shutil
from pathlib import Path

import polars as pl

from siggeo.config import repo_root
from siggeo.runs import (
    clear_run_control,
    initialize_run,
    load_run_status,
    mark_phase_paused,
    tracked_phase,
    update_run_control,
)


def test_initialize_run_creates_expected_files() -> None:
    temp_root = repo_root() / "results" / "test_scratch"
    temp_root.mkdir(parents=True, exist_ok=True)
    try:
        config_path = temp_root / "analysis_test.yaml"
        config_path.write_text(
            "\n".join(
                [
                    "project_name: signed_intervention_geometry",
                    "run_name: unit_test_run",
                    "pipeline: smoke_official",
                    "mode: smoke",
                    "outputs:",
                    f"  result_dir: {temp_root.as_posix()}/results",
                    "seed: 123",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        context = initialize_run(config_path)

        assert context.run_dir == temp_root / "results" / "unit_test_run"
        assert (context.run_dir / "figures").is_dir()
        assert (context.run_dir / "tables").is_dir()
        assert (context.run_dir / "configs" / "analysis_test.yaml").exists()
        assert (context.run_dir / "metrics.json").exists()
        assert (context.run_dir / "provenance.json").exists()
        assert (context.run_dir / "run_status.json").exists()
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def test_tracked_phase_updates_checkpoint_status() -> None:
    temp_root = repo_root() / "results" / "test_scratch_status"
    temp_root.mkdir(parents=True, exist_ok=True)
    try:
        config_path = temp_root / "analysis_test.yaml"
        config_path.write_text(
            "\n".join(
                [
                    "project_name: signed_intervention_geometry",
                    "run_name: unit_test_run_status",
                    "pipeline: smoke_official",
                    "mode: smoke",
                    "outputs:",
                    f"  result_dir: {temp_root.as_posix()}/results",
                    "seed: 123",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        with tracked_phase(config_path, "02_make_overlap_gene_table") as context:
            output_path = context.run_dir / "tables" / "overlap_gene_table.parquet"
            pl.DataFrame(
                {
                    "gene_id": ["101"],
                    "display_symbol": ["GENEA"],
                    "canonical_symbol": ["GENEA"],
                    "keep_for_analysis": [True],
                    "low_replicate_support": [False],
                }
            ).write_parquet(output_path)

        status = load_run_status(config_path)
        first_phase = status["phases"][0]
        assert status["progress"]["completed_phases"] == 1
        assert first_phase["phase_id"] == "02_make_overlap_gene_table"
        assert first_phase["status"] == "completed"
        assert first_phase["checkpoint_reached"] is True
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def test_paused_phase_remains_resumable() -> None:
    temp_root = repo_root() / "results" / "test_scratch_paused"
    temp_root.mkdir(parents=True, exist_ok=True)
    try:
        config_path = temp_root / "analysis_test.yaml"
        config_path.write_text(
            "\n".join(
                [
                    "project_name: signed_intervention_geometry",
                    "run_name: unit_test_run_paused",
                    "pipeline: smoke_official",
                    "mode: smoke",
                    "outputs:",
                    f"  result_dir: {temp_root.as_posix()}/results",
                    "seed: 123",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        context = initialize_run(config_path)
        mark_phase_paused(
            context.run_dir,
            context.config,
            "02_make_overlap_gene_table",
            "Paused manually during testing.",
        )

        status = load_run_status(config_path)
        first_phase = status["phases"][0]
        assert status["status"] == "paused"
        assert status["current_phase_id"] == "02_make_overlap_gene_table"
        assert status["next_phase_id"] == "02_make_overlap_gene_table"
        assert first_phase["status"] == "paused"
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def test_run_control_metadata_round_trips() -> None:
    temp_root = repo_root() / "results" / "test_scratch_control"
    temp_root.mkdir(parents=True, exist_ok=True)
    try:
        config_path = temp_root / "analysis_test.yaml"
        config_path.write_text(
            "\n".join(
                [
                    "project_name: signed_intervention_geometry",
                    "run_name: unit_test_run_control",
                    "pipeline: smoke_official",
                    "mode: smoke",
                    "outputs:",
                    f"  result_dir: {temp_root.as_posix()}/results",
                    "seed: 123",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        context = initialize_run(config_path)
        update_run_control(
            context.run_dir,
            context.config,
            {"mode": "live_suspend", "pids": [123, 456]},
        )

        status = load_run_status(config_path)
        assert status["control"]["mode"] == "live_suspend"
        assert status["control"]["pids"] == [123, 456]

        cleared = clear_run_control(context.run_dir, context.config)
        assert "control" not in cleared
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def test_failed_phase_with_output_checkpoint_stays_failed() -> None:
    temp_root = repo_root() / "results" / "test_scratch_failed_output"
    temp_root.mkdir(parents=True, exist_ok=True)
    try:
        config_path = temp_root / "analysis_test.yaml"
        config_path.write_text(
            "\n".join(
                [
                    "project_name: signed_intervention_geometry",
                    "run_name: unit_test_failed_output",
                    "pipeline: smoke_official",
                    "mode: smoke",
                    "outputs:",
                    f"  result_dir: {temp_root.as_posix()}/results",
                    "seed: 123",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        try:
            with tracked_phase(config_path, "02_make_overlap_gene_table") as context:
                output_path = context.run_dir / "tables" / "overlap_gene_table.parquet"
                output_path.write_bytes(b"checkpoint")
                raise RuntimeError("boom")
        except RuntimeError:
            pass

        status = load_run_status(config_path)
        first_phase = status["phases"][0]
        assert status["status"] == "failed"
        assert first_phase["status"] == "failed"
        assert first_phase["checkpoint_reached"] is True
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def test_completed_phase_with_legacy_schema_output_fails_validation() -> None:
    temp_root = repo_root() / "results" / "test_scratch_schema_validation"
    temp_root.mkdir(parents=True, exist_ok=True)
    try:
        config_path = temp_root / "analysis_test.yaml"
        config_path.write_text(
            "\n".join(
                [
                    "project_name: signed_intervention_geometry",
                    "run_name: unit_test_schema_validation",
                    "pipeline: smoke_official",
                    "mode: smoke",
                    "outputs:",
                    f"  result_dir: {temp_root.as_posix()}/results",
                    "seed: 123",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        context = initialize_run(config_path)
        legacy_path = context.run_dir / "tables" / "pipeline_agreement.csv"
        legacy_path.parent.mkdir(parents=True, exist_ok=True)
        legacy_path.write_text("comparison_name,overall_agreement\nbaseline,1.0\n", encoding="utf-8")
        (context.run_dir / "tables" / "confound_summary.csv").write_text(
            "pipeline,confound_name,test_name,statistic_name,statistic_value,p_value,supports_positive_story,notes\n"
            "official,artifact_sensitivity,agreement,agreement_rate,0.8,,true,ok\n",
            encoding="utf-8",
        )
        (context.run_dir / "tables" / "regime_prevalence.csv").write_text(
            "pipeline,regime_label,n_genes,fraction_of_analyzed,n_regime_members\nofficial,aligned,1,1.0,1\n",
            encoding="utf-8",
        )
        (context.run_dir / "tables" / "regime_prevalence_comparison.csv").write_text(
            "comparison_name,regime_label,base_n_genes,base_fraction_of_analyzed,base_n_regime_members,other_n_genes,other_fraction_of_analyzed,other_n_regime_members,fraction_delta\n"
            "baseline,aligned,1,1.0,1,1,1.0,1,0.0\n",
            encoding="utf-8",
        )
        (context.run_dir / "tables" / "ablation_summary.csv").write_text(
            "run_name,pipeline,plate_centering,artifact_sensitivity,n_confident_noninactive,agreement_vs_official,decision_note\n"
            "unit_test_schema_validation,smoke_official,true,false,1,0.8,ok\n",
            encoding="utf-8",
        )

        status = load_run_status(config_path)
        phase_payload = next(phase for phase in status["phases"] if phase["phase_id"] == "08_run_confounds_and_ablations")
        assert phase_payload["status"] == "failed"
        assert phase_payload["outputs_validated"] is False
        assert "overall_agreement_primary_only" in " ".join(phase_payload["validation_errors"])
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)
