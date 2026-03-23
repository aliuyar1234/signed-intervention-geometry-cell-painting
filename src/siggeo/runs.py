from __future__ import annotations

import json
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import polars as pl

from siggeo.config import AnalysisConfig, load_analysis_config, repo_root
from siggeo.io.atomic import write_text_atomic
from siggeo.provenance import collect_provenance, write_json


RUN_SUBDIRS = ("tables", "figures", "logs", "source_data", "paper", "configs")


@dataclass(frozen=True)
class RunContext:
    config: AnalysisConfig
    run_dir: Path


@dataclass(frozen=True)
class RunPhaseSpec:
    phase_id: str
    label: str
    script_name: str
    output_templates: tuple[str, ...]


RUN_PHASES: tuple[RunPhaseSpec, ...] = (
    RunPhaseSpec(
        phase_id="02_make_overlap_gene_table",
        label="Overlap Gene Table",
        script_name="02_make_overlap_gene_table.py",
        output_templates=("tables/overlap_gene_table.parquet",),
    ),
    RunPhaseSpec(
        phase_id="03_build_centered_matrices",
        label="Centered Matrices",
        script_name="03_build_centered_matrices.py",
        output_templates=(
            "tables/feature_intersection.txt",
            "tables/centered_wells_{pipeline}_orf.parquet",
            "tables/centered_wells_{pipeline}_crispr.parquet",
        ),
    ),
    RunPhaseSpec(
        phase_id="04_compute_gene_modality_stats",
        label="Gene Activity Stats",
        script_name="04_compute_gene_modality_stats.py",
        output_templates=("tables/gene_modality_stats.parquet",),
    ),
    RunPhaseSpec(
        phase_id="05_compute_signed_geometry",
        label="Signed Geometry",
        script_name="05_compute_signed_geometry.py",
        output_templates=("tables/same_gene_signed_geometry.parquet",),
    ),
    RunPhaseSpec(
        phase_id="06_run_retrieval",
        label="Retrieval",
        script_name="06_run_retrieval.py",
        output_templates=("tables/retrieval_ranks.parquet", "tables/retrieval_summary.csv"),
    ),
    RunPhaseSpec(
        phase_id="07_run_enrichment",
        label="Enrichment",
        script_name="07_run_enrichment.py",
        output_templates=("tables/enrichment_reactome.csv", "tables/enrichment_corum.csv"),
    ),
    RunPhaseSpec(
        phase_id="08_run_confounds_and_ablations",
        label="Confounds And Ablations",
        script_name="08_run_confounds_and_ablations.py",
        output_templates=(
            "tables/confound_summary.csv",
            "tables/regime_prevalence.csv",
            "tables/pipeline_agreement.csv",
            "tables/regime_prevalence_comparison.csv",
            "tables/ablation_summary.csv",
        ),
    ),
    RunPhaseSpec(
        phase_id="09_make_figures",
        label="Figures",
        script_name="09_make_figures.py",
        output_templates=(
            "figures/figure1_dataset.png",
            "figures/figure2_regime_map.png",
            "figures/figure3_stability.png",
            "figures/figure4_retrieval.png",
            "figures/figure5_enrichment.png",
            "figures/figure6_confounds.png",
            "source_data/figure1_dataset.csv",
            "source_data/figure2_regime_map.csv",
            "source_data/figure3_stability.csv",
            "source_data/figure4_retrieval.csv",
            "source_data/figure5_enrichment.csv",
            "source_data/figure6_confounds.csv",
        ),
    ),
    RunPhaseSpec(
        phase_id="10_assemble_paper_bundle",
        label="Paper Bundle",
        script_name="10_assemble_paper_bundle.py",
        output_templates=(
            "paper/claim_evidence_matrix.csv",
            "paper/manuscript.md",
            "paper/decision_summary.md",
            "paper/figure_manifest.csv",
            "paper/limitations_checklist.md",
        ),
    ),
    RunPhaseSpec(
        phase_id="11_make_release_report",
        label="Release Report",
        script_name="11_make_release_report.py",
        output_templates=("release_report.json",),
    ),
)

PHASE_INDEX = {phase.phase_id: phase for phase in RUN_PHASES}


def _required_columns_for_output(relative_path: str) -> set[str]:
    if relative_path == "tables/overlap_gene_table.parquet":
        return {"gene_id", "display_symbol", "canonical_symbol", "keep_for_analysis", "low_replicate_support"}
    if relative_path.startswith("tables/centered_wells_") and relative_path.endswith(".parquet"):
        return {
            "Metadata_Plate",
            "Metadata_Well",
            "Metadata_JCP2022",
            "Metadata_Batch",
            "Metadata_Source",
            "is_negative_control",
            "centering_source",
        }
    if relative_path == "tables/gene_modality_stats.parquet":
        return {"gene_id", "canonical_symbol", "modality", "strength_z", "rep_z", "active_bool", "n_wells"}
    if relative_path == "tables/same_gene_signed_geometry.parquet":
        return {
            "gene_id",
            "canonical_symbol",
            "regime_label_prebootstrap",
            "bootstrap_consensus_label",
            "bootstrap_consensus",
            "regime_label",
        }
    if relative_path == "tables/retrieval_ranks.parquet":
        return {"query_gene_id", "candidate_gene_id", "score_name", "rank", "direction"}
    if relative_path == "tables/retrieval_summary.csv":
        return {"score_name", "direction", "mrr", "hits_at_1", "n_queries"}
    if relative_path in {"tables/enrichment_reactome.csv", "tables/enrichment_corum.csv"}:
        return {
            "resource",
            "background_mode",
            "regime",
            "term_id",
            "term_name",
            "q_value",
            "overlap_genes",
        }
    if relative_path == "tables/confound_summary.csv":
        return {"confound_name", "test_name", "statistic_name", "statistic_value", "supports_positive_story"}
    if relative_path == "tables/regime_prevalence.csv":
        return {"pipeline", "regime_label", "n_genes", "fraction_of_analyzed", "n_regime_members"}
    if relative_path == "tables/pipeline_agreement.csv":
        return {
            "comparison_name",
            "has_real_comparison",
            "overall_agreement_primary_only",
            "aligned_agreement",
            "inverse_agreement",
            "orthogonal_agreement",
            "asymmetric_orf_agreement",
            "asymmetric_crispr_agreement",
        }
    if relative_path == "tables/regime_prevalence_comparison.csv":
        return {
            "comparison_name",
            "regime_label",
            "base_n_genes",
            "base_n_regime_members",
            "other_n_genes",
            "other_n_regime_members",
            "fraction_delta",
        }
    if relative_path == "tables/ablation_summary.csv":
        return {
            "run_name",
            "pipeline",
            "plate_centering",
            "artifact_sensitivity",
            "n_confident_noninactive",
            "agreement_vs_official",
            "decision_note",
        }
    if relative_path == "paper/claim_evidence_matrix.csv":
        return {"claim_id", "claim_text", "support_type", "supported", "required_artifacts", "evidence_detail"}
    if relative_path == "paper/figure_manifest.csv":
        return {"figure_name", "source_data_name"}
    return set()


def _validate_output_file(path: Path, relative_path: str) -> list[str]:
    errors: list[str] = []
    suffix = path.suffix.lower()
    try:
        if suffix == ".json":
            json.loads(path.read_text(encoding="utf-8"))
        elif suffix == ".md":
            if not path.read_text(encoding="utf-8").strip():
                errors.append(f"{relative_path}: markdown output is empty")
        elif suffix == ".txt":
            if relative_path == "tables/feature_intersection.txt" and not path.read_text(encoding="utf-8").strip():
                errors.append(f"{relative_path}: feature intersection is empty")
        elif suffix == ".png":
            if path.stat().st_size <= 0:
                errors.append(f"{relative_path}: figure file is empty")
        elif suffix == ".csv":
            frame = pl.read_csv(path)
            missing = sorted(_required_columns_for_output(relative_path) - set(frame.columns))
            if missing:
                errors.append(f"{relative_path}: missing required columns {missing}")
        elif suffix == ".parquet":
            frame = pl.read_parquet(path)
            missing = sorted(_required_columns_for_output(relative_path) - set(frame.columns))
            if missing:
                errors.append(f"{relative_path}: missing required columns {missing}")
    except Exception as exc:  # pragma: no cover - defensive parse failure path
        errors.append(f"{relative_path}: failed validation ({exc})")
    return errors


def _validate_phase_outputs(run_dir: Path, config: AnalysisConfig, phase: RunPhaseSpec) -> list[str]:
    errors: list[str] = []
    for output_path in _phase_output_paths(run_dir, config, phase):
        if not output_path.exists():
            continue
        relative_path = output_path.relative_to(run_dir).as_posix()
        errors.extend(_validate_output_file(output_path, relative_path))
    return errors


def _write_text(path: Path, content: str) -> Path:
    return write_text_atomic(path, content)


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _status_path(run_dir: Path) -> Path:
    return run_dir / "run_status.json"


def _phase_output_paths(run_dir: Path, config: AnalysisConfig, phase: RunPhaseSpec) -> list[Path]:
    return [run_dir / template.format(pipeline=config.pipeline) for template in phase.output_templates]


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _empty_phase_payload(run_dir: Path, config: AnalysisConfig, phase: RunPhaseSpec) -> dict[str, Any]:
    output_paths = _phase_output_paths(run_dir, config, phase)
    relative = [path.relative_to(run_dir).as_posix() for path in output_paths]
    return {
        "phase_id": phase.phase_id,
        "label": phase.label,
        "script_name": phase.script_name,
        "status": "pending",
        "started_at_utc": None,
        "completed_at_utc": None,
        "last_error": None,
        "required_outputs": relative,
        "present_outputs": [],
        "missing_outputs": relative,
        "n_outputs_present": 0,
        "n_outputs_total": len(relative),
        "checkpoint_reached": False,
    }


def _normalize_status_payload(
    run_dir: Path,
    config: AnalysisConfig,
    stored_payload: dict[str, Any],
) -> dict[str, Any]:
    stored_phase_map = {
        str(item.get("phase_id")): dict(item)
        for item in stored_payload.get("phases", [])
        if item.get("phase_id")
    }
    normalized_phases: list[dict[str, Any]] = []
    for phase in RUN_PHASES:
        output_paths = _phase_output_paths(run_dir, config, phase)
        relative = [path.relative_to(run_dir).as_posix() for path in output_paths]
        present = [rel for rel, path in zip(relative, output_paths, strict=False) if path.exists()]
        missing = [rel for rel, path in zip(relative, output_paths, strict=False) if not path.exists()]
        outputs_complete = len(present) == len(relative)
        validation_errors = _validate_phase_outputs(run_dir, config, phase) if outputs_complete else []
        outputs_valid = outputs_complete and not validation_errors
        stored_phase = stored_phase_map.get(phase.phase_id, _empty_phase_payload(run_dir, config, phase))
        stored_status = str(stored_phase.get("status", "pending"))
        if stored_status == "running":
            phase_status = "running"
        elif stored_status == "paused":
            phase_status = "paused"
        elif stored_status == "failed":
            phase_status = "failed"
        elif outputs_valid:
            phase_status = "completed"
        elif outputs_complete:
            phase_status = "failed"
        else:
            phase_status = "pending"
        normalized_phases.append(
            {
                "phase_id": phase.phase_id,
                "label": phase.label,
                "script_name": phase.script_name,
                "status": phase_status,
                "started_at_utc": stored_phase.get("started_at_utc"),
                "completed_at_utc": stored_phase.get("completed_at_utc") if outputs_valid else None,
                "last_error": (
                    "; ".join(validation_errors)
                    if validation_errors
                    else None
                    if outputs_valid
                    else stored_phase.get("last_error")
                ),
                "required_outputs": relative,
                "present_outputs": present,
                "missing_outputs": missing,
                "n_outputs_present": len(present),
                "n_outputs_total": len(relative),
                "checkpoint_reached": outputs_complete,
                "validation_errors": validation_errors,
                "outputs_validated": outputs_valid,
            }
        )

    completed_count = sum(1 for phase in normalized_phases if phase["status"] == "completed")
    running_phase = next((phase["phase_id"] for phase in normalized_phases if phase["status"] == "running"), None)
    paused_phase = next((phase["phase_id"] for phase in normalized_phases if phase["status"] == "paused"), None)
    failed_phase = next((phase["phase_id"] for phase in normalized_phases if phase["status"] == "failed"), None)
    next_phase = next((phase["phase_id"] for phase in normalized_phases if phase["status"] != "completed"), None)
    if completed_count == len(normalized_phases):
        overall_status = "completed"
    elif running_phase is not None:
        overall_status = "running"
    elif paused_phase is not None:
        overall_status = "paused"
    elif failed_phase is not None:
        overall_status = "failed"
    elif completed_count > 0:
        overall_status = "partial"
    else:
        overall_status = "initialized"

    payload = {
        "run_name": config.run_name,
        "pipeline": config.pipeline,
        "mode": config.mode,
        "status": overall_status,
        "current_phase_id": running_phase or paused_phase,
        "next_phase_id": next_phase,
        "completed_phase_ids": [phase["phase_id"] for phase in normalized_phases if phase["status"] == "completed"],
        "remaining_phase_ids": [phase["phase_id"] for phase in normalized_phases if phase["status"] != "completed"],
        "progress": {
            "completed_phases": completed_count,
            "total_phases": len(normalized_phases),
            "percent": round((100.0 * completed_count / len(normalized_phases)) if normalized_phases else 100.0, 1),
        },
        "started_at_utc": stored_payload.get("started_at_utc"),
        "last_updated_at_utc": _now_utc_iso(),
        "phases": normalized_phases,
    }
    control = stored_payload.get("control")
    if isinstance(control, dict) and control:
        payload["control"] = control
    return payload


def initialize_run(config_path: str | Path) -> RunContext:
    config = load_analysis_config(config_path)
    run_dir = config.result_dir / config.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    for name in RUN_SUBDIRS:
        (run_dir / name).mkdir(parents=True, exist_ok=True)

    config_snapshot = run_dir / "configs" / config.path.name
    write_text_atomic(config_snapshot, config.path.read_text(encoding="utf-8"))

    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        write_json(metrics_path, {"status": "initialized", "run_name": config.run_name})

    notes_path = run_dir / "notes.md"
    if not notes_path.exists():
        _write_text(
            notes_path,
            "\n".join(
                [
                    "# Notes",
                    "",
                    f"- Run initialized for `{config.run_name}`.",
                    "- Scientific outputs pending implementation.",
                ]
            )
            + "\n",
        )

    provenance = collect_provenance(
        {
            "project_name": config.project_name,
            "run_name": config.run_name,
            "pipeline": config.pipeline,
            "mode": config.mode,
            "seed": config.seed,
            "config_path": str(config.path),
        }
    )
    write_json(run_dir / "provenance.json", provenance)

    status_path = _status_path(run_dir)
    stored_status = _load_json(status_path)
    if not stored_status:
        stored_status = {"started_at_utc": _now_utc_iso(), "phases": []}
    write_json(status_path, _normalize_status_payload(run_dir, config, stored_status))
    return RunContext(config=config, run_dir=run_dir)


def register_phase_stub(
    config_path: str | Path,
    phase_name: str,
    extra: dict[str, Any] | None = None,
) -> RunContext:
    context = initialize_run(config_path)
    payload = {
        "status": "stub",
        "phase": phase_name,
        "run_name": context.config.run_name,
        "message": "Phase entrypoint is wired, but the scientific implementation is still pending.",
    }
    if extra:
        payload.update(extra)
    write_json(context.run_dir / "logs" / f"{phase_name}.json", payload)
    return context


def load_metrics(run_dir: Path) -> dict[str, Any]:
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        return {}
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def update_metrics(run_dir: Path, updates: dict[str, Any]) -> dict[str, Any]:
    payload = load_metrics(run_dir)
    payload.update(updates)
    write_json(run_dir / "metrics.json", payload)
    return payload


def update_run_control(run_dir: Path, config: AnalysisConfig, control: dict[str, Any]) -> dict[str, Any]:
    stored = _load_json(_status_path(run_dir))
    stored["control"] = control
    payload = _normalize_status_payload(run_dir, config, stored)
    write_json(_status_path(run_dir), payload)
    return payload


def clear_run_control(run_dir: Path, config: AnalysisConfig) -> dict[str, Any]:
    stored = _load_json(_status_path(run_dir))
    stored.pop("control", None)
    payload = _normalize_status_payload(run_dir, config, stored)
    write_json(_status_path(run_dir), payload)
    return payload


def refresh_run_status(run_dir: Path, config: AnalysisConfig) -> dict[str, Any]:
    payload = _normalize_status_payload(run_dir, config, _load_json(_status_path(run_dir)))
    write_json(_status_path(run_dir), payload)
    return payload


def load_run_status(config_path: str | Path) -> dict[str, Any]:
    context = initialize_run(config_path)
    return refresh_run_status(context.run_dir, context.config)


def _mutate_phase_status(
    run_dir: Path,
    config: AnalysisConfig,
    phase_id: str,
    *,
    status: str,
    last_error: str | None = None,
    set_completed_at: bool = False,
) -> dict[str, Any]:
    if phase_id not in PHASE_INDEX:
        raise KeyError(f"Unknown phase_id: {phase_id}")
    stored = _load_json(_status_path(run_dir))
    phase_map = {str(item.get("phase_id")): dict(item) for item in stored.get("phases", []) if item.get("phase_id")}
    current = phase_map.get(phase_id, _empty_phase_payload(run_dir, config, PHASE_INDEX[phase_id]))
    now = _now_utc_iso()
    current["status"] = status
    current["started_at_utc"] = current.get("started_at_utc") or now
    current["last_error"] = last_error
    if set_completed_at:
        current["completed_at_utc"] = now
    phase_map[phase_id] = current
    stored["phases"] = [phase_map.get(phase.phase_id, _empty_phase_payload(run_dir, config, phase)) for phase in RUN_PHASES]
    payload = _normalize_status_payload(run_dir, config, stored)
    if status in {"running", "paused"}:
        payload["status"] = status
        payload["current_phase_id"] = phase_id
    write_json(_status_path(run_dir), payload)
    return payload


def mark_phase_started(run_dir: Path, config: AnalysisConfig, phase_id: str) -> dict[str, Any]:
    return _mutate_phase_status(run_dir, config, phase_id, status="running", last_error=None)


def mark_phase_failed(run_dir: Path, config: AnalysisConfig, phase_id: str, message: str) -> dict[str, Any]:
    return _mutate_phase_status(run_dir, config, phase_id, status="failed", last_error=message)


def mark_phase_paused(run_dir: Path, config: AnalysisConfig, phase_id: str, message: str) -> dict[str, Any]:
    return _mutate_phase_status(run_dir, config, phase_id, status="paused", last_error=message)


def mark_phase_completed(run_dir: Path, config: AnalysisConfig, phase_id: str) -> dict[str, Any]:
    payload = _mutate_phase_status(run_dir, config, phase_id, status="completed", last_error=None, set_completed_at=True)
    payload = refresh_run_status(run_dir, config) if payload else payload
    phase_payload = next((phase for phase in payload.get("phases", []) if phase.get("phase_id") == phase_id), None)
    if phase_payload and phase_payload.get("status") != "completed":
        raise RuntimeError(
            f"Phase {phase_id} produced outputs that failed validation: "
            f"{'; '.join(phase_payload.get('validation_errors', [])) or 'unknown validation error'}"
        )
    return payload


@contextmanager
def tracked_phase(config_path: str | Path, phase_id: str) -> Iterator[RunContext]:
    context = initialize_run(config_path)
    mark_phase_started(context.run_dir, context.config, phase_id)
    try:
        yield context
    except Exception as exc:
        mark_phase_failed(context.run_dir, context.config, phase_id, str(exc))
        raise
    else:
        try:
            mark_phase_completed(context.run_dir, context.config, phase_id)
        except Exception as exc:
            mark_phase_failed(context.run_dir, context.config, phase_id, str(exc))
            raise


def get_pending_phase_ids(status_payload: dict[str, Any], *, include_running: bool = True) -> list[str]:
    pending_statuses = {"pending", "failed", "paused"}
    if include_running:
        pending_statuses.add("running")
    return [phase["phase_id"] for phase in status_payload.get("phases", []) if phase.get("status") in pending_statuses]


def _phase_id_range(from_phase: str | None, through_phase: str | None) -> tuple[int, int]:
    phase_ids = [phase.phase_id for phase in RUN_PHASES]
    start = phase_ids.index(from_phase) if from_phase is not None else 0
    end = phase_ids.index(through_phase) if through_phase is not None else len(phase_ids) - 1
    if end < start:
        raise ValueError("through_phase must not come before from_phase.")
    return start, end


def _phase_extra_args(config: AnalysisConfig, phase_id: str) -> list[str]:
    if phase_id == "08_run_confounds_and_ablations" and bool(
        config.raw.get("preprocess", {}).get("artifact_sensitivity_filter", False)
    ):
        return ["--mode", "artifact_sensitivity"]
    return []


def execute_pipeline(
    config_path: str | Path,
    *,
    from_phase: str | None = None,
    through_phase: str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    context = initialize_run(config_path)
    status_payload = refresh_run_status(context.run_dir, context.config)
    start_idx, end_idx = _phase_id_range(from_phase, through_phase)
    selected = RUN_PHASES[start_idx : end_idx + 1]
    if force:
        phases_to_run = list(selected)
    else:
        pending = set(get_pending_phase_ids(status_payload, include_running=True))
        phases_to_run = [phase for phase in selected if phase.phase_id in pending]

    for phase in phases_to_run:
        command = [
            sys.executable,
            str(repo_root() / "scripts" / phase.script_name),
            "--config",
            str(context.config.path),
            *_phase_extra_args(context.config, phase.phase_id),
        ]
        subprocess.run(command, cwd=repo_root(), check=True)
    return refresh_run_status(context.run_dir, context.config)
