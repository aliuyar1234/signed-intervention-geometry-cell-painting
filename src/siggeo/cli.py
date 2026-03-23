from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from siggeo import __version__
from siggeo.config import load_analysis_config, repo_root
from siggeo.run_control import pause_run_live, resume_run_live
from siggeo.runs import execute_pipeline, initialize_run, load_run_status


app = typer.Typer(
    add_completion=False,
    help="CLI for the signed intervention geometry research pipeline.",
    no_args_is_help=True,
)
console = Console()


def _print_status(status_payload: dict[str, object]) -> None:
    progress = dict(status_payload.get("progress", {}))
    control = status_payload.get("control")
    summary = Table(title=f"Run Status: {status_payload['run_name']}", show_header=False)
    summary.add_row("Pipeline", str(status_payload.get("pipeline")))
    summary.add_row("Mode", str(status_payload.get("mode")))
    summary.add_row("Status", str(status_payload.get("status")))
    summary.add_row(
        "Progress",
        f"{progress.get('completed_phases', 0)}/{progress.get('total_phases', 0)} "
        f"({progress.get('percent', 0.0)}%)",
    )
    summary.add_row("Current phase", str(status_payload.get("current_phase_id") or "-"))
    summary.add_row("Next phase", str(status_payload.get("next_phase_id") or "-"))
    summary.add_row("Remaining", ", ".join(status_payload.get("remaining_phase_ids", [])) or "-")
    if isinstance(control, dict) and control:
        summary.add_row("Pause mode", str(control.get("mode") or "-"))
        summary.add_row("Live PIDs", ", ".join(str(pid) for pid in control.get("pids", [])) or "-")
    console.print(summary)

    phases = Table(title="Phase Checkpoints")
    phases.add_column("Phase")
    phases.add_column("Label")
    phases.add_column("Status")
    phases.add_column("Outputs")
    phases.add_column("Missing")
    for phase in status_payload.get("phases", []):
        status = str(phase.get("status"))
        style = {
            "completed": "green",
            "running": "yellow",
            "paused": "magenta",
            "failed": "red",
            "pending": "cyan",
        }.get(status, "white")
        phases.add_row(
            str(phase.get("phase_id")),
            str(phase.get("label")),
            f"[{style}]{status}[/{style}]",
            f"{phase.get('n_outputs_present', 0)}/{phase.get('n_outputs_total', 0)}",
            ", ".join(phase.get("missing_outputs", [])) or "-",
        )
    console.print(phases)


@app.command("repo-root")
def repo_root_command() -> None:
    """Print the repository root used for config resolution."""
    console.print(str(repo_root()))


@app.command("show-config")
def show_config(config: Path = typer.Option(..., "--config", exists=True, readable=True)) -> None:
    """Render a minimal config summary."""
    loaded = load_analysis_config(config)
    console.print(
        {
            "project_name": loaded.project_name,
            "run_name": loaded.run_name,
            "pipeline": loaded.pipeline,
            "mode": loaded.mode,
            "result_dir": str(loaded.result_dir),
            "seed": loaded.seed,
        }
    )


@app.command("init-run")
def init_run(config: Path = typer.Option(..., "--config", exists=True, readable=True)) -> None:
    """Create the deterministic run folder and provenance snapshot."""
    context = initialize_run(config)
    console.print(f"Initialized run: {context.config.run_name}")
    console.print(f"Run directory: {context.run_dir}")


@app.command("run-status")
def run_status(config: Path = typer.Option(..., "--config", exists=True, readable=True)) -> None:
    """Show checkpoint progress, remaining phases, and output coverage for a run."""
    _print_status(load_run_status(config))


@app.command("run-pipeline")
def run_pipeline(
    config: Path = typer.Option(..., "--config", exists=True, readable=True),
    from_phase: str | None = typer.Option(None, "--from-phase", help="Start from a specific phase id."),
    through_phase: str | None = typer.Option(None, "--through-phase", help="Stop after a specific phase id."),
    force: bool = typer.Option(False, "--force", help="Rerun selected phases even if their outputs exist."),
) -> None:
    """Run or resume the pipeline from the latest checkpointed state."""
    final_status = execute_pipeline(
        config,
        from_phase=from_phase,
        through_phase=through_phase,
        force=force,
    )
    _print_status(final_status)


@app.command("resume-run")
def resume_run(
    config: Path = typer.Option(..., "--config", exists=True, readable=True),
    from_phase: str | None = typer.Option(None, "--from-phase", help="Resume from a specific phase id."),
    through_phase: str | None = typer.Option(None, "--through-phase", help="Stop after a specific phase id."),
) -> None:
    """Resume only the unfinished phases for a run."""
    final_status = execute_pipeline(
        config,
        from_phase=from_phase,
        through_phase=through_phase,
        force=False,
    )
    _print_status(final_status)


@app.command("pause-run")
def pause_run(config: Path = typer.Option(..., "--config", exists=True, readable=True)) -> None:
    """Live-suspend a currently running pipeline on Windows."""
    _print_status(pause_run_live(config))


@app.command("resume-live")
def resume_live(config: Path = typer.Option(..., "--config", exists=True, readable=True)) -> None:
    """Resume a run that was live-suspended with pause-run."""
    _print_status(resume_run_live(config))


@app.callback()
def main_callback(version: bool = typer.Option(False, "--version", help="Show version and exit.")) -> None:
    if version:
        console.print(__version__)
        raise typer.Exit()


def main() -> None:
    app()


if __name__ == "__main__":
    main()
