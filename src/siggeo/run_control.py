from __future__ import annotations

import ctypes
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from siggeo.config import load_analysis_config
from siggeo.runs import (
    RUN_PHASES,
    clear_run_control,
    initialize_run,
    load_run_status,
    mark_phase_paused,
    mark_phase_started,
    update_run_control,
)


@dataclass(frozen=True)
class LiveProcessInfo:
    pid: int
    parent_pid: int | None
    name: str
    role: str
    command_line: str


def supports_live_suspend() -> bool:
    return sys.platform == "win32"


def _powershell_json(command: str) -> Any:
    completed = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            "[Console]::OutputEncoding=[System.Text.Encoding]::UTF8; " + command,
        ],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    payload = completed.stdout.strip()
    if not payload:
        return []
    return json.loads(payload)


def _list_windows_processes() -> list[dict[str, Any]]:
    raw = _powershell_json(
        "Get-CimInstance Win32_Process | "
        "Select-Object ProcessId,ParentProcessId,Name,CreationDate,CommandLine | "
        "ConvertTo-Json -Compress"
    )
    if isinstance(raw, list):
        return [dict(item) for item in raw]
    if isinstance(raw, dict):
        return [dict(raw)]
    return []


def discover_run_processes(config_path: str | Path) -> list[LiveProcessInfo]:
    if not supports_live_suspend():
        return []
    config = load_analysis_config(config_path)
    config_token = str(config.path).lower()
    current_pid = os.getpid()
    phase_script_names = tuple(phase.script_name.lower() for phase in RUN_PHASES)
    matches: list[LiveProcessInfo] = []
    for item in _list_windows_processes():
        pid = int(item.get("ProcessId") or 0)
        if pid <= 0 or pid == current_pid:
            continue
        command_line = str(item.get("CommandLine") or "")
        lowered = command_line.lower()
        if config_token not in lowered:
            continue

        role: str | None = None
        if any(script_name in lowered for script_name in phase_script_names):
            role = "worker"
        elif "-m siggeo.cli run-pipeline" in lowered or "-m siggeo.cli resume-run" in lowered:
            role = "coordinator"
        elif " run-pipeline " in lowered or " resume-run " in lowered:
            role = "wrapper"
        if role is None:
            continue

        matches.append(
            LiveProcessInfo(
                pid=pid,
                parent_pid=int(item.get("ParentProcessId")) if item.get("ParentProcessId") is not None else None,
                name=str(item.get("Name") or ""),
                role=role,
                command_line=command_line,
            )
        )

    role_order = {"worker": 0, "coordinator": 1, "wrapper": 2}
    return sorted(matches, key=lambda item: (role_order.get(item.role, 99), item.pid))


def _open_process_handle(pid: int) -> int:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    process_suspend_resume = 0x0800
    handle = kernel32.OpenProcess(process_suspend_resume, False, pid)
    if not handle:
        raise OSError(ctypes.get_last_error(), f"Failed to open process {pid} for suspend/resume.")
    return handle


def _suspend_process(pid: int) -> None:
    handle = _open_process_handle(pid)
    try:
        ntdll = ctypes.WinDLL("ntdll")
        status = ntdll.NtSuspendProcess(handle)
        if status != 0:
            raise OSError(status, f"NtSuspendProcess failed for pid {pid} with status {status}.")
    finally:
        ctypes.WinDLL("kernel32", use_last_error=True).CloseHandle(handle)


def _resume_process(pid: int) -> None:
    handle = _open_process_handle(pid)
    try:
        ntdll = ctypes.WinDLL("ntdll")
        status = ntdll.NtResumeProcess(handle)
        if status != 0:
            raise OSError(status, f"NtResumeProcess failed for pid {pid} with status {status}.")
    finally:
        ctypes.WinDLL("kernel32", use_last_error=True).CloseHandle(handle)


def pause_run_live(config_path: str | Path) -> dict[str, Any]:
    if not supports_live_suspend():
        raise RuntimeError("Live pause is currently implemented only on Windows.")

    status_payload = load_run_status(config_path)
    if str(status_payload.get("status")) != "running":
        raise RuntimeError("Run is not currently running. Use run-pipeline or resume-run instead.")
    current_phase = str(status_payload.get("current_phase_id") or "")
    if not current_phase:
        raise RuntimeError("No active phase found for the running run.")

    targets = [process for process in discover_run_processes(config_path) if process.role in {"worker", "coordinator"}]
    if not targets:
        raise RuntimeError("No live run processes were found for this config.")

    for process in targets:
        _suspend_process(process.pid)

    context = initialize_run(config_path)
    message = (
        "Live-suspended via pause-run. Resume with resume-live to continue the in-memory worker, "
        "or use resume-run only if the live processes are no longer available."
    )
    mark_phase_paused(context.run_dir, context.config, current_phase, message)
    update_run_control(
        context.run_dir,
        context.config,
        {
            "mode": "live_suspend",
            "pids": [process.pid for process in targets],
            "processes": [asdict(process) for process in targets],
        },
    )
    return load_run_status(config_path)


def resume_run_live(config_path: str | Path) -> dict[str, Any]:
    if not supports_live_suspend():
        raise RuntimeError("Live resume is currently implemented only on Windows.")

    status_payload = load_run_status(config_path)
    if str(status_payload.get("status")) != "paused":
        raise RuntimeError("Run is not paused.")
    control = status_payload.get("control")
    if not isinstance(control, dict) or control.get("mode") != "live_suspend":
        raise RuntimeError("This run is checkpoint-paused, not live-suspended. Use resume-run instead.")

    current_phase = str(status_payload.get("current_phase_id") or status_payload.get("next_phase_id") or "")
    if not current_phase:
        raise RuntimeError("No resumable phase was found in the paused run status.")

    raw_pids = control.get("pids") or []
    pids = [int(pid) for pid in raw_pids if int(pid) > 0]
    if not pids:
        targets = [process for process in discover_run_processes(config_path) if process.role in {"worker", "coordinator"}]
        pids = [process.pid for process in targets]
    if not pids:
        raise RuntimeError("No suspended live processes were found. Use resume-run instead.")

    # Resume the coordinator before the worker so the parent is ready to observe the child again.
    coordinator_pids = [pid for pid in pids if pid in {process.pid for process in discover_run_processes(config_path) if process.role == "coordinator"}]
    worker_pids = [pid for pid in pids if pid not in coordinator_pids]
    for pid in coordinator_pids + worker_pids:
        _resume_process(pid)

    context = initialize_run(config_path)
    clear_run_control(context.run_dir, context.config)
    mark_phase_started(context.run_dir, context.config, current_phase)
    return load_run_status(config_path)
