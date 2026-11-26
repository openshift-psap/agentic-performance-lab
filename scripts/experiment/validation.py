"""Trace output and functional-conformance gates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from trace_replay_sim.functional_conformance import verify_trace, write_report
from trace_replay_sim.output_validation import (
    OutputValidationError,
    validate_trace_output,
)

from .config import LOG


def validate_collected_trace(
    trace_dir: Path,
    session_id: str,
    backend: Any,
    session_events: list[dict[str, Any]],
) -> None:
    """Fail the trace when its measurement output or replay conformance is invalid."""
    input_payload = json.loads(
        (trace_dir / "input_corpus.json").read_text(encoding="utf-8")
    )
    input_sessions = input_payload.get("sessions") or []
    if len(input_sessions) != 1:
        raise RuntimeError(
            f"invalid input corpus for {session_id}: expected exactly one session"
        )
    expected_session = input_sessions[0]
    expected_turns = len(expected_session.get("turns") or [])
    expected_tool_calls = sum(
        len(turn.get("tool_calls") or [])
        for turn in expected_session.get("turns") or []
    )
    try:
        output_evidence = validate_trace_output(
            trace_dir,
            expected_turns=expected_turns,
            expected_tool_calls=expected_tool_calls,
            backend_dir_name="appworld"
            if backend.benchmark == "appworld"
            else "backend",
        )
    except OutputValidationError as exc:
        (trace_dir / "output_validation_error.json").write_text(
            str(exc) + "\n", encoding="utf-8"
        )
        raise RuntimeError(
            f"measurement output validation failed for {session_id}: {exc}"
        ) from exc
    (trace_dir / "output_validation.json").write_text(
        json.dumps(output_evidence, indent=2) + "\n", encoding="utf-8"
    )
    if not session_events and expected_tool_calls > 0:
        raise RuntimeError(
            f"no real {backend.name} backend events captured for {session_id}"
        )
    conformance = verify_trace(
        trace_dir,
        appworld_events=session_events if backend.benchmark == "appworld" else None,
        backend_events=session_events,
        backend_kind=backend.benchmark,
        backend_event_name=backend.event_name,
    )
    write_report(trace_dir, conformance)
    for check in conformance.get("checks", []):
        if check.get("status") == "WARN":
            LOG.warning(
                "functional conformance warning for %s: %s: %s",
                session_id,
                check.get("name"),
                check.get("details") or "",
            )
    if not conformance.get("passed"):
        failed_checks = [
            check["name"]
            for check in conformance.get("checks", [])
            if check.get("status") == "FAIL"
        ]
        raise RuntimeError(
            f"functional conformance failed for {session_id}: {', '.join(failed_checks)}"
        )


def exit_code_warnings(out: Path) -> list[dict[str, Any]]:
    """Collect SWE exit-code mismatches reported by completed trace gates."""
    warnings: list[dict[str, Any]] = []
    for report_path in sorted((out / "traces").glob("*/functional_conformance.json")):
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for check in report.get("checks") or []:
            if (
                check.get("name") != "swe_recorded_exit_codes"
                or check.get("status") != "WARN"
            ):
                continue
            warnings.append(
                {
                    "session_id": str(
                        report.get("session_id") or report_path.parent.name
                    ),
                    "mismatched_turns": int(check.get("mismatched_turns") or 0),
                    "turn_indices": list(check.get("mismatched_turn_indices") or []),
                }
            )
    return warnings
