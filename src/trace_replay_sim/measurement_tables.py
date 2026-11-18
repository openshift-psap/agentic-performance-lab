"""Tabular latency and resource reports for a measured replay trace."""

from __future__ import annotations

import csv
import math
import statistics
from pathlib import Path
from typing import Any


LATENCY_FIELDS = (
    "trace_id", "turn", "context_assembly_ms", "context_boundary_status",
    "context_start_source", "model_call_ms", "context_to_model_end_ms", "model_ttft_ms",
    "model_decode_proxy_ms", "response_processing_ms", "model_end_to_tool_start_ms",
    "pre_execution_gap_ms", "tool_call_total_ms", "openshell_execution_ms",
    "tool_failure_ms", "sandbox_init_ms",
    "tool_count", "input_tokens", "output_tokens", "prompt_chars", "request_bytes",
    "input_messages", "input_message_chars", "system_prompt_chars", "tool_definition_chars",
)
CONTEXT_BOUNDARY_FIELDS = (
    "trace_id", "turn", "start_source", "start_epoch_ms", "model_call_start_epoch_ms",
    "model_endpoint_recv_epoch_ms", "model_endpoint_body_read_epoch_ms",
    "agent_pre_model_ms", "model_request_arrival_ms", "model_request_body_read_ms",
    "total_pre_model_ms", "boundary_status", "prompt_chars", "request_bytes",
)
RESOURCE_FIELDS = (
    "component", "instance", "epoch_ns", "within_measured_window", "cpu_cores", "memory_mib",
    "working_set_mib", "cpu_usage_usec", "nr_throttled", "throttled_usec",
)
BACKEND_FIELDS = (
    "session_id", "turn", "call", "tool", "execution_mode", "elapsed_ms",
    "sandbox_returncode", "sandbox_command_wall_ms", "sandbox_user_cpu_ms",
    "sandbox_system_cpu_ms", "result_bytes", "arguments_sha256", "run_key",
)


def _write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "mean": "", "p50": "", "p95": "", "min": "", "max": ""}
    ordered = sorted(values)
    return {
        "n": len(ordered), "mean": round(statistics.mean(ordered), 3),
        "p50": round(statistics.median(ordered), 3),
        "p95": round(ordered[math.floor(0.95 * (len(ordered) - 1))], 3),
        "min": round(ordered[0], 3), "max": round(ordered[-1], 3),
    }


def _latency_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        start = _number(row.get("context_start_ms"))
        end = _number(row.get("model_end_ms"))
        ordered = row.get("context_boundary_status") in ("ordered", "ordered_missing_recv")
        combined = (
            round(end - start, 3)
            if ordered and start is not None and end is not None and end >= start
            else None
        )
        result.append({
            "trace_id": row.get("trace_id"), "turn": row.get("turn"),
            "context_assembly_ms": row.get("context_assembly_received_ms"),
            "context_boundary_status": row.get("context_boundary_status"),
            "context_start_source": row.get("context_start_source"),
            "model_call_ms": row.get("model_call_ms"),
            "context_to_model_end_ms": combined,
            "model_ttft_ms": row.get("mock_ttft_ms"),
            "model_decode_proxy_ms": row.get("mock_decode_ms"),
            "response_processing_ms": row.get("response_processing_ms"),
            "model_end_to_tool_start_ms": row.get("response_processing_to_tool_start_ms"),
            "pre_execution_gap_ms": row.get("pre_execution_gap_ms"),
            "tool_call_total_ms": row.get("tool_dispatch_ms"),
            "openshell_execution_ms": row.get("exec_ms"),
            "tool_failure_ms": row.get("tool_failure_ms"),
            "sandbox_init_ms": row.get("sandbox_init_ms"),
            "tool_count": row.get("tool_count"),
            "input_tokens": row.get("input_tokens"),
            "output_tokens": row.get("output_tokens"),
            "prompt_chars": row.get("prompt_chars"),
            "request_bytes": row.get("request_bytes"),
            "input_messages": row.get("input_messages"),
            "input_message_chars": row.get("input_message_chars"),
            "system_prompt_chars": row.get("system_prompt_chars"),
            "tool_definition_chars": row.get("tool_definition_chars"),
        })
    return result


def _context_boundary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Expose observed event edges without presenting their gap as pure prompt building."""
    result = []
    for row in rows:
        start = _number(row.get("context_start_ms"))
        model = _number(row.get("model_start_ms"))
        receive = _number(row.get("model_endpoint_recv_ms"))
        body = _number(row.get("model_endpoint_body_read_ms"))
        status = str(row.get("context_boundary_status") or "missing_endpoint_edge")
        ordered = status in ("ordered", "ordered_missing_recv")
        result.append({
            "trace_id": row.get("trace_id"), "turn": row.get("turn"),
            "start_source": row.get("context_start_source"),
            "start_epoch_ms": start, "model_call_start_epoch_ms": model,
            "model_endpoint_recv_epoch_ms": receive,
            "model_endpoint_body_read_epoch_ms": body,
            "agent_pre_model_ms": round(model - start, 3) if ordered and start is not None and model is not None else "",
            "model_request_arrival_ms": round(receive - model, 3) if ordered and receive is not None and model is not None else "",
            "model_request_body_read_ms": round(body - receive, 3) if ordered and body is not None and receive is not None else "",
            "total_pre_model_ms": round(body - start, 3) if ordered and start is not None and body is not None else "",
            "boundary_status": status,
            "prompt_chars": row.get("prompt_chars"), "request_bytes": row.get("request_bytes"),
        })
    return result


def _efficiency_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        sandbox_init = float(row.get("sandbox_init_bucket_ms") or 0)
        context = float(row.get("context_assembly_bucket_ms") or 0)
        response = float(row.get("response_processing_bucket_ms") or 0)
        dispatch = float(row.get("tool_dispatch_overhead_bucket_ms") or 0)
        execution = float(row.get("sandbox_execution_bucket_ms") or 0)
        failure = float(row.get("tool_failure_bucket_ms") or 0)
        non_model = sandbox_init + context + response + dispatch + execution + failure
        result.append({
            "turn": int(row.get("turn") or 0),
            "model_call_ms": float(row.get("model_call_ms") or 0),
            "sandbox_init_ms": sandbox_init,
            "prompt_boundary_ms": context,
            "raw_prompt_boundary_ms": float(row.get("prompt_boundary_ms") or 0),
            "response_processing_ms": response,
            "response_processing_raw_ms": float(row.get("response_processing_full_ms") or 0),
            "tool_dispatch_overhead_ms": dispatch,
            "openclaw_observed_overhead_ms": context + response + dispatch,
            "openshell_exec_ms": execution,
            "tool_failure_ms": failure,
            "non_model_path_ms": non_model,
            "openshell_share_pct": 100 * execution / non_model if non_model else 0.0,
        })
    return result


def _resource_rows(cgroup_dir: Path, measured_window: tuple[float, float] | None) -> list[dict[str, Any]]:
    sources = [path for path in sorted(cgroup_dir.glob("*.csv")) if path.name != "openshell_sandbox.csv"]
    sandbox_sources = sorted((cgroup_dir / "sandbox_samples").glob("*.csv"))
    sources += sandbox_sources or list(cgroup_dir.glob("openshell_sandbox.csv"))
    result = []
    for source in sources:
        component = "openshell_sandbox" if source in sandbox_sources or source.stem == "openshell_sandbox" else source.stem
        previous: tuple[int, float] | None = None
        try:
            handle = source.open(encoding="utf-8")
        except OSError:
            continue
        with handle:
            for raw in csv.DictReader(handle):
                try:
                    epoch = int(raw["epoch_ns"])
                    usage = float(raw["cpu_usage_usec"])
                    memory = float(raw["memory_bytes"])
                except (KeyError, TypeError, ValueError):
                    continue
                if epoch <= 0 or usage < 0 or memory < 0:
                    continue
                in_window = (
                    int(measured_window[0] * 1e9) <= epoch <= int(measured_window[1] * 1e9)
                    if measured_window else "unknown"
                )
                cpu_cores = None
                if (previous and epoch > previous[0] and usage >= previous[1]
                        and (not measured_window or (in_window and previous[0] >= int(measured_window[0] * 1e9)))):
                    cpu_cores = (usage - previous[1]) / ((epoch - previous[0]) / 1000)
                previous = (epoch, usage)
                working = _number(raw.get("memory_working_set_bytes"))
                result.append({
                    "component": component, "instance": source.stem,
                    "epoch_ns": epoch, "cpu_cores": round(cpu_cores, 6) if cpu_cores is not None else "",
                    "within_measured_window": in_window,
                    "memory_mib": round(memory / 1048576, 3),
                    "working_set_mib": round(working / 1048576, 3) if working is not None else "",
                    "cpu_usage_usec": int(usage),
                    "nr_throttled": raw.get("nr_throttled", ""),
                    "throttled_usec": raw.get("throttled_usec", ""),
                })
    return sorted(result, key=lambda row: (str(row["component"]), str(row["instance"]), int(row["epoch_ns"])))


def write_profile_tables(
    rows: list[dict[str, Any]], tool_rows: list[dict[str, Any]], prom: Path | None,
    out: Path, measured_window: tuple[float, float] | None = None,
) -> list[str]:
    """Write stable CSV tables; raw JSON and sampler data remain available for audit."""
    latency = _latency_rows(rows)
    context_boundaries = _context_boundary_rows(rows)
    efficiency = _efficiency_rows(rows)
    resources = _resource_rows(prom / "cgroup", measured_window) if prom else []
    _write_csv(out / "per_turn_latency.csv", LATENCY_FIELDS, latency)
    _write_csv(out / "context_boundary_events.csv", CONTEXT_BOUNDARY_FIELDS, context_boundaries)
    _write_csv(out / "per_turn_efficiency.csv", tuple(efficiency[0]) if efficiency else ("turn",), efficiency)
    tool_fields = tuple(dict.fromkeys(key for row in tool_rows for key in row)) or ("trace_id", "tool_name", "duration_ms")
    _write_csv(out / "per_tool_latency.csv", tool_fields, tool_rows)
    stages = (
        "context_assembly_ms", "model_call_ms", "context_to_model_end_ms", "model_ttft_ms",
        "model_decode_proxy_ms", "response_processing_ms",
        "model_end_to_tool_start_ms", "pre_execution_gap_ms",
        "tool_call_total_ms", "openshell_execution_ms", "tool_failure_ms",
        "sandbox_init_ms",
    )
    latency_summary = [{"stage": stage, "unit": "ms", **_stats([value for row in latency if (value := _number(row.get(stage))) is not None])} for stage in stages]
    _write_csv(out / "stage_latency_summary.csv", ("stage", "unit", "n", "mean", "p50", "p95", "min", "max"), latency_summary)
    _write_csv(out / "resource_samples.csv", RESOURCE_FIELDS, resources)
    resource_summary = []
    for component, instance in sorted({(str(row["component"]), str(row["instance"])) for row in resources}):
        samples = [
            row for row in resources
            if row["component"] == component and row["instance"] == instance
            and row["within_measured_window"] is not False
        ]
        cpu = [value for row in samples if (value := _number(row["cpu_cores"])) is not None]
        memory = [float(row["memory_mib"]) for row in samples]
        working = [value for row in samples if (value := _number(row["working_set_mib"])) is not None]
        cpu_seconds = (int(samples[-1]["cpu_usage_usec"]) - int(samples[0]["cpu_usage_usec"])) / 1e6 if len(samples) > 1 else None
        resource_summary.append({
            "component": component, "instance": instance, "scope": "measured_driver_window" if measured_window else "available_sampler_window", "samples": len(samples),
            "cpu_samples": len(cpu), "cpu_core_seconds": round(max(0.0, cpu_seconds), 6) if cpu_seconds is not None else "",
            "cpu_mean_cores": _stats(cpu)["mean"], "cpu_p95_cores": _stats(cpu)["p95"], "cpu_peak_cores": _stats(cpu)["max"],
            "memory_mean_mib": _stats(memory)["mean"], "memory_p95_mib": _stats(memory)["p95"], "memory_peak_mib": _stats(memory)["max"],
            "working_set_peak_mib": _stats(working)["max"],
        })
    _write_csv(out / "resource_summary.csv", (
        "component", "instance", "scope", "samples", "cpu_samples", "cpu_core_seconds",
        "cpu_mean_cores", "cpu_p95_cores", "cpu_peak_cores", "memory_mean_mib",
        "memory_p95_mib", "memory_peak_mib", "working_set_peak_mib",
    ), resource_summary)
    (out / "README.md").write_text(
        "# Measured trace tables\n\n"
        "- `per_turn_latency.csv`: one row per recorded model turn. `input_messages`, `input_message_chars`, `prompt_chars`, and `request_bytes` come from the per-call OpenClaw OTEL model span and show context/request growth directly; they are sizes, not durations. `context_assembly_ms` starts at this turn's prior tool-result span end, or at the first harness-run start, and ends when the model endpoint has received the complete request body. It includes agent preparation and request transfer, not only prompt-building CPU time. `context_to_model_end_ms` is the elapsed time from that same context start to model-call completion. It combines context assembly and model call without counting their overlapping request-transfer interval twice. Missing or unordered boundaries remain blank and are identified by `context_boundary_status`; no shorter model-start gap is substituted. On tool turns, response processing runs from model completion to the start of the measured OpenShell execution span; this includes the pre-execution handoff. `model_end_to_tool_start_ms` and `pre_execution_gap_ms` show its two parts when available. `tool_call_total_ms` is the broad OpenClaw wrapper span and overlaps response processing and OpenShell execution; do not add it to them. On turns without a tool, response processing ends at the next model call when available.\n"
        "- `context_boundary_events.csv`: timestamps and component gaps behind each context value. `model_call_start` is an OpenClaw span boundary, not proof of the exact socket-send instant. `agent_pre_model_ms` may include orchestration as well as prompt preparation; `model_request_arrival_ms` and `model_request_body_read_ms` describe the model-endpoint handoff. An inconsistent or missing boundary is flagged and its gaps are left blank.\n"
        "- `stage_latency_summary.csv`: count, mean, p50, p95, minimum, and maximum for each per-turn latency column. Empty measurements are excluded.\n"
        "- `per_tool_latency.csv`: OpenClaw tool span duration and outcome for each recorded tool call. This is a broader span than the sandbox command wall time.\n"
        "- `per_turn_efficiency.csv`: turn-boundary phase decomposition retained for conformance checks. Response and tool work following model turn N appear on the next boundary row.\n"
        "- `resource_samples.csv`: raw cgroup sample timestamps with CPU rates derived from adjacent counter samples and memory in MiB. `within_measured_window` identifies points inside the client driver window.\n"
        "- `resource_summary.csv`: CPU and memory statistics for samples inside the measured driver window when available. Zero samples means the sampler missed that window; it is not a zero-utilization result.\n"
        "- `data/backend/backend_calls.csv` or `data/appworld/backend_calls.csv`: per-call backend and sandbox command timings, located beside this profile.\n\n"
        "Raw Jaeger, model-edge, and cgroup files remain alongside these tables for audit.\n",
        encoding="utf-8",
    )
    return [path.name for path in sorted(out.glob("*.csv"))]


def write_backend_calls(events: list[dict[str, Any]], out: Path) -> None:
    rows = []
    for event in events:
        rows.append({
            "session_id": event.get("session_id"), "turn": event.get("turn"), "call": event.get("call"),
            "tool": event.get("tool"), "execution_mode": event.get("execution_mode"),
            "elapsed_ms": event.get("elapsed_ms"), "sandbox_returncode": event.get("sandbox_returncode"),
            "sandbox_command_wall_ms": round(int(event["command_wall_ns"]) / 1e6, 3) if event.get("command_wall_ns") is not None else "",
            "sandbox_user_cpu_ms": round(int(event["command_user_cpu_us"]) / 1000, 3) if event.get("command_user_cpu_us") is not None else "",
            "sandbox_system_cpu_ms": round(int(event["command_system_cpu_us"]) / 1000, 3) if event.get("command_system_cpu_us") is not None else "",
            "result_bytes": event.get("result_bytes"), "arguments_sha256": event.get("arguments_sha256"),
            "run_key": event.get("run_key"),
        })
    _write_csv(out, BACKEND_FIELDS, rows)
