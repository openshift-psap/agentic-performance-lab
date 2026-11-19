"""multi-turn profiler for OpenClaw/OpenShell-

This module intentionally does not reuse the legacy A-E residual buckets.  It
joins official OpenClaw OTEL spans with the sim edge clock and
reports measurements separately from derived proxies.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any

from .measurement_tables import write_profile_tables


def _tags(span: dict[str, Any]) -> dict[str, Any]:
    return {str(item.get("key")): item.get("value") for item in span.get("tags", [])}


def _span_is_error(span: dict[str, Any]) -> bool:
    """Return true for a tool span that ended as an error/rejection.

    A tool wrapper can be short even when no sandbox command ran.  Keeping
    this classification at the span boundary prevents failed wrappers from
    being counted as OpenShell execution.
    """
    tags = _tags(span)
    status = str(tags.get("otel.status_code") or "").upper()
    error_tag = tags.get("error")
    return (
        status == "ERROR"
        or str(tags.get("openclaw.errorCategory") or "") not in ("", "None", "null")
        or error_tag is True
        or str(error_tag).lower() == "true"
    )


def _span_ms(span: dict[str, Any] | None) -> float | None:
    if not span:
        return None
    value = span.get("duration")
    return float(value) / 1000.0 if value is not None else None


def _start_ms(span: dict[str, Any] | None) -> float | None:
    if not span:
        return None
    value = span.get("startTime")
    return float(value) / 1000.0 if value is not None else None


def _end_ms(span: dict[str, Any] | None) -> float | None:
    start, duration = _start_ms(span), _span_ms(span)
    return start + duration if start is not None and duration is not None else None


def _number(value: Any) -> float | None:
    try:
        return float(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _first(tags: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if tags.get(key) not in (None, ""):
            return tags[key]
    return None


def _parent(span: dict[str, Any]) -> str | None:
    refs = span.get("references") or []
    return next((ref.get("spanID") for ref in refs if ref.get("refType") == "CHILD_OF"), None)


def _stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "mean": None, "p50": None, "p95": None, "min": None, "max": None}
    ordered = sorted(values)
    return {
        "n": len(values),
        "mean": round(statistics.mean(values), 3),
        "p50": round(statistics.median(values), 3),
        "p95": round(ordered[min(len(ordered) - 1, math.floor(0.95 * (len(ordered) - 1)))], 3),
        "min": round(min(values), 3),
        "max": round(max(values), 3),
    }


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_sandbox_lifecycle(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        value = _load_json(path)
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _load_jsonl(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _edge_index(edges: list[dict[str, Any]]) -> dict[tuple[str, int], dict[str, Any]]:
    return {
        (str(edge.get("trace_id") or ""), int(edge.get("turn_seq", -1))): edge
        for edge in edges
        if not edge.get("warmup") and not edge.get("phantom") and not edge.get("miss") and not edge.get("exhausted")
    }


def _load_driver_requests(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and value.get("t_send_wall_ns") is not None:
            rows.append(value)
    return rows


def _context_span(contexts: list[dict[str, Any]], model: dict[str, Any]) -> dict[str, Any] | None:
    model_id = model.get("spanID")
    for span in contexts:
        if model_id in {ref.get("spanID") for ref in span.get("references") or []}:
            return span
    # Older exporters may omit references. Only accept an unambiguous nearby span.
    model_start = _start_ms(model)
    nearby = [span for span in contexts if model_start is not None and _start_ms(span) is not None
              and abs(_start_ms(span) - model_start) < 1000]
    if len(nearby) == 1:
        return nearby[0]
    # OpenClaw 2026.9.x emits one run-level context span before the first
    # provider call, rather than one context span per internal model call.
    before_model = [
        span for span in contexts
        if model_start is not None and _start_ms(span) is not None and _end_ms(span) is not None
        and _start_ms(span) < model_start and _end_ms(span) <= model_start + 100
    ]
    return max(before_model, key=lambda span: _end_ms(span) or 0) if before_model else None


def _trace_rows(trace: dict[str, Any], edges: dict[tuple[str, str], dict[str, Any]]) -> list[dict[str, Any]]:
    trace_id = str(trace.get("traceID") or "")
    spans = trace.get("spans") or []
    models = sorted((span for span in spans if span.get("operationName") == "openclaw.model.call"),
                    key=lambda span: _start_ms(span) or 0)
    contexts = [span for span in spans if span.get("operationName") == "openclaw.context.assembled"]
    tools = sorted((span for span in spans if span.get("operationName") == "openclaw.tool.execution"),
                   key=lambda span: _start_ms(span) or 0)
    execs = sorted((span for span in spans if span.get("operationName") == "openclaw.exec" and not _span_is_error(span)),
                   key=lambda span: _start_ms(span) or 0)
    successful_tools = [span for span in tools if not _span_is_error(span)]
    failed_tools = [span for span in tools if _span_is_error(span)]
    # Newer OpenClaw builds report the sandbox duration directly on
    # openclaw.tool.execution and omit the former duplicate openclaw.exec span.
    if not execs:
        execs = successful_tools
    else:
        # Mixed exports can contain nested exec spans for some calls and only
        # successful wrapper spans for others.  Add only uncovered wrappers;
        # failed wrappers remain failures and never become sandbox execution.
        def covered(wrapper: dict[str, Any]) -> bool:
            wrapper_start = _start_ms(wrapper) or 0.0
            wrapper_end = _end_ms(wrapper) or wrapper_start
            return any(
                max(wrapper_start, _start_ms(item) or 0.0) <= min(wrapper_end, _end_ms(item) or 0.0) + 1.0
                for item in execs
            )
        execs.extend(wrapper for wrapper in successful_tools if not covered(wrapper))
        execs.sort(key=lambda span: _start_ms(span) or 0)
    harness = next((span for span in spans if span.get("operationName") == "openclaw.harness.run"), None)
    rows: list[dict[str, Any]] = []

    for turn, model in enumerate(models):
        tags = _tags(model)
        start = _start_ms(model)
        end = _end_ms(model)
        next_start = _start_ms(models[turn + 1]) if turn + 1 < len(models) else _end_ms(harness)
        context = _context_span(contexts, model) if turn == 0 else None
        context_tags = _tags(context or {})
        context_span_ms = _span_ms(context)
        context_start = _start_ms(context)
        context_end = _end_ms(context)
        run_start = _start_ms(harness)
        # The native context span is run-scoped in this OpenClaw build.  In
        # the collected traces it can also carry a clock-skew warning and
        # begin before the run by minutes.  Such a duration is not a valid
        # per-turn measurement.  The request-to-model boundary below is the
        # authoritative pre-LLM preparation window for this run.
        context_span_valid = (
            context is not None
            and context_span_ms is not None
            and context_start is not None
            and context_end is not None
            and start is not None
            and run_start is not None
            and context_start >= run_start - 100.0
            and context_end <= start + 100.0
            and not any("clock skew" in str(warning).lower() for warning in context.get("warnings") or [])
        )
        if not context_span_valid:
            context_span_ms = None
        following_tools = [tool for tool in tools if end is not None and (_start_ms(tool) or 0) >= end
                           and (next_start is None or (_start_ms(tool) or 0) < next_start)]
        first_tool = following_tools[0] if following_tools else None
        following_execs = [item for item in execs if end is not None and (_start_ms(item) or 0) >= end
                           and (next_start is None or (_start_ms(item) or 0) < next_start)]
        tool_dispatch_ms = _span_ms(first_tool)
        exec_ms = sum(_span_ms(item) or 0 for item in following_execs) if following_execs else None
        following_failures = [item for item in failed_tools if end is not None and (_start_ms(item) or 0) >= end
                              and (next_start is None or (_start_ms(item) or 0) < next_start)]
        tool_failure_ms = sum(_span_ms(item) or 0 for item in following_failures) if following_failures else None
        # Use the tool result for *this* transition. Reusing a tool from an
        # earlier turn would accidentally charge an intervening model response
        # to context assembly when the current turn has no tool result.
        previous_model_end = _end_ms(models[turn - 1]) if turn else None
        previous_tools = [
            span for span in tools
            if previous_model_end is not None and start is not None
            and _start_ms(span) is not None and _end_ms(span) is not None
            and previous_model_end <= _start_ms(span) <= _end_ms(span) <= start
        ]
        previous_tool = max(previous_tools, key=lambda span: _end_ms(span) or 0) if previous_tools else None
        if previous_tool is not None:
            previous_end = _end_ms(previous_tool)
            context_start_source = "prior_tool_result"
        elif turn == 0:
            previous_end = _start_ms(harness)
            context_start_source = "harness_run_start"
        else:
            previous_end = None
            context_start_source = "missing_prior_tool_result"
        context_assembly_window_ms = max(0.0, start - previous_end) if start is not None and previous_end is not None else None
        # Unlike model.call.start, this edge is recorded after the simulator
        # has read the complete request body. It includes request transfer.
        edge = edges.get((trace_id, str(model.get("spanID") or "")), {})
        endpoint_recv_wall_ns = edge.get("t_recv_wall_ns")
        body_read_wall_ns = edge.get("t_body_read_wall_ns")
        endpoint_recv_ms = int(endpoint_recv_wall_ns) / 1e6 if endpoint_recv_wall_ns is not None else None
        body_read_ms = int(body_read_wall_ns) / 1e6 if body_read_wall_ns is not None else None
        context_boundary_status = "missing_endpoint_edge"
        if previous_end is None:
            context_boundary_status = "missing_start_boundary"
        elif start is not None and endpoint_recv_ms is not None and body_read_ms is not None:
            context_boundary_status = (
                "ordered" if previous_end <= start <= endpoint_recv_ms <= body_read_ms
                and (end is None or body_read_ms <= end)
                else "inconsistent_order"
            )
        elif start is not None and body_read_ms is not None:
            context_boundary_status = (
                "ordered_missing_recv" if previous_end <= start <= body_read_ms
                and (end is None or body_read_ms <= end)
                else "inconsistent_order"
            )
        context_assembly_received_ms = None
        if context_boundary_status in ("ordered", "ordered_missing_recv"):
            elapsed = body_read_ms - previous_end
            if elapsed < 60_000:
                context_assembly_received_ms = round(elapsed, 3)
        # `openclaw.exec` is emitted after the LLM has requested a tool.  It
        # is therefore actual sandbox command execution, not pre-LLM sandbox
        # access/initialization.  Keep the cold command attribution separate
        # for compatibility, but do not label it as pre-LLM initialization.
        sandbox_cold_exec_ms = None
        if following_execs and following_execs[0].get("spanID") == (execs[0].get("spanID") if execs else None):
            sandbox_cold_exec_ms = _span_ms(following_execs[0])
        next_edge = (
            edges.get((trace_id, str(models[turn + 1].get("spanID") or "")), {})
            if turn + 1 < len(models) else {}
        )
        response_processing_full_ms = None
        if edge.get("t_first_emit_ns") is not None and next_edge.get("t_recv_ns") is not None:
            decode_ms = None
            if edge.get("t_end_emit_ns") is not None:
                decode_ms = (int(edge["t_end_emit_ns"]) - int(edge["t_first_emit_ns"])) / 1e6
            full_gap_ms = (int(next_edge["t_recv_ns"]) - int(edge["t_first_emit_ns"])) / 1e6
            response_processing_full_ms = max(
                0.0,
                full_gap_ms - (decode_ms or 0.0) - (exec_ms or 0.0) - (tool_failure_ms or 0.0),
            )
        model_ms = _span_ms(model)
        # Context assembly ends after the model call has begun (the endpoint
        # must read the request body). Use the union of those two intervals,
        # not their sum, so request transfer is counted only once.
        context_to_model_end_ms = (
            round(end - previous_end, 3)
            if context_boundary_status in ("ordered", "ordered_missing_recv")
            and previous_end is not None and end is not None and end >= previous_end
            else None
        )
        ttft_ms = _number(_first(tags, "openclaw.model_call.time_to_first_byte_ms"))
        response_processing = None
        response_processing_to_tool_start_ms = None
        pre_execution_gap_ms = None
        response_boundary = "unavailable"
        if end is not None:
            tool_start = _start_ms(first_tool)
            exec_start = _start_ms(following_execs[0]) if following_execs else None
            if tool_start is not None and exec_start is not None and exec_start >= tool_start:
                pre_execution_gap_ms = round(exec_start - tool_start, 3)
            if tool_start is not None:
                response_processing_to_tool_start_ms = max(0.0, tool_start - end)
            # Response processing includes the pre-execution tool handoff.
            # The wrapper span is diagnostic: it also contains OpenShell
            # execution and must not be added to this response interval.
            boundary = exec_start if exec_start is not None else tool_start if first_tool else next_start
            if boundary is not None:
                response_processing = max(0.0, boundary - end)
                response_boundary = (
                    "model_end_to_openshell_exec_start" if exec_start is not None
                    else "model_end_to_tool_start" if first_tool else "model_end_to_next_model"
                )
        rows.append({
            "trace_id": trace_id,
            "model_span_id": str(model.get("spanID") or ""),
            "turn": turn,
            "observation_unit": _first(tags, "openclaw.model_call.observation_unit") or "request",
            "model": _first(tags, "gen_ai.request.model", "openclaw.model"),
            "provider": _first(tags, "gen_ai.system", "gen_ai.provider.name", "openclaw.provider"),
            "api": tags.get("openclaw.api"),
            "transport": tags.get("openclaw.transport"),
            "model_call_ms": model_ms,
            "context_to_model_end_ms": context_to_model_end_ms,
            "ttft_ms": ttft_ms,
            "decode_proxy_ms": max(0.0, model_ms - ttft_ms) if model_ms is not None and ttft_ms is not None else None,
            "context_assembly_ms": context_assembly_received_ms,
            "context_span_direct_ms": context_span_ms if context_span_ms and context_span_ms > 0 else None,
            "context_span_present": context is not None,
            "context_span_duration_ms": context_span_ms,
            "context_measurement_status": context_boundary_status,
            "context_span_status": (
                "measured" if context_span_valid and context_span_ms and context_span_ms > 0
                else "invalid_clock_skew_or_run_scope" if context is not None
                else "missing"
            ),
            "context_tokens": _number(_first(context_tags, "openclaw.context.tokens")),
            "history_chars": _number(_first(context_tags, "openclaw.history.size", "openclaw.context.history_text_chars")),
            "prompt_chars": _number(_first(tags, "openclaw.model_call.prompt.total_chars", "openclaw.prompt.size")),
            "input_messages": _number(tags.get("openclaw.model_call.prompt.input_messages_count")),
            "input_message_chars": _number(tags.get("openclaw.model_call.prompt.input_messages_chars")),
            "system_prompt_chars": _number(tags.get("openclaw.model_call.prompt.system_prompt_chars")),
            "tool_definition_count": _number(tags.get("openclaw.model_call.prompt.tool_definitions_count")),
            "tool_definition_chars": _number(tags.get("openclaw.model_call.prompt.tool_definitions_chars")),
            "request_bytes": _number(tags.get("openclaw.model_call.request_bytes")),
            "response_bytes": _number(tags.get("openclaw.model_call.response_bytes")),
            "input_tokens": _number(_first(tags, "gen_ai.usage.input_tokens", "openclaw.model_call.usage.input_tokens")),
            "output_tokens": _number(_first(tags, "gen_ai.usage.output_tokens", "openclaw.model_call.usage.output_tokens")),
            "cache_read_tokens": _number(_first(tags, "gen_ai.usage.cache_read_input_tokens", "gen_ai.usage.cache_read")),
            "cache_write_tokens": _number(_first(tags, "gen_ai.usage.cache_creation_input_tokens", "gen_ai.usage.cache_write")),
            "response_processing_ms": response_processing,
            "response_processing_to_tool_start_ms": response_processing_to_tool_start_ms,
            "pre_execution_gap_ms": pre_execution_gap_ms,
            "response_processing_full_ms": response_processing_full_ms,
            "context_assembly_residual_ms": (
                max(0.0, response_processing_full_ms - response_processing)
                if response_processing_full_ms is not None and response_processing is not None
                else None
            ),
            "response_boundary": response_boundary,
            "tool_dispatch_ms": tool_dispatch_ms,
            "tool_dispatch_overhead_ms": max(0.0, tool_dispatch_ms - exec_ms)
            if tool_dispatch_ms is not None and exec_ms is not None and first_tool not in following_failures else 0.0,
            "tool_count": len(following_tools),
            "exec_ms": exec_ms,
            "tool_failure_ms": tool_failure_ms,
            "tool_failure_count": len(following_failures),
            "context_assembly_window_ms": context_assembly_window_ms,
            "context_assembly_received_ms": context_assembly_received_ms,
            "context_start_ms": previous_end,
            "context_start_source": context_start_source,
            "model_endpoint_recv_ms": endpoint_recv_ms,
            "model_endpoint_body_read_ms": body_read_ms,
            "context_boundary_status": context_boundary_status,
            "request_start_ms": run_start,
            "model_start_ms": start,
            "model_end_ms": end,
            "pre_model_sandbox_access_ms": None,
            "pre_model_sandbox_access_status": "not_instrumented",
            "sandbox_init_ms": None,
            "sandbox_reuse_ms": None,
            "sandbox_cold_exec_ms": sandbox_cold_exec_ms,
            "exec_count": len(following_execs),
            "harness_ms": _span_ms(harness),
            "mock_ttft_ms": _number(edge.get("ttft_measured_ms")),
            "mock_decode_ms": _number(edge.get("decode_measured_ms")),
            "mock_prefill_ms": _number(edge.get("prefill_ms")),
            "mock_output_tokens": _number(edge.get("output_tokens")),
            "mock_input_tokens": _number(edge.get("input_tokens")),
            "mock_turn_start_ns": edge.get("t_recv_ns"),
            "mock_turn_end_ns": edge.get("t_end_emit_ns"),
        })
    return rows


def _model_edge_index(
    edges: list[dict[str, Any]], raw_traces: list[dict[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    """Join mock edges to model spans by their recorded parent span ID.

    Older exports without parent IDs remain unpaired. Equal row counts do not
    establish identity when a dropped span and a retried span cancel out.
    """
    grouped: dict[str, list[dict[str, Any]]] = {}
    for edge in edges:
        trace_id = str(edge.get("trace_id") or "")
        if trace_id and int(edge.get("turn_seq", -1)) >= 0:
            grouped.setdefault(trace_id, []).append(edge)
    indexed: dict[tuple[str, str], dict[str, Any]] = {}
    for trace in raw_traces:
        trace_id = str(trace.get("traceID") or "")
        trace_edges = grouped.get(trace_id, [])
        ordered = sorted(
            trace_edges,
            key=lambda edge: (int(edge.get("turn_seq", 0)), int(edge.get("t_recv_ns") or 0)),
        )
        if any(edge.get("parent_span_id") for edge in ordered):
            for edge in ordered:
                parent = str(edge.get("parent_span_id") or "")
                if parent:
                    indexed[(trace_id, parent)] = edge
    return indexed


def build_rows(raw_traces: list[dict[str, Any]], edges: list[dict[str, Any]], driver_requests: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    edge_map = _model_edge_index(edges, raw_traces)
    rows: list[dict[str, Any]] = []
    for trace in raw_traces:
        rows.extend(_trace_rows(trace, edge_map))
    requests = driver_requests or []
    for trace_id in {row["trace_id"] for row in rows}:
        trace_edges = sorted(
            [edge for edge in edges if str(edge.get("trace_id") or "") == trace_id],
            key=lambda edge: int(edge.get("turn_seq", 0)),
        )
        if not trace_edges:
            continue
        first_recv = int(trace_edges[0].get("t_recv_wall_ns") or 0)
        candidates = [request for request in requests if int(request.get("t_send_wall_ns") or 0) <= first_recv]
        request_start = max(candidates, key=lambda request: int(request.get("t_send_wall_ns") or 0)).get("t_send_wall_ns") if candidates else None
        for index, edge in enumerate(trace_edges):
            previous_end = request_start if index == 0 else trace_edges[index - 1].get("t_end_emit_wall_ns")
            if previous_end is None or edge.get("t_recv_wall_ns") is None:
                continue
            parent = str(edge.get("parent_span_id") or "")
            match = next((row for row in rows if row["trace_id"] == trace_id and row.get("model_span_id") == parent), None) if parent else None
            if match:
                match["prompt_boundary_ms"] = round((int(edge["t_recv_wall_ns"]) - int(previous_end)) / 1e6, 3)
        ordered_rows = sorted(
            [row for row in rows if row["trace_id"] == trace_id],
            key=lambda row: int(row.get("turn", 0)),
        )
        for index, row in enumerate(ordered_rows):
            if index == 0:
                row["context_assembly_bucket_ms"] = row.get("context_assembly_received_ms")
                row["response_processing_bucket_ms"] = 0.0
                # The first model call has not produced a response yet.  Its
                # tool/exec spans belong to the transition into the next
                # turn, never to the initial user-request phase.
                row["tool_dispatch_overhead_bucket_ms"] = 0.0
                row["sandbox_execution_bucket_ms"] = 0.0
                row["tool_failure_bucket_ms"] = 0.0
            else:
                previous = ordered_rows[index - 1]
                row["context_assembly_bucket_ms"] = row.get("context_assembly_received_ms")
                # Keep the primary waterfall on the measured response span.
                # The full model-to-continuation gap includes dispatch/wait
                # time and is retained in the detailed efficiency fields.
                row["response_processing_bucket_ms"] = previous.get("response_processing_ms") or 0.0
                row["tool_dispatch_overhead_bucket_ms"] = 0.0
                row["sandbox_execution_bucket_ms"] = previous.get("exec_ms") or 0.0
                row["tool_failure_bucket_ms"] = previous.get("tool_failure_ms") or 0.0
            # The current OpenClaw export has no direct pre-LLM sandbox-init
            # span. Keep this separate from post-LLM command execution so a
            # first tool call cannot be mislabeled as sandbox initialization.
            row["sandbox_init_bucket_ms"] = row.get("sandbox_init_ms")
    return rows


def _collapse_logical_rows(
    rows: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    driver_requests: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Order every model row by the recorded session turn it replays.

    A client-boundary driver request carries a whole OpenClaw tool loop, so
    each model call inside that trace is a recorded turn of its own.  The mock
    edge for the call carries the session-level turn index; re-keying the rows
    from it keeps per-turn artifacts at recorded-turn grain instead of
    collapsing a multi-turn request into a single row.
    """
    by_trace: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_trace.setdefault(str(row["trace_id"]), []).append(row)

    edge_groups: dict[str, list[dict[str, Any]]] = {}
    for edge in edges:
        trace_id = str(edge.get("trace_id") or "")
        if trace_id and int(edge.get("turn_seq", -1)) >= 0:
            edge_groups.setdefault(trace_id, []).append(edge)

    ordered: list[tuple[int, float, dict[str, Any]]] = []
    for trace_id, trace_rows in by_trace.items():
        ordered_rows = sorted(trace_rows, key=lambda row: int(row.get("turn", 0)))
        trace_edges = sorted(
            edge_groups.get(trace_id, []),
            key=lambda edge: int(edge.get("t_recv_wall_ns") or 0),
        )
        request_start: Any = None
        if trace_edges:
            first_recv = int(trace_edges[0].get("t_recv_wall_ns") or 0)
            candidates = [
                request
                for request in driver_requests
                if int(request.get("t_send_wall_ns") or 0) <= first_recv
            ]
            request_start = (
                max(candidates, key=lambda request: int(request.get("t_send_wall_ns") or 0)).get("t_send_wall_ns")
                if candidates
                else None
            )
        for position, row in enumerate(ordered_rows):
            edge = trace_edges[position] if position < len(trace_edges) else None
            prior_end = request_start if position == 0 else None
            if position > 0 and position - 1 < len(trace_edges):
                prior_end = trace_edges[position - 1].get("t_end_emit_wall_ns")
            if edge is not None and prior_end is not None and edge.get("t_recv_wall_ns") is not None:
                row["prompt_boundary_ms"] = round(
                    (int(edge["t_recv_wall_ns"]) - int(prior_end)) / 1e6,
                    3,
                )
            elif position == 0:
                row["prompt_boundary_ms"] = row.get("context_assembly_window_ms")
            turn_seq = (
                int(edge.get("turn_seq"))
                if edge is not None and int(edge.get("turn_seq", -1)) >= 0
                else None
            )
            row["_logical_edge"] = edge
            ordered.append((
                turn_seq if turn_seq is not None else 1 << 30,
                float(row.get("model_start_ms") or 0.0),
                row,
            ))

    ordered.sort(key=lambda item: (item[0], item[1]))
    logical: list[dict[str, Any]] = []
    previous: dict[str, Any] | None = None
    previous_edge: dict[str, Any] | None = None
    for turn, (_, _, row) in enumerate(ordered):
        edge = row.pop("_logical_edge", None)
        if turn == 0:
            row["context_assembly_bucket_ms"] = row.get("context_assembly_received_ms")
            row["response_processing_bucket_ms"] = 0.0
            row["tool_dispatch_overhead_bucket_ms"] = 0.0
            row["sandbox_execution_bucket_ms"] = 0.0
            row["tool_failure_bucket_ms"] = 0.0
        else:
            if edge and previous_edge and edge.get("t_recv_wall_ns") is not None and previous_edge.get("t_end_emit_wall_ns") is not None:
                row["prompt_boundary_ms"] = round(max(
                    0.0,
                    (int(edge["t_recv_wall_ns"]) - int(previous_edge["t_end_emit_wall_ns"])) / 1e6,
                ), 3)
            # The primary waterfall reports the measured response-processing
            # span only.  The larger model-to-continuation interval contains
            # OpenClaw/OpenShell dispatch waiting and is retained in the
            # detailed efficiency artifact, not mislabeled as response work.
            row["response_processing_bucket_ms"] = previous.get("response_processing_ms") or 0.0 if previous else 0.0
            row["tool_dispatch_overhead_bucket_ms"] = 0.0
            row["sandbox_execution_bucket_ms"] = previous.get("exec_ms") or 0.0 if previous else 0.0
            row["tool_failure_bucket_ms"] = previous.get("tool_failure_ms") or 0.0 if previous else 0.0
            row["context_assembly_bucket_ms"] = row.get("context_assembly_received_ms")
        row["turn"] = turn
        row["sandbox_init_bucket_ms"] = row.get("sandbox_init_ms")
        logical.append(row)
        previous = row
        previous_edge = edge
    return logical


def _timing_segments_for_logical_rows(
    segments: list[dict[str, Any]],
    logical_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Keep the baseline timing artifact at the same logical-turn grain."""
    by_trace: dict[str, dict[str, Any]] = {}
    for segment in segments:
        by_trace.setdefault(str(segment.get("trace_id") or ""), segment)
    collapsed: list[dict[str, Any]] = []
    for row in logical_rows:
        segment = dict(by_trace.get(str(row.get("trace_id") or ""), {}))
        segment.update(
            {
                "trace_id": row.get("trace_id"),
                "turn_idx": row.get("turn"),
                "sandbox_init_ms": row.get("sandbox_init_bucket_ms") or 0.0,
                "sandbox_exec_ms": row.get("sandbox_execution_bucket_ms") or 0.0,
                "context_assembly_ms": row.get("context_assembly_bucket_ms") or 0.0,
                "response_processing_ms": row.get("response_processing_bucket_ms") or 0.0,
                "post_model_response_processing_ms": row.get("response_processing_full_ms"),
                "post_model_sandbox_exec_ms": row.get("exec_ms"),
                "post_model_tool_failure_ms": row.get("tool_failure_ms"),
                "model_call_total_ms": row.get("model_call_ms"),
                "harness_total_ms": row.get("harness_ms"),
                "input_tokens": row.get("input_tokens"),
                "output_tokens": row.get("output_tokens"),
                "prompt_chars": row.get("prompt_chars"),
                "prompt_tokens_est": row.get("prompt_tokens_est"),
                "system_prompt_chars": row.get("system_prompt_chars"),
            }
        )
        collapsed.append(segment)
    return collapsed


def profile(*, traces: Path, mock_edges: Path | None, out: Path, prom: Path | None = None, driver_requests: Path | None = None, sandbox_lifecycle: Path | None = None) -> dict[str, Any]:
    raw = _load_json(traces)
    if isinstance(raw, dict):
        raw = raw.get("data") or raw.get("traces") or []
    edges = _load_jsonl(mock_edges)
    driver_requests_rows = _load_driver_requests(driver_requests)
    measured_window = None
    if driver_requests:
        try:
            driver_summary = _load_json(driver_requests.parent / "summary.json")
            measured_window = (float(driver_summary["start_unix"]), float(driver_summary["end_unix"]))
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            pass
    edge_trace_ids = {str(edge.get("trace_id") or "") for edge in edges if edge.get("trace_id")}
    if edge_trace_ids:
        raw = [trace for trace in raw if str(trace.get("traceID") or "") in edge_trace_ids]
    built_rows = build_rows(raw, edges, driver_requests_rows)
    # Client-boundary replay emits one raw Jaeger trace per client message and
    # the harness runs the tool loop inside it, so a multi-trace session still
    # has one model row per recorded turn.  Re-key those rows from the mock
    # edges into recorded-turn order.
    rows = (
        _collapse_logical_rows(built_rows, edges, driver_requests_rows)
        if len(raw) > 1
        else built_rows
    )
    lifecycle_path = sandbox_lifecycle or (traces.parent / "sandbox_lifecycle.json")
    lifecycle = _load_sandbox_lifecycle(lifecycle_path)
    lifecycle_init_ms = float(lifecycle.get("sandbox_init_ms") or 0.0)
    lifecycle_reuse_raw = lifecycle.get("sandbox_reuse_ms")
    lifecycle_reuse_ms = (
        float(lifecycle_reuse_raw)
        if lifecycle_reuse_raw is not None
        else None
    )
    lifecycle_request_to_action_raw = lifecycle.get("sandbox_request_to_first_action_ms")
    lifecycle_request_to_action_ms = (
        float(lifecycle_request_to_action_raw)
        if lifecycle_request_to_action_raw is not None
        else None
    )
    lifecycle_status = str(lifecycle.get("status") or "not_instrumented")
    if rows and lifecycle_init_ms > 0:
        rows[0]["sandbox_init_ms"] = lifecycle_init_ms
        rows[0]["sandbox_init_bucket_ms"] = lifecycle_init_ms
        rows[0]["sandbox_init_status"] = lifecycle_status
    elif rows:
        rows[0]["sandbox_init_status"] = lifecycle_status
    if rows:
        rows[0]["sandbox_reuse_ms"] = lifecycle_reuse_ms
        rows[0]["sandbox_reuse_status"] = (
            str(lifecycle.get("sandbox_reuse_measurement") or "observed")
            if lifecycle_reuse_ms is not None
            else "not_separately_observed"
        )
        rows[0]["sandbox_request_to_first_action_ms"] = lifecycle_request_to_action_ms
        rows[0]["sandbox_request_to_first_action_status"] = (
            str(lifecycle.get("sandbox_request_to_first_action_measurement") or "observed")
            if lifecycle_request_to_action_ms is not None
            else str(lifecycle.get("sandbox_request_to_first_action_measurement") or "not_measured")
        )
    out.mkdir(parents=True, exist_ok=True)
    (out / "per_turn.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    if rows:
        fieldnames = list(dict.fromkeys(key for row in rows for key in row))
        with (out / "per_turn.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
    # Keep the established profiler-v2 contract: per-tool artifacts live next
    # to per-turn artifacts and are derived from the same filtered trace set.
    from .jaeger_export import compute_timing_segments, extract_per_turn
    _, tool_rows, _ = extract_per_turn(raw)
    (out / "per_tool.json").write_text(json.dumps(tool_rows, indent=2), encoding="utf-8")
    if tool_rows:
        tool_fields = list(dict.fromkeys(key for row in tool_rows for key in row))
        with (out / "per_tool.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=tool_fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(tool_rows)
    # Refresh the baseline-compatible timing artifact from the same raw span
    # set.  Keeping this in the profiling step prevents stale timing buckets
    # from being mixed with newly generated per-turn tables.
    timing_segments = _timing_segments_for_logical_rows(
        compute_timing_segments(raw),
        rows,
    )
    (traces.parent / "timing_segments.json").write_text(json.dumps(timing_segments, indent=2), encoding="utf-8")
    (out / "timing_segments.json").write_text(json.dumps(timing_segments, indent=2), encoding="utf-8")
    direct_keys = ["context_assembly_ms", "model_call_ms", "context_to_model_end_ms", "ttft_ms", "response_processing_ms", "tool_dispatch_ms", "exec_ms", "tool_failure_ms"]
    summary = {
        "schema": "openclaw-profiler-v2",
        "turns": len(rows),
        "traces": len({row["trace_id"] for row in rows}),
        "observation_units": sorted({row["observation_unit"] for row in rows}),
        "direct_measurements": {key: _stats([row[key] for row in rows if row.get(key) is not None]) for key in direct_keys},
        "baseline_buckets": {
            "context_assembly_ms": _stats([row["context_assembly_bucket_ms"] for row in rows if row.get("context_assembly_bucket_ms") is not None]),
            "context_assembly_complete_request_ms": _stats([row["context_assembly_received_ms"] for row in rows if row.get("context_assembly_received_ms") is not None]),
            "context_assembly_legacy_model_start_ms": _stats([row["context_assembly_window_ms"] for row in rows if row.get("context_assembly_window_ms") is not None]),
            "context_span_direct_ms": _stats([row["context_span_direct_ms"] for row in rows if row.get("context_span_direct_ms") is not None]),
            "prompt_boundary_ms": _stats([row["prompt_boundary_ms"] for row in rows if row.get("prompt_boundary_ms") is not None]),
            "response_processing_ms": _stats([row["response_processing_full_ms"] for row in rows if row.get("response_processing_full_ms") is not None]),
            "sandbox_execution_ms": _stats([row["exec_ms"] for row in rows if row.get("exec_ms") is not None]),
            "tool_failure_ms": _stats([row["tool_failure_ms"] for row in rows if row.get("tool_failure_ms") is not None]),
            "sandbox_initialization_ms": _stats([row["sandbox_init_ms"] for row in rows if row.get("sandbox_init_ms") is not None]),
            "sandbox_cold_exec_ms": _stats([row["sandbox_cold_exec_ms"] for row in rows if row.get("sandbox_cold_exec_ms") is not None]),
        },
        "context_evidence": {
            "input_tokens": _stats([row["input_tokens"] for row in rows if row.get("input_tokens") is not None]),
            "prompt_chars": _stats([row["prompt_chars"] for row in rows if row.get("prompt_chars") is not None]),
            "system_prompt_chars": _stats([row["system_prompt_chars"] for row in rows if row.get("system_prompt_chars") is not None]),
            "tool_definition_chars": _stats([row["tool_definition_chars"] for row in rows if row.get("tool_definition_chars") is not None]),
        },
        "coverage": {key: sum(row.get(key) is not None for row in rows) for key in direct_keys},
        "definitions": {
            "context_assembly_ms": "Per-turn interval from this transition's tool-result span end (or the first harness-run start) to complete request-body receipt at the model endpoint. Missing or unordered boundaries have no value; this includes agent preparation and request transfer, not only prompt-building CPU time.",
            "context_assembly_received_ms": "Same per-turn complete-request boundary as context_assembly_ms; requires an OpenClaw model span joined to the mock edge by parent span ID.",
            "context_span_direct_ms": "Direct openclaw.context.assembled span only when its timestamps are within the run and have no clock-skew warning; otherwise it is excluded.",
            "model_call_ms": "Supporting provider span; excluded from the baseline overhead buckets.",
            "context_to_model_end_ms": "Elapsed time from the prior tool-result or initial harness boundary through model-call completion. This is the union of context assembly and model call, without double-counting request transfer.",
            "response_processing_ms": "OpenClaw model-call end to the start of the first measured OpenShell execution span; includes the pre-execution handoff. If no execution span exists, ends at the tool wrapper or next model call. This is elapsed time, not attributed CPU time.",
            "response_processing_to_tool_start_ms": "OpenClaw model-call end to the start of the outer tool wrapper; a narrower diagnostic interval.",
            "pre_execution_gap_ms": "Outer tool wrapper start to the start of the measured OpenShell execution span; included in response_processing_ms when both spans exist.",
            "tool_dispatch_ms": "Legacy name for the complete outer OpenClaw tool-call span; overlaps response_processing_ms and sandbox_execution_ms and is called tool_call_total_ms in the new latency tables.",
            "response_processing_full_ms": "First streamed model event to next prompt minus mock decode and sandbox execution; baseline-style agent processing bucket.",
            "sandbox_execution_ms": "Sum of openclaw.exec spans, falling back to openclaw.tool.execution when the newer schema omits openclaw.exec.",
            "sandbox_initialization_ms": "Measured prewarm request through OpenClaw and OpenShell using a harmless exec before the first replay turn; pod create-to-ready remains separately recorded as cold-start evidence.",
            "sandbox_cold_exec_ms": "Compatibility field containing the first actual openclaw.exec command duration; it is command execution and must not be interpreted as cold-start or warm-pool initialization.",
            "context_per_turn_latency": "Measured from OpenClaw's tool-result/harness boundary through the mock model endpoint's complete request receipt. The native OpenClaw context.assembled event is run-level only in the deployed image.",
            "prompt_boundary_ms": "Driver send to the first mock prompt, then previous mock turn end to the next mock prompt; includes all OpenClaw work in that observable boundary.",
        },
        "tables": write_profile_tables(rows, tool_rows, prom, out / "tables", measured_window),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out / "PROFILE.md").write_text(
        "# OpenClaw Profiler v2\n\n"
        "This report separates direct OpenTelemetry measurements from derived proxies.\n\n"
        + "\n".join(f"- `{key}`: {value}" for key, value in summary["definitions"].items())
        + "\n",
        encoding="utf-8",
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evidence-first OpenClaw multi-turn profiler")
    parser.add_argument("--traces", required=True, type=Path)
    parser.add_argument("--mock-edges", type=Path)
    parser.add_argument("--prom", type=Path)
    parser.add_argument("--driver-requests", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(profile(traces=args.traces, mock_edges=args.mock_edges, prom=args.prom, driver_requests=args.driver_requests, out=args.out), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
