import csv
import json

import pytest

from trace_replay_sim.output_validation import OutputValidationError, validate_trace_output


def write_csv(path, fields, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def valid_trace(tmp_path):
    trace = tmp_path / "trace"
    tables = trace / "data/shell/profile-v2/tables"
    cgroup = trace / "data/shell/prometheus/cgroup"
    backend = trace / "data/backend"
    for path in (tables, cgroup, backend, trace / "data/shell/traces"):
        path.mkdir(parents=True)
    (trace / "data/shell/traces/raw_traces.json").write_text('[{"traceID":"t","spans":[]}]')
    (trace / "data/shell/mock_edges.jsonl").write_text('{"session_id":"s"}\n')
    (trace / "data/shell/sandbox_lifecycle.json").write_text(json.dumps({
        "status": "ready", "prewarm_requested": True, "prewarm_status": "measured",
        "provisioning_mode": "warm_reused", "sandbox_init_ms": 0.0,
        "sandbox_request_to_first_action_ms": 200.0, "readiness_evidence": "prewarm",
    }))
    (trace / "data/shell/prometheus/collect_manifest.json").write_text('{}')
    for name in ("openclaw_gateway.csv", "openshell_sandbox.csv"):
        (cgroup / name).write_text("epoch_ns,cpu_usage_usec,memory_bytes\n1000000000,1,2\n1100000000,2,3\n")
    (trace / "data/shell/profile-v2/per_turn.json").write_text('[{"turn":0},{"turn":1}]')
    (trace / "data/shell/profile-v2/per_tool.json").write_text('[{"tool":"a"}]')
    write_csv(tables / "per_turn_latency.csv", ["turn", "model_call_ms"], [{"turn": 0, "model_call_ms": 10}, {"turn": 1, "model_call_ms": 11}])
    fields = ["turn", "sandbox_init_ms", "prompt_boundary_ms", "response_processing_ms", "tool_dispatch_overhead_ms", "openshell_exec_ms", "tool_failure_ms"]
    write_csv(tables / "per_turn_efficiency.csv", fields, [
        dict(zip(fields, (0, 2, 5, 0, 0, 0, 0))),
        dict(zip(fields, (1, 0, 3, 4, 1, 6, 0))),
    ])
    write_csv(tables / "per_tool_latency.csv", ["tool_name", "duration_ms"], [{"tool_name": "a", "duration_ms": 6}])
    write_csv(tables / "stage_latency_summary.csv", ["stage", "n"], [{"stage": "model_call_ms", "n": 2}])
    write_csv(tables / "resource_samples.csv", ["component", "cpu_cores"], [{"component": "openclaw_gateway", "cpu_cores": 0.01}])
    write_csv(tables / "resource_summary.csv", ["component", "samples"], [{"component": "openclaw_gateway", "samples": 2}])
    write_csv(backend / "backend_calls.csv", ["tool", "turn", "call"], [{"tool": "a", "turn": 0, "call": 0}])
    return trace


def test_output_validation_accepts_complete_tables(tmp_path):
    evidence = validate_trace_output(valid_trace(tmp_path), expected_turns=2, expected_tool_calls=1, backend_dir_name="backend")
    assert evidence["passed"] is True
    assert evidence["tables"]["data/shell/profile-v2/tables/per_turn_latency.csv"] == 2


def test_output_validation_rejects_first_turn_response_processing(tmp_path):
    trace = valid_trace(tmp_path)
    path = trace / "data/shell/profile-v2/tables/per_turn_efficiency.csv"
    path.write_text(path.read_text().replace("0,2,5,0,0,0,0", "0,2,5,9,0,0,0", 1))
    with pytest.raises(OutputValidationError, match="turn 0 must not contain response_processing_ms"):
        validate_trace_output(trace, expected_turns=2, expected_tool_calls=1, backend_dir_name="backend")


def test_output_validation_rejects_missing_resource_table(tmp_path):
    trace = valid_trace(tmp_path)
    (trace / "data/shell/profile-v2/tables/resource_samples.csv").unlink()
    with pytest.raises(OutputValidationError, match="resource_samples.csv"):
        validate_trace_output(trace, expected_turns=2, expected_tool_calls=1, backend_dir_name="backend")
