import csv

from trace_replay_sim.measurement_tables import write_profile_tables


def test_resource_summary_uses_only_samples_inside_driver_window(tmp_path):
    cgroup = tmp_path / "prom/cgroup"
    cgroup.mkdir(parents=True)
    (cgroup / "openclaw_gateway.csv").write_text(
        "epoch_ns,cpu_usage_usec,memory_bytes,memory_working_set_bytes\n"
        "1000000000,0,1048576,524288\n"
        "2000000000,1000000,2097152,1048576\n"
        "3000000000,3000000,3145728,1572864\n"
        "4000000000,4000000,99999999,99999999\n",
        encoding="utf-8",
    )
    out = tmp_path / "tables"
    files = write_profile_tables([], [], tmp_path / "prom", out, (2.0, 3.0))
    assert "resource_summary.csv" in files
    rows = list(csv.DictReader((out / "resource_samples.csv").open(encoding="utf-8")))
    assert [row["within_measured_window"] for row in rows] == ["False", "True", "True", "False"]
    assert rows[1]["cpu_cores"] == ""  # Previous counter sample was outside the measured window.
    assert float(rows[2]["cpu_cores"]) == 2.0
    summary = list(csv.DictReader((out / "resource_summary.csv").open(encoding="utf-8")))[0]
    assert summary["samples"] == "2"
    assert summary["cpu_samples"] == "1"
    assert summary["cpu_core_seconds"] == "2.0"
    assert summary["memory_peak_mib"] == "3.0"
    assert not list(tmp_path.rglob("*.png"))


def test_latency_table_separates_response_from_overlapping_tool_wrapper(tmp_path):
    out = tmp_path / "tables"
    write_profile_tables([{
        "trace_id": "trace-tool", "turn": 0,
        "response_processing_ms": 1282.0,
        "response_processing_to_tool_start_ms": 10.0,
        "pre_execution_gap_ms": 1272.0,
        "tool_dispatch_ms": 1606.0,
        "exec_ms": 333.0,
        "input_messages": 47,
        "input_message_chars": 105490,
        "prompt_chars": 141000,
        "request_bytes": 120318,
    }], [], None, out)
    row = list(csv.DictReader((out / "per_turn_latency.csv").open(encoding="utf-8")))[0]
    assert row["response_processing_ms"] == "1282.0"
    assert row["pre_execution_gap_ms"] == "1272.0"
    assert row["tool_call_total_ms"] == "1606.0"
    assert row["openshell_execution_ms"] == "333.0"
    assert row["input_messages"] == "47"
    assert row["input_message_chars"] == "105490"
    assert row["prompt_chars"] == "141000"
    assert row["request_bytes"] == "120318"
    assert "tool_dispatch_ms" not in row


def test_context_boundary_table_exposes_timestamps_and_flags_bad_order(tmp_path):
    out = tmp_path / "tables"
    write_profile_tables([{
        "trace_id": "good", "turn": 1, "context_start_source": "prior_tool_span_end",
        "context_start_ms": 1000.0, "model_start_ms": 1009.0,
        "model_end_ms": 1050.0, "model_call_ms": 41.0,
        "model_endpoint_recv_ms": 1019.0, "model_endpoint_body_read_ms": 1020.0,
        "context_boundary_status": "ordered", "context_assembly_received_ms": 20.0,
    }, {
        "trace_id": "bad", "turn": 2, "context_start_source": "prior_tool_span_end",
        "context_start_ms": 2000.0, "model_start_ms": 2010.0,
        "model_endpoint_recv_ms": 2009.0, "model_endpoint_body_read_ms": 2020.0,
        "context_boundary_status": "inconsistent_order",
    }], [], None, out)
    rows = list(csv.DictReader((out / "context_boundary_events.csv").open(encoding="utf-8")))
    assert rows[0]["agent_pre_model_ms"] == "9.0"
    assert rows[0]["model_request_arrival_ms"] == "10.0"
    assert rows[0]["model_request_body_read_ms"] == "1.0"
    assert rows[0]["total_pre_model_ms"] == "20.0"
    assert rows[1]["boundary_status"] == "inconsistent_order"
    assert rows[1]["total_pre_model_ms"] == ""

    latency = list(csv.DictReader((out / "per_turn_latency.csv").open(encoding="utf-8")))
    assert latency[0]["context_assembly_ms"] == "20.0"
    assert latency[0]["context_to_model_end_ms"] == "50.0"  # 20 + 41 - 11 ms overlap
    assert latency[0]["context_boundary_status"] == "ordered"
    assert latency[1]["context_assembly_ms"] == ""
    assert latency[1]["context_to_model_end_ms"] == ""


def test_context_latency_does_not_fall_back_to_incomplete_model_start_gap(tmp_path):
    out = tmp_path / "tables"
    write_profile_tables([{
        "trace_id": "missing-endpoint", "turn": 2,
        "context_assembly_window_ms": 17.0,
        "context_boundary_status": "missing_endpoint_edge",
    }], [], None, out)
    row = list(csv.DictReader((out / "per_turn_latency.csv").open(encoding="utf-8")))[0]
    assert row["context_assembly_ms"] == ""
    assert row["context_boundary_status"] == "missing_endpoint_edge"
