from trace_replay_sim.profiler_v2 import _collapse_logical_rows, _model_edge_index, _trace_rows


def test_context_ends_after_model_endpoint_reads_complete_body():
    trace = {"traceID": "trace-a", "spans": [
        {"spanID": "run", "operationName": "openclaw.harness.run", "startTime": 1_000_000, "duration": 500_000},
        {"spanID": "model", "operationName": "openclaw.model.call", "startTime": 1_100_000, "duration": 200_000},
    ]}
    edges = {("trace-a", "model"): {"t_body_read_wall_ns": 1_150_000_000}}
    row = _trace_rows(trace, edges)[0]
    assert row["context_assembly_window_ms"] == 100.0
    assert row["context_assembly_received_ms"] == 150.0
    assert row["context_assembly_ms"] == 150.0
    assert _collapse_logical_rows([row], [], [])[0]["context_assembly_bucket_ms"] == 150.0


def test_context_boundaries_preserve_ordered_events_and_reject_reversed_endpoint_clock():
    trace = {"traceID": "trace-a", "spans": [
        {"spanID": "run", "operationName": "openclaw.harness.run", "startTime": 1_000_000, "duration": 500_000},
        {"spanID": "model", "operationName": "openclaw.model.call", "startTime": 1_100_000, "duration": 200_000},
    ]}
    ordered = _trace_rows(trace, {("trace-a", "model"): {
        "t_recv_wall_ns": 1_130_000_000, "t_body_read_wall_ns": 1_150_000_000,
    }})[0]
    assert ordered["context_start_source"] == "harness_run_start"
    assert ordered["context_start_ms"] == 1000.0
    assert ordered["model_endpoint_recv_ms"] == 1130.0
    assert ordered["model_endpoint_body_read_ms"] == 1150.0
    assert ordered["context_boundary_status"] == "ordered"
    assert ordered["context_assembly_received_ms"] == 150.0

    reversed_edge = _trace_rows(trace, {("trace-a", "model"): {
        "t_recv_wall_ns": 1_090_000_000, "t_body_read_wall_ns": 1_150_000_000,
    }})[0]
    assert reversed_edge["context_boundary_status"] == "inconsistent_order"
    assert reversed_edge["context_assembly_received_ms"] is None


def test_context_uses_only_this_turns_tool_result_and_requires_complete_request():
    trace = {"traceID": "trace-a", "spans": [
        {"spanID": "run", "operationName": "openclaw.harness.run", "startTime": 1_000_000, "duration": 2_000_000},
        {"spanID": "model-0", "operationName": "openclaw.model.call", "startTime": 1_100_000, "duration": 100_000},
        {"spanID": "tool", "operationName": "openclaw.tool.execution", "startTime": 1_210_000, "duration": 300_000},
        {"spanID": "model-1", "operationName": "openclaw.model.call", "startTime": 1_520_000, "duration": 100_000},
        {"spanID": "model-2", "operationName": "openclaw.model.call", "startTime": 1_700_000, "duration": 100_000},
    ]}
    edges = {
        ("trace-a", "model-0"): {"t_recv_wall_ns": 1_110_000_000, "t_body_read_wall_ns": 1_120_000_000},
        ("trace-a", "model-1"): {"t_recv_wall_ns": 1_530_000_000, "t_body_read_wall_ns": 1_540_000_000},
        ("trace-a", "model-2"): {"t_recv_wall_ns": 1_710_000_000, "t_body_read_wall_ns": 1_720_000_000},
    }
    rows = _trace_rows(trace, edges)
    assert rows[1]["context_start_source"] == "prior_tool_result"
    assert rows[1]["context_assembly_ms"] == 30.0
    assert rows[2]["context_start_source"] == "missing_prior_tool_result"
    assert rows[2]["context_assembly_ms"] is None
    assert rows[2]["context_boundary_status"] == "missing_start_boundary"


def test_context_rejects_request_receipt_after_model_call_ended():
    trace = {"traceID": "trace-a", "spans": [
        {"spanID": "run", "operationName": "openclaw.harness.run", "startTime": 1_000_000, "duration": 500_000},
        {"spanID": "model", "operationName": "openclaw.model.call", "startTime": 1_100_000, "duration": 20_000},
    ]}
    row = _trace_rows(trace, {("trace-a", "model"): {
        "t_recv_wall_ns": 1_110_000_000, "t_body_read_wall_ns": 1_150_000_000,
    }})[0]
    assert row["context_boundary_status"] == "inconsistent_order"
    assert row["context_assembly_ms"] is None


def test_response_processing_runs_until_measured_tool_execution_starts():
    trace = {"traceID": "trace-tool", "spans": [
        {"spanID": "run", "operationName": "openclaw.harness.run", "startTime": 1_000_000, "duration": 3_000_000},
        {"spanID": "model", "operationName": "openclaw.model.call", "startTime": 1_100_000, "duration": 100_000},
        {"spanID": "tool", "operationName": "openclaw.tool.execution", "startTime": 1_210_000, "duration": 1_606_000},
        {"spanID": "exec", "operationName": "openclaw.exec", "startTime": 2_482_000, "duration": 333_000},
    ]}
    row = _trace_rows(trace, {})[0]
    assert row["response_processing_to_tool_start_ms"] == 10.0
    assert row["pre_execution_gap_ms"] == 1272.0
    assert row["response_processing_ms"] == 1282.0
    assert row["exec_ms"] == 333.0
    assert row["tool_dispatch_ms"] == 1606.0
    assert row["response_boundary"] == "model_end_to_openshell_exec_start"


def test_mock_edges_join_by_parent_even_if_a_model_span_is_missing():
    traces = [{"traceID": "trace-a", "spans": [
        {"spanID": "first", "operationName": "openclaw.model.call", "startTime": 1_000_000},
        {"spanID": "third", "operationName": "openclaw.model.call", "startTime": 3_000_000},
    ]}]
    edges = [
        {"trace_id": "trace-a", "turn_seq": 0, "parent_span_id": "first"},
        {"trace_id": "trace-a", "turn_seq": 1, "parent_span_id": "missing"},
        {"trace_id": "trace-a", "turn_seq": 2, "parent_span_id": "third"},
    ]
    indexed = _model_edge_index(edges, traces)
    assert indexed[("trace-a", "third")]["turn_seq"] == 2
    assert ("trace-a", "1") not in indexed


def test_missing_parent_ids_are_not_joined():
    traces = [{"traceID": "trace-a", "spans": [
        {"spanID": "first", "operationName": "openclaw.model.call", "startTime": 1_000_000},
        {"spanID": "third", "operationName": "openclaw.model.call", "startTime": 3_000_000},
    ]}]
    edges = [{"trace_id": "trace-a", "turn_seq": index} for index in range(3)]
    assert _model_edge_index(edges, traces) == {}


def test_equal_edge_and_span_counts_without_parent_ids_are_not_joined():
    traces = [{"traceID": "trace-a", "spans": [
        {"spanID": "first", "operationName": "openclaw.model.call", "startTime": 100},
        {"spanID": "retry", "operationName": "openclaw.model.call", "startTime": 200},
    ]}]
    edges = [
        {"trace_id": "trace-a", "turn_seq": 0},
        {"trace_id": "trace-a", "turn_seq": 1},
    ]
    assert _model_edge_index(edges, traces) == {}


def test_client_turn_rows_keep_every_model_call_and_rekey_from_mock_edges():
    rows = [
        {
            "trace_id": "trace-a",
            "turn": 0,
            "model_start_ms": 100.0,
            "exec_ms": 200.0,
            "response_processing_ms": 7.0,
            "response_processing_full_ms": 700.0,
        },
        {
            "trace_id": "trace-a",
            "turn": 1,
            "model_start_ms": 150.0,
            "exec_ms": None,
            "response_processing_ms": 8.0,
        },
        {
            "trace_id": "trace-b",
            "turn": 0,
            "model_start_ms": 30.0,
            "exec_ms": 180.0,
            "response_processing_ms": 6.0,
            "response_processing_full_ms": 650.0,
        },
    ]
    edges = [
        {"trace_id": "trace-a", "turn_seq": 0, "t_recv_wall_ns": 1_000_000_000, "t_end_emit_wall_ns": 1_100_000_000},
        {"trace_id": "trace-a", "turn_seq": 1, "t_recv_wall_ns": 1_150_000_000, "t_end_emit_wall_ns": 1_200_000_000},
        {"trace_id": "trace-b", "turn_seq": 2, "t_recv_wall_ns": 1_250_000_000, "t_end_emit_wall_ns": 1_300_000_000},
    ]
    requests = [
        {"t_send_wall_ns": 900_000_000},
        {"t_send_wall_ns": 1_220_000_000},
    ]

    result = _collapse_logical_rows(rows, edges, requests)

    assert len(result) == 3
    assert [row["turn"] for row in result] == [0, 1, 2]
    assert result[0]["sandbox_execution_bucket_ms"] == 0.0
    assert result[0]["response_processing_bucket_ms"] == 0.0
    assert result[0]["context_assembly_bucket_ms"] is None
    assert result[1]["sandbox_execution_bucket_ms"] == 200.0
    assert result[1]["response_processing_bucket_ms"] == 7.0
    # A broad prompt boundary does not substitute for the complete request
    # receipt measurement, even when it could be made into a residual.
    assert result[1]["context_assembly_bucket_ms"] is None
    assert result[2]["context_assembly_bucket_ms"] is None
    assert result[2]["response_processing_bucket_ms"] == 8.0


def test_cross_trace_context_is_not_inferred_from_response_and_tool():
    rows = [
        {"trace_id": "first", "turn": 0, "model_start_ms": 0.0,
         "response_processing_ms": 40.0, "exec_ms": 500.0},
        {"trace_id": "second", "turn": 0, "model_start_ms": 1.0},
    ]
    edges = [
        {"trace_id": "first", "turn_seq": 0, "t_recv_wall_ns": 1_000_000_000,
         "t_end_emit_wall_ns": 1_100_000_000},
        {"trace_id": "second", "turn_seq": 1, "t_recv_wall_ns": 1_640_000_000,
         "t_end_emit_wall_ns": 1_700_000_000},
    ]
    requests = [{"t_send_wall_ns": 900_000_000}, {"t_send_wall_ns": 1_600_000_000}]

    result = _collapse_logical_rows(rows, edges, requests)

    assert result[1]["prompt_boundary_ms"] == 540.0
    assert result[1]["response_processing_bucket_ms"] == 40.0
    assert result[1]["sandbox_execution_bucket_ms"] == 500.0
    assert result[1]["context_assembly_bucket_ms"] is None
