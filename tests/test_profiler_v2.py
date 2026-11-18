from trace_replay_sim.profiler_v2 import _collapse_logical_rows, _model_edge_index


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
    assert result[0]["context_assembly_bucket_ms"] == 100.0
    assert result[1]["sandbox_execution_bucket_ms"] == 200.0
    assert result[1]["response_processing_bucket_ms"] == 7.0
    # The broad 50 ms boundary cannot also contain 200 ms of execution.
    # In inconsistent legacy rows, clamp the residual instead of adding
    # the entire boundary to response and execution a second time.
    assert result[1]["context_assembly_bucket_ms"] == 0.0
    assert result[2]["context_assembly_bucket_ms"] == 42.0
    assert result[2]["response_processing_bucket_ms"] == 8.0


def test_cross_trace_context_is_residual_after_response_and_tool():
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
    assert result[1]["context_assembly_bucket_ms"] == 0.0
