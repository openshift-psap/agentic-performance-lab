import csv

from scripts.analyze_multi_agent_results import write_combined_tables


def test_final_latency_table_has_requested_columns_and_context_to_model_end_boundary(tmp_path):
    experiment = tmp_path / "experiment"
    source = experiment / "traces" / "trace-a" / "data/shell/profile-v2/tables"
    source.mkdir(parents=True)
    with (source / "per_turn_latency.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=(
            "trace_id", "turn", "context_assembly_ms", "context_to_model_end_ms",
            "model_call_ms", "response_processing_ms", "openshell_execution_ms",
            "input_tokens", "output_tokens", "prompt_chars", "request_bytes",
            "input_messages", "input_message_chars",
        ))
        writer.writeheader()
        writer.writerow({
            "trace_id": "otel-a", "turn": 3, "context_assembly_ms": 20,
            "context_to_model_end_ms": 150, "model_call_ms": 140,
            "response_processing_ms": 10, "openshell_execution_ms": 300,
            "input_tokens": 2000, "output_tokens": 100,
            "prompt_chars": 8000, "request_bytes": 8500,
            "input_messages": 7, "input_message_chars": 4000,
        })
    out = experiment / "analysis"
    out.mkdir()
    write_combined_tables(experiment, [{
        "trace": "trace-a", "session_id": "session-a",
        "workload": "software_engineering_qa", "agent": "aharush-agent",
    }], out)
    with (out / "per_turn_latency.csv").open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        row = next(reader)
        assert reader.fieldnames == [
            "trace", "session_id", "workload", "agent", "trace_id", "turn",
            "context_assembly_ms", "response_processing_ms", "openshell_execution_ms",
            "input_tokens", "output_tokens", "prompt_chars", "request_bytes",
            "input_messages", "input_message_chars",
        ]
    assert row["context_assembly_ms"] == "150"
    assert row["response_processing_ms"] == "10"
    assert row["openshell_execution_ms"] == "300"
