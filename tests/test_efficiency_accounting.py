import json

from scripts.analyze_multi_agent_results import _normalized_efficiency, write_concurrency_timeline


def test_efficiency_uses_nonoverlapping_stage_buckets(tmp_path):
    profile = tmp_path / "data/shell/profile-v2"
    profile.mkdir(parents=True)
    (profile / "per_turn.json").write_text(json.dumps([
        {"turn": 0, "prompt_boundary_ms": 100.0,
         "context_assembly_bucket_ms": 100.0,
         "response_processing_ms": 40.0, "exec_ms": 500.0,
         "response_processing_full_ms": 540.0},
        {"turn": 1, "prompt_boundary_ms": 540.0,
         "context_assembly_bucket_ms": 0.0,
         "response_processing_bucket_ms": 40.0,
         "sandbox_execution_bucket_ms": 500.0},
    ]), encoding="utf-8")

    rows = _normalized_efficiency(tmp_path)

    assert rows[1]["raw_prompt_boundary_ms"] == 540.0
    assert rows[1]["prompt_boundary_ms"] == 0.0
    assert rows[1]["response_processing_ms"] == 40.0
    assert rows[1]["openshell_exec_ms"] == 500.0
    assert rows[1]["non_model_path_ms"] == 540.0


def test_observed_concurrency_comes_from_measured_windows(tmp_path):
    summary = write_concurrency_timeline([
        {"trace": "a", "measured_start_unix": 10.0, "measured_end_unix": 20.0},
        {"trace": "b", "measured_start_unix": 12.0, "measured_end_unix": 15.0},
        {"trace": "c", "measured_start_unix": 21.0, "measured_end_unix": 25.0},
    ], tmp_path)
    assert summary["peak_active_sessions"] == 2
    assert summary["measured_sessions"] == 3
    assert round(summary["mean_active_sessions"], 3) == round(17 / 15, 3)
