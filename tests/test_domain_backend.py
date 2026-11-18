import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from domain_backend import DomainBackend


def test_domain_backends_keep_session_state(tmp_path):
    swe = DomainBackend("software_engineering_qa", tmp_path / "swe")
    first = swe.execute({
        "session_id": "s1",
        "benchmark": "swebench",
        "tool": "mcp__environment__bash",
        "arguments": {"command": "pwd"},
        "execution_mode": "sandbox",
        "sandbox_result": {"output": "/testbed", "returncode": 0, "repo_root": "/testbed", "base_commit": "abc"},
        "turn": 0,
        "call": 0,
    })
    second = swe.execute({
        "session_id": "s1",
        "benchmark": "swebench",
        "tool": "mcp__environment__bash",
        "arguments": {"command": "pytest -q"},
        "execution_mode": "sandbox",
        "sandbox_result": {"output": "2 passed", "returncode": 0, "repo_root": "/testbed", "base_commit": "abc"},
        "turn": 1,
        "call": 0,
    })
    assert first["ok"] and second["ok"]
    assert swe.events[-1]["state_version"] == 2
    assert swe.sessions["s1"]["tests_run"] == 1
    assert swe.events[-1]["execution_mode"] == "sandbox"
    assert swe.events[-1]["sandbox_base_commit"] == "abc"

    research = DomainBackend("research_deep_search", tmp_path / "research")
    result = research.execute({"session_id": "r1", "benchmark": "browsecompplus", "tool": "mcp__environment__search", "arguments": {"query": "agent systems"}, "turn": 0, "call": 0})
    assert json.loads(result["result"])[0]["source"] == "trace-replay-browsecompplus-document-index"
    assert research.sessions["r1"]["searches"] == ["agent systems"]

    support = DomainBackend("customer_support_ops", tmp_path / "support")
    result = support.execute({"session_id": "c1", "benchmark": "tau2_airline", "tool": "book_flight", "arguments": {"route": "A-B"}, "turn": 0, "call": 0})
    assert json.loads(result["result"])["status"] == "created"
    assert support.sessions["c1"]["bookings"]


def test_research_results_match_recorded_byte_length(tmp_path):
    research = DomainBackend("research_deep_search", tmp_path / "research")
    for tool, arguments, expected_type in (
        ("mcp__environment__search", {"query": "agent systems"}, list),
        ("mcp__environment__get_document", {"docid": "47683"}, dict),
    ):
        response = research.execute({
            "session_id": "r1",
            "benchmark": "browsecompplus",
            "tool": tool,
            "arguments": arguments,
            "recorded_result_bytes": 7005,
        })
        result = response["result"]
        assert len(result.encode("utf-8")) == 7005
        assert isinstance(json.loads(result), expected_type)
        assert "Synthetic research text" in result


def test_run_key_isolates_repeated_session_state_and_support_size(tmp_path):
    support = DomainBackend("customer_support_ops", tmp_path / "support")
    request = {"session_id": "same", "benchmark": "tau2_airline", "tool": "message",
               "arguments": {"text": "hello"}, "recorded_result_bytes": 312}
    first = support.execute({**request, "run_key": "a" * 32})
    second = support.execute({**request, "run_key": "b" * 32})
    assert len(first["result"].encode("utf-8")) == 312
    assert len(second["result"].encode("utf-8")) == 312
    assert first["state_version"] == second["state_version"] == 1
    assert len(support.sessions) == 2


def test_swe_backend_marks_missing_dependency_even_when_shell_exit_is_zero(tmp_path):
    backend = DomainBackend("software_engineering_qa", tmp_path / "swe")
    backend.execute({
        "session_id": "swe-missing-dependency",
        "benchmark": "swebench",
        "tool": "mcp__environment__bash",
        "arguments": {"command": "pytest -q | tee test.log"},
        "execution_mode": "sandbox",
        "sandbox_result": {
            "output": "ModuleNotFoundError: No module named 'pytest'",
            "returncode": 0,
            "repo_root": "/testbed",
            "base_commit": "abc",
        },
        "turn": 0,
        "call": 0,
    })
    assert backend.events[-1]["sandbox_dependency_failure"] is True


def test_swe_backend_does_not_treat_source_text_as_missing_dependency(tmp_path):
    backend = DomainBackend("software_engineering_qa", tmp_path / "swe")
    backend.execute({
        "session_id": "swe-source-text", "benchmark": "swebench",
        "tool": "mcp__environment__bash", "arguments": {"command": "grep ImportError source.py"},
        "execution_mode": "sandbox",
        "sandbox_result": {
            "output": 'source.py:12: print("ImportError: example")',
            "returncode": 0, "repo_root": "/testbed", "base_commit": "abc",
        },
        "turn": 0, "call": 0,
    })
    assert backend.events[-1]["sandbox_dependency_failure"] is False
