"""Read-only CLI checks for workload allocation and trajectory selection."""

import json
import subprocess
from pathlib import Path

import pytest

from trace_replay_sim.workload_allocation import allocate_custom_workloads


ROOT = Path(__file__).resolve().parents[1]


def _corpus(path: Path, count: int = 5) -> Path:
    workloads = (
        "software_engineering_qa",
        "personal_assistant",
        "research_deep_search",
        "customer_support_ops",
    )
    sessions = [
        {
            "session_id": f"{workload}-{index}",
            "workload": workload,
            "benchmark": workload,
            "turns": [{"index": 0, "tool_calls": [{"name": "exec", "arguments": {}}]}],
        }
        for workload in workloads for index in range(count)
    ]
    path.write_text(json.dumps({"sessions": sessions}), encoding="utf-8")
    return path


def _dry_run(corpus: Path, *args: str) -> subprocess.CompletedProcess[str]:
    version = subprocess.run(["bash", "-c", 'printf "%s" "${BASH_VERSINFO[0]}"'],
                             text=True, capture_output=True, check=False)
    if int(version.stdout or 0) < 4:
        pytest.skip("bench requires Bash 4+; this macOS host has Bash 3.2")
    return subprocess.run(
        ["bash", str(ROOT / "scripts/bench"), "--dry-run", "--corpus", str(corpus), *args],
        cwd=ROOT, text=True, capture_output=True, check=False,
    )


def test_percentage_allocator_preserves_total_and_stable_ties() -> None:
    assert allocate_custom_workloads(10, "swe:33,pa:33,cs:34") == {
        "swe": 3, "pa": 3, "research": 0, "cs": 4,
    }
    assert allocate_custom_workloads(3, "swe:50,pa:50") == {
        "swe": 2, "pa": 1, "research": 0, "cs": 0,
    }


def test_invalid_custom_allocation_rejected() -> None:
    with pytest.raises(ValueError, match="must equal agent count"):
        allocate_custom_workloads(10, "swe=3,pa=3,cs=3")
    with pytest.raises(ValueError, match="do not mix"):
        allocate_custom_workloads(10, "swe:50,pa=5")


def test_percentage_allocation_keeps_every_agent(tmp_path: Path) -> None:
    result = _dry_run(_corpus(tmp_path / "corpus.json"), "--agents", "10",
                      "--workload-dist", "swe:33,pa:33,cs:34")
    assert result.returncode == 0, result.stderr
    assert "software_engineering_qa=3" in result.stdout
    assert "personal_assistant=3" in result.stdout
    assert "customer_support_ops=4" in result.stdout


def test_requested_trajectory_population_is_checked(tmp_path: Path) -> None:
    corpus = _corpus(tmp_path / "corpus.json", count=2)
    result = _dry_run(corpus, "--agents", "1", "--workload-dist", "swe", "--traces", "3")
    assert result.returncode != 0
    assert "requires --traces=3" in result.stderr


def test_custom_allocation_rejects_mixed_and_short_counts(tmp_path: Path) -> None:
    corpus = _corpus(tmp_path / "corpus.json")
    short = _dry_run(corpus, "--agents", "10", "--workload-dist", "swe=3,pa=3,cs=3")
    assert short.returncode != 0
    assert "must equal agent count" in short.stderr
    mixed = _dry_run(corpus, "--agents", "10", "--workload-dist", "swe:50,pa=5")
    assert mixed.returncode != 0
    assert "do not mix counts and percentages" in mixed.stderr
