import base64
import json
import subprocess
import sys
from types import SimpleNamespace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from run_experiment import _swe_sandbox_command, _swe_task_runtime_preflight_code, _swe_command_map, _collect_backend_events
from trace_replay_sim.swe_manifest import SweManifestError, load_swe_manifest, validate_swe_sessions


MANIFEST = Path(__file__).parents[1] / "config/swebench-trace-manifest.json"


def test_swe_command_map_preserves_repeated_command_occurrences():
    command = "python -m pytest"
    session = {"turns": [{"tool_calls": [
        {"swe_original_command": command,
         "mapped_command": ["python3", f"/tmp/trace-replay-tool-calls/0000-00{index}.py"]}
        for index in range(2)
    ]}]}
    mapping = _swe_command_map(session)
    assert len(mapping) == 1
    assert next(iter(mapping.values())) == [
        "/tmp/trace-replay-tool-calls/0000-000.py",
        "/tmp/trace-replay-tool-calls/0000-001.py",
    ]


def test_checked_in_manifest_has_real_pinned_task_environments():
    tasks = load_swe_manifest(MANIFEST)
    assert len(tasks) == 25
    task = tasks["e3bfdef4aca3_061b9f48"]
    assert task.repo == "astropy/astropy"
    assert len(task.base_commit) == 40
    assert task.image.startswith("swebench/")


def test_swe_sessions_fail_closed_without_a_repository_mapping():
    tasks = load_swe_manifest(MANIFEST)
    with pytest.raises(SweManifestError, match="no real SWE environment mapping"):
        validate_swe_sessions([{"session_id": "missing", "benchmark": "swebench"}], tasks)


def test_generated_swe_command_carries_pinned_task_identity():
    tasks = load_swe_manifest(MANIFEST)
    task = tasks["e3bfdef4aca3_061b9f48"]
    command = _swe_sandbox_command(
        "s",
        "software_engineering_qa",
        "swebench",
        0,
        0,
        "mcp__environment__bash",
        {"command": "git status --short"},
        "http://backend",
        task,
    )
    encoded = command[2].split("b64decode('", 1)[1].split("'", 1)[0]
    payload = json.loads(base64.b64decode(encoded))
    assert payload["swe_task"]["base_commit"] == task.base_commit
    assert payload["swe_task"]["repo"] == task.repo
    assert payload["run_key"] == ""
    compile(command[2], "<generated-swe-command>", "exec")
    task_image_command = _swe_sandbox_command(
        "s", "software_engineering_qa", "swebench", 0, 0,
        "mcp__environment__bash", {"command": "python3 -V"}, "http://backend", task,
        use_task_image_runtime=True,
    )
    assert "trace-replay-swe-runtime.json" in task_image_command[2]
    assert "PYTHONPATH" in task_image_command[2]
    assert "export PATH=/opt/miniconda3/envs/testbed/bin" not in task_image_command[2]
    compile(task_image_command[2], "<generated-task-image-command>", "exec")


def test_swe_command_carries_run_key_and_collector_accepts_matching_event():
    tasks = load_swe_manifest(MANIFEST)
    run_key = "a" * 32
    command = _swe_sandbox_command(
        "session", "software_engineering_qa", "swebench", 0, 0,
        "mcp__environment__bash", {"command": "python3 -V"},
        "http://backend", tasks["e3bfdef4aca3_061b9f48"], run_key=run_key,
    )
    encoded = command[2].split("b64decode('", 1)[1].split("'", 1)[0]
    assert json.loads(base64.b64decode(encoded))["run_key"] == run_key

    event = {
        "event": "swebench_tool", "session_id": "session", "trace_id": "",
        "run_key": run_key, "timestamp_unix_ns": 1001 * 10**9,
        "turn": 0, "call": 0, "tool": "mcp__environment__bash",
    }
    class ClusterStub:
        def run(self, _args, check=False):
            return json.dumps(event)

    backend = SimpleNamespace(namespace="aharush-test", name="aharush-backend", event_name="swebench_tool", port=8092)
    assert _collect_backend_events(ClusterStub(), backend, "session", 1000, {"b" * 32}, run_key=run_key, end=1002) == [event]
    assert _collect_backend_events(ClusterStub(), backend, "session", 1000, {"b" * 32}, run_key="c" * 32, end=1002) == []


@pytest.mark.parametrize("module_parent", ["", "lib"])
def test_task_image_runtime_preflight_selects_python_for_pinned_checkout(tmp_path, module_parent):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    module_dir = checkout / module_parent / "repo"
    module_dir.mkdir(parents=True)
    (module_dir / "__init__.py").write_text("VALUE = 1\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(checkout)], check=True)
    subprocess.run(["git", "-C", str(checkout), "add", "."], check=True)
    subprocess.run(
        ["git", "-C", str(checkout), "-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-qm", "initial"],
        check=True,
    )
    commit = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
    task = {"base_commit": commit, "image": "swebench/test:latest", "repo": "owner/repo"}
    code = _swe_task_runtime_preflight_code(task)
    compile(code, "<swe-runtime-preflight>", "exec")
    code = code.replace("'/sandbox/.openshell/workspace'", repr(str(checkout)))
    code = code.replace("'/tmp/trace-replay-swe-runtime.json'", repr(str(tmp_path / "runtime.json")))
    code = code.replace("'/tmp/trace-replay-swe-python-bin'", repr(str(tmp_path / "python-bin")))
    completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    runtime = json.loads((tmp_path / "runtime.json").read_text())
    assert runtime["base_commit"] == commit
    assert runtime["module_origin"].startswith(str(checkout))
    assert (tmp_path / "python-bin" / "python3").resolve() == Path(runtime["python"]).resolve()


def test_task_image_runtime_preflight_rejects_wrong_base_commit(tmp_path):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    subprocess.run(["git", "init", "-q", str(checkout)], check=True)
    code = _swe_task_runtime_preflight_code({"base_commit": "0" * 40, "image": "swebench/test", "repo": "owner/repo"})
    code = code.replace("'/sandbox/.openshell/workspace'", repr(str(checkout)))
    completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert completed.returncode != 0
    assert "checkout does not match task base commit" in completed.stderr


def test_manifest_round_trip_is_json_serializable():
    tasks = load_swe_manifest(MANIFEST)
    payload = {session_id: task.public_dict() for session_id, task in tasks.items()}
    assert json.loads(json.dumps(payload))["e3bfdef4aca3_061b9f48"]["base_commit"]


def test_manifest_rejects_unpinned_or_mismatched_metadata(tmp_path):
    payload = {
        "schema": "agentic.trace-replay.swe-task-manifest.v1",
        "tasks": [{
            "session_id": "s1",
            "instance_id": "owner__repo-1",
            "repo": "owner/repo",
            "repo_url": "https://github.com/owner/other.git",
            "base_commit": "not-a-commit",
            "image": "swebench/image:latest",
        }],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(SweManifestError):
        load_swe_manifest(path)
