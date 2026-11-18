import json
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from appworld_service import AppWorldAdapter


def test_appworld_worlds_are_run_scoped_and_independent(tmp_path, monkeypatch):
    created = []
    barrier = threading.Barrier(2)

    class FakeWorld:
        def __init__(self, task_id, experiment_name):
            created.append((task_id, experiment_name))
            self.task = SimpleNamespace(supervisor={"email": "tester@example.com"})

        def execute(self, code):
            barrier.wait(timeout=2)
            return json.dumps({"ok": True})

    monkeypatch.setitem(sys.modules, "appworld", SimpleNamespace(AppWorld=FakeWorld))
    mapping = tmp_path / "task-map.json"
    mapping.write_text(json.dumps({"same": "task-1"}), encoding="utf-8")
    adapter = AppWorldAdapter(str(tmp_path), str(mapping))
    replies = []

    def run(key):
        replies.append(adapter.execute({"session_id": "same", "run_key": key,
                                        "tool": "mcp__environment__calendar__list_events",
                                        "arguments": {}, "turn": 0, "call": 0}))

    threads = [threading.Thread(target=run, args=(key,)) for key in ("a" * 32, "b" * 32)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=3)
    assert len(replies) == 2 and all(reply["ok"] for reply in replies)
    assert len(created) == 2
    assert created[0][1] != created[1][1]
    assert len(adapter.worlds) == 2
    assert all(event["queue_wait_ms"] >= 0 for event in adapter.events)
