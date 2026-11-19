import base64
import json
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_experiment import _workload_command  # noqa: E402
from smoke_browsecompplus_corpus import decode_tool_result  # noqa: E402


def test_real_research_manifest_is_owned_internal_and_pinned_to_approved_node():
    manifest = ROOT / "k8s/browsecompplus-real-backend.yaml"
    resources = list(yaml.safe_load_all(manifest.read_text(encoding="utf-8")))
    assert [resource["kind"] for resource in resources] == [
        "PersistentVolumeClaim", "Deployment", "Service",
    ]
    assert all(resource["metadata"]["name"].startswith("browsecompplus-real-") for resource in resources)
    assert all(resource["metadata"]["namespace"] == "trace-replay" for resource in resources)
    assert resources[0]["spec"]["storageClassName"] == "rook-cephfs-fast"

    deployment = resources[1]
    pod_spec = deployment["spec"]["template"]["spec"]
    download_args = pod_spec["initContainers"][0]["args"]
    assert download_args[download_args.index("--revision") + 1] == "b3f37f70c33829eb09d04784a54277a31871fd63"
    assert "nodeSelector" not in pod_spec
    assert pod_spec["containers"][0]["args"][-2:] == ["--browsecomp-index-path", "/data/indexes/bm25"]
    assert resources[2]["spec"]["type"] == "ClusterIP"
    assert "Route" not in {resource["kind"] for resource in resources}


def test_recorded_submit_response_length_is_passed_to_backend_adapter():
    command = _workload_command(
        "session-1", "research_deep_search", "browsecompplus", 4, 0,
        "mcp__environment__submit", {"exact_answer": "answer"},
        "http://127.0.0.1:8093", recorded_result_bytes=0,
    )
    match = re.search(r"b64decode\('([^']+)'\)", command[2])
    assert match
    request = json.loads(base64.b64decode(match.group(1)))
    assert request["tool"] == "mcp__environment__submit"
    assert request["recorded_result_bytes"] == 0


def test_runtime_image_pins_pyserini_and_java21():
    dockerfile = (ROOT / "containers/browsecompplus/Dockerfile").read_text(encoding="utf-8")
    assert "eclipse-temurin:21-jre-jammy" in dockerfile
    assert "python:3.10-slim-bookworm" in dockerfile
    assert '"pyserini==1.2.0"' in dockerfile


def test_local_smoke_accepts_recorded_empty_tool_responses():
    assert decode_tool_result("mcp__environment__search", "", "") == (None, "")
    assert decode_tool_result("mcp__environment__get_document", "", "") == (None, "")
    assert decode_tool_result("mcp__environment__submit", "", "") == (None, "")
    assert decode_tool_result("mcp__environment__submit", "null", "null") == (None, "")

