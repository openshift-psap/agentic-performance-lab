#!/usr/bin/env python3
"""Small, workload-specific stateful APIs for trace-replay benchmarking.

This service is intentionally separate from the LLM simulator.  It receives
tool calls over an internal HTTP boundary, owns per-session state, performs
domain operations, and emits auditable events. For SWE, it never owns or
pretends to own a repository: OpenShell executes the command against a staged
repository checkout and this service records the returned evidence. The implementations
are benchmark environments for the replay corpus; they are not claims of
official SWE-bench, BrowseComp, or τ-bench evaluation correctness.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


MAX_REQUEST_BYTES = 1_048_576
VOLATILE_KEYS = {"access_token", "api_key", "authorization", "password", "refresh_token", "token"}
EVENT_NAMES = {
    "software_engineering_qa": "swebench_tool",
    "research_deep_search": "browsecompplus_tool",
    "customer_support_ops": "tau2_tool",
}


def _canonical(value: Any, key: str | None = None) -> Any:
    if key and key.lower() in VOLATILE_KEYS:
        return "<redacted>"
    if isinstance(value, dict):
        return {str(k): _canonical(value[k], str(k)) for k in sorted(value)}
    if isinstance(value, list):
        return [_canonical(item) for item in value]
    return value


def arguments_sha256(arguments: dict[str, Any]) -> str:
    payload = json.dumps(_canonical(arguments), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _json_result_with_byte_budget(value: Any, byte_budget: int | None) -> tuple[str, bool]:
    """Serialize real retrieval data within the recorded response-size budget.

    Long indexed text is shortened at UTF-8 boundaries. When the resulting JSON
    is shorter than the recorded byte count, trailing JSON whitespace preserves
    the response size without adding invented document content.
    """
    separators = (",", ":")
    serialized = json.dumps(value, ensure_ascii=False, separators=separators)
    if byte_budget is None:
        return serialized, False
    if byte_budget < 0:
        raise ValueError("recorded result size cannot be negative")
    original_size = len(serialized.encode("utf-8"))
    if original_size <= byte_budget:
        return serialized + (" " * (byte_budget - original_size)), False

    bounded = copy.deepcopy(value)
    text_slots: list[tuple[dict[str, Any], str, bytes]] = []
    candidates = bounded if isinstance(bounded, list) else [bounded]
    for candidate in candidates:
        if isinstance(candidate, dict) and isinstance(candidate.get("text"), str):
            text_slots.append((candidate, "text", candidate["text"].encode("utf-8")))
    if not text_slots:
        raise ValueError(
            f"recorded result byte budget {byte_budget} is too small for the retrieval response envelope"
        )

    for candidate, field, _ in text_slots:
        candidate[field] = ""
    minimum = json.dumps(bounded, ensure_ascii=False, separators=separators)
    minimum_size = len(minimum.encode("utf-8"))
    if minimum_size > byte_budget:
        raise ValueError(
            f"recorded result byte budget {byte_budget} is too small for the retrieval response envelope ({minimum_size} bytes)"
        )

    total_text_bytes = sum(len(raw) for _, _, raw in text_slots)
    remaining_budget = byte_budget - minimum_size
    serialized = minimum
    # Escaped characters can use more bytes than their UTF-8 source. Recompute
    # the proportional content allowance until it fits, then add whitespace.
    for _ in range(32):
        for candidate, field, raw in text_slots:
            per_text_budget = (
                round(len(raw) * remaining_budget / total_text_bytes)
                if total_text_bytes else 0
            )
            candidate[field] = raw[:per_text_budget].decode("utf-8", errors="ignore")
        serialized = json.dumps(bounded, ensure_ascii=False, separators=separators)
        actual_size = len(serialized.encode("utf-8"))
        if actual_size <= byte_budget:
            return serialized + (" " * (byte_budget - actual_size)), True
        remaining_budget = max(0, remaining_budget - (actual_size - byte_budget) - 1)
    raise ValueError("could not fit actual retrieval text into the recorded response-size budget")


def now_ns() -> int:
    return time.clock_gettime_ns(getattr(time, "CLOCK_MONOTONIC_RAW", time.CLOCK_MONOTONIC))


class DomainBackend:
    def __init__(
        self,
        workload: str,
        root: Path,
        *,
        research_mode: str = "synthetic",
        browsecomp_index_path: str = "",
        browsecomp_k: int = 5,
    ):
        if workload not in EVENT_NAMES:
            raise ValueError(f"unsupported domain workload: {workload}")
        if research_mode not in {"synthetic", "browsecompplus"}:
            raise ValueError(f"unsupported research mode: {research_mode}")
        if workload == "research_deep_search" and research_mode == "browsecompplus" and not browsecomp_index_path:
            raise ValueError("BrowseComp-Plus mode requires --browsecomp-index-path or BROWSECOMP_INDEX_PATH")
        self.workload = workload
        self.root = root
        self.research_mode = research_mode if workload == "research_deep_search" else "not_applicable"
        self.browsecomp = (
            BrowseCompPlusBM25(browsecomp_index_path, browsecomp_k)
            if self.research_mode == "browsecompplus"
            else None
        )
        self.root.mkdir(parents=True, exist_ok=True)
        self.sessions: dict[str | tuple[str, str], dict[str, Any]] = {}
        self.events: list[dict[str, Any]] = []
        self.lock = threading.Lock()

    def _state(self, session_id: str, benchmark: str, run_key: str = "") -> dict[str, Any]:
        state_key: str | tuple[str, str] = (run_key, session_id) if run_key else session_id
        state = self.sessions.get(state_key)
        if state is not None:
            return state
        state = {
            "session_id": session_id,
            "benchmark": benchmark,
            "version": 0,
            "authenticated": False,
            "documents_read": [],
            "searches": [],
            "submissions": [],
            "files_touched": [],
            "tests_run": 0,
            "support_updates": 0,
            "bookings": {},
        }
        self.sessions[state_key] = state
        return state

    def _swe_result(
        self,
        state: dict[str, Any],
        tool: str,
        arguments: dict[str, Any],
        *,
        execution_mode: str = "backend",
        sandbox_result: dict[str, Any] | None = None,
    ) -> str:
        command = str(arguments.get("command") or arguments.get("cmd") or "")
        session_root = self.root / state["session_id"]
        state["files_touched"].append(str(arguments.get("path") or command or tool))
        lowered = command.lower()
        if execution_mode == "sandbox":
            result = sandbox_result
            if not isinstance(result, dict) or "returncode" not in result:
                return json.dumps({"output": "missing OpenShell sandbox result", "returncode": 125})
            if any(token in lowered for token in ("pytest", "unittest", "tox", "test")):
                state["tests_run"] += 1
            return json.dumps({
                "output": str(result.get("output") or ""),
                "returncode": int(result.get("returncode") or 0),
                "execution": "openshell-sandbox",
                "repo_root": str(result.get("repo_root") or ""),
                "base_commit": str(result.get("base_commit") or ""),
            })
        return json.dumps({
            "output": "SWE execution must come from the OpenShell sandbox with a staged repository",
            "returncode": 125,
            "execution": "backend-fallback-rejected",
        })

    def _research_result(
        self,
        state: dict[str, Any],
        tool: str,
        arguments: dict[str, Any],
        recorded_result_bytes: int | None = None,
    ) -> str:
        if self.research_mode == "browsecompplus":
            response, _, _ = self._browsecompplus_result(
                state, tool, arguments, recorded_result_bytes
            )
            return response

        query = str(arguments.get("query") or arguments.get("q") or arguments.get("url") or tool)
        lowered = tool.lower()
        if any(word in lowered for word in ("search", "query", "find")):
            state["searches"].append(query)
            digest = hashlib.sha256(query.encode()).hexdigest()[:8]
            result = [{
                "docid": digest,
                "score": 1.0,
                "snippet": f"Indexed research result for: {query}",
                "source": "trace-replay-browsecompplus-document-index",
            }]
            text_field = "snippet"
            item = result[0]
        else:
            docid = str(arguments.get("docid") or arguments.get("id") or "unknown")
            state["documents_read"].append(docid)
            if recorded_result_bytes is None:
                return json.dumps({
                    "docid": docid,
                    "title": f"Research document {docid}",
                    "content": f"Document content retrieved from the session-scoped index for {docid}.",
                })
            result = {
                "docid": docid,
                "title": f"Research document {docid}",
                "text": f"Synthetic research content for document {docid}.",
            }
            text_field = "text"
            item = result

        if recorded_result_bytes is None:
            return json.dumps(result)
        if recorded_result_bytes == 0:
            return ""
        # Keep the tool response JSON-shaped while matching the trace's UTF-8
        # payload size. The recorded content itself is never sent to the backend.
        item[text_field] = ""
        base = json.dumps(result)
        padding = recorded_result_bytes - len(base.encode("utf-8"))
        if padding < 0:
            raise ValueError("recorded research result is too short for its response envelope")
        words = "Synthetic research text for trace replay. "
        item[text_field] = (words * ((padding // len(words)) + 1))[:padding]
        return json.dumps(result)

    def _browsecompplus_result(
        self,
        state: dict[str, Any],
        tool: str,
        arguments: dict[str, Any],
        recorded_result_bytes: int | None,
    ) -> tuple[str, list[str], bool]:
        if self.browsecomp is None:
            raise RuntimeError("BrowseComp-Plus retriever is not configured")
        lowered = tool.lower()
        retrieved_docids: list[str] = []
        if lowered.endswith("search"):
            query = str(arguments.get("query") or arguments.get("q") or arguments.get("url") or "")
            if not query:
                raise ValueError("BrowseComp-Plus search requires a non-empty query")
            with self.lock:
                state["searches"].append(query)
            result = self.browsecomp.search(query)
            retrieved_docids = [str(item["docid"]) for item in result if item.get("docid") is not None]
        elif "get_document" in lowered:
            docid = str(
                arguments.get("docid")
                or arguments.get("doc_id")
                or arguments.get("document_id")
                or arguments.get("id")
                or ""
            )
            if not docid:
                raise ValueError("BrowseComp-Plus get_document requires docid or id")
            with self.lock:
                state["documents_read"].append(docid)
            result = self.browsecomp.get_document(docid)
            if result is not None:
                retrieved_docids = [str(result["docid"])]
        elif lowered.endswith("submit"):
            if "exact_answer" not in arguments:
                raise ValueError("BrowseComp-Plus submit requires exact_answer")
            with self.lock:
                state["submissions"].append({
                    "exact_answer": str(arguments.get("exact_answer") or ""),
                    "explanation": str(arguments.get("explanation") or ""),
                    "confidence": arguments.get("confidence"),
                })
            # Submission is recorded locally; answer scoring is outside this
            # replay backend. BrowseComp-Plus traces return null or empty text.
            result = None
        else:
            raise ValueError(f"unsupported BrowseComp-Plus tool: {tool}")

        if recorded_result_bytes == 0:
            # Still execute the real lookup and preserve its IDs in the audit
            # event, but keep an empty recorded tool response empty.
            return "", retrieved_docids, bool(result)
        response, truncated = _json_result_with_byte_budget(result, recorded_result_bytes)
        return response, retrieved_docids, truncated

    def _support_result(
        self, state: dict[str, Any], tool: str, arguments: dict[str, Any],
        recorded_result_bytes: int | None = None,
    ) -> str:
        lowered = tool.lower()
        if "login" in lowered or "auth" in lowered:
            state["authenticated"] = True
        state["support_updates"] += 1
        if any(word in lowered for word in ("book", "reserve", "create")):
            booking_id = f"BK-{state['session_id'][:8]}-{state['version'] + 1:03d}"
            state["bookings"][booking_id] = {"status": "created", "arguments": _canonical(arguments)}
            result = json.dumps({"ok": True, "booking_id": booking_id, "status": "created"})
        else:
            result = json.dumps({
            "ok": True,
            "policy_state": "authenticated" if state["authenticated"] else "anonymous",
            "updates": state["support_updates"],
            "result": f"Support operation {tool} completed against the session policy store.",
        })
        if recorded_result_bytes is None:
            return result
        if recorded_result_bytes < 0:
            raise ValueError("recorded result size cannot be negative")
        if recorded_result_bytes < len(result.encode("utf-8")):
            # The controlled backend cannot preserve its full JSON envelope
            # in a shorter recorded response; retain the measured byte size.
            return result[:recorded_result_bytes]
        return result + (" " * (recorded_result_bytes - len(result.encode("utf-8"))))

    def execute(self, request: dict[str, Any]) -> dict[str, Any]:
        session_id = str(request.get("session_id") or "")
        if not session_id:
            return {"ok": False, "error": "session_id is required"}
        tool = str(request.get("tool") or "unknown")
        run_key = str(request.get("run_key") or "")
        if run_key and not re.fullmatch(r"[0-9a-f]{32}", run_key):
            return {"ok": False, "error": "invalid replay run key"}
        arguments = dict(request.get("arguments") or {})
        started = now_ns()
        retrieved_docids: list[str] = []
        response_truncated_to_budget = False
        with self.lock:
            state = self._state(session_id, str(request.get("benchmark") or self.workload), run_key)
            state["version"] += 1
            state_version = state["version"]
            live_research = (
                self.workload == "research_deep_search"
                and self.research_mode == "browsecompplus"
            )
            if live_research:
                result = None
            elif self.workload == "software_engineering_qa":
                result = self._swe_result(
                    state,
                    tool,
                    arguments,
                    execution_mode=str(request.get("execution_mode") or "backend"),
                    sandbox_result=request.get("sandbox_result"),
                )
            elif self.workload == "research_deep_search":
                result = self._research_result(
                    state, tool, arguments, request.get("recorded_result_bytes")
                )
            else:
                result = self._support_result(state, tool, arguments, request.get("recorded_result_bytes"))
        if live_research:
            # Do not hold the session-state lock while the real retriever runs:
            # concurrent agent searches should be able to reach the local
            # search index together instead of being serialized by this service.
            result, retrieved_docids, response_truncated_to_budget = self._browsecompplus_result(
                state, tool, arguments, request.get("recorded_result_bytes")
            )
        event = {
            "event": EVENT_NAMES[self.workload],
            "timestamp_unix_ns": time.time_ns(),
            "trace_id": str(request.get("trace_id") or ""),
            "run_key": run_key,
            "backend": f"trace-replay-{self.workload}-backend",
            "session_id": session_id,
            "workload": self.workload,
            "benchmark": request.get("benchmark"),
            "turn": request.get("turn"),
            "call": request.get("call"),
            "tool": tool,
            "arguments_sha256": arguments_sha256(arguments),
            "elapsed_ms": round((now_ns() - started) / 1e6, 3),
            "result_bytes": len(result.encode("utf-8")),
            "state_version": state_version,
            "execution_mode": str(request.get("execution_mode") or "backend"),
        }
        if self.workload == "research_deep_search":
            event["retrieval_mode"] = self.research_mode
            if live_research:
                if "get_document" in tool.lower() or tool.lower().endswith("search"):
                    event["response_size_policy"] = "recorded_result_bytes"
                    event["recorded_result_bytes"] = request.get("recorded_result_bytes")
                if "get_document" in tool.lower() or tool.lower().endswith("search"):
                    event["retrieved_docids"] = retrieved_docids
                    event["response_truncated_to_trace_budget"] = response_truncated_to_budget
        if request.get("execution_mode") == "sandbox":
            sandbox_result = request.get("sandbox_result")
            if isinstance(sandbox_result, dict):
                event["sandbox_repo_root"] = str(sandbox_result.get("repo_root") or "")
                event["sandbox_base_commit"] = str(sandbox_result.get("base_commit") or "")
                event["sandbox_returncode"] = int(sandbox_result.get("returncode") or 0)
                # A pipeline such as `pytest | tee` may report status 0
                # even when Python printed a genuine import failure.
                # Anchor to diagnostic lines so source-code snippets that
                # merely mention these strings are not treated as errors.
                event["sandbox_dependency_failure"] = bool(re.search(
                    r"^\s*(?:(?:ModuleNotFoundError|ImportError):|(?:\S*/)?python(?:\d+(?:\.\d+)?)?:\s*No module named\b)",
                    str(sandbox_result.get("output") or ""), re.IGNORECASE | re.MULTILINE,
                ))
                if sandbox_result.get("operation") == "command":
                    for field in (
                        "command_started_epoch_ns", "command_ended_epoch_ns", "command_wall_ns",
                        "command_user_cpu_us", "command_system_cpu_us",
                        "command_voluntary_context_switches", "command_involuntary_context_switches",
                        "command_minor_faults", "command_major_faults",
                        "command_effective_cpu_ids", "command_requested_cpu_ids",
                        "command_cgroup_quota_requested", "command_cgroup_quota_applied",
                        "command_cgroup_quota_original",
                        "command_cgroup_usage_usec", "command_cgroup_user_usec",
                        "command_cgroup_system_usec", "command_cgroup_nr_throttled",
                        "command_cgroup_throttled_usec", "command_cgroup_pressure_some_usec",
                        "command_cgroup_pressure_full_usec",
                    ):
                        if field in sandbox_result:
                            event[field] = sandbox_result[field]
                if int(sandbox_result.get("returncode") or 0) != 0:
                    event["sandbox_output_prefix"] = str(sandbox_result.get("output") or "")[:512]
        with self.lock:
            self.events.append(event)
        print(json.dumps(event, separators=(",", ":")), flush=True)
        return {"ok": True, "result": result, "state_version": event["state_version"]}



class BrowseCompPlusBM25:
    """Direct adapter around BrowseComp-Plus's upstream BM25 index API.

    It mirrors the upstream BM25 searcher's lookup behavior: search returns
    ranked document IDs, scores, and full indexed text; get_document reads a
    full document from the same local Lucene index. No MCP process or network
    hop is involved. Pyserini and Java are loaded only when this mode is enabled.
    """

    def __init__(self, index_path: str, k: int = 5):
        # Pyserini builds an OpenAI client while its lucene search module is
        # imported. BM25 retrieval never calls that client, and containers
        # normally carry no OpenAI key, so the import would fail with
        # "Missing credentials" without a placeholder.
        os.environ.setdefault("OPENAI_API_KEY", "unused-by-browsecomp-backend")
        from pyserini.search.lucene import LuceneSearcher

        if k < 1:
            raise ValueError("BrowseComp-Plus result count must be positive")
        self.searcher = LuceneSearcher(index_path)
        self.k = k

    @staticmethod
    def _contents(raw_document: str) -> str:
        payload = json.loads(raw_document)
        return str(payload["contents"])

    def search(self, query: str) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for hit in self.searcher.search(query, self.k):
            text = self._contents(hit.lucene_document.get("raw"))
            results.append({"docid": hit.docid, "score": hit.score, "text": text})
        return results

    def get_document(self, docid: str) -> dict[str, str] | None:
        document = self.searcher.doc(docid)
        if document is None:
            return None
        return {"docid": docid, "text": self._contents(document.raw())}

def serve(
    workload: str,
    host: str,
    port: int,
    root: Path,
    research_mode: str,
    browsecomp_index_path: str,
    browsecomp_k: int,
) -> None:
    backend = DomainBackend(
        workload,
        root,
        research_mode=research_mode,
        browsecomp_index_path=browsecomp_index_path,
        browsecomp_k=browsecomp_k,
    )

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *values: Any) -> None:
            return

        def send_json(self, status: int, payload: Any) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/health":
                self.send_json(200, {
                    "status": "ok",
                    "workload": workload,
                    "research_mode": backend.research_mode,
                    "browsecomp_index_loaded": backend.browsecomp is not None,
                    "events": len(backend.events),
                })
            elif self.path == "/events":
                self.send_json(200, {"events": backend.events})
            else:
                self.send_json(404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/execute":
                self.send_json(404, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length") or 0)
                if length <= 0 or length > MAX_REQUEST_BYTES:
                    raise ValueError("invalid request size")
                response = backend.execute(json.loads(self.rfile.read(length).decode("utf-8")))
                self.send_json(200 if response.get("ok") else 400, response)
            except Exception as exc:  # noqa: BLE001
                self.send_json(400, {"ok": False, "error": str(exc)})

    print(f"domain backend listening workload={workload} port={port}", flush=True)
    ThreadingHTTPServer((host, port), Handler).serve_forever()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", required=True, choices=sorted(EVENT_NAMES))
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--root", default="/data/sessions")
    parser.add_argument(
        "--research-mode",
        choices=("synthetic", "browsecompplus"),
        default=os.environ.get("TRACE_REPLAY_RESEARCH_MODE", "synthetic"),
        help="Use the legacy synthetic research response or direct BrowseComp-Plus BM25 retrieval.",
    )
    parser.add_argument(
        "--browsecomp-index-path",
        default=os.environ.get("BROWSECOMP_INDEX_PATH", ""),
        help="Path to the local BrowseComp-Plus BM25 Lucene index.",
    )
    parser.add_argument(
        "--browsecomp-k",
        type=int,
        default=int(os.environ.get("BROWSECOMP_K", "5")),
        help="Number of BrowseComp-Plus BM25 hits to return (default: 5).",
    )
    args = parser.parse_args()
    serve(
        args.workload,
        args.host,
        args.port,
        Path(args.root),
        args.research_mode,
        args.browsecomp_index_path,
        args.browsecomp_k,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
