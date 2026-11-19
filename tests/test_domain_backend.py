import json
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from domain_backend import DomainBackend, _json_result_with_byte_budget


def install_fake_browsecomp_index(monkeypatch):
    documents = {
        "doc-17": {"contents": "Real indexed BrowseComp-Plus document seventeen."},
        "doc-23": {"contents": "Another indexed document with distinct content."},
    }

    class FakeRawDocument:
        def __init__(self, text):
            self.text = text

        def raw(self):
            return json.dumps(self.text)

    class FakeHit:
        def __init__(self, docid, score):
            self.docid = docid
            self.score = score
            self.lucene_document = {"raw": json.dumps(documents[docid])}

    class FakeLuceneSearcher:
        def __init__(self, index_path):
            assert index_path == "/fixture/browsecomp-index"
            self.queries = []

        def search(self, query, k):
            self.queries.append((query, k))
            return [FakeHit("doc-17", 9.25), FakeHit("doc-23", 7.5)][:k]

        def doc(self, docid):
            document = documents.get(docid)
            return FakeRawDocument(document) if document is not None else None

    pyserini = types.ModuleType("pyserini")
    search = types.ModuleType("pyserini.search")
    lucene = types.ModuleType("pyserini.search.lucene")
    lucene.LuceneSearcher = FakeLuceneSearcher
    monkeypatch.setitem(sys.modules, "pyserini", pyserini)
    monkeypatch.setitem(sys.modules, "pyserini.search", search)
    monkeypatch.setitem(sys.modules, "pyserini.search.lucene", lucene)


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


def test_browsecompplus_search_returns_real_index_content_and_audit_ids(tmp_path, monkeypatch):
    install_fake_browsecomp_index(monkeypatch)
    research = DomainBackend(
        "research_deep_search",
        tmp_path / "research",
        research_mode="browsecompplus",
        browsecomp_index_path="/fixture/browsecomp-index",
        browsecomp_k=2,
    )

    response = research.execute({
        "session_id": "r-real",
        "benchmark": "browsecompplus",
        "tool": "mcp__environment__search",
        "arguments": {"query": "aurora colors"},
        "recorded_result_bytes": 110,
        "turn": 3,
        "call": 0,
    })

    result = json.loads(response["result"])
    assert [item["docid"] for item in result] == ["doc-17", "doc-23"]
    assert [item["score"] for item in result] == [9.25, 7.5]
    assert "Real indexed BrowseComp-Plus document seventeen.".startswith(result[0]["text"])
    assert "Another indexed document with distinct content.".startswith(result[1]["text"])
    assert len(response["result"].encode("utf-8")) == 110
    assert research.events[-1]["response_size_policy"] == "recorded_result_bytes"
    assert research.events[-1]["recorded_result_bytes"] == 110
    assert research.browsecomp.searcher.queries == [("aurora colors", 2)]
    assert research.events[-1]["retrieval_mode"] == "browsecompplus"
    assert research.events[-1]["retrieved_docids"] == ["doc-17", "doc-23"]
    assert research.sessions["r-real"]["searches"] == ["aurora colors"]


def test_browsecompplus_get_document_returns_full_indexed_text(tmp_path, monkeypatch):
    install_fake_browsecomp_index(monkeypatch)
    research = DomainBackend(
        "research_deep_search", tmp_path / "research",
        research_mode="browsecompplus", browsecomp_index_path="/fixture/browsecomp-index",
    )
    response = research.execute({
        "session_id": "r-doc", "benchmark": "browsecompplus",
        "tool": "mcp__environment__get_document", "arguments": {"docid": "doc-17"},
        "recorded_result_bytes": 45,
    })
    result = json.loads(response["result"])
    assert result["docid"] == "doc-17"
    assert "Real indexed BrowseComp-Plus document seventeen.".startswith(result["text"])
    assert len(response["result"].encode("utf-8")) == 45
    assert research.events[-1]["retrieved_docids"] == ["doc-17"]
    assert research.sessions["r-doc"]["documents_read"] == ["doc-17"]


def test_browsecompplus_missing_document_is_json_null(tmp_path, monkeypatch):
    install_fake_browsecomp_index(monkeypatch)
    research = DomainBackend(
        "research_deep_search", tmp_path / "research",
        research_mode="browsecompplus", browsecomp_index_path="/fixture/browsecomp-index",
    )
    response = research.execute({
        "session_id": "r-missing", "benchmark": "browsecompplus",
        "tool": "mcp__environment__get_document", "arguments": {"docid": "missing"},
    })
    assert response["result"] == "null"
    assert research.events[-1]["retrieved_docids"] == []


def test_browsecompplus_submit_records_answer_without_scoring(tmp_path, monkeypatch):
    install_fake_browsecomp_index(monkeypatch)
    research = DomainBackend(
        "research_deep_search", tmp_path / "research",
        research_mode="browsecompplus", browsecomp_index_path="/fixture/browsecomp-index",
    )
    response = research.execute({
        "session_id": "r-submit", "benchmark": "browsecompplus",
        "tool": "mcp__environment__submit",
        "arguments": {"exact_answer": "answer", "explanation": "because", "confidence": 72},
    })
    assert response["result"] == "null"
    assert research.sessions["r-submit"]["submissions"] == [{
        "exact_answer": "answer", "explanation": "because", "confidence": 72,
    }]
    assert research.events[-1]["retrieval_mode"] == "browsecompplus"
    assert "retrieved_docids" not in research.events[-1]


def test_browsecompplus_submit_preserves_empty_recorded_response(tmp_path, monkeypatch):
    install_fake_browsecomp_index(monkeypatch)
    research = DomainBackend(
        "research_deep_search", tmp_path / "research",
        research_mode="browsecompplus", browsecomp_index_path="/fixture/browsecomp-index",
    )
    response = research.execute({
        "session_id": "r-empty-submit", "benchmark": "browsecompplus",
        "tool": "mcp__environment__submit",
        "arguments": {"exact_answer": "answer"}, "recorded_result_bytes": 0,
    })
    assert response["result"] == ""


def test_browsecompplus_empty_recorded_search_still_performs_and_audits_lookup(tmp_path, monkeypatch):
    install_fake_browsecomp_index(monkeypatch)
    research = DomainBackend(
        "research_deep_search", tmp_path / "research",
        research_mode="browsecompplus", browsecomp_index_path="/fixture/browsecomp-index",
    )
    response = research.execute({
        "session_id": "r-empty-search", "benchmark": "browsecompplus",
        "tool": "mcp__environment__search", "arguments": {"query": "aurora colors"},
        "recorded_result_bytes": 0, "turn": 2, "call": 0,
    })
    assert response["result"] == ""
    assert research.browsecomp.searcher.queries == [("aurora colors", 5)]
    assert research.events[-1]["retrieved_docids"] == ["doc-17", "doc-23"]
    assert research.events[-1]["result_bytes"] == 0


def test_browsecompplus_mode_fails_closed_for_unmapped_tools(tmp_path, monkeypatch):
    install_fake_browsecomp_index(monkeypatch)
    research = DomainBackend(
        "research_deep_search", tmp_path / "research",
        research_mode="browsecompplus", browsecomp_index_path="/fixture/browsecomp-index",
    )
    try:
        research.execute({
            "session_id": "r-unknown", "benchmark": "browsecompplus",
            "tool": "mcp__environment__unknown", "arguments": {},
        })
    except ValueError as exc:
        assert "unsupported BrowseComp-Plus tool" in str(exc)
    else:
        raise AssertionError("real retrieval mode silently accepted an unmapped tool")
    assert not research.events


def test_browsecompplus_byte_budget_keeps_valid_utf8_json_and_exact_size():
    value = {"docid": "x", "text": "évidence " * 100}
    response, truncated = _json_result_with_byte_budget(value, 75)
    assert truncated
    assert len(response.encode("utf-8")) == 75
    parsed = json.loads(response)
    assert value["text"].startswith(parsed["text"])


def test_browsecompplus_fails_if_recorded_budget_cannot_fit_response_envelope():
    with pytest.raises(ValueError, match="too small for the retrieval response envelope"):
        _json_result_with_byte_budget([{"docid": "id", "score": 1.0, "text": "actual"}], 1)


def test_browsecompplus_requires_an_explicit_index_path(tmp_path):
    with pytest.raises(ValueError, match="requires --browsecomp-index-path"):
        DomainBackend("research_deep_search", tmp_path / "research", research_mode="browsecompplus")


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
