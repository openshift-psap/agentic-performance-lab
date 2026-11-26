import pytest

from trace_replay_sim.mock_plan import (
    PER_AGENT,
    SHARED,
    SIDECAR,
    MockPlan,
    mock_plan,
    mock_topology,
)


def test_default_topology_is_sidecar():
    assert mock_topology({}) == SIDECAR


@pytest.mark.parametrize("value", ["shared", " Shared ", "per_agent", "PER_AGENT"])
def test_topology_is_normalized(value):
    assert mock_topology({"TRACE_REPLAY_MOCK_TOPOLOGY": value}) in {SHARED, PER_AGENT}


def test_unknown_topology_is_rejected():
    with pytest.raises(ValueError, match="unsupported mock topology"):
        mock_topology({"TRACE_REPLAY_MOCK_TOPOLOGY": "remote"})
    with pytest.raises(ValueError, match="unsupported mock topology"):
        mock_plan("remote", "ns", "run", agents=2)


def test_shared_plan_creates_one_mock():
    plan = mock_plan(SHARED, "ns", "run", agents=8)
    assert plan.services == ("run-mock-llm",)
    assert plan.endpoint(1) == plan.endpoint(8) == "http://run-mock-llm.ns.svc.cluster.local:8080/v1"


def test_per_agent_plan_creates_one_mock_per_agent():
    plan = mock_plan(PER_AGENT, "ns", "run", agents=3)
    assert plan.services == ("run-mock-llm-01", "run-mock-llm-02", "run-mock-llm-03")
    assert plan.endpoint(2) == "http://run-mock-llm-02.ns.svc.cluster.local:8080/v1"


def test_sidecar_plan_creates_no_mock_deployment():
    plan = mock_plan(SIDECAR, "ns", "run", agents=4)
    assert plan.services == ()
    assert plan.sidecar
    assert plan.endpoint(3) == "http://127.0.0.1:8080/v1"


def test_recorder_follows_the_mock_placement():
    agent = "run-openclaw-01"
    assert mock_plan(SIDECAR, "ns", "run", 2).recorder(1, agent) == agent
    assert mock_plan(SHARED, "ns", "run", 2).recorder(1, agent) == "run-mock-llm"
    assert mock_plan(PER_AGENT, "ns", "run", 2).recorder(2, agent) == "run-mock-llm-02"


def test_plan_holds_its_namespace_and_prefix():
    plan = MockPlan(PER_AGENT, "ns", "run", ("run-mock-llm-01",))
    assert (plan.namespace, plan.prefix, plan.sidecar) == ("ns", "run", False)
