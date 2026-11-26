"""Placement of the mock LLM for one replay run."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping

SHARED = "shared"
SIDECAR = "sidecar"
PER_AGENT = "per_agent"
TOPOLOGIES = (SHARED, SIDECAR, PER_AGENT)
SIDECAR_ENDPOINT = "http://127.0.0.1:8080/v1"


def mock_topology(env: Mapping[str, str] | None = None) -> str:
    """Read the requested mock placement."""
    value = (env or os.environ).get("TRACE_REPLAY_MOCK_TOPOLOGY", SIDECAR).strip().lower()
    if value not in TOPOLOGIES:
        raise ValueError(f"unsupported mock topology {value!r}; expected one of {TOPOLOGIES}")
    return value


def mock_plan(topology: str, namespace: str, prefix: str, agents: int) -> MockPlan:
    """Name the mock Deployments that serve the agents of one run."""
    if topology not in TOPOLOGIES:
        raise ValueError(f"unsupported mock topology {topology!r}; expected one of {TOPOLOGIES}")
    if topology == SIDECAR:
        return MockPlan(topology, namespace, prefix)
    if topology == SHARED:
        return MockPlan(topology, namespace, prefix, (f"{prefix}-mock-llm",))
    return MockPlan(
        topology,
        namespace,
        prefix,
        tuple(f"{prefix}-mock-llm-{index:02d}" for index in range(1, agents + 1)),
    )


@dataclass(frozen=True)
class MockPlan:
    """Where each agent's mock runs and how it is reached."""

    topology: str
    namespace: str
    prefix: str
    services: tuple[str, ...] = ()

    @property
    def sidecar(self) -> bool:
        """Whether the mock runs inside each agent pod."""
        return self.topology == SIDECAR

    def pod(self, index: int) -> str:
        """Mock Deployment that serves an agent slot."""
        return self.services[0] if self.topology == SHARED else self.services[index - 1]

    def endpoint(self, index: int) -> str:
        """Model base URL for an agent slot."""
        if self.sidecar:
            return SIDECAR_ENDPOINT
        return f"http://{self.pod(index)}.{self.namespace}.svc.cluster.local:8080/v1"

    def recorder(self, index: int, agent: str) -> str:
        """Deployment that records mock edges for an agent slot."""
        return agent if self.sidecar else self.pod(index)
