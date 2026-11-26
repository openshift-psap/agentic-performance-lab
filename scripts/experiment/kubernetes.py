"""Cluster access, node placement, and pod capacity."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import threading
import time
import urllib.request
from collections import Counter, defaultdict
from typing import Any

from trace_replay_sim.cluster_planning import (
    validate_agent_count,
)

from .config import (
    BASE_JAEGER,
    BASE_OPENCLAW,
    DEFAULT_KUEUE_NAMESPACE,
    DEFAULT_KUEUE_WEBHOOK_SERVICE,
    LOG,
    READ_ONLY_OC_COMMANDS,
    STRICT_OWNERSHIP,
    assert_owned_namespace,
    assert_owned_target,
    is_transient_api_error,
    log_phase,
    parse_pod_quantity,
)


class Cluster:
    def __init__(
        self,
        kubeconfig: str | None,
        namespace: str,
        node: str,
        nodes: tuple[str, ...] | None = None,
        api_workers: int = 16,
    ):
        if api_workers < 1:
            raise ValueError("api_workers must be positive")
        self.env = os.environ.copy()
        if kubeconfig:
            self.env["KUBECONFIG"] = kubeconfig
        self.namespace = namespace
        self.node = node
        self.nodes = tuple(nodes or (node,))
        # A 100-agent run can have hundreds of concurrent polling and cgroup
        # operations.  Bound oc subprocesses so the local controller cannot
        # exhaust its file-descriptor limit while the Kubernetes workloads
        # remain fully concurrent.
        self._api_semaphore = threading.BoundedSemaphore(api_workers)

    def run(
        self, args: list[str], *, input_text: str | None = None, check: bool = True
    ) -> str:
        if STRICT_OWNERSHIP:
            mutating = {
                "apply",
                "create",
                "replace",
                "delete",
                "patch",
                "scale",
                "set",
                "label",
                "annotate",
                "exec",
                "cp",
            }
            if any(token in mutating for token in args):
                try:
                    namespace = args[args.index("-n") + 1]
                except (ValueError, IndexError) as exc:
                    raise RuntimeError(
                        f"strict ownership requires an explicit namespace: {args}"
                    ) from exc
                assert_owned_namespace(namespace)
        LOG.debug("oc %s", " ".join(args))
        command = next(
            (token for token in args if token in READ_ONLY_OC_COMMANDS), None
        )
        max_attempts = 4 if command is not None else 1
        # A 100-agent run must not leave the local controller blocked forever
        # on a single API-server read. Job waits intentionally keep their
        # Kubernetes timeout; point reads get a bounded client timeout.
        client_timeout = (
            30
            if command
            in {"get", "config", "whoami", "version", "api-resources", "cluster-info"}
            else None
        )
        result = None
        for attempt in range(max_attempts):
            try:
                with self._api_semaphore:
                    result = subprocess.run(
                        ["oc", *args],
                        input=input_text,
                        text=True,
                        capture_output=True,
                        env=self.env,
                        timeout=client_timeout,
                    )
            except subprocess.TimeoutExpired as exc:
                if attempt + 1 >= max_attempts:
                    raise RuntimeError(
                        f"oc {' '.join(args)} timed out after {client_timeout}s"
                    ) from exc
                delay = min(2**attempt, 8)
                LOG.warning(
                    "timeout on oc %s; retrying in %ss (%d/%d)",
                    " ".join(args),
                    delay,
                    attempt + 1,
                    max_attempts - 1,
                )
                time.sleep(delay)
                continue
            if (
                result.returncode == 0
                or not is_transient_api_error(result.stderr)
                or attempt + 1 >= max_attempts
            ):
                break
            delay = min(2**attempt, 8)
            LOG.warning(
                "transient Kubernetes API error on oc %s; retrying in %ss (%d/%d)",
                " ".join(args),
                delay,
                attempt + 1,
                max_attempts - 1,
            )
            time.sleep(delay)
        assert result is not None
        if check and result.returncode:
            raise RuntimeError(f"oc {' '.join(args)} failed: {result.stderr.strip()}")
        return result.stdout

    def delete_owned(self, namespace: str, kind: str, name: str) -> None:
        """Delete only an existing object carrying this run's ownership label."""
        assert_owned_target(namespace, name)
        raw = self.run(["-n", namespace, "get", kind, name], check=False).strip()
        if not raw:
            return
        current = json.loads(
            self.run(["-n", namespace, "get", kind, name, "-o", "json"])
        )
        labels = current.get("metadata", {}).get("labels") or {}
        if labels.get("managed-by") != "trace-replay-experiment":
            raise RuntimeError(f"refusing to delete unowned {kind} {namespace}/{name}")
        self.run(["-n", namespace, "delete", kind, name, "--ignore-not-found"])

    def json(self, args: list[str]) -> dict[str, Any]:
        return json.loads(self.run([*args, "-o", "json"]))

    def apply_json(self, value: dict[str, Any]) -> None:
        """Apply a run-owned object in the experiment namespace with retries.

        The Kueue admission webhook can briefly refuse otherwise valid
        idempotent writes.  All callers of this helper construct experiment
        resources in ``self.namespace``; retrying transient API failures here
        keeps a single webhook flap from aborting a long 100-agent run.
        """
        self.apply_json_in_with_retry(self.namespace, value)

    def replace_json(self, value: dict[str, Any]) -> None:
        """Replace an existing object without creating a huge last-applied annotation."""
        assert_owned_target(
            self.namespace, str(value.get("metadata", {}).get("name") or "")
        )
        self.run(
            ["-n", self.namespace, "replace", "-f", "-"], input_text=json.dumps(value)
        )

    def apply_json_in(self, namespace: str, value: dict[str, Any]) -> None:
        assert_owned_target(namespace, str(value.get("metadata", {}).get("name") or ""))
        self.run(["-n", namespace, "apply", "-f", "-"], input_text=json.dumps(value))

    def apply_json_in_with_retry(self, namespace: str, value: dict[str, Any]) -> None:
        """Retry an idempotent apply for an experiment-owned object.

        OpenShift admission webhooks can time out or refuse connections while
        the API remains healthy enough for a subsequent request.  This helper
        is used only for resources constructed by this experiment controller;
        it must not turn arbitrary cluster writes into an unbounded retry loop.
        """
        # Admission webhooks can remain unavailable for several minutes even
        # after the read-only preflight has succeeded.  Keep this longer
        # window limited to run-owned experiment resources.
        for attempt in range(30):
            try:
                self.apply_json_in(namespace, value)
                return
            except RuntimeError as exc:
                if not is_transient_api_error(str(exc)) or attempt >= 29:
                    raise
                delay = min(2**attempt, 30)
                name = str(value.get("metadata", {}).get("name") or "<unnamed>")
                LOG.warning(
                    "transient apply failed for %s; retrying in %ss (%d/29)",
                    name,
                    delay,
                    attempt + 1,
                )
                time.sleep(delay)

    def apply_yaml_in(self, namespace: str, value: str) -> None:
        if STRICT_OWNERSHIP:
            try:
                import yaml
            except ImportError as exc:
                raise RuntimeError(
                    "strict ownership requires PyYAML to inspect manifests"
                ) from exc
            for resource in yaml.safe_load_all(value):
                if resource:
                    assert_owned_target(
                        namespace, str(resource.get("metadata", {}).get("name") or "")
                    )
        self.run(["-n", namespace, "apply", "-f", "-"], input_text=value)

    def upsert_owned_configmap(self, namespace: str, value: dict[str, Any]) -> None:
        """Retry an idempotent owned ConfigMap operation across API flaps."""
        assert_owned_target(namespace, str(value.get("metadata", {}).get("name") or ""))
        for attempt in range(8):
            try:
                self._upsert_owned_configmap_once(namespace, value)
                return
            except RuntimeError as exc:
                if not is_transient_api_error(str(exc)) or attempt >= 7:
                    raise
                delay = min(2**attempt, 30)
                name = str(value.get("metadata", {}).get("name") or "<unnamed>")
                LOG.warning(
                    "transient ConfigMap operation failed for %s; retrying in %ss (%d/7)",
                    name,
                    delay,
                    attempt + 1,
                )
                time.sleep(delay)

    def _upsert_owned_configmap_once(
        self, namespace: str, value: dict[str, Any]
    ) -> None:
        """Create/replace an owned ConfigMap without client-side apply metadata.

        Session corpora can be hundreds of KiB; the client-side apply
        annotation would exceed Kubernetes' annotation limit even though the
        ConfigMap data itself is valid.
        """
        name = str(value.get("metadata", {}).get("name") or "")
        existing = self.run(
            ["-n", namespace, "get", "configmap", name], check=False
        ).strip()
        if not existing:
            try:
                self.run(
                    ["-n", namespace, "create", "-f", "-"], input_text=json.dumps(value)
                )
            except RuntimeError as exc:
                # A create request can be accepted by the API server even if
                # the client receives a timeout.  Confirm ownership before
                # surfacing the failure so a retry remains idempotent.
                if not is_transient_api_error(str(exc)):
                    raise
                # The create may also have been rejected before persistence,
                # or a concurrent cleanup may have removed the object between
                # the create and this confirmation read.  A NotFound here is
                # therefore still transient: re-run the idempotent create
                # instead of turning the confirmation probe into a fatal
                # error.
                try:
                    current = self.json(["-n", namespace, "get", "configmap", name])
                except RuntimeError as confirmation_exc:
                    if "not found" in str(confirmation_exc).lower():
                        raise exc
                    raise
                labels = current.get("metadata", {}).get("labels") or {}
                if labels.get("managed-by") != "trace-replay-experiment":
                    raise
            return
        current = self.json(["-n", namespace, "get", "configmap", name])
        labels = current.get("metadata", {}).get("labels") or {}
        if labels.get("managed-by") != "trace-replay-experiment":
            raise RuntimeError(
                f"refusing to modify unowned ConfigMap {namespace}/{name}"
            )
        # Reusing a run-owned corpus is intentionally idempotent.  Avoid a
        # full GET/replace round trip for large, unchanged payloads: repeated
        # rewrites create needless API-server and admission load and can make
        # a healthy cluster appear stalled during resume.
        desired_labels = value.get("metadata", {}).get("labels") or {}
        if (
            labels == desired_labels
            and current.get("data", {}) == value.get("data", {})
            and current.get("binaryData") == value.get("binaryData")
        ):
            return
        metadata = current.setdefault("metadata", {})
        metadata["labels"] = desired_labels or labels
        annotations = metadata.get("annotations") or {}
        annotations.pop("kubectl.kubernetes.io/last-applied-configuration", None)
        if annotations:
            metadata["annotations"] = annotations
        else:
            metadata.pop("annotations", None)
        current["data"] = value.get("data", {})
        if value.get("binaryData") is not None:
            current["binaryData"] = value["binaryData"]
        else:
            current.pop("binaryData", None)
        for key in (
            "uid",
            "creationTimestamp",
            "managedFields",
        ):  # resourceVersion is required by replace
            metadata.pop(key, None)
        self.run(
            ["-n", namespace, "replace", "-f", "-"], input_text=json.dumps(current)
        )

    def upsert_owned_secret(self, namespace: str, value: dict[str, Any]) -> None:
        """Create/replace an experiment-owned Secret without logging its data."""
        name = str(value.get("metadata", {}).get("name") or "")
        assert_owned_target(namespace, name)
        existing = self.run(
            ["-n", namespace, "get", "secret", name], check=False
        ).strip()
        if not existing:
            self.run(
                ["-n", namespace, "create", "-f", "-"], input_text=json.dumps(value)
            )
            return
        current = self.json(["-n", namespace, "get", "secret", name])
        labels = current.get("metadata", {}).get("labels") or {}
        if labels.get("managed-by") != "trace-replay-experiment":
            raise RuntimeError(f"refusing to modify unowned Secret {namespace}/{name}")
        metadata = current.setdefault("metadata", {})
        metadata["labels"] = value.get("metadata", {}).get("labels", labels)
        annotations = metadata.get("annotations") or {}
        annotations.pop("kubectl.kubernetes.io/last-applied-configuration", None)
        if annotations:
            metadata["annotations"] = annotations
        else:
            metadata.pop("annotations", None)
        current["type"] = value.get("type", current.get("type", "Opaque"))
        current["data"] = value.get("data", {})
        for key in ("uid", "creationTimestamp", "managedFields"):
            metadata.pop(key, None)
        self.run(
            ["-n", namespace, "replace", "-f", "-"], input_text=json.dumps(current)
        )

    def wait_rollout(self, deployment: str) -> None:
        try:
            self.run(
                [
                    "-n",
                    self.namespace,
                    "rollout",
                    "status",
                    f"deployment/{deployment}",
                    "--timeout=600s",
                ]
            )
        except RuntimeError as exc:
            # During a webhook/API flap, `oc rollout status` can report a
            # stale progress-deadline failure even though the Deployment has
            # already converged.  Accept only the unambiguous healthy state;
            # otherwise preserve the original failure.
            if "progress deadline" not in str(exc).lower():
                raise
            current = self.json(["-n", self.namespace, "get", "deployment", deployment])
            desired = int((current.get("spec") or {}).get("replicas") or 0)
            status = current.get("status") or {}
            ready = int(status.get("readyReplicas") or 0)
            updated = int(status.get("updatedReplicas") or 0)
            available = int(status.get("availableReplicas") or 0)
            if (
                desired
                and ready >= desired
                and updated >= desired
                and available >= desired
            ):
                LOG.warning(
                    "accepting converged Deployment %s after stale progress-deadline status "
                    "(ready=%d updated=%d available=%d desired=%d)",
                    deployment,
                    ready,
                    updated,
                    available,
                    desired,
                )
                return
            raise

    def wait_rollout_in(self, namespace: str, deployment: str) -> None:
        self.run(
            [
                "-n",
                namespace,
                "rollout",
                "status",
                f"deployment/{deployment}",
                "--timeout=600s",
            ]
        )


def _wait_for_job_success(
    cluster: Cluster, namespace: str, job: str, timeout: float = 900.0
) -> None:
    """Wait for a one-shot setup Job without creating a watch stream."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        status = cluster.json(["-n", namespace, "get", "job", job]).get("status") or {}
        if int(status.get("succeeded") or 0) > 0:
            return
        if int(status.get("failed") or 0) > 0 or any(
            condition.get("type") == "Failed" and condition.get("status") == "True"
            for condition in status.get("conditions") or []
        ):
            logs = cluster.run(
                ["-n", namespace, "logs", f"job/{job}", "--all-containers=true"],
                check=False,
            ).strip()
            raise RuntimeError(f"Job {namespace}/{job} failed:\n{logs[-8000:]}")
        time.sleep(3)
    raise RuntimeError(f"timed out waiting for Job {namespace}/{job}")


def _free_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_jaeger(
    base: str, process: subprocess.Popen[str], timeout: float = 30.0
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"Jaeger port-forward exited with status {process.returncode}"
            )
        try:
            with urllib.request.urlopen(f"{base}/api/services", timeout=2):
                return
        except Exception:
            time.sleep(0.5)
    raise RuntimeError(f"Jaeger port-forward did not become ready: {base}")


def _preflight_pod_capacity(
    cluster: Cluster,
    agent_nodes: list[str] | tuple[str, ...],
    additional_pods_by_node: dict[str, int] | None = None,
) -> None:
    """Reject a run that would exceed a node's pod budget before mutation.

    A concurrent trace uses one OpenClaw pod, one trace Job pod, and one
    OpenShell sandbox pod. ``additional_pods_by_node`` accounts for
    run-scoped services that have not been created yet. This is deliberately
    conservative and counts only Pending/Running pods; completed Jobs do not
    consume scheduler capacity.
    """
    grouped: dict[str, int] = {}
    for node in agent_nodes:
        grouped[node] = grouped.get(node, 0) + 1
    failures: list[str] = []
    for node in additional_pods_by_node or {}:
        grouped.setdefault(node, 0)
    for node, agent_count in grouped.items():
        node_obj = cluster.json(["get", "node", node])
        allocatable = node_obj.get("status", {}).get("allocatable", {})
        raw_limit = allocatable.get("pods") or node_obj.get("status", {}).get(
            "capacity", {}
        ).get("pods")
        try:
            pod_limit = parse_pod_quantity(raw_limit)
        except (TypeError, ValueError):
            LOG.warning(
                "preflight: unable to read pod capacity for %s; skipping pod-budget check",
                node,
            )
            continue
        pods = cluster.json(
            ["get", "pods", "-A", "--field-selector", f"spec.nodeName={node}"]
        )
        active = sum(
            1
            for item in pods.get("items", [])
            if str(item.get("status", {}).get("phase") or "") in {"Pending", "Running"}
        )
        required = agent_count * 3 + int((additional_pods_by_node or {}).get(node, 0))
        remaining = pod_limit - active
        LOG.info(
            "preflight: node=%s active_pods=%d pod_limit=%d required_for_run=%d remaining=%d",
            node,
            active,
            pod_limit,
            required,
            remaining,
        )
        if active + required > pod_limit:
            failures.append(
                f"{node}: {active} active pods + {required} required pods exceeds allocatable pod limit {pod_limit}"
            )
    if failures:
        raise RuntimeError(
            "pod-capacity preflight failed before cluster mutation:\n- "
            + "\n- ".join(failures)
        )


def _capacity_aware_agent_assignments(
    cluster: Cluster,
    agent_count: int,
    target_nodes: tuple[str, ...],
    additional_pods_by_node: dict[str, int] | None = None,
) -> list[str]:
    """Assign slots using current pod headroom without evicting anything.

    The two-node run can start with different unrelated pod occupancy. A fixed
    50/50 split can therefore fail even when the aggregate headroom is enough.
    This read-only calculation reserves the same three active pods per agent
    plus any not-yet-created run-scoped pods supplied by the caller.
    It never changes scheduling, pods, nodes, or quotas.
    """
    remaining: dict[str, int] = {}
    for node in target_nodes:
        node_obj = cluster.json(["get", "node", node])
        allocatable = node_obj.get("status", {}).get("allocatable", {})
        raw_limit = allocatable.get("pods") or node_obj.get("status", {}).get(
            "capacity", {}
        ).get("pods")
        try:
            pod_limit = parse_pod_quantity(raw_limit)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"unable to read pod capacity for approved node {node}"
            ) from exc
        pods = cluster.json(
            ["get", "pods", "-A", "--field-selector", f"spec.nodeName={node}"]
        )
        active = sum(
            1
            for item in pods.get("items", [])
            if str(item.get("status", {}).get("phase") or "") in {"Pending", "Running"}
        )
        additional = int((additional_pods_by_node or {}).get(node, 0))
        remaining[node] = (pod_limit - active - additional) // 3
        LOG.info(
            "preflight placement: node=%s active_pods=%d pod_limit=%d safe_agent_slots=%d",
            node,
            active,
            pod_limit,
            max(0, remaining[node]),
        )
    if sum(max(0, count) for count in remaining.values()) < agent_count:
        raise RuntimeError(
            "insufficient non-destructive pod headroom for the requested run: "
            f"requested={agent_count}, safe_slots="
            + str(sum(max(0, count) for count in remaining.values()))
        )
    if any(remaining[node] < 1 for node in target_nodes):
        raise RuntimeError(
            "each approved target node must have at least one safe agent slot"
        )

    assignments: list[str] = []
    cursor = 0
    while len(assignments) < agent_count:
        for _ in target_nodes:
            node = target_nodes[cursor % len(target_nodes)]
            cursor += 1
            if remaining[node] <= 0:
                continue
            assignments.append(node)
            remaining[node] -= 1
            break
        else:
            raise RuntimeError("capacity-aware placement exhausted unexpectedly")
    return assignments


def _balance_workloads_across_nodes(
    queue: list[dict[str, Any]],
    capacity_assignments: list[str],
) -> list[str]:
    """Permute safe node slots so each workload is represented on each node.

    The capacity planner determines how many agent slots each approved node
    can host. A simple round-robin assignment to the interleaved workload
    queue puts every SWE trace on one node in a two-node run. Reordering the
    *same* safe slots removes that experiment-design confound.
    """
    if len(queue) != len(capacity_assignments):
        raise ValueError("one queue entry is required per agent slot")
    remaining = Counter(capacity_assignments)
    original = Counter(capacity_assignments)
    workload_node_counts: dict[str, Counter[str]] = defaultdict(Counter)
    assigned: list[str] = []
    for item in queue:
        workload = str(item.get("workload") or "")
        node = min(
            (candidate for candidate, slots in remaining.items() if slots > 0),
            key=lambda candidate: (
                workload_node_counts[workload][candidate] / original[candidate],
                (original[candidate] - remaining[candidate]) / original[candidate],
                candidate,
            ),
        )
        assigned.append(node)
        remaining[node] -= 1
        workload_node_counts[workload][node] += 1
    return assigned


def _preflight_kueue_admission(
    cluster: Cluster,
    namespace: str = DEFAULT_KUEUE_NAMESPACE,
    webhook_service: str = DEFAULT_KUEUE_WEBHOOK_SERVICE,
) -> None:
    """Verify the Job admission path before creating any run-owned objects.

    The benchmark creates real Kubernetes Jobs. On this cluster those Jobs are
    admitted through Kueue's mutating webhook. A Running controller pod is not
    sufficient: the Service must have a ready endpoint and the controller
    pod must report Ready. This read-only gate prevents a 100-agent run from
    producing a partial corpus when admission is already broken.
    """
    service = cluster.json(["-n", namespace, "get", "service", webhook_service])
    if not service.get("spec", {}).get("clusterIP"):
        raise RuntimeError(
            f"Kueue admission preflight failed: service {namespace}/{webhook_service} has no ClusterIP"
        )

    endpoint_addresses: list[str] = []
    endpoints = cluster.json(["-n", namespace, "get", "endpoints", webhook_service])
    for subset in endpoints.get("subsets") or []:
        endpoint_addresses.extend(
            str(address.get("ip"))
            for address in subset.get("addresses") or []
            if address.get("ip")
        )
    if not endpoint_addresses:
        # EndpointSlice is the preferred API on newer Kubernetes versions.
        slices = cluster.json(
            [
                "-n",
                namespace,
                "get",
                "endpointslice",
                "-l",
                f"kubernetes.io/service-name={webhook_service}",
            ]
        )
        for item in slices.get("items") or []:
            for endpoint in item.get("endpoints") or []:
                conditions = endpoint.get("conditions") or {}
                if conditions.get("ready") is False:
                    continue
                endpoint_addresses.extend(
                    str(address) for address in endpoint.get("addresses") or []
                )
    if not endpoint_addresses:
        raise RuntimeError(
            f"Kueue admission preflight failed: {namespace}/{webhook_service} has no ready endpoints; "
            "refusing to create trace Jobs"
        )

    pods = cluster.json(
        ["-n", namespace, "get", "pods", "-l", "control-plane=controller-manager"]
    )
    ready_pods = []
    for pod in pods.get("items") or []:
        statuses = (pod.get("status") or {}).get("containerStatuses") or []
        if statuses and all(status.get("ready") is True for status in statuses):
            ready_pods.append(str((pod.get("metadata") or {}).get("name") or ""))
    if not ready_pods:
        raise RuntimeError(
            f"Kueue admission preflight failed: no Ready controller-manager pod in {namespace}; "
            "refusing to create trace Jobs"
        )
    LOG.info(
        "preflight: Kueue admission healthy service=%s/%s endpoints=%d ready_controllers=%s",
        namespace,
        webhook_service,
        len(endpoint_addresses),
        ",".join(name for name in ready_pods if name) or "unknown",
    )


def preflight(
    cluster: Cluster,
    agents: int,
    system_ns: str,
    openshell_ns: str,
    *,
    allow_cross_node: bool = False,
    agent_nodes: list[str] | tuple[str, ...] | None = None,
    kueue_namespace: str = DEFAULT_KUEUE_NAMESPACE,
    kueue_webhook_service: str = DEFAULT_KUEUE_WEBHOOK_SERVICE,
) -> None:
    log_phase(
        "preflight: validating cluster, target nodes and system services "
        f"({', '.join(cluster.nodes)})"
    )
    context = cluster.run(["config", "current-context"])
    user = cluster.run(["whoami"])
    _preflight_kueue_admission(cluster, kueue_namespace, kueue_webhook_service)
    for target_node in cluster.nodes:
        node = cluster.json(["get", "node", target_node])
        ready = any(
            c.get("type") == "Ready" and c.get("status") == "True"
            for c in node.get("status", {}).get("conditions", [])
        )
        if not ready or (
            node.get("spec", {}).get("unschedulable") and not allow_cross_node
        ):
            raise RuntimeError(f"target node is not ready/schedulable: {target_node}")
    LOG.info(
        "context=%s user=%s nodes=%s",
        context.strip(),
        user.strip(),
        ",".join(cluster.nodes),
    )
    cluster.wait_rollout_in(system_ns, BASE_JAEGER)
    gateway_statefulsets = cluster.run(
        [
            "-n",
            openshell_ns,
            "get",
            "statefulset",
            "-l",
            "app.kubernetes.io/name=openshell",
            "--no-headers",
        ],
        check=False,
    )
    if not gateway_statefulsets.strip():
        raise RuntimeError(
            f"no OpenShell gateway StatefulSet found in namespace {openshell_ns}"
        )
    services = cluster.run(
        ["-n", cluster.namespace, "get", "svc", BASE_OPENCLAW], check=False
    )
    if not services.strip():
        raise RuntimeError(
            f"base {BASE_OPENCLAW} service is missing; refusing to invent a deployment template"
        )
    validate_agent_count(agents)
    if agent_nodes is not None and not allow_cross_node:
        _preflight_pod_capacity(cluster, agent_nodes)


def _choose_safe_auxiliary_node(cluster: Cluster, target_nodes: tuple[str, ...]) -> str:
    """Choose an approved node for short-lived controller-created Jobs.

    Agent and sandbox placement remains tied to each agent's recorded node.
    Temporary driver/staging Jobs do not mount those RWO workspaces and can
    therefore run on any healthy approved node.  Do not bypass a disk-pressure
    taint: choose another approved node instead, or fail before creating more
    workload if none is safe.
    """
    blocked: list[str] = []
    for node_name in target_nodes:
        node = cluster.json(["get", "node", node_name])
        ready = any(
            condition.get("type") == "Ready" and condition.get("status") == "True"
            for condition in node.get("status", {}).get("conditions", [])
        )
        taints = node.get("spec", {}).get("taints") or []
        disk_blocked = any(
            taint.get("key") == "node.kubernetes.io/disk-pressure"
            and taint.get("effect") in {"NoSchedule", "NoExecute"}
            for taint in taints
        )
        if ready and not disk_blocked:
            return node_name
        blocked.append(node_name)
    raise RuntimeError(
        "no approved healthy node is available for temporary experiment Jobs; "
        f"disk-pressure or readiness blocked: {', '.join(blocked)}"
    )


def _retry_cluster_read_stage(
    label: str,
    operation: Any,
    timeout_seconds: float,
    interval_seconds: float,
) -> Any:
    """Keep read-only preparation alive while the API/Kueue path recovers."""
    deadline = time.monotonic() + timeout_seconds
    attempt = 0
    while True:
        try:
            return operation()
        except RuntimeError as exc:
            message = str(exc).lower()
            retryable = (
                is_transient_api_error(message)
                or "timed out after" in message
                or "context deadline" in message
                or "kueue admission preflight failed" in message
                or "no ready endpoints" in message
            )
            if not retryable or time.monotonic() >= deadline:
                raise
            attempt += 1
            remaining = max(0.0, deadline - time.monotonic())
            delay = min(interval_seconds, remaining)
            log_phase(
                f"{label}: cluster read not ready; retrying in {delay:.1f}s "
                f"(attempt {attempt}, {remaining:.0f}s remaining)"
            )
            time.sleep(delay)


def _wait_for_phase(
    cluster: Cluster,
    namespace: str,
    kind: str,
    name: str,
    expected: str,
    timeout: int = 600,
) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        obj = cluster.json(["-n", namespace, "get", kind, name])
        phase = str(obj.get("status", {}).get("phase") or "")
        if phase == expected:
            return
        if phase in {"Failed", "Unknown"}:
            raise RuntimeError(f"{kind}/{name} entered unexpected phase {phase}")
        time.sleep(5)
    raise RuntimeError(f"timed out waiting for {kind}/{name} phase={expected}")


def _wait_for_pvc_bound(
    cluster: Cluster, namespace: str, pvc_name: str, timeout: int = 600
) -> None:
    """Wait for a PVC to bind, including WaitForFirstConsumer storage classes."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        phase = str(
            cluster.json(["-n", namespace, "get", "pvc", pvc_name])
            .get("status", {})
            .get("phase")
            or ""
        )
        if phase == "Bound":
            return
        if phase == "Lost":
            raise RuntimeError(
                f"AppWorld PVC {namespace}/{pvc_name} entered Lost state"
            )
        time.sleep(5)
    raise RuntimeError(
        f"timed out waiting for AppWorld PVC {namespace}/{pvc_name} to bind"
    )
