"""Experiment lifecycle: preflight, deployment, execution, and reporting."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from trace_replay_sim.backend_registry import (
    BackendConfigurationError,
    load_backends,
    parse_workload_plan,
    plan_payload,
    validate_backends,
)
from trace_replay_sim.cluster_planning import balanced_node_assignments
from trace_replay_sim.mock_plan import MockPlan, mock_topology
from trace_replay_sim.swe_manifest import (
    SweManifestError,
    load_swe_manifest,
    manifest_sha256,
)
from trace_replay_sim.workloads import WORKLOADS

from .agent_manager import (
    _collect_owned_agent_bindings,
    _ensure_openshell_client_secret,
    _resume_agent_layout,
    deploy_agents,
    ensure_driver_config,
    ensure_mock_llm_config,
    verify_agent_mock_endpoints,
)
from .cleanup import cleanup_trace_resources
from .config import (
    LOG,
    OWNED_NAMESPACES,
    RESOURCE_PREFIX,
    ROOT,
    STRICT_OWNERSHIP,
    ConfigError,
    agent_name_prefix,
    log_phase,
    safe_name,
    workspace_name,
)
from .kubernetes import (
    Cluster,
    _balance_workloads_across_nodes,
    _capacity_aware_agent_assignments,
    _choose_safe_auxiliary_node,
    _preflight_pod_capacity,
    _retry_cluster_read_stage,
    preflight,
)
from .metrics import (
    JaegerPortForward,
    _start_shared_gateway_sampler,
    _stop_cgroup_samplers,
)
from .sandbox_manager import (
    _resolve_backend_service_endpoints,
    _rewrite_queue_backend_endpoints,
    ensure_appworld_data,
    ensure_shared_workspace,
    ensure_workload_backends,
    prewarm_pool_sandboxes,
    prewarm_swe_sandboxes,
    stage_swe_repositories,
    verify_appworld_data,
)
from .trace_manager import (
    _swe_command_map,
    build_queue,
    collect_trace,
    create_trace_job,
    prepare_appworld_corpus,
    prepare_workload_corpus,
)
from .validation import exit_code_warnings


@dataclass(frozen=True)
class Workloads:
    """Selected workloads and their real backends."""

    allocation: dict[str, int]
    backends: dict[str, Any]
    swe_manifest: dict[str, Any]

    @property
    def appworld_selected(self) -> bool:
        return (
            "personal_assistant" in self.allocation
            and self.backends["personal_assistant"].benchmark == "appworld"
        )


@dataclass
class Experiment:
    """Replay corpus, queue, and per-agent SWE staging state."""

    queue: list[dict[str, Any]]
    runtime_corpus: Path
    runtime_payload: dict[str, Any]
    tool_cpu_ids: tuple[int, ...]
    tool_cgroup_quota: str
    sandbox_cpu_limit: Any
    swe_task_images_by_agent: list[str | None]
    swe_tasks_by_agent: dict[int, dict[str, Any]]
    swe_items_by_agent: dict[int, dict[str, Any]]
    swe_script_dirs_by_agent: dict[int, Path]
    swe_command_maps_by_agent: list[dict[str, list[str]] | None]


@dataclass(frozen=True)
class Targets:
    """Placement and storage targets for this run."""

    nodes: tuple[str, ...]
    workspace_pvcs: dict[str, str]


@dataclass(frozen=True)
class RunIdentity:
    """Names derived from the output directory."""

    run_hash: str
    run_stem: str
    backend_run_id: str
    auxiliary_job_node: str


@dataclass(frozen=True)
class Storage:
    """Workspace naming and the PVC backing each node."""

    workspace_prefix: str
    agent_prefix: str
    workspace_names: list[str]
    pvcs_by_node: dict[str, str]


@dataclass(frozen=True)
class ResumeLayout:
    """Owned fleet slots preserved by a partial resume."""

    agent_nodes: list[str]
    pending: list[int]
    missing: list[int]


@dataclass
class Fleet:
    """Deployed agents and their mock LLM wiring."""

    agent_nodes: list[str]
    agent_bindings: list[tuple[str, str, str]]
    mock: MockPlan
    warm_pool_size: int
    warm_slots_by_node: dict[str, list[int]]


def run(args: argparse.Namespace) -> int:
    """Execute one controller run."""
    if args.execute and not args.backends_only:
        _acquire_controller_lock(args)
    cluster = Cluster(
        args.kubeconfig,
        args.namespace,
        args.target_nodes[0],
        args.target_nodes,
        args.cluster_api_workers,
    )
    workloads = _select_workloads(args)
    targets = Targets(args.target_nodes, args.workspace_pvcs)
    if args.backends_only:
        return _run_backends_only(args, cluster, workloads, targets)
    experiment = _prepare_queue(args, workloads)
    if args.plan:
        return _write_plan(args, workloads, experiment)
    resume = _resume_layout(args, cluster, experiment)
    _capacity_preflight(args, cluster, resume)
    identity = _run_identity(args, cluster)
    appworld_endpoint = _appworld_tool_endpoint(args, workloads)
    if workloads.appworld_selected and not args.resume_existing_run:
        _stage_appworld_data(args, cluster, workloads, targets)
    agent_nodes = _reserve_backends(
        args, cluster, workloads, experiment, identity, resume
    )
    backend_endpoints = _resolve_backends(args, cluster, workloads, experiment)
    if workloads.appworld_selected:
        appworld_endpoint = backend_endpoints["personal_assistant"] + "/execute"
    node_groups = write_deployment_plan(
        args,
        workloads.allocation,
        experiment.swe_task_images_by_agent,
        len(experiment.queue),
        agent_nodes,
    )
    mock = ensure_mock_llm_config(
        cluster,
        experiment.runtime_corpus,
        args.system_namespace,
        identity.run_stem,
        args.agents,
        mock_topology(),
    )
    ensure_driver_config(cluster)
    client_secret = _ensure_openshell_client_secret(
        cluster,
        args.openshell_client_secret_namespace or args.openshell_namespace,
        args.openshell_client_secret,
        f"{RESOURCE_PREFIX}-openshell-client-tls-{identity.run_hash}",
        ROOT / "results" / "mlperf-cpu-20261111" / "oidc-private" / "access-token",
    )
    storage = _provision_storage(args, cluster, targets, identity, node_groups)
    internal_endpoints = [
        backend_endpoints[workload] for workload in workloads.allocation
    ]
    fleet = _deploy_fleet(
        args,
        cluster,
        workloads,
        experiment,
        storage,
        mock,
        identity,
        agent_nodes,
        resume,
        internal_endpoints,
        appworld_endpoint,
        client_secret,
    )
    _stage_swe(args, cluster, experiment, storage, identity, fleet)
    fleet = _prewarm_pool(args, cluster, experiment, storage, identity, fleet)
    if args.deployment_only:
        log_phase(
            "deployment-only: agents and workspace wiring are ready; trace Jobs were not created"
        )
        return 0
    _execute(args, cluster, workloads, experiment, storage, fleet, identity)
    _write_experiment_json(args, experiment, fleet, agent_nodes)
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/analyze_multi_agent_results.py"),
            "--experiment",
            str(args.out),
        ],
        check=True,
    )
    log_phase(f"complete: results in {args.out}")
    return 0


def _acquire_controller_lock(args: argparse.Namespace) -> None:
    """Refuse to run two controllers against the same fleet."""
    lock_identity = args.reuse_existing_agent_prefix or str(args.out.resolve())
    lock_hash = hashlib.sha256(lock_identity.encode("utf-8")).hexdigest()[:16]
    lock_path = Path("/tmp") / f"trace-replay-controller-{lock_hash}.lock"
    handle = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.seek(0)
        owner = handle.read().strip() or "unknown"
        raise ConfigError(
            f"another controller already owns fleet {lock_identity!r} (lock owner {owner})"
        ) from None
    handle.seek(0)
    handle.truncate()
    handle.write(f"pid={os.getpid()} out={args.out}\n")
    handle.flush()


def _select_workloads(args: argparse.Namespace) -> Workloads:
    """Resolve the workload allocation and its backend definitions."""
    try:
        allocation = (
            {"personal_assistant": args.agents}
            if args.appworld and not args.workload_plan
            else parse_workload_plan(
                args.workload_plan, args.agents, args.traces_per_workload
            )
        )
        backends = load_backends(args.backend_manifest)
        validate_backends(backends, list(allocation))
        if STRICT_OWNERSHIP:
            owned_prefix = f"{RESOURCE_PREFIX}-"
            for workload in allocation:
                spec = backends[workload]
                if (
                    spec.namespace not in OWNED_NAMESPACES
                    or not spec.name.startswith(owned_prefix)
                    or not spec.service.startswith(owned_prefix)
                ):
                    raise ConfigError(
                        f"strict ownership refuses backend {workload}: {spec.namespace}/{spec.name}"
                    )
        swe_manifest = (
            load_swe_manifest(args.swe_task_manifest)
            if allocation.get("software_engineering_qa", 0) > 0
            else {}
        )
    except (
        BackendConfigurationError,
        SweManifestError,
        OSError,
        json.JSONDecodeError,
    ) as exc:
        raise ConfigError(str(exc)) from exc
    return Workloads(allocation, backends, swe_manifest)


def _run_backends_only(
    args: argparse.Namespace, cluster: Cluster, workloads: Workloads, targets: Targets
) -> int:
    """Deploy and health-check the selected workload backends, then stop."""
    _cluster_preflight(args, cluster)
    if workloads.appworld_selected:
        _stage_appworld_data(args, cluster, workloads, targets)
    ensure_workload_backends(
        cluster,
        workloads.backends,
        workloads.allocation,
        "backends-only",
        appworld_task_map=args.appworld_task_map
        if workloads.appworld_selected
        else None,
        appworld_data_pvc=args.appworld_data_pvc
        if workloads.appworld_selected
        else None,
    )
    log_phase(
        "backend gate: all selected workload backends are ready; no agents or sandboxes were created"
    )
    return 0


def _prepare_queue(args: argparse.Namespace, workloads: Workloads) -> Experiment:
    """Build the replay queue and the per-agent SWE staging state."""
    args.out.mkdir(parents=True, exist_ok=True)
    corpus = args.corpus
    if args.appworld:
        if set(workloads.allocation) != {"personal_assistant"}:
            raise ConfigError(
                "--appworld is only valid for an AppWorld-only allocation"
            )
        endpoint = (
            args.appworld_endpoint
            or f"{workloads.backends['personal_assistant'].base_url}/execute"
        )
        corpus = prepare_appworld_corpus(
            args.corpus,
            args.out / "appworld-corpus.json",
            endpoint,
            args.appworld_task_map,
        )
    initial_queue = build_queue(
        corpus,
        args.out,
        workloads.allocation,
        args.session_id,
        args.selected_session_ids or None,
        args.max_traces,
        args.trace_offset,
    )
    if not initial_queue:
        raise ConfigError(f"session not found in corpus: {args.session_id}")
    selected_ids = {
        str(item["session"].get("session_id") or "") for item in initial_queue
    }
    if not args.appworld:
        corpus = prepare_workload_corpus(
            corpus,
            args.out / "workload-backend-corpus.json",
            workloads.backends,
            workloads.swe_manifest,
            selected_session_ids=selected_ids,
            tool_cpu_ids=args.tool_cpu_ids,
            tool_cgroup_quota=args.tool_cgroup_quota,
            swe_task_image_runtime=args.swe_sandbox_task_image,
        )
        queue = build_queue(
            corpus, args.out, workloads.allocation, session_ids=selected_ids
        )
    else:
        queue = initial_queue
    expected_queue = (
        sum(workloads.allocation.values())
        if args.session_id is None
        and not args.selected_session_ids
        and args.max_traces is None
        else None
    )
    if expected_queue is not None and len(queue) != expected_queue:
        raise ConfigError(
            f"workload plan requires exactly {expected_queue} traces, found {len(queue)}"
        )
    if args.require_traces is not None and len(queue) != args.require_traces:
        raise ConfigError(
            f"expected exactly {args.require_traces} traces, found {len(queue)}"
        )
    if args.selected_agent_indices and len(args.selected_agent_indices) != len(queue):
        raise ConfigError(
            "--agent-indices must contain exactly one agent index per selected trace"
        )
    # SWE repository transfer requires creating the OpenShell sandbox before
    # measurement. Such a slot cannot also be presented as a cold sandbox
    # sample, regardless of the requested partial warm-pool ratio.
    for position, item in enumerate(queue):
        slot = (
            (args.selected_agent_indices[position] - 1)
            if args.selected_agent_indices
            else ((position + args.agent_offset) % args.agents)
        )
        if (
            item["workload"] == "software_engineering_qa"
            and slot >= args.warm_pool_size
        ):
            raise ConfigError(
                f"SWE agent slot {slot + 1} must be in the warm pool because repository staging "
                "creates its sandbox before measurement; increase --warm-pool-size or change the cohort"
            )
    swe_task_images_by_agent: list[str | None] = [None] * args.agents
    swe_tasks_by_agent: dict[int, dict[str, Any]] = {}
    swe_items_by_agent: dict[int, dict[str, Any]] = {}
    swe_script_dirs_by_agent: dict[int, Path] = {}
    swe_command_maps_by_agent: list[dict[str, list[str]] | None] = [None] * args.agents
    swe_assignments: dict[int, str] = {}
    for position, item in enumerate(queue):
        binding_index = (
            (args.selected_agent_indices[position] - 1)
            if args.selected_agent_indices
            else ((position + args.agent_offset) % args.agents)
        )
        task = item["session"].get("swe_task") or {}
        if item["workload"] != "software_engineering_qa":
            continue
        image = str(task.get("image") or "")
        session_id = str(item["session"].get("session_id") or "")
        if not image:
            raise ConfigError(f"SWE trace {session_id} has no pinned task image")
        previous = swe_assignments.get(binding_index)
        if previous and previous != session_id:
            raise ConfigError(
                f"agent {binding_index + 1} would run multiple SWE repositories; "
                "use one agent per SWE trace so its OpenShell task image remains pinned"
            )
        swe_assignments[binding_index] = session_id
        swe_task_images_by_agent[binding_index] = image
        swe_tasks_by_agent[binding_index] = dict(task)
        swe_items_by_agent[binding_index] = item
        swe_command_maps_by_agent[binding_index] = _swe_command_map(item["session"])
        swe_script_dirs_by_agent[binding_index] = (
            args.out
            / "swe-tool-scripts"
            / hashlib.sha256(session_id.encode()).hexdigest()[:16]
        )
    if args.allow_cross_node and any(swe_command_maps_by_agent):
        raise ConfigError(
            "recorded SWE bash commands require the node-pinned OpenShell SSH adapter"
        )
    runtime_corpus = args.out / "runtime-replay-corpus.json"
    runtime_corpus.parent.mkdir(parents=True, exist_ok=True)
    runtime_payload = {
        "dataset": "Exgentic/agent-llm-traces",
        "tool_backend": "workload-backends",
        "backend_plan": plan_payload(workloads.backends, workloads.allocation),
        "sessions": [item["session"] for item in queue],
    }
    if workloads.swe_manifest:
        runtime_payload["swe_task_manifest"] = str(args.swe_task_manifest)
        runtime_payload["swe_task_manifest_sha256"] = manifest_sha256(
            args.swe_task_manifest
        )
        runtime_payload["swe_tasks"] = {
            session_id: session.get("swe_task")
            for session_id, session in (
                (str(item["session"].get("session_id") or ""), item["session"])
                for item in queue
            )
            if session.get("swe_task")
        }
    runtime_corpus.write_text(json.dumps(runtime_payload, indent=2), encoding="utf-8")
    (args.out / "backend_plan.json").write_text(
        json.dumps(plan_payload(workloads.backends, workloads.allocation), indent=2)
        + "\n",
        encoding="utf-8",
    )
    log_phase(
        f"queue: {len(queue)} traces "
        f"({', '.join(f'{workload}={count}' for workload, count in workloads.allocation.items())}), "
        f"{args.agents} agents"
    )
    return Experiment(
        queue=queue,
        runtime_corpus=runtime_corpus,
        runtime_payload=runtime_payload,
        tool_cpu_ids=args.tool_cpu_ids,
        tool_cgroup_quota=args.tool_cgroup_quota,
        sandbox_cpu_limit=args.sandbox_cpu_limit,
        swe_task_images_by_agent=swe_task_images_by_agent,
        swe_tasks_by_agent=swe_tasks_by_agent,
        swe_items_by_agent=swe_items_by_agent,
        swe_script_dirs_by_agent=swe_script_dirs_by_agent,
        swe_command_maps_by_agent=swe_command_maps_by_agent,
    )


def write_deployment_plan(
    args: argparse.Namespace,
    allocation: dict[str, int],
    swe_task_images: list[str | None],
    traces: int,
    agent_nodes: list[str],
) -> dict[str, list[int]]:
    """Persist the agent-to-node placement plan and return groups by node."""
    node_groups: dict[str, list[int]] = {node: [] for node in args.target_nodes}
    warm_slots_by_node: dict[str, list[int]] = {node: [] for node in args.target_nodes}
    for index, node in enumerate(agent_nodes, start=1):
        node_groups[node].append(index)
        if index <= args.warm_pool_size:
            warm_slots_by_node[node].append(index)
    (args.out / "deployment_plan.json").write_text(
        json.dumps(
            {
                "schema": "agentic.trace-replay.deployment-plan.v2",
                "agents": args.agents,
                "traces": traces,
                "workload_allocation": allocation,
                "target_nodes": list(args.target_nodes),
                "agent_node_assignments": {
                    str(index): node for index, node in enumerate(agent_nodes, start=1)
                },
                "agent_indices_by_node": node_groups,
                "warm_pool_size": args.warm_pool_size,
                "warm_pool_slots_by_node": warm_slots_by_node,
                "workspace_access_mode": args.workspace_access_mode,
                "workspace_storage_class": args.workspace_storage_class,
                "controlled_node_placement": not args.allow_cross_node,
                "swe_task_images_by_agent": {
                    str(index + 1): image
                    for index, image in enumerate(swe_task_images)
                    if image
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return node_groups


def _write_plan(
    args: argparse.Namespace, workloads: Workloads, experiment: Experiment
) -> int:
    """Write the cluster-free deployment plan."""
    agent_nodes = list(balanced_node_assignments(args.agents, args.target_nodes))
    node_groups = write_deployment_plan(
        args,
        workloads.allocation,
        experiment.swe_task_images_by_agent,
        len(experiment.queue),
        agent_nodes,
    )
    for workload in WORKLOADS:
        LOG.info(
            "  %s: %d traces / %d agents / backend=%s",
            workload,
            sum(item["workload"] == workload for item in experiment.queue),
            workloads.allocation.get(workload, 0),
            workloads.backends[workload].name,
        )
    for node in args.target_nodes:
        LOG.info("  node %s: %d agent slots", node, len(node_groups[node]))
    return 0


def _resume_layout(
    args: argparse.Namespace, cluster: Cluster, experiment: Experiment
) -> ResumeLayout | None:
    """Inventory an owned partial fleet that this run resumes."""
    if not args.resume_existing_run:
        return None
    planned_resume_nodes: list[str] | None = None
    deployment_plan_path = args.out / "deployment_plan.json"
    if deployment_plan_path.exists():
        deployment_plan = json.loads(deployment_plan_path.read_text(encoding="utf-8"))
        assignments = deployment_plan.get("agent_node_assignments") or {}
        candidate = [
            str(assignments.get(str(index)) or "")
            for index in range(1, args.agents + 1)
        ]
        if len(candidate) == args.agents and all(
            node in args.target_nodes for node in candidate
        ):
            planned_resume_nodes = candidate
    agent_nodes, pending, missing = _retry_cluster_read_stage(
        "resume inventory",
        lambda: _resume_agent_layout(
            cluster,
            args.reuse_existing_agent_prefix,
            args.agents,
            args.resume_preferred_node,
            planned_resume_nodes,
        ),
        args.cluster_retry_timeout,
        args.cluster_retry_interval,
    )
    # Pending existing Deployments keep their recorded node selector; only
    # truly missing slots would require placement on the preferred node.
    moved_or_missing = set(missing) if planned_resume_nodes is None else set()
    unsupported_swe = sorted(
        index
        for index in moved_or_missing
        if (index - 1) in experiment.swe_tasks_by_agent
    )
    if unsupported_swe:
        raise RuntimeError(
            "refusing to move or recreate SWE agents during partial resume; "
            f"repository staging is tied to the original node: {unsupported_swe}"
        )
    log_phase(
        f"resume: preserving {args.agents - len(pending) - len(missing)} "
        f"owned agents; reconciling {len(pending)} pending slots and creating "
        f"{len(missing)} missing slots from the recorded placement plan"
    )
    return ResumeLayout(agent_nodes, pending, missing)


def _cluster_preflight(args: argparse.Namespace, cluster: Cluster) -> None:
    preflight(
        cluster,
        args.agents,
        args.system_namespace,
        args.openshell_namespace,
        allow_cross_node=args.allow_cross_node,
        kueue_namespace=args.kueue_namespace,
        kueue_webhook_service=args.kueue_webhook_service,
    )


def _capacity_preflight(
    args: argparse.Namespace,
    cluster: Cluster,
    resume: ResumeLayout | None,
) -> None:
    """Validate pod budget and immutable baseline services before mutation."""
    if resume is not None:
        resume_count = len(resume.pending) + len(resume.missing)
        if resume_count:
            resume_pending_nodes = (
                [resume.agent_nodes[index - 1] for index in resume.pending]
                if resume.agent_nodes
                else []
            )
            # Missing slots retain the recorded balanced placement whenever a
            # deployment plan is available. Charging every missing slot to the
            # preferred node makes a valid multi-node resume fail its
            # conservative pod-budget check before any mutation.
            resume_missing_nodes = (
                [resume.agent_nodes[index - 1] for index in resume.missing]
                if resume.agent_nodes
                else [args.resume_preferred_node] * len(resume.missing)
            )
            capacity_nodes = resume_pending_nodes + resume_missing_nodes
            additional_by_node = {
                node: capacity_nodes.count(node) for node in set(capacity_nodes)
            }
            _retry_cluster_read_stage(
                "resume capacity preflight",
                lambda: _preflight_pod_capacity(
                    cluster,
                    capacity_nodes,
                    additional_pods_by_node=additional_by_node,
                ),
                args.cluster_retry_timeout,
                args.cluster_retry_interval,
            )
    else:
        backend_capacity_allowance = {
            args.target_nodes[0]: 5
        }  # four backends + mock LLM
        backend_capacity_probe = _retry_cluster_read_stage(
            "capacity placement preflight",
            lambda: _capacity_aware_agent_assignments(
                cluster,
                args.agents,
                args.target_nodes,
                additional_pods_by_node=backend_capacity_allowance,
            ),
            args.cluster_retry_timeout,
            args.cluster_retry_interval,
        )
        _retry_cluster_read_stage(
            "pod capacity preflight",
            lambda: _preflight_pod_capacity(
                cluster,
                backend_capacity_probe,
                additional_pods_by_node=backend_capacity_allowance,
            ),
            args.cluster_retry_timeout,
            args.cluster_retry_interval,
        )
    _retry_cluster_read_stage(
        "cluster preflight",
        lambda: _cluster_preflight(args, cluster),
        args.cluster_retry_timeout,
        args.cluster_retry_interval,
    )


def _run_identity(args: argparse.Namespace, cluster: Cluster) -> RunIdentity:
    """Derive run-scoped names and pick the node for temporary Jobs."""
    # Temporary staging/trace-driver Jobs do not mount the agent's RWO
    # workspace. Keep them on one healthy approved node so a disk-pressure
    # taint on the agent's node cannot strand the whole run, while the agents
    # and their sandboxes remain on their original nodes.
    auxiliary_job_node = _choose_safe_auxiliary_node(cluster, args.target_nodes)
    log_phase(
        f"temporary experiment Jobs will use healthy approved node {auxiliary_job_node}"
    )
    run_hash = hashlib.sha256(str(args.out.resolve()).encode("utf-8")).hexdigest()[:8]
    run_stem = (
        f"{safe_name(args.out.name)[:24].rstrip('-')}-{run_hash}" or f"run-{run_hash}"
    )
    return RunIdentity(
        run_hash=run_hash,
        run_stem=run_stem,
        backend_run_id=run_stem[:30],
        auxiliary_job_node=auxiliary_job_node,
    )


def _appworld_tool_endpoint(
    args: argparse.Namespace, workloads: Workloads
) -> str | None:
    """Endpoint sandbox commands use to reach the AppWorld backend."""
    # Mixed-workload runs cannot use the AppWorld-only --appworld mode, but
    # their personal-assistant sandbox commands still call the AppWorld
    # backend. Keep the endpoint in the sandbox network policy whenever the
    # selected allocation contains AppWorld traces.
    if not workloads.appworld_selected:
        return None
    return (
        args.appworld_endpoint
        or f"{workloads.backends['personal_assistant'].base_url}/execute"
    )


def _stage_appworld_data(
    args: argparse.Namespace, cluster: Cluster, workloads: Workloads, targets: Targets
) -> None:
    """Stage or verify the AppWorld dataset the backend reads."""
    namespace = workloads.backends["personal_assistant"].namespace
    if args.skip_appworld_staging:
        verify_appworld_data(
            cluster, namespace, targets.nodes[0], args.appworld_data_pvc
        )
    else:
        ensure_appworld_data(
            cluster,
            namespace,
            targets.nodes[0],
            args.appworld_data_pvc,
            args.workspace_storage_class,
            args.appworld_data,
        )


def _reserve_backends(
    args: argparse.Namespace,
    cluster: Cluster,
    workloads: Workloads,
    experiment: Experiment,
    identity: RunIdentity,
    resume: ResumeLayout | None,
) -> list[str]:
    """Ensure workload backends are healthy and return the agent placement."""
    if resume is not None:
        for workload in workloads.allocation:
            spec = workloads.backends[workload]
            if not cluster.run(
                ["-n", spec.namespace, "get", "svc", spec.service], check=False
            ).strip():
                raise RuntimeError(
                    f"cannot resume: backend service is missing for {workload}: "
                    f"{spec.namespace}/{spec.service}"
                )
            if cluster.run(
                ["-n", spec.namespace, "get", "deployment", spec.name], check=False
            ).strip():
                cluster.wait_rollout_in(spec.namespace, spec.name)
        return list(resume.agent_nodes or [])
    ensure_workload_backends(
        cluster,
        workloads.backends,
        workloads.allocation,
        identity.backend_run_id,
        appworld_task_map=args.appworld_task_map
        if workloads.appworld_selected
        else None,
        appworld_data_pvc=args.appworld_data_pvc
        if workloads.appworld_selected
        else None,
    )
    additional_pods_by_node = {args.target_nodes[0]: 1}  # run-scoped mock LLM
    agent_nodes = _capacity_aware_agent_assignments(
        cluster,
        args.agents,
        args.target_nodes,
        additional_pods_by_node=additional_pods_by_node,
    )
    if (
        len(experiment.queue) == args.agents
        and not args.selected_agent_indices
        and args.agent_offset == 0
    ):
        agent_nodes = _balance_workloads_across_nodes(experiment.queue, agent_nodes)
    _preflight_pod_capacity(
        cluster,
        agent_nodes,
        additional_pods_by_node=additional_pods_by_node,
    )
    return agent_nodes


def _resolve_backends(
    args: argparse.Namespace,
    cluster: Cluster,
    workloads: Workloads,
    experiment: Experiment,
) -> dict[str, str]:
    """Rewrite sandbox commands to reach owned backends by ClusterIP."""
    # OpenShell sandboxes have an isolated network namespace and do not
    # resolve Kubernetes Service DNS names. Resolve only the selected,
    # experiment-owned Services to ClusterIPs and use those addresses for the
    # queued tool commands and the sandbox network policy. This avoids any
    # cluster-wide DNS or policy changes.
    endpoints = _resolve_backend_service_endpoints(
        cluster, workloads.backends, workloads.allocation
    )
    _rewrite_queue_backend_endpoints(
        experiment.queue,
        workloads.backends,
        endpoints,
        args.out / "swe-tool-scripts",
    )
    experiment.runtime_payload["sessions"] = [
        item["session"] for item in experiment.queue
    ]
    experiment.runtime_corpus.write_text(
        json.dumps(experiment.runtime_payload, indent=2), encoding="utf-8"
    )
    return endpoints


def _provision_storage(
    args: argparse.Namespace,
    cluster: Cluster,
    targets: Targets,
    identity: RunIdentity,
    node_groups: dict[str, list[int]],
) -> Storage:
    """Create or validate the workspace PVCs and derive workspace names."""
    # OpenShell limits workspace names to 19 characters. In strict mode the
    # workspace also prefixes the generated Sandbox resource, so retain the
    # experiment owner's name while leaving room for agent slot 100.
    default_workspace_prefix = (
        f"{RESOURCE_PREFIX}-{identity.run_hash[:7]}"
        if STRICT_OWNERSHIP
        else f"replay-{identity.run_hash}"
    )
    workspace_prefix = args.reuse_existing_workspace_prefix or default_workspace_prefix
    agent_prefix = args.reuse_existing_agent_prefix or agent_name_prefix(
        args.out.name, identity.run_hash
    )
    log_phase(
        f"deployment: workspace prefix={workspace_prefix} agent prefix={agent_prefix}"
    )
    workspace_names = [
        workspace_name(workspace_prefix, index) for index in range(1, args.agents + 1)
    ]
    # RWX uses one shared workspace across nodes. RWO uses one PVC per node,
    # which keeps every sandbox and its block volume on the same node and does
    # not depend on CephFS. A single user-supplied RWO PVC is intentionally
    # rejected for multi-node runs because it cannot be mounted safely on both
    # nodes.
    pvcs_by_node: dict[str, str] = {}
    if args.reuse_gateway_managed_workspaces:
        log_phase(
            "storage: reusing gateway-managed OpenShell workspaces (no controller PVC)"
        )
    elif args.workspace_access_mode == "ReadWriteMany":
        pvc_name = (
            args.workspace_pvc
            or f"{RESOURCE_PREFIX}-openshell-workspaces-{identity.run_hash}"
        )
        ensure_shared_workspace(
            cluster,
            args.openshell_namespace,
            pvc_name,
            args.workspace_storage_class,
            workspace_names,
            args.workspace_access_mode,
            target_node=targets.nodes[0],
            allow_shared=args.allow_shared_workspace_pvc,
        )
        pvcs_by_node = {node: pvc_name for node in targets.nodes}
    else:
        if len(targets.nodes) > 1 and args.workspace_pvc:
            raise ConfigError(
                "--workspace-pvc cannot be a single RWO PVC for a multi-node run; "
                "omit it for per-node PVCs"
            )
        if args.resume_existing_run or args.reuse_existing_agent_prefix:
            # A resumed or explicitly reused fleet already owns and mounts
            # these RWO PVCs. Validate the explicit node-to-PVC map and reuse
            # it as-is; launching a redundant prep Job can block behind the
            # existing attachment (or be evicted under node disk pressure).
            if set(targets.workspace_pvcs) != set(targets.nodes):
                raise RuntimeError(
                    "reusing ReadWriteOnce workspaces requires an explicit owned PVC for every target node"
                )
            for node in targets.nodes:
                pvc_name = targets.workspace_pvcs[node]
                pvc = cluster.json(
                    ["-n", args.openshell_namespace, "get", "pvc", pvc_name]
                )
                labels = pvc.get("metadata", {}).get("labels") or {}
                if labels.get("managed-by") != "trace-replay-experiment":
                    raise RuntimeError(
                        f"refusing to reuse unowned workspace PVC {args.openshell_namespace}/{pvc_name}"
                    )
                spec = pvc.get("spec", {})
                if "ReadWriteOnce" not in (spec.get("accessModes") or []) or (
                    args.workspace_storage_class
                    and spec.get("storageClassName") != args.workspace_storage_class
                ):
                    raise RuntimeError(
                        f"existing PVC {args.openshell_namespace}/{pvc_name} has incompatible workspace settings"
                    )
                if pvc.get("status", {}).get("phase") != "Bound":
                    raise RuntimeError(
                        f"existing workspace PVC {args.openshell_namespace}/{pvc_name} is not Bound"
                    )
                pvcs_by_node[node] = pvc_name
            log_phase(
                f"storage: reusing existing node-local workspaces ({len(pvcs_by_node)} PVCs)"
            )
        else:
            for node in targets.nodes:
                node_hash = hashlib.sha256(node.encode("utf-8")).hexdigest()[:6]
                pvc_name = (
                    targets.workspace_pvcs.get(node)
                    or args.workspace_pvc
                    or f"{RESOURCE_PREFIX}-openshell-workspaces-{identity.run_hash}-{node_hash}"
                )
                node_workspaces = [
                    workspace_names[index - 1] for index in node_groups[node]
                ]
                ensure_shared_workspace(
                    cluster,
                    args.openshell_namespace,
                    pvc_name,
                    args.workspace_storage_class,
                    node_workspaces,
                    args.workspace_access_mode,
                    target_node=node,
                    allow_shared=args.allow_shared_workspace_pvc,
                )
                pvcs_by_node[node] = pvc_name
    return Storage(workspace_prefix, agent_prefix, workspace_names, pvcs_by_node)


def _active_swe_images(
    args: argparse.Namespace, experiment: Experiment
) -> list[str | None] | None:
    """Task images used as sandbox images, when that mode is enabled."""
    return experiment.swe_task_images_by_agent if args.swe_sandbox_task_image else None


def _deploy_agents(
    args: argparse.Namespace,
    cluster: Cluster,
    experiment: Experiment,
    storage: Storage,
    mock: MockPlan,
    internal_endpoints: list[str],
    appworld_endpoint: str | None,
    client_secret: str,
    agent_nodes: list[str],
    *,
    reuse: bool,
    reconcile_indices: list[int] | None,
    reuse_existing_workspaces: bool,
) -> list[tuple[str, str, str]]:
    return deploy_agents(
        cluster,
        args.agents,
        reuse=reuse,
        workspace_prefix=storage.workspace_prefix,
        deployment_prefix=storage.agent_prefix,
        shared_workspace_pvcs=storage.pvcs_by_node,
        appworld_endpoint=appworld_endpoint,
        internal_api_endpoints=internal_endpoints,
        mock=mock,
        gateway_endpoint=args.openshell_gateway_endpoint,
        openshell_client_secret=client_secret,
        allow_cross_node=args.allow_cross_node,
        agent_nodes=agent_nodes,
        reconcile_indices=reconcile_indices,
        reuse_existing_workspaces=reuse_existing_workspaces,
        simulated_ttft_ms=args.simulated_ttft,
        simulated_itl_ms=args.simulated_itl,
        sandbox_cpu_limit=experiment.sandbox_cpu_limit,
        sandbox_images=_active_swe_images(args, experiment),
        swe_command_maps=experiment.swe_command_maps_by_agent,
    )


def _deploy_fleet(
    args: argparse.Namespace,
    cluster: Cluster,
    workloads: Workloads,
    experiment: Experiment,
    storage: Storage,
    mock: MockPlan,
    identity: RunIdentity,
    agent_nodes: list[str],
    resume: ResumeLayout | None,
    internal_endpoints: list[str],
    appworld_endpoint: str | None,
    client_secret: str,
) -> Fleet:
    """Create or reconcile the agent fleet and verify its mock routing."""
    if resume is None:
        agent_bindings = _deploy_agents(
            args,
            cluster,
            experiment,
            storage,
            mock,
            internal_endpoints,
            appworld_endpoint,
            client_secret,
            agent_nodes,
            reuse=args.reuse_agents or bool(args.reuse_existing_agent_prefix),
            reconcile_indices=args.selected_agent_indices or None,
            reuse_existing_workspaces=bool(args.reuse_existing_workspace_prefix),
        )
    else:
        agent_bindings = _resume_deploy(
            args,
            cluster,
            experiment,
            storage,
            mock,
            identity,
            resume,
            workloads,
            internal_endpoints,
            appworld_endpoint,
            client_secret,
        )
    verify_agent_mock_endpoints(
        cluster,
        agent_bindings,
        mock,
        args.selected_agent_indices or None,
    )
    return Fleet(
        agent_nodes=agent_nodes,
        agent_bindings=agent_bindings,
        mock=mock,
        warm_pool_size=min(args.warm_pool_size, len(agent_bindings)),
        warm_slots_by_node={},
    )


def _resume_deploy(
    args: argparse.Namespace,
    cluster: Cluster,
    experiment: Experiment,
    storage: Storage,
    mock: MockPlan,
    identity: RunIdentity,
    resume: ResumeLayout,
    workloads: Workloads,
    internal_endpoints: list[str],
    appworld_endpoint: str | None,
    client_secret: str,
) -> list[tuple[str, str, str]]:
    """Deploy missing slots and reconcile the existing owned fleet."""
    images = _active_swe_images(args, experiment)
    if resume.pending:
        deploy_agents(
            cluster,
            len(resume.pending),
            reuse=True,
            workspace_prefix=storage.workspace_prefix,
            deployment_prefix=storage.agent_prefix,
            shared_workspace_pvcs=storage.pvcs_by_node,
            appworld_endpoint=appworld_endpoint,
            internal_api_endpoints=internal_endpoints,
            mock=mock,
            gateway_endpoint=args.openshell_gateway_endpoint,
            openshell_client_secret=client_secret,
            allow_cross_node=args.allow_cross_node,
            agent_nodes=[resume.agent_nodes[index - 1] for index in resume.pending],
            agent_indices=resume.pending,
            simulated_ttft_ms=args.simulated_ttft,
            simulated_itl_ms=args.simulated_itl,
            sandbox_cpu_limit=experiment.sandbox_cpu_limit,
            sandbox_images=[images[index - 1] for index in resume.pending]
            if images is not None
            else None,
            swe_command_maps=[
                experiment.swe_command_maps_by_agent[index - 1]
                for index in resume.pending
            ],
        )
    if resume.missing:
        deploy_agents(
            cluster,
            len(resume.missing),
            reuse=False,
            workspace_prefix=storage.workspace_prefix,
            deployment_prefix=storage.agent_prefix,
            shared_workspace_pvcs=storage.pvcs_by_node,
            appworld_endpoint=appworld_endpoint,
            internal_api_endpoints=internal_endpoints,
            mock=mock,
            gateway_endpoint=args.openshell_gateway_endpoint,
            openshell_client_secret=client_secret,
            allow_cross_node=args.allow_cross_node,
            agent_nodes=[resume.agent_nodes[index - 1] for index in resume.missing],
            agent_indices=resume.missing,
            simulated_ttft_ms=args.simulated_ttft,
            simulated_itl_ms=args.simulated_itl,
            sandbox_cpu_limit=experiment.sandbox_cpu_limit,
            sandbox_images=[images[index - 1] for index in resume.missing]
            if images is not None
            else None,
            swe_command_maps=[
                experiment.swe_command_maps_by_agent[index - 1]
                for index in resume.missing
            ],
        )
    bindings = _retry_cluster_read_stage(
        "agent binding inventory",
        lambda: _collect_owned_agent_bindings(
            cluster, storage.agent_prefix, storage.workspace_prefix, args.agents
        ),
        args.cluster_retry_timeout,
        args.cluster_retry_interval,
    )
    # A resumed run normally reconciles every existing agent to the fresh
    # run-scoped mock service. Once that reconciliation has completed, retries
    # of the same run can trust the owned inventory and avoid another 100-slot
    # rewrite that adds no experiment value.
    if args.skip_resume_agent_reconcile:
        # Keep warm reuse read-only. The live endpoint verifier below is
        # authoritative and avoids serial gateway rollouts.
        log_phase(
            "deployment: reusing warm agents without rollout; verifying live routing"
        )
    else:
        log_phase("deployment: reconciling existing agents to the run-scoped mock LLM")
        reconcile_indices = args.selected_agent_indices or list(
            range(1, args.agents + 1)
        )
        # A targeted recovery queue must not perform 100 per-Deployment reads
        # and Service applies merely to refresh a sparse subset. The batch
        # inventory above already established the complete, ordered fleet;
        # reconcile only the selected slots, then refresh that inventory with
        # two list calls.
        deploy_agents(
            cluster,
            len(reconcile_indices),
            reuse=True,
            workspace_prefix=storage.workspace_prefix,
            deployment_prefix=storage.agent_prefix,
            shared_workspace_pvcs=storage.pvcs_by_node,
            appworld_endpoint=appworld_endpoint,
            internal_api_endpoints=internal_endpoints,
            mock=mock,
            gateway_endpoint=args.openshell_gateway_endpoint,
            openshell_client_secret=client_secret,
            allow_cross_node=args.allow_cross_node,
            # The binding inventory can temporarily omit a non-running existing
            # Deployment. Resume inventory is the authoritative complete 1..N
            # placement map and preserves that slot's node.
            agent_nodes=[bindings[index - 1][2] for index in reconcile_indices],
            agent_indices=reconcile_indices,
            reconcile_indices=reconcile_indices,
            reuse_existing_workspaces=True,
            simulated_ttft_ms=args.simulated_ttft,
            simulated_itl_ms=args.simulated_itl,
            sandbox_cpu_limit=experiment.sandbox_cpu_limit,
            sandbox_images=[images[index - 1] for index in reconcile_indices]
            if images is not None
            else None,
            swe_command_maps=[
                experiment.swe_command_maps_by_agent[index - 1]
                for index in reconcile_indices
            ],
        )
        bindings = _retry_cluster_read_stage(
            "post-reconcile agent binding inventory",
            lambda: _collect_owned_agent_bindings(
                cluster, storage.agent_prefix, storage.workspace_prefix, args.agents
            ),
            args.cluster_retry_timeout,
            args.cluster_retry_interval,
        )
    write_deployment_plan(
        args,
        workloads.allocation,
        experiment.swe_task_images_by_agent,
        len(experiment.queue),
        [node for _, _, node in bindings],
    )
    return bindings


def _stage_swe(
    args: argparse.Namespace,
    cluster: Cluster,
    experiment: Experiment,
    storage: Storage,
    identity: RunIdentity,
    fleet: Fleet,
) -> None:
    """Create SWE sandboxes and copy pinned repositories onto their workspaces."""
    # OpenShell workspace creation initializes the per-workspace subpaths and
    # can clear files written before the workspace exists. Stage SWE
    # repositories only after all agent workspaces have been created and
    # reconciled. The sandbox-specific staging gate must still verify the
    # repository after OpenShell creates the actual sandbox; the PVC-level
    # staging below is only the first half of that handshake.
    # A sandbox also initializes its workspace at first use. Create the
    # agent-scoped SWE sandboxes through the real OpenClaw deployment before
    # staging, so OpenShell cannot clear a repository after it is copied.
    prewarm_swe_sandboxes(
        cluster,
        experiment.swe_items_by_agent,
        fleet.agent_bindings,
        args.out,
        args.system_namespace,
        identity.run_hash,
        args.allow_cross_node,
        workspace_pvcs=storage.pvcs_by_node,
        placement_node=identity.auxiliary_job_node,
        openshell_ns=args.openshell_namespace,
    )
    stage_swe_repositories(
        cluster,
        args.openshell_namespace,
        identity.run_hash,
        experiment.swe_tasks_by_agent,
        storage.workspace_names,
        fleet.agent_nodes,
        storage.pvcs_by_node,
        staging_node=identity.auxiliary_job_node,
        checkpoint_dir=args.out,
        scripts_by_agent=experiment.swe_script_dirs_by_agent,
        task_image_runtime=args.swe_sandbox_task_image,
    )
    # Repository staging already verified every commit directly on the exact
    # PVC subpath mounted by its OpenShell workspace. The synthetic prewarm
    # sandboxes exit after initialization, so requiring those pods to remain
    # Running here would turn a successful PVC handoff into a false timeout.
    if experiment.swe_tasks_by_agent:
        log_phase(
            "SWE backend: PVC repository handoff verified; skipping exited prewarm-pod liveness check"
        )


def _prewarm_pool(
    args: argparse.Namespace,
    cluster: Cluster,
    experiment: Experiment,
    storage: Storage,
    identity: RunIdentity,
    fleet: Fleet,
) -> Fleet:
    """Admit the warm-pool sandboxes and record the pool layout."""
    if args.resume_existing_run:
        # Existing agent pods are already counted as active. Before creating
        # trace Jobs and OpenShell sandboxes, reserve the two additional pods
        # required per agent and validate each node independently.
        trace_capacity_by_node: dict[str, int] = {}
        for _, _, node in fleet.agent_bindings:
            trace_capacity_by_node[node] = trace_capacity_by_node.get(node, 0) + 2
        _preflight_pod_capacity(
            cluster, [], additional_pods_by_node=trace_capacity_by_node
        )
    warm_pool_size = min(args.warm_pool_size, len(fleet.agent_bindings))
    pool_items_by_agent: dict[int, dict[str, Any]] = {}
    for position, item in enumerate(experiment.queue):
        binding_index = (
            (args.selected_agent_indices[position] - 1)
            if args.selected_agent_indices
            else ((position + args.agent_offset) % len(fleet.agent_bindings))
        )
        if binding_index < warm_pool_size:
            pool_items_by_agent[binding_index] = item
    prewarm_pool_sandboxes(
        cluster,
        pool_items_by_agent,
        fleet.agent_bindings,
        args.out,
        args.system_namespace,
        identity.run_hash,
        args.allow_cross_node,
        workspace_pvcs=storage.pvcs_by_node,
        placement_node=identity.auxiliary_job_node,
        openshell_ns=args.openshell_namespace,
    )
    warm_slots_by_node: dict[str, list[int]] = {node: [] for node in args.target_nodes}
    for index, (_, _, node) in enumerate(fleet.agent_bindings, start=1):
        if index <= warm_pool_size:
            warm_slots_by_node[node].append(index)
    warm_pool = {
        "schema": "agentic.trace-replay.warm-pool.v1",
        "policy": "agent-scoped-ready-sandbox-slots",
        "requested_size": args.warm_pool_size,
        "active_size": warm_pool_size,
        "cold_start_fallback": True,
        "target_nodes": list(args.target_nodes),
        "slots_by_node": warm_slots_by_node,
        "slots": [
            {
                "slot": index,
                "agent": agent,
                "workspace": workspace,
                "node": node,
                "prewarm_requested": index <= warm_pool_size,
            }
            for index, (agent, workspace, node) in enumerate(
                fleet.agent_bindings, start=1
            )
        ],
    }
    (args.out / "warm_pool.json").write_text(
        json.dumps(warm_pool, indent=2) + "\n", encoding="utf-8"
    )
    log_phase(
        f"sandbox warm pool: {warm_pool_size} prewarm slots across {len(args.target_nodes)} nodes; "
        f"{max(0, len(fleet.agent_bindings) - warm_pool_size)} agents use cold-start fallback"
    )
    return Fleet(
        agent_nodes=fleet.agent_nodes,
        agent_bindings=fleet.agent_bindings,
        mock=fleet.mock,
        warm_pool_size=warm_pool_size,
        warm_slots_by_node=warm_slots_by_node,
    )


def _execute(
    args: argparse.Namespace,
    cluster: Cluster,
    workloads: Workloads,
    experiment: Experiment,
    storage: Storage,
    fleet: Fleet,
    identity: RunIdentity,
) -> None:
    """Run the queue with live Jaeger forwarding and gateway sampling."""
    jaeger = JaegerPortForward(cluster, args.system_namespace)
    samplers: list[tuple[subprocess.Popen[str], Any]] = []
    try:
        try:
            samplers = _start_shared_gateway_sampler(
                cluster, args.openshell_namespace, args.out
            )
        except Exception as exc:  # noqa: BLE001
            LOG.warning("shared OpenShell gateway sampler unavailable: %s", exc)
        jaeger.wait_ready()
        log_phase(f"telemetry: Jaeger ready at {jaeger.url}")
        TraceDispatcher(
            args=args,
            cluster=cluster,
            workloads=workloads,
            experiment=experiment,
            storage=storage,
            fleet=fleet,
            identity=identity,
            jaeger=jaeger,
        ).run_all()
    finally:
        _stop_cgroup_samplers(samplers)
        jaeger.close()


class TraceDispatcher:
    """Execute every queued trace against its bound agent."""

    def __init__(
        self,
        *,
        args: argparse.Namespace,
        cluster: Cluster,
        workloads: Workloads,
        experiment: Experiment,
        storage: Storage,
        fleet: Fleet,
        identity: RunIdentity,
        jaeger: JaegerPortForward,
    ) -> None:
        self.args = args
        self.cluster = cluster
        self.workloads = workloads
        self.experiment = experiment
        self.storage = storage
        self.fleet = fleet
        self.identity = identity
        self.jaeger = jaeger
        self.errors_path = args.out / "errors.jsonl"
        self.errors_path.unlink(missing_ok=True)
        self._errors_lock = threading.Lock()
        self._dispatch_lock = threading.Lock()
        self._next_dispatch_at = 0.0
        self._jaeger_io_lock = threading.Lock()

    def _wait_for_job_dispatch_slot(self) -> None:
        """Rate-limit only Job admission; agent execution stays concurrent."""
        if self.args.job_dispatch_interval <= 0:
            return
        with self._dispatch_lock:
            now = time.monotonic()
            delay = max(0.0, self._next_dispatch_at - now)
            self._next_dispatch_at = (
                max(now, self._next_dispatch_at) + self.args.job_dispatch_interval
            )
        if delay > 0:
            time.sleep(delay)

    def run_all(self) -> None:
        """Dispatch the whole queue and persist the run summary."""
        execution_workers = self.args.execution_workers or min(
            16, len(self.fleet.agent_bindings)
        )
        log_phase(
            f"trace controller concurrency: {execution_workers} workers for "
            f"{len(self.fleet.agent_bindings)} deployed agents"
        )
        with ThreadPoolExecutor(
            max_workers=execution_workers, thread_name_prefix="trace-replay-agent"
        ) as pool:
            futures = [
                pool.submit(self._run_one, position, item)
                for position, item in enumerate(self.experiment.queue)
            ]
            results = [future.result() for future in as_completed(futures)]
        failed = [result for result in results if not result.get("ok")]
        warnings = exit_code_warnings(self.args.out)
        if warnings:
            LOG.warning(
                "recorded exit codes differ in %d of %d replayed sessions "
                "(%d turns ended with a different exit code): %s",
                len(warnings),
                len(results),
                sum(item["mismatched_turns"] for item in warnings),
                "; ".join(
                    f"{item['session_id']}={item['mismatched_turns']} turn(s) {item['turn_indices']}"
                    for item in warnings
                ),
            )
        (self.args.out / "run_summary.json").write_text(
            json.dumps(
                {
                    "total": len(results),
                    "completed": len(results) - len(failed),
                    "failed": len(failed),
                    "exit_code_warnings": warnings,
                    "agents": len(self.fleet.agent_bindings),
                    "execution_workers": execution_workers,
                    "tool_cpu_ids": list(self.args.tool_cpu_ids),
                    "tool_cgroup_quota": self.args.tool_cgroup_quota or None,
                    "tool_cgroup_period_us": (
                        self.args.tool_cgroup_period_us
                        if self.args.tool_cgroup_quota
                        else None
                    ),
                    "job_dispatch_interval_seconds": self.args.job_dispatch_interval,
                    "warm_pool_size": self.fleet.warm_pool_size,
                    "cold_start_fallback": True,
                    "target_nodes": list(self.args.target_nodes),
                    "agent_workspaces": {
                        agent: {"workspace": workspace, "node": node}
                        for agent, workspace, node in self.fleet.agent_bindings
                    },
                },
                indent=2,
            )
        )
        if failed:
            raise RuntimeError(f"{len(failed)} traces failed; see {self.errors_path}")

    def _run_one(self, position: int, item: dict[str, Any]) -> dict[str, Any]:
        """Run one queue entry, retrying transient trace failures."""
        args = self.args
        binding_index = (
            (args.selected_agent_indices[position] - 1)
            if args.selected_agent_indices
            else ((position + args.agent_offset) % len(self.fleet.agent_bindings))
        )
        agent, workspace, agent_node = self.fleet.agent_bindings[binding_index]
        warm_pool_slot = (
            binding_index + 1 if binding_index < self.fleet.warm_pool_size else None
        )
        # The pool is filled before measurement; the driver sends only the real client request.
        prewarm_requested = False
        session_id = str(item["session"].get("session_id") or "")
        for attempt in range(args.retries + 1):
            job = ""
            corpus_configmap = ""
            trace_dir: Path | None = None
            try:
                log_phase(
                    f"trace {position + 1}/{len(self.experiment.queue)}: {session_id} "
                    f"workload={item['workload']} agent={agent} attempt={attempt + 1}"
                )
                start = time.time()
                if attempt > 0 and item["workload"] == "software_engineering_qa":
                    # The preceding attempt may have edited the staged
                    # repository without moving HEAD. Recopy the pinned
                    # snapshot into this run-owned workspace before any
                    # diagnostic retry, bypassing the clean-stage cache.
                    stage_swe_repositories(
                        self.cluster,
                        args.openshell_namespace,
                        self.identity.run_hash,
                        {
                            binding_index: self.experiment.swe_tasks_by_agent[
                                binding_index
                            ]
                        },
                        self.storage.workspace_names,
                        self.fleet.agent_nodes,
                        self.storage.pvcs_by_node,
                        staging_node=self.identity.auxiliary_job_node,
                        checkpoint_dir=None,
                        scripts_by_agent={
                            binding_index: self.experiment.swe_script_dirs_by_agent[
                                binding_index
                            ]
                        },
                        task_image_runtime=args.swe_sandbox_task_image,
                    )
                # Kubernetes Jobs are immutable; include a per-process suffix
                # so reruns never reuse a completed Job name. The suffix must
                # also be unique when several experiment controllers run
                # concurrently (concurrency sweeps launch one process per task
                # and they frequently start in the same second), so mix in a
                # stable hash of the output directory instead of the start time
                # alone.
                out_key = hashlib.sha256(
                    str(args.out.resolve()).encode("utf-8")
                ).hexdigest()[:6]
                run_id = f"{safe_name(args.out.name)[:6] or 'run'}-{int(start) % 1000000:06d}-{out_key}"
                self._wait_for_job_dispatch_slot()
                job, corpus_configmap, trace_dir, session_id = create_trace_job(
                    self.cluster,
                    item,
                    agent,
                    args.out / "traces",
                    args.system_namespace,
                    run_id,
                    attempt,
                    allow_cross_node=args.allow_cross_node,
                    prewarm_requested=prewarm_requested,
                    warm_pool_slot=warm_pool_slot,
                    warm_pool_size=self.fleet.warm_pool_size,
                    agent_node=agent_node,
                    placement_node=self.identity.auxiliary_job_node,
                    user_key=(
                        f"trace-replay:{session_id}:swe-prewarm-{self.identity.run_hash}"
                        if item["workload"] == "software_engineering_qa"
                        else None
                    ),
                    client_gap_ms=args.client_gap_ms,
                )
                self._collect(
                    job,
                    trace_dir,
                    session_id,
                    start,
                    agent,
                    workspace,
                    agent_node,
                    warm_pool_slot,
                    item,
                    binding_index,
                )
                metadata_path = trace_dir / "trace.json"
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                metadata["attempt_number"] = attempt + 1
                metadata["attempts_requested"] = args.retries + 1
                metadata["benchmark_eligible"] = attempt == 0
                metadata_path.write_text(
                    json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
                )
                self._record_attempt(
                    {
                        "session_id": session_id,
                        "position": position,
                        "agent": agent,
                        "attempt": attempt + 1,
                        "ok": True,
                        "start_unix": start,
                        "end_unix": time.time(),
                    }
                )
                log_phase(f"trace complete: {trace_dir}")
                return {
                    "position": position,
                    "session_id": session_id,
                    "agent": agent,
                    "ok": True,
                    "attempt": attempt + 1,
                }
            except Exception as exc:  # noqa: BLE001
                self._record_attempt(
                    {
                        "session_id": session_id,
                        "position": position,
                        "agent": agent,
                        "attempt": attempt + 1,
                        "ok": False,
                        "start_unix": locals().get("start"),
                        "end_unix": time.time(),
                        "error": str(exc),
                    }
                )
                if trace_dir is not None and job:
                    self._preserve_failed_logs(job, trace_dir)
                if attempt >= args.retries:
                    error = {
                        "position": position,
                        "session_id": session_id,
                        "agent": agent,
                        "ok": False,
                        "attempts": attempt + 1,
                        "error": str(exc),
                    }
                    with self._errors_lock:
                        with self.errors_path.open("a", encoding="utf-8") as handle:
                            handle.write(json.dumps(error) + "\n")
                    LOG.error("trace failed permanently: %s", error)
                    return error
                LOG.warning(
                    "trace failed; retrying: session=%s error=%s", session_id, exc
                )
            finally:
                cleanup_trace_resources(self.cluster, job, corpus_configmap)
        raise AssertionError("unreachable")

    def _collect(
        self,
        job: str,
        trace_dir: Path,
        session_id: str,
        start: float,
        agent: str,
        workspace: str,
        agent_node: str,
        warm_pool_slot: int | None,
        item: dict[str, Any],
        binding_index: int,
    ) -> None:
        """Collect one trace, restoring the owned Jaeger forward if it breaks."""
        args = self.args

        def collect() -> None:
            collect_trace(
                self.cluster,
                job,
                trace_dir,
                session_id,
                start,
                args.system_namespace,
                self.jaeger.url,
                agent,
                workspace,
                args.openshell_namespace,
                self.workloads.backends[item["workload"]],
                self.fleet.mock.recorder(binding_index + 1, agent),
                allow_cross_node=args.allow_cross_node,
                prewarm_requested=False,
                warm_pool_slot=warm_pool_slot,
                warm_pool_size=self.fleet.warm_pool_size,
                expected_node=agent_node,
                jaeger_supplier=self.jaeger.ensure,
                jaeger_io_lock=self._jaeger_io_lock,
                expected_cgroup_quota=self.experiment.tool_cgroup_quota,
            )

        try:
            collect()
        except Exception as exc:  # noqa: BLE001
            message = str(exc).lower()
            if not any(
                token in message
                for token in ("jaeger", "connection refused", "urlopen error")
            ):
                raise
            LOG.warning(
                "Jaeger collection failed; restoring the owned port-forward: %s", exc
            )
            with self._jaeger_io_lock:
                self.jaeger.ensure(force=True)
            collect()

    def _record_attempt(self, record: dict[str, Any]) -> None:
        with self._errors_lock:
            with (self.args.out / "attempts.jsonl").open(
                "a", encoding="utf-8"
            ) as handle:
                handle.write(json.dumps(record) + "\n")

    def _preserve_failed_logs(self, job: str, trace_dir: Path) -> None:
        try:
            failure_log = self.cluster.run(
                [
                    "-n",
                    self.cluster.namespace,
                    "logs",
                    f"job/{job}",
                    "--all-containers=true",
                ],
                check=False,
            )
            if failure_log:
                (trace_dir / "failed-job.log").write_text(failure_log, encoding="utf-8")
        except Exception as log_exc:  # noqa: BLE001
            LOG.warning("unable to preserve failed Job logs for %s: %s", job, log_exc)


def _write_experiment_json(
    args: argparse.Namespace,
    experiment: Experiment,
    fleet: Fleet,
    agent_nodes: list[str],
) -> None:
    """Persist the experiment-level manifest."""
    (args.out / "experiment.json").write_text(
        json.dumps(
            {
                "agents": args.agents,
                "traces": len(experiment.queue),
                "workloads": WORKLOADS,
                "node": args.target_nodes[0],
                "nodes": list(args.target_nodes),
                "agent_node_assignments": {
                    str(index): node for index, node in enumerate(agent_nodes, start=1)
                },
                "allow_cross_node": args.allow_cross_node,
                "tool_cpu_ids": list(args.tool_cpu_ids),
                "tool_cgroup_quota": args.tool_cgroup_quota or None,
                "tool_cgroup_period_us": (
                    args.tool_cgroup_period_us if args.tool_cgroup_quota else None
                ),
                "warm_pool_size": fleet.warm_pool_size,
                "warm_pool_policy": "agent-scoped warm reuse with measured cold-start fallback",
                "measurement_caveat": (
                    "includes cross-node scheduling and network latency"
                    if args.allow_cross_node
                    else None
                ),
            },
            indent=2,
        )
    )
