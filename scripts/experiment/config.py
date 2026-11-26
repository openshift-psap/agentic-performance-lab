"""Run configuration: environment, ownership rules, and paths."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from trace_replay_sim.cluster_planning import parse_target_nodes, validate_agent_count

ROOT = Path(__file__).resolve().parents[2]

SRC = ROOT / "src"

# Baseline objects are explicitly owned by this benchmark. Run-scoped objects
# use an additional hash suffix; these names are template/telemetry services
# and are never deleted by the controller.
RESOURCE_PREFIX = os.environ.get("TRACE_REPLAY_RESOURCE_PREFIX", "trace-replay").strip()

# The OpenClaw OpenShell plugin addresses the gateway by *name* whenever one is
# configured, and the CLI then resolves that name from its own registry before
# it can build an SSH proxy.  A raw endpoint URL is not a registry entry, so
# SSH-based transfers (workspace seeding) fail with a closed connection and the
# seed leaves a broken workspace behind.  Registering the endpoint under a
# stable name in the agent pod's CLI config keeps every call (exec, upload,
# ssh-config) on one resolvable identity.
OPENSHELL_GATEWAY_NAME = os.environ.get(
    "TRACE_REPLAY_OPENSHELL_GATEWAY_NAME", "aharush-tracesim"
).strip()

STRICT_OWNERSHIP = os.environ.get("TRACE_REPLAY_STRICT_OWNERSHIP", "").lower() in {
    "1",
    "true",
    "yes",
}

OWNED_NAMESPACES = {
    name.strip()
    for name in os.environ.get("TRACE_REPLAY_OWNED_NAMESPACES", "").split(",")
    if name.strip()
}

# OpenShell's gateway creates Sandbox resources in the namespace where that
# shared gateway is installed.  Strict mode may therefore allow that one
# existing namespace as a target for this run's explicitly prefixed resources;
# every object written there still has to pass assert_owned_target().
EXTERNAL_SANDBOX_NAMESPACE = os.environ.get(
    "TRACE_REPLAY_EXTERNAL_SANDBOX_NAMESPACE", ""
).strip()


def assert_owned_target(namespace: str, name: str) -> None:
    """Fail closed before any strict-mode write outside this run's objects."""
    if not STRICT_OWNERSHIP:
        return
    if namespace not in OWNED_NAMESPACES or not name.startswith(f"{RESOURCE_PREFIX}-"):
        raise RuntimeError(f"strict ownership refused {namespace}/{name}")


def assert_owned_namespace(namespace: str) -> None:
    if STRICT_OWNERSHIP and namespace not in OWNED_NAMESPACES:
        raise RuntimeError(f"strict ownership refused namespace {namespace}")


BASE_JAEGER = f"{RESOURCE_PREFIX}-jaeger"


BASE_MOCK_LLM = f"{RESOURCE_PREFIX}-mock-llm"


BASE_OPENCLAW = f"{RESOURCE_PREFIX}-openclaw-shell"


BASE_OPENCLAW_CONFIG = f"{RESOURCE_PREFIX}-openclaw-shell-config"


APPWORLD_IMAGE = "ghcr.io/stonybrooknlp/appworld:latest"


APPWORLD_DATA_CHECK = (
    "test -f /appworld/data/base_dbs/spotify.db "
    "&& test -d /appworld/data/tasks && printf PRESENT"
)


# The experiment image's OpenClaw runtime and external OpenShell plugin must
# use compatible plugin APIs.  This image is the public OpenShell-enabled build
# used by the existing replay runs; callers may override it explicitly when
# validating another approved image.
EXPERIMENT_OPENCLAW_IMAGE = os.environ.get(
    "TRACE_REPLAY_OPENCLAW_IMAGE",
    "quay.io/sallyom/openclaw-openshell:latest",
)


EXPERIMENT_OPENSHELL_PLUGIN_VERSION = os.environ.get(
    "TRACE_REPLAY_OPENSHELL_PLUGIN_VERSION",
    "2026.8.1-beta.2",
).strip()


LOG = logging.getLogger("run-experiment")


TRANSIENT_API_MARKERS = (
    "tls handshake timeout",
    "context deadline exceeded",
    "server was unable to return a response",
    "i/o timeout",
    "connection reset by peer",
    "failed calling webhook",
    "no endpoints available for service",
    "connection refused",
    "unexpected eof",
    "error: eof",
    "internal error occurred",
    "timed out after",
    "too many requests",
    "storage is (re)initializing",
    "admission plugin",
    "caches not synchronized",
    "client connection lost",
    "stream error",
    "received from peer",
    # A broken oc exec stream can reach the target Python process as a
    # syntactically valid command with a truncated/non-gzip tar payload.  The
    # API transport often reports no useful error on the source side, so the
    # target-side tarfile exception is the only transient signal available.
    "not a gzip file",
    "unexpected end of data",
    "not yet ready to handle request",
    "serviceunavailable",
    "service unavailable",
    "server is currently unable to handle the request",
    # Concurrent experiment controllers can legitimately share template
    # objects (for example the driver ConfigMap).  The loser of a `replace`
    # gets a resourceVersion conflict; re-reading and retrying is correct.
    "the object has been modified",
    "operation cannot be fulfilled",
    "please apply your changes to the latest version",
    # A get-then-create upsert can lose a race with a concurrent controller
    # or with its own earlier attempt; re-read and replace instead of failing.
    "alreadyexists",
)


READ_ONLY_OC_COMMANDS = {
    "get",
    "wait",
    "rollout",
    "whoami",
    "config",
    "version",
    "api-resources",
    "cluster-info",
}


DEFAULT_KUEUE_NAMESPACE = "kueue-system"


DEFAULT_KUEUE_WEBHOOK_SERVICE = "kueue-webhook-service"


MOCK_CORPUS_CHUNK_BYTES = 900_000


SANDBOX_POLL_INTERVAL_SECONDS = 2.0


RESULT_POLL_INTERVAL_SECONDS = 3.0


def safe_name(value: str) -> str:
    return "".join(
        char if char.isalnum() or char == "-" else "-" for char in value.lower()
    ).strip("-")


def is_transient_api_error(message: str) -> bool:
    lowered = message.lower()
    return any(marker in lowered for marker in TRANSIENT_API_MARKERS)


def pack_mock_corpus_sessions(
    sessions: list[dict[str, Any]],
    *,
    max_bytes: int = MOCK_CORPUS_CHUNK_BYTES,
) -> list[list[dict[str, Any]]]:
    """Pack replay sessions into ConfigMap-safe JSON chunks.

    A full multi-workload run can contain tens of MiB of replay data. Keeping
    one ConfigMap per session creates unnecessary API-server and admission
    traffic, while one ConfigMap for the whole corpus exceeds Kubernetes'
    object-size limit. Packing whole sessions into bounded chunks preserves
    directory-based loading and materially reduces the number of write
    operations without splitting any stateful session.
    """
    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")
    parts: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for session in sessions:
        candidate = current + [session]
        encoded_size = len(
            json.dumps({"sessions": candidate}, separators=(",", ":")).encode("utf-8")
        )
        if current and encoded_size > max_bytes:
            parts.append(current)
            current = []
            encoded_size = len(
                json.dumps({"sessions": [session]}, separators=(",", ":")).encode(
                    "utf-8"
                )
            )
        if encoded_size > max_bytes:
            raise ValueError(
                f"replay session exceeds the mock corpus chunk limit ({max_bytes} bytes)"
            )
        current.append(session)
    if current:
        parts.append(current)
    return parts


def parse_pod_quantity(value: Any) -> int:
    """Parse the Kubernetes quantity used for ``status.allocatable.pods``.

    Kubernetes may serialize a pod count as either ``1000`` or the compact
    quantity ``1k``.  Pod capacity is an integer count, so accepting the
    documented decimal SI suffixes keeps preflight checks equivalent across
    API-server versions without weakening the capacity gate.
    """
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    text = str(value).strip()
    if text.isdigit():
        return int(text)
    match = re.fullmatch(r"(\d+)([kKmMgGtTpPeE])", text)
    if match:
        multipliers = {
            "k": 10**3,
            "K": 10**3,
            "m": 10**6,
            "M": 10**6,
            "g": 10**9,
            "G": 10**9,
            "t": 10**12,
            "T": 10**12,
            "p": 10**15,
            "P": 10**15,
            "e": 10**18,
            "E": 10**18,
        }
        return int(match.group(1)) * multipliers[match.group(2)]
    raise ValueError(f"unsupported Kubernetes pod quantity: {value!r}")


def parse_tool_cgroup_quota(value: str, period_us: int = 100_000) -> str:
    """Translate a core count into the cgroup-v2 ``cpu.max`` quota form.

    ``cpu.max`` expects ``"<quota_us> <period_us>"`` (or ``"max <period_us>"``).
    Expressing the experiment's independent variable as quota gives an exact
    CPU-time rate per sandbox that is independent of who else shares the
    physical cores, unlike cpuset affinity alone.
    """
    try:
        cores = float(str(value).strip())
    except ValueError as exc:
        raise ValueError("tool cgroup quota must be a number of cores") from exc
    if not 0.0 < cores <= 1024.0:
        raise ValueError(
            "tool cgroup quota must be greater than 0 and at most 1024 cores"
        )
    if not 1_000 <= int(period_us) <= 1_000_000:
        raise ValueError(
            "tool cgroup quota period must be between 1000 and 1000000 microseconds"
        )
    quota_us = int(round(cores * int(period_us)))
    if quota_us < 1_000:
        raise ValueError(
            "tool cgroup quota is below the 1 ms minimum for the chosen period"
        )
    return f"{quota_us} {int(period_us)}"


def tool_cgroup_quota_cpu_quantity(quota: str) -> str:
    """Translate a ``cpu.max`` quota string into a Kubernetes CPU quantity.

    ``openshell sandbox create --cpu`` is the supported way to pin CPU on the
    Kubernetes driver: it becomes the sandbox agent container's ``resources``
    limit, which the kubelet turns into the same ``cpu.max`` quota.  Enforcing
    it at pod creation avoids relying on a writable cgroupfs inside the
    sandbox, which the container runtime mounts read-only.
    """
    parts = str(quota).split()
    if len(parts) != 2 or parts[0] == "max":
        raise ValueError(
            f"tool cgroup quota must be a finite '<quota> <period>': {quota!r}"
        )
    try:
        cores = int(parts[0]) / int(parts[1])
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError(f"invalid tool cgroup quota: {quota!r}") from exc
    millicores = int(round(cores * 1000))
    if millicores < 1:
        raise ValueError("tool cgroup quota is below 1 millicore")
    return f"{millicores}m"


def workspace_name(prefix: str, index: int) -> str:
    """Return an OpenShell-compatible workspace name for an agent slot.

    OpenShell currently limits workspace names to 19 characters.  The normal
    run prefix fits for slots 1..99, but the literal ``-100`` suffix would
    overflow by one character.  Keep the slot visible and derive a compact,
    deterministic run token for overflow slots so every agent still gets a
    unique, run-scoped workspace.
    """
    candidate = f"{prefix}-{index}"
    if len(candidate) <= 19:
        return candidate
    token = hashlib.sha256(f"{prefix}:{index}".encode("utf-8")).hexdigest()[:7]
    bounded = f"{RESOURCE_PREFIX[:7]}-{token}-{index}"
    if len(bounded) > 19:
        raise ValueError(f"cannot create an OpenShell workspace name for agent {index}")
    return bounded


def agent_name_prefix(out_name: str, run_hash: str) -> str:
    """Return a compact run-scoped prefix for per-agent Kubernetes objects.

    Agent ConfigMaps append ``-config-NN`` and use the object name as a label
    value, which Kubernetes limits to 63 bytes.
    """
    fixed = f"{RESOURCE_PREFIX}--{run_hash}-openclaw"
    max_out_name = 63 - len("-config-100") - len(fixed)
    if max_out_name < 1:
        raise ValueError("resource prefix and run hash leave no room for agent names")
    return f"{RESOURCE_PREFIX}-{safe_name(out_name)[:max_out_name].rstrip('-')}-{run_hash}-openclaw"


def parse_workspace_pvcs(
    spec: str | None, target_nodes: tuple[str, ...]
) -> dict[str, str]:
    """Parse an explicit node-to-RWO-PVC map without guessing ownership."""
    if not spec:
        return {}
    mapping: dict[str, str] = {}
    for item in spec.split(","):
        node, separator, pvc = item.partition("=")
        node = node.strip()
        pvc = pvc.strip()
        if not separator or not node or not pvc:
            raise ValueError("--workspace-pvcs must use node=pvc,node=pvc syntax")
        if node in mapping:
            raise ValueError(f"duplicate node in --workspace-pvcs: {node}")
        if node not in target_nodes:
            raise ValueError(
                f"--workspace-pvcs names a node not present in --nodes: {node}"
            )
        mapping[node] = pvc
    missing = [node for node in target_nodes if node not in mapping]
    if missing:
        raise ValueError(
            f"--workspace-pvcs is missing target nodes: {', '.join(missing)}"
        )
    return mapping


def log_phase(message: str) -> None:
    LOG.info("[%s] %s", datetime.now().astimezone().strftime("%H:%M:%S"), message)


CGROUP_SAMPLE_COLUMNS = (
    "epoch_ns",
    "cpu_usage_usec",
    "memory_bytes",
    "user_usec",
    "system_usec",
    "nr_throttled",
    "throttled_usec",
    "cpu_quota",
    "cpu_period",
    "cpu_pressure_some_avg10",
    "cpu_pressure_some_total",
    "cpu_pressure_full_avg10",
    "cpu_pressure_full_total",
    "memory_pressure_some_avg10",
    "memory_pressure_full_avg10",
    "io_pressure_some_avg10",
    "memory_inactive_file_bytes",
    "memory_working_set_bytes",
    "memory_anon_bytes",
    "memory_file_bytes",
)


class ConfigError(RuntimeError):
    """A validated control-plane setting is invalid."""


def build_parser() -> argparse.ArgumentParser:
    """Build the controller command line."""
    parser = argparse.ArgumentParser(
        description="Run the complete multi-workload OpenClaw/OpenShell experiment"
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        help="recorded trace corpus; not needed with --backends-only",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="experiment output directory; not needed with --backends-only",
    )
    parser.add_argument("--agents", type=int, default=20)
    parser.add_argument(
        "--warm-pool-size",
        type=int,
        default=10,
        help="number of agent-scoped OpenShell sandboxes admitted to the prewarm pool; excess agents use cold-start fallback",
    )
    parser.add_argument("--traces-per-workload", type=int, default=5)
    parser.add_argument(
        "--client-gap-ms",
        type=float,
        default=0.0,
        help="controlled pause between recorded client messages",
    )
    parser.add_argument(
        "--max-traces",
        type=int,
        help="cap the total number of traces after workload partitioning",
    )
    parser.add_argument(
        "--trace-offset",
        type=int,
        default=0,
        help="skip this many sessions per workload before selecting traces",
    )
    parser.add_argument(
        "--require-traces",
        type=int,
        help="fail unless the final queue has exactly this many traces",
    )
    parser.add_argument("--session-id", help="run exactly one recorded session")
    parser.add_argument(
        "--session-ids", help="comma-separated recorded session IDs to run concurrently"
    )
    parser.add_argument(
        "--agent-offset",
        type=int,
        default=0,
        help="offset queue positions into the deployed agent list",
    )
    parser.add_argument(
        "--agent-indices",
        help="comma-separated 1-based agent indices for the selected queue",
    )
    parser.add_argument(
        "--appworld", action="store_true", help="legacy shortcut for AppWorld-only runs"
    )
    parser.add_argument("--appworld-endpoint", default=None)
    parser.add_argument(
        "--appworld-data",
        type=Path,
        default=Path(
            os.environ.get(
                "TRACE_REPLAY_APPWORLD_DATA",
                str(ROOT.parent / "appworld-runtime" / "data"),
            )
        ),
        help="local AppWorld data directory to stage into the owned backend PVC",
    )
    parser.add_argument(
        "--appworld-task-map",
        type=Path,
        default=Path(
            os.environ.get(
                "TRACE_REPLAY_APPWORLD_TASK_MAP",
                str(ROOT.parent / "appworld-runtime" / "session-task-map.json"),
            )
        ),
        help="validated session-to-AppWorld-task mapping JSON",
    )
    parser.add_argument(
        "--appworld-data-pvc",
        default=os.environ.get(
            "TRACE_REPLAY_APPWORLD_DATA_PVC", "trace-replay-appworld-data"
        ),
        help="RWO PVC used by the AppWorld backend",
    )
    parser.add_argument(
        "--skip-appworld-staging",
        action="store_true",
        default=(
            os.environ.get("TRACE_REPLAY_SKIP_APPWORLD_STAGING", "").strip().lower()
            in {"1", "true", "yes", "on"}
        ),
        help=(
            "reuse an existing, pre-populated AppWorld data PVC instead of "
            "staging from --appworld-data; the PVC does not need the controller "
            "ownership label"
        ),
    )
    parser.add_argument(
        "--workload-plan",
        help="exact allocation, e.g. personal_assistant=5,software_engineering_qa=5,research_deep_search=5,customer_support_ops=5",
    )
    parser.add_argument(
        "--backend-manifest",
        type=Path,
        help="JSON manifest describing real workload backends",
    )
    parser.add_argument(
        "--swe-task-manifest",
        type=Path,
        default=ROOT / "config/swebench-trace-manifest.json",
        help="pinned SWE task metadata mapping traces to repository commits and task images",
    )
    parser.add_argument(
        "--swe-sandbox-task-image",
        action="store_true",
        help="create each SWE OpenShell sandbox from its task image instead of the generic workload image",
    )
    parser.add_argument("--namespace", default="trace-replay")
    parser.add_argument("--system-namespace", default="trace-replay")
    parser.add_argument("--openshell-namespace", default="openshell-tracesim")
    parser.add_argument(
        "--kueue-namespace",
        default=os.environ.get("TRACE_REPLAY_KUEUE_NAMESPACE", DEFAULT_KUEUE_NAMESPACE),
        help="namespace containing the Kueue admission webhook",
    )
    parser.add_argument(
        "--kueue-webhook-service",
        default=os.environ.get(
            "TRACE_REPLAY_KUEUE_WEBHOOK_SERVICE", DEFAULT_KUEUE_WEBHOOK_SERVICE
        ),
        help="Kueue webhook Service used for Job admission",
    )
    parser.add_argument(
        "--openshell-gateway-endpoint",
        default=os.environ.get(
            "TRACE_REPLAY_OPENSHELL_GATEWAY_ENDPOINT",
            "http://openshell.openshell-tracesim.svc.cluster.local:8080",
        ),
        help="OpenShell gateway endpoint used by the agent and workspace provisioning",
    )
    parser.add_argument(
        "--openshell-client-secret",
        default=os.environ.get(
            "TRACE_REPLAY_OPENSHELL_CLIENT_SECRET", "openshell-client-tls"
        ),
        help="client TLS Secret name in the OpenShell gateway namespace",
    )
    parser.add_argument(
        "--openshell-client-secret-namespace",
        default=os.environ.get("TRACE_REPLAY_OPENSHELL_CLIENT_SECRET_NAMESPACE"),
        help=(
            "namespace containing the source OpenShell client Secret; defaults to "
            "--openshell-namespace"
        ),
    )
    parser.add_argument(
        "--node",
        default=os.environ.get("TRACE_REPLAY_NODE"),
        help="backwards-compatible single target node",
    )
    parser.add_argument(
        "--nodes",
        help="comma-separated target nodes; agents are assigned round-robin in this order",
    )
    parser.add_argument(
        "--allow-cross-node",
        action="store_true",
        help="allow experiment-owned agents, Jobs, and sandboxes to schedule on any available node",
    )
    parser.add_argument(
        "--workspace-pvc",
        default=os.environ.get("TRACE_REPLAY_WORKSPACE_PVC"),
        help="owned RWX workspace PVC; defaults to a unique run-scoped name",
    )
    parser.add_argument(
        "--workspace-pvcs",
        help="explicit node-to-owned-RWO-PVC map, e.g. node-a=pvc-a,node-b=pvc-b",
    )
    parser.add_argument(
        "--workspace-storage-class",
        default=os.environ.get("TRACE_REPLAY_WORKSPACE_STORAGE_CLASS", ""),
        help="StorageClass for workspace PVCs; empty uses the cluster default",
    )
    parser.add_argument(
        "--workspace-access-mode",
        choices=("ReadWriteMany", "ReadWriteOnce"),
        default=os.environ.get("TRACE_REPLAY_WORKSPACE_ACCESS_MODE", "ReadWriteMany"),
    )
    parser.add_argument(
        "--allow-shared-workspace-pvc",
        action="store_true",
        default=os.environ.get("TRACE_REPLAY_ALLOW_SHARED_WORKSPACE_PVC") == "1",
        help="place several agent workspaces on one PVC (requires the gateway admission gate to be off)",
    )
    parser.add_argument("--kubeconfig", default=os.environ.get("KUBECONFIG"))
    parser.add_argument("--jaeger", default="http://127.0.0.1:16686")
    parser.add_argument("--reuse-agents", action="store_true")
    parser.add_argument(
        "--reuse-existing-agent-prefix",
        help="reuse only this owned run's OpenClaw deployment prefix",
    )
    parser.add_argument(
        "--reuse-existing-workspace-prefix",
        help="reuse only this owned run's workspace prefix",
    )
    parser.add_argument(
        "--resume-existing-run",
        action="store_true",
        help="resume an owned partial fleet without recreating running agents",
    )
    parser.add_argument(
        "--skip-resume-agent-reconcile",
        action="store_true",
        help="on a resume, trust the already reconciled owned fleet and skip the full 100-slot rewrite",
    )
    parser.add_argument(
        "--resume-preferred-node",
        help="healthy approved node for missing or pending agents during --resume-existing-run",
    )
    parser.add_argument(
        "--deployment-only",
        action="store_true",
        help="stop after agent deployment and workspace wiring; do not create trace Jobs",
    )
    parser.add_argument("--retries", type=int, default=0)
    parser.add_argument(
        "--simulated-ttft",
        type=float,
        default=None,
        help="simulated time-to-first-token in ms (overrides calibrated timing)",
    )
    parser.add_argument(
        "--simulated-itl",
        type=float,
        default=None,
        help="simulated inter-token latency in ms (overrides calibrated timing)",
    )
    parser.add_argument(
        "--tool-cpu-ids",
        default="",
        help="comma-separated logical CPU IDs for SWE shell commands; empty leaves commands unrestricted",
    )
    parser.add_argument(
        "--tool-cgroup-quota",
        default="",
        help=(
            "CPU cores per sandbox (e.g. 0.5, 2), enforced as the sandbox pod "
            "resources.limits.cpu at creation; empty leaves the limit unchanged"
        ),
    )
    parser.add_argument(
        "--tool-cgroup-period-us",
        type=int,
        default=100_000,
        help="cpu.max period used to derive the core count when --tool-cgroup-quota is set (default 100000)",
    )
    parser.add_argument(
        "--execution-workers",
        type=int,
        default=None,
        help=(
            "maximum number of trace controllers issuing Kubernetes API requests concurrently; "
            "the deployed agent count is unchanged (default: all agents)"
        ),
    )
    parser.add_argument(
        "--job-dispatch-interval",
        type=float,
        default=2.0,
        help="minimum seconds between trace Job admission requests (0 disables throttling)",
    )
    parser.add_argument(
        "--cluster-api-workers",
        type=int,
        default=6,
        help="maximum concurrent oc subprocesses used by the controller (default: 6)",
    )
    parser.add_argument(
        "--cluster-retry-timeout",
        type=float,
        default=1800.0,
        help="seconds to keep retrying transient read-only cluster preparation failures",
    )
    parser.add_argument(
        "--cluster-retry-interval",
        type=float,
        default=10.0,
        help="seconds between transient cluster preparation retries",
    )
    parser.add_argument(
        "--backends-only",
        action="store_true",
        help="deploy and health-check selected workload backends, then stop before agents/sandboxes",
    )
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--execute", action="store_true")
    return parser


def parse_args(
    argv: list[str] | None = None,
    parser: argparse.ArgumentParser | None = None,
) -> argparse.Namespace:
    """Parse and validate controller arguments."""
    args = (parser or build_parser()).parse_args(argv)
    if args.backends_only and args.plan:
        raise ConfigError(
            "--backends-only requires --execute and cannot be combined with --plan"
        )
    if not args.backends_only and args.plan == args.execute:
        raise ConfigError("choose exactly one of --plan or --execute")
    if args.backends_only and not args.execute:
        raise ConfigError("--backends-only requires --execute")
    if not args.backends_only and args.corpus is None:
        raise ConfigError("--corpus is required unless --backends-only is used")
    if not args.backends_only and args.out is None:
        raise ConfigError("--out is required unless --backends-only is used")
    try:
        validate_agent_count(args.agents)
        target_nodes = parse_target_nodes(args.node, args.nodes)
        workspace_pvcs = parse_workspace_pvcs(args.workspace_pvcs, target_nodes)
    except ValueError as exc:
        raise ConfigError(str(exc))
    if STRICT_OWNERSHIP:
        allowed_node = os.environ.get("TRACE_REPLAY_ALLOWED_NODE", "").strip()
        if len(target_nodes) != 1 or (allowed_node and target_nodes != (allowed_node,)):
            raise ConfigError(
                "strict ownership requires exactly one target node, matching "
                "TRACE_REPLAY_ALLOWED_NODE when it is set"
            )
        owned_prefix = f"{RESOURCE_PREFIX}-"
        if not OWNED_NAMESPACES or any(
            not name.startswith(owned_prefix) and name != EXTERNAL_SANDBOX_NAMESPACE
            for name in OWNED_NAMESPACES
        ):
            raise ConfigError(
                f"strict ownership requires {owned_prefix}* namespaces, except for the explicitly "
                "configured OpenShell sandbox namespace"
            )
        if (
            EXTERNAL_SANDBOX_NAMESPACE
            and args.openshell_namespace != EXTERNAL_SANDBOX_NAMESPACE
        ):
            raise ConfigError(
                "strict ownership permits the external namespace only as --openshell-namespace"
            )
        if {
            args.namespace,
            args.system_namespace,
            args.openshell_namespace,
        } - OWNED_NAMESPACES:
            raise ConfigError(
                "strict ownership refuses a namespace outside TRACE_REPLAY_OWNED_NAMESPACES"
            )
        if args.allow_cross_node:
            raise ConfigError("strict ownership refuses --allow-cross-node")
        for name in (args.workspace_pvc, *workspace_pvcs.values()):
            if name and not name.startswith(owned_prefix):
                raise ConfigError(
                    f"strict ownership refuses PVC name outside {owned_prefix}*: {name}"
                )
    if len(target_nodes) > 1 and args.allow_cross_node:
        raise ConfigError(
            "--allow-cross-node is incompatible with controlled multi-node placement; "
            "omit it when using --nodes"
        )
    if workspace_pvcs and args.workspace_access_mode != "ReadWriteOnce":
        raise ConfigError(
            "--workspace-pvcs is only valid with --workspace-access-mode ReadWriteOnce"
        )
    if workspace_pvcs and args.workspace_pvc:
        raise ConfigError("use either --workspace-pvc or --workspace-pvcs, not both")
    if args.warm_pool_size < 0 or args.warm_pool_size > args.agents:
        raise ConfigError("--warm-pool-size must be between 0 and --agents")
    if (
        args.execution_workers is not None
        and not 1 <= args.execution_workers <= args.agents
    ):
        raise ConfigError("--execution-workers must be between 1 and --agents")
    if args.job_dispatch_interval < 0:
        raise ConfigError("--job-dispatch-interval cannot be negative")
    try:
        tool_cpu_ids = tuple(
            int(value.strip())
            for value in args.tool_cpu_ids.split(",")
            if value.strip()
        )
    except ValueError:
        raise ConfigError(
            "--tool-cpu-ids requires comma-separated nonnegative integers"
        )
    if any(value < 0 for value in tool_cpu_ids) or len(tool_cpu_ids) != len(
        set(tool_cpu_ids)
    ):
        raise ConfigError("--tool-cpu-ids requires unique nonnegative logical CPU IDs")
    if args.tool_cgroup_quota.strip():
        try:
            tool_cgroup_quota = parse_tool_cgroup_quota(
                args.tool_cgroup_quota, args.tool_cgroup_period_us
            )
        except ValueError as exc:
            raise ConfigError(str(exc))
    else:
        tool_cgroup_quota = ""
    if tool_cgroup_quota:
        try:
            sandbox_cpu_limit = tool_cgroup_quota_cpu_quantity(tool_cgroup_quota)
        except ValueError as exc:
            raise ConfigError(str(exc))
    else:
        sandbox_cpu_limit = None
    if not 1 <= args.cluster_api_workers <= 128:
        raise ConfigError("--cluster-api-workers must be between 1 and 128")
    if args.cluster_retry_timeout < 0:
        raise ConfigError("--cluster-retry-timeout cannot be negative")
    if args.cluster_retry_interval <= 0:
        raise ConfigError("--cluster-retry-interval must be positive")
    if args.max_traces is not None and args.max_traces < 1:
        raise ConfigError("--max-traces must be positive")
    if args.session_id and args.session_ids:
        raise ConfigError("use only one of --session-id or --session-ids")
    if args.require_traces is not None and args.require_traces < 1:
        raise ConfigError("--require-traces must be positive")
    if bool(args.reuse_existing_agent_prefix) != bool(
        args.reuse_existing_workspace_prefix
    ):
        raise ConfigError(
            "--reuse-existing-agent-prefix and --reuse-existing-workspace-prefix must be provided together"
        )
    if args.skip_resume_agent_reconcile and not args.resume_existing_run:
        raise ConfigError(
            "--skip-resume-agent-reconcile requires --resume-existing-run"
        )
    if args.resume_existing_run:
        if (
            not args.reuse_existing_agent_prefix
            or not args.reuse_existing_workspace_prefix
        ):
            raise ConfigError(
                "--resume-existing-run requires the owned agent and workspace prefixes"
            )
        if not args.resume_preferred_node:
            raise ConfigError("--resume-existing-run requires --resume-preferred-node")
        if args.resume_preferred_node not in target_nodes:
            raise ConfigError("--resume-preferred-node must be included in --nodes")
        if args.workspace_access_mode != "ReadWriteOnce":
            raise ConfigError(
                "--resume-existing-run currently requires the existing per-node ReadWriteOnce workspaces"
            )
        # A resumed fleet may safely run a filtered queue when explicit
        # session IDs and matching agent indices are supplied.  This keeps the
        # existing deployments/workspaces warm while retrying only unfinished
        # traces.  The complete queue remains the default when no filter is
        # provided; max-traces is still rejected because it is positional and
        # cannot identify the intended warm agent slots.
        if args.max_traces is not None:
            raise ConfigError(
                "--resume-existing-run cannot be combined with --max-traces"
            )
        if args.reuse_agents:
            raise ConfigError(
                "--resume-existing-run already enables owned partial reuse"
            )
    if args.deployment_only and not args.execute:
        raise ConfigError("--deployment-only requires --execute")
    # Older explicitly-owned OpenShell fleets may use the gateway-managed
    # workspace (no controller PVC).  Reusing such a fleet must not force a
    # new PVC or attach an unrelated volume; fresh runs still require the
    # normal RWX/RWO storage checks below.
    reuse_gateway_managed_workspaces = (
        bool(args.reuse_existing_agent_prefix)
        and bool(args.reuse_existing_workspace_prefix)
        and not args.workspace_pvc
        and not workspace_pvcs
    )
    if (
        args.reuse_existing_agent_prefix
        and not reuse_gateway_managed_workspaces
        and not args.workspace_pvc
        and not workspace_pvcs
        and args.workspace_access_mode == "ReadWriteMany"
    ):
        raise ConfigError("--workspace-pvc is required when reusing an existing run")
    if args.trace_offset < 0:
        raise ConfigError("--trace-offset cannot be negative")
    if args.client_gap_ms < 0:
        raise ConfigError("--client-gap-ms must be nonnegative")
    if args.agent_offset < 0 or args.agent_offset >= args.agents:
        raise ConfigError("--agent-offset must be between 0 and agents-1")
    selected_session_ids = {
        value.strip() for value in (args.session_ids or "").split(",") if value.strip()
    }
    try:
        selected_agent_indices = [
            int(value.strip())
            for value in (args.agent_indices or "").split(",")
            if value.strip()
        ]
    except ValueError:
        raise ConfigError(f"invalid --agent-indices: {args.agent_indices}") from None
    if selected_agent_indices and any(
        index < 1 or index > args.agents for index in selected_agent_indices
    ):
        raise ConfigError("--agent-indices values must be between 1 and agents")
    if selected_agent_indices and len(set(selected_agent_indices)) != len(
        selected_agent_indices
    ):
        raise ConfigError("--agent-indices must not contain duplicates")

    args.target_nodes = target_nodes
    args.workspace_pvcs = workspace_pvcs
    args.tool_cpu_ids = tool_cpu_ids
    args.tool_cgroup_quota = tool_cgroup_quota
    args.sandbox_cpu_limit = sandbox_cpu_limit
    args.selected_session_ids = selected_session_ids
    args.selected_agent_indices = selected_agent_indices
    args.reuse_gateway_managed_workspaces = reuse_gateway_managed_workspaces
    return args
