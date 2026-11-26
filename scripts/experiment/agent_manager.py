"""Agent deployments, mock LLM wiring, and OpenClaw plugin setup."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import re
import shlex
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from trace_replay_sim.cluster_planning import (
    is_allowed_node,
)
from trace_replay_sim.mock_plan import MockPlan, mock_plan

from .config import (
    BASE_JAEGER,
    BASE_MOCK_LLM,
    BASE_OPENCLAW,
    BASE_OPENCLAW_CONFIG,
    EXPERIMENT_OPENCLAW_IMAGE,
    EXPERIMENT_OPENSHELL_PLUGIN_VERSION,
    MOCK_CORPUS_CHUNK_BYTES,
    OPENSHELL_GATEWAY_NAME,
    RESOURCE_PREFIX,
    ROOT,
    log_phase,
    pack_mock_corpus_sessions,
    safe_name,
    workspace_name,
)
from .kubernetes import Cluster
from .sandbox_manager import _ensure_openshell_workspace


def _clone_config(
    cluster: Cluster,
    source: str,
    target: str,
    workspace: str,
    *,
    pin_sandbox: bool = True,
    shared_workspace_pvc: str | None = None,
    mock: MockPlan | None = None,
    agent_index: int = 1,
    gateway_endpoint: str | None = None,
) -> str:
    config = cluster.json(["-n", cluster.namespace, "get", "configmap", source])
    for key in ("uid", "resourceVersion", "creationTimestamp", "managedFields"):
        config.get("metadata", {}).pop(key, None)
    config["metadata"]["name"] = target
    config["metadata"]["labels"] = {
        "app": target,
        "managed-by": "trace-replay-experiment",
    }
    raw = json.loads(config["data"]["openclaw.json"])
    entry = (
        raw.setdefault("plugins", {})
        .setdefault("entries", {})
        .setdefault("openshell", {})
    )
    # Use a per-agent wrapper that injects the Kubernetes driver configuration
    # at sandbox-create time. The OpenShell sandbox pod is created by the
    # gateway after the workspace exists, so setting a nodeSelector only on the
    # OpenClaw deployment does not constrain the sandbox pod itself.
    entry["config"]["command"] = (
        "/opt/openshell/bin/openshell-pinned"
        if pin_sandbox
        else "/opt/openshell/bin/openshell"
    )
    # The supported plugin version for the public OpenClaw image reads the
    # workspace from OPENSHELL_WORKSPACE.  Keeping it out of plugin config also
    # lets the init container validate the config before the plugin is installed.
    entry["config"].pop("workspace", None)
    # OpenClaw remote mode seeds and clears remoteWorkspaceDir on first use.
    # Keep its two remote roots below a dedicated PVC mount so seeding cannot
    # erase the staged SWE repository mounted at /sandbox/.openshell/workspace.
    # These paths also exist as writable mounts before the OpenClaw plugin
    # connects, avoiding mkdir attempts against the sandbox image filesystem.
    entry["config"]["remoteWorkspaceDir"] = "/sandbox/openclaw/workspace"
    entry["config"]["remoteAgentWorkspaceDir"] = "/sandbox/openclaw/agent"
    if gateway_endpoint:
        # Address the gateway by registered name (see OPENSHELL_GATEWAY_NAME).
        # The name is registered in the CLI config by the gateway container's
        # startup script, which is what lets SSH proxying (and therefore
        # workspace seeding) work; passing the bare endpoint URL is not
        # resolvable.  Drop any endpoint left over from the base config so the
        # two identities cannot disagree.
        entry["config"].pop("gatewayEndpoint", None)
        entry["config"]["gateway"] = OPENSHELL_GATEWAY_NAME
    # Long research traces can legitimately spend several minutes crossing
    # repeated model/tool boundaries.  The base deployment carries a 180s
    # sandbox timeout for ordinary interactive use; raise it only in this
    # experiment-owned config so a long recorded session is not truncated.
    entry["config"]["timeoutSeconds"] = int(
        os.environ.get("TRACE_REPLAY_OPEN_SHELL_TIMEOUT_SECONDS", "900")
    )
    if mock:
        for provider in raw.get("models", {}).get("providers", {}).values():
            if isinstance(provider, dict) and provider.get("baseUrl"):
                provider["baseUrl"] = mock.endpoint(agent_index)
                # OpenClaw requires a credential for endpoint providers even
                # when the in-cluster mock does not authenticate requests.
                # Without this field it fails before contacting the simulator.
                provider["apiKey"] = "change-me-gateway-token"
    # The replay corpus contains deliberately long research/tool-output
    # histories.  The experiment uses a deterministic simulator, so there is
    # no inference-memory trade-off here; the harness must still accept the
    # recorded context in order to measure its assembly and OpenShell path.
    # Keep this override scoped to experiment-owned OpenClaw configs.
    context_limit = int(os.environ.get("TRACE_REPLAY_REPLAY_CONTEXT_LENGTH", "262144"))
    model_sets = [raw.get("agents", {}).get("defaults", {}).get("models", {})]
    model_sets.extend(
        agent.get("models", {})
        for agent in raw.get("agents", {}).get("list", [])
        if isinstance(agent, dict)
    )
    for models in model_sets:
        if not isinstance(models, dict):
            continue
        for model_config in models.values():
            if not isinstance(model_config, dict):
                continue
            params = model_config.setdefault("params", {})
            if isinstance(params, dict):
                params["contextLength"] = context_limit
    rendered_config = json.dumps(raw, sort_keys=True, separators=(",", ":"))
    config["data"]["openclaw.json"] = rendered_config
    cluster.apply_json(config)
    return hashlib.sha256(rendered_config.encode("utf-8")).hexdigest()


def _ensure_openshell_client_secret(
    cluster: Cluster,
    source_namespace: str,
    source_name: str,
    target_name: str,
    oidc_token_path: Path | None = None,
) -> str:
    """Copy the gateway client bundle into the agent namespace under an owned name.

    OpenShell's Kubernetes TLS Secret is namespaced. The gateway and OpenClaw
    agent usually live in separate namespaces, so a fresh run publishes a
    run-scoped copy for the agent pod. Secret data is never logged.
    """
    source = cluster.json(["-n", source_namespace, "get", "secret", source_name])
    existing = cluster.run(
        ["-n", cluster.namespace, "get", "secret", target_name], check=False
    ).strip()
    if existing:
        current = cluster.json(["-n", cluster.namespace, "get", "secret", target_name])
        labels = current.get("metadata", {}).get("labels", {})
        if labels.get("managed-by") != "trace-replay-experiment":
            raise RuntimeError(
                f"refusing to modify unowned client Secret {cluster.namespace}/{target_name}"
            )
    target = {
        "apiVersion": "v1",
        "kind": "Secret",
        "metadata": {
            "name": target_name,
            "namespace": cluster.namespace,
            "labels": {"managed-by": "trace-replay-experiment"},
        },
        "type": source.get("type", "Opaque"),
        "data": source.get("data", {}),
    }
    if oidc_token_path is not None:
        token_path = oidc_token_path.resolve()
        if not token_path.is_file():
            raise RuntimeError(
                f"OpenShell OIDC access token file is missing: {token_path}"
            )
        target["data"]["oidc_token.json"] = base64.b64encode(
            json.dumps(
                {
                    "access_token": token_path.read_text(encoding="utf-8").strip(),
                    "issuer": "https://aharush-oidc.aharush-agentic-cpu.svc.cluster.local:8443",
                    "client_id": "aharush-openshell-cli",
                    "expires_at": int(time.time()) + 36 * 3600,
                }
            ).encode("utf-8")
        ).decode("ascii")
    cluster.apply_json_in(cluster.namespace, target)
    return target_name


def deploy_agents(
    cluster: Cluster,
    agents: int,
    *,
    reuse: bool,
    workspace_prefix: str,
    deployment_prefix: str | None = None,
    shared_workspace_pvc: str | None = None,
    appworld_endpoint: str | None = None,
    internal_api_endpoints: list[str] | None = None,
    mock: MockPlan | None = None,
    gateway_endpoint: str | None = None,
    openshell_client_secret: str | None = None,
    allow_cross_node: bool = False,
    agent_nodes: list[str] | tuple[str, ...] | None = None,
    shared_workspace_pvcs: dict[str, str] | None = None,
    sandbox_images: list[str | None] | None = None,
    swe_command_maps: list[dict[str, list[str]] | None] | None = None,
    agent_indices: list[int] | tuple[int, ...] | None = None,
    reconcile_indices: list[int] | tuple[int, ...] | None = None,
    reuse_existing_workspaces: bool = False,
    simulated_ttft_ms: float | None = None,
    simulated_itl_ms: float | None = None,
    sandbox_cpu_limit: str | None = None,
) -> list[tuple[str, str, str]]:
    log_phase(
        f"deployment: preparing {agents} OpenClaw agents with isolated OpenShell workspaces"
    )
    planned_nodes = tuple(agent_nodes or (cluster.node,))
    if len(planned_nodes) != agents:
        raise ValueError(
            "agent_nodes must contain exactly one node assignment per agent"
        )
    if sandbox_images is not None and len(sandbox_images) != agents:
        raise ValueError(
            "sandbox_images must contain exactly one image assignment per agent"
        )
    if swe_command_maps is not None and len(swe_command_maps) != agents:
        raise ValueError("swe_command_maps must contain exactly one mapping per agent")
    global_indices = tuple(agent_indices or range(1, agents + 1))
    if (
        len(global_indices) != agents
        or len(set(global_indices)) != agents
        or any(index < 1 for index in global_indices)
    ):
        raise ValueError(
            "agent_indices must contain unique positive global agent indices"
        )
    reconcile_set = set(reconcile_indices or global_indices)
    if not reconcile_set.issubset(global_indices):
        raise ValueError(
            "reconcile_indices must refer to deployed global agent indices"
        )
    log_phase(
        f"deployment: workspace storage map={shared_workspace_pvcs or shared_workspace_pvc or 'per-sandbox-default'}"
    )

    openshell_cli_version = os.environ.get(
        "TRACE_REPLAY_OPENSHELL_CLI_VERSION", "0.1.8"
    ).strip()
    if not re.fullmatch(r"\d+\.\d+\.\d+", openshell_cli_version):
        raise ValueError(f"invalid OpenShell CLI version: {openshell_cli_version!r}")

    def pin_openshell_cli_version(deployment_obj: dict[str, Any]) -> bool:
        """Keep the agent-side CLI aligned with the upstream gateway release."""
        for init_container in (
            deployment_obj.get("spec", {})
            .get("template", {})
            .get("spec", {})
            .get("initContainers", [])
        ):
            if init_container.get("name") != "install-openshell-cli":
                continue
            args = init_container.setdefault("args", [])
            for index, arg in enumerate(args):
                script, count = re.subn(
                    r'(?m)^(\s*ver\s*=\s*)["\'][^"\']+["\']',
                    rf'\g<1>"{openshell_cli_version}"',
                    str(arg),
                    count=1,
                )
                if count:
                    changed = script != arg
                    args[index] = script
                    return changed
            raise RuntimeError(
                "OpenClaw deployment has no recognized OpenShell CLI version in install-openshell-cli"
            )
        raise RuntimeError(
            "OpenClaw deployment has no install-openshell-cli init container"
        )

    def pin_openshell_plugin_version(deployment_obj: dict[str, Any]) -> bool:
        """Match the external plugin release to the OpenClaw runtime API."""
        spec = deployment_obj.get("spec", {}).get("template", {}).get("spec", {})
        for init_container in spec.get("initContainers", []):
            if init_container.get("name") != "init-config":
                continue
            args = init_container.setdefault("args", [""])
            command = str(args[0] if args else "")
            replacement = (
                f"@openclaw/openshell-sandbox@{EXPERIMENT_OPENSHELL_PLUGIN_VERSION}"
            )
            updated, count = re.subn(
                r"@openclaw/openshell-sandbox(?:@[^\s]+)?",
                replacement,
                command,
                count=1,
            )
            if not count:
                raise RuntimeError(
                    "OpenClaw init-config has no OpenShell plugin install command"
                )
            args[0] = updated
            return updated != command
        raise RuntimeError("OpenClaw deployment has no init-config container")

    def install_plugin_compat_shim(deployment_obj: dict[str, Any]) -> bool:
        """Patch the public image's stale SDK export in the shared OpenClaw home."""
        # OpenShell's Kubernetes API in this environment enforces a 19-character
        # sandbox-name limit, while the pinned plugin's generated name can be
        # up to 50 characters (``openclaw-<scope>-<hash>``).  Keep the first
        # readable character and retain the hash suffix for per-session
        # uniqueness.  Bump the marker so an already-deployed experiment fleet
        # receives this fix on the next idempotent reconciliation.
        marker = "# trace-replay-compat-isPathInside-v2-openshell-name"
        shim = (
            "              # trace-replay-compat-isPathInside-v1\n"
            '              node --input-type=module -e \'import fs from "node:fs"; import { execFileSync } from "node:child_process"; const files=execFileSync("find",["/home/node/.openclaw/npm/projects","-path","*/@openclaw/openshell-sandbox/dist/index.js","-print"],{encoding:"utf8"}).trim().split("\\n").filter(Boolean); for (const file of files) { let text=fs.readFileSync(file,"utf8"); if (text.includes("trace-replay-compat-isPathInside-v1")) continue; const needle="import { isPathInside, root } from \\"openclaw/plugin-sdk/file-access-runtime\\";"; if (!text.includes(needle)) continue; const replacement="import { root } from \\"openclaw/plugin-sdk/file-access-runtime\\";\\n// trace-replay-compat-isPathInside-v1\\nconst isPathInside=(basePath,candidatePath)=>{const base=path.resolve(basePath);const candidate=path.resolve(candidatePath);return candidate===base||candidate.startsWith(base+path.sep);};"; fs.writeFileSync(file,text.replace(needle,replacement)); }\'\n'
        )
        # Use a shell-native, idempotent replacement here.  Keeping the
        # compatibility edit on the shared OpenClaw home avoids modifying the
        # base image and works for both fresh and resumed deployments.
        shim = (
            "              # trace-replay-compat-isPathInside-v2-openshell-name\n"
            "              find /home/node/.openclaw/npm/projects -path '*/@openclaw/openshell-sandbox/dist/index.js' -exec sed -i 's@import { isPathInside, root } from \"openclaw/plugin-sdk/file-access-runtime\";@import { root } from \"openclaw/plugin-sdk/file-access-runtime\"; /* trace-replay-compat-isPathInside-v1 */ const isPathInside=(basePath,candidatePath)=>{const base=path.resolve(basePath);const candidate=path.resolve(candidatePath);return candidate===base||candidate.startsWith(base+path.sep);};@' {} +\n"
            "              # The upstream plugin builds names as openclaw-<scope>-<hash>\n"
            "              # (up to 50 chars); this gateway accepts at most 19.\n"
            "              find /home/node/.openclaw/npm/projects -path '*/@openclaw/openshell-sandbox/dist/index.js' -exec sed -i 's@slice(0, 32)@slice(0, 1)@g' {} +\n"
        )
        for init_container in (
            deployment_obj.get("spec", {})
            .get("template", {})
            .get("spec", {})
            .get("initContainers", [])
        ):
            if init_container.get("name") != "init-config":
                continue
            args = init_container.setdefault("args", [""])
            command = str(args[0] if args else "")
            if marker in command:
                return False
            anchor = "node openclaw.mjs doctor"
            anchor_pos = command.find(anchor)
            if anchor_pos < 0:
                raise RuntimeError(
                    "base OpenClaw deployment init-config has no doctor step"
                )
            line_start = command.rfind("\n", 0, anchor_pos) + 1
            args[0] = command[:line_start] + shim + command[line_start:]
            return True
        raise RuntimeError("base OpenClaw deployment has no init-config container")

    def install_swe_preflight_allowlist(
        deployment_obj: dict[str, Any], command_map: dict[str, list[str]] | None
    ) -> bool:
        """Permit only corpus-pinned SWE commands past OpenClaw's exec guards.

        OpenClaw still receives each original bash command and routes exec to
        its OpenShell sandbox. The SSH adapter then maps the command to its
        staged script. The pinned 2026.8.1-beta.2 bundle otherwise rejects
        recorded heredoc/interpreter combinations and rejects two recorded
        shell strings with unclosed quotes before the adapter runs. The
        experiment-owned overlay changes these guards only for exact command
        hashes present in this session's validated command map. The original
        shell string is never executed: the SSH adapter selects its script.
        """
        if not command_map:
            return False
        hashes = sorted(command_map)
        if not all(re.fullmatch(r"[0-9a-f]{64}", value) for value in hashes):
            raise ValueError("invalid SWE command hash allowlist")
        spec = deployment_obj["spec"]["template"]["spec"]
        bundle = "bash-tools-BoFtiF3w.js"
        volume_name = "swe-preflight-patch"
        if not any(
            volume.get("name") == volume_name for volume in spec.get("volumes", [])
        ):
            spec.setdefault("volumes", []).append({"name": volume_name, "emptyDir": {}})
        patch_program = """
const fs = require("node:fs");
const source = "/app/dist/bash-tools-BoFtiF3w.js";
const destination = "/swe-preflight-patch/bash-tools-BoFtiF3w.js";
let code = fs.readFileSync(source, "utf8");
const needle = 'if (hasInterpreterInvocation && hasComplexSyntax && (hasInterpreterSegmentScriptHint || hasInterpreterPipelineScriptHint || hasProcessSubstitution && isDirectInterpreterCommand)) throw new Error("exec preflight:';
const replacement = 'if (hasInterpreterInvocation && hasComplexSyntax && (hasInterpreterSegmentScriptHint || hasInterpreterPipelineScriptHint || hasProcessSubstitution && isDirectInterpreterCommand) && !String(process.env.TRACE_REPLAY_SWE_ALLOWED_COMMAND_HASHES || "").split(",").includes(crypto.createHash("sha256").update(params.command).digest("hex"))) throw new Error("exec preflight:';
if (code.split(needle).length !== 2) throw new Error("pinned OpenClaw preflight bundle changed");
code = code.replace(needle, replacement);
fs.writeFileSync(destination, code);
const sshSource = "/app/dist/ssh-backend-Bu8TBAEJ.js";
const sshDestination = "/swe-preflight-patch/ssh-backend-Bu8TBAEJ.js";
let sshCode = fs.readFileSync(sshSource, "utf8");
const sshNeedle = 'function buildValidatedExecRemoteCommand(params) {\\n\\tassertValidExecRemoteCommand(params.command);';
const sshReplacement = 'function buildValidatedExecRemoteCommand(params) {\\n\\tif (!String(process.env.TRACE_REPLAY_SWE_ALLOWED_COMMAND_HASHES || "").split(",").includes(hashTextSha256(params.command))) assertValidExecRemoteCommand(params.command);';
if (sshCode.split(sshNeedle).length !== 2) throw new Error("pinned OpenClaw SSH backend changed");
sshCode = sshCode.replace(sshNeedle, sshReplacement);
fs.writeFileSync(sshDestination, sshCode);
"""
        marker = "# trace-replay-swe-preflight-allowlist-v1"
        for container in spec.get("initContainers", []):
            if container.get("name") != "init-config":
                continue
            mounts = container.setdefault("volumeMounts", [])
            if not any(mount.get("name") == volume_name for mount in mounts):
                mounts.append(
                    {"name": volume_name, "mountPath": "/swe-preflight-patch"}
                )
            args = container.setdefault("args", [""])
            command = str(args[0] if args else "")
            if marker not in command:
                anchor = "node openclaw.mjs doctor"
                anchor_pos = command.find(anchor)
                if anchor_pos < 0:
                    raise RuntimeError("base OpenClaw init-config has no doctor step")
                line_start = command.rfind("\n", 0, anchor_pos) + 1
                insertion = (
                    marker
                    + "\nnode <<'TRACE_REPLAY_SWE_PATCH'\n"
                    + patch_program
                    + "\nTRACE_REPLAY_SWE_PATCH\n"
                )
                args[0] = command[:line_start] + insertion + command[line_start:]
            break
        else:
            raise RuntimeError("base OpenClaw deployment has no init-config container")
        for container in spec.get("containers", []):
            if container.get("name") != "gateway":
                continue
            if container.get("image") != EXPERIMENT_OPENCLAW_IMAGE:
                raise RuntimeError(
                    "SWE preflight overlay requires the pinned OpenClaw image"
                )
            env = container.setdefault("env", [])
            env[:] = [
                item
                for item in env
                if item.get("name") != "TRACE_REPLAY_SWE_ALLOWED_COMMAND_HASHES"
            ]
            env.append(
                {
                    "name": "TRACE_REPLAY_SWE_ALLOWED_COMMAND_HASHES",
                    "value": ",".join(hashes),
                }
            )
            mounts = container.setdefault("volumeMounts", [])
            if not any(mount.get("name") == volume_name for mount in mounts):
                mounts.append(
                    {
                        "name": volume_name,
                        "mountPath": f"/app/dist/{bundle}",
                        "subPath": bundle,
                        "readOnly": True,
                    }
                )
            ssh_bundle = "ssh-backend-Bu8TBAEJ.js"
            if not any(
                mount.get("mountPath") == f"/app/dist/{ssh_bundle}" for mount in mounts
            ):
                mounts.append(
                    {
                        "name": volume_name,
                        "mountPath": f"/app/dist/{ssh_bundle}",
                        "subPath": ssh_bundle,
                        "readOnly": True,
                    }
                )
            break
        else:
            raise RuntimeError("base OpenClaw deployment has no gateway container")
        annotation = "trace-replay.openai.com/swe-preflight-hashes"
        revision = hashlib.sha256(",".join(hashes).encode()).hexdigest()
        metadata = deployment_obj["spec"]["template"].setdefault("metadata", {})
        annotations = metadata.setdefault("annotations", {})
        changed = annotations.get(annotation) != revision
        annotations[annotation] = revision
        return changed

    def node_for(index: int) -> str:
        return planned_nodes[index - 1]

    def global_index_for(index: int) -> int:
        return global_indices[index - 1]

    def sandbox_image_for(index: int) -> str | None:
        return sandbox_images[index - 1] if sandbox_images is not None else None

    def swe_command_map_for(index: int) -> dict[str, list[str]] | None:
        return swe_command_maps[index - 1] if swe_command_maps is not None else None

    def workspace_pvc_for(node: str) -> str | None:
        if shared_workspace_pvcs:
            try:
                return shared_workspace_pvcs[node]
            except KeyError as exc:
                raise ValueError(
                    f"no workspace PVC configured for target node {node}"
                ) from exc
        return shared_workspace_pvc

    def install_local_mock_sidecar(deployment_obj: dict[str, Any]) -> bool:
        """Run an isolated replay simulator beside this OpenClaw gateway."""
        if not mock or not mock.sidecar:
            return False
        prefix = mock.prefix
        source_name = f"{prefix}-mock-src"
        configmaps = cluster.json(
            ["-n", cluster.namespace, "get", "configmaps", "-l", f"run={prefix}"]
        ).get("items", [])
        corpus_names = sorted(
            str(x.get("metadata", {}).get("name") or "")
            for x in configmaps
            if "-corpus-" in str(x.get("metadata", {}).get("name") or "")
        )
        if not corpus_names:
            raise RuntimeError(f"local mock assets missing for {prefix}")
        template = deployment_obj.setdefault("spec", {}).setdefault("template", {})
        metadata = template.setdefault("metadata", {}).setdefault("annotations", {})
        revision = hashlib.sha256(
            (prefix + ":" + ",".join(corpus_names)).encode()
        ).hexdigest()
        changed = metadata.get("trace-replay.openai.com/local-mock-sha256") != revision
        if simulated_ttft_ms is not None or simulated_itl_ms is not None:
            changed = True  # env-var timing override requires a deployment update
        metadata["trace-replay.openai.com/local-mock-sha256"] = revision
        spec = template.setdefault("spec", {})
        managed = {
            "mock-src",
            "mock-edges",
            *(f"mock-corpus-{i:03d}" for i in range(len(corpus_names))),
        }
        spec["volumes"] = [
            v for v in spec.get("volumes", []) if v.get("name") not in managed
        ]
        spec["volumes"].extend(
            [
                {"name": "mock-src", "configMap": {"name": source_name}},
                *[
                    {"name": f"mock-corpus-{i:03d}", "configMap": {"name": name}}
                    for i, name in enumerate(corpus_names)
                ],
                {"name": "mock-edges", "emptyDir": {}},
            ]
        )
        spec["containers"] = [
            c for c in spec.get("containers", []) if c.get("name") != "mock-llm"
        ]
        _mock_env = [
            {"name": "REPLAY_CORPUS", "value": "/data/corpus"},
            {"name": "REPLAY_MODE", "value": "turns"},
            {"name": "REPLAY_TIMING", "value": "humaneval-vllm-c1"},
            {"name": "MAP_TOOLS_TO_SANDBOX", "value": "1"},
            {"name": "MOCK_EDGES_PATH", "value": "/edges/mock_edges.jsonl"},
            {
                "name": "OTLP_ENDPOINT",
                "value": f"http://{BASE_JAEGER}.{cluster.namespace}.svc.cluster.local:4318",
            },
        ]
        if simulated_ttft_ms is not None or simulated_itl_ms is not None:
            _mock_env.extend(
                [
                    {"name": "SIMULATED_TIMING_OVERRIDE", "value": "1"},
                    {
                        "name": "SIMULATED_TTFT_MS",
                        "value": str(
                            simulated_ttft_ms if simulated_ttft_ms is not None else 0
                        ),
                    },
                    {
                        "name": "SIMULATED_ITL_MS",
                        "value": str(
                            simulated_itl_ms if simulated_itl_ms is not None else 0
                        ),
                    },
                ]
            )
        spec["containers"].append(
            {
                "name": "mock-llm",
                "image": "python:3.12-slim",
                "command": ["python", "/app/mock_llm.py"],
                "ports": [{"name": "mock", "containerPort": 8080}],
                "env": _mock_env,
                "resources": {
                    "requests": {"cpu": "50m", "memory": "128Mi"},
                    "limits": {"cpu": "1", "memory": "512Mi"},
                },
                "readinessProbe": {
                    "httpGet": {"path": "/health", "port": "mock"},
                    "initialDelaySeconds": 1,
                    "periodSeconds": 2,
                    "failureThreshold": 30,
                },
                "volumeMounts": [
                    {"name": "mock-src", "mountPath": "/app", "readOnly": True},
                    *[
                        {
                            "name": f"mock-corpus-{i:03d}",
                            "mountPath": f"/data/corpus/part-{i:03d}",
                            "readOnly": True,
                        }
                        for i in range(len(corpus_names))
                    ],
                    {"name": "mock-edges", "mountPath": "/edges"},
                ],
            }
        )
        return changed

    def pin_existing_deployment(
        name: str,
        workspace: str,
        node: str,
        workspace_pvc: str | None,
        index: int,
        config_sha256: str | None = None,
    ) -> bool:
        """Refresh node and sandbox pinning on an explicitly owned deployment."""
        deployment_obj = cluster.json(
            ["-n", cluster.namespace, "get", "deployment", name]
        )
        shim_changed = install_plugin_compat_shim(deployment_obj)
        cli_changed = pin_openshell_cli_version(deployment_obj)
        plugin_changed = pin_openshell_plugin_version(deployment_obj)
        sidecar_changed = install_local_mock_sidecar(deployment_obj)
        preflight_changed = install_swe_preflight_allowlist(
            deployment_obj, swe_command_map_for(index)
        )
        template = deployment_obj.setdefault("spec", {}).setdefault("template", {})
        template_metadata = template.setdefault("metadata", {})
        template_annotations = template_metadata.setdefault("annotations", {})
        config_hash_changed = False
        if config_sha256:
            annotation = "trace-replay.openai.com/openclaw-config-sha256"
            config_hash_changed = template_annotations.get(annotation) != config_sha256
            template_annotations[annotation] = config_sha256
        template_spec = template.setdefault("spec", {})
        startup_changed = False
        if allow_cross_node:
            template_spec.pop("nodeSelector", None)
        else:
            template_spec["nodeSelector"] = {"kubernetes.io/hostname": node}
            startup_changed = _install_pinned_openshell_wrapper(
                deployment_obj,
                node,
                workspace,
                workspace_pvc,
                appworld_endpoint,
                internal_api_endpoints=internal_api_endpoints,
                gateway_endpoint=gateway_endpoint,
                sandbox_image=sandbox_image_for(index),
                swe_command_map=swe_command_map_for(index),
                sandbox_cpu_limit=sandbox_cpu_limit,
            )
        for container in template_spec.get("containers", []):
            if container.get("name") != "gateway":
                continue
            # Explicitly reused fleets may be pinned to a compatible OpenClaw
            # build.  Do not replace that image with the mutable experiment
            # `latest` tag: a restart can then reject the fleet's state DB
            # schema.  Fresh deployments set EXPERIMENT_OPENCLAW_IMAGE in
            # create_agent() below.
            env = container.setdefault("env", [])
            workspace_entry = next(
                (
                    item
                    for item in env
                    if item.get("name") == "OPENSHELL_FLEET_WORKSPACE"
                ),
                None,
            )
            if workspace_entry is None:
                env.append({"name": "OPENSHELL_FLEET_WORKSPACE", "value": workspace})
            else:
                workspace_entry["value"] = workspace
        cluster.apply_json(deployment_obj)
        return (
            shim_changed
            or config_hash_changed
            or sidecar_changed
            or cli_changed
            or plugin_changed
            or startup_changed
            or preflight_changed
        )

    if reuse and deployment_prefix:
        # A recovery run may reuse one explicitly named, experiment-owned
        # fleet. Refresh those deployments concurrently so a validation fix
        # does not require another long serial rollout. Every object is
        # ownership-checked before it is changed.
        def refresh_existing(index: int) -> tuple[str, str, str]:
            # ``index`` is the local position in this reconciliation batch;
            # ``global_index`` is the original fleet slot.  Resume calls can
            # pass a sparse ``agent_indices`` list (for example, [74]) while
            # providing a one-element ``agent_nodes`` list.  Using the global
            # slot to index the local node list misroutes that case and can
            # raise IndexError before any trace Jobs are created.
            global_index = global_index_for(index)
            stem = safe_name(deployment_prefix)
            name = f"{stem}-{global_index:02d}"
            config = f"{stem}-config-{global_index:02d}"
            workspace = workspace_name(workspace_prefix, global_index)
            target_node = node_for(index)
            target_workspace_pvc = workspace_pvc_for(target_node)
            log_phase(
                f"deployment slot {global_index:02d}: node={target_node} "
                f"workspace={workspace} pvc={target_workspace_pvc or 'per-sandbox-default'}"
            )
            if not cluster.run(
                ["-n", cluster.namespace, "get", "deployment", name], check=False
            ).strip():
                raise RuntimeError(
                    f"expected owned deployment is missing: {cluster.namespace}/{name}"
                )
            deployment_obj = cluster.json(
                ["-n", cluster.namespace, "get", "deployment", name]
            )
            if (deployment_obj.get("metadata", {}).get("labels") or {}).get(
                "managed-by"
            ) != "trace-replay-experiment":
                raise RuntimeError(
                    f"refusing to reuse unowned deployment {cluster.namespace}/{name}"
                )
            if global_index not in reconcile_set:
                gateway = next(
                    (
                        container
                        for container in deployment_obj.get("spec", {})
                        .get("template", {})
                        .get("spec", {})
                        .get("containers", [])
                        if container.get("name") == "gateway"
                    ),
                    {},
                )
                configured_workspace = next(
                    (
                        item.get("value")
                        for item in gateway.get("env", [])
                        if item.get("name") == "OPENSHELL_FLEET_WORKSPACE"
                    ),
                    None,
                )
                return name, str(configured_workspace or workspace), target_node
            # The wrapper and its driver configuration are rendered into the
            # gateway container filesystem by the deployment template.  Only
            # restart when reconciliation actually changes the init script;
            # this keeps resume idempotent and avoids needlessly rolling an
            # already-fixed fleet on every retry.
            needs_restart = True
            init_config_text = "\n".join(
                str(item)
                for container in deployment_obj.get("spec", {})
                .get("template", {})
                .get("spec", {})
                .get("initContainers", [])
                if container.get("name") == "init-config"
                for item in (container.get("args") or [])
            )
            # Applying a changed Deployment template starts a new ReplicaSet
            # asynchronously.  The trace watcher must not race that rollout;
            # otherwise it can try to exec into a gateway container that is
            # still in init (or no longer exists).
            wrapper_needs_rollout = (
                "trace-replay-openshell-node-selector-v31-" not in init_config_text
                or "sleep 2147483647" not in init_config_text
                or "openshell-pinned"
                not in "\n".join(
                    str(container)
                    for container in deployment_obj.get("spec", {})
                    .get("template", {})
                    .get("initContainers", [])
                )
            )
            config_obj = cluster.json(
                ["-n", cluster.namespace, "get", "configmap", config]
            )
            config_name_is_expected = config == f"{stem}-config-{global_index:02d}"
            if (config_obj.get("metadata", {}).get("labels") or {}).get(
                "managed-by"
            ) != "trace-replay-experiment" and not config_name_is_expected:
                raise RuntimeError(
                    f"refusing to reuse unowned ConfigMap {cluster.namespace}/{config}"
                )
            # The init container materializes the plugin and config into the
            # pod's emptyDir home. Reconcile the template first and restart
            # only when the rendered gateway actually changed. Rebooting an
            # already-correct reused fleet adds avoidable rollout/API load and
            # can spend most of the experiment window waiting for readiness.
            # (The ConfigMap is mounted only by init-config; updating it does
            # not restart an existing pod.)
            needs_restart = False
            config_sha256 = _clone_config(
                cluster,
                BASE_OPENCLAW_CONFIG,
                config,
                workspace,
                pin_sandbox=not allow_cross_node,
                shared_workspace_pvc=target_workspace_pvc,
                mock=mock,
                agent_index=index,
                gateway_endpoint=gateway_endpoint,
            )
            deployment_changed = pin_existing_deployment(
                name,
                workspace,
                target_node,
                target_workspace_pvc,
                index,
                config_sha256=config_sha256,
            )
            if deployment_changed or wrapper_needs_rollout:
                needs_restart = True
            if needs_restart:
                cluster.run(
                    [
                        "-n",
                        cluster.namespace,
                        "rollout",
                        "restart",
                        f"deployment/{name}",
                    ]
                )
                cluster.wait_rollout(name)
            cluster.apply_json(
                {
                    "apiVersion": "v1",
                    "kind": "Service",
                    "metadata": {
                        "name": name,
                        "labels": {
                            "app": name,
                            "managed-by": "trace-replay-experiment",
                        },
                    },
                    "spec": {
                        "clusterIP": "None",
                        "selector": {"app": name},
                        "ports": [
                            {"name": "gateway", "port": 18790, "targetPort": 18790}
                        ],
                    },
                }
            )
            deployed = cluster.json(
                ["-n", cluster.namespace, "get", "deployment", name]
            )
            gateway = next(
                (
                    container
                    for container in deployed["spec"]["template"]["spec"].get(
                        "containers", []
                    )
                    if container.get("name") == "gateway"
                ),
                {},
            )
            configured_workspace = next(
                (
                    item.get("value")
                    for item in gateway.get("env", [])
                    if item.get("name") == "OPENSHELL_FLEET_WORKSPACE"
                ),
                None,
            )
            if configured_workspace:
                workspace = str(configured_workspace)
            if not reuse_existing_workspaces or workspace.startswith(
                f"{RESOURCE_PREFIX}-"
            ):
                _ensure_openshell_workspace(
                    cluster,
                    name,
                    workspace,
                    gateway_endpoint
                    or "http://openshell.openshell-tracesim.svc.cluster.local:8080",
                )
            return name, workspace, target_node

        with ThreadPoolExecutor(max_workers=min(8, agents)) as pool:
            bindings = list(pool.map(refresh_existing, range(1, agents + 1)))
        return bindings

    def create_agent(index: int) -> tuple[str, str, str]:
        target_node = node_for(index)
        target_workspace_pvc = workspace_pvc_for(target_node)
        global_index = global_index_for(index)
        if reuse:
            if deployment_prefix:
                stem = safe_name(deployment_prefix)
                name = f"{stem}-{global_index:02d}"
                config = f"{stem}-config-{global_index:02d}"
            else:
                name = f"{RESOURCE_PREFIX}-openclaw-agent-{global_index}"
                config = f"{RESOURCE_PREFIX}-openclaw-config-{global_index}"
        else:
            stem = safe_name(deployment_prefix or f"{RESOURCE_PREFIX}-openclaw-agent")
            name = f"{stem}-{global_index:02d}"
            config = f"{stem}-config-{global_index:02d}"
        service = name
        workspace = workspace_name(workspace_prefix, global_index)
        existing = cluster.run(
            ["-n", cluster.namespace, "get", "deployment", name], check=False
        ).strip()
        if reuse and existing:
            deployment_obj = cluster.json(
                ["-n", cluster.namespace, "get", "deployment", name]
            )
            labels = deployment_obj.get("metadata", {}).get("labels") or {}
            if labels.get("managed-by") != "trace-replay-experiment":
                raise RuntimeError(
                    f"refusing to reuse unowned deployment {cluster.namespace}/{name}"
                )
            config_obj = cluster.json(
                ["-n", cluster.namespace, "get", "configmap", config]
            )
            config_labels = config_obj.get("metadata", {}).get("labels") or {}
            if config_labels.get("managed-by") != "trace-replay-experiment":
                raise RuntimeError(
                    f"refusing to reuse unowned ConfigMap {cluster.namespace}/{config}"
                )
            config_sha256 = _clone_config(
                cluster,
                BASE_OPENCLAW_CONFIG,
                config,
                workspace,
                pin_sandbox=not allow_cross_node,
                shared_workspace_pvc=target_workspace_pvc,
                mock=mock,
                agent_index=index,
                gateway_endpoint=gateway_endpoint,
            )
            pin_existing_deployment(
                name,
                workspace,
                target_node,
                target_workspace_pvc,
                index,
                config_sha256=config_sha256,
            )
            cluster.run(
                ["-n", cluster.namespace, "rollout", "restart", f"deployment/{name}"]
            )
            cluster.wait_rollout(name)
            service_obj = {
                "apiVersion": "v1",
                "kind": "Service",
                "metadata": {
                    "name": service,
                    "labels": {"app": name, "managed-by": "trace-replay-experiment"},
                },
                "spec": {
                    "clusterIP": "None",
                    "selector": {"app": name},
                    "ports": [{"name": "gateway", "port": 18790, "targetPort": 18790}],
                },
            }
            cluster.apply_json(service_obj)
            deployed = cluster.json(
                ["-n", cluster.namespace, "get", "deployment", name]
            )
            gateway = next(
                (
                    container
                    for container in deployed["spec"]["template"]["spec"].get(
                        "containers", []
                    )
                    if container.get("name") == "gateway"
                ),
                {},
            )
            configured_workspace = next(
                (
                    item.get("value")
                    for item in gateway.get("env", [])
                    if item.get("name") == "OPENSHELL_FLEET_WORKSPACE"
                ),
                None,
            )
            if configured_workspace:
                workspace = str(configured_workspace)
            if not reuse_existing_workspaces or workspace.startswith(
                f"{RESOURCE_PREFIX}-"
            ):
                _ensure_openshell_workspace(
                    cluster,
                    name,
                    workspace,
                    gateway_endpoint
                    or "http://openshell.openshell-tracesim.svc.cluster.local:8080",
                )
            return service, workspace, target_node
        if existing:
            raise RuntimeError(
                f"refusing to modify existing fresh-run deployment {cluster.namespace}/{name}; "
                "use a new output directory/run identity"
            )
        if cluster.run(
            ["-n", cluster.namespace, "get", "configmap", config], check=False
        ).strip():
            raise RuntimeError(
                f"refusing to modify existing fresh-run ConfigMap {cluster.namespace}/{config}; "
                "use a new output directory/run identity"
            )
        if cluster.run(
            ["-n", cluster.namespace, "get", "service", service], check=False
        ).strip():
            raise RuntimeError(
                f"refusing to modify existing fresh-run Service {cluster.namespace}/{service}; "
                "use a new output directory/run identity"
            )
        _clone_config(
            cluster,
            BASE_OPENCLAW_CONFIG,
            config,
            workspace,
            pin_sandbox=not allow_cross_node,
            shared_workspace_pvc=target_workspace_pvc,
            mock=mock,
            agent_index=index,
            gateway_endpoint=gateway_endpoint,
        )
        deployment = cluster.json(
            [
                "-n",
                cluster.namespace,
                "get",
                "deployment",
                name if existing else BASE_OPENCLAW,
            ]
        )
        for key in (
            "uid",
            "resourceVersion",
            "generation",
            "creationTimestamp",
            "managedFields",
            "status",
        ):
            deployment.get("metadata", {}).pop(key, None)
        deployment["metadata"]["name"] = name
        deployment["metadata"]["labels"] = {
            "app": name,
            "managed-by": "trace-replay-experiment",
        }
        spec = deployment["spec"]
        spec["replicas"] = 1
        spec["selector"]["matchLabels"] = {"app": name}
        template = spec["template"]
        template["metadata"]["labels"] = {
            "app": name,
            "managed-by": "trace-replay-experiment",
        }
        template["metadata"].setdefault("annotations", {})[
            "trace-replay.experiment/reconciled-at"
        ] = str(time.time())
        if allow_cross_node:
            template["spec"].pop("nodeSelector", None)
        else:
            template["spec"]["nodeSelector"] = {"kubernetes.io/hostname": target_node}
        for volume in template["spec"].get("volumes", []):
            if volume.get("name") == "openclaw-shell-home":
                volume.clear()
                volume.update({"name": "openclaw-shell-home", "emptyDir": {}})
            if volume.get("name") == "config-template":
                volume["configMap"] = {"name": config}
            if volume.get("name") == "openshell-mtls" and openshell_client_secret:
                volume.setdefault("secret", {})["secretName"] = openshell_client_secret
        for container in template["spec"].get("containers", []):
            if container.get("name") == "gateway":
                container["image"] = EXPERIMENT_OPENCLAW_IMAGE
                env = [
                    item
                    for item in container.setdefault("env", [])
                    if item.get("name") != "OPENSHELL_FLEET_WORKSPACE"
                ]
                env.append({"name": "OPENSHELL_FLEET_WORKSPACE", "value": workspace})
                container["env"] = env
        for container in template["spec"].get("initContainers", []):
            if container.get("name") == "init-config":
                container["image"] = EXPERIMENT_OPENCLAW_IMAGE
        install_local_mock_sidecar(deployment)
        install_plugin_compat_shim(deployment)
        pin_openshell_cli_version(deployment)
        pin_openshell_plugin_version(deployment)
        install_swe_preflight_allowlist(deployment, swe_command_map_for(index))
        if not allow_cross_node:
            _install_pinned_openshell_wrapper(
                deployment,
                target_node,
                workspace,
                target_workspace_pvc,
                appworld_endpoint,
                internal_api_endpoints=internal_api_endpoints,
                gateway_endpoint=gateway_endpoint,
                sandbox_image=sandbox_image_for(index),
                swe_command_map=swe_command_map_for(index),
                sandbox_cpu_limit=sandbox_cpu_limit,
            )
        cluster.apply_json(deployment)
        service_obj = {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {
                "name": service,
                "labels": {"app": name, "managed-by": "trace-replay-experiment"},
            },
            "spec": {
                "clusterIP": "None",
                "selector": {"app": name},
                "ports": [{"name": "gateway", "port": 18790, "targetPort": 18790}],
            },
        }
        cluster.apply_json(service_obj)
        cluster.wait_rollout(name)
        _ensure_openshell_workspace(
            cluster,
            name,
            workspace,
            gateway_endpoint
            or "http://openshell.openshell-tracesim.svc.cluster.local:8080",
        )
        return service, workspace, target_node

    with ThreadPoolExecutor(max_workers=min(8, agents)) as pool:
        return list(pool.map(create_agent, range(1, agents + 1)))


def verify_agent_mock_endpoints(
    cluster: Cluster,
    agent_bindings: list[tuple[str, str, str]],
    mock: MockPlan,
    agent_indices: list[int] | tuple[int, ...] | None = None,
) -> None:
    """Prove that each selected running gateway loaded this run's mock URL.

    Updating the per-agent ConfigMap is not sufficient: the ConfigMap is read
    by an init container and an already-running gateway keeps its old rendered
    config.  Read the live file from the gateway so trace dispatch cannot begin
    with a stale data-plane endpoint.
    """
    indices = list(agent_indices or range(1, len(agent_bindings) + 1))

    def verify(index: int) -> None:
        expected = mock.endpoint(index)
        try:
            agent = agent_bindings[index - 1][0]
        except IndexError as exc:
            raise RuntimeError(
                f"agent endpoint verification references missing slot {index}"
            ) from exc
        rendered = cluster.run(
            [
                "-n",
                cluster.namespace,
                "exec",
                f"deployment/{agent}",
                "-c",
                "gateway",
                "--",
                "cat",
                "/home/node/.openclaw/openclaw.json",
            ]
        )
        try:
            payload = json.loads(rendered)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"agent {agent} has an unreadable live OpenClaw config"
            ) from exc
        urls = {
            str(provider.get("baseUrl") or "")
            for provider in (
                payload.get("models", {}).get("providers", {}) or {}
            ).values()
            if isinstance(provider, dict) and provider.get("baseUrl")
        }
        if urls != {expected}:
            raise RuntimeError(
                f"agent {agent} live mock endpoint mismatch: expected {expected}, observed {sorted(urls)}"
            )

    with ThreadPoolExecutor(max_workers=min(8, len(indices))) as pool:
        list(pool.map(verify, indices))
    log_phase(
        f"deployment: verified live mock endpoint on {len(indices)} selected agents"
    )


def _resume_agent_layout(
    cluster: Cluster,
    deployment_prefix: str,
    agents: int,
    preferred_node: str,
    planned_nodes: list[str] | tuple[str, ...] | None = None,
) -> tuple[list[str], list[int], list[int]]:
    """Inspect an owned partial fleet without changing cluster state.

    Running agents retain their observed node. Existing non-running agents
    retain the node selector already recorded on their Deployment; only
    absent slots are returned for controlled placement on ``preferred_node``.
    The caller performs the only mutations after capacity and ownership checks.
    """
    stem = safe_name(deployment_prefix)
    deployment_items = (
        cluster.json(
            [
                "-n",
                cluster.namespace,
                "get",
                "deploy",
                "-l",
                "managed-by=trace-replay-experiment",
            ]
        ).get("items")
        or []
    )
    deployments = {
        str(item.get("metadata", {}).get("name") or ""): item
        for item in deployment_items
        if str(item.get("metadata", {}).get("name") or "").startswith(f"{stem}-")
    }
    pod_items = (
        cluster.json(
            [
                "-n",
                cluster.namespace,
                "get",
                "pods",
                "-l",
                "managed-by=trace-replay-experiment",
            ]
        ).get("items")
        or []
    )
    running_pods_by_app: dict[str, dict[str, Any]] = {}
    for pod in pod_items:
        labels = pod.get("metadata", {}).get("labels") or {}
        app = str(labels.get("app") or "")
        if (
            app.startswith(f"{stem}-")
            and pod.get("status", {}).get("phase") == "Running"
        ):
            running_pods_by_app[app] = pod
    assignments: list[str] = []
    pending: list[int] = []
    missing: list[int] = []
    for index in range(1, agents + 1):
        name = f"{stem}-{index:02d}"
        deployment = deployments.get(name)
        if deployment is None:
            assignments.append(
                planned_nodes[index - 1]
                if planned_nodes is not None
                else preferred_node
            )
            missing.append(index)
            continue
        labels = deployment.get("metadata", {}).get("labels") or {}
        if labels.get("managed-by") != "trace-replay-experiment":
            raise RuntimeError(
                f"refusing to resume unowned deployment {cluster.namespace}/{name}"
            )
        pod = running_pods_by_app.get(name)
        if pod is not None:
            node = str(pod.get("spec", {}).get("nodeName") or "")
            if not is_allowed_node(node):
                raise RuntimeError(
                    f"owned deployment {name} is running on an unapproved node: {node or '<none>'}"
                )
            assignments.append(node)
            continue
        node_selector = (
            ((deployment.get("spec") or {}).get("template") or {})
            .get("spec", {})
            .get("nodeSelector", {})
        )
        node = str(node_selector.get("kubernetes.io/hostname") or preferred_node)
        if not is_allowed_node(node):
            raise RuntimeError(
                f"owned deployment {name} has an unapproved node selector: {node}"
            )
        assignments.append(node)
        pending.append(index)
    return assignments, pending, missing


def _collect_owned_agent_bindings(
    cluster: Cluster,
    deployment_prefix: str,
    workspace_prefix: str,
    agents: int,
) -> list[tuple[str, str, str]]:
    """Return the post-resume fleet in stable global agent-index order."""
    stem = safe_name(deployment_prefix)
    deployment_items = (
        cluster.json(
            [
                "-n",
                cluster.namespace,
                "get",
                "deploy",
                "-l",
                "managed-by=trace-replay-experiment",
            ]
        ).get("items")
        or []
    )
    deployments = {
        str(item.get("metadata", {}).get("name") or ""): item
        for item in deployment_items
        if str(item.get("metadata", {}).get("name") or "").startswith(f"{stem}-")
    }
    pod_items = (
        cluster.json(
            [
                "-n",
                cluster.namespace,
                "get",
                "pods",
                "-l",
                "managed-by=trace-replay-experiment",
            ]
        ).get("items")
        or []
    )
    running_pods_by_app: dict[str, dict[str, Any]] = {}
    for pod in pod_items:
        labels = pod.get("metadata", {}).get("labels") or {}
        app = str(labels.get("app") or "")
        if (
            app.startswith(f"{stem}-")
            and pod.get("status", {}).get("phase") == "Running"
        ):
            running_pods_by_app[app] = pod
    bindings: list[tuple[str, str, str]] = []
    for index in range(1, agents + 1):
        name = f"{stem}-{index:02d}"
        deployment = deployments.get(name)
        if deployment is None:
            raise RuntimeError(
                f"resumed agent deployment is missing: {cluster.namespace}/{name}"
            )
        labels = deployment.get("metadata", {}).get("labels") or {}
        if labels.get("managed-by") != "trace-replay-experiment":
            raise RuntimeError(
                f"refusing to collect unowned deployment {cluster.namespace}/{name}"
            )
        pod = running_pods_by_app.get(name)
        if pod is None:
            raise RuntimeError(
                f"resumed agent is not Running: {cluster.namespace}/{name}"
            )
        node = str(pod.get("spec", {}).get("nodeName") or "")
        if not is_allowed_node(node):
            raise RuntimeError(
                f"resumed deployment {name} is running on an unapproved node: {node or '<none>'}"
            )
        workspace = workspace_name(workspace_prefix, index)
        for container in (
            deployment.get("spec", {})
            .get("template", {})
            .get("spec", {})
            .get("containers", [])
        ):
            if container.get("name") != "gateway":
                continue
            for entry in container.get("env", []):
                if entry.get("name") == "OPENSHELL_FLEET_WORKSPACE" and entry.get(
                    "value"
                ):
                    workspace = str(entry["value"])
                    break
        bindings.append((name, workspace, node))
    return bindings


def _install_pinned_openshell_wrapper(
    deployment: dict[str, Any],
    node: str,
    workspace: str,
    shared_workspace_pvc: str | None,
    appworld_endpoint: str | None = None,
    internal_api_endpoints: list[str] | None = None,
    gateway_endpoint: str | None = None,
    sandbox_image: str | None = None,
    sandbox_cpu_limit: str | None = None,
    swe_command_map: dict[str, list[str]] | None = None,
) -> bool:
    """Install a wrapper that pins and storage-wires sandbox pods at create time."""
    kubernetes_config: dict[str, Any] = {
        "pod": {"node_selector": {"kubernetes.io/hostname": node}},
    }
    if shared_workspace_pvc:
        kubernetes_config.update(
            {
                "volumes": [
                    {
                        "name": "trace-replay-shared-workspace",
                        "persistent_volume_claim": {
                            "claim_name": shared_workspace_pvc,
                            "read_only": False,
                        },
                    }
                ],
                "containers": {
                    "agent": {
                        "volume_mounts": [
                            {
                                "name": "trace-replay-shared-workspace",
                                # Keep the staged repo separate from OpenClaw's
                                # remote-mode seed and agent workspace directories.
                                "mount_path": "/sandbox/.openshell/workspace",
                                "sub_path": workspace,
                                "read_only": False,
                            },
                            {
                                "name": "trace-replay-shared-workspace",
                                "mount_path": "/sandbox/openclaw",
                                "sub_path": f"{workspace}-oclaw",
                                "read_only": False,
                            },
                        ],
                    },
                },
            }
        )
    driver_config_json = json.dumps(
        {"kubernetes": kubernetes_config}, separators=(",", ":")
    )
    if sandbox_image is not None:
        if not re.fullmatch(r"[A-Za-z0-9._/:@-]+", sandbox_image):
            raise ValueError(f"unsupported sandbox image reference: {sandbox_image!r}")
    endpoints = list(internal_api_endpoints or [])
    if appworld_endpoint and appworld_endpoint not in endpoints:
        endpoints.append(appworld_endpoint)
    network_policies: list[str] = []
    for index, endpoint in enumerate(endpoints, start=1):
        parsed = urlparse(endpoint)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or "\n" in endpoint
        ):
            raise ValueError(
                f"unsupported internal API endpoint for sandbox policy: {endpoint}"
            )
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        tls_line = "\n        tls: terminate" if parsed.scheme == "https" else ""
        policy_name = f"trace-replay_internal_api_{index}"
        network_policies.append(
            f"  {policy_name}:\n"
            f"    name: {policy_name}\n"
            "    endpoints:\n"
            f"      - host: {parsed.hostname}\n"
            f"        port: {port}\n"
            "        protocol: rest\n"
            "        enforcement: enforce\n"
            "        access: read-write"
            f"{tls_line}\n"
            "    binaries:\n"
            # SWE task images vary by Python minor version and installation
            # prefix. Keep backend access scoped to Python 3 interpreters and
            # the single backend endpoint declared above.
            # The OpenShell sandbox image resolves `python3` through a
            # uv-managed venv (/sandbox/.venv/bin/python3 ->
            # /sandbox/.uv/python/cpython-*/bin/python3.*). The supervisor
            # matches the resolved executable path, so without these entries
            # the in-sandbox call is refused with
            # `transparent_tcp_policy_denied` even when the Kubernetes
            # NetworkPolicy allows the destination.
            "      - path: /sandbox/.venv/bin/python3\n"
            "      - path: /sandbox/.venv/bin/python3.*\n"
            "      - path: /sandbox/.uv/python/*/bin/python3\n"
            "      - path: /sandbox/.uv/python/*/bin/python3.*\n"
            "      - path: /usr/sbin/python3\n"
            "      - path: /usr/sbin/python3.*\n"
            "      - path: /usr/bin/python3\n"
            "      - path: /usr/bin/python3.*\n"
            "      - path: /usr/bin/python3.13\n"
            "      - path: /usr/local/bin/python3\n"
            "      - path: /usr/local/bin/python3.*\n"
            "      - path: /opt/conda/bin/python3\n"
            "      - path: /opt/conda/bin/python3.*\n"
            "      - path: /opt/conda/envs/testbed/bin/python3\n"
            "      - path: /opt/conda/envs/testbed/bin/python3.*\n"
            "      - path: /opt/miniconda3/bin/python3\n"
            "      - path: /opt/miniconda3/bin/python3.*\n"
            "      - path: /opt/miniconda3/envs/testbed/bin/python3\n"
            "      - path: /opt/miniconda3/envs/testbed/bin/python3.*\n"
        )
    policy_yaml = ""
    if network_policies:
        policy_yaml = (
            "version: 1\n"
            "filesystem_policy:\n"
            "  include_workdir: true\n"
            "  read_only:\n"
            "    - /usr\n"
            "    - /lib\n"
            "    - /etc\n"
            "    - /app\n"
            "    - /opt/miniconda3\n"
            "    - /opt/miniconda\n"
            "    - /opt/conda\n"
            "    - /opt/venv\n"
            "    - /venv\n"
            "    - /var/log\n"
            "    - /proc\n"
            "    - /dev/urandom\n"
            "  read_write:\n"
            "    - /tmp\n"
            "    - /dev/null\n"
            "    - /sandbox/.openshell/workspace\n"
            "    - /sandbox/openclaw\n"
            "landlock:\n"
            "  compatibility: best_effort\n"
            "network_policies:\n" + "".join(network_policies)
        )
    policy_literal = json.dumps(policy_yaml)
    # The init container keeps the wrapper in its command script. Include the
    # effective driver configuration in the marker so a reused deployment is
    # refreshed when its node, workspace, PVC, policy, or endpoint changes.
    wrapper_fingerprint = hashlib.sha256(
        "\n".join(
            (
                driver_config_json,
                policy_yaml,
                gateway_endpoint or "",
                sandbox_image or "",
                sandbox_cpu_limit or "",
                json.dumps(swe_command_map or {}, sort_keys=True),
            )
        ).encode()
    ).hexdigest()[:12]
    # Bump the marker whenever wrapper semantics change so a reused agent
    # cannot retain an older wrapper that would duplicate an existing --from
    # flag or otherwise skip the refreshed driver configuration.
    # The published plugin currently creates the sandbox with a terminal
    # ``true`` command.  On this OpenShell gateway that command is the sandbox
    # process itself, so the pod exits immediately and later ``sandbox exec``
    # calls fail with exit 127.  Keep the experiment-owned sandbox alive until
    # the OpenClaw agent is finished; this is limited to the generated wrapper
    # and does not modify the gateway or any cluster infrastructure.
    # v27 installs a pinned OpenSSH client in the experiment pod. OpenClaw's
    # OpenShell backend uses the OpenShell SSH proxy for remote commands and
    # workspace transfers; replacing the SSH client with a `sandbox exec`
    # shim breaks that protocol.
    # v29 forces OpenShell sandbox creation to detach. The pinned CLI supports
    # `--detach`, but the OpenClaw plugin's capability check can miss it and
    # otherwise wait forever while attached to its persistent main process.
    # Keep `sleep infinity` as that process; later tool commands use separate
    # SSH execution channels.
    # v24 mounts its remote roots on writable, experiment-owned PVC storage,
    # separate from the staged SWE repository.
    # v30 makes file uploads copy into the destination directory (0.0.116
    # otherwise replaces a directory with the uploaded file, corrupting the
    # OpenClaw workspace seed).
    marker = f"# trace-replay-openshell-node-selector-v31-{wrapper_fingerprint}"
    policy_injection = ""
    if policy_yaml:
        policy_injection = (
            f"  fs.writeFileSync('/tmp/trace-replay-appworld-policy.yaml', {policy_literal});\n"
            "  injected.push('--policy', '/tmp/trace-replay-appworld-policy.yaml');\n"
        )
    wrapper = "".join(
        [
            f"\n              {marker}\n",
            "              cat > /opt/openshell/bin/openshell-pinned <<'TRACE_REPLAY_OPEN_SHELL_WRAPPER'\n",
            "#!/usr/bin/env node\n",
            "const { spawnSync } = require('child_process');\n",
            "const fs = require('fs');\n",
            "const args = process.argv.slice(2);\n",
            "const sandboxIndex = args.indexOf('sandbox');\n",
            "const workspace = process.env.OPENSHELL_FLEET_WORKSPACE || process.env.OPENSHELL_WORKSPACE || 'aharush-default';\n",
            "const safeWorkspace = workspace.replace(/[^A-Za-z0-9-]/g, '-');\n",
            "const sandboxName = safeWorkspace.startsWith('aharush-') ? safeWorkspace.slice(0, 19) : ('aharush-' + safeWorkspace).slice(0, 19);\n",
            # OpenClaw's upstream backend calls `get` before `create`, then uses
            # the same runtime name for `ssh-config`. Keep these names aligned
            # with the experiment's Kubernetes-safe shortened create name.
            "if (sandboxIndex >= 0 && ['get', 'ssh-config'].includes(args[sandboxIndex + 1]) && args[sandboxIndex + 2]) {\n",
            "  args[sandboxIndex + 2] = sandboxName;\n",
            "}\n",
            # The OpenClaw plugin may already add a driver-config-json argument
            # (for example, its own node selector).  That argument must not mask
            # the experiment's storage wiring: on this cluster the default
            # per-sandbox PVC has no provisioner, while the approved node-local
            # workspace PVC is supplied by this wrapper.  Replace any existing
            # driver config for the create call, preserving the experiment-owned
            # node/PVC/policy configuration deterministically.
            "if (sandboxIndex >= 0 && args[sandboxIndex + 1] === 'create') {\n",
            f"  const sandboxImage = {json.dumps(sandbox_image)};\n",
            f"  const sandboxCpu = {json.dumps(sandbox_cpu_limit)};\n",
            f"  const injected = ['--driver-config-json', {json.dumps(driver_config_json)}];\n",
            "  for (let i = args.length - 1; i >= 0; i--) {\n",
            "    if (args[i] === '--driver-config-json') { args.splice(i, 2); continue; }\n",
            "    if (args[i].startsWith('--driver-config-json=')) { args.splice(i, 1); }\n",
            "    if (args[i] === '--policy') { args.splice(i, 2); continue; }\n",
            "    if (args[i].startsWith('--policy=')) { args.splice(i, 1); }\n",
            "    if (args[i] === '--name') { args.splice(i, 2); continue; }\n",
            "    if (args[i].startsWith('--name=')) { args.splice(i, 1); }\n",
            "    if (args[i] === '--cpu') { args.splice(i, 2); continue; }\n",
            "    if (args[i].startsWith('--cpu=')) { args.splice(i, 1); }\n",
            "  }\n",
            "  injected.unshift('--name', sandboxName);\n",
            "  if (sandboxImage) {\n",
            "    const fromIndex = args.indexOf('--from');\n",
            "    const fromEqualsIndex = args.findIndex((arg) => arg.startsWith('--from='));\n",
            "    if (fromIndex >= 0 && fromIndex + 1 < args.length) args[fromIndex + 1] = sandboxImage;\n",
            "    else if (fromEqualsIndex >= 0) args[fromEqualsIndex] = '--from=' + sandboxImage;\n",
            "    else injected.unshift('--from', sandboxImage);\n",
            "  }\n",
            "  if (sandboxCpu) injected.push('--cpu', sandboxCpu);\n",
            policy_injection,
            "  args.splice(sandboxIndex + 2, 0, ...injected);\n",
            "}\n",
            # OpenClaw seeds its remote workspace through `sandbox upload` using
            # the plugin's full runtime name. Match it to the shortened run name.
            "if (sandboxIndex >= 0 && args[sandboxIndex + 1] === 'upload' && args[sandboxIndex + 3]) {\n",
            "  args[sandboxIndex + 3] = sandboxName;\n",
            "  // CLI 0.0.116 treats the remote argument as an exact destination path,\n",
            "  // so uploading a single file replaces an existing directory with that\n",
            "  // file.  The plugin uploads each workspace entry separately and expects\n",
            "  // copy-into-directory semantics; normalize file sources to\n",
            "  // <dest>/<basename> so workspace seeding cannot clobber the seed roots.\n",
            "  const uploadRemote = args[args.length - 1];\n",
            "  const uploadLocal = args[args.length - 2];\n",
            "  if (uploadRemote && uploadLocal && !uploadRemote.startsWith('-')) {\n",
            "    let uploadIsFile = false;\n",
            "    try { uploadIsFile = fs.lstatSync(uploadLocal).isFile(); } catch (err) { uploadIsFile = false; }\n",
            "    const uploadBase = uploadLocal.replace(/\\/+$/, '').split('/').pop();\n",
            "    const uploadRemoteBase = uploadRemote.replace(/\\/+$/, '').split('/').pop();\n",
            "    if (uploadIsFile && uploadBase && uploadBase !== uploadRemoteBase) {\n",
            "      args[args.length - 1] = uploadRemote.replace(/\\/+$/, '') + '/' + uploadBase;\n",
            "    }\n",
            "  }\n",
            "}\n",
            "if (sandboxIndex >= 0 && args[sandboxIndex + 1] === 'exec') {\n",
            "  for (let i = args.length - 1; i >= 0; i--) {\n",
            "    if (args[i] === '--name' && i + 1 < args.length) { args[i + 1] = sandboxName; continue; }\n",
            "    if (args[i].startsWith('--name=')) { args[i] = '--name=' + sandboxName; continue; }\n",
            "    if (args[i] === '--policy') { args.splice(i, 2); continue; }\n",
            "    if (args[i].startsWith('--policy=')) { args.splice(i, 1); }\n",
            "  }\n",
            "}\n",
            "if (sandboxIndex >= 0 && args[sandboxIndex + 1] === 'create') {\n",
            "  let separator = args.indexOf('--', sandboxIndex + 2);\n",
            "  if (separator >= 0 && args[separator + 1] === 'true') {\n",
            "    args.splice(separator + 1, 1, 'sh', '-c', 'exec sleep 2147483647');\n",
            "    separator = args.indexOf('--', sandboxIndex + 2);\n",
            "  }\n",
            "  if (!args.includes('--detach')) {\n",
            "    if (separator >= 0) args.splice(separator, 0, '--detach');\n",
            "    else args.push('--detach');\n",
            "  }\n",
            "}\n",
            "const result = spawnSync('/opt/openshell/bin/openshell', args, { stdio: 'inherit' });\n",
            "process.exit(result.status ?? 1);\n",
            "TRACE_REPLAY_OPEN_SHELL_WRAPPER\n",
            "              chmod 755 /opt/openshell/bin/openshell-pinned\n",
        ]
    )
    # Install the real SSH client into the existing per-pod OpenShell CLI
    # volume. The base OpenClaw image is Debian 12 and already contains the
    # client's shared-library dependencies. Pin and verify the Debian package
    # so this run does not rely on a mutable image push or an unpinned package.
    ssh_install_script = """set -eu
tmp=/tmp/aharush-openssh-client
mkdir -p "$tmp/lists/partial" "$tmp/cache/archives/partial" "$tmp/pkg" "$tmp/root"
timeout 120s apt-get -o Dir::State::lists="$tmp/lists" -o Dir::Cache="$tmp/cache" -o APT::Get::List-Cleanup=0 update
(cd "$tmp/pkg" && timeout 60s apt-get -o Dir::State::lists="$tmp/lists" -o Dir::Cache="$tmp/cache" download 'openssh-client=1:9.2p1-2+deb12u10')
pkg=$(find "$tmp/pkg" -maxdepth 1 -type f -name 'openssh-client_*.deb' -print -quit)
test -n "$pkg"
printf '%s  %s\\n' '42c250b8b9110382488c53c066a960bc564ddac2cb9e449f47b6cdbb5fc1cb60' "$pkg" | sha256sum -c -
dpkg-deb -x "$pkg" "$tmp/root"
cp "$tmp/root/usr/bin/ssh" /opt/openshell/bin/ssh
chmod 755 /opt/openshell/bin/ssh
/opt/openshell/bin/ssh -V
"""
    if swe_command_map:
        adapter_path = Path(__file__).with_name("swe_ssh_adapter.cjs")
        adapter_source = adapter_path.read_text(encoding="utf-8")
        encoded_map = base64.b64encode(
            json.dumps(swe_command_map, sort_keys=True).encode()
        ).decode()
        adapter_source = adapter_source.replace("__SWE_MAP_BASE64__", encoded_map)
        ssh_install_script += (
            "mv /opt/openshell/bin/ssh /opt/openshell/bin/ssh-real\n"
            "cat > /opt/openshell/bin/ssh <<'TRACE_REPLAY_SWE_SSH_ADAPTER'\n"
            + adapter_source
            + "\nTRACE_REPLAY_SWE_SSH_ADAPTER\n"
            "chmod 755 /opt/openshell/bin/ssh\n"
        )
    init_containers = deployment["spec"]["template"]["spec"].setdefault(
        "initContainers", []
    )
    init_containers[:] = [
        item
        for item in init_containers
        if item.get("name")
        not in {
            "install-ssh-wrapper",
            "install-openssh-client",
            "aharush-install-openssh-client",
        }
    ]
    init_containers.append(
        {
            "name": "aharush-install-openssh-client",
            "image": EXPERIMENT_OPENCLAW_IMAGE,
            "command": ["/bin/sh", "-lc"],
            "args": [ssh_install_script],
            "volumeMounts": [{"mountPath": "/opt/openshell", "name": "openshell-cli"}],
        }
    )
    startup_changed = False
    if gateway_endpoint:
        endpoint_literal = json.dumps(gateway_endpoint)
        endpoint_cert_key = re.sub(r"[^A-Za-z0-9._-]", "_", gateway_endpoint)
        endpoint_cert_dir = (
            f"/home/node/.config/openshell/gateways/{endpoint_cert_key}/mtls"
        )
        alias_cert_dir = (
            f"/home/node/.config/openshell/gateways/{OPENSHELL_GATEWAY_NAME}/mtls"
        )
        endpoint_cert_marker = (
            "# trace-replay-openshell-endpoint-certs-v2-"
            + hashlib.sha256(gateway_endpoint.encode("utf-8")).hexdigest()[:10]
        )
    for container in deployment["spec"]["template"]["spec"].get("containers", []):
        if container.get("name") == "gateway":
            env = container.setdefault("env", [])
            path_env = next((item for item in env if item.get("name") == "PATH"), None)
            if path_env is None:
                env.append(
                    {
                        "name": "PATH",
                        "value": "/opt/openshell/bin:/usr/local/bin:/usr/bin:/bin",
                    }
                )
            else:
                path_env["value"] = "/opt/openshell/bin:/usr/local/bin:/usr/bin:/bin"
            workspace_env = next(
                (item for item in env if item.get("name") == "OPENSHELL_WORKSPACE"),
                None,
            )
            if workspace_env is None:
                env.append({"name": "OPENSHELL_WORKSPACE", "value": workspace})
            else:
                workspace_env["value"] = workspace
            if gateway_endpoint:
                gateway_endpoint_env = next(
                    (
                        item
                        for item in env
                        if item.get("name") == "OPENSHELL_GATEWAY_ENDPOINT"
                    ),
                    None,
                )
                if gateway_endpoint_env is None:
                    env.append(
                        {
                            "name": "OPENSHELL_GATEWAY_ENDPOINT",
                            "value": gateway_endpoint,
                        }
                    )
                else:
                    gateway_endpoint_env["value"] = gateway_endpoint
        if container.get("name") != "gateway":
            continue
        for index, arg in enumerate(container.get("args", [])):
            if not isinstance(arg, str) or "openshell" not in arg:
                continue
            updated_arg = arg
            if gateway_endpoint and "EP=" in updated_arg:
                updated_arg = re.sub(
                    r'EP=\\?"[^"\n]+\\?"',
                    f"EP={endpoint_literal}",
                    updated_arg,
                    count=1,
                )
            if (
                gateway_endpoint
                and endpoint_cert_marker not in updated_arg
                and "trace-replay-openshell-mtls-endpoint-dir-v1" not in updated_arg
            ):
                mtls_setup = "\n".join(
                    [
                        endpoint_cert_marker,
                        f'mkdir -p "{endpoint_cert_dir}"',
                        f'mkdir -p "{alias_cert_dir}"',
                        f'cp /openshell-mtls/ca.crt /openshell-mtls/tls.crt /openshell-mtls/tls.key "{endpoint_cert_dir}/"',
                        f'cp /openshell-mtls/ca.crt /openshell-mtls/tls.crt /openshell-mtls/tls.key "{alias_cert_dir}/"',
                        f'chmod 600 "{endpoint_cert_dir}/tls.key"',
                        f'chmod 600 "{alias_cert_dir}/tls.key"',
                    ]
                )
                # Older OpenShell-enabled agent deployments materialize a
                # gateway alias directly instead of assigning `EP` and using
                # `gateway add`.  Put a second copy of the mTLS bundle at the
                # CLI's URL-derived location in either form.
                ep_pattern = re.compile(r'(?m)^([ \t]*EP=\\?"[^"\n]+\\?"[ \t]*)$')
                if ep_pattern.search(updated_arg):
                    updated_arg, injected = ep_pattern.subn(
                        lambda match: match.group(1) + "\n" + mtls_setup,
                        updated_arg,
                        count=1,
                    )
                else:
                    alias_chmod = re.compile(
                        r"(?m)^([ \t]*chmod 600 /home/node/\.config/openshell/gateways/openshell/mtls/tls\.key[ \t]*)$"
                    )
                    updated_arg, injected = alias_chmod.subn(
                        lambda match: match.group(1) + "\n" + mtls_setup,
                        updated_arg,
                        count=1,
                    )
                if not injected:
                    raise RuntimeError(
                        "OpenClaw gateway command has neither an endpoint assignment nor the expected mTLS alias setup"
                    )
            if gateway_endpoint:
                # The plugin addresses the gateway by name (`--gateway <name>`),
                # and the CLI only builds an SSH proxy for a registered name.
                # Register this run's endpoint before OpenClaw starts so
                # workspace seeding (SSH upload) can resolve the gateway.
                registry_marker = (
                    "# trace-replay-openshell-gateway-registry-v1-"
                    + OPENSHELL_GATEWAY_NAME
                )
                if registry_marker not in updated_arg:
                    registry_line = " ".join(
                        [
                            "timeout 30s /opt/openshell/bin/openshell gateway add",
                            "--local",
                            f"--name {shlex.quote(OPENSHELL_GATEWAY_NAME)}",
                            shlex.quote(gateway_endpoint),
                            ">/dev/null 2>&1",
                            "|| /opt/openshell/bin/openshell gateway list 2>/dev/null",
                            f"| grep -q {shlex.quote(OPENSHELL_GATEWAY_NAME)}",
                        ]
                    )
                    updated_arg, injected_registry = re.subn(
                        r"(?m)^(exec )",
                        registry_marker + "\n" + registry_line + "\n" + r"\1",
                        updated_arg,
                        count=1,
                    )
                    if not injected_registry:
                        raise RuntimeError(
                            "OpenClaw gateway command has no exec line for gateway registration"
                        )
            startup_changed = startup_changed or updated_arg != arg
            container["args"][index] = updated_arg
            break
    for init_container in deployment["spec"]["template"]["spec"].get(
        "initContainers", []
    ):
        if init_container.get("name") != "init-config":
            continue
        args = init_container.setdefault("args", [""])
        if marker not in args[0]:
            args[0] += wrapper
            # The wrapper marker includes a fingerprint of the effective
            # driver config (including the shared PVC and subPath).  Report
            # this change so callers that reuse an existing Deployment roll
            # its pod before sending sandbox-create requests; otherwise the
            # Deployment template is updated but the old pod keeps using its
            # previous wrapper.
            startup_changed = True
        return startup_changed
    raise RuntimeError("base OpenClaw deployment has no init-config container")


def ensure_driver_config(cluster: Cluster) -> None:
    """Publish the driver and its local package as one owned ConfigMap.

    Trace jobs use the stock Python image.  Mounting only ``driver.py`` is not
    sufficient because the driver imports the benchmark package from ``src``.
    A small source archive keeps the job reproducible without requiring an
    image build or a registry push.
    """
    name = f"{RESOURCE_PREFIX}-experiment-driver"
    bundle = io.BytesIO()
    package_root = ROOT / "src/trace_replay_sim"
    with tarfile.open(fileobj=bundle, mode="w:gz") as archive:
        archive.add(ROOT / "src/trace_replay_sim/driver.py", arcname="driver.py")
        for source in sorted(package_root.rglob("*.py")):
            archive.add(
                source,
                arcname=str(
                    Path("trace_replay_sim") / source.relative_to(package_root)
                ),
            )
    cluster.upsert_owned_configmap(
        cluster.namespace,
        {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": {
                "name": name,
                "labels": {"managed-by": "trace-replay-experiment"},
            },
            "binaryData": {
                "source_bundle.tgz": base64.b64encode(bundle.getvalue()).decode("ascii")
            },
        },
    )


def ensure_mock_llm_config(
    cluster: Cluster,
    replay_corpus: Path,
    system_ns: str,
    run_token: str,
    agents: int,
    topology: str,
) -> MockPlan:
    """Create the run-owned mock corpus and the mocks for one topology."""
    # Keep the run hash in the resource name.  Truncating only the beginning
    # caused different runs with the same long output prefix to overwrite one
    # another's mock corpus and Deployment.
    token = safe_name(run_token).strip("-")
    token = f"{token[:14].rstrip('-')}-{token[-8:]}" if len(token) > 8 else token
    prefix = f"{RESOURCE_PREFIX}-{token}"
    source_name = f"{prefix}-mock-src"
    replay_payload = json.loads(replay_corpus.read_text(encoding="utf-8"))
    sessions = replay_payload.get("sessions") or []
    if not sessions:
        raise RuntimeError("owned mock corpus is empty")
    replay_code = (ROOT / "src/trace_replay_sim/mock_llm.py").read_text(
        encoding="utf-8"
    )
    corpus_revision = hashlib.sha256(
        (
            replay_code
            + "\n"
            + json.dumps(replay_payload, sort_keys=True, separators=(",", ":"))
        ).encode("utf-8")
    ).hexdigest()
    cluster.upsert_owned_configmap(
        system_ns,
        {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": {
                "name": source_name,
                "labels": {"managed-by": "trace-replay-experiment", "run": prefix},
            },
            "data": {"mock_llm.py": replay_code},
        },
    )
    corpus_volumes: list[dict[str, Any]] = []
    corpus_mounts: list[dict[str, Any]] = []
    corpus_parts = pack_mock_corpus_sessions(sessions)
    log_phase(
        f"mock corpus: packing {len(sessions)} sessions into "
        f"{len(corpus_parts)} ConfigMaps (limit={MOCK_CORPUS_CHUNK_BYTES} bytes)"
    )
    for index, part_sessions in enumerate(corpus_parts):
        part_name = f"{prefix}-corpus-{index:03d}"
        cluster.upsert_owned_configmap(
            system_ns,
            {
                "apiVersion": "v1",
                "kind": "ConfigMap",
                "metadata": {
                    "name": part_name,
                    "labels": {"managed-by": "trace-replay-experiment", "run": prefix},
                },
                "data": {
                    "replay.json": json.dumps(
                        {"sessions": part_sessions}, separators=(",", ":")
                    )
                },
            },
        )
        volume_name = f"corpus-{index:03d}"
        mount_path = f"/data/corpus/part-{index:03d}"
        corpus_volumes.append({"name": volume_name, "configMap": {"name": part_name}})
        corpus_mounts.append(
            {"name": volume_name, "mountPath": mount_path, "readOnly": True}
        )
    plan = mock_plan(topology, system_ns, prefix, agents)
    for service_name in plan.services:
        base = cluster.json(["-n", system_ns, "get", "deployment", BASE_MOCK_LLM])
        for key in (
            "uid",
            "resourceVersion",
            "generation",
            "creationTimestamp",
            "managedFields",
            "status",
        ):
            base.get("metadata", {}).pop(key, None)
        base["metadata"]["name"] = service_name
        base["metadata"]["labels"] = {
            "app": service_name,
            "managed-by": "trace-replay-experiment",
            "run": prefix,
        }
        spec = base["spec"]
        spec["replicas"] = 1
        spec["selector"]["matchLabels"] = {"app": service_name}
        template = spec["template"]
        template["metadata"]["labels"] = {
            "app": service_name,
            "managed-by": "trace-replay-experiment",
            "run": prefix,
        }
        # The mock process loads its corpus once at startup.  A resumed invocation
        # can keep the same run-scoped Service name while changing the selected
        # sessions, so make the corpus revision part of the pod template.  This
        # forces a bounded rollout instead of silently serving the previous
        # in-memory corpus through an apparently healthy Service.
        template["metadata"].setdefault("annotations", {})[
            "trace-replay.openai.com/mock-corpus-sha256"
        ] = corpus_revision
        template["spec"]["nodeSelector"] = {"kubernetes.io/hostname": cluster.node}
        for container in template["spec"].get("containers", []):
            if container.get("name") != "mock-llm":
                continue
            env = {
                item["name"]: item.get("value", "")
                for item in container.get("env", [])
                if item.get("name")
            }
            env.update(
                {
                    "REPLAY_CORPUS": "/data/corpus",
                    "MOCK_EDGES_PATH": "/edges/mock_edges.jsonl",
                    "OTLP_ENDPOINT": f"http://{BASE_JAEGER}.{system_ns}.svc.cluster.local:4318",
                }
            )
            container["env"] = [
                {"name": name, "value": value} for name, value in env.items()
            ]
            mounts = [
                mount
                for mount in container.get("volumeMounts", [])
                if mount.get("name") not in {"src", "corpus", "edges"}
            ]
            mounts.extend(
                [
                    {"name": "src", "mountPath": "/app", "readOnly": True},
                    *corpus_mounts,
                    {"name": "edges", "mountPath": "/edges"},
                ]
            )
            container["volumeMounts"] = mounts
        template["spec"]["volumes"] = [
            {"name": "src", "configMap": {"name": source_name}},
            *corpus_volumes,
            {"name": "edges", "emptyDir": {}},
        ]
        cluster.apply_json_in_with_retry(system_ns, base)
        cluster.apply_json_in_with_retry(
            system_ns,
            {
                "apiVersion": "v1",
                "kind": "Service",
                "metadata": {
                    "name": service_name,
                    "labels": {
                        "app": service_name,
                        "managed-by": "trace-replay-experiment",
                        "run": prefix,
                    },
                },
                "spec": {
                    "selector": {"app": service_name},
                    "ports": [{"name": "http", "port": 8080, "targetPort": 8080}],
                },
            },
        )
        cluster.wait_rollout_in(system_ns, service_name)
    mocks = ", ".join(plan.services) if plan.services else "in-pod sidecars"
    log_phase(f"deployment: mock LLM ready ({mocks}, {len(corpus_parts)} corpus parts)")
    return plan
