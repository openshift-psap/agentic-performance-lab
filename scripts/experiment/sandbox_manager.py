"""OpenShell workspaces, sandboxes, and workload backends."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import re
import shlex
import subprocess
import tarfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import (
    APPWORLD_DATA_CHECK,
    APPWORLD_IMAGE,
    RESOURCE_PREFIX,
    ROOT,
    is_transient_api_error,
    log_phase,
)
from .kubernetes import (
    Cluster,
    _wait_for_job_success,
    _wait_for_phase,
    _wait_for_pvc_bound,
)
from .trace_manager import create_trace_job


def ensure_shared_workspace(
    cluster: Cluster,
    openshell_ns: str,
    pvc_name: str,
    storage_class: str,
    workspace_names: list[str],
    access_mode: str,
    *,
    target_node: str | None = None,
    allow_shared: bool = False,
) -> None:
    """Provision one experiment-owned shared workspace and its agent subpaths.

    OpenShell's default Kubernetes workspace is one PVC per sandbox.  The
    benchmark instead uses one PVC and a unique subPath for each agent; this
    removes per-sandbox volume attachment pressure while preserving workspace
    isolation at the agent path boundary.  ReadWriteMany is the normal mode
    for CephFS.  ReadWriteOnce is supported for a block-storage fallback when
    every sandbox is pinned to the same node.
    """
    if not workspace_names:
        raise ValueError("workspace_names must not be empty")
    if len(workspace_names) != 1 and not allow_shared:
        raise ValueError(
            "the OpenShell Kubernetes driver requires an approval label per workspace; "
            "pass --allow-shared-workspace-pvc to place several agent workspaces on one claim"
        )
    if access_mode not in {"ReadWriteMany", "ReadWriteOnce"}:
        raise ValueError(f"unsupported workspace access mode: {access_mode}")
    node = target_node or cluster.node
    attachment_labels = {"openshell.ai/sandbox-attachable": "true"}
    if len(workspace_names) == 1:
        # The workspace-scoped approval label matches one workspace only; with
        # several subpaths on one claim the gateway must not enforce it.
        attachment_labels["openshell.ai/sandbox-attachable-workspace"] = (
            workspace_names[0]
        )
    existing = cluster.run(
        ["-n", openshell_ns, "get", "pvc", pvc_name], check=False
    ).strip()
    if existing:
        pvc = json.loads(
            cluster.run(["-n", openshell_ns, "get", "pvc", pvc_name, "-o", "json"])
        )
        labels = pvc.get("metadata", {}).get("labels") or {}
        if labels.get("managed-by") != "trace-replay-experiment":
            raise RuntimeError(
                f"refusing to reuse unowned PVC {openshell_ns}/{pvc_name}"
            )
        for key, expected in attachment_labels.items():
            observed = labels.get(key)
            if observed and observed != expected:
                raise RuntimeError(
                    f"owned PVC {openshell_ns}/{pvc_name} is approved for a different workspace"
                )
        missing_labels = [
            f"{key}={value}"
            for key, value in attachment_labels.items()
            if not labels.get(key)
        ]
        if missing_labels:
            cluster.run(["-n", openshell_ns, "label", "pvc", pvc_name, *missing_labels])
        spec = pvc.get("spec", {})
        if access_mode not in (spec.get("accessModes") or []) or (
            storage_class and spec.get("storageClassName") != storage_class
        ):
            raise RuntimeError(
                f"existing PVC {openshell_ns}/{pvc_name} has incompatible access mode/storage class settings"
            )
    else:
        workspace_spec: dict[str, Any] = {
            "accessModes": [access_mode],
            "resources": {"requests": {"storage": "10Gi"}},
            "volumeMode": "Filesystem",
        }
        if storage_class:
            workspace_spec["storageClassName"] = storage_class
        cluster.apply_json_in(
            openshell_ns,
            {
                "apiVersion": "v1",
                "kind": "PersistentVolumeClaim",
                "metadata": {
                    "name": pvc_name,
                    "labels": {
                        "managed-by": "trace-replay-experiment",
                        "component": "openshell-shared-workspace",
                        **attachment_labels,
                    },
                },
                "spec": workspace_spec,
            },
        )
    # Keep the preparation Job scoped to this run's workspace prefix.  Delete
    # a stale copy first so a retry can recreate it with the current subpaths.
    prep_name = f"{RESOURCE_PREFIX}-{workspace_names[0][:39]}-workspace-prep"
    cluster.delete_owned(openshell_ns, "job", prep_name)
    names_literal = ",".join(repr(name) for name in workspace_names)
    prep = {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": prep_name,
            "labels": {
                "managed-by": "trace-replay-experiment",
                "component": "openshell-workspace-prep",
            },
        },
        "spec": {
            "backoffLimit": 0,
            "activeDeadlineSeconds": 900,
            "ttlSecondsAfterFinished": 86400,
            "template": {
                "metadata": {
                    "labels": {
                        "managed-by": "trace-replay-experiment",
                        "component": "openshell-workspace-prep",
                    }
                },
                "spec": {
                    "restartPolicy": "Never",
                    "securityContext": {"fsGroupChangePolicy": "OnRootMismatch"},
                    "nodeSelector": {"kubernetes.io/hostname": node},
                    "containers": [
                        {
                            "name": "prep",
                            "image": "python:3.12-slim",
                            "command": ["python", "-c"],
                            "args": [
                                "import os\n"
                                "def ensure(path):\n"
                                "    if os.path.lexists(path) and not os.path.isdir(path):\n"
                                "        os.unlink(path)\n"
                                "    os.makedirs(path, exist_ok=True)\n"
                                "    os.chmod(path, 0o2777)\n"
                                f"names=({names_literal},)\n"
                                "for name in names:\n"
                                "    ensure('/mnt/workspaces/' + name)\n"
                                "    ensure('/mnt/workspaces/' + name + '-oclaw')\n"
                                "    ensure('/mnt/workspaces/' + name + '-oclaw/workspace')\n"
                                "    ensure('/mnt/workspaces/' + name + '-oclaw/agent')\n"
                            ],
                            "volumeMounts": [
                                {
                                    "name": "shared-workspace",
                                    "mountPath": "/mnt/workspaces",
                                }
                            ],
                        }
                    ],
                    "volumes": [
                        {
                            "name": "shared-workspace",
                            "persistentVolumeClaim": {"claimName": pvc_name},
                        }
                    ],
                },
            },
        },
    }
    cluster.apply_json_in(openshell_ns, prep)
    # Block StorageClasses commonly use WaitForFirstConsumer. Applying the
    # node-pinned prep Job before waiting for Bound lets the scheduler select
    # the requested target node and triggers provisioning without a
    # PVC/consumer deadlock.
    cluster.run(
        [
            "-n",
            openshell_ns,
            "wait",
            "--for=jsonpath={.status.phase}=Bound",
            f"pvc/{pvc_name}",
            "--timeout=300s",
        ]
    )
    # A retry can follow cleanup of many failed sandboxes.  Their volume
    # detachments may delay the node-pinned prep pod beyond five minutes even
    # though the PVC and Job are healthy; keep this bounded but give the
    # cluster enough time to settle.
    cluster.run(
        [
            "-n",
            openshell_ns,
            "wait",
            "--for=condition=complete",
            f"job/{prep_name}",
            "--timeout=600s",
        ]
    )
    cluster.delete_owned(openshell_ns, "job", prep_name)
    log_phase(
        f"storage: shared {access_mode} workspace ready ({pvc_name}, {len(workspace_names)} subpaths)"
    )


def _swe_task_runtime_preflight_code(task: dict[str, Any]) -> str:
    """Select a working Python from this task image before measured replay.

    The source checkout is pinned separately from the image, so require the
    selected interpreter to resolve the project from that checkout rather
    than silently importing the image's original /testbed tree.
    """
    identity = json.dumps(
        {
            "base_commit": str(task.get("base_commit") or ""),
            "image": str(task.get("image") or ""),
            "module": str(task.get("repo") or "").rsplit("/", 1)[-1].replace("-", "_"),
        }
    )
    return (
        "import importlib.util,json,os,shutil,subprocess,sys\n"
        f"task={identity}\n"
        "root='/sandbox/.openshell/workspace'\n"
        "head=subprocess.run(['git','-c','safe.directory='+root,'-C',root,'rev-parse','HEAD'],capture_output=True,text=True,check=False)\n"
        "if head.returncode or head.stdout.strip()!=task['base_commit']: raise RuntimeError('SWE runtime preflight: checkout does not match task base commit')\n"
        "candidates=['/opt/miniconda3/envs/testbed/bin/python','/opt/conda/envs/testbed/bin/python','/opt/miniconda/envs/testbed/bin/python','/opt/venv/bin/python','/venv/bin/python','/usr/local/bin/python3','/usr/bin/python3']\n"
        "system=shutil.which('python3')\n"
        "if system: candidates.append(system)\n"
        "probe=\"import importlib.util,json,os,sys; s=importlib.util.find_spec(sys.argv[1]); print(json.dumps({'origin':s.origin if s else None,'locations':list(s.submodule_search_locations or []) if s else [],'version':sys.version.split()[0]}))\"\n"
        "env=os.environ.copy();env['PYTHONPATH']=':'.join((root,os.path.join(root,'src'),os.path.join(root,'lib')))\n"
        "errors=[];chosen=None\n"
        "for python in dict.fromkeys(candidates):\n"
        " if not os.path.isfile(python) or not os.access(python,os.X_OK): continue\n"
        " try: result=subprocess.run([python,'-c',probe,task['module']],cwd=root,env=env,capture_output=True,text=True,timeout=30,check=False)\n"
        " except (OSError,subprocess.TimeoutExpired) as exc: errors.append(python+': '+str(exc)); continue\n"
        " if result.returncode: errors.append(python+': '+result.stderr.strip()[-400:]); continue\n"
        " try: details=json.loads(result.stdout.strip())\n"
        " except ValueError: errors.append(python+': invalid probe output'); continue\n"
        " paths=[details.get('origin') or '']+details.get('locations',[])\n"
        " if not any(path and os.path.realpath(path).startswith(os.path.realpath(root)+os.sep) for path in paths):\n"
        "  errors.append(python+': project resolves outside pinned checkout'); continue\n"
        " chosen={'python':python,'python_version':details['version'],'module':task['module'],'module_origin':details.get('origin'),'base_commit':task['base_commit'],'image':task['image']}\n"
        " break\n"
        "if chosen is None: raise RuntimeError('SWE task image has no Python that resolves the pinned project: '+'; '.join(errors))\n"
        "shim='/tmp/trace-replay-swe-python-bin'\n"
        "os.makedirs(shim,exist_ok=True)\n"
        "os.chmod(shim,0o755)\n"
        "for name in ('python','python3'):\n"
        " link=os.path.join(shim,name)\n"
        " if os.path.lexists(link): os.unlink(link)\n"
        " os.symlink(chosen['python'],link)\n"
        "chosen['path_prefix']=shim+':'+os.path.dirname(chosen['python'])\n"
        "path='/tmp/trace-replay-swe-runtime.json'\n"
        "with open(path,'w',encoding='utf-8') as handle: json.dump(chosen,handle)\n"
        "os.chmod(path,0o644)\n"
        "print(json.dumps(chosen))\n"
    )


def stage_swe_repositories(
    cluster: Cluster,
    openshell_ns: str,
    run_hash: str,
    tasks_by_agent: dict[int, dict[str, Any]],
    workspace_names: list[str],
    agent_nodes: list[str],
    workspace_pvcs: dict[str, str],
    staging_node: str | None = None,
    checkpoint_dir: Path | None = None,
    scripts_by_agent: dict[int, Path] | None = None,
    task_image_runtime: bool = False,
) -> None:
    """Copy real SWE task snapshots into the agent's shared workspace.

    The exact trajectory base commit is fetched and staged separately from
    the image. In task-image mode, compiled extensions and generated version
    modules from the image are copied where the package directories exist;
    the runtime preflight then verifies that Python resolves the pinned
    checkout. This is replay setup, not timed tool work.
    """
    if not tasks_by_agent:
        return

    def _stage_one_once(agent_index: int, task: dict[str, Any]) -> None:
        image = str(task.get("image") or "")
        repo_url = str(task.get("repo_url") or "")
        commit = str(task.get("base_commit") or "")
        if not repo_url.startswith("https://github.com/"):
            raise RuntimeError(
                f"SWE task for agent {agent_index + 1} has no approved public GitHub repository URL"
            )
        workspace = workspace_names[agent_index]
        node = agent_nodes[agent_index]
        checkpoint = (
            (checkpoint_dir / "swe-staging" / f"{agent_index + 1:03d}.json")
            if checkpoint_dir
            else None
        )
        if (
            checkpoint
            and not scripts_by_agent
            and not task_image_runtime
            and _staged_repository_is_valid(
                cluster,
                openshell_ns,
                workspace,
                workspace_pvcs[node],
                commit,
                checkpoint,
            )
        ):
            log_phase(
                f"SWE repository staging already verified; reusing agent {agent_index + 1}"
            )
            return
        job = f"{RESOURCE_PREFIX}-swe-stage-{run_hash}-{agent_index + 1:03d}"
        extension_copy = (
            'python3 -c \'import os,shutil;src="/testbed";dst="/stage/repo";'
            "files=[os.path.join(root,name) for root,_,names in os.walk(src) for name in names "
            'if name.endswith(".so") or name=="_version.py"];'
            "[shutil.copy2(path,os.path.join(dst,os.path.relpath(path,src))) for path in files "
            "if os.path.isdir(os.path.dirname(os.path.join(dst,os.path.relpath(path,src)))) "
            'and (path.endswith(".so") or not os.path.exists(os.path.join(dst,os.path.relpath(path,src))))]\'; '
            if task_image_runtime
            else ""
        )
        command = (
            "set -eux; "
            "export GIT_CONFIG_GLOBAL=/tmp/trace-replay-gitconfig; "
            "git config --global --add safe.directory /testbed; "
            # Keep repository staging inside the short-lived Job's own
            # emptyDir.  The OpenShell workspace PVC carries a sandbox-
            # specific SELinux label, so a separate staging pod must not
            # write to it directly. Fetch the exact trajectory base commit
            # from its public repository, verify it, then stream that tree
            # into the already-running sandbox below. The task image tag is
            # mutable and may contain a different checkout.
            'stage_root=/stage/repo; rm -rf "$stage_root"; mkdir -p "$stage_root"; '
            'git config --global --add safe.directory "$stage_root"; '
            'git -C "$stage_root" init; '
            f'git -C "$stage_root" remote add origin {shlex.quote(repo_url)}; '
            f'git -C "$stage_root" fetch --no-tags --depth=1 origin {shlex.quote(commit)}; '
            'git -C "$stage_root" checkout --detach FETCH_HEAD; '
            f'test "$(git -C "$stage_root" rev-parse HEAD)" = "{commit}"; '
            f"{extension_copy}"
            "touch /stage/ready; sleep 1800"
        )
        job_obj = {
            "apiVersion": "batch/v1",
            "kind": "Job",
            "metadata": {
                "name": job,
                "labels": {
                    "managed-by": "trace-replay-experiment",
                    "component": "swe-repository-stage",
                },
            },
            "spec": {
                "backoffLimit": 0,
                "activeDeadlineSeconds": 1800,
                "ttlSecondsAfterFinished": 86400,
                "template": {
                    "metadata": {
                        "labels": {
                            "managed-by": "trace-replay-experiment",
                            "component": "swe-repository-stage",
                        }
                    },
                    "spec": {
                        "restartPolicy": "Never",
                        "securityContext": {"fsGroupChangePolicy": "OnRootMismatch"},
                        # The target workspace PVC is RWO, so stage on the
                        # agent's approved node rather than an auxiliary node.
                        "nodeSelector": {"kubernetes.io/hostname": node},
                        "containers": [
                            {
                                "name": "stage",
                                "image": image,
                                "command": ["/bin/sh", "-lc"],
                                "args": [command],
                                # Let OpenShift's restricted SCC assign the project UID.
                                # Staging only reads /testbed and writes to the Job's
                                # emptyDir, so it does not need root privileges.
                                "volumeMounts": [
                                    {"name": "stage-data", "mountPath": "/stage"},
                                ],
                            }
                        ],
                        "volumes": [
                            {"name": "stage-data", "emptyDir": {}},
                        ],
                    },
                },
            },
        }
        cluster.delete_owned(openshell_ns, "job", job)
        # A delete request is asynchronous.  Wait for the old run-owned Job
        # to disappear before recreating the same name; otherwise Kubernetes
        # can leave the controller observing the stale pod/spec.
        delete_deadline = time.time() + 120
        while time.time() < delete_deadline:
            if not cluster.run(
                ["-n", openshell_ns, "get", "job", job], check=False
            ).strip():
                break
            time.sleep(1)
        else:
            raise RuntimeError(f"timed out deleting previous SWE staging Job {job}")
        cluster.apply_json_in(openshell_ns, job_obj)
        try:
            deadline = time.time() + 1800
            stage_pod = ""
            while time.time() < deadline:
                pod_items = (
                    cluster.json(
                        ["-n", openshell_ns, "get", "pods", "-l", f"job-name={job}"]
                    ).get("items")
                    or []
                )
                running = [
                    pod
                    for pod in pod_items
                    if (pod.get("status") or {}).get("phase") == "Running"
                ]
                if running:
                    stage_pod = str(running[0].get("metadata", {}).get("name") or "")
                    ready = cluster.run(
                        [
                            "-n",
                            openshell_ns,
                            "exec",
                            stage_pod,
                            "-c",
                            "stage",
                            "--",
                            "sh",
                            "-lc",
                            "test -f /stage/ready && echo READY",
                        ],
                        check=False,
                    ).strip()
                    if ready == "READY":
                        break
                status = (
                    cluster.json(["-n", openshell_ns, "get", "job", job]).get("status")
                    or {}
                )
                if int(status.get("failed") or 0) > 0 or any(
                    condition.get("type") == "Failed"
                    and condition.get("status") == "True"
                    for condition in status.get("conditions") or []
                ):
                    logs = cluster.run(
                        ["-n", openshell_ns, "logs", f"job/{job}", "-c", "stage"],
                        check=False,
                    ).strip()
                    raise RuntimeError(
                        f"SWE repository staging Job failed for agent {agent_index + 1}:\n{logs[-8000:]}"
                    )
                time.sleep(5)
            else:
                raise RuntimeError(
                    f"timed out waiting for SWE repository staging Job {job}"
                )
            sandbox = None
            sandbox_deadline = time.time() + 300
            while time.time() < sandbox_deadline and sandbox is None:
                sandbox = _sandbox_for_workspace(
                    cluster,
                    openshell_ns,
                    workspace_pvcs[node],
                    workspace,
                )
                if sandbox is None:
                    time.sleep(3)
            if sandbox is None:
                raise RuntimeError(
                    f"timed out locating OpenShell sandbox for agent {agent_index + 1}"
                )
            sandbox_pod, sandbox_container = sandbox
            source_code = (
                "import sys,tarfile; t=tarfile.open(fileobj=sys.stdout.buffer, mode='w|gz'); "
                "t.add('/stage/repo', arcname='.'); t.close()"
            )
            target_code = (
                "import os,shutil,subprocess,sys,tarfile; root='/sandbox/.openshell/workspace'; "
                "os.makedirs(root,exist_ok=True); "
                "[shutil.rmtree(os.path.join(root,n)) if os.path.isdir(os.path.join(root,n)) and not os.path.islink(os.path.join(root,n)) else os.unlink(os.path.join(root,n)) for n in os.listdir(root)]; "
                "tarfile.open(fileobj=sys.stdin.buffer, mode='r|gz').extractall(root); "
                # The extraction runs as root inside the sandbox pod, but the
                # agent executes tools as the sandbox user.  Give the staged
                # tree explicit write access instead of relying on an
                # image-specific uid, otherwise every edit fails with EACCES.
                "subprocess.run(['chmod','-R','a+rwX',root],check=True)"
            )
            source = subprocess.Popen(
                [
                    "oc",
                    "-n",
                    openshell_ns,
                    "exec",
                    "-i",
                    stage_pod,
                    "-c",
                    "stage",
                    "--",
                    "python3",
                    "-c",
                    source_code,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=cluster.env,
            )
            assert source.stdout is not None
            target = subprocess.Popen(
                [
                    "oc",
                    "-n",
                    openshell_ns,
                    "exec",
                    "-i",
                    sandbox_pod,
                    "-c",
                    sandbox_container,
                    "--",
                    "python3",
                    "-c",
                    target_code,
                ],
                stdin=source.stdout,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=cluster.env,
            )
            source.stdout.close()
            target_out, target_err = target.communicate(timeout=600)
            source_err = source.stderr.read() if source.stderr is not None else b""
            source_rc = source.wait(timeout=30)
            if target.returncode != 0 or source_rc != 0:
                raise RuntimeError(
                    f"sandbox repository transfer failed: source_rc={source_rc} target_rc={target.returncode} "
                    f"source={source_err.decode(errors='replace')[-2000:]} target={target_err.decode(errors='replace')[-2000:]}"
                )
            observed = cluster.run(
                [
                    "-n",
                    openshell_ns,
                    "exec",
                    sandbox_pod,
                    "-c",
                    sandbox_container,
                    "--",
                    "sh",
                    "-lc",
                    "git -c safe.directory=/sandbox/.openshell/workspace -C /sandbox/.openshell/workspace rev-parse HEAD",
                ],
                check=False,
            ).strip()
            if observed != commit:
                raise RuntimeError(
                    f"SWE sandbox repository mismatch for agent {agent_index + 1}: expected {commit}, observed {observed or '<empty>'}"
                )
            runtime: dict[str, Any] | None = None
            if task_image_runtime:
                runtime_code = _swe_task_runtime_preflight_code(task)
                runtime_result = cluster.run(
                    [
                        "-n",
                        openshell_ns,
                        "exec",
                        sandbox_pod,
                        "-c",
                        sandbox_container,
                        "--",
                        "python3",
                        "-c",
                        runtime_code,
                    ],
                ).strip()
                try:
                    runtime = json.loads(runtime_result)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(
                        f"SWE task-image runtime preflight failed for agent {agent_index + 1}: {runtime_result[-2000:]}"
                    ) from exc
                if (
                    runtime.get("base_commit") != commit
                    or runtime.get("image") != image
                ):
                    raise RuntimeError(
                        f"SWE task-image runtime identity mismatch for agent {agent_index + 1}"
                    )
                log_phase(
                    f"SWE agent {agent_index + 1}: task Python {runtime['python_version']} "
                    f"resolves {runtime['module']} from pinned checkout"
                )
            script_dir = (scripts_by_agent or {}).get(agent_index)
            if script_dir is not None:
                scripts = sorted(
                    script_dir.glob("[0-9][0-9][0-9][0-9]-[0-9][0-9][0-9].py")
                )
                if not scripts:
                    raise RuntimeError(
                        f"SWE replay scripts missing for agent {agent_index + 1}: {script_dir}"
                    )
                bundle = io.BytesIO()
                with tarfile.open(fileobj=bundle, mode="w:gz") as archive:
                    for script in scripts:
                        archive.add(script, arcname=script.name)
                install_code = (
                    "import os,sys,tarfile;root='/tmp/trace-replay-tool-calls';"
                    "os.makedirs(root,exist_ok=True);"
                    "archive=tarfile.open(fileobj=sys.stdin.buffer,mode='r|gz');"
                    "[(open(os.path.join(root,item.name),'wb').write(archive.extractfile(item).read()),"
                    "os.chmod(os.path.join(root,item.name),0o644)) for item in archive if item.isfile() "
                    "and item.name.endswith('.py') and '/' not in item.name]"
                )
                install = subprocess.run(
                    [
                        "oc",
                        "-n",
                        openshell_ns,
                        "exec",
                        "-i",
                        sandbox_pod,
                        "-c",
                        sandbox_container,
                        "--",
                        "python3",
                        "-c",
                        install_code,
                    ],
                    input=bundle.getvalue(),
                    capture_output=True,
                    env=cluster.env,
                    timeout=120,
                    check=False,
                )
                if install.returncode != 0:
                    raise RuntimeError(
                        f"SWE replay script installation failed for agent {agent_index + 1}: "
                        + install.stderr.decode(errors="replace")[-1000:]
                    )
            if checkpoint:
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                temporary = checkpoint.with_suffix(".tmp")
                temporary.write_text(
                    json.dumps(
                        {
                            "schema": "trace-replay.swe-staging.v1",
                            "agent_index": agent_index + 1,
                            "workspace": workspace,
                            "node": node,
                            "base_commit": commit,
                            "task_image_runtime": runtime,
                            "verified_at": datetime.now(timezone.utc).isoformat(),
                        },
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                os.replace(temporary, checkpoint)
        except Exception as exc:
            logs = cluster.run(
                ["-n", openshell_ns, "logs", f"job/{job}", "-c", "stage"], check=False
            ).strip()
            raise RuntimeError(
                f"SWE repository staging failed for agent {agent_index + 1}: {exc}\n{logs[-4000:]}"
            ) from exc
        finally:
            cluster.delete_owned(openshell_ns, "job", job)

    def stage_one(agent_index: int, task: dict[str, Any]) -> None:
        """Stage one snapshot, retrying transient API stream resets.

        ``oc exec`` transports the repository tar over an API-server stream.
        A connection reset can truncate an otherwise valid tarball even when
        both pods and etcd remain healthy.  The per-agent Job and sandbox
        transfer are idempotent, so retry the complete snapshot operation a
        small bounded number of times instead of aborting the full run.
        """
        for attempt in range(3):
            try:
                _stage_one_once(agent_index, task)
                return
            except Exception as exc:
                if attempt >= 2 or not is_transient_api_error(str(exc)):
                    raise
                delay = min(2**attempt, 20)
                log_phase(
                    f"SWE repository staging transient failure for agent {agent_index + 1}; "
                    f"retrying in {delay}s ({attempt + 1}/2): {exc}"
                )
                time.sleep(delay)

    log_phase(
        f"SWE backend: staging {len(tasks_by_agent)} real repository snapshots into shared workspaces"
    )
    with ThreadPoolExecutor(max_workers=min(2, len(tasks_by_agent))) as pool:
        futures = [
            pool.submit(stage_one, agent_index, task)
            for agent_index, task in tasks_by_agent.items()
        ]
        for future in as_completed(futures):
            future.result()
    log_phase(
        f"SWE backend: staged {len(tasks_by_agent)} repository snapshots with verified base commits"
    )


def _sandbox_for_workspace(
    cluster: Cluster,
    openshell_ns: str,
    pvc: str,
    workspace: str,
) -> tuple[str, str] | None:
    """Find the OpenShell sandbox pod mounting one workspace subpath."""
    pods = (
        cluster.json(["-n", openshell_ns, "get", "pods", "-o", "json"]).get("items")
        or []
    )
    for pod in pods:
        if (pod.get("status") or {}).get("phase") != "Running":
            continue
        volumes = pod.get("spec", {}).get("volumes") or []
        matching_names = {
            str(volume.get("name"))
            for volume in volumes
            if (volume.get("persistentVolumeClaim") or {}).get("claimName") == pvc
        }
        if not matching_names:
            continue
        for container in pod.get("spec", {}).get("containers") or []:
            for mount in container.get("volumeMounts") or []:
                if (
                    mount.get("name") in matching_names
                    and mount.get("subPath") == workspace
                ):
                    return str(pod.get("metadata", {}).get("name")), str(
                        container.get("name") or "agent"
                    )
    return None


def _remove_failed_sandbox_for_workspace(
    cluster: Cluster,
    openshell_ns: str,
    workspace: str,
) -> None:
    """Remove only a failed, run-owned Sandbox so prewarm can recreate it.

    OpenShell keeps the Sandbox CR after a failed pod.  A later prewarm call
    can then report success while reusing the dead object, leaving repository
    staging with no Running pod to receive the workspace.  The workspace
    annotation and name prefix are both checked before deletion; running or
    unrelated sandboxes are never touched.
    """
    if not workspace.startswith(f"{RESOURCE_PREFIX}-"):
        return
    for item in cluster.json(["-n", openshell_ns, "get", "sandbox"]).get("items") or []:
        metadata = item.get("metadata") or {}
        name = str(metadata.get("name") or "")
        annotations = metadata.get("annotations") or {}
        if annotations.get("openshell.ai/sandbox-workspace") != workspace:
            continue
        if not name.startswith(f"{workspace}--"):
            continue
        conditions = (item.get("status") or {}).get("conditions") or []
        failed = any(
            condition.get("type") == "Finished"
            and condition.get("status") == "True"
            and condition.get("reason") == "PodFailed"
            for condition in conditions
        )
        if failed:
            # A failed Sandbox may retain a controller finalizer.  Do not make
            # the experiment wait for that finalizer; the OpenShell controller
            # will finish cleanup asynchronously while prewarm recreates the
            # run-owned slot.
            cluster.run(
                [
                    "-n",
                    openshell_ns,
                    "delete",
                    "sandbox",
                    name,
                    "--ignore-not-found",
                    "--wait=false",
                ]
            )


def _staged_repository_is_valid(
    cluster: Cluster,
    openshell_ns: str,
    workspace: str,
    pvc: str,
    expected_commit: str,
    checkpoint: Path,
) -> bool:
    """Return true only when a persisted staging checkpoint is still live.

    A checkpoint is an optimization, not authority: the exact sandbox pod is
    rediscovered and its mounted workspace HEAD is checked before reuse.  This
    prevents a stale marker from hiding a failed or recreated sandbox.
    """
    if not expected_commit or not checkpoint.exists():
        return False
    try:
        data = json.loads(checkpoint.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if data.get("workspace") != workspace or data.get("base_commit") != expected_commit:
        return False
    found = _sandbox_for_workspace(cluster, openshell_ns, pvc, workspace)
    if not found:
        return False
    pod, container = found
    observed = cluster.run(
        [
            "-n",
            openshell_ns,
            "exec",
            pod,
            "-c",
            container,
            "--",
            "sh",
            "-lc",
            "git -c safe.directory=/sandbox/.openshell/workspace -C /sandbox/.openshell/workspace rev-parse HEAD",
        ],
        check=False,
    ).strip()
    return observed == expected_commit


def _verify_swe_sandboxes(
    cluster: Cluster,
    openshell_ns: str,
    tasks_by_agent: dict[int, dict[str, Any]],
    workspace_names: list[str],
    agent_nodes: list[str],
    workspace_pvcs: dict[str, str],
    timeout: float = 300.0,
) -> None:
    """Verify staged repositories from inside the actual OpenShell sandboxes."""
    if not tasks_by_agent:
        return
    deadline = time.time() + timeout
    pending = set(tasks_by_agent)
    while pending and time.time() < deadline:
        for agent_index in list(pending):
            workspace = workspace_names[agent_index]
            pvc = workspace_pvcs[agent_nodes[agent_index]]
            found = _sandbox_for_workspace(cluster, openshell_ns, pvc, workspace)
            if not found:
                continue
            pod, container = found
            task = tasks_by_agent[agent_index]
            expected = str(task.get("base_commit") or "")
            observed = cluster.run(
                [
                    "-n",
                    openshell_ns,
                    "exec",
                    pod,
                    "-c",
                    container,
                    "--",
                    "sh",
                    "-lc",
                    "git -c safe.directory=/sandbox/.openshell/workspace -C /sandbox/.openshell/workspace rev-parse HEAD",
                ],
                check=False,
            ).strip()
            if observed != expected:
                raise RuntimeError(
                    f"SWE sandbox repository mismatch for agent {agent_index + 1}: "
                    f"expected {expected}, observed {observed or '<empty>'}"
                )
            pending.remove(agent_index)
        if pending:
            time.sleep(3)
    if pending:
        raise RuntimeError(
            "timed out locating OpenShell sandboxes after prewarm for agents "
            + ", ".join(str(index + 1) for index in sorted(pending))
        )


def prewarm_swe_sandboxes(
    cluster: Cluster,
    items_by_agent: dict[int, dict[str, Any]],
    agent_bindings: list[tuple[str, str, str]],
    out: Path,
    system_ns: str,
    run_hash: str,
    allow_cross_node: bool,
    workspace_pvcs: dict[str, str] | None = None,
    placement_node: str | None = None,
    openshell_ns: str | None = None,
) -> None:
    """Create the real agent-scoped sandboxes before SWE repository staging.

    OpenShell initializes the workspace when its sandbox is first created.  A
    synthetic exec through the same OpenClaw deployment creates that exact
    sandbox; staging then happens while it is mounted, and the measured Job
    reuses the agent-scoped sandbox.
    """
    if not items_by_agent:
        return
    sandbox_ns = openshell_ns or cluster.namespace

    def one(agent_index: int, item: dict[str, Any]) -> None:
        agent, _, node = agent_bindings[agent_index]
        workspace = agent_bindings[agent_index][1]
        if workspace_pvcs:
            _remove_failed_sandbox_for_workspace(cluster, sandbox_ns, workspace)
            swe_task = item.get("session", {}).get("swe_task") or {}
            checkpoint = out / "swe-staging" / f"{agent_index + 1:03d}.json"
            if _staged_repository_is_valid(
                cluster,
                sandbox_ns,
                workspace,
                workspace_pvcs[node],
                str(swe_task.get("base_commit") or ""),
                checkpoint,
            ):
                log_phase(
                    f"SWE sandbox staging already verified; reusing agent {agent_index + 1}"
                )
                return
        session_id = str(item["session"].get("session_id") or "")
        user_key = f"trace-replay:{session_id}:swe-prewarm-{run_hash}"
        job = cm = ""
        try:
            job, cm, _, _ = create_trace_job(
                cluster,
                item,
                agent,
                out / "traces",
                system_ns,
                f"swp{run_hash[:6]}{agent_index + 1:03d}",
                allow_cross_node=allow_cross_node,
                prewarm_requested=True,
                warm_pool_size=0,
                agent_node=node,
                placement_node=placement_node,
                prewarm_only=True,
                user_key=user_key,
            )
            _wait_for_job_success(cluster, cluster.namespace, job, timeout=900)
            if workspace_pvcs:
                deadline = time.time() + 300
                while time.time() < deadline:
                    if _sandbox_for_workspace(
                        cluster, sandbox_ns, workspace_pvcs[node], workspace
                    ):
                        break
                    time.sleep(3)
                else:
                    raise RuntimeError(
                        f"prewarm completed but no Running OpenShell sandbox was found for agent {agent_index + 1} "
                        f"workspace {workspace}"
                    )
            log_phase(
                f"SWE sandbox prewarm complete: agent {agent_index + 1} ({session_id})"
            )
        finally:
            for kind, name in (("job", job), ("configmap", cm)):
                if name:
                    cluster.delete_owned(cluster.namespace, kind, name)

    log_phase(
        f"SWE backend: prewarming {len(items_by_agent)} agent-scoped sandboxes before staging"
    )
    # Two at a time keeps sandbox admission and etcd pressure bounded while
    # avoiding a serial 25-sandbox cold start.
    with ThreadPoolExecutor(max_workers=min(2, len(items_by_agent))) as pool:
        futures = [
            pool.submit(one, index, item) for index, item in items_by_agent.items()
        ]
        for future in as_completed(futures):
            future.result()
    log_phase(
        "SWE backend: all task sandboxes are live; repository staging may proceed"
    )


def _appworld_data_pod(
    name: str, pvc_name: str, node: str, command: str
) -> dict[str, Any]:
    """Build a short-lived pod that mounts the AppWorld data volume."""
    return {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {
            "name": name,
            "labels": {
                "managed-by": "trace-replay-experiment",
                "component": "appworld-data-probe",
            },
        },
        "spec": {
            "restartPolicy": "Never",
            "securityContext": {"fsGroupChangePolicy": "OnRootMismatch"},
            "nodeSelector": {"kubernetes.io/hostname": node},
            "containers": [
                {
                    "name": "appworld-data",
                    "image": APPWORLD_IMAGE,
                    "command": ["/bin/sh", "-lc", command],
                    "volumeMounts": [
                        {"name": "appworld-data", "mountPath": "/appworld"}
                    ],
                }
            ],
            "volumes": [
                {
                    "name": "appworld-data",
                    "persistentVolumeClaim": {"claimName": pvc_name},
                }
            ],
        },
    }


def _appworld_data_present(cluster: Cluster, namespace: str, pod: str) -> bool:
    result = cluster.run(
        ["-n", namespace, "exec", pod, "--", "sh", "-lc", APPWORLD_DATA_CHECK],
        check=False,
    ).strip()
    return "PRESENT" in result


def _appworld_backend_pod(cluster: Cluster, namespace: str) -> str:
    return cluster.run(
        [
            "-n",
            namespace,
            "get",
            "pod",
            "-l",
            f"app={RESOURCE_PREFIX}-appworld-backend",
            "-o",
            "jsonpath={.items[0].metadata.name}",
        ],
        check=False,
    ).strip()


def _appworld_backend_pvc(cluster: Cluster, namespace: str, pod: str) -> str:
    spec = cluster.json(["-n", namespace, "get", "pod", pod]).get("spec", {})
    for volume in spec.get("volumes") or []:
        if volume.get("name") == "appworld-data":
            return str(
                (volume.get("persistentVolumeClaim") or {}).get("claimName") or ""
            )
    return ""


def verify_appworld_data(
    cluster: Cluster, namespace: str, node: str, pvc_name: str
) -> None:
    """Verify a customer-owned, pre-populated AppWorld PVC without staging.

    This mode deliberately does not require the controller ownership label: the
    PVC belongs to the customer.  It only reads the volume, checks that the
    expected AppWorld directories exist, and never writes to it.
    """
    if not cluster.run(["-n", namespace, "get", "pvc", pvc_name], check=False).strip():
        raise RuntimeError(
            f"AppWorld data PVC {namespace}/{pvc_name} is missing; "
            "create it or point --appworld-data-pvc at an existing claim"
        )
    # If the backend from an earlier run still holds this RWO claim, reading
    # through that pod is the only safe check: a second pod cannot mount the
    # same claim concurrently.  This also makes reruns idempotent.
    backend_pod = _appworld_backend_pod(cluster, namespace)
    backend_uses_pvc = (
        bool(backend_pod)
        and _appworld_backend_pvc(cluster, namespace, backend_pod) == pvc_name
    )
    if backend_pod and backend_uses_pvc:
        _wait_for_phase(cluster, namespace, "pod", backend_pod, "Running", timeout=600)
        if not _appworld_data_present(cluster, namespace, backend_pod):
            raise RuntimeError(
                f"AppWorld data PVC {namespace}/{pvc_name} is not pre-populated; "
                "expected /appworld/data/base_dbs/spotify.db and /appworld/data/tasks"
            )
        log_phase(
            f"appworld: verified pre-populated data PVC {namespace}/{pvc_name} via {backend_pod}"
        )
        return
    probe_name = f"{RESOURCE_PREFIX}-appworld-data-check"
    if cluster.run(["-n", namespace, "get", "pod", probe_name], check=False).strip():
        cluster.delete_owned(namespace, "pod", probe_name)
    cluster.apply_json_in(
        namespace, _appworld_data_pod(probe_name, pvc_name, node, "sleep 900")
    )
    try:
        # A WaitForFirstConsumer claim binds only once the probe pod exists.
        _wait_for_pvc_bound(cluster, namespace, pvc_name)
        _wait_for_phase(cluster, namespace, "pod", probe_name, "Running", timeout=600)
        if not _appworld_data_present(cluster, namespace, probe_name):
            raise RuntimeError(
                f"AppWorld data PVC {namespace}/{pvc_name} is not pre-populated; "
                "expected /appworld/data/base_dbs/spotify.db and /appworld/data/tasks"
            )
    finally:
        cluster.delete_owned(namespace, "pod", probe_name)
    log_phase(f"appworld: verified pre-populated data PVC {namespace}/{pvc_name}")


def rewrite_appworld_claim_name(manifest_text: str, pvc_name: str) -> str:
    """Point the AppWorld backend manifest at the configured PVC name."""
    if not re.fullmatch(r"[a-z0-9]([-a-z0-9.]*[a-z0-9])?", pvc_name or ""):
        raise RuntimeError(f"invalid AppWorld PVC name: {pvc_name!r}")
    return re.sub(
        r"(claimName:\s*)[^\s#]+",
        lambda match: f"{match.group(1)}{pvc_name}",
        manifest_text,
    )


def ensure_appworld_data(
    cluster: Cluster,
    namespace: str,
    node: str,
    pvc_name: str,
    storage_class: str,
    local_data: Path,
) -> None:
    """Stage the real AppWorld data into an owned, node-local RWO PVC.

    AppWorld is stateful and its data is not part of the public container image.
    The PVC is deliberately RWO because the benchmark backend is a single pod
    pinned to the experiment node. Existing PVCs are reused only when they carry
    the controller's ownership label.
    """
    required = [local_data / "base_dbs", local_data / "tasks"]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(
            f"AppWorld data is incomplete; missing: {', '.join(missing)}"
        )

    pvc_spec: dict[str, Any] = {
        "accessModes": ["ReadWriteOnce"],
        "resources": {"requests": {"storage": "1Gi"}},
    }
    if storage_class:
        pvc_spec["storageClassName"] = storage_class
    pvc = {
        "apiVersion": "v1",
        "kind": "PersistentVolumeClaim",
        "metadata": {
            "name": pvc_name,
            "namespace": namespace,
            "labels": {
                "managed-by": "trace-replay-experiment",
                "component": "appworld-data",
            },
        },
        "spec": pvc_spec,
    }
    existing = cluster.run(
        ["-n", namespace, "get", "pvc", pvc_name], check=False
    ).strip()
    if existing:
        current = cluster.json(["-n", namespace, "get", "pvc", pvc_name])
        labels = current.get("metadata", {}).get("labels") or {}
        if labels.get("managed-by") != "trace-replay-experiment":
            raise RuntimeError(
                f"refusing to use unowned AppWorld PVC {namespace}/{pvc_name}"
            )
    else:
        cluster.apply_json_in(namespace, pvc)
    stage_name = f"{RESOURCE_PREFIX}-appworld-data-stage"
    stage = _appworld_data_pod(
        stage_name, pvc_name, node, "mkdir -p /appworld/data && sleep 900"
    )
    cluster.apply_json_in(namespace, stage)
    try:
        # Block StorageClasses that use WaitForFirstConsumer bind only once a
        # consumer exists; the node selector pins the pod and volume to one node.
        _wait_for_pvc_bound(cluster, namespace, pvc_name)
        _wait_for_phase(cluster, namespace, "pod", stage_name, "Running", timeout=600)
        present = cluster.run(
            [
                "-n",
                namespace,
                "exec",
                stage_name,
                "--",
                "sh",
                "-lc",
                "test -f /appworld/data/base_dbs/spotify.db && printf PRESENT",
            ],
            check=False,
        ).strip()
        if "PRESENT" not in present:
            log_phase(
                f"appworld: staging real data from {local_data} into {namespace}/{pvc_name}"
            )
            cluster.run(
                [
                    "-n",
                    namespace,
                    "rsync",
                    f"{local_data}/",
                    f"pod/{stage_name}:/appworld/data/",
                ]
            )
        cluster.run(
            [
                "-n",
                namespace,
                "exec",
                stage_name,
                "--",
                "sh",
                "-lc",
                "test -f /appworld/data/base_dbs/spotify.db && test -d /appworld/data/tasks",
            ],
        )
    finally:
        cluster.delete_owned(namespace, "pod", stage_name)


def ensure_workload_backends(
    cluster: Cluster,
    backends: dict[str, Any],
    allocation: dict[str, int],
    run_id: str,
    *,
    appworld_task_map: Path | None = None,
    appworld_data_pvc: str | None = None,
) -> dict[str, Any]:
    """Deploy/check only explicitly configured, experiment-owned backends.

    A managed backend manifest must contain resources whose names begin with
    the configured ``RESOURCE_PREFIX``. Existing managed resources must carry this experiment's
    ownership label before they are updated. Source changes roll only those
    owned backend deployments. This function has no generic fixture fallback.
    """
    resolved: dict[str, Any] = {}
    for workload in allocation:
        spec = backends[workload]
        if spec.deployment_mode == "managed":
            owned_objects = [("deployment", spec.name), ("service", spec.service)]
            if spec.benchmark == "browsecompplus":
                owned_objects.append(
                    (
                        "persistentvolumeclaim",
                        spec.pvc or f"{RESOURCE_PREFIX}-browsecompplus-index",
                    )
                )
            for kind, name in owned_objects:
                raw = cluster.run(
                    ["-n", spec.namespace, "get", kind, name, "-o", "json"], check=False
                ).strip()
                if raw:
                    current = json.loads(raw)
                    labels = current.get("metadata", {}).get("labels") or {}
                    if labels.get("managed-by") != "trace-replay-experiment":
                        raise RuntimeError(
                            f"refusing to update unowned backend {kind} {spec.namespace}/{name}"
                        )
            if not spec.manifest:
                raise RuntimeError(f"managed backend {workload} has no manifest")
            manifest_path = Path(spec.manifest)
            if not manifest_path.is_absolute():
                manifest_path = ROOT / manifest_path
            if not manifest_path.exists():
                raise RuntimeError(
                    f"managed backend manifest not found for {workload}: {manifest_path}"
                )
            source_name = f"{spec.service}-src"
            if spec.benchmark == "appworld":
                if appworld_task_map is None or not appworld_task_map.exists():
                    raise RuntimeError(
                        "AppWorld requires a session-to-task mapping file; "
                        "pass --appworld-task-map"
                    )
                try:
                    mapping = json.loads(appworld_task_map.read_text(encoding="utf-8"))
                except json.JSONDecodeError as exc:
                    raise RuntimeError(
                        f"invalid AppWorld task map: {appworld_task_map}"
                    ) from exc
                if not isinstance(mapping, dict) or not mapping:
                    raise RuntimeError(
                        f"AppWorld task map is empty: {appworld_task_map}"
                    )
                source_data = {
                    "appworld_service.py": (
                        ROOT / "scripts/appworld_service.py"
                    ).read_text(encoding="utf-8")
                }
                cluster.upsert_owned_secret(
                    spec.namespace,
                    {
                        "apiVersion": "v1",
                        "kind": "Secret",
                        "metadata": {
                            "name": f"{RESOURCE_PREFIX}-appworld-task-map",
                            "labels": {
                                "managed-by": "trace-replay-experiment",
                                "component": "appworld-task-map",
                            },
                        },
                        "type": "Opaque",
                        "data": {
                            "session-task-map.json": base64.b64encode(
                                json.dumps(mapping, indent=2).encode("utf-8")
                            ).decode("ascii")
                        },
                    },
                )
            else:
                source_data = {
                    "domain_backend.py": (ROOT / "scripts/domain_backend.py").read_text(
                        encoding="utf-8"
                    )
                }
            cluster.upsert_owned_configmap(
                spec.namespace,
                {
                    "apiVersion": "v1",
                    "kind": "ConfigMap",
                    "metadata": {
                        "name": source_name,
                        "labels": {
                            "managed-by": "trace-replay-experiment",
                            "component": "workload-backend",
                            "workload": workload,
                        },
                    },
                    "data": source_data,
                },
            )
            manifest_text = manifest_path.read_text(encoding="utf-8")
            if spec.benchmark == "appworld" and appworld_data_pvc:
                # The checked-in template points at the default claim.  Keep it
                # usable as-is, but bind a customer-owned PVC when one is
                # supplied.
                manifest_text = rewrite_appworld_claim_name(
                    manifest_text, appworld_data_pvc
                )
            # Backend services are single, run-owned control-plane services.
            # Render their node selector from the requested control node
            # instead of silently applying the historical hard-coded node
            # from the checked-in template.
            manifest_text = re.sub(
                r"(kubernetes\.io/hostname:\s*)[^\s#]+",
                rf"\g<1>{cluster.node}",
                manifest_text,
            )
            documents = manifest_text.split("\n---")
            for document in documents:
                try:
                    resource = (
                        json.loads(document)
                        if document.lstrip().startswith("{")
                        else None
                    )
                except json.JSONDecodeError:
                    resource = None
                if resource:
                    name = str(resource.get("metadata", {}).get("name") or "")
                    if name and not name.startswith(f"{RESOURCE_PREFIX}-"):
                        raise RuntimeError(
                            f"managed backend manifest contains unowned resource {name}"
                        )
                else:
                    names = re.findall(
                        r"^\s+name:\s*([A-Za-z0-9][A-Za-z0-9-]*)\s*$",
                        document,
                        re.MULTILINE,
                    )
                    if names and not names[0].startswith(f"{RESOURCE_PREFIX}-"):
                        raise RuntimeError(
                            f"managed backend manifest contains unowned resource {names[0]}"
                        )
            cluster.apply_yaml_in(spec.namespace, manifest_text)
            # The source ConfigMap is mounted at pod start, so an unchanged
            # Deployment template would keep serving stale backend code. Bind
            # the source hash to the pod template to roll the backend whenever
            # the local source changes.
            source_sha = hashlib.sha256(
                "\n".join(
                    f"{name}\n{text}" for name, text in sorted(source_data.items())
                ).encode()
            ).hexdigest()
            cluster.run(
                [
                    "-n",
                    spec.namespace,
                    "patch",
                    "deployment",
                    spec.name,
                    "--type=merge",
                    "-p",
                    json.dumps(
                        {
                            "spec": {
                                "template": {
                                    "metadata": {
                                        "annotations": {
                                            "trace-replay.openai.com/backend-source-sha256": source_sha,
                                        }
                                    }
                                }
                            }
                        }
                    ),
                ]
            )
        # A real backend must expose its Service. For an existing backend this
        # is strictly read-only; for a managed one it follows the apply above.
        service = cluster.run(
            ["-n", spec.namespace, "get", "svc", spec.service], check=False
        ).strip()
        if not service:
            raise RuntimeError(
                f"backend service is missing for {workload}: {spec.namespace}/{spec.service}"
            )
        deployment = cluster.run(
            ["-n", spec.namespace, "get", "deployment", spec.name], check=False
        ).strip()
        if deployment:
            cluster.wait_rollout_in(spec.namespace, spec.name)
        resolved[workload] = spec
        log_phase(
            f"backend ready: {workload} -> {spec.base_url} ({spec.deployment_mode})"
        )
    return resolved


def _resolve_backend_service_endpoints(
    cluster: Cluster,
    backends: dict[str, Any],
    workloads: dict[str, int],
) -> dict[str, str]:
    """Resolve selected backend Services to stable in-cluster IP endpoints.

    OpenShell sandboxes use an isolated network namespace and cannot rely on
    Kubernetes DNS resolution.  Service ClusterIPs are the narrowest
    experiment-scoped endpoint that works from those sandboxes; this helper is
    read-only and never changes the Service or cluster DNS configuration.
    """
    resolved: dict[str, str] = {}
    for workload in workloads:
        spec = backends[workload]
        service = cluster.json(["-n", spec.namespace, "get", "svc", spec.service])
        cluster_ip = str(service.get("spec", {}).get("clusterIP") or "")
        if not cluster_ip or cluster_ip == "None":
            raise RuntimeError(
                f"backend service {spec.namespace}/{spec.service} has no ClusterIP; "
                "cannot safely route sandbox traffic"
            )
        resolved[workload] = f"http://{cluster_ip}:{spec.port}"
    return resolved


def _rewrite_queue_backend_endpoints(
    queue: list[dict[str, Any]],
    backends: dict[str, Any],
    endpoint_map: dict[str, str],
    swe_script_root: Path | None = None,
) -> None:
    """Replace DNS backend URLs in queued sandbox commands with Service IPs."""
    for item in queue:
        session = item.get("session") or {}
        for turn in session.get("turns") or []:
            for tool in turn.get("tool_calls") or []:
                command = tool.get("mapped_command")
                if not isinstance(command, list):
                    continue
                rewritten = [str(part) for part in command]
                for workload, new_endpoint in endpoint_map.items():
                    spec = backends[workload]
                    old = spec.base_url.rstrip("/")
                    new = new_endpoint.rstrip("/")
                    rewritten = [part.replace(old, new) for part in rewritten]
                # AppWorld commands are transported as a base64-encoded
                # Python client to stay below OpenShell's remote-command
                # limit.  Rewrite URLs inside that payload as well; a plain
                # string replacement cannot see the embedded Service DNS
                # name and would leave sandboxes using an unreachable DNS
                # endpoint.
                if (
                    len(rewritten) >= 3
                    and rewritten[0] == "/bin/sh"
                    and rewritten[1] == "-c"
                ):
                    match = re.fullmatch(
                        r"echo ([A-Za-z0-9+/=]+)\|base64 -d\|python3", rewritten[2]
                    )
                    if match:
                        try:
                            decoded = base64.b64decode(match.group(1)).decode("utf-8")
                        except (ValueError, UnicodeDecodeError):
                            decoded = ""
                        if decoded:
                            for workload, new_endpoint in endpoint_map.items():
                                spec = backends[workload]
                                old = spec.base_url.rstrip("/")
                                new = new_endpoint.rstrip("/")
                                decoded = decoded.replace(old, new)
                            rewritten[2] = (
                                "echo "
                                + base64.b64encode(decoded.encode("utf-8")).decode(
                                    "ascii"
                                )
                                + "|base64 -d|python3"
                            )
                tool["mapped_command"] = rewritten
        if (
            swe_script_root is not None
            and item.get("workload") == "software_engineering_qa"
        ):
            session_id = str(session.get("session_id") or "")
            script_dir = (
                swe_script_root / hashlib.sha256(session_id.encode()).hexdigest()[:16]
            )
            for script in script_dir.glob("*.py"):
                source = script.read_text(encoding="utf-8")
                for workload, new_endpoint in endpoint_map.items():
                    source = source.replace(
                        backends[workload].base_url.rstrip("/"),
                        new_endpoint.rstrip("/"),
                    )
                script.write_text(source, encoding="utf-8")


def _ensure_openshell_workspace(
    cluster: Cluster,
    deployment: str,
    workspace: str,
    gateway_endpoint: str,
) -> None:
    cli = "/opt/openshell/bin/openshell"
    common = [
        "-n",
        cluster.namespace,
        "exec",
        f"deployment/{deployment}",
        "-c",
        "gateway",
        "--",
        cli,
    ]
    existing = cluster.run(
        [
            *common,
            "workspace",
            "get",
            workspace,
            "--gateway-endpoint",
            gateway_endpoint,
        ],
        check=False,
    )
    if existing.strip():
        return
    if not workspace.startswith(f"{RESOURCE_PREFIX}-"):
        raise RuntimeError(
            f"refusing to create OpenShell workspace outside the owned prefix: {workspace}"
        )
    cluster.run(
        [
            *common,
            "workspace",
            "create",
            "--name",
            workspace,
            "--gateway-endpoint",
            gateway_endpoint,
        ]
    )


def prewarm_pool_sandboxes(
    cluster: Cluster,
    items_by_agent: dict[int, dict[str, Any]],
    agent_bindings: list[tuple[str, str, str]],
    out: Path,
    system_ns: str,
    run_hash: str,
    allow_cross_node: bool,
    *,
    workspace_pvcs: dict[str, str] | None = None,
    placement_node: str | None = None,
    openshell_ns: str | None = None,
) -> list[dict[str, Any]]:
    """Fill and verify agent-scoped sandbox slots before measured dispatch."""
    if not items_by_agent:
        return []
    sandbox_ns = openshell_ns or cluster.namespace
    evidence: list[dict[str, Any]] = []
    evidence_lock = threading.Lock()

    def one(agent_index: int, item: dict[str, Any]) -> None:
        agent, workspace, node = agent_bindings[agent_index]
        session_id = str(item["session"].get("session_id") or "")
        was_ready = bool(
            workspace_pvcs
            and _sandbox_for_workspace(
                cluster, sandbox_ns, workspace_pvcs[node], workspace
            )
        )
        started = time.time()
        job = cm = ""
        try:
            job, cm, _, _ = create_trace_job(
                cluster,
                item,
                agent,
                out / "pool-prewarm" / "traces",
                system_ns,
                f"pool{run_hash[:6]}{agent_index + 1:03d}",
                allow_cross_node=allow_cross_node,
                prewarm_requested=True,
                warm_pool_size=0,
                agent_node=node,
                placement_node=placement_node,
                prewarm_only=True,
                user_key=f"trace-replay:{session_id}:pool-ready-{run_hash}",
            )
            _wait_for_job_success(cluster, cluster.namespace, job, timeout=900)
            if workspace_pvcs:
                deadline = time.time() + 300
                while time.time() < deadline:
                    if _sandbox_for_workspace(
                        cluster, sandbox_ns, workspace_pvcs[node], workspace
                    ):
                        break
                    time.sleep(3)
                else:
                    raise RuntimeError(
                        f"pool prewarm completed but no Ready sandbox was found for agent {agent_index + 1}"
                    )
            with evidence_lock:
                evidence.append(
                    {
                        "agent_index": agent_index + 1,
                        "agent": agent,
                        "workspace": workspace,
                        "node": node,
                        "session_id": session_id,
                        "sandbox_ready_before_fill": was_ready,
                        "pool_fill_wall_ms": round((time.time() - started) * 1000.0, 3),
                        "status": "ready",
                    }
                )
            log_phase(
                f"sandbox pool: slot {agent_index + 1} ready ({'reused' if was_ready else 'verified'})"
            )
        finally:
            for kind, name in (("job", job), ("configmap", cm)):
                if name:
                    cluster.delete_owned(cluster.namespace, kind, name)

    log_phase(
        f"sandbox pool: filling {len(items_by_agent)} ready slots before measurement"
    )
    with ThreadPoolExecutor(max_workers=min(2, len(items_by_agent))) as pool:
        futures = [
            pool.submit(one, index, item) for index, item in items_by_agent.items()
        ]
        for future in as_completed(futures):
            future.result()
    evidence.sort(key=lambda item: int(item["agent_index"]))
    (out / "warm_pool_evidence.json").write_text(
        json.dumps(evidence, indent=2) + "\n", encoding="utf-8"
    )
    return evidence
