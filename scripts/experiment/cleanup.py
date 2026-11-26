"""Cleanup of run-scoped Kubernetes objects."""

from __future__ import annotations

from .config import LOG
from .kubernetes import Cluster


def delete_owned(cluster: Cluster, namespace: str, kind: str, name: str) -> None:
    """Delete one owned object, never letting cleanup mask the trace result."""
    try:
        cluster.delete_owned(namespace, kind, name)
    except Exception as exc:  # noqa: BLE001
        LOG.warning("owned cleanup failed for %s/%s: %s", kind, name, exc)


def cleanup_trace_resources(cluster: Cluster, job: str, corpus_configmap: str) -> None:
    """Remove the Job and corpus ConfigMap created for one trace attempt."""
    for kind, name in (("job", job), ("configmap", corpus_configmap)):
        if name:
            delete_owned(cluster, cluster.namespace, kind, name)
