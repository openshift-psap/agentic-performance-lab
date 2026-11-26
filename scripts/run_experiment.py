#!/usr/bin/env python3
"""Controller entrypoint for the workload-partitioned OpenClaw benchmark.

The :mod:`experiment` package owns deployment, execution, collection, and
analysis. This module parses the command line and dispatches to it; the
re-exported helpers keep the controller's long-standing import surface stable.
"""

# ruff: noqa: E402 -- the package path must exist before its modules import.

from __future__ import annotations

import logging
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
for path in (str(SCRIPTS.parent / "src"), str(SCRIPTS)):
    if path not in sys.path:
        sys.path.insert(0, path)

from experiment.agent_manager import _install_pinned_openshell_wrapper
from experiment.config import (
    CGROUP_SAMPLE_COLUMNS,
    ConfigError,
    agent_name_prefix,
    assert_owned_target,
    build_parser,
    pack_mock_corpus_sessions,
    parse_args,
    parse_tool_cgroup_quota,
    tool_cgroup_quota_cpu_quantity,
    workspace_name,
)
from experiment.kubernetes import (
    _balance_workloads_across_nodes,
    _preflight_kueue_admission,
)
from experiment.metrics import _collect_backend_events, _merge_sandbox_cgroup_samples
from experiment.sandbox_manager import (
    _swe_task_runtime_preflight_code,
    ensure_shared_workspace,
    rewrite_appworld_claim_name,
    verify_appworld_data,
)
from experiment.scheduler import run
from experiment.trace_manager import (
    _swe_command_map,
    _swe_sandbox_command,
    _workload_command,
    build_queue,
    create_trace_job,
    prepare_workload_corpus,
)

__all__ = [
    "CGROUP_SAMPLE_COLUMNS",
    "_balance_workloads_across_nodes",
    "_collect_backend_events",
    "_install_pinned_openshell_wrapper",
    "_merge_sandbox_cgroup_samples",
    "_preflight_kueue_admission",
    "_swe_command_map",
    "_swe_sandbox_command",
    "_swe_task_runtime_preflight_code",
    "_workload_command",
    "agent_name_prefix",
    "assert_owned_target",
    "build_queue",
    "create_trace_job",
    "ensure_shared_workspace",
    "main",
    "pack_mock_corpus_sessions",
    "parse_tool_cgroup_quota",
    "prepare_workload_corpus",
    "rewrite_appworld_claim_name",
    "tool_cgroup_quota_cpu_quantity",
    "verify_appworld_data",
    "workspace_name",
]


def main(argv: list[str] | None = None) -> int:
    """Run the controller from the command line."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = build_parser()
    try:
        return run(parse_args(argv, parser))
    except ConfigError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
