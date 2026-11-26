"""Telemetry samplers and backend event collection."""

from __future__ import annotations

import csv
import json
import signal
import subprocess
import threading
import time
import urllib.request
from bisect import bisect_right
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import BASE_JAEGER, CGROUP_SAMPLE_COLUMNS, ROOT, log_phase
from .kubernetes import Cluster, _free_local_port, _wait_for_jaeger


def _start_cgroup_sampler(
    cluster: Cluster, namespace: str, pod: str, container: str, output: Path
) -> tuple[subprocess.Popen[str], Any]:
    output.parent.mkdir(parents=True, exist_ok=True)
    # `oc cp` requires tar in the destination container. OpenShell sandbox
    # images intentionally do not include tar, so stream the sampler through
    # stdin instead. This works with the minimal sandbox image as long as sh
    # and cat are present, and avoids changing the image or cluster state.
    sampler = (ROOT / "scripts/cgroup_sampler.sh").read_text(encoding="utf-8")
    cluster.run(
        [
            "-n",
            namespace,
            "exec",
            "-i",
            pod,
            "-c",
            container,
            "--",
            "sh",
            "-c",
            "cat > /tmp/trace-replay-cgroup-sampler.sh && chmod 700 /tmp/trace-replay-cgroup-sampler.sh",
        ],
        input_text=sampler,
    )
    handle = output.open("w", encoding="utf-8")
    process = subprocess.Popen(
        [
            "oc",
            "-n",
            namespace,
            "exec",
            pod,
            "-c",
            container,
            "--",
            "env",
            "CGROUP_SAMPLE_INTERVAL=0.100",
            "sh",
            "/tmp/trace-replay-cgroup-sampler.sh",
        ],
        env=cluster.env,
        stdout=handle,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    return process, handle


def _start_shared_gateway_sampler(
    cluster: Cluster,
    openshell_ns: str,
    out: Path,
) -> list[tuple[subprocess.Popen[str], Any]]:
    """Sample the shared gateway once for the run, never once per agent."""
    pod = cluster.run(
        [
            "-n",
            openshell_ns,
            "get",
            "pod",
            "-l",
            "app=openshell",
            "-o",
            "jsonpath={.items[0].metadata.name}",
        ],
        check=False,
    ).strip()
    if not pod:
        return []
    container = (
        cluster.run(
            [
                "-n",
                openshell_ns,
                "get",
                "pod",
                pod,
                "-o",
                "jsonpath={.spec.containers[0].name}",
            ],
            check=False,
        ).strip()
        or "gateway"
    )
    return [
        _start_cgroup_sampler(
            cluster,
            openshell_ns,
            pod,
            container,
            out / "shared_openshell_gateway_cgroup.csv",
        )
    ]


def _stop_cgroup_samplers(samplers: list[tuple[subprocess.Popen[str], Any]]) -> None:
    for process, _ in samplers:
        process.terminate()
    for process, handle in samplers:
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        handle.close()


def _cpu_quota_ratio(quota: str | None, period: str | None) -> float | None:
    """Return the CPU-time rate encoded by ``cpu.max``, or None if unbounded."""
    try:
        quota_us = int(str(quota).strip())
        period_us = int(str(period).strip())
    except (TypeError, ValueError):
        return None
    if quota_us <= 0 or period_us <= 0:
        return None
    return quota_us / period_us


def _merge_sandbox_cgroup_samples(cgroup_dir: Path, expected_quota: str = "") -> None:
    """Merge one file per sandbox into the canonical resource series.

    Multiple OpenShell sandboxes can be observed during one trace.  Writing
    all samplers directly to one file lets concurrent stdout streams
    interleave and corrupt rows.  Keep each sampler isolated, then merge only
    complete numeric rows after sampling stops.  Rows from older samplers that
    lack the throttling or PSI columns are padded with zeroes so downstream
    readers see one stable schema.

    When ``expected_quota`` is set, every sample must show that CPU rate.  The
    limit is applied to the sandbox pod at creation; this gate makes a silently
    unbounded sandbox fail the run instead of producing invalid measurements.
    """
    raw_dir = cgroup_dir / "sandbox_samples"
    sources: dict[str, list[dict[str, str]]] = {}
    expected_ratio = (
        _cpu_quota_ratio(*expected_quota.split()) if expected_quota else None
    )
    if expected_quota and expected_ratio is None:
        raise ValueError(f"invalid expected sandbox CPU quota: {expected_quota!r}")
    for path in sorted(raw_dir.glob("*.csv")) if raw_dir.exists() else []:
        try:
            with path.open(encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    try:
                        epoch_ns = int(row.get("epoch_ns") or 0)
                        cpu_usage_usec = int(row.get("cpu_usage_usec") or 0)
                        memory_bytes = int(row.get("memory_bytes") or 0)
                    except ValueError:
                        continue
                    if not (epoch_ns > 0 and cpu_usage_usec >= 0 and memory_bytes >= 0):
                        continue
                    if expected_ratio is not None:
                        observed_ratio = _cpu_quota_ratio(
                            row.get("cpu_quota"), row.get("cpu_period")
                        )
                        if (
                            observed_ratio is None
                            or abs(observed_ratio - expected_ratio) > 1e-6
                        ):
                            raise RuntimeError(
                                f"sandbox CPU quota mismatch in {path.name}: "
                                f"expected {expected_quota}, observed "
                                f"{row.get('cpu_quota')} {row.get('cpu_period')}"
                            )
                    merged = {
                        column: (row.get(column) or "0")
                        for column in CGROUP_SAMPLE_COLUMNS
                    }
                    merged["epoch_ns"] = str(epoch_ns)
                    merged["cpu_usage_usec"] = str(cpu_usage_usec)
                    merged["memory_bytes"] = str(memory_bytes)
                    sources.setdefault(path.name, []).append(merged)
        except OSError:
            continue
    if expected_ratio is not None and not any(sources.values()):
        raise RuntimeError(
            f"sandbox CPU quota {expected_quota} was not observed in any cgroup sample"
        )
    for source_rows in sources.values():
        source_rows.sort(key=lambda row: int(row["epoch_ns"]))
        if any(
            int(current["cpu_usage_usec"]) < int(previous["cpu_usage_usec"])
            for previous, current in zip(source_rows, source_rows[1:])
        ):
            raise RuntimeError("sandbox CPU counter reset within one sampler series")

    if len(sources) <= 1:
        rows = next(iter(sources.values()), [])
    else:
        # A cumulative counter cannot be concatenated across cgroups: a rate
        # computed between pod A and pod B would be meaningless. Interpolate
        # each pod's counter on the union of timestamps, subtract its initial
        # value, then sum the increments. Before its first sample a pod adds
        # zero; after its last sample its counter remains constant.
        counter_columns = (
            "cpu_usage_usec",
            "user_usec",
            "system_usec",
            "nr_throttled",
            "throttled_usec",
            "cpu_pressure_some_total",
            "cpu_pressure_full_total",
        )
        gauge_columns = (
            "cpu_pressure_some_avg10",
            "cpu_pressure_full_avg10",
            "memory_pressure_some_avg10",
            "memory_pressure_full_avg10",
            "io_pressure_some_avg10",
        )
        timelines = [
            ([int(row["epoch_ns"]) for row in source_rows], source_rows)
            for source_rows in sources.values()
            if source_rows
        ]
        timestamps = sorted(
            {timestamp for times, _ in timelines for timestamp in times}
        )
        rows = []
        for timestamp in timestamps:
            aggregate: dict[str, str] = {
                column: "0" for column in CGROUP_SAMPLE_COLUMNS
            }
            aggregate["epoch_ns"] = str(timestamp)
            counters = {column: 0.0 for column in counter_columns}
            gauges = {column: 0.0 for column in gauge_columns}
            memory = 0
            working_set = 0
            inactive_file = 0
            memory_anon = 0
            memory_file = 0
            quota_cores = 0.0
            for times, source_rows in timelines:
                position = bisect_right(times, timestamp) - 1
                if position < 0:
                    continue
                first = source_rows[0]
                before = source_rows[position]
                after = (
                    source_rows[position + 1]
                    if position + 1 < len(source_rows)
                    else before
                )
                fraction = (
                    (timestamp - times[position])
                    / (times[position + 1] - times[position])
                    if position + 1 < len(times)
                    and times[position + 1] > times[position]
                    else 0.0
                )
                for column in counter_columns:
                    left = float(before[column])
                    right = float(after[column])
                    baseline = float(first[column])
                    counters[column] += max(
                        0.0, left + fraction * (right - left) - baseline
                    )
                if timestamp <= times[-1]:
                    memory += int(before["memory_bytes"])
                    working_set += int(before["memory_working_set_bytes"])
                    inactive_file += int(before["memory_inactive_file_bytes"])
                    memory_anon += int(before["memory_anon_bytes"])
                    memory_file += int(before["memory_file_bytes"])
                    quota_cores += (
                        _cpu_quota_ratio(
                            before.get("cpu_quota"), before.get("cpu_period")
                        )
                        or 0.0
                    )
                    for column in gauge_columns:
                        gauges[column] = max(gauges[column], float(before[column]))
            aggregate["memory_bytes"] = str(memory)
            aggregate["memory_working_set_bytes"] = str(working_set)
            aggregate["memory_inactive_file_bytes"] = str(inactive_file)
            aggregate["memory_anon_bytes"] = str(memory_anon)
            aggregate["memory_file_bytes"] = str(memory_file)
            aggregate["cpu_quota"] = (
                str(round(quota_cores * 100000)) if quota_cores else "max"
            )
            aggregate["cpu_period"] = "100000"
            aggregate.update(
                {column: str(round(value)) for column, value in counters.items()}
            )
            aggregate.update({column: str(value) for column, value in gauges.items()})
            rows.append(aggregate)
        (cgroup_dir / "openshell_sandbox_merge.json").write_text(
            json.dumps(
                {
                    "schema": "agentic.trace-replay.sandbox-cgroup-merge.v1",
                    "sources": sorted(sources),
                    "method": "sum per-sandbox baseline-subtracted counters at union timestamps; sum active memory gauges",
                    "pressure_avg10_method": "max of active sources; inspect raw sandbox_samples for per-pod values",
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    output = cgroup_dir / "openshell_sandbox.csv"
    with output.open("w", encoding="utf-8") as handle:
        handle.write(",".join(CGROUP_SAMPLE_COLUMNS) + "\n")
        for row in rows:
            handle.write(
                ",".join(row[column] for column in CGROUP_SAMPLE_COLUMNS) + "\n"
            )


def _valid_cgroup_row_count(path: Path) -> int:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return 0
    count = 0
    for line in lines[1:]:
        fields = line.split(",")
        if len(fields) < 3:
            continue
        try:
            epoch_ns, cpu_usage_usec, memory_bytes = (
                int(field) for field in fields[:3]
            )
        except ValueError:
            continue
        if epoch_ns > 0 and cpu_usage_usec >= 0 and memory_bytes >= 0:
            count += 1
    return count


def _epoch_from_k8s(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _decode_json_log_events(
    text: str, session_id: str, event_name: str
) -> list[dict[str, Any]]:
    """Extract this session's AppWorld events from a pod log stream.

    The AppWorld deployment has existed in two forms: one writes a raw event
    file, while the current deployment emits JSON events directly to stdout.
    Keep the collector independent of that deployment detail and only retain
    canonical tool-event objects for the requested replay session.
    """
    events: list[dict[str, Any]] = []
    for line in text.splitlines():
        candidate = line.strip()
        if not candidate:
            continue
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            # Container log prefixes or wrapped JSON should not make an
            # otherwise valid event disappear.
            start = candidate.find("{")
            if start < 0:
                continue
            try:
                value = json.loads(candidate[start:])
            except json.JSONDecodeError:
                continue
        if not isinstance(value, dict):
            continue
        if (
            value.get("event") != event_name
            or str(value.get("session_id") or "") != session_id
        ):
            continue
        events.append(value)
    return events


def _collect_appworld_events(
    cluster: Cluster,
    system_ns: str,
    appworld_pod: str,
    session_id: str,
    start: float,
) -> list[dict[str, Any]]:
    """Collect only current-run AppWorld events from the Kubernetes log window.

    Raw files and unbounded pod logs are deliberately excluded here.  Both can
    contain events from an earlier replay with the same dataset session ID,
    which would make a run look valid even when the current sandbox never
    reached the backend.
    """
    since = (
        datetime.fromtimestamp(max(0.0, start - 5.0), timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )
    recent_logs = cluster.run(
        ["-n", system_ns, "logs", appworld_pod, "--timestamps", "--since-time", since],
        check=False,
    )
    return _decode_json_log_events(recent_logs, session_id, "appworld_tool")


def _collect_backend_events(
    cluster: Cluster,
    backend: Any,
    session_id: str,
    start: float,
    trace_ids: set[str],
    *,
    run_key: str = "",
    end: float | None = None,
) -> list[dict[str, Any]]:
    """Collect only the current session from the resolved backend deployment.

    Newer workload pods emit canonical events to stdout, while the existing
    backend keeps the same events in its read-only ``/events``
    endpoint and suppresses request logs.  Prefer the bounded log window and
    fall back to that endpoint through ``oc exec`` when the log stream is
    empty; this changes no backend or cluster state.
    """
    if not trace_ids:
        raise ValueError(
            "current Jaeger trace IDs are required for backend event collection"
        )
    since = (
        datetime.fromtimestamp(max(0.0, start - 5.0), timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )

    def current_events(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        # A shared backend may retain older attempts with the same session,
        # turn and call. Only events carrying this driver's Jaeger trace IDs
        # are admissible; choose the latest matching duplicate.
        latest: dict[tuple[Any, Any, str], dict[str, Any]] = {}
        for event in candidates:
            event_trace_id = str(event.get("trace_id") or "")
            event_run_key = str(event.get("run_key") or "")
            trace_matches = bool(event_trace_id and event_trace_id in trace_ids)
            run_matches = bool(
                not event_trace_id
                and run_key
                and event_run_key == run_key
                and end is not None
                and (start - 5.0) * 1e9
                <= int(event.get("timestamp_unix_ns") or 0)
                <= (end + 5.0) * 1e9
            )
            if not (trace_matches or run_matches):
                continue
            key = (event.get("turn"), event.get("call"), str(event.get("tool") or ""))
            previous = latest.get(key)
            if previous is None or int(event.get("timestamp_unix_ns") or 0) >= int(
                previous.get("timestamp_unix_ns") or 0
            ):
                latest[key] = event
        return list(latest.values())

    # Concurrent experiment controllers read the same shared backend log
    # stream.  A transient API/stream failure would otherwise look like "this
    # session emitted no events" and fail output validation, so retry an empty
    # read before falling back to the read-only /events endpoint.
    for attempt in range(3):
        logs = cluster.run(
            [
                "-n",
                backend.namespace,
                "logs",
                f"deployment/{backend.name}",
                "--timestamps",
                "--since-time",
                since,
            ],
            check=False,
        )
        events = current_events(
            _decode_json_log_events(logs, session_id, backend.event_name)
        )
        if events:
            return events
        if logs.strip() or attempt == 2:
            break
        time.sleep(2.0)

    # The domain backend exposes /events on its loopback listener.  Read it
    # from the existing deployment so the controller does not need a new
    # Service, port-forward, or pod mutation.
    port = int(getattr(backend, "port", 0) or 0)
    if not port:
        return []
    fetch = (
        "import json,urllib.request; "
        f"print(json.dumps(json.load(urllib.request.urlopen('http://127.0.0.1:{port}/events'))))"
    )
    raw = cluster.run(
        [
            "-n",
            backend.namespace,
            "exec",
            f"deployment/{backend.name}",
            "--",
            "python",
            "-c",
            fetch,
        ],
        check=False,
    ).strip()
    payload: Any = None
    for line in reversed(raw.splitlines()):
        candidate = line.strip()
        if not candidate:
            continue
        try:
            payload = json.loads(candidate)
            break
        except json.JSONDecodeError:
            continue
    if not isinstance(payload, dict) or not isinstance(payload.get("events"), list):
        return []
    return current_events(
        [
            event
            for event in payload["events"]
            if isinstance(event, dict)
            and event.get("event") == backend.event_name
            and str(event.get("session_id") or "") == session_id
        ]
    )


class JaegerPortForward:
    """Keep a controller-owned local port-forward to the Jaeger service alive."""

    def __init__(self, cluster: Cluster, system_namespace: str) -> None:
        self.cluster = cluster
        self.system_namespace = system_namespace
        self._lock = threading.Lock()
        self._process = self._start()

    def _start(self) -> tuple[subprocess.Popen[str], str]:
        port = _free_local_port()
        process = subprocess.Popen(
            [
                "oc",
                "-n",
                self.system_namespace,
                "port-forward",
                f"svc/{BASE_JAEGER}",
                f"{port}:16686",
            ],
            env=self.cluster.env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return process, f"http://127.0.0.1:{port}"

    @property
    def url(self) -> str:
        return self._process[1]

    def ensure(self, *, force: bool = False) -> str:
        """Return a reachable Jaeger URL, restarting the forward when needed."""
        with self._lock:
            process, url = self._process
            if not force and process.poll() is None:
                try:
                    with urllib.request.urlopen(f"{url}/api/services", timeout=2):
                        return url
                except Exception:  # noqa: BLE001
                    pass
            if process.poll() is None:
                process.send_signal(signal.SIGTERM)
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=3)
            self._process = self._start()
            _wait_for_jaeger(self._process[1], self._process[0])
            log_phase(f"telemetry: Jaeger port-forward restored at {self._process[1]}")
            return self._process[1]

    def wait_ready(self) -> None:
        """Wait for the very first forward to serve requests."""
        _wait_for_jaeger(self.url, self._process[0])

    def close(self) -> None:
        process, _ = self._process
        process.send_signal(signal.SIGTERM)
        process.wait(timeout=10)
