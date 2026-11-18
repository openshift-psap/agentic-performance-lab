"""Exact, deterministic allocation of agents to benchmark workloads."""

from __future__ import annotations

import re
import sys


ORDER = ("swe", "pa", "research", "cs")


def allocate_custom_workloads(agents: int, specification: str) -> dict[str, int]:
    if agents <= 0:
        raise ValueError("agent count must be positive")
    allocations = dict.fromkeys(ORDER, 0)
    mode = None
    seen: set[str] = set()
    for part in specification.split(","):
        match = re.fullmatch(r"(swe|pa|research|cs)([:=])([0-9]+)", part)
        if not match:
            raise ValueError(f"invalid workload allocation: {part!r}")
        key, separator, value = match.groups()
        if key in seen:
            raise ValueError(f"duplicate workload: {key}")
        seen.add(key)
        current_mode = "percent" if separator == ":" else "count"
        if mode is not None and mode != current_mode:
            raise ValueError("do not mix counts and percentages")
        mode = current_mode
        allocations[key] = int(value)

    if mode == "count":
        if sum(allocations.values()) != agents:
            raise ValueError(f"workload counts ({sum(allocations.values())}) must equal agent count ({agents})")
        return allocations
    if sum(allocations.values()) != 100:
        raise ValueError(f"workload percentages must sum to 100 (got {sum(allocations.values())})")
    scaled = {key: agents * allocations[key] for key in ORDER}
    counts = {key: scaled[key] // 100 for key in ORDER}
    unassigned = agents - sum(counts.values())
    # Stable ties follow ORDER. No agent silently disappears through rounding.
    for key in sorted(ORDER, key=lambda name: -(scaled[name] % 100))[:unassigned]:
        counts[key] += 1
    return counts


def main() -> int:
    try:
        allocation = allocate_custom_workloads(int(sys.argv[1]), sys.argv[2])
    except (IndexError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    print(",".join(str(allocation[key]) for key in ORDER))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
