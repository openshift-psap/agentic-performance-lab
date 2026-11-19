#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-plan}"
case "$MODE" in
  plan|execute) ;;
  *) echo "usage: $0 [plan|execute]" >&2; exit 2 ;;
esac

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CORPUS="results/client-boundary-100agents-20261025/hundred-agents-25x4-warm-reuse/runtime-replay-corpus.json"
if [[ ! -f "$CORPUS" ]]; then
  echo "research corpus not found: $CORPUS" >&2
  exit 2
fi
if [[ "$MODE" == "execute" && ( -z "${KUBECONFIG:-}" || ! -f "$KUBECONFIG" ) ]]; then
  echo "set KUBECONFIG to the approved cluster credentials before running this script" >&2
  exit 2
fi

SESSION_IDS="$(python3 - "$CORPUS" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
sessions = [s for s in payload.get("sessions", []) if s.get("workload") == "research_deep_search"]
if len(sessions) != 25:
    raise SystemExit(f"expected 25 research sessions, found {len(sessions)}")
print(",".join(str(s["session_id"]) for s in sessions))
PY
)"

export TRACE_REPLAY_RESOURCE_PREFIX=trace-replay
# Opt-in guard: refuses any namespace, resource or node outside this prefix.
# It requires exactly one target node, so enable it together with
# TRACE_REPLAY_ALLOWED_NODE when you want the stricter behaviour.
export TRACE_REPLAY_STRICT_OWNERSHIP="${TRACE_REPLAY_STRICT_OWNERSHIP:-0}"
# Set to a node name to pin the run, or leave empty to let the scheduler decide.
export TRACE_REPLAY_ALLOWED_NODE="${TRACE_REPLAY_ALLOWED_NODE:-}"
export TRACE_REPLAY_OWNED_NAMESPACES=trace-replay,openshell-tracesim
export TRACE_REPLAY_EXTERNAL_SANDBOX_NAMESPACE=openshell-tracesim
export TRACE_REPLAY_OPENSHELL_CLI_VERSION=0.1.8
export TRACE_REPLAY_OPENSHELL_GATEWAY_NAME=openshell-tracesim

OUT_ROOT="${TRACE_REPLAY_BROWSECOMP_OUT_ROOT:-results/browsecompplus-real-25}"
OUT="$OUT_ROOT/$MODE"
if [[ -e "$OUT" ]]; then
  echo "refusing to reuse existing output directory: $OUT" >&2
  exit 2
fi
mkdir -p "$OUT"

python3 scripts/run_experiment.py "--$MODE" \
  --backend-manifest config/backend-manifest.browsecompplus-real.example.json \
  ${TRACE_REPLAY_ALLOWED_NODE:+--nodes "$TRACE_REPLAY_ALLOWED_NODE"} \
  --namespace trace-replay \
  --system-namespace trace-replay \
  --openshell-namespace openshell-tracesim \
  --openshell-gateway-endpoint https://openshell-tracesim.openshell-tracesim.svc.cluster.local:8080 \
  --openshell-client-secret openshell-tracesim-client-tls \
  --workspace-storage-class "${TRACE_REPLAY_WORKSPACE_STORAGE_CLASS:-rook-cephfs-fast}" \
  --agents 1 \
  --warm-pool-size 1 \
  --execution-workers 1 \
  --workload-plan personal_assistant=0,software_engineering_qa=0,research_deep_search=1,customer_support_ops=0 \
  --session-ids "$SESSION_IDS" \
  --require-traces 25 \
  --corpus "$CORPUS" \
  --out "$OUT"
