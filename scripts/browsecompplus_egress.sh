#!/usr/bin/env bash
# Allow this run's OpenShell sandboxes to reach the BrowseComp-Plus backend.
#
# OpenShell fences sandbox pods with the `openshell-sandbox-workloads`
# NetworkPolicy (boundary-role=workload, policyTypes Ingress+Egress, no egress
# rules), so an in-sandbox HTTP call to the retrieval service is dropped and the
# trace records zero backend calls.  NetworkPolicies are additive, so this script
# adds one narrow egress allowance per sandbox boundary pair of the current run:
#
#   sandbox pods of pair <id> -> browsecompplus-real-backend:8093
#
# Nothing else becomes reachable: no DNS, no internet, no other Service.  The
# policy is owned by this experiment (label browsecompplus-real-egress) and
# `delete` removes it again.
#
# Usage:
#   scripts/browsecompplus_egress.sh apply <run-prefix>   # after the pool is ready
#   scripts/browsecompplus_egress.sh status
#   scripts/browsecompplus_egress.sh delete
#
# <run-prefix> is the run marker shown in the sandbox pod name, e.g. 38e4aa1
# for <prefix>-execute-<id>-openclaw-01.
set -euo pipefail

ACTION="${1:-status}"
NS_OPENSSHELL="${TRACE_REPLAY_OPENSHELL_NAMESPACE:-openshell-tracesim}"
NS_BACKEND="${TRACE_REPLAY_NAMESPACE:-trace-replay}"
BACKEND_APP="browsecompplus-real-backend"
PORT=8093
# Only sandboxes of the current run are opened up, never other experiments'.
MATCH="${2:-${TRACE_REPLAY_RUN_PREFIX:-}}"
LABEL="app=browsecompplus-real-egress"

pairs() {
  oc get pods -n "$NS_OPENSSHELL" -l openshell.ai/boundary-role=workload -o json \
    | python3 -c '
import json, sys
match = sys.argv[1]
pods = json.load(sys.stdin).get("items", [])
seen = sorted({
    p["metadata"]["labels"]["openshell.ai/boundary-pair"]
    for p in pods
    if match in p["metadata"]["name"]
    and p["metadata"].get("labels", {}).get("openshell.ai/boundary-pair")
})
print("\n".join(seen))
' "$MATCH"
}

apply_policy() {
  local pair="$1"
  local name="browsecompplus-real-egress-${pair:0:8}"
  oc apply -f - <<YAML
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: ${name}
  namespace: ${NS_OPENSSHELL}
  labels:
    app: browsecompplus-real-egress
    managed-by: trace-replay-experiment
spec:
  podSelector:
    matchLabels:
      openshell.ai/boundary-role: workload
      openshell.ai/boundary-pair: ${pair}
  policyTypes:
    - Egress
  egress:
    - to:
        - namespaceSelector:
            matchLabels:
              kubernetes.io/metadata.name: ${NS_BACKEND}
          podSelector:
            matchLabels:
              app: ${BACKEND_APP}
      ports:
        - protocol: TCP
          port: ${PORT}
YAML
}

case "$ACTION" in
  apply)
    if [[ -z "$MATCH" ]]; then
      echo "usage: $0 apply <run-prefix>  (e.g. the 38e4aa1 from <prefix>-execute-<id>-*)" >&2
      exit 2
    fi
    found=()
    while IFS= read -r pair; do
      [[ -n "$pair" ]] && found+=("$pair")
    done < <(pairs)
    if [[ ${#found[@]} -eq 0 ]]; then
      echo "no sandbox pods matching ${MATCH} in ${NS_OPENSSHELL}; start the run first" >&2
      exit 2
    fi
    for pair in "${found[@]}"; do
      apply_policy "$pair"
    done
    ;;
  delete)
    oc delete networkpolicy -n "$NS_OPENSSHELL" -l "$LABEL" --ignore-not-found
    ;;
  status)
    oc get networkpolicy -n "$NS_OPENSSHELL" -l "$LABEL" || true
    ;;
  *)
    echo "usage: $0 [apply|delete|status]" >&2
    exit 2
    ;;
esac
