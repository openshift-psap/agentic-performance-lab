#!/usr/bin/env bash
# Build the BrowseComp-Plus runtime image inside the approved cluster.
#
# The build context is minimized to the Dockerfile and scripts/domain_backend.py
# so the 20+ GiB of local results are never uploaded. The resulting image is
# written only to the internal registry of trace-replay.
set -euo pipefail

NAMESPACE="${TRACE_REPLAY_NAMESPACE:-trace-replay}"
BUILD="${TRACE_REPLAY_BUILD_NAME:-browsecompplus-real-runtime}"
TAG="${TRACE_REPLAY_BUILD_TAG:-20261119}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONTEXT="$(mktemp -d "${TMPDIR:-/tmp}/browsecompplus-real-build.XXXXXX")"
trap 'rm -rf "$CONTEXT"' EXIT

mkdir -p "$CONTEXT/scripts"
cp "$ROOT/containers/browsecompplus/Dockerfile" "$CONTEXT/Dockerfile"
cp "$ROOT/scripts/domain_backend.py" "$CONTEXT/scripts/domain_backend.py"

oc apply -f "$ROOT/k8s/browsecompplus-real-build.yaml"
oc start-build "$BUILD" -n "$NAMESPACE" --from-dir "$CONTEXT" --follow
oc get "istag/$BUILD:$TAG" -n "$NAMESPACE" -o jsonpath='{.image.dockerImageReference}{"\n"}'
