# BrowseComp-Plus retrieval integration

Research tool calls use the same basic mapping pattern as SWE: the harness
replays the recorded tool name and arguments, and our research backend maps
them to an operation against the local BrowseComp-Plus BM25 index.
`search(query)` retrieves ranked IDs, BM25 scores, and indexed text;
`get_document(docid)` retrieves the selected document. The actual text is
truncated at a UTF-8 boundary when needed to fit the recorded tool-result byte
budget, so retrieval content stays real without making replay context much
larger than the trace. The response is padded only with trailing JSON whitespace
when necessary to match that byte count. `submit` stores the recorded answer and
returns its recorded empty/`null` response; it does not score the answer. There
is no separate MCP server.

## Configuration

The default remains the legacy synthetic mode. To enable direct BM25 retrieval,
install the optional package dependencies and provide the local index path:

```sh
pip install '.[browsecomp]'
TRACE_REPLAY_RESEARCH_MODE=browsecompplus \
BROWSECOMP_INDEX_PATH=/data/browsecomp-plus/indexes/bm25 \
python scripts/domain_backend.py --workload research_deep_search --port 8093
```

The BM25 index must be the prebuilt BrowseComp-Plus index or an equivalent index
built from its fixed corpus. The prepared cluster manifest pins the HF dataset
revision to `b3f37f70c33829eb09d04784a54277a31871fd63` so the index bytes are
stable across runs. Upstream's implementation uses Pyserini/Lucene and Java 21.
The result count defaults to five and can be changed with
`BROWSECOMP_K`.

To fetch the same BM25 index locally, use the upstream download command:

```sh
hf download Tevatron/browsecomp-plus-indexes \
  --repo-type dataset --revision b3f37f70c33829eb09d04784a54277a31871fd63 \
  --include 'bm25/*' --local-dir ./indexes
```

When direct retrieval is enabled, recorded result sizes set a text budget for
search and document retrieval. The backend returns actual ranked document IDs,
scores, and text from the index, and the event log records the size policy and
retrieved IDs. Submit calls store their recorded answer locally but do not score
it. If the index or its dependencies are unavailable, startup or the tool call
fails; the backend does not silently fall back to synthetic text. Health
reports whether the index was loaded.

## Measurement boundary and limits

The measured backend call includes the local BM25 lookup and result formatting.
The corpus is fixed, so this path retrieves from indexed BrowseComp-Plus
documents rather than fetching pages from the live web.

The agent harness still receives recorded model continuations. It executes the
real search/document lookup and receives actual indexed content, while later
model decisions remain replayed rather than generated from that content. This
validates the tool-execution path, not a live end-to-end research agent.

## Local 25-session smoke run

The selected corpus has 25 research sessions and 867 recorded tool calls:
786 `search`, 63 `get_document`, and 18 `submit`. After installing the
`browsecomp` optional extra, downloading the upstream BM25 index locally, and
confirming Java 21 is available, run:

```sh
python scripts/smoke_browsecompplus_corpus.py \
  --index /path/to/indexes/bm25 \
  --index-revision b3f37f70c33829eb09d04784a54277a31871fd63 \
  --output results/browsecompplus-local-smoke/calls.jsonl
```

The runner starts the backend bound to `127.0.0.1`, replays calls sequentially,
checks response byte counts, submit bodies, the number of audit events, and
retrieved document IDs, then writes per-call timings, byte counts, IDs, and
errors. It does not store retrieved document text and does not contact
OpenShift. It is a backend integration test, not an OpenClaw performance run or
an answer-accuracy evaluation.

## Cluster deployment preparation

Files prepared for a future owned deployment:

- `containers/browsecompplus/Dockerfile` builds the Python 3.10, Java 21,
  Pyserini runtime and HF downloader.
- `config/backend-manifest.browsecompplus-real.example.json` selects the
  real-retrieval mode and uses `browsecompplus-real-*` resource names in the existing
  `trace-replay` namespace.
- `k8s/browsecompplus-real-backend.yaml` defines an owned PVC, one backend
  replica, and a ClusterIP service. Pinning the pods to a specific node is left
  as a commented `nodeSelector` so the manifest applies to a stock cluster.

Before using these files, the runtime image must be built and placed in the
internal OpenShift image registry, and the selected namespace must have storage
available for the 8 GiB index PVC. The init container downloads only the
upstream BM25 files to that PVC; no results, source, or images are uploaded to a
public service. This repository change does not apply the manifest or inspect
cluster state. The prior generic backend manifest remains unchanged.

### Building the runtime image in the cluster

The image is built by the cluster itself; nothing is uploaded from outside and
the result stays in the internal registry.

```sh
scripts/build_browsecompplus_image.sh
```

The script applies `k8s/browsecompplus-real-build.yaml` (ImageStream plus a
binary BuildConfig), uploads only
`containers/browsecompplus/Dockerfile` and `scripts/domain_backend.py`, and
streams the build log. It outputs
`image-registry.openshift-image-registry.svc:5000/trace-replay/browsecompplus-real-runtime:latest`,
which is exactly the tag the backend manifest references.

The base images (`python:3.10-slim-bookworm`, `eclipse-temurin:21-jre-jammy`) are
floating tags; pin them by digest if a bit-identical rebuild is required. The
Python dependencies are already pinned (`pyserini==1.2.0`,
`huggingface_hub==0.36.0`).

## Sandbox network path

The research tool calls must run inside the OpenShell sandbox, otherwise the
cross-workload comparison is meaningless. The sandbox is fenced twice, and both
layers have to allow the retrieval call:

1. Kubernetes: `openshell-sandbox-workloads` selects the sandbox pods
   (`openshell.ai/boundary-role=workload`) and declares `policyTypes:
   [Ingress, Egress]` with no egress rules. A call to the backend is dropped.
2. OpenShell supervisor: the sandbox policy generated by
   `scripts/run_experiment.py` is binary aware. Only the declared endpoint is
   reachable, and only from allowlisted interpreter paths.

The two failure modes look different, which is the fastest way to tell them
apart:

- Layer 1 missing: TCP retransmits until the client timeout (`urlopen error
  timed out`, up to 180 s), and the backend records nothing.
- Layer 2 missing: the call is refused immediately and the supervisor logs
  `DENIED ... -> <ip>:8093 [reason:transparent_tcp_policy_denied]`. The backend
  records nothing either, but the tool call returns in about a second.

Layer 1 is a per-run allowance, because the sandbox boundary pair only exists
once the run creates its sandbox:

```sh
scripts/browsecompplus_egress.sh apply <run-prefix>   # e.g. 2981a48
scripts/browsecompplus_egress.sh status
scripts/browsecompplus_egress.sh delete               # after the run
```

`apply` finds the boundary pairs of sandbox pods whose name contains the run
prefix and creates one NetworkPolicy per pair
(`browsecompplus-real-egress-<pair8>`, label
`app=browsecompplus-real-egress`). Each policy allows exactly that pair's
sandbox pods to reach the `browsecompplus-real-backend` pods on TCP 8093 via
`namespaceSelector` and `podSelector`. Nothing else becomes reachable: no DNS,
no other Service, no external address. Other runs' sandboxes are never touched.
Run `apply` right after the sandbox pod appears (the pool needs about 20 s
before measurement) and `delete` when the run is done.

Layer 2 travels with the run. The sandbox image resolves `python3` through a
uv-managed venv (`/sandbox/.venv/bin/python3` ->
`/sandbox/.uv/python/cpython-*/bin/python3.*`), and the supervisor matches the
resolved executable path, so the policy allowlist carries those paths next to
the generic `/usr/bin/python3*` and `/opt/conda/...` entries.

The endpoint itself is the Service ClusterIP, not the Service DNS name: the
runtime corpus is rewritten from
`http://browsecompplus-real-backend.<namespace>.svc.cluster.local:8093` to
`http://<clusterIP>:8093` before the queue is dispatched, so the sandbox never
needs DNS egress.

The prepared single-agent cluster command runs the 25 research sessions
serially and checks that the queue contains exactly 25:

```sh
scripts/run_browsecompplus_25.sh plan
scripts/run_browsecompplus_25.sh execute
```

The runtime image tag in the backend manifest is a placeholder for the image
that must first be built and stored in the internal registry. The plan command
does not install the image or deploy any workload.
