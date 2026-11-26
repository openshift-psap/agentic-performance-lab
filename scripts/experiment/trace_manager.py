"""Corpus preparation, trace Jobs, and trace collection."""

from __future__ import annotations

import base64
import hashlib
import json
import re
import shlex
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from trace_replay_sim.audit_corpus import classify_session as audit_replay_session
from trace_replay_sim.collect import collect as collect_prometheus
from trace_replay_sim.collect import skipped_report as skipped_prometheus_report
from trace_replay_sim.collect import write_report as write_prometheus_report
from trace_replay_sim.jaeger_export import (
    compute_timing_segments,
    extract_per_turn,
)
from trace_replay_sim.jaeger_export import (
    export as export_jaeger,
)
from trace_replay_sim.measurement_tables import write_backend_calls
from trace_replay_sim.per_turn_analysis import analyze as analyze_per_turn
from trace_replay_sim.profiler_v2 import profile as profile_v2
from trace_replay_sim.swe_manifest import (
    SweManifestError,
    SweTask,
    validate_swe_sessions,
)
from trace_replay_sim.workloads import (
    WORKLOADS,
    classify_session,
    select_tool_rich_sessions,
    split_corpus,
)

from .config import (
    LOG,
    RESOURCE_PREFIX,
    RESULT_POLL_INTERVAL_SECONDS,
    SANDBOX_POLL_INTERVAL_SECONDS,
    log_phase,
    safe_name,
)
from .kubernetes import Cluster
from .metrics import (
    _collect_backend_events,
    _epoch_from_k8s,
    _merge_sandbox_cgroup_samples,
    _start_cgroup_sampler,
    _stop_cgroup_samplers,
)
from .validation import validate_collected_trace


def _swe_command_map(session: dict[str, Any]) -> dict[str, list[str]]:
    """Index recorded bash commands to prepared scripts for the SSH boundary."""
    mapping: dict[str, list[str]] = {}
    for turn in session.get("turns") or []:
        for tool in turn.get("tool_calls") or []:
            original = tool.get("swe_original_command")
            mapped = tool.get("mapped_command")
            if not isinstance(original, str) or not original:
                continue
            if (
                not isinstance(mapped, list)
                or len(mapped) != 2
                or mapped[0] != "python3"
            ):
                raise ValueError(
                    "SWE original command has no prepared execution script"
                )
            script = str(mapped[1])
            if not re.fullmatch(
                r"/tmp/trace-replay-tool-calls/[0-9]{4}-[0-9]{3}\.py", script
            ):
                raise ValueError(f"invalid SWE execution script: {script}")
            digest = hashlib.sha256(original.encode("utf-8")).hexdigest()
            mapping.setdefault(digest, []).append(script)
    return mapping


def _appworld_command(
    session_id: str,
    turn: int,
    call: int,
    name: str,
    arguments: dict[str, Any],
    endpoint: str,
) -> list[str]:
    payload = base64.b64encode(
        json.dumps(
            {
                "session_id": session_id,
                "turn": turn,
                "call": call,
                "tool": name,
                "arguments": arguments,
            },
            separators=(",", ":"),
        ).encode()
    ).decode()
    endpoint_literal = json.dumps(endpoint)
    # Keep the executable command below OpenClaw/OpenShell's remote-command
    # limit.  The former shell/base64 pipeline expanded larger AppWorld calls
    # beyond the transport boundary and returned before reaching the backend.
    # A direct Python argv avoids that extra shell layer while retaining a
    # quote-safe base64 payload.
    code = (
        "import base64 as b,json as j,os,urllib.request as u;"
        f'p=j.loads(b.b64decode("{payload}"));'
        "p['trace_id']=os.environ.get('TRACE_REPLAY_TRACE_ID','');"
        "p['run_key']=os.environ.get('TRACE_REPLAY_RUN_KEY','');p=j.dumps(p).encode();"
        f'r=u.Request({endpoint_literal},p,{{"Content-Type":"application/json"}});'
        "x=j.loads(u.urlopen(r,timeout=180).read());"
        'v=x.get("result",x.get("error",x));'
        'print(v if isinstance(v,str) else j.dumps(v),end="")'
    )
    return ["python3", "-c", code]


def prepare_appworld_corpus(
    source: Path,
    out: Path,
    endpoint: str,
    task_map: Path,
) -> Path:
    """Create a trace corpus whose AppWorld tools call the stateful adapter."""
    payload = json.loads(source.read_text(encoding="utf-8"))
    mapping = json.loads(task_map.read_text(encoding="utf-8"))
    if not isinstance(mapping, dict) or not mapping:
        raise RuntimeError(f"AppWorld task map is empty: {task_map}")
    mapped = 0
    for session in payload.get("sessions") or []:
        session_id = str(session.get("session_id") or "")
        if session_id not in mapping:
            raise RuntimeError(
                f"AppWorld task map has no entry for replay session {session_id}; "
                f"use a corpus with a validated mapping or rebuild {task_map}"
            )
        for turn in session.get("turns") or []:
            for call_index, tool in enumerate(turn.get("tool_calls") or []):
                name = str(tool.get("name") or "")
                if name.startswith("mcp__environment__"):
                    tool["mapped_command"] = _appworld_command(
                        session_id,
                        int(turn.get("index", 0)),
                        call_index,
                        name,
                        dict(tool.get("arguments") or {}),
                        endpoint,
                    )
                    mapped += 1
    if mapped == 0:
        raise RuntimeError(
            f"--appworld requested but corpus contains no mcp__environment__ tool calls: {source}"
        )
    payload["tool_backend"] = "appworld"
    payload["appworld_endpoint"] = endpoint
    payload["appworld_task_map_sha256"] = hashlib.sha256(
        task_map.read_bytes()
    ).hexdigest()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log_phase(
        f"corpus: prepared AppWorld-backed replay with {mapped} mapped tool calls"
    )
    return out


def _workload_command(
    session_id: str,
    workload: str,
    benchmark: str,
    turn: int,
    call: int,
    name: str,
    arguments: dict[str, Any],
    endpoint: str,
    recorded_result_bytes: int | None = None,
) -> list[str]:
    """Build a sandbox command for a workload-specific internal API.

    The backend receives the original tool identity and redacted arguments;
    it owns the domain state and returns the tool result.  No recorded result
    is embedded in the command, so a successful run proves that the selected
    backend, rather than the LLM simulator, produced the tool response.
    """
    request = {
        "session_id": session_id,
        "workload": workload,
        "benchmark": benchmark,
        "turn": turn,
        "call": call,
        "tool": name,
        "arguments": arguments,
    }
    if recorded_result_bytes is not None and (
        workload == "customer_support_ops"
        or (
            workload == "research_deep_search"
            and (
                name.endswith("__search")
                or name.endswith("__get_document")
                or name.endswith("__submit")
            )
        )
    ):
        request["recorded_result_bytes"] = recorded_result_bytes
    payload = base64.b64encode(
        json.dumps(request, separators=(",", ":")).encode()
    ).decode()
    endpoint_literal = json.dumps(endpoint.rstrip("/"))
    code = (
        "import base64,json,os,sys,urllib.request,urllib.error;"
        f"p=json.loads(base64.b64decode('{payload}'));"
        "p['trace_id']=os.environ.get('TRACE_REPLAY_TRACE_ID','');"
        "p['run_key']=os.environ.get('TRACE_REPLAY_RUN_KEY','');"
        f"r=urllib.request.Request(os.environ.get('WORKLOAD_BACKEND_URL',{endpoint_literal})+'/execute',"
        "data=json.dumps(p).encode(),headers={'Content-Type':'application/json'});"
        "\ntry:\n response=json.loads(urllib.request.urlopen(r,timeout=180).read())"
        "\nexcept urllib.error.HTTPError as e:\n"
        " raw=e.read().decode('utf-8','replace')\n"
        " try: response=json.loads(raw)\n"
        " except Exception: response={'ok':False,'error':f'HTTP {e.code}: {raw}'}"
        "\nexcept urllib.error.URLError as e:\n"
        " print(f'workload backend transport error: {e}',file=sys.stderr); raise SystemExit(75)"
        "\nexcept Exception as e:\n"
        " print(f'workload backend request error: {e}',file=sys.stderr); raise SystemExit(75)"
        "\nif not response.get('ok',True):\n"
        " print(response.get('error',response),file=sys.stderr); raise SystemExit(2)"
        "\nvalue=response.get('result',response.get('error',response))\n"
        "print(value if isinstance(value,str) else json.dumps(value),end='')"
    )
    return ["python3", "-c", code]


def _swe_sandbox_command(
    session_id: str,
    workload: str,
    benchmark: str,
    turn: int,
    call: int,
    name: str,
    arguments: dict[str, Any],
    endpoint: str,
    task: SweTask,
    tool_cpu_ids: tuple[int, ...] = (),
    tool_cgroup_quota: str = "",
    use_task_image_runtime: bool = False,
    run_key: str = "",
    locale_override: str = "",
    recorded_temp_paths: tuple[str, ...] = (),
) -> list[str]:
    """Execute the recorded SWE command in an OpenShell sandbox with a staged checkout.

    The backend remains the audit/state boundary, but the repository command
    itself runs in OpenShell.  This prevents the benchmark service from
    pretending that a command changed a repository when it did not.
    """
    payload = base64.b64encode(
        json.dumps(
            {
                "session_id": session_id,
                "workload": workload,
                "benchmark": benchmark,
                "turn": turn,
                "call": call,
                "tool": name,
                "arguments": arguments,
                "swe_task": task.public_dict(),
                "tool_cpu_ids": list(tool_cpu_ids),
                "tool_cgroup_quota": tool_cgroup_quota,
                "use_task_image_runtime": use_task_image_runtime,
                "run_key": run_key,
                "locale_override": locale_override,
                "recorded_temp_paths": list(recorded_temp_paths),
            },
            separators=(",", ":"),
        ).encode()
    ).decode()
    endpoint_literal = json.dumps(endpoint.rstrip("/"))
    code = (
        "import base64,glob,json,os,re,resource,shlex,subprocess,sys,time,urllib.request,urllib.error;"
        f"p=json.loads(base64.b64decode('{payload}'));"
        "p['trace_id']=os.environ.get('TRACE_REPLAY_TRACE_ID','');"
        "p['run_key']=os.environ.get('TRACE_REPLAY_RUN_KEY') or p.get('run_key','');"
        "a=p.get('arguments') or {};task=p.get('swe_task') or {};tool_name=str(p.get('tool') or '');"
        "command=str(a.get('command') or a.get('cmd') or '');"
        "repo_root='/sandbox/.openshell/workspace'\n"
        "try:\n os.symlink(repo_root,'/testbed')\nexcept FileExistsError:\n pass\nexcept OSError:\n pass\n"
        "repo_root='/sandbox/.openshell/workspace';cwd=repo_root;"
        "os.environ['GIT_CONFIG_COUNT']='1';os.environ['GIT_CONFIG_KEY_0']='safe.directory';os.environ['GIT_CONFIG_VALUE_0']=repo_root;"
        "result={};verified_base_commit='';"
        "\ntry:\n"
        " if not os.path.isdir(repo_root) or not task.get('base_commit'):\n  raise RuntimeError('SWE workspace does not provide the staged /testbed repository')\n"
        " head=subprocess.run(['git','-C',cwd,'rev-parse','HEAD'],capture_output=True,text=True,timeout=20,check=False)\n"
        " if head.returncode != 0 or head.stdout.strip() != str(task.get('base_commit')):\n  raise RuntimeError('SWE staged repository commit mismatch: expected '+str(task.get('base_commit'))+' observed '+head.stdout.strip())\n"
        " verified_base_commit=head.stdout.strip()\n"
        " if command:\n"
        "  command=command.replace('/testbed',repo_root)\n"
        "  temp_map_path='/tmp/trace-replay-swe-temp-map-'+str(p.get('run_key') or '')+'.json'\n"
        "  try:\n   with open(temp_map_path,encoding='utf-8') as handle: temp_map=json.load(handle)\n"
        "  except (OSError,ValueError): temp_map={}\n"
        "  for recorded_path,actual_path in sorted(temp_map.items(),key=lambda item:-len(item[0])):\n"
        "   if os.path.isdir(actual_path): command=command.replace(recorded_path,actual_path)\n"
        "  if p.get('use_task_image_runtime'):\n"
        "   with open('/tmp/trace-replay-swe-runtime.json',encoding='utf-8') as handle: runtime=json.load(handle)\n"
        "   if runtime.get('base_commit') != task.get('base_commit') or runtime.get('image') != task.get('image'):\n"
        "    raise RuntimeError('SWE task-image runtime does not match the selected task')\n"
        "   python_bin=runtime['python'];repo_src=os.path.join(repo_root,'src');repo_lib=os.path.join(repo_root,'lib')\n"
        "   runtime_path=runtime['path_prefix'];command='export PATH='+shlex.quote(runtime_path)+':$PATH; export PYTHONPATH='+shlex.quote(':'.join((repo_root,repo_src,repo_lib)))+':$PYTHONPATH; '+command\n"
        "  if p.get('locale_override') == 'C':\n"
        "   command='export LANG=C LC_ALL=C PYTHONCOERCECLOCALE=0 PYTHONUTF8=0; '+command\n"
        "  requested=set(int(cpu) for cpu in p.get('tool_cpu_ids') or [])\n"
        "  original=os.sched_getaffinity(0)\n"
        "  if requested and not requested.issubset(original):\n"
        "   raise RuntimeError('requested tool CPUs are outside sandbox effective affinity')\n"
        "  if requested: os.sched_setaffinity(0,requested)\n"
        "  effective=sorted(os.sched_getaffinity(0))\n"
        "  quota=str(p.get('tool_cgroup_quota') or '').strip()\n"
        "  cg_rel=''\n"
        "  try:\n"
        "   for cg_line in open('/proc/self/cgroup'):\n"
        "    cg_parts=cg_line.rstrip().split(':',2)\n"
        "    if len(cg_parts)==3 and cg_parts[1]=='': cg_rel=cg_parts[2]\n"
        "  except OSError: cg_rel=''\n"
        "  cg_dir=os.path.join('/sys/fs/cgroup',cg_rel.lstrip('/')) if cg_rel not in ('','/') else '/sys/fs/cgroup'\n"
        "  def _cg_stat():\n"
        "   values={}\n"
        "   try:\n"
        "    for cg_line in open(os.path.join(cg_dir,'cpu.stat')):\n"
        "     cg_parts=cg_line.split()\n"
        "     if len(cg_parts)>=2:\n"
        "      try: values[cg_parts[0]]=int(cg_parts[1])\n"
        "      except ValueError: pass\n"
        "   except OSError: pass\n"
        "   for cg_kind in ('some','full'):\n"
        "    try:\n"
        "     for cg_line in open(os.path.join(cg_dir,'cpu.pressure')):\n"
        "      cg_parts=cg_line.split()\n"
        "      if cg_parts and cg_parts[0]==cg_kind:\n"
        "       for cg_token in cg_parts[1:]:\n"
        "        cg_key,_,cg_value=cg_token.partition('=')\n"
        "        if cg_value:\n"
        "         try: values['pressure_'+cg_kind+'_'+cg_key]=float(cg_value)\n"
        "         except ValueError: pass\n"
        "    except OSError: pass\n"
        "   return values\n"
        # Projected processes in the sandbox usually run as a non-root SSH
        # user, so they cannot read the pod cgroup even when root can.  The
        # sandbox CPU limit is applied at pod creation and verified by the
        # root-owned sampler, not from inside the tool process.
        "  quota_original=None;quota_applied=None\n"
        "  before_cg=_cg_stat()\n"
        "  try:\n"
        "   before=resource.getrusage(resource.RUSAGE_CHILDREN)\n"
        "   started_epoch_ns=time.time_ns();started_ns=time.monotonic_ns()\n"
        "   completed=subprocess.run(['bash','-lc',command],cwd=cwd,capture_output=True,text=True,timeout=180,check=False)\n"
        "   command_output=completed.stdout+completed.stderr\n"
        "   recorded_paths=list(dict.fromkeys(p.get('recorded_temp_paths') or []))\n"
        "   actual_paths=list(dict.fromkeys(re.findall(r'/tmp/tmp[A-Za-z0-9_-]+',command_output)))\n"
        "   if completed.returncode == 0 and len(recorded_paths) == len(actual_paths) == 1 and os.path.isdir(actual_paths[0]):\n"
        "    temp_map[recorded_paths[0]]=actual_paths[0]\n"
        "    with open(temp_map_path,'w',encoding='utf-8') as handle: json.dump(temp_map,handle)\n"
        "   for recorded_path,actual_path in temp_map.items(): command_output=command_output.replace(actual_path,recorded_path)\n"
        "   ended_ns=time.monotonic_ns();ended_epoch_ns=time.time_ns()\n"
        "   after=resource.getrusage(resource.RUSAGE_CHILDREN)\n"
        "  finally:\n"
        "   after_cg=_cg_stat()\n"
        "   if requested: os.sched_setaffinity(0,original)\n"
        "  def _cg_delta(key):\n"
        "   if key in before_cg and key in after_cg: return after_cg[key]-before_cg[key]\n"
        "   return None\n"
        "  result={'output':command_output[-64000:],'returncode':completed.returncode,'operation':'command',"
        "'command_started_epoch_ns':started_epoch_ns,'command_ended_epoch_ns':ended_epoch_ns,"
        "'command_wall_ns':ended_ns-started_ns,'command_user_cpu_us':round((after.ru_utime-before.ru_utime)*1000000),"
        "'command_system_cpu_us':round((after.ru_stime-before.ru_stime)*1000000),"
        "'command_voluntary_context_switches':after.ru_nvcsw-before.ru_nvcsw,"
        "'command_involuntary_context_switches':after.ru_nivcsw-before.ru_nivcsw,"
        "'command_minor_faults':after.ru_minflt-before.ru_minflt,"
        "'command_major_faults':after.ru_majflt-before.ru_majflt,"
        "'command_effective_cpu_ids':effective,'command_requested_cpu_ids':sorted(requested),"
        "'command_cgroup_quota_requested':quota,'command_cgroup_quota_applied':quota_applied,"
        "'command_cgroup_quota_original':quota_original,"
        "'command_cgroup_usage_usec':_cg_delta('usage_usec'),"
        "'command_cgroup_user_usec':_cg_delta('user_usec'),"
        "'command_cgroup_system_usec':_cg_delta('system_usec'),"
        "'command_cgroup_nr_throttled':_cg_delta('nr_throttled'),"
        "'command_cgroup_throttled_usec':_cg_delta('throttled_usec'),"
        "'command_cgroup_pressure_some_usec':_cg_delta('pressure_some_total'),"
        "'command_cgroup_pressure_full_usec':_cg_delta('pressure_full_total'),"
        "'command_cgroup_pressure_some_avg10':after_cg.get('pressure_some_avg10'),"
        "'command_cgroup_pressure_full_avg10':after_cg.get('pressure_full_avg10')}\n"
        " elif tool_name in ('Read','ReadMcpResourceTool'):\n  path=str(a.get('file_path') or a.get('path') or a.get('uri') or '')\n  path=path[7:] if path.startswith('file://') else path\n  path=path.replace('/testbed',repo_root,1)\n  if not (path == repo_root or path.startswith(repo_root+'/')):\n   raise RuntimeError('SWE read path is outside the staged repository')\n  with open(path,encoding='utf-8',errors='replace') as handle: result={'output':handle.read()[-64000:],'returncode':0,'operation':'read'}\n"
        " elif tool_name in ('Edit','Write'):\n  path=str(a.get('file_path') or a.get('path') or '')\n  path=path.replace('/testbed',repo_root,1)\n  if not (path == repo_root or path.startswith(repo_root+'/')):\n   raise RuntimeError('SWE write path is outside the staged repository')\n  if tool_name == 'Edit':\n   with open(path,encoding='utf-8') as handle: text=handle.read()\n   old=str(a.get('old_string') or ''); new=str(a.get('new_string') or '')\n   if not old or old not in text: raise RuntimeError('SWE edit target was not found')\n   with open(path,'w',encoding='utf-8') as handle: handle.write(text.replace(old,new,1))\n  else:\n   os.makedirs(os.path.dirname(path) or repo_root,exist_ok=True)\n   with open(path,'w',encoding='utf-8') as handle: handle.write(str(a.get('content') or ''))\n  result={'output':'repository file updated','returncode':0,'operation':'write'}\n"
        " elif tool_name in ('Grep','Glob'):\n  base=str(a.get('path') or '/testbed')\n  base=base.replace('/testbed',repo_root,1)\n  if not (base == repo_root or base.startswith(repo_root+'/')): raise RuntimeError('SWE search path is outside the staged repository')\n  if tool_name == 'Glob':\n   pattern=str(a.get('pattern') or '**/*'); result={'output':'\\n'.join(glob.glob(os.path.join(base,pattern),recursive=True))[-64000:],'returncode':0,'operation':'search'}\n  else:\n   pattern=str(a.get('pattern') or ''); found=subprocess.run(['grep','-R','-n','--',pattern,base],cwd=cwd,capture_output=True,text=True,timeout=60,check=False); result={'output':(found.stdout+found.stderr)[-64000:],'returncode':found.returncode,'operation':'search'}\n"
        " else:\n  result={'output':'control tool acknowledged after validating the real SWE repository','returncode':0,'operation':'control'}\n"
        " result.update({'repo_root':cwd,'base_commit':head.stdout.strip()})\n"
        "except subprocess.TimeoutExpired:\n result={'output':'command timed out after 180 seconds','returncode':124,'repo_root':cwd,'base_commit':verified_base_commit}\n"
        "except Exception as e:\n result={'output':str(e),'returncode':1,'repo_root':cwd if verified_base_commit else '','base_commit':verified_base_commit}\n"
        "p['execution_mode']='sandbox';p['sandbox_result']=result;"
        f"r=urllib.request.Request(os.environ.get('WORKLOAD_BACKEND_URL',{endpoint_literal})+'/execute',"
        "data=json.dumps(p).encode(),headers={'Content-Type':'application/json'});"
        "\ntry:\n response=json.loads(urllib.request.urlopen(r,timeout=180).read())\n"
        "except urllib.error.HTTPError as e:\n raw=e.read().decode('utf-8','replace');\n"
        " response={'ok':False,'error':f'HTTP {e.code}: {raw}'}\n"
        "except Exception as e:\n response={'ok':False,'error':str(e)}\n"
        "\nif not response.get('ok',True):\n print(response.get('error',response),file=sys.stderr); raise SystemExit(2)\n"
        "value=response.get('result',result);print(value if isinstance(value,str) else json.dumps(value),end='')"
    )
    return ["python3", "-c", code]


def prepare_workload_corpus(
    source: Path,
    out: Path,
    backends: dict[str, Any],
    swe_manifest: dict[str, SweTask] | None = None,
    selected_session_ids: set[str] | None = None,
    tool_cpu_ids: tuple[int, ...] = (),
    tool_cgroup_quota: str = "",
    swe_task_image_runtime: bool = False,
) -> Path:
    """Map selected trace tools to the resolved backend API boundaries."""
    payload = json.loads(source.read_text(encoding="utf-8"))
    swe_run_key = hashlib.sha256(str(out.parent.resolve()).encode("utf-8")).hexdigest()[
        :32
    ]
    if selected_session_ids is not None:
        payload["sessions"] = [
            session
            for session in (payload.get("sessions") or [])
            if str(session.get("session_id") or "") in selected_session_ids
        ]
        if not payload["sessions"]:
            raise RuntimeError("selected workload corpus contains no sessions to map")
    if swe_manifest is not None:
        validate_swe_sessions(payload.get("sessions") or [], swe_manifest)
    mapped: dict[str, int] = {workload: 0 for workload in WORKLOADS}
    for session in payload.get("sessions") or []:
        session_id = str(session.get("session_id") or "")
        classification = classify_session(session)
        workload = str(session.get("workload") or classification["workload"])
        # The ingested full replay corpus is benchmark-labelled; preserve the
        # normalized workload metadata so downstream queueing and collection
        # use the same classification as split_corpus().
        session.setdefault("workload", workload)
        session.setdefault(
            "workload_profile",
            str(session.get("benchmark") or classification["profile"]),
        )
        if workload == "software_engineering_qa":
            if swe_manifest is None or session_id not in swe_manifest:
                raise SweManifestError(
                    f"SWE trace has no real repository mapping: {session_id}"
                )
            session["swe_task"] = swe_manifest[session_id].public_dict()
        # Reproduce the command locale when the recorded trajectory explicitly
        # reports it. Django's PostgreSQL shell test depends on this setting.
        locale_override = (
            "C"
            if any(
                "Locale encoding: ANSI_X3.4-1968"
                in str(tool.get("recorded_result") or "")
                or "UnicodeEncodeError: 'ascii' codec"
                in str(tool.get("recorded_result") or "")
                for turn in session.get("turns") or []
                for tool in turn.get("tool_calls") or []
            )
            else ""
        )
        spec = backends.get(workload)
        if spec is None:
            continue
        benchmark = str(session.get("benchmark") or spec.benchmark)
        for turn in session.get("turns") or []:
            for call_index, tool in enumerate(turn.get("tool_calls") or []):
                name = str(tool.get("name") or "")
                if workload == "personal_assistant" and name.startswith(
                    "mcp__environment__"
                ):
                    tool["mapped_command"] = _appworld_command(
                        session_id,
                        int(turn.get("index", 0)),
                        call_index,
                        name,
                        dict(tool.get("arguments") or {}),
                        spec.base_url + "/execute",
                    )
                    mapped[workload] += 1
                elif workload == "software_engineering_qa":
                    arguments = dict(tool.get("arguments") or {})
                    recorded_command = str(
                        arguments.get("command") or arguments.get("cmd") or ""
                    )
                    if name in {"mcp__environment__bash", "bash"} and recorded_command:
                        # OpenClaw sees the recorded shell command. The SSH
                        # adapter selects the prepared execution script later.
                        tool["swe_original_command"] = recorded_command
                    full_command = _swe_sandbox_command(
                        session_id,
                        workload,
                        benchmark,
                        int(turn.get("index", 0)),
                        call_index,
                        name,
                        arguments,
                        spec.base_url,
                        swe_manifest[session_id],
                        tool_cpu_ids,
                        tool_cgroup_quota,
                        swe_task_image_runtime,
                        swe_run_key,
                        locale_override,
                        tuple(
                            dict.fromkeys(
                                re.findall(
                                    r"/tmp/tmp[A-Za-z0-9_-]+",
                                    str(tool.get("recorded_result") or ""),
                                )
                            )
                        ),
                    )
                    script_name = f"{int(turn.get('index', 0)):04d}-{call_index:03d}.py"
                    script_dir = (
                        out.parent
                        / "swe-tool-scripts"
                        / hashlib.sha256(session_id.encode()).hexdigest()[:16]
                    )
                    script_dir.mkdir(parents=True, exist_ok=True)
                    (script_dir / script_name).write_text(
                        full_command[2] + "\n", encoding="utf-8"
                    )
                    tool["mapped_command"] = [
                        "python3",
                        f"/tmp/trace-replay-tool-calls/{script_name}",
                    ]
                    mapped[workload] += 1
                elif workload != "personal_assistant":
                    arguments = dict(tool.get("arguments") or {})
                    tool["mapped_command"] = _workload_command(
                        session_id,
                        workload,
                        benchmark,
                        int(turn.get("index", 0)),
                        call_index,
                        name,
                        arguments,
                        spec.base_url,
                        len(tool["recorded_result"].encode("utf-8"))
                        if isinstance(tool.get("recorded_result"), str)
                        else None,
                    )
                    mapped[workload] += 1
    present_workloads = {
        str(session.get("workload") or classify_session(session)["workload"])
        for session in payload.get("sessions") or []
    }
    recorded_calls: dict[str, int] = {}
    for session in payload.get("sessions") or []:
        workload = str(session.get("workload") or classify_session(session)["workload"])
        recorded_calls[workload] = recorded_calls.get(workload, 0) + sum(
            len(turn.get("tool_calls") or []) for turn in session.get("turns") or []
        )
    # A workload whose selected sessions genuinely record zero tool calls
    # (a conversational-only retry selection) correctly maps zero calls; only
    # a silent mapping drop -- recorded calls exist but none mapped -- fails.
    if any(
        mapped.get(workload, 0) == 0 and recorded_calls.get(workload, 0) > 0
        for workload in present_workloads
    ):
        missing = ", ".join(
            f"{workload}={mapped.get(workload, 0)}"
            for workload in sorted(present_workloads)
        )
        raise RuntimeError(
            f"backend mapping produced no calls for one or more workloads: {missing}"
        )
    payload["tool_backend"] = "workload-specific"
    payload["backend_mapping_counts"] = mapped
    payload["backend_fidelity"] = {
        "software_engineering_qa": (
            "real OpenShell commands against pinned source checkout in the task image"
            if swe_task_image_runtime
            else "real OpenShell commands against pinned source checkout; task-image dependencies are not guaranteed"
        ),
        "personal_assistant": "stateful AppWorld API execution",
        "research_deep_search": (
            "BrowseComp-Plus BM25 retrieval over the configured local index; model continuations remain recorded"
            if backends.get("research_deep_search")
            and backends["research_deep_search"].research_mode == "browsecompplus"
            else "local generated responses with recorded byte lengths where available; no live document service"
        ),
        "customer_support_ops": "local generated responses with recorded byte lengths where available; no live support service",
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    payload_rows = []
    for session in payload.get("sessions") or []:
        workload = str(session.get("workload") or "")
        for turn in session.get("turns") or []:
            for call_index, tool in enumerate(turn.get("tool_calls") or []):
                command = tool.get("mapped_command") or []
                original_bytes = len(
                    json.dumps(
                        tool.get("arguments") or {}, separators=(",", ":")
                    ).encode("utf-8")
                )
                mapped_bytes = len(
                    json.dumps(
                        {"command": shlex.join([str(part) for part in command])}
                    ).encode("utf-8")
                )
                payload_rows.append(
                    {
                        "session_id": session.get("session_id"),
                        "workload": workload,
                        "turn": turn.get("index"),
                        "call": call_index,
                        "recorded_args_bytes": original_bytes,
                        "mapped_args_bytes_before_trace_headers": mapped_bytes,
                        "recorded_result_bytes": (
                            len(tool["recorded_result"].encode("utf-8"))
                            if isinstance(tool.get("recorded_result"), str)
                            else None
                        ),
                        "recorded_output_tokens_for_turn": turn.get("output_tokens"),
                        "recorded_args_four_char_fragments": (original_bytes + 3) // 4,
                        "mapped_args_four_char_fragments": (mapped_bytes + 3) // 4,
                        "ratio": round(mapped_bytes / max(1, original_bytes), 3),
                    }
                )
    (out.parent / "tool_payload_fidelity.json").write_text(
        json.dumps(
            {
                "schema": "agentic.trace-replay.tool-payload-fidelity.v1",
                "scope": "arguments only; mapped byte count excludes trace/run environment headers",
                "calls": payload_rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log_phase(f"corpus: mapped workload tool calls through internal APIs ({mapped})")
    return out


def _require_client_boundaries(queue: list[dict[str, Any]]) -> None:
    """Refuse sessions whose mid-session user messages were never recorded.

    A turn that ends without a tool call hands control back to the user.  If
    the corpus has no client boundary there, replaying it as one request would
    stop at that turn and silently drop the rest of the conversation.
    """
    missing: list[str] = []
    for item in queue:
        session = item["session"]
        turns = session.get("turns") or []
        needs_boundaries = any(not turn.get("tool_calls") for turn in turns[:-1])
        if needs_boundaries and not session.get("client_events"):
            missing.append(str(session.get("session_id") or "?"))
    if missing:
        raise RuntimeError(
            "corpus sessions contain mid-session user messages but no client_events "
            "(rebuild the corpus with `trace-replay-sim ingest`): "
            + ", ".join(missing[:5])
        )


def build_queue(
    source: Path,
    out: Path,
    per_workload: int | dict[str, int],
    session_id: str | None = None,
    session_ids: set[str] | None = None,
    max_traces: int | None = None,
    workload_offset: int = 0,
) -> list[dict[str, Any]]:
    partition = out / "workloads"
    # Split the complete corpus before selecting tool-rich sessions. Capping
    # here would bias workloads whose first sessions are conversational
    # no-ops, even when later sessions contain the required tool cycles.
    split_corpus(source, partition, limit_per_workload=None)
    workload_queues: dict[str, list[dict[str, Any]]] = {}
    for workload in WORKLOADS:
        workload_path = partition / f"{workload}.json"
        if not workload_path.exists():
            continue
        payload = json.loads(workload_path.read_text(encoding="utf-8"))
        limit = (
            per_workload.get(workload, 0)
            if isinstance(per_workload, dict)
            else per_workload
        )
        if limit < 1 and not (session_id or session_ids):
            # Zero-count workloads are not part of the run; explicit session
            # selection remains authoritative for the workloads it names.
            continue
        # The benchmark is explicitly a multi-turn/tool-calling benchmark.
        # Exclude recorded conversational sessions with no tool calls before
        # applying the offset/limit; otherwise a workload can silently yield
        # a one-turn no-op and invalidate the profiler tables.
        tool_rich_sessions = [
            session
            for session in payload["sessions"]
            if any(turn.get("tool_calls") for turn in session.get("turns") or [])
        ]
        if session_id or session_ids:
            # Explicit operator selection is authoritative: conversational
            # (tool-less) sessions stay eligible when their IDs are listed.
            # The tool-rich filter still guards every automatic selection.
            sessions = list(payload["sessions"])
        else:
            # Select after tool filtering and balance benchmark profiles. This
            # keeps customer_support_ops from becoming airline-only merely
            # because tau2 airline rows happen to come first in the corpus.
            sessions = select_tool_rich_sessions(
                tool_rich_sessions[workload_offset:],
                limit,
            )
        for index, session in enumerate(sessions, start=workload_offset):
            if session_id and str(session.get("session_id")) != session_id:
                continue
            if session_ids and str(session.get("session_id")) not in session_ids:
                continue
            workload_queues.setdefault(workload, []).append(
                {"workload": workload, "index": index, "session": session}
            )
    # Interleave workloads instead of admitting every long SWE trace first.
    # This balances backend pressure and prevents one workload from occupying
    # the complete active-controller window.
    queue: list[dict[str, Any]] = []
    while any(workload_queues.values()):
        for workload in WORKLOADS:
            items = workload_queues.get(workload) or []
            if items:
                queue.append(items.pop(0))
    queue = queue[:max_traces] if max_traces is not None else queue
    nested_sessions = [
        str(item["session"].get("session_id") or "")
        for item in queue
        if audit_replay_session(item["session"])["nested_or_subagent_tools"]
    ]
    if nested_sessions:
        raise RuntimeError(
            "nested/subagent trajectories require a graph-aware replay harness: "
            + ", ".join(nested_sessions[:10])
        )
    out.mkdir(parents=True, exist_ok=True)
    (out / "selected_trajectory_audit.json").write_text(
        json.dumps(
            {
                "schema": "agentic.trace-replay.selected-trajectory-audit.v1",
                "selected_sessions": len(queue),
                "sessions": [audit_replay_session(item["session"]) for item in queue],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    _require_client_boundaries(queue)
    return queue


def create_trace_job(
    cluster: Cluster,
    item: dict[str, Any],
    agent: str,
    out: Path,
    system_ns: str,
    run_id: str,
    attempt: int = 0,
    *,
    allow_cross_node: bool = False,
    prewarm_requested: bool = True,
    warm_pool_slot: int | None = None,
    warm_pool_size: int = 0,
    agent_node: str | None = None,
    placement_node: str | None = None,
    prewarm_only: bool = False,
    user_key: str | None = None,
    client_gap_ms: float = 0.0,
) -> tuple[str, str, Path, str]:
    workload = safe_name(item["workload"])
    index = int(item["index"])
    trace_dir = out / f"{index:03d}-{item['session']['session_id']}__{workload}"
    # Preserve incomplete attempt evidence outside the active result tree.
    # Retrying in-place would mix old mock edges with the new attempt.
    if attempt > 0 and trace_dir.exists():
        failed_root = out.parent / "failed-attempts" / trace_dir.name
        failed_root.mkdir(parents=True, exist_ok=True)
        trace_dir.rename(failed_root / f"attempt-{attempt}-{time.time_ns()}")
    trace_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = trace_dir / "input_corpus.json"
    corpus_path.write_text(
        json.dumps(
            {"dataset": "Exgentic/agent-llm-traces", "sessions": [item["session"]]},
            indent=2,
        )
    )
    cm = f"{RESOURCE_PREFIX}-{run_id}-corpus-{workload[:20]}-{index:03d}"
    job = f"{RESOURCE_PREFIX}-{run_id}-trace-{workload[:20]}-{index:03d}" + (
        f"-r{attempt}" if attempt else ""
    )
    if prewarm_only:
        # Prewarm Jobs are short-lived control traffic; keep their resource
        # names well below Kubernetes' 63-character name/label limit even
        # when the workload name is long.
        cm = f"{RESOURCE_PREFIX}-swp-cm-{run_id}-{index:03d}"[:63].rstrip("-")
        job = f"{RESOURCE_PREFIX}-swp-{run_id}-{index:03d}"[:63].rstrip("-")
    cluster.upsert_owned_configmap(
        cluster.namespace,
        {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": {
                "name": cm,
                "labels": {
                    "managed-by": "trace-replay-experiment",
                    "component": "trace-corpus",
                    "run": f"{RESOURCE_PREFIX}-{run_id}",
                    "workload": workload,
                },
            },
            "data": {"replay.json": corpus_path.read_text(encoding="utf-8")},
        },
    )
    driver_args = [
        "--corpus",
        "/data/replay.json",
        "--out",
        "/results",
        "--layer",
        "shell",
        "--url",
        f"http://{agent}.{cluster.namespace}.svc.cluster.local:18790",
        "--model",
        "openclaw/perf_agent",
        "--token",
        "change-me-gateway-token",
        "--timeout",
        "1800",
        "--stream",
    ]
    if item["session"].get("client_events"):
        # The driver only plays the user: one request per recorded client
        # message.  The harness owns the tool loop in between, so a model turn
        # without a tool call never truncates the rest of the conversation.
        driver_args.append("--client-turns")
        if client_gap_ms > 0:
            driver_args.extend(["--client-gap-ms", str(client_gap_ms)])
    if prewarm_requested:
        driver_args.append("--prewarm")
    if prewarm_only:
        driver_args.append("--prewarm-only")
    if user_key:
        driver_args.extend(["--user-key", user_key])
    driver_container: dict[str, Any] = {
        "name": "driver",
        "image": "python:3.12-slim",
        "command": ["python", "/app/driver.py"],
        # Drive only the user side of the recorded conversation while keeping
        # one harness session.  Traces with real user messages mid-session get
        # one request per client boundary (--client-turns); the harness owns
        # the tool loop, so the driver never sends a request per model turn.
        "args": driver_args,
        "volumeMounts": [
            {"name": "driver-app", "mountPath": "/app"},
            {"name": "corpus", "mountPath": "/data"},
            {"name": "results", "mountPath": "/results"},
        ],
    }
    pod_spec: dict[str, Any] = {
        "restartPolicy": "Never",
        "initContainers": [
            {
                "name": "init-driver",
                "image": "python:3.12-slim",
                "command": ["python", "-c"],
                "args": [
                    "import tarfile; tarfile.open('/bundle/source_bundle.tgz', 'r:gz').extractall('/app')"
                ],
                "volumeMounts": [
                    {"name": "source", "mountPath": "/bundle", "readOnly": True},
                    {"name": "driver-app", "mountPath": "/app"},
                ],
            }
        ],
        "containers": [
            driver_container,
            {
                # Keep the shared emptyDir mounted after the driver exits so the
                # collector can copy telemetry before the pod terminates.  The
                # controller acknowledges the copy by creating the marker below;
                # a fixed sleep is unsafe because a realistic multi-turn replay
                # can outlive the sleep and leave the collector unable to exec or
                # cp from a completed pod.
                "name": "result-hold",
                "image": "python:3.12-slim",
                "command": ["sh", "-c"],
                "args": [
                    "while [ ! -f /results/.trace-replay-collector-done ]; do sleep 1; done"
                ],
                "volumeMounts": [{"name": "results", "mountPath": "/results"}],
            },
        ],
        "volumes": [
            {
                "name": "source",
                "configMap": {"name": f"{RESOURCE_PREFIX}-experiment-driver"},
            },
            {"name": "driver-app", "emptyDir": {}},
            {"name": "corpus", "configMap": {"name": cm}},
            {"name": "results", "emptyDir": {}},
        ],
    }
    if prewarm_only:
        # The synthetic prewarm has no collector phase.  Let the Job finish
        # immediately after the driver exits so the controller can stage the
        # repository while the agent-scoped OpenShell sandbox remains alive.
        pod_spec["containers"] = [driver_container]
    if placement_node:
        # Trace/prewarm drivers and SWE staging use only ephemeral scratch
        # space and reach the agent over its Service.  Keep them off an
        # approved node currently under disk pressure without moving the
        # agent-scoped OpenShell sandbox itself.
        pod_spec["nodeSelector"] = {"kubernetes.io/hostname": placement_node}
    elif not allow_cross_node:
        pod_spec["nodeSelector"] = {
            "kubernetes.io/hostname": agent_node or cluster.node
        }
    trace_labels = {
        "app": "trace-replay-experiment",
        "component": "trace-job",
        "managed-by": "trace-replay-experiment",
        "run": f"{RESOURCE_PREFIX}-{run_id}",
        "workload": workload,
        "trace-replay-warm-pool": "true" if warm_pool_size else "false",
        "trace-replay-warm-pool-slot": str(warm_pool_slot or 0),
        "trace-replay-prewarm-requested": "true" if prewarm_requested else "false",
    }
    job_obj = {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {"name": job, "labels": trace_labels},
        "spec": {
            "backoffLimit": 0,
            "ttlSecondsAfterFinished": 600,
            "template": {
                "metadata": {"labels": dict(trace_labels)},
                "spec": pod_spec,
            },
        },
    }
    cluster.apply_json(job_obj)
    return job, cm, trace_dir, str(item["session"].get("session_id") or "")


def _trace_has_session(trace: dict[str, Any], session_id: str) -> bool:
    return session_id in json.dumps(trace, separators=(",", ":"))


def _trace_overlaps_window(trace: dict[str, Any], start_us: int, end_us: int) -> bool:
    for span in trace.get("spans") or []:
        span_start = int(span.get("startTime") or 0)
        span_end = span_start + int(span.get("duration") or 0)
        if span_start <= end_us and span_end >= start_us:
            return True
    return False


def _rewrite_filtered_jaeger(
    trace_dir: Path, session_id: str, start_us: int, end_us: int
) -> int:
    traces_dir = trace_dir / "data/shell/traces"
    raw_path = traces_dir / "raw_traces.json"
    if not raw_path.exists():
        return 0
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    edge_path = trace_dir / "data/shell/mock_edges.jsonl"
    edge_trace_ids: set[str] = set()
    warmup_trace_ids: set[str] = set()
    if edge_path.exists():
        for line in edge_path.read_text(encoding="utf-8").splitlines():
            try:
                edge = json.loads(line)
                edge_trace_id = str(edge.get("trace_id") or "")
            except json.JSONDecodeError:
                continue
            if edge_trace_id:
                if edge.get("warmup"):
                    warmup_trace_ids.add(edge_trace_id)
                else:
                    edge_trace_ids.add(edge_trace_id)
    # A session can be retried, so an old mock edge is not sufficient identity.
    # First anchor Jaeger data to this driver's session *and* execution interval.
    # Edge IDs are only a narrowing hint; they must never reintroduce an earlier
    # retry of the same session into this trace's telemetry.
    interval_traces = [
        trace
        for trace in raw
        if _trace_has_session(trace, session_id)
        and _trace_overlaps_window(trace, start_us, end_us)
    ]
    if edge_trace_ids:
        narrowed = [
            trace
            for trace in interval_traces
            if str(trace.get("traceID") or "") in edge_trace_ids
        ]
        filtered = narrowed or interval_traces
    else:
        filtered = interval_traces
    warmup_filtered = [
        trace for trace in raw if str(trace.get("traceID") or "") in warmup_trace_ids
    ]
    (traces_dir / "raw_traces.json").write_text(
        json.dumps(filtered, indent=2), encoding="utf-8"
    )
    (traces_dir / "warmup_raw_traces.json").write_text(
        json.dumps(warmup_filtered, indent=2), encoding="utf-8"
    )
    per_turn, per_tool, sessions = extract_per_turn(filtered)
    (traces_dir / "per_turn_segments.json").write_text(
        json.dumps(per_turn, indent=2), encoding="utf-8"
    )
    (traces_dir / "per_tool_segments.json").write_text(
        json.dumps(per_tool, indent=2), encoding="utf-8"
    )
    (traces_dir / "session_summary.json").write_text(
        json.dumps(sessions, indent=2), encoding="utf-8"
    )
    (traces_dir / "timing_segments.json").write_text(
        json.dumps(compute_timing_segments(filtered), indent=2), encoding="utf-8"
    )
    return len(filtered)


def _first_openshell_action_ms(trace_dir: Path) -> float | None:
    """Return the first real OpenClaw-to-OpenShell action span."""
    path = trace_dir / "data/shell/traces/raw_traces.json"
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    candidates: list[tuple[int, float]] = []
    for payload in raw if isinstance(raw, list) else []:
        for span in payload.get("spans") or []:
            if span.get("operationName") != "openclaw.exec":
                continue
            try:
                candidates.append(
                    (
                        int(span.get("startTime") or 0),
                        float(span.get("duration") or 0.0) / 1000.0,
                    )
                )
            except (TypeError, ValueError):
                continue
    return (
        round(min(candidates, key=lambda item: item[0])[1], 3) if candidates else None
    )


def _conversational_trace(trace_dir: Path) -> bool:
    """True when the trace's recorded session has zero tool calls.

    Conversational traces never execute an OpenShell action; their
    sandbox-acquisition and OpenShell execution metrics are not applicable
    and are excluded from tool-stage aggregates by the analysis layer.
    """
    try:
        payload = json.loads(
            (trace_dir / "input_corpus.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError):
        return False
    sessions = payload.get("sessions") or []
    if len(sessions) != 1:
        return False
    return not any(turn.get("tool_calls") for turn in sessions[0].get("turns") or [])


def _warmup_openshell_span_ms(trace_dir: Path) -> float | None:
    """Return the measured OpenShell span from the warmup request, if exported.

    The full warmup request includes model-simulator and harness time. This
    helper isolates the OpenShell command span so it is not mislabeled as a
    zero-cost warm reuse or as cold sandbox initialization.
    """
    path = trace_dir / "data/shell/traces/warmup_raw_traces.json"
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    durations: list[float] = []
    seen: set[str] = set()
    for payload in raw if isinstance(raw, list) else []:
        for span in payload.get("spans") or []:
            if span.get("operationName") not in {"openclaw.exec", "openshell.exec"}:
                continue
            span_id = str(span.get("spanID") or "")
            if span_id and span_id in seen:
                continue
            if span_id:
                seen.add(span_id)
            try:
                durations.append(float(span.get("duration") or 0.0) / 1000.0)
            except (TypeError, ValueError):
                continue
    if not durations:
        return None
    return round(sum(durations), 3)


def collect_trace(
    cluster: Cluster,
    job: str,
    trace_dir: Path,
    session_id: str,
    start: float,
    system_ns: str,
    jaeger: str,
    agent: str,
    workspace: str,
    openshell_ns: str,
    backend: Any,
    mock_deployment: str,
    *,
    allow_cross_node: bool = False,
    prewarm_requested: bool = True,
    warm_pool_slot: int | None = None,
    warm_pool_size: int = 0,
    expected_node: str | None = None,
    jaeger_supplier: Any | None = None,
    jaeger_io_lock: threading.Lock | None = None,
    expected_cgroup_quota: str = "",
) -> None:
    openclaw_pod = cluster.run(
        [
            "-n",
            cluster.namespace,
            "get",
            "pod",
            "-l",
            f"app={agent}",
            "-o",
            "jsonpath={.items[0].metadata.name}",
        ]
    ).strip()
    openclaw_node = cluster.run(
        [
            "-n",
            cluster.namespace,
            "get",
            "pod",
            openclaw_pod,
            "-o",
            "jsonpath={.spec.nodeName}",
        ]
    ).strip()
    cgroup_dir = trace_dir / "data/shell/prometheus/cgroup"
    samplers = [
        _start_cgroup_sampler(
            cluster,
            cluster.namespace,
            openclaw_pod,
            "gateway",
            cgroup_dir / "openclaw_gateway.csv",
        ),
    ]
    # The OpenShell gateway is shared across agents. Its run-level sampler is
    # started once by main(), not inside each per-agent cgroup window.
    lifecycle_records: dict[str, dict[str, Any]] = {}
    lifecycle_sampler_started: set[str] = set()
    # This flag is used by both lifecycle paths, including completed-Job recovery.
    warm_pool_slot_ready = False
    watcher_stop = threading.Event()
    watcher_errors: list[str] = []

    def sandbox_matches(pod: dict[str, Any], pod_name: str) -> bool:
        """Match only this run's sandbox, without assuming pod-name syntax.

        OpenShell versions have used both ``<workspace>--<id>`` and
        ``<workspace>-<id>`` pod names.  The workspace is unique per run and
        starts with ``trace-replay-``; checking names, labels, and annotations keeps
        collection version-tolerant while excluding unrelated sandboxes.
        """
        metadata = pod.get("metadata") or {}
        values = [pod_name]
        values.extend(str(value) for value in (metadata.get("labels") or {}).values())
        values.extend(
            str(value) for value in (metadata.get("annotations") or {}).values()
        )

        # Workspace names are prefixes of OpenShell pod names.  Require a
        # delimiter after the full workspace so workspace ``...-1`` cannot
        # accidentally match ``...-10`` through ``...-19`` from other agents.
        def has_workspace_boundary(value: str) -> bool:
            start = value.find(workspace)
            while start >= 0:
                end = start + len(workspace)
                if end == len(value) or not value[end].isalnum():
                    return True
                start = value.find(workspace, start + 1)
            return False

        return any(has_workspace_boundary(value) for value in values)

    def watch_sandbox() -> None:
        while not watcher_stop.wait(SANDBOX_POLL_INTERVAL_SECONDS):
            try:
                pods = cluster.json(["-n", openshell_ns, "get", "pods"])
                for pod in pods.get("items", []):
                    name = str(pod.get("metadata", {}).get("name") or "")
                    spec = pod.get("spec", {})
                    if name == "openshell-0" or not sandbox_matches(pod, name):
                        continue
                    metadata = pod.get("metadata", {})
                    status = pod.get("status", {})
                    conditions = status.get("conditions") or []
                    ready_condition = next(
                        (
                            condition
                            for condition in conditions
                            if condition.get("type") == "Ready"
                        ),
                        {},
                    )
                    created_unix = _epoch_from_k8s(metadata.get("creationTimestamp"))
                    ready_unix = (
                        (
                            _epoch_from_k8s(ready_condition.get("lastTransitionTime"))
                            or time.time()
                        )
                        if ready_condition.get("status") == "True"
                        else None
                    )
                    record = lifecycle_records.setdefault(
                        name, {"pod": name, "observed_unix": time.time()}
                    )
                    record.update(
                        {
                            "node": spec.get("nodeName"),
                            "phase": status.get("phase"),
                            "created_unix": created_unix,
                            "started_unix": _epoch_from_k8s(status.get("startTime")),
                            "ready_unix": ready_unix,
                            "ready": ready_condition.get("status") == "True",
                            "ready_reason": ready_condition.get("reason"),
                            "last_observed_unix": time.time(),
                        }
                    )
                    # A Pending pod has no nodeName yet.  Do not abort the
                    # watcher during that normal scheduling window; enforce the
                    # node constraint only after Kubernetes has assigned one.
                    assigned_node = spec.get("nodeName")
                    expected_sandbox_node = expected_node or cluster.node
                    if (
                        not allow_cross_node
                        and assigned_node
                        and assigned_node != expected_sandbox_node
                    ):
                        watcher_errors.append(
                            f"sandbox {name} scheduled on {spec.get('nodeName')}, expected {expected_sandbox_node}"
                        )
                        return
                    containers = [
                        str(item.get("name")) for item in spec.get("containers", [])
                    ]
                    container = (
                        "agent"
                        if "agent" in containers
                        else (containers[0] if containers else "")
                    )
                    if (
                        status.get("phase") == "Running"
                        and container
                        and name not in lifecycle_sampler_started
                    ):
                        sampler_path = (
                            cgroup_dir / "sandbox_samples" / f"{safe_name(name)}.csv"
                        )
                        try:
                            samplers.append(
                                _start_cgroup_sampler(
                                    cluster, openshell_ns, name, container, sampler_path
                                )
                            )
                            lifecycle_sampler_started.add(name)
                        except Exception:
                            # A transient exec failure under concurrent load
                            # must not kill the watcher: the pod stays absent
                            # from lifecycle_sampler_started and the next poll
                            # retries the sampler for it.
                            pass
            except Exception as exc:  # noqa: BLE001
                watcher_errors.append(str(exc))
                return

    watcher = threading.Thread(
        target=watch_sandbox, name="trace-replay-sandbox-cgroup-watcher", daemon=True
    )
    watcher.start()
    pod = ""
    completed_before_collection = False
    result_deadline = time.monotonic() + 7200
    while time.monotonic() < result_deadline:
        pod = cluster.run(
            [
                "-n",
                cluster.namespace,
                "get",
                "pod",
                "-l",
                f"job-name={job}",
                "-o",
                "jsonpath={.items[0].metadata.name}",
            ],
            check=False,
        ).strip()
        if pod:
            phase = cluster.run(
                [
                    "-n",
                    cluster.namespace,
                    "get",
                    "pod",
                    pod,
                    "-o",
                    "jsonpath={.status.phase}",
                ],
                check=False,
            ).strip()
            if phase in {"Failed", "Unknown"}:
                raise RuntimeError(
                    f"replay pod {pod} entered {phase} before writing results"
                )
            # A completed pod cannot be exec'd or copied from by OpenShift.
            # Treat this as a hard collection failure rather than polling
            # forever; the normal path keeps result-hold running until the
            # controller acknowledges the copy below.
            if phase == "Succeeded":
                # Some Kubernetes runtimes can mark a multi-container Job
                # complete during the short handoff between the driver and
                # result-holder containers.  The result volume may still be
                # copyable from the terminated pod, so try that recovery path
                # before declaring the trace lost.
                completed_before_collection = True
                break
            ready = cluster.run(
                [
                    "-n",
                    cluster.namespace,
                    "exec",
                    "-c",
                    "result-hold",
                    pod,
                    "--",
                    "sh",
                    "-c",
                    "test -s /results/summary.json && echo ready",
                ],
                check=False,
            ).strip()
            if ready == "ready":
                break
        time.sleep(RESULT_POLL_INTERVAL_SECONDS)
    else:
        raise TimeoutError(f"timed out waiting for replay results from {job}")

    shell = trace_dir / "data/shell"
    shell.mkdir(parents=True, exist_ok=True)
    cluster.run(
        [
            "-n",
            cluster.namespace,
            "cp",
            "-c",
            "result-hold",
            f"{pod}:/results/.",
            str(shell),
        ]
    )
    if not (shell / "summary.json").is_file():
        raise RuntimeError(
            f"replay pod {pod} completed before result collection and no summary.json was copyable"
        )
    if not completed_before_collection:
        cluster.run(
            [
                "-n",
                cluster.namespace,
                "exec",
                "-c",
                "result-hold",
                pod,
                "--",
                "sh",
                "-c",
                "touch /results/.trace-replay-collector-done",
            ]
        )
        try:
            cluster.run(
                [
                    "-n",
                    cluster.namespace,
                    "wait",
                    "--for=condition=complete",
                    f"job/{job}",
                    "--timeout=7200s",
                ]
            )
        finally:
            watcher_stop.set()
            watcher.join(timeout=5)
            _stop_cgroup_samplers(samplers)
            _merge_sandbox_cgroup_samples(cgroup_dir, expected_cgroup_quota)
    else:
        watcher_stop.set()
        watcher.join(timeout=5)
        _stop_cgroup_samplers(samplers)
        _merge_sandbox_cgroup_samples(cgroup_dir, expected_cgroup_quota)
        ready_records = [
            record
            for record in lifecycle_records.values()
            if record.get("ready")
            and record.get("created_unix")
            and record.get("ready_unix")
        ]
        driver_summary = {}
        try:
            driver_summary = json.loads(
                (shell / "summary.json").read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError):
            pass
        warmup = driver_summary.get("warmup") or {}
        warmup_start = float(warmup.get("start_unix") or 0.0)
        warmup_end = float(warmup.get("end_unix") or 0.0)
        warmup_ms = float(warmup.get("duration_ms") or 0.0)
        created_during_warmup = bool(
            ready_records
            and warmup_start
            and warmup_end
            and warmup_start - 1.0
            <= float(ready_records[0]["created_unix"])
            <= warmup_end + 1.0
        )
        warmup_succeeded = warmup.get("ok") is True
        driver_ok = int(driver_summary.get("ok") or 0) > 0
        created_to_ready_ms = (
            round(
                max(
                    0.0,
                    ready_records[0]["ready_unix"] - ready_records[0]["created_unix"],
                )
                * 1000.0,
                3,
            )
            if ready_records
            else None
        )
        # Keep the one-time pool-fill request separate from the measured
        # trace path.  A sandbox that was already ready must not appear as
        # seconds of "sandbox init" in the first turn waterfall; its actual
        # OpenShell tool execution is measured independently.
        sandbox_reuse_ms: float | None = None
        # Pool-fill evidence is durable run state. The per-trace watcher starts
        # after pool fill, so it cannot observe the already-ready sandbox itself.
        # Pool size is global; only this agent's assigned slot can establish
        # warm reuse. A partial pool must leave unassigned agents cold.
        warm_pool_slot_ready = warm_pool_slot is not None
        try:
            pool_evidence = json.loads(
                (trace_dir.parent.parent / "warm_pool_evidence.json").read_text(
                    encoding="utf-8"
                )
            )
            warm_pool_slot_ready = warm_pool_slot is not None and any(
                isinstance(record, dict)
                and record.get("status") == "ready"
                and record.get("agent") == agent
                and record.get("workspace") == workspace
                for record in pool_evidence
            )
        except (OSError, json.JSONDecodeError):
            pass
        if prewarm_requested:
            if ready_records:
                readiness_evidence = "sandbox_pod_ready_observed_during_pool_prewarm"
                lifecycle_status = "ready" if warmup_succeeded else "not_ready"
                provisioning_mode = (
                    "cold_pool_fill" if created_during_warmup else "warm_reused"
                )
            elif not lifecycle_records and warmup_succeeded:
                # OpenShell can reuse an already-running agent-scoped sandbox.
                # The harmless prewarm exec is direct evidence of readiness;
                # no new pod is expected in this case.
                readiness_evidence = "successful_open_shell_prewarm_no_new_pod"
                lifecycle_status = "ready"
                provisioning_mode = "warm_reused"
            else:
                readiness_evidence = "sandbox_pod_not_ready"
                lifecycle_status = "not_ready"
                provisioning_mode = "warm_pool_unavailable"
            sandbox_init_ms = 0.0
            sandbox_reuse_ms = None
            if provisioning_mode == "warm_reused":
                sandbox_init_measurement = (
                    "already-ready warm reuse; cold sandbox initialization is absent from the measured trace path; "
                    "OpenShell tool execution measured separately"
                )
            else:
                sandbox_init_measurement = "sandbox filled before the measured trace; one-time pool-fill cost is in prewarm_request_ms"
            prewarm_status = "measured" if warmup_ms > 0 else "missing"
        elif warm_pool_slot_ready:
            # Pool fill completed before dispatch and would have raised on failure.
            # The per-trace watcher starts after that fill, so it cannot be required
            # to rediscover the already-ready sandbox.
            # This preserves the distinction from a true cold fallback.

            readiness_evidence = "verified_ready_pool_slot_before_measured_trace"
            lifecycle_status = "ready" if driver_ok else "not_ready"
            provisioning_mode = "warm_reused"
            sandbox_init_ms = 0.0
            sandbox_reuse_ms = None
            sandbox_init_measurement = "sandbox pool filled before measured request; request-to-first-action is separate"
            prewarm_status = "measured"
        elif ready_records:
            # No pool slot was available. The first real tool call created a
            # sandbox, so retain Kubernetes create-to-ready as the cold-start
            # measurement instead of silently labeling the trace warm.
            readiness_evidence = "sandbox_pod_ready_observed_without_prewarm"
            lifecycle_status = "ready" if driver_ok else "not_ready"
            provisioning_mode = "cold_fallback"
            sandbox_init_ms = created_to_ready_ms or 0.0
            sandbox_reuse_ms = None
            sandbox_init_measurement = (
                "Kubernetes sandbox pod create-to-ready after warm-pool exhaustion"
            )
            prewarm_status = "not_requested"
        elif not lifecycle_records and driver_ok:
            # An existing agent-scoped sandbox can still be reused even when
            # this trace was admitted after the ten warm slots. In that case
            # the successful tool path is the readiness evidence and no cold
            # lifecycle interval is fabricated.
            readiness_evidence = "successful_open_shell_tool_no_new_pod"
            lifecycle_status = "ready"
            provisioning_mode = "warm_reused_without_prewarm"
            sandbox_init_ms = 0.0
            sandbox_reuse_ms = 0.0
            sandbox_init_measurement = (
                "existing agent-scoped sandbox reused without synthetic prewarm; "
                "OpenShell tool execution measured separately"
            )
            prewarm_status = "not_requested"
        else:
            readiness_evidence = "sandbox_pod_not_ready"
            lifecycle_status = "not_ready"
            provisioning_mode = "cold_fallback_unobserved"
            sandbox_init_ms = 0.0
            sandbox_reuse_ms = None
            sandbox_init_measurement = "missing sandbox lifecycle evidence"
            prewarm_status = "not_requested"
        lifecycle = {
            "schema": "agentic.trace-replay.sandbox-lifecycle.v1",
            "status": lifecycle_status,
            "prewarm_requested": prewarm_requested,
            "prewarm_status": prewarm_status,
            "prewarm_start_unix": warmup_start or None,
            "prewarm_end_unix": warmup_end or None,
            "prewarm_request_ms": round(warmup_ms, 3) if warmup_ms > 0 else None,
            "sandbox_init_ms": sandbox_init_ms,
            "sandbox_reuse_ms": sandbox_reuse_ms,
            "sandbox_init_measurement": sandbox_init_measurement,
            "provisioning_mode": provisioning_mode,
            "warm_pool_size": warm_pool_size,
            "warm_pool_slot": warm_pool_slot,
            "readiness_evidence": readiness_evidence,
            "workspace": workspace,
            "target_node": expected_node
            or (cluster.node if not allow_cross_node else "cross-node-allowed"),
            "openclaw_node": openclaw_node,
            "created_unix": ready_records[0]["created_unix"] if ready_records else None,
            "ready_unix": ready_records[0]["ready_unix"] if ready_records else None,
            "created_to_ready_ms": created_to_ready_ms,
            "pods": sorted(
                lifecycle_records.values(), key=lambda item: item.get("pod", "")
            ),
        }
        (trace_dir / "data/shell/sandbox_lifecycle.json").write_text(
            json.dumps(lifecycle, indent=2) + "\n", encoding="utf-8"
        )
    if "lifecycle" not in locals():
        # Build lifecycle evidence whenever the collection branch above did
        # not initialize it.  This covers both completed-pod recovery and the
        # normal live result-holder path, while keeping one schema for every
        # trace.
        ready_records = [
            record
            for record in lifecycle_records.values()
            if record.get("ready")
            and record.get("created_unix")
            and record.get("ready_unix")
        ]
        try:
            recovered_summary = json.loads(
                (shell / "summary.json").read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError):
            recovered_summary = {}
        recovered_warmup = recovered_summary.get("warmup") or {}
        recovered_warmup_start = float(recovered_warmup.get("start_unix") or 0.0)
        recovered_warmup_end = float(recovered_warmup.get("end_unix") or 0.0)
        recovered_warmup_ms = float(recovered_warmup.get("duration_ms") or 0.0)
        recovered_warmup_ok = recovered_warmup.get("ok") is True
        recovered_driver_ok = int(recovered_summary.get("ok") or 0) > 0
        recovered_created_to_ready_ms = (
            round(
                max(
                    0.0,
                    ready_records[0]["ready_unix"] - ready_records[0]["created_unix"],
                )
                * 1000.0,
                3,
            )
            if ready_records
            else None
        )
        if prewarm_requested:
            lifecycle_status = (
                "ready" if recovered_warmup_ok or recovered_driver_ok else "not_ready"
            )
            provisioning_mode = (
                "warm_reused"
                if recovered_warmup_ok and not ready_records
                else "cold_pool_fill"
            )
            sandbox_init_ms = 0.0
            sandbox_reuse_ms = None
            sandbox_init_measurement = "completed-pod recovery after prewarm; cold sandbox initialization is outside the measured trace path"
            prewarm_status = "measured" if recovered_warmup_ms > 0 else "missing"
            readiness_evidence = "completed_pod_result_copy_after_prewarm"
        elif warm_pool_slot is not None:
            lifecycle_status = "ready" if recovered_driver_ok else "not_ready"
            provisioning_mode = "warm_reused"
            sandbox_init_ms = 0.0
            sandbox_reuse_ms = None
            sandbox_init_measurement = (
                "completed-pod recovery after a verified ready pool slot"
            )
            prewarm_status = "measured"
            readiness_evidence = "completed_pod_result_copy_after_ready_pool_fill"
        else:
            lifecycle_status = "ready" if recovered_driver_ok else "not_ready"
            provisioning_mode = "cold_fallback"
            sandbox_init_ms = recovered_created_to_ready_ms or 0.0
            sandbox_reuse_ms = None
            sandbox_init_measurement = (
                "Kubernetes sandbox pod create-to-ready after warm-pool exhaustion"
            )
            prewarm_status = "not_requested"
            readiness_evidence = "completed_pod_result_copy_without_prewarm"
        lifecycle = {
            "schema": "agentic.trace-replay.sandbox-lifecycle.v1",
            "status": lifecycle_status,
            "prewarm_requested": prewarm_requested,
            "prewarm_status": prewarm_status,
            "prewarm_start_unix": recovered_warmup_start or None,
            "prewarm_end_unix": recovered_warmup_end or None,
            "prewarm_request_ms": round(recovered_warmup_ms, 3)
            if recovered_warmup_ms > 0
            else None,
            "sandbox_init_ms": sandbox_init_ms,
            "sandbox_reuse_ms": sandbox_reuse_ms,
            "sandbox_init_measurement": sandbox_init_measurement,
            "provisioning_mode": provisioning_mode,
            "warm_pool_size": warm_pool_size,
            "warm_pool_slot": warm_pool_slot,
            "readiness_evidence": readiness_evidence,
            "workspace": workspace,
            "target_node": expected_node
            or (cluster.node if not allow_cross_node else "cross-node-allowed"),
            "openclaw_node": openclaw_node,
            "created_unix": ready_records[0]["created_unix"] if ready_records else None,
            "ready_unix": ready_records[0]["ready_unix"] if ready_records else None,
            "created_to_ready_ms": recovered_created_to_ready_ms,
            "pods": sorted(
                lifecycle_records.values(), key=lambda item: item.get("pod", "")
            ),
        }
        (trace_dir / "data/shell/sandbox_lifecycle.json").write_text(
            json.dumps(lifecycle, indent=2) + "\n", encoding="utf-8"
        )
    if watcher_errors:
        raise RuntimeError(f"sandbox cgroup watcher failed: {watcher_errors[0]}")
    if lifecycle.get("status") != "ready":
        raise RuntimeError(
            f"sandbox lifecycle did not reach Ready for {session_id}: {lifecycle}"
        )
    end = time.time()
    (shell / "terminal.log").write_text(
        cluster.run(["-n", cluster.namespace, "logs", pod, "-c", "driver"], check=False)
    )

    # The mock edge recorder is the authoritative source for model TTFB and
    # response completion boundaries. This is how the established profiler
    # produced the reference waterfall.
    def read_mock_edges() -> str:
        return cluster.run(
            [
                "-n",
                cluster.namespace,
                "exec",
                f"deployment/{mock_deployment}",
                "-c",
                "mock-llm",
                "--",
                "cat",
                "/edges/mock_edges.jsonl",
            ],
            check=False,
        )

    # The recorder is shared by all traces. Serialize only this small read so
    # concurrent oc exec streams cannot race while the recorder is appending.
    if jaeger_io_lock is None:
        mock_edges = read_mock_edges()
    else:
        with jaeger_io_lock:
            mock_edges = read_mock_edges()
    traces = shell / "traces"
    # The worker nodes and the local Mac are not tightly clock-synchronized.
    # Query a bounded window around the client clock, then _rewrite_filtered_jaeger
    # keeps only this session's trace. Without this skew allowance, valid spans
    # can be missed even though Jaeger contains them.
    start_dt = datetime.fromtimestamp(start - 5, timezone.utc)
    # Jaeger ingestion is asynchronous in this deployment. Give the collector
    # a bounded window to make spans visible before treating telemetry as lost.
    export_summary: dict[str, Any] = {}
    for export_attempt in range(1, 4):
        if export_attempt > 1:
            time.sleep(5)

        def export_current_jaeger() -> dict[str, Any]:
            current_jaeger = (
                jaeger_supplier() if jaeger_supplier is not None else jaeger
            )
            return export_jaeger(
                traces,
                start_dt,
                datetime.fromtimestamp(end + 30, timezone.utc),
                jaeger_base=current_jaeger,
                service="openclaw-gateway-shell",
            )

        if jaeger_io_lock is None:
            export_summary = export_current_jaeger()
        else:
            # Jaeger export is the only serialized part of collection.  The
            # trace Jobs, cgroup samplers, backend calls, and local analysis
            # remain concurrent.  Holding this lock prevents one worker from
            # restarting the shared port-forward while another worker is
            # querying it.
            with jaeger_io_lock:
                export_summary = export_current_jaeger()
        if export_summary.get("traces", 0):
            break
        LOG.warning(
            "Jaeger returned no traces yet; retry %d/3 for %s",
            export_attempt,
            session_id,
        )
    if not export_summary.get("traces", 0):
        raise RuntimeError(
            f"Jaeger returned no shell traces for session {session_id} after bounded retries"
        )
    raw_path = traces / "raw_traces.json"
    raw = json.loads(raw_path.read_text(encoding="utf-8")) if raw_path.exists() else []
    if not any(_trace_has_session(trace, session_id) for trace in raw):
        raise RuntimeError(
            f"Jaeger returned traces but none matched session {session_id}"
        )
    driver_start_us = int(start * 1_000_000)
    driver_end_us = int(end * 1_000_000)
    try:
        driver_summary = json.loads(
            (shell / "summary.json").read_text(encoding="utf-8")
        )
        driver_start_us = int(
            float(driver_summary.get("start_unix", start)) * 1_000_000
        )
        driver_end_us = int(float(driver_summary.get("end_unix", end)) * 1_000_000)
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        pass
    matched = _rewrite_filtered_jaeger(
        trace_dir, session_id, driver_start_us, driver_end_us
    )
    if matched == 0:
        raise RuntimeError(
            f"No Jaeger traces matched session/time window for {session_id}"
        )

    # The sidecar log persists across retries. Keep only records that belong to
    # the Jaeger traces selected for this exact driver invocation, rather than
    # accepting all records for the same session in a broad wall-clock window.
    filtered_raw = (
        json.loads(raw_path.read_text(encoding="utf-8")) if raw_path.exists() else []
    )
    current_trace_ids = {str(trace.get("traceID") or "") for trace in filtered_raw}
    session_edges: list[str] = []
    for line in mock_edges.splitlines():
        try:
            edge = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (
            edge.get("session_id") == session_id
            and str(edge.get("trace_id") or "") in current_trace_ids
        ):
            session_edges.append(line)
    if not session_edges:
        raise RuntimeError(
            f"No current mock edges matched Jaeger trace IDs for session {session_id}"
        )
    (shell / "mock_edges.jsonl").write_text(
        "\n".join(session_edges) + "\n", encoding="utf-8"
    )
    lifecycle_path = shell / "sandbox_lifecycle.json"
    try:
        lifecycle = json.loads(lifecycle_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        lifecycle = {}
    if lifecycle.get("provisioning_mode") == "warm_reused" and _conversational_trace(
        trace_dir
    ):
        # Conversational traces (zero recorded tool calls) never execute an
        # OpenShell action, so sandbox-acquisition and OpenShell execution
        # metrics are not applicable. The verified pool-slot readiness
        # evidence is preserved and the analysis layer excludes these traces
        # from tool-stage aggregates. The no-prewarm warm-reuse mode keeps
        # the unchanged validator from demanding a request-to-first-action
        # duration that cannot exist for a conversational trace.
        lifecycle["provisioning_mode"] = "warm_reused_without_prewarm"
        lifecycle["sandbox_request_to_first_action_ms"] = None
        lifecycle["sandbox_request_to_first_action_measurement"] = (
            "not_applicable: conversational trace performed no OpenShell tool execution"
        )
        lifecycle["sandbox_reuse_ms"] = None
        lifecycle["sandbox_reuse_measurement"] = (
            "not_applicable: no OpenShell tool execution"
        )
        lifecycle["openshell_execution"] = "not_applicable"
        lifecycle["conversational_trace"] = True
        lifecycle_path.write_text(
            json.dumps(lifecycle, indent=2) + "\n", encoding="utf-8"
        )
    elif lifecycle.get("provisioning_mode") == "warm_reused":
        first_action_ms = _first_openshell_action_ms(trace_dir)
        if first_action_ms is None or first_action_ms <= 0:
            raise RuntimeError(
                f"warm pool was ready but no first OpenShell action span was captured for {session_id}"
            )
        lifecycle["sandbox_request_to_first_action_ms"] = first_action_ms
        lifecycle["sandbox_request_to_first_action_measurement"] = (
            "first measured openclaw.exec span: OpenClaw request to a ready OpenShell sandbox through completion of the first real agent action"
        )
        lifecycle["sandbox_reuse_ms"] = None
        lifecycle["sandbox_reuse_measurement"] = (
            "gateway telemetry measures request-to-first-action, not lookup alone"
        )
        lifecycle_path.write_text(
            json.dumps(lifecycle, indent=2) + "\n", encoding="utf-8"
        )
    try:
        collect_prometheus(
            shell / "prometheus",
            start=start,
            end=end,
            ns_openclaw=cluster.namespace,
            ns_openshell=openshell_ns,
            openclaw_pod=openclaw_pod,
        )
    except Exception as exc:
        # Prometheus is optional telemetry.  Never turn a collector outage into
        # a failed trace; keep the manifest so downstream analysis and the
        # output gate can report the gap explicitly.
        reason = f"Prometheus collection failed; continuing without metrics: {exc}"
        log_phase(f"telemetry: {reason}")
        write_prometheus_report(
            shell / "prometheus",
            skipped_prometheus_report(
                start=start,
                end=end,
                reason=reason,
                ns_openclaw=cluster.namespace,
                ns_openshell=openshell_ns,
                openclaw_pod=openclaw_pod,
            ),
        )
    profile_v2(
        traces=traces / "raw_traces.json",
        mock_edges=shell / "mock_edges.jsonl",
        prom=shell / "prometheus",
        out=shell / "profile-v2",
        driver_requests=shell / "requests.jsonl",
        sandbox_lifecycle=shell / "sandbox_lifecycle.json",
    )
    analyze_per_turn(traces, shell / "prometheus", trace_dir / "analysis")
    backend_dir = trace_dir / (
        "data/appworld" if backend.benchmark == "appworld" else "data/backend"
    )
    backend_dir.mkdir(parents=True, exist_ok=True)
    session_events = _collect_backend_events(
        cluster,
        backend,
        session_id,
        start,
        current_trace_ids,
        run_key=hashlib.sha256(
            str(trace_dir.parents[1].resolve()).encode("utf-8")
        ).hexdigest()[:32],
        end=end,
    )
    (backend_dir / "events.jsonl").write_text(
        "\n".join(json.dumps(event, separators=(",", ":")) for event in session_events)
        + ("\n" if session_events else ""),
        encoding="utf-8",
    )
    write_backend_calls(session_events, backend_dir / "backend_calls.csv")
    validate_collected_trace(trace_dir, session_id, backend, session_events)
    metadata = {
        "job": job,
        "agent": agent,
        "session_id": session_id,
        "start": start,
        "end": end,
        "target_node": cluster.node if not allow_cross_node else "cross-node-allowed",
        "openclaw_node": openclaw_node,
        "sandbox_nodes": sorted(
            {
                str(record.get("node"))
                for record in lifecycle_records.values()
                if record.get("node")
            }
        ),
    }
    (trace_dir / "trace.json").write_text(json.dumps(metadata, indent=2))
