import shutil
import subprocess
from pathlib import Path

import pytest


def test_ssh_boundary_maps_recorded_command_after_openclaw(tmp_path: Path):
    if not shutil.which("node"):
        pytest.skip("Node.js is unavailable")
    adapter = Path(__file__).parents[1] / "scripts/swe_ssh_adapter.cjs"
    javascript = r"""
const assert = require('node:assert/strict');
const crypto = require('node:crypto');
const adapter = require(process.argv[1]);
const state = process.argv[2];
const command = 'find /testbed -name "*.py" | head -3';
const quote = value => "'" + value.replaceAll("'", "'\"'\"'") + "'";
const remote = ['/bin/sh', '-c', "cd '/sandbox/openclaw' && " + command].map(quote).join(' ');
const digest = crypto.createHash('sha256').update(command).digest('hex');
const mapping = {[digest]: [
  '/tmp/trace-replay-tool-calls/0000-000.py',
  '/tmp/trace-replay-tool-calls/0001-000.py',
]};
assert.equal(adapter.rewriteRemote(remote, {}, state), remote);
assert.equal(adapter.commandParts(adapter.rewriteRemote(remote, mapping, state)).command,
             "python3 '/tmp/trace-replay-tool-calls/0000-000.py'");
assert.equal(adapter.commandParts(adapter.rewriteRemote(remote, mapping, state)).command,
             "python3 '/tmp/trace-replay-tool-calls/0001-000.py'");
assert.throws(() => adapter.rewriteRemote(remote, mapping, state),
              /exceeds recorded trajectory/);
"""
    result = subprocess.run(
        ["node", "-e", javascript, str(adapter), str(tmp_path / "claims")],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
