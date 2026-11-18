#!/usr/bin/env node
// Rewrite only recorded SWE bash commands after OpenClaw has selected exec.
// Other SSH operations, including workspace transfer, pass through unchanged.
const crypto = require('node:crypto');
const fs = require('node:fs');
const path = require('node:path');
const { spawnSync } = require('node:child_process');

function splitShellWords(input) {
  const words = [];
  let word = '';
  let quote = '';
  let started = false;
  for (let i = 0; i < input.length; i++) {
    const char = input[i];
    if (quote === "'") {
      if (char === "'") quote = '';
      else word += char;
    } else if (quote === '"') {
      if (char === '"') quote = '';
      else if (char === '\\' && i + 1 < input.length) word += input[++i];
      else word += char;
    } else if (char === "'" || char === '"') {
      quote = char;
      started = true;
    } else if (char === '\\' && i + 1 < input.length) {
      word += input[++i];
      started = true;
    } else if (/\s/.test(char)) {
      if (started) { words.push(word); word = ''; started = false; }
    } else {
      word += char;
      started = true;
    }
  }
  if (quote) throw new Error('unterminated SSH command quote');
  if (started) words.push(word);
  return words;
}

function shellQuote(value) {
  return "'" + value.replace(/'/g, "'\\''") + "'";
}

function commandParts(remote) {
  const words = splitShellWords(remote);
  const shell = words.findIndex((word, i) => word === '/bin/sh' && words[i + 1] === '-c');
  if (shell < 0 || shell + 2 >= words.length) return null;
  const body = words[shell + 2];
  if (body.startsWith('cd ')) {
    const separator = body.indexOf(' && ');
    if (separator < 0) return null;
    return { words, bodyIndex: shell + 2, prefix: body.slice(0, separator + 4), command: body.slice(separator + 4) };
  }
  return { words, bodyIndex: shell + 2, prefix: '', command: body };
}

function claimScript(hash, scripts, stateRoot) {
  fs.mkdirSync(stateRoot, { recursive: true, mode: 0o700 });
  const lock = path.join(stateRoot, hash + '.lock');
  const countFile = path.join(stateRoot, hash + '.count');
  for (let attempt = 0; attempt < 100; attempt++) {
    try { fs.mkdirSync(lock); break; }
    catch (error) {
      if (error.code !== 'EEXIST' || attempt === 99) throw error;
      Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, 10);
    }
  }
  try {
    const index = fs.existsSync(countFile) ? Number(fs.readFileSync(countFile, 'utf8')) : 0;
    if (!Number.isSafeInteger(index) || index < 0 || index >= scripts.length) {
      throw new Error('SWE command occurrence exceeds recorded trajectory');
    }
    fs.writeFileSync(countFile, String(index + 1), { mode: 0o600 });
    return scripts[index];
  } finally {
    fs.rmdirSync(lock);
  }
}

function rewriteRemote(remote, commandMap, stateRoot) {
  const parts = commandParts(remote);
  if (!parts) return remote;
  const hash = crypto.createHash('sha256').update(parts.command).digest('hex');
  const scripts = commandMap[hash];
  if (!scripts) return remote;
  const script = claimScript(hash, scripts, stateRoot);
  if (!/^\/tmp\/trace-replay-tool-calls\/[0-9]{4}-[0-9]{3}\.py$/.test(script)) {
    throw new Error('invalid SWE adapter script path');
  }
  parts.words[parts.bodyIndex] = parts.prefix + 'python3 ' + shellQuote(script);
  return parts.words.map(shellQuote).join(' ');
}

if (require.main === module) {
  const args = process.argv.slice(2);
  const map = JSON.parse(Buffer.from('__SWE_MAP_BASE64__', 'base64').toString('utf8'));
  if (args.includes('-F') && Object.keys(map).length && args.length) {
    try { args[args.length - 1] = rewriteRemote(args[args.length - 1], map, '/tmp/trace-replay-swe-claims'); }
    catch (error) { process.stderr.write('SWE SSH adapter: ' + error.message + '\n'); process.exit(75); }
  }
  const result = spawnSync('/opt/openshell/bin/ssh-real', args, { stdio: 'inherit' });
  process.exit(result.status ?? 1);
}

module.exports = { splitShellWords, commandParts, rewriteRemote };
