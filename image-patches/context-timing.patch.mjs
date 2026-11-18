import fs from "node:fs";
import path from "node:path";

const dist = process.env.OPENCLAW_DIST || "/app/dist";
const builtinPath = fs
  .readdirSync(dist)
  .filter((name) => name.startsWith("builtin-openclaw-") && name.endsWith(".js"))
  .map((name) => path.join(dist, name))
  .find((file) => fs.readFileSync(file, "utf8").includes("function observeEmbeddedAttemptPrompt"));
const runtimePath = fs
  .readdirSync(dist)
  .filter((name) => name.startsWith("runtime-api-") && name.endsWith(".js"))
  .map((name) => path.join(dist, name))
  .find((file) => fs.readFileSync(file, "utf8").includes('spanWithDuration("openclaw.context.assembled"'));

if (!builtinPath || !runtimePath) {
  throw new Error("Could not locate OpenClaw context instrumentation bundles");
}

let builtin = fs.readFileSync(builtinPath, "utf8");
const functionNeedle = "function observeEmbeddedAttemptPrompt(input) {\n\tconst { attempt } = input;";
const patchedFunctionNeedle = "const contextAssemblyStartMs = input.contextAssemblyStartMs ?? Date.now();";
if (!builtin.includes(patchedFunctionNeedle)) {
  if (!builtin.includes(functionNeedle)) throw new Error("Context prompt function shape changed");
  builtin = builtin.replace(
    functionNeedle,
    "function observeEmbeddedAttemptPrompt(input) {\n\tconst contextAssemblyStartMs = input.contextAssemblyStartMs ?? Date.now();\n\tconst { attempt } = input;",
  );
}
const eventNeedle = "trace: freezeDiagnosticTraceContext(createChildDiagnosticTraceContext(input.runTrace))";
if (!builtin.includes(`${eventNeedle},\n\t\tcontextAssemblyStartMs`)) {
  if (!builtin.includes(eventNeedle)) throw new Error("Context diagnostic event shape changed");
  builtin = builtin.replace(eventNeedle, `${eventNeedle},\n\t\tcontextAssemblyStartMs`);
}
const promptCallNeedle = "\t\t\t\tattempt,\n\t\t\t\tcontextTokenBudget:";
if (!builtin.includes("contextAssemblyStartMs: promptStartedAt")) {
  if (!builtin.includes(promptCallNeedle)) throw new Error("prompt assembly call shape changed");
  builtin = builtin.replace(
    promptCallNeedle,
    "\t\t\t\tattempt,\n\t\t\t\tcontextAssemblyStartMs: promptStartedAt,\n\t\t\t\tcontextTokenBudget:",
  );
}
fs.writeFileSync(builtinPath, builtin);

let runtime = fs.readFileSync(runtimePath, "utf8");
const spanNeedle = 'spanWithDuration("openclaw.context.assembled", spanAttrs, 0, {';
// This fixes the run-level initial-prompt span only. Per-model-call context
// latency is measured by joining tool-result and model-endpoint receipt edges.
// spanWithDuration takes an elapsed duration, not a start timestamp. Passing
// Date.now() as its third argument moves the span back to 1970.
const patchedSpanNeedle = 'spanWithDuration("openclaw.context.assembled", spanAttrs, evt.contextAssemblyStartMs == null ? 0 : Math.max(0, evt.ts - evt.contextAssemblyStartMs), {';
if (!runtime.includes(patchedSpanNeedle)) {
  if (!runtime.includes(spanNeedle)) throw new Error("Context span recorder shape changed");
  runtime = runtime.replace(spanNeedle, patchedSpanNeedle);
}
fs.writeFileSync(runtimePath, runtime);

console.log(JSON.stringify({ builtinPath, runtimePath }));
