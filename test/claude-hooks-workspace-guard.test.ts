/**
 * Tests for the prompt hook's workspace-parent guard.
 *
 * A session opened at a workspace parent (graft/workspace.json) federates
 * `ask` across every child repo — minutes of CPU per prompt, past any hook
 * budget, with the pack silently dropped at `if (!ask) return`. The guard
 * skips retrieval there entirely; these tests pin both directions: no child
 * spawn under the guard, and the ask still fires (with --no-refresh) in a
 * plain repo.
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import { chmodSync, existsSync, mkdtempSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { main } from "../src/claude/hooks.js";

/** JS stub the hook's `graft ask` child resolves via GRAFT_TEST_CLI; records
 * its argv into a sentinel file so the test can observe both whether the
 * child ran and with which flags. */
function writeAskStub(dir: string, sentinel: string): string {
  const stub = join(dir, "ask-stub.cjs");
  writeFileSync(
    stub,
    "const fs = require('fs');\n" +
      `fs.writeFileSync(${JSON.stringify(sentinel)}, JSON.stringify(process.argv));\n` +
      "console.log(JSON.stringify({ query: 'stub', mode: 'lexical', hits: [] }));\n",
  );
  chmodSync(stub, 0o755);
  return stub;
}

async function runPrompt(dir: string, prompt: string): Promise<void> {
  process.env.CLAUDE_PROJECT_DIR = dir;
  process.env.GRAFT_TEST_STDIN = JSON.stringify({ prompt, session_id: "guard-test" });
  try {
    await main("prompt");
  } finally {
    delete process.env.GRAFT_TEST_STDIN;
    delete process.env.CLAUDE_PROJECT_DIR;
    delete process.env.GRAFT_TEST_CLI;
  }
}

const PROMPT = "how does the deep lane select repositories today";

test("prompt hook skips graft ask on a workspace parent", async () => {
  const d = mkdtempSync(join(tmpdir(), "graft-guard-"));
  try {
    const sentinel = join(d, "sentinel-must-not-exist");
    process.env.GRAFT_TEST_CLI = writeAskStub(d, sentinel);
    mkdirSync(join(d, "graft"), { recursive: true });
    writeFileSync(join(d, "graft", "workspace.json"), JSON.stringify({ version: 1, children: ["repoA"] }));
    await runPrompt(d, PROMPT);
    assert.equal(existsSync(sentinel), false, "workspace parent must not spawn a federated ask child");
  } finally {
    rmSync(d, { recursive: true, force: true });
  }
});

test("prompt hook still asks (with --no-refresh) in a plain repo", async () => {
  const d = mkdtempSync(join(tmpdir(), "graft-guard-"));
  try {
    const sentinel = join(d, "sentinel");
    process.env.GRAFT_TEST_CLI = writeAskStub(d, sentinel);
    mkdirSync(join(d, "graft"), { recursive: true });
    await runPrompt(d, PROMPT);
    assert.equal(existsSync(sentinel), true, "plain repo must still spawn the ask child");
    const argv: string[] = JSON.parse(readFileSync(sentinel, "utf8"));
    assert.ok(argv.includes("--no-refresh"), `expected --no-refresh in argv, got ${argv.join(" ")}`);
  } finally {
    rmSync(d, { recursive: true, force: true });
  }
});
