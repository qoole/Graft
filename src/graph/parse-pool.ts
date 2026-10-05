/**
 * Parent half of the Tier-1 parse pool (`parse-worker.ts` is the worker half).
 *
 * Sizing: `GRAFT_PARSE_CONCURRENCY` overrides the default worker count. `1`
 * means "stay on the main thread" — the caller runs jobs inline and this pool
 * is never spawned, which keeps an escape hatch for debugging and lets the
 * benchmark harness compare the two modes on identical input.
 */
import { availableParallelism } from "node:os";
import { existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { Worker } from "node:worker_threads";
import type { ParseJob, ParseOutcome } from "./parse-worker.js";

export type { ParseJob, ParseOutcome } from "./parse-worker.js";

export const DEFAULT_PARSE_WORKERS = 4;

/**
 * The worker entry sits beside this module — compiled `.js` in `dist/`, `.ts`
 * under tsx. A `new URL` string is opaque to module resolvers (and tsx's .js→.ts
 * remap is version-dependent), so look at what is actually on disk instead of
 * guessing an extension into a string.
 */
function workerEntryUrl(): URL {
  const here = new URL(".", import.meta.url);
  for (const candidate of ["parse-worker.js", "parse-worker.ts"]) {
    const url = new URL(candidate, here);
    if (existsSync(url)) return url;
  }
  throw new Error(
    `parse worker entry not found beside ${fileURLToPath(here)} — run the build first`,
  );
}

/** Worker count for `jobCount` cache-miss files. Never more workers than jobs,
 * never more than CPUs; `1` (or an invalid env value) falls back to inline. */
export function parseWorkerCount(jobCount: number): number {
  const raw = process.env.GRAFT_PARSE_CONCURRENCY;
  const requested = raw === undefined || raw === "" ? DEFAULT_PARSE_WORKERS : Number(raw);
  if (!Number.isFinite(requested) || requested <= 1) return 1;
  const cpus = availableParallelism();
  return Math.max(1, Math.min(Math.floor(requested), cpus, jobCount));
}

export interface ParsedFileResult {
  index: number;
  rel: string;
  outcome: ParseOutcome;
}

/**
 * Parse `jobs` across `parseWorkerCount(jobs.length)` fresh worker threads.
 * Jobs are partitioned strided (round-robin) so languages/mixes balance without
 * a scheduler; each worker streams one result message per file, and the parent
 * collects them into a map keyed by the caller's original file index — merge
 * order is the caller's business, never completion order.
 *
 * Workers are spawned per call and terminated after; a build pays the pool
 * once. A worker that dies mid-partition rejects — callers surface the error
 * rather than silently degrading to a partial parse (correctness over silent
 * fallback).
 */
export async function parseJobsInWorkers(
  jobs: ParseJob[],
  onFileDone?: (rel: string) => void,
): Promise<Map<number, ParsedFileResult>> {
  const workerCount = parseWorkerCount(jobs.length);
  if (workerCount <= 1 || jobs.length === 0) return new Map();

  const results = new Map<number, ParsedFileResult>();
  const workers: Worker[] = [];
  let outstanding = workerCount;

  return await new Promise<Map<number, ParsedFileResult>>((resolveP, rejectP) => {
    for (let w = 0; w < workerCount; w++) {
      const partition = jobs.filter((_, k) => k % workerCount === w);
      const entry = workerEntryUrl();
      // A `.ts` entry only exists under tsx, and the loader must be stated for
      // the worker explicitly — workers don't reliably inherit the parent's
      // `--import` registration across Node versions (CI's Node 20 didn't,
      // which is exactly the failure this fixed).
      const worker = entry.href.endsWith(".ts")
        ? new Worker(entry, { execArgv: ["--import", "tsx"] })
        : new Worker(entry);
      workers.push(worker);
      const fail = (err: Error): void => {
        for (const wk of workers) void wk.terminate();
        rejectP(err);
      };
      worker.on("message", (msg: { type: string; index?: number; rel?: string; outcome?: ParseOutcome }) => {
        if (msg.type === "result" && msg.index !== undefined && msg.rel !== undefined && msg.outcome) {
          results.set(msg.index, { index: msg.index, rel: msg.rel, outcome: msg.outcome });
          onFileDone?.(msg.rel);
        } else if (msg.type === "done") {
          outstanding--;
          worker.unref?.();
          if (outstanding === 0) resolveP(results);
        }
      });
      worker.on("error", fail);
      worker.on("exit", (code) => {
        if (code !== 0) fail(new Error(`parse worker exited with code ${code}`));
      });
      worker.postMessage({ type: "parse", jobs: partition });
    }
  });
}
