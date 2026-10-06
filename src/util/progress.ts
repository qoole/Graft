/**
 * Terminal progress renderer for long build phases, plus LLM call accounting.
 *
 * The old build output was two unconditional `\r` writers (cli.ts) plus a
 * console.error per synthesis batch — unobservable through a pipe, child-
 * anonymous in a workspace, and it repainted over its own errors. This module
 * is the single output path for build chatter:
 *
 * - TTY: one self-overwriting row; last tick wins across overlapping phases
 *   (the deep overlap runs two at once by design).
 * - non-TTY: heartbeat — a full line every HEARTBEAT_MS or HEARTBEAT_TICKS
 *   units, so a piped/CI run shows steady, greppable progress.
 * - `note()` (errors, warnings) flushes the dirty row first, so a failure can
 *   never be repainted away.
 * - Phase transitions are short full lines; spans accumulate per phase.
 * - `llmBegin()`/`llmEnd()` bracket every chat-completion call (installed by
 *   wrapping the transport in `ai/llm/factory.ts`), so the ✓ block can report
 *   call count, average seconds per call, and the share of wallclock spent
 *   inside the LLM — per phase and overall.
 *
 * One build per process: module state, not a class.
 */

const HEARTBEAT_MS = 30_000;
const HEARTBEAT_TICKS = 100;

let quiet = false;
let verbose = false;
const tty = process.stderr.isTTY === true;

let scopeLabel = ""; // "[2/40 ZCode]" — workspace child scope
let currentPhase = ""; // "parse" | "summarize" | ...
let phaseStart = 0;
const phaseDurations: Record<string, number> = {};
const phaseLlmMs: Record<string, number> = {};
let dirty = false; // a \r row is on screen
let lastHeartbeatMs = 0;
let lastHeartbeatIndex = 0;
let startedAt = 0;

// LLM accounting
let llmCalls = 0;
let llmMs = 0;
let llmInFlight = 0;
let llmMaxInFlight = 0;

/** Display order for the timing summary; unlisted phases go last. */
const PHASE_ORDER = ["parse", "summarize", "synthesize", "crux"];

function write(s: string): void {
  process.stderr.write(s);
}

function secondsLabel(ms: number): string {
  if (ms < 1000) return "<1s";
  if (ms < 60_000) return `${Math.round(ms / 1000)}s`;
  const m = Math.floor(ms / 60_000);
  const s = Math.round((ms % 60_000) / 1000);
  return `${m}m${s > 0 ? `${s}s` : ""}`;
}

function prefix(): string {
  return scopeLabel ? `${scopeLabel} ` : "";
}

function phaseName(p: string): string {
  if (p === "enrich") return "crux";
  return p;
}

function drawRow(index: number, total: number, file: string): void {
  const filePart = tty ? `: ${file.slice(0, 40).padEnd(40)}` : `: ${file}`;
  write(`\r\x1b[K${prefix()}${currentPhase} ${index + 1}/${total}${filePart}`);
  dirty = true;
}

export const progress = {
  /** quiet: ticks and phase lines suppressed (results + errors only). */
  configure(opts: { quiet?: boolean; verbose?: boolean }): void {
    quiet = opts.quiet === true;
    verbose = opts.verbose === true;
    startedAt = Date.now();
  },

  /** Workspace child scope — every line carries it until changed. */
  setScope(label: string): void {
    this.flush();
    scopeLabel = label;
  },

  /** Phase transition. Records the previous phase's span; prints a short
   * transition line unless quiet. Repeating the label is a no-op, so callers
   * can hand every onProgress event here. */
  phase(rawLabel: string): void {
    const label = phaseName(rawLabel);
    if (label === currentPhase) return;
    const now = Date.now();
    if (currentPhase && phaseStart) {
      phaseDurations[currentPhase] = (phaseDurations[currentPhase] ?? 0) + (now - phaseStart);
    }
    currentPhase = label;
    phaseStart = now;
    // Heartbeat spacing is per-phase: without this, a phase that ends near the
    // 100-tick mark swallows the next phase's early ticks entirely.
    lastHeartbeatIndex = 0;
    if (quiet) return;
    this.flush();
    write(`${prefix()}── ${label}\n`);
    dirty = false;
  },

  /** Progress tick. TTY redraws the row; non-TTY heartbeats (one full line
   * every HEARTBEAT_MS or HEARTBEAT_TICKS units). quiet: no-op. */
  tick(index: number, total: number, file?: string): void {
    if (quiet) return;
    const now = Date.now();
    if (!tty) {
      const dueMs = now - lastHeartbeatMs >= HEARTBEAT_MS;
      const dueTicks = index - lastHeartbeatIndex >= HEARTBEAT_TICKS;
      if (!dueMs && !dueTicks) return;
      lastHeartbeatMs = now;
      lastHeartbeatIndex = index;
      write(`${prefix()}${currentPhase} ${index + 1}/${total}: ${file ?? ""}\n`);
      return;
    }
    drawRow(index, total, file ?? "");
  },

  /** A full stderr line (errors, warnings). Flushes the dirty row first so
   * the next tick can't repaint over it. Goes through console.error so tests
   * that capture stderr via the console see these lines. */
  note(line: string): void {
    this.flush();
    console.error(line);
  },

  /** Newline if a `\r` row is on screen. Call before any direct stderr write. */
  flush(): void {
    if (!dirty) return;
    write("\n");
    dirty = false;
  },

  /** Bracket one chat-completion call. `t0` = Date.now() at call start. */
  llmBegin(): number {
    llmCalls++;
    llmInFlight++;
    if (llmInFlight > llmMaxInFlight) llmMaxInFlight = llmInFlight;
    return Date.now();
  },

  llmEnd(t0: number): void {
    const ms = Date.now() - t0;
    llmInFlight = Math.max(0, llmInFlight - 1);
    llmMs += ms;
    phaseLlmMs[currentPhase] = (phaseLlmMs[currentPhase] ?? 0) + ms;
  },

  /** Seconds elapsed since configure(). */
  elapsedMs(): number {
    return Date.now() - startedAt;
  },

  /** `── timing` block line, or "" when nothing was measured. */
  timing(): string {
    const parts: string[] = [];
    const entries = Object.entries(phaseDurations);
    entries.sort((a, b) => {
      const ia = PHASE_ORDER.indexOf(a[0]);
      const ib = PHASE_ORDER.indexOf(b[0]);
      return (ia === -1 ? 99 : ia) - (ib === -1 ? 99 : ib);
    });
    for (const [k, ms] of entries) {
      const llm = phaseLlmMs[k];
      parts.push(llm && llm > 500 ? `${k} ${secondsLabel(ms)} (llm ${secondsLabel(llm)})` : `${k} ${secondsLabel(ms)}`);
    }
    if (parts.length === 0) return "";
    return `  timing: ${parts.join(" · ")} · total ${secondsLabel(Date.now() - startedAt)}`;
  },

  /** `── llm` block line, or "" when no calls were made. */
  llmSummary(): string {
    if (llmCalls === 0) return "";
    const avg = llmMs / llmCalls;
    return (
      `  llm: ${llmCalls} calls · avg ${secondsLabel(avg)}/call · ${secondsLabel(llmMs)} in llm ` +
      `(peak ${llmMaxInFlight} in flight)`
    );
  },
};
