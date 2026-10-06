/**
 * Terminal progress renderer for long build phases, plus LLM call accounting.
 *
 * One output path for build chatter. Two regimes:
 *
 * - Single repo: one scope, one phase stream — TTY draws a self-overwriting
 *   row, a pipe gets heartbeats.
 * - Workspace: children build CONCURRENTLY, so a single last-tick-wins row is
 *   a lie (labels from one child, counters from another). Each child takes a
 *   {@link scope} handle; ticks update that child's entry, and the row shows
 *   AGGREGATES across children (per-phase sums + slowest child + llm in
 *   flight). Per-child detail appears only where it's readable: transition
 *   lines and heartbeats carry the scope prefix, completions print their own
 *   ✓ line.
 *
 * `note()` (errors, warnings) flushes the row first, so a failure can never
 * be repainted away. Every chat completion funnels through `llmBegin()`/
 * `llmEnd()` (installed on the transport in `ai/llm/factory.ts`) to feed the
 * `llm:` summary line: calls, avg seconds per call, total, peak in flight.
 *
 * One build per process: module state, not a class.
 */

const HEARTBEAT_MS = 30_000;
const HEARTBEAT_TICKS = 100;
const MAX_LAGGING_SHOWN = 3;

let quiet = false;
const tty = process.stderr.isTTY === true;

let dirty = false; // a \r row is on screen
let startedAt = 0;
let lastHeartbeatMs = 0;

// LLM accounting
let llmCalls = 0;
let llmMs = 0;
let llmInFlight = 0;
let llmMaxInFlight = 0;

interface ScopeState {
  label: string;
  phase: string;
  index: number;
  total: number;
}

const scopes = new Map<string, ScopeState>(); // key = label

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

function phaseName(p: string): string {
  return p === "enrich" ? "crux" : p;
}

function flush(): void {
  if (!dirty) return;
  write("\n");
  dirty = false;
}

/** One full line to stderr — the only sanctioned way to break the row. */
function fullLine(line: string): void {
  flush();
  write(`${line}\n`);
}

/** Aggregate across active scopes, grouped by phase:
 * `summarize 3456/21000 (12) · synthesize 40/300 (3)`, plus the slowest
 * children by fraction remaining. */
function aggregate(exclude = ""): string {
  const byPhase = new Map<string, { done: number; total: number; count: number }>();
  const lagging: { label: string; frac: number; index: number; total: number; phase: string }[] = [];
  for (const s of scopes.values()) {
    if (s.total <= 0) continue;
    const g = byPhase.get(s.phase) ?? { done: 0, total: 0, count: 0 };
    g.done += s.index + 1;
    g.total += s.total;
    g.count++;
    byPhase.set(s.phase, g);
    lagging.push({ label: s.label, frac: (s.index + 1) / s.total, index: s.index, total: s.total, phase: s.phase });
  }
  const parts = [...byPhase.entries()]
    .sort((a, b) => b[1].total - a[1].total)
    .slice(0, 3)
    .map(([phase, g]) => `${phase} ${g.done}/${g.total} (${g.count})`);
  lagging.sort((a, b) => a.frac - b.frac);
  const slow = lagging.filter((l) => l.label !== exclude).slice(0, MAX_LAGGING_SHOWN);
  const slowPart = slow
    .map((l) => `${l.label} ${l.index + 1}/${l.total}`)
    .join(" · ");
  const llmPart = llmInFlight > 0 ? ` · llm ${llmInFlight}` : "";
  const base = parts.length > 0 ? parts.join(" · ") : "starting";
  return slowPart ? `${base} · slowest ${slowPart}${llmPart}` : `${base}${llmPart}`;
}

function drawRow(): void {
  const label = scopes.size === 1 ? [...scopes.keys()][0] + " " : "";
  write(`\r\x1b[K${label}${aggregate(scopes.size === 1 ? [...scopes.keys()][0] : "")}`);
  dirty = true;
}

export interface ProgressScope {
  /** Phase transition for this child — one prefixed line. */
  phase(rawLabel: string): void;
  /** Progress tick — updates this child's entry; drives the aggregate row. */
  tick(index: number, total: number, file?: string): void;
  /** Child finished — drop it from the active set. */
  done(): void;
}

export const progress = {
  configure(opts: { quiet?: boolean; verbose?: boolean }): void {
    quiet = opts.quiet === true;
    startedAt = Date.now();
  },

  /** A concurrent unit (workspace child). Its ticks feed the aggregate row;
   * its phase transitions and heartbeats carry its own prefix. */
  scope(label: string): ProgressScope {
    scopes.set(label, { label, phase: "", index: 0, total: 0 });
    return {
      phase(rawLabel: string): void {
        const st = scopes.get(label);
        if (!st) return;
        const next = phaseName(rawLabel);
        if (next === st.phase) return;
        st.phase = next;
        st.index = 0;
        st.total = 0;
        if (!quiet) fullLine(`[${label}] ── ${next}`);
      },
      tick(index: number, total: number, _file?: string): void {
        const st = scopes.get(label);
        if (!st) return;
        st.index = index;
        st.total = total;
        if (quiet) return;
        const now = Date.now();
        if (!tty) {
          if (now - lastHeartbeatMs < HEARTBEAT_MS && index - lastHeartbeatIndexFor(label) < HEARTBEAT_TICKS) return;
          lastHeartbeatMs = now;
          noteHeartbeatIndex(label, index);
          fullLine(`[${label}] ${aggregate()}`);
          return;
        }
        drawRow();
      },
      done(): void {
        scopes.delete(label);
        if (scopes.size === 0) flush();
      },
    };
  },

  /** Single-stream path (single-repo build): implicit lone scope. */
  phase(rawLabel: string): void {
    const label = "\0single";
    let st = scopes.get(label);
    if (!st) st = scopes.set(label, { label: "", phase: "", index: 0, total: 0 }).get(label)!;
    const next = phaseName(rawLabel);
    if (next === st.phase) return;
    st.phase = next;
    st.index = 0;
    st.total = 0;
    if (!quiet) fullLine(`── ${next}`);
  },

  tick(index: number, total: number, file?: string): void {
    if (quiet) return;
    const now = Date.now();
    if (!tty) {
      if (now - lastHeartbeatMs < HEARTBEAT_MS && index - lastHeartbeatIndexFor("\0single") < HEARTBEAT_TICKS) return;
      lastHeartbeatMs = now;
      noteHeartbeatIndex("\0single", index);
      const st = scopes.get("\0single");
      fullLine(`── ${st?.phase ?? "build"} ${index + 1}/${total}${file ? `: ${file}` : ""}`);
      return;
    }
    const st = scopes.get("\0single");
    if (st) {
      st.index = index;
      st.total = total;
    }
    drawRow();
  },

  /** A full stderr line (errors, warnings). Flushes the row first so the
   * next tick can't repaint over it. console.error so stderr-capturing
   * tests still see it. */
  note(line: string): void {
    flush();
    console.error(line);
  },

  /** Newline if a `\r` row is on screen. Call before any direct stderr write. */
  flush,

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
  },

  /** `  timing: ...` line, or "" when nothing was measured. */
  timing(): string {
    const total = Date.now() - startedAt;
    if (total <= 0) return "";
    return `  timing: total ${secondsLabel(total)}`;
  },

  /** `  llm: ...` line, or "" when no calls were made. */
  llmSummary(): string {
    if (llmCalls === 0) return "";
    const avg = llmMs / llmCalls;
    return (
      `  llm: ${llmCalls} calls · avg ${secondsLabel(avg)}/call · ${secondsLabel(llmMs)} in llm ` +
      `(peak ${llmMaxInFlight} in flight)`
    );
  },
};

// Per-scope heartbeat counters (tick spacing is per child, not global).
const heartbeatIndex = new Map<string, number>();
function lastHeartbeatIndexFor(key: string): number {
  return heartbeatIndex.get(key) ?? Number.NEGATIVE_INFINITY;
}
function noteHeartbeatIndex(key: string, index: number): void {
  heartbeatIndex.set(key, index);
}
