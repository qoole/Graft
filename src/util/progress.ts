/**
 * Terminal progress renderer for long build phases, plus LLM call accounting.
 *
 * Design constraints learned from real workspace runs at WS_CONCURRENCY=50:
 *
 * - A human glances at the terminal and wants three things: how far along
 *   overall, what is slow right now, did it stall. Everything else is noise.
 * - Every number on the row must be MONOTONE. Aggregate denominators shift as
 *   concurrent children enter/leave phases, so "3456/28865" can drop to
 *   "786/5955" mid-run — unreadable. The row therefore shows cumulative DONE
 *   counts only (files summarized, batches synthesized, nodes cruxed), which
 *   only ever climb.
 * - The row redraws at most every REDRAW_MS — at -j 50 ticks arrive hundreds
 *   of times a second and per-tick repaints are a strobe.
 * - Per-child detail belongs in EVENT lines, not the row: `✓ [12/40] name ·
 *   38s` completions and (non-TTY/verbose) prefixed phase transitions. In a
 *   40-child run, 160 `── summarize` lines are spam, so TTY hides them unless
 *   --verbose.
 * - `note()` (errors, warnings) flushes the row first: a failure can never be
 *   repainted away.
 * - LLM accounting: every chat completion funnels through `llmBegin()`/
 *   `llmEnd()` (installed on the transport in `ai/llm/factory.ts`) → the
 *   `llm:` footer: calls, avg seconds per call, total in-llm, peak in flight.
 *
 * One build per process: module state, not a class.
 */

const REDRAW_MS = 250;
const PULSE_MS = 1000;
const THROBBER = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
const HEARTBEAT_MS = 30_000;

let quiet = false;
let verbose = false;
const tty = process.stderr.isTTY === true;

let dirty = false; // a \r row is on screen
let startedAt = 0;
let lastHeartbeatMs = 0;
let lastDrawMs = 0;
let lastFile = ""; // single-repo: last tick's file, shown on the row
let frameIdx = 0;
let pulse: ReturnType<typeof setInterval> | undefined;

// Cumulative done-counters per activity — the row's core. Only climb.
const cumulative: Record<string, number> = {};
// Per-scope last seen index, per phase — the delta source for `cumulative`.
const lastIndex = new Map<string, number>(); // key = `${scope}\0${phase}`

// LLM accounting
let llmCalls = 0;
let llmMs = 0;
let llmInFlight = 0;
let llmMaxInFlight = 0;
const llmStarts: number[] = []; // start timestamps of calls still in flight

/** Age of the oldest in-flight call — the "is it moving" signal: climbing
 * means the model is still thinking; frozen at 0:00 with N>0 means dead. */
function llmOldest(): number {
  return llmStarts.length > 0 ? Date.now() - Math.min(...llmStarts) : 0;
}

interface ScopeState {
  label: string;
  phase: string;
  index: number;
  total: number;
  file: string;
}

const scopes = new Map<string, ScopeState>(); // key = label ("\0single" for one-repo)

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
  lastDrawMs = 0; // next tick may draw immediately
}

/** One full line to stderr — the only sanctioned way to break the row. */
function fullLine(line: string): void {
  flush();
  write(`${line}\n`);
}

function fmt(n: number): string {
  return n.toLocaleString("en-US");
}

/** Slowest active child by fraction of its current phase. */
function slowest(): string {
  let best: ScopeState | undefined;
  for (const s of scopes.values()) {
    if (s.total <= 0 || !s.phase) continue;
    const frac = (s.index + 1) / s.total;
    if (!best || frac < (best.index + 1) / best.total) best = s;
  }
  if (!best) return "";
  return `${best.label} ${phaseName(best.phase)} ${fmt(best.index + 1)}/${fmt(best.total)}`;
}

/** The row. Two shapes:
 * - single repo: the live phase WITH its denominator (static here, unlike the
 *   workspace's shifting aggregates) + llm in flight + current file;
 * - workspace: cumulative counters + active children + slowest child. */
function rowText(): string {
  const single = scopes.size === 1 ? scopes.get("\0single") : undefined;
  if (single) {
    if (!single.phase) return "starting";
    const head =
      single.total > 0
        ? `${single.phase} ${fmt(single.index + 1)}/${fmt(single.total)}`
        : `${single.phase}`;
    const parts = [head];
    if (llmInFlight > 0) parts.push(`llm ${llmInFlight} in flight · oldest ${secondsLabel(llmOldest())}`);
    return parts.join(" · ");
  }
  const parts: string[] = [`${scopes.size} active`];
  const acts = Object.entries(cumulative).filter(([, n]) => n > 0);
  for (const [phase, n] of acts) parts.push(`${phase} ${fmt(n)}`);
  if (llmInFlight > 0) parts.push(`llm ${llmInFlight} in flight · oldest ${secondsLabel(llmOldest())}`);
  const slow = slowest();
  if (slow) parts.push(`slowest ${slow}`);
  return parts.join(" · ") || "starting";
}

function draw(force = false): void {
  if (quiet) return;
  const now = Date.now();
  if (!force && now - lastDrawMs < REDRAW_MS) return;
  lastDrawMs = now;
  const label = scopes.size === 1 && scopes.has("\0single") && lastFile ? `: ${lastFile.slice(0, 48)}` : "";
  write(`\r\x1b[K${THROBBER[frameIdx]} ${rowText()}${label}`);
  dirty = true;
}

function recordTick(scopeKey: string, phase: string, index: number): void {
  const key = `${scopeKey}\0${phase}`;
  const prev = lastIndex.get(key);
  const delta = prev === undefined ? index + 1 : index - prev;
  lastIndex.set(key, index);
  if (delta > 0) cumulative[phase] = (cumulative[phase] ?? 0) + delta;
}

function heartbeatLine(): string {
  const slow: string[] = [];
  const arr = [...scopes.values()].filter((s) => s.total > 0);
  arr.sort((a, b) => (a.index + 1) / a.total - (b.index + 1) / b.total);
  for (const s of arr.slice(0, 3)) slow.push(`[${s.label}] ${phaseName(s.phase)} ${fmt(s.index + 1)}/${fmt(s.total)}`);
  return `${rowText()}${slow.length ? ` · ${slow.join(" · ")}` : ""}`;
}

export interface ProgressScope {
  phase(rawLabel: string): void;
  tick(index: number, total: number, file?: string): void;
  done(): void;
}

function makeScope(label: string): ProgressScope {
  scopes.set(label, { label, phase: "", index: 0, total: 0, file: "" });
  return {
    phase(rawLabel: string): void {
      const st = scopes.get(label);
      if (!st) return;
      const next = phaseName(rawLabel);
      if (next === st.phase) return;
      st.phase = next;
      st.index = 0;
      st.total = 0;
      if (!tty) fullLine(`[${label}] ── ${next}`);
    },
    tick(index: number, total: number, file?: string): void {
      const st = scopes.get(label);
      if (!st) return;
      st.index = index;
      st.total = total;
      if (file) st.file = file;
      recordTick(label, st.phase, index);
      const now = Date.now();
      if (!tty && now - lastHeartbeatMs >= HEARTBEAT_MS) {
        lastHeartbeatMs = now;
        fullLine(heartbeatLine());
        return;
      }
      draw();
    },
    done(): void {
      scopes.delete(label);
      draw(true);
    },
  };
}

/**
 * Proof-of-life: repaint the row (and advance the throbber) every second
 * whether or not any ticks arrive — slow in-flight calls mean no completions
 * for minutes, and a frozen "oldest <1s" is exactly the lie this module
 * exists to stop. Non-TTY: the heartbeat runs off the pulse too, so a run
 * whose calls all take minutes still logs a line every HEARTBEAT_MS.
 */
function startPulse(): void {
  if (pulse) return;
  pulse = setInterval(() => {
    if (quiet) return;
    frameIdx = (frameIdx + 1) % THROBBER.length;
    const now = Date.now();
    if (tty) {
      if (dirty) draw(true);
    } else if (now - lastHeartbeatMs >= HEARTBEAT_MS) {
      lastHeartbeatMs = now;
      fullLine(heartbeatLine());
    }
  }, PULSE_MS);
  pulse.unref?.();
}

export const progress = {
  configure(opts: { quiet?: boolean; verbose?: boolean }): void {
    quiet = opts.quiet === true;
    verbose = opts.verbose === true;
    startedAt = Date.now();
    startPulse();
  },

  /** A concurrent unit (workspace child). */
  scope(label: string): ProgressScope {
    return makeScope(label);
  },

  /** Single-stream path (single-repo build): implicit lone scope. */
  phase(rawLabel: string): void {
    let st = scopes.get("\0single");
    if (!st) st = scopes.set("\0single", { label: "", phase: "", index: 0, total: 0, file: "" }).get("\0single")!;
    const next = phaseName(rawLabel);
    if (next === st.phase) return;
    st.phase = next;
    st.index = 0;
    st.total = 0;
    // The row already names the phase; under the deep overlap the concept and
    // graph passes alternate every few seconds and transition lines were a
    // strobe. Non-TTY keeps them (greppable log).
  },

  tick(index: number, total: number, file?: string): void {
    const st = scopes.get("\0single");
    if (!st) return;
    st.index = index;
    st.total = total;
    if (file) lastFile = file;
    recordTick("\0single", st.phase, index);
    const now = Date.now();
    if (!tty && now - lastHeartbeatMs >= HEARTBEAT_MS) {
      lastHeartbeatMs = now;
      fullLine(`── ${phaseName(st.phase)} ${fmt(index + 1)}/${fmt(total)}${lastFile ? `: ${lastFile}` : ""}`);
      return;
    }
    draw();
  },

  /** A full stderr line (errors, warnings). Flushes the row first so the
   * next tick can't repaint over it. */
  note(line: string): void {
    flush();
    console.error(line);
  },

  /** Newline if a `\r` row is on screen. Call before any stdout write too,
   * so ✓ lines never append to the row. */
  flush,
  /** Force a row repaint now (bypasses the throttle). */
  redraw: () => draw(true),

  llmBegin(): number {
    llmCalls++;
    llmInFlight++;
    const t0 = Date.now();
    llmStarts.push(t0);
    if (llmInFlight > llmMaxInFlight) llmMaxInFlight = llmInFlight;
    return t0;
  },

  llmEnd(t0: number): void {
    const ms = Date.now() - t0;
    llmInFlight = Math.max(0, llmInFlight - 1);
    const at = llmStarts.indexOf(t0);
    if (at >= 0) llmStarts.splice(at, 1);
    llmMs += ms;
  },

  /** `  timing: ...` footer line, or "" when nothing was measured. */
  timing(): string {
    const total = Date.now() - startedAt;
    if (total <= 0) return "";
    return `  timing: total ${secondsLabel(total)}`;
  },

  /** `  llm: ...` footer line, or "" when no calls were made. */
  llmSummary(): string {
    if (llmCalls === 0) return "";
    const avg = llmMs / llmCalls;
    return (
      `  llm: ${fmt(llmCalls)} calls · avg ${secondsLabel(avg)}/call · ${secondsLabel(llmMs)} in llm ` +
      `(peak ${llmMaxInFlight} in flight)`
    );
  },
};
