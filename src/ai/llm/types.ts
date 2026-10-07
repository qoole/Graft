/**
 * Provider-neutral chat transport.
 *
 * Every LLM call in graft — the engine's single-shot summarize/synthesize/crux
 * ops, and any multi-turn tool-use loop built on top — goes through one interface,
 * {@link ChatModel}. Adapters translate this neutral shape to a concrete SDK
 * (OpenAI-compatible or native Anthropic). Nothing above this layer knows which
 * provider is in play, so adding a provider is a new adapter, never a change to
 * a call site.
 *
 * The design carries three affordances that let a single interface serve the
 * multi-turn loop as cleanly as a one-shot call:
 *   - {@link ChatResponse.assistant} — a ready-to-append assistant message, so a
 *     caller continues a conversation without touching provider shapes.
 *   - {@link Message.providerRaw} — opaque verbatim replay of a message the same
 *     adapter produced (preserves Anthropic thinking-block signatures, which must
 *     be echoed back unchanged).
 *   - {@link Usage} normalized to uncached-input, so token accounting is identical
 *     across providers.
 */

/** A single tool invocation requested by the model. `args` is ALREADY parsed. */
export interface ToolCall {
  id: string;
  name: string;
  /** Parsed argument object. Adapters (de)serialize to their wire format. */
  args: unknown;
}

/** One turn in a conversation, in provider-neutral form. */
export interface Message {
  role: "system" | "user" | "assistant" | "tool";
  /** Plain-text content. Empty string when a turn is purely tool calls/results. */
  content: string;
  /** Assistant turns only: tool calls the model asked for. */
  toolCalls?: ToolCall[];
  /** Tool turns only: the id of the {@link ToolCall} this result answers. */
  toolCallId?: string;
  /**
   * Request a provider cache marker (`cache_control: ephemeral`) on THIS message.
   * The transport only understands this per-message flag; any sliding/rotation
   * policy lives in the caller.
   */
  cacheBreakpoint?: boolean;
  /**
   * Opaque, provider-tagged payload from the adapter that produced this message.
   * When the SAME adapter replays it, the original wire form is sent verbatim
   * (preserving e.g. Anthropic thinking blocks + signatures). A different adapter
   * ignores it and reconstructs from {@link content}/{@link toolCalls}.
   */
  providerRaw?: { provider: string; raw: unknown };
}

/** A tool the model may call. `parameters` is a JSON Schema object. */
export interface ToolSpec {
  name: string;
  description: string;
  parameters: Record<string, unknown>;
}

/**
 * How the model must shape its reply:
 *   - `text`  — free-form assistant text.
 *   - `json`  — a JSON object (mapped to a forced synthetic tool on both providers,
 *               since that is the one structured-output mechanism they share).
 *   - `tool`  — force a call to the named tool (structured output with a schema).
 */
export type ResponseFormat =
  | { kind: "text" }
  | { kind: "json" }
  | { kind: "tool"; name: string };

/**
 * Token accounting, normalized across providers. `input` is UNCACHED input only
 * (OpenAI reports cached tokens inside `prompt_tokens`; Anthropic reports them
 * separately — adapters reconcile so callers can sum these directly).
 */
export interface Usage {
  input: number;
  output: number;
  cacheRead: number;
  cacheCreate: number;
}

import { createRequire } from "node:module";

export interface ChatRequest {
  messages: Message[];
  tools?: ToolSpec[];
  /** Default: `{ kind: "text" }`. */
  responseFormat?: ResponseFormat;
  /** Providers that reject it (current Anthropic models) drop it in the adapter. */
  temperature?: number;
  /** Max output tokens. Required by Anthropic; adapters supply a default. */
  maxTokens?: number;
  /** Per-request transport timeout in ms — overrides the transport default.
   * Synthesis batches (a whole char-budget of summaries per call) set this
   * higher than per-file calls: GRAFT_SYNTH_TIMEOUT, default 600s. */
  timeoutMs?: number;
}

export interface ChatResponse {
  /** Concatenated assistant text (empty when the turn only made tool calls). */
  text: string;
  toolCalls: ToolCall[];
  usage: Usage;
  stopReason: string | null;
  /** The assistant turn as a neutral message — push straight back into `messages`. */
  assistant: Message;
}

/**
 * A configured connection to one model. `label` is a stable `<provider>:<model>`
 * string recorded in the graph manifest.
 */
export interface ChatModel {
  readonly label: string;
  create(req: ChatRequest): Promise<ChatResponse>;
}

/**
 * How many times the transport retries a failed request before the error reaches
 * the caller. Both SDKs retry only what is worth retrying (429 and 5xx, honouring
 * `Retry-After`) with exponential backoff, so this is the knob that separates a
 * transient rate limit from a real failure — #127 lost whole files to the first
 * 429 a shared gateway threw. Env-overridable because a metered corporate gateway
 * may want fewer, and a flaky local proxy more.
 */
export function transportRetries(): number {
  const raw = Number(process.env.GRAFT_LLM_RETRIES);
  return Number.isFinite(raw) && raw >= 0 ? Math.floor(raw) : 4;
}

/**
 * Per-ATTEMPT timeout in ms. Without it the SDKs sit at their 10-minute
 * default, and one hung connection through the retry ladder reads as a call
 * that is "16 minutes old and climbing" — minutes of wallclock spent waiting
 * on a socket that stopped answering. A reasoning model on a full synthesis
 * batch needs real minutes, so the default is generous (300s); dial it down
 * (and GRAFT_LLM_RETRIES with it) when your endpoint fails fast instead.
 */
export function transportTimeoutMs(): number {
  const raw = Number(process.env.GRAFT_LLM_TIMEOUT);
  return Number.isFinite(raw) && raw > 0 ? Math.floor(raw) * 1000 : 300_000;
}

/**
 * undici dispatcher with the head/body timeouts DISABLED: the SDK's own
 * `timeout` (transportTimeoutMs / per-request timeoutMs) must be the single
 * canceller. undici's default headersTimeout is 5 minutes, which silently
 * truncated our longer budgets — the SDK timer believed it had minutes left
 * while the dispatcher had already killed the socket ("configure a matching
 * undici fetch ... headersTimeout" is the SDK saying exactly this).
 */
let undici: { fetch: unknown; dispatcher: unknown } | undefined;
/** The undici fetch + a no-head-timeout Agent, from ONE undici copy. Cast at
 * the call site: openai bundles its own undici-types, so the real instances
 * and the expected types are different declarations of the same shape — and
 * mixing copies (Agent from one, fetch from Node's built-in) makes every
 * request die as "Connection error." */
export function llmUndici(): { fetch: unknown; dispatcher: unknown } {
  if (!undici) {
    const u = createRequire(import.meta.url)("undici") as typeof import("undici");
    undici = { fetch: u.fetch, dispatcher: new u.Agent({ headersTimeout: 0, bodyTimeout: 0 }) };
  }
  return undici;
}

