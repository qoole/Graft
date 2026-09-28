/**
 * An OpenAI-compatible endpoint that does not behave like the spec: a 200 whose
 * body is an error object (or has no `choices` at all), and a forced
 * `tool_choice` that comes back as something other than a tool call.
 *
 * Network-free — every test drives a STUB client that replays the exact bodies
 * such an endpoint produces, and records what went out on the wire, so the
 * recovery ladder is asserted step by step.
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import type OpenAI from "openai";
import { OpenAIChatModel } from "../src/ai/llm/openai.js";
import type { ChatRequest } from "../src/ai/llm/types.js";

/** A stub client that answers each call with the next queued body. */
function stubClient(...bodies: unknown[]): { client: OpenAI; calls: any[] } {
  const calls: any[] = [];
  const client = {
    chat: {
      completions: {
        create: async (params: any) => {
          calls.push(params);
          const body = bodies[Math.min(calls.length - 1, bodies.length - 1)];
          if (body instanceof Error) throw body;
          return body;
        },
      },
    },
  } as unknown as OpenAI;
  return { client, calls };
}

/** No-op backoff, recording the waits so the retry ladder is observable. */
function instantSleep(): { sleep: (ms: number) => Promise<void>; waits: number[] } {
  const waits: number[] = [];
  return { sleep: async (ms) => void waits.push(ms), waits };
}

/** The 200-with-an-error-body an unhappy gateway answers with. */
function erroredBody(message = "Provider returned an empty response", code: string | number = 502) {
  return { error: { message, code } };
}

function textBody(content: string) {
  return {
    choices: [{ message: { content, tool_calls: [] }, finish_reason: "stop" }],
    usage: { prompt_tokens: 10, completion_tokens: 5 },
  };
}

function toolBody(name: string, args: unknown) {
  return {
    choices: [
      {
        message: {
          content: null,
          tool_calls: [{ id: "c1", type: "function", function: { name, arguments: JSON.stringify(args) } }],
        },
        finish_reason: "tool_calls",
      },
    ],
    usage: { prompt_tokens: 10, completion_tokens: 5 },
  };
}

const TOOL_REQ: ChatRequest = {
  messages: [{ role: "system", content: "sys" }, { role: "user", content: "go" }],
  tools: [{ name: "record_graph", description: "d", parameters: { type: "object" } }],
  responseFormat: { kind: "tool", name: "record_graph" },
};

/** Capture console.error so the one-off notices neither leak nor collide. */
async function withCapturedError<T>(fn: () => Promise<T>): Promise<{ result: T; err: string[] }> {
  const err: string[] = [];
  const orig = console.error;
  console.error = (...args: unknown[]) => void err.push(args.map(String).join(" "));
  try {
    return { result: await fn(), err };
  } finally {
    console.error = orig;
  }
}

async function withRetries<T>(value: string, fn: () => Promise<T>): Promise<T> {
  const orig = process.env.GRAFT_LLM_RETRIES;
  process.env.GRAFT_LLM_RETRIES = value;
  try {
    return await fn();
  } finally {
    if (orig === undefined) delete process.env.GRAFT_LLM_RETRIES;
    else process.env.GRAFT_LLM_RETRIES = orig;
  }
}

/**
 * Matched by name, not by `instanceof`, so this file also loads against a
 * build of the adapter that has no such class — the tests then fail on their
 * own assertions instead of on an import.
 */
function isProviderResponseError(e: unknown): e is Error & { code?: unknown; retryable?: boolean } {
  return e instanceof Error && e.name === "ProviderResponseError";
}

test("openai: a 200 carrying an error object is retried, and the retry succeeds", async () => {
  const { client, calls } = stubClient(erroredBody(), textBody("hello"));
  const { sleep, waits } = instantSleep();
  const m = new OpenAIChatModel({ apiKey: "x", model: "errored-then-fine", client, sleep });

  const res = await withRetries("2", () => m.create({ messages: [{ role: "user", content: "hi" }] }));

  assert.equal(calls.length, 2);
  assert.deepEqual(waits, [500]); // one backoff, not a tight retry loop
  assert.equal(res.text, "hello");
});

test("openai: an errored 200 spends the retry budget, then throws a typed retryable error", async () => {
  const { client, calls } = stubClient(erroredBody("upstream refused", "bad_gateway"));
  const { sleep, waits } = instantSleep();
  const m = new OpenAIChatModel({ apiKey: "x", model: "always-errored", client, sleep });

  const err = await withRetries("1", () =>
    m.create({ messages: [{ role: "user", content: "hi" }] }).then(
      () => assert.fail("expected the errored body to surface"),
      (e: unknown) => e,
    ),
  );

  assert.equal(calls.length, 2); // GRAFT_LLM_RETRIES=1 → one retry
  assert.deepEqual(waits, [500]);
  assert.ok(isProviderResponseError(err), `expected a ProviderResponseError, got ${String(err)}`);
  assert.equal(err.message, "upstream refused"); // the provider's own words, not ours
  assert.equal(err.code, "bad_gateway");
  assert.equal((err as { retryable?: boolean }).retryable, true);
  assert.equal(err instanceof TypeError, false);
});

test("openai: a 200 with no choices is a typed error, never a TypeError from choices[0]", async () => {
  for (const [what, body] of [
    ["an empty object", {}],
    ["choices but none in it", { choices: [] }],
    ["a body that is not JSON at all", "gateway timeout"],
  ] as const) {
    const { client, calls } = stubClient(body);
    const { sleep } = instantSleep();
    const m = new OpenAIChatModel({ apiKey: "x", model: `no-choices-${String(what.length)}`, client, sleep });

    const err = await withRetries("0", () =>
      m.create({ messages: [{ role: "user", content: "hi" }] }).then(
        () => assert.fail(`expected ${what} to surface as an error`),
        (e: unknown) => e,
      ),
    );

    assert.equal(calls.length, 1);
    assert.ok(isProviderResponseError(err), `${what}: got ${String(err)}`);
    assert.equal(err instanceof TypeError, false);
  }
});

test("openai: forced → required → auto fallback reaches a tool call", async () => {
  const { client, calls } = stubClient(erroredBody(), textBody("I would rather explain it"), toolBody("record_graph", { nodes: [1, 2] }));
  const { sleep } = instantSleep();
  const m = new OpenAIChatModel({ apiKey: "x", model: "ladder-model", client, sleep });

  const { result, err } = await withCapturedError(() => m.create(TOOL_REQ));

  assert.deepEqual(
    calls.map((c) => c.tool_choice),
    [{ type: "function", function: { name: "record_graph" } }, "required", "auto"],
  );
  assert.deepEqual(calls.map((c) => c.messages.length), [2, 2, 3]); // the auto rung names the tool
  assert.match(String(calls[2].messages[1].content), /record_graph/);
  assert.equal(result.toolCalls.length, 1);
  assert.deepEqual(result.toolCalls[0].args, { nodes: [1, 2] });
  assert.equal(err.length, 1); // the downgrade is announced once
});

test("openai: JSON written into content is returned as the JSON text json mode asked for", async () => {
  const fenced = "```json\n{\"correct\":true}\n```";
  const { client, calls } = stubClient(textBody("Sure, here you go"), textBody("Still explaining."), textBody(fenced));
  const { sleep } = instantSleep();
  const m = new OpenAIChatModel({ apiKey: "x", model: "json-in-content", client, sleep });

  const { result } = await withCapturedError(() =>
    m.create({ messages: [{ role: "user", content: "grade" }], responseFormat: { kind: "json" } }),
  );

  assert.equal(calls.length, 3);
  assert.equal(calls.at(-1)?.tool_choice, "auto");
  assert.deepEqual(JSON.parse(result.text), { correct: true }); // fence stripped, not passed through
  assert.equal(result.toolCalls.length, 0); // the synthetic tool stays hidden
});

test("openai: a tool payload left in content surfaces as a call to the named tool", async () => {
  const { client, calls } = stubClient(textBody('Here is the graph: {"nodes":[1]}'));
  const { sleep } = instantSleep();
  const m = new OpenAIChatModel({ apiKey: "x", model: "tool-in-content", client, sleep });

  const { result } = await withCapturedError(() => m.create(TOOL_REQ));

  assert.equal(calls.length, 1); // the payload was in the reply; no second ask
  assert.equal(result.toolCalls.length, 1);
  assert.equal(result.toolCalls[0].name, "record_graph");
  assert.deepEqual(result.toolCalls[0].args, { nodes: [1] });
});

test("openai: an emulated tool call in content on the last rung is unwrapped", async () => {
  const emulated = '[{"name":"record_graph","parameters":{"nodes":[2]}}]';
  const { client, calls } = stubClient(textBody("prose"), textBody("more prose"), textBody(emulated));
  const { sleep } = instantSleep();
  const m = new OpenAIChatModel({ apiKey: "x", model: "emulated-last-rung", client, sleep });

  const { result } = await withCapturedError(() => m.create(TOOL_REQ));

  assert.equal(calls.at(-1)?.tool_choice, "auto");
  assert.equal(result.toolCalls.length, 1);
  assert.deepEqual(result.toolCalls[0].args, { nodes: [2] });
});

test("openai: a reply with no payload at all is handed back untouched, for the caller to judge", async () => {
  const { client, calls } = stubClient(textBody("I am not going to call a tool."));
  const { sleep } = instantSleep();
  const m = new OpenAIChatModel({ apiKey: "x", model: "pure-prose", client, sleep });

  const { result } = await withCapturedError(() => m.create(TOOL_REQ));

  assert.equal(calls.length, 3);
  assert.equal(result.toolCalls.length, 0);
  assert.equal(result.text, "I am not going to call a tool."); // nothing invented, nothing hidden
});

test("openai: a payload already in content is not re-requested on another rung", async () => {
  const { client, calls } = stubClient(toolBody("record_graph", { nodes: [7] }));
  const { sleep } = instantSleep();
  const m = new OpenAIChatModel({ apiKey: "x", model: "emulated-wrapper", client, sleep });

  const { result } = await withCapturedError(() => m.create(TOOL_REQ));

  assert.equal(calls.length, 1);
  assert.deepEqual(result.toolCalls[0].args, { nodes: [7] });
});

test("openai: once a model is known to ignore a forced tool_choice, later calls go straight to auto", async () => {
  const { client, calls } = stubClient(textBody("here is a prose answer"));
  const { sleep } = instantSleep();
  const m = new OpenAIChatModel({ apiKey: "x", model: "remembered-model", client, sleep });

  const first = await withCapturedError(() => m.create(TOOL_REQ));
  assert.equal(calls.length, 3); // forced, required, auto
  assert.equal(first.err.length, 1);
  assert.match(first.err[0]!, /remembered-model/);

  calls.length = 0;
  const second = await withCapturedError(() => m.create(TOOL_REQ));

  assert.equal(calls.length, 1);
  assert.equal(calls[0].tool_choice, "auto");
  assert.equal(second.err.length, 0); // noted once, not once per call
});

test("openai: a second instance of the same endpoint + model skips the forced rung too", async () => {
  const first = stubClient(textBody("prose"));
  const { sleep } = instantSleep();
  await withCapturedError(
    () => new OpenAIChatModel({ apiKey: "x", model: "shared-endpoint-model", baseUrl: "http://stub", client: first.client, sleep }).create(TOOL_REQ),
  );

  const second = stubClient(textBody("prose"));
  const m = new OpenAIChatModel({ apiKey: "x", model: "shared-endpoint-model", baseUrl: "http://stub", client: second.client, sleep });
  await withCapturedError(() => m.create(TOOL_REQ));

  assert.equal(first.calls.length, 3);
  assert.equal(second.calls.length, 1);
  assert.equal(second.calls[0].tool_choice, "auto");
});

test("openai: an endpoint that honors the forced tool_choice sees exactly one call, unchanged", async () => {
  const { client, calls } = stubClient(toolBody("record_graph", { nodes: [1] }));
  const { sleep } = instantSleep();
  const m = new OpenAIChatModel({ apiKey: "x", model: "well-behaved-model", client, sleep });

  const { result, err } = await withCapturedError(() => m.create(TOOL_REQ));

  assert.equal(calls.length, 1);
  assert.deepEqual(calls[0].tool_choice, { type: "function", function: { name: "record_graph" } });
  assert.deepEqual(calls[0].messages, TOOL_REQ.messages.map((msg) => ({ role: msg.role, content: msg.content })));
  assert.deepEqual(result.toolCalls[0].args, { nodes: [1] });
  assert.equal(err.length, 0);
});
