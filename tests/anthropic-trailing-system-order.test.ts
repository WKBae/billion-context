import test from "node:test";
import assert from "node:assert/strict";
import { once } from "node:events";
import type { Config, CoreMessage } from "acp-kernel";
import { createCore, createInitialState, defaultConfig } from "acp-kernel";
import { anthropicToCore, type AnthropicRequestBody } from "acp-kernel/wire";
import { startServer } from "../src/server.ts";
import type { ProxyOptions } from "../src/config.ts";
import { SessionStore, _setStoreForTest } from "../src/persist.ts";
import { _setForTest as setRegistryForTest } from "../src/registry.ts";
import { insertBeforeSystemTail } from "../src/util.ts";
import { appendTrailingUserText } from "../src/wire-body.ts";
import type { Session } from "../src/session.ts";
import { runCompressLoop, createAnthropicAdapter } from "../src/loop/index.ts";
import { buildCompressSystemPrompt } from "../src/compress-tool.ts";
import { startFakeUpstream, VALIDATORS } from "./wire-contract-fakes.ts";

// WC-023: Anthropic accepts a mid-conversation role:"system" message only as
// the last entry or directly before an assistant turn. Claude Code closes
// messages[] with a per-request system note, so every user turn bili adds
// (nudge, anchor notes, retrievals, loop extras, post-assembly hints) must
// land in front of that trailing system run. The validation-parity fake
// enforces WC-023, so a regression relays its 400 instead of passing.

const BASE_SYSTEM = Array.from({ length: 10 }, (_, i) => `line ${i + 1}`).join("\n");
const TRAILING_SYSTEM = { role: "system", content: "<total_tokens>900000 tokens left</total_tokens>" };

interface Harness {
    proxyUrl: string;
    close(): Promise<void>;
}

async function startHarness(fakeUrl: string): Promise<Harness> {
    _setStoreForTest(new SessionStore({ enabled: false }));
    setRegistryForTest({});
    const opts: ProxyOptions = {
        port: 0,
        host: "127.0.0.1",
        upstream: fakeUrl,
        routes: { [fakeUrl]: { models: { "claude-test": { context: 1_000_000 } } } },
        modelContextLimit: 1_000_000,
        kernelConfig: defaultConfig(1_000_000),
        compress: { injectTool: true, injectNudge: true },
        // The anchor reconciler turns a localized head-system edit into a
        // sysNotes user turn — the deterministic proxy-added trailing turn.
        stableSystemAnchor: true,
        promptCache: { routing: "auto" },
        sessionHeader: "x-acp-session",
        log: false,
        debug: false,
        passthrough: false,
        autoUpdate: false,
        compat: { roles: {} },
        streamErrorShape: "protocol",
        passthroughSource: null,
        autoRestartOnUpdate: false,
        updateTag: "latest",
        advisoryCheck: false,
        releaseNotesCheck: false,
        mitm: { enabled: false, domains: [] },
    };
    const proxy = await startServer(opts);
    await once(proxy, "listening");
    const port = (proxy.address() as { port: number }).port;
    return {
        proxyUrl: `http://127.0.0.1:${port}/bili/${fakeUrl}`,
        close: () => new Promise<void>((resolve, reject) => proxy.close((e) => (e ? reject(e) : resolve()))),
    };
}

async function sendTurn(h: Harness, sessionId: string, system: string, messages: unknown[]): Promise<number> {
    const res = await fetch(`${h.proxyUrl}/v1/messages`, {
        method: "POST",
        headers: { "content-type": "application/json", "x-api-key": "test", "x-acp-session": sessionId },
        body: JSON.stringify({ model: "claude-test", max_tokens: 1024, stream: true, system, messages }),
    });
    await res.text();
    return res.status;
}

type Msg = { role: string; content: unknown };

const textOf = (m: Msg | undefined): string => (typeof m?.content === "string" ? m.content : JSON.stringify(m?.content ?? ""));

async function driftTurns(sessionId: string, withTrailingSystem: boolean): Promise<{ status: number; violations: string[]; messages: Msg[] }> {
    const fake = await startFakeUpstream("anthropic");
    let h: Harness | undefined;
    try {
        h = await startHarness(fake.url);
        const turn1 = [{ role: "user", content: "first question" }, ...(withTrailingSystem ? [TRAILING_SYSTEM] : [])];
        assert.equal(await sendTurn(h, sessionId, BASE_SYSTEM, turn1), 200, "turn 1 baseline completes");
        const turn2 = [
            { role: "user", content: "first question" },
            { role: "assistant", content: "first answer" },
            { role: "user", content: "second question" },
            ...(withTrailingSystem ? [TRAILING_SYSTEM] : []),
        ];
        const status = await sendTurn(h, sessionId, `${BASE_SYSTEM}\nUPDATED LINE`, turn2);
        const body = fake.requests[1]?.body as { messages?: Msg[] } | undefined;
        return { status, violations: [...fake.violations], messages: body?.messages ?? [] };
    } finally {
        if (h) await h.close();
        await fake.close();
    }
}

test("WC-023: a proxy-added user note lands before the client's trailing system message", async () => {
    const { status, violations, messages } = await driftTurns("wc023-trailing", true);
    assert.equal(status, 200, "turn 2 completes — a WC-023 violation would relay the fake's 400");
    assert.deepEqual(violations, [], `fake upstream violations:\n${violations.join("\n")}`);
    const last = messages[messages.length - 1];
    assert.equal(last?.role, "system", "the client's system message stays last");
    assert.ok(textOf(last).includes("total_tokens"), "the trailing system message is the client's own");
    const prev = messages[messages.length - 2];
    assert.equal(prev?.role, "user");
    assert.ok(textOf(prev).includes("UPDATED"), "the system-drift note sits directly before the trailing system message");
    const marked = messages.flatMap((m, i) => (Array.isArray(m.content) && m.content.some((b) => (b as { cache_control?: unknown }).cache_control !== undefined) ? [i] : []));
    assert.ok(marked.length > 0, "bili stamps message breakpoints (no client-managed caching)");
    assert.ok(marked.every((i) => i < messages.length - 2), `breakpoints stay ahead of the note and the trailing system run (marked: ${marked.join(",")})`);
});

test("WC-023 negative control: without a trailing system message the note is still appended last", async () => {
    const { status, violations, messages } = await driftTurns("wc023-plain", false);
    assert.equal(status, 200);
    assert.deepEqual(violations, []);
    const last = messages[messages.length - 1];
    assert.equal(last?.role, "user");
    assert.ok(textOf(last).includes("UPDATED"), "the note is the final message");
});

test("insertBeforeSystemTail: inserts before the trailing system run only", () => {
    const u = { role: "user", content: "u" };
    const a = { role: "assistant", content: "a" };
    const s1 = { role: "system", content: "s1" };
    const s2 = { role: "system", content: "s2" };
    const n = { role: "user", content: "n" };
    assert.deepEqual(insertBeforeSystemTail([u, s1, s2], [n]), [u, n, s1, s2]);
    assert.deepEqual(insertBeforeSystemTail([u, s1, a, u], [n]), [u, s1, a, u, n], "a mid-history system message is not a tail");
    assert.deepEqual(insertBeforeSystemTail([u], [n]), [u, n]);
    assert.deepEqual(insertBeforeSystemTail([u, s1], []), [u, s1]);
});

test("appendTrailingUserText (anthropic): merges into the user turn before a trailing system message", () => {
    const out = appendTrailingUserText("anthropic", JSON.stringify({ messages: [{ role: "user", content: "q" }, TRAILING_SYSTEM] }), "hint")!;
    const parsed = JSON.parse(out) as { messages: Msg[] };
    assert.deepEqual(parsed.messages, [{ role: "user", content: "q\n\nhint" }, TRAILING_SYSTEM]);
    assert.deepEqual(VALIDATORS.anthropic(parsed).filter((v) => v.startsWith("WC-023")), []);

    const afterAssistant = appendTrailingUserText("anthropic", JSON.stringify({ messages: [{ role: "user", content: "q" }, { role: "assistant", content: "a" }, TRAILING_SYSTEM] }), "hint")!;
    const parsed2 = JSON.parse(afterAssistant) as { messages: Msg[] };
    assert.deepEqual(parsed2.messages.map((m) => m.role), ["user", "assistant", "user", "system"], "a fresh user turn is spliced in before the system run");
    // Other wires keep the plain append.
    const openai = appendTrailingUserText("openai", JSON.stringify({ messages: [{ role: "user", content: "q" }, { role: "system", content: "s" }] }), "hint")!;
    assert.deepEqual((JSON.parse(openai) as { messages: Msg[] }).messages.map((m) => m.role), ["user", "system", "user"]);
});

test("compress loop (anthropic): a round-1 continuation nudge lands before the client's trailing system message", async () => {
    const clientBody = { model: "claude-test", stream: true, max_tokens: 10, messages: [{ role: "user", content: "q" }, TRAILING_SYSTEM] };
    const ev = (type: string, data: unknown): string => `event: ${type}\ndata: ${JSON.stringify(data)}\n\n`;
    const start = ev("message_start", { type: "message_start", message: { id: "msg_1", type: "message", role: "assistant", model: "claude-test", content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: 5, output_tokens: 0 } } });
    const text = (t: string) => ev("content_block_start", { type: "content_block_start", index: 0, content_block: { type: "text", text: "" } }) + ev("content_block_delta", { type: "content_block_delta", index: 0, delta: { type: "text_delta", text: t } });
    // Visible text, then EOF: the loop re-fetches once with a continuation nudge.
    const partial = start + text("The answer is");
    const good = start + text(" forty-two.") + ev("content_block_stop", { type: "content_block_stop", index: 0 }) + ev("message_delta", { type: "message_delta", delta: { stop_reason: "end_turn", stop_sequence: null }, usage: { output_tokens: 3 } }) + ev("message_stop", { type: "message_stop" });
    const bodies: string[] = [];
    const orig = globalThis.fetch;
    globalThis.fetch = (async (_url: unknown, init?: { body?: unknown }) => {
        bodies.push(String(init?.body));
        return new Response(good, { status: 200 });
    }) as typeof fetch;
    try {
        const ctx = {
            core: createCore(),
            config: { modelContextLimit: 200000 } as Config,
            messages: anthropicToCore(clientBody as AnthropicRequestBody).msgs as CoreMessage[],
            session: {
                id: "wc023-loop", meta: {}, metadata: {}, state: createInitialState(), createdAt: Date.now(), lastSeen: Date.now(),
                stats: { requests: 0, tokensSaved: 0, inputTokens: 0, cachedTokens: 0, outputTokens: 0, cacheSamples: 0, lastInputTokens: 0, contextTokens: 0, compressCreditTokens: 0, retrieveCalls: 0, retrieveHits: 0, retrieveMisses: 0, storedBytes: 0, storeBytesSaved: 0, rangeRestores: 0 },
                blockContents: new Map(), inFlight: 0, persisted: false, pendingRetrievals: [],
            } as Session,
            log: () => {},
            protocol: "anthropic" as const,
        };
        const stream = new Response(partial).body!;
        for await (const _ of runCompressLoop(stream, ctx, clientBody, { url: "http://mock", headers: {} }, createAnthropicAdapter(clientBody), buildCompressSystemPrompt())) { /* drain */ }
    } finally {
        globalThis.fetch = orig;
    }
    assert.equal(bodies.length, 1, "exactly one continuation re-fetch");
    const sent = JSON.parse(bodies[0]!) as { messages: Msg[] };
    assert.deepEqual(VALIDATORS.anthropic(sent).filter((v) => v.startsWith("WC-023")), [], "re-fetch body satisfies WC-023");
    const last = sent.messages[sent.messages.length - 1];
    assert.equal(last?.role, "system", "the client's system message stays last");
    assert.ok(textOf(sent.messages[sent.messages.length - 2]).includes("cut off mid-transmission"), "the continuation nudge sits directly before it");
});
