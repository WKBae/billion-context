// Claude Code closes every request with a role:"system" note
// (<total_tokens>…</total_tokens>) that carries the request's only
// message-level cache_control. A fresh Claude Code process re-sends the
// history without the old notes, so its first note repeats the content-derived
// id of an older note a fold already covered, and prune drops it with the
// breakpoint on it. The client's breakpoint must move to the nearest earlier
// surviving message instead of leaving the request with none.
import assert from "node:assert";
import http from "node:http";
import { once } from "node:events";
import test from "node:test";

process.env.NODE_ENV = "test";

import { defaultConfig } from "acp-kernel";
import { startServer } from "../src/server.ts";
import { carryPrunedCacheControls } from "../src/loop/cache-control.ts";
import type { ProxyOptions } from "../src/config.ts";
import { SessionStore, _setStoreForTest } from "../src/persist.ts";
import { _setForTest as setRegistryForTest } from "../src/registry.ts";

const anthropicSse = (event: string, data: unknown): string =>
    `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;

// Enough distinct mass per message that the folded range clears the kernel's
// 5000-char minCompressRange floor (#402) and the compress actually commits.
const FILLER = (seed: number, kb: number): string => {
    const para = `Paragraph ${seed}: the module under review passed its integration suite across all four regions without any reported regressions. `;
    const unit = Math.ceil((kb * 1024) / para.length);
    return Array.from({ length: unit }, (_, i) => para.replace(String(seed), `${seed}-${i}`)).join("");
};

function compressSse(startId: string, endId: string): string {
    const args = JSON.stringify({ startId, endId, topic: "fold", summary: "Folded the early subsystem-analysis turns; state-recovery failure modes and their remediation notes are retained in this summary." });
    return [
        anthropicSse("message_start", { type: "message_start", message: { id: "msg_r1", role: "assistant", usage: { input_tokens: 55 } } }),
        anthropicSse("content_block_start", { type: "content_block_start", index: 0, content_block: { type: "tool_use", id: "toolu_c_note", name: "compress", input: {} } }),
        anthropicSse("content_block_delta", { type: "content_block_delta", index: 0, delta: { type: "input_json_delta", partial_json: args } }),
        anthropicSse("content_block_stop", { type: "content_block_stop", index: 0 }),
        anthropicSse("message_delta", { type: "message_delta", delta: { stop_reason: "tool_use", stop_sequence: null }, usage: { output_tokens: 12 } }),
        anthropicSse("message_stop", { type: "message_stop" }),
    ].join("");
}

const TEXT_SSE = [
    anthropicSse("message_start", { type: "message_start", message: { id: "msg_r2", role: "assistant", usage: { input_tokens: 20 } } }),
    anthropicSse("content_block_start", { type: "content_block_start", index: 0, content_block: { type: "text", text: "" } }),
    anthropicSse("content_block_delta", { type: "content_block_delta", index: 0, delta: { type: "text_delta", text: "Done after compress" } }),
    anthropicSse("content_block_stop", { type: "content_block_stop", index: 0 }),
    anthropicSse("message_delta", { type: "message_delta", delta: { stop_reason: "end_turn", stop_sequence: null }, usage: { output_tokens: 4 } }),
    anthropicSse("message_stop", { type: "message_stop" }),
].join("");

interface Harness {
    proxyPort: number;
    upstreamPort: number;
    captured: string[];
    close(): Promise<void>;
}

async function startHarness(): Promise<Harness> {
    const captured: string[] = [];
    let call = 0;
    const upstream = http.createServer((req, res) => {
        const chunks: Buffer[] = [];
        req.on("data", (c: Buffer) => chunks.push(c));
        req.on("end", () => {
            captured.push(Buffer.concat(chunks).toString("utf8"));
            // Fold m00001..m00003 (u1, the mid-history note, a1) on the first
            // round; every later upstream call just answers.
            const sse = call === 0 ? compressSse("m00001", "m00003") : TEXT_SSE;
            call += 1;
            res.writeHead(200, { "content-type": "text/event-stream" });
            res.end(sse);
        });
    });
    upstream.listen(0, "127.0.0.1");
    await once(upstream, "listening");
    const upstreamPort = (upstream.address() as { port: number }).port;

    _setStoreForTest(new SessionStore({ enabled: false }));
    setRegistryForTest({});
    const proxy = await startServer({
        port: 0,
        host: "127.0.0.1",
        upstream: "http://127.0.0.1",
        routes: { [`http://127.0.0.1:${upstreamPort}`]: { models: { "claude-test": { context: 400_000 } } } },
        modelContextLimit: 400_000,
        kernelConfig: defaultConfig(400_000),
        compress: { injectTool: true, injectNudge: true },
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
    } as ProxyOptions);
    await once(proxy, "listening");
    const proxyPort = (proxy.address() as { port: number }).port;

    return {
        proxyPort,
        upstreamPort,
        captured,
        close: async () => {
            proxy.close();
            proxy.closeIdleConnections?.();
            await once(proxy, "close");
            upstream.close();
            upstream.closeIdleConnections?.();
            await once(upstream, "close");
        },
    };
}

type Block = { type?: string; text?: string; cache_control?: unknown };
type WireMessage = { role: string; content: string | Block[] };

const NOTE = "<total_tokens>15000000 tokens left</total_tokens>";
const CC = { type: "ephemeral" };

const text = (role: string, value: string, mark?: unknown): WireMessage => ({
    role,
    content: [{ type: "text", text: value, ...(mark ? { cache_control: mark } : {}) }],
});

// u1, note, a1, u2, a2, u3, a3, trigger → refs m00001..m00008. The fold covers
// m00001..m00003, outside the protected recent tail.
const history = (): WireMessage[] => [
    text("user", "user turn 1: examine subsystem 1 and report every failure mode. " + FILLER(1, 8)),
    { role: "system", content: NOTE },
    text("assistant", "assistant 1: subsystem 1 shows three failure modes. " + FILLER(101, 8)),
    text("user", "user turn 2: examine subsystem 2. " + FILLER(2, 8)),
    text("assistant", "assistant 2: subsystem 2 is stable. " + FILLER(102, 8)),
    text("user", "user turn 3: examine subsystem 3. " + FILLER(3, 8)),
    text("assistant", "assistant 3: subsystem 3 leaks file handles. " + FILLER(103, 8)),
    text("user", "TRIGGER_TURN: patch the refresh-token rotation. " + FILLER(900, 1)),
];

const markedBlocks = (raw: string): Block[] => {
    const sent = JSON.parse(raw) as { messages: WireMessage[] };
    return sent.messages.flatMap((m) => (typeof m.content === "string" ? [] : m.content)).filter((b) => b.cache_control !== undefined);
};

test("a fresh trailing note pruned as a covered id hands its breakpoint to the message before it", async () => {
    const h = await startHarness();
    const post = (messages: WireMessage[]) =>
        fetch(`http://127.0.0.1:${h.proxyPort}/bili/http://127.0.0.1:${h.upstreamPort}/v1/messages`, {
            method: "POST",
            headers: { "content-type": "application/json", "x-acp-session": "pruned-client-mark" },
            body: JSON.stringify({ model: "claude-test", max_tokens: 1024, stream: true, system: "You are a coding agent.", messages }),
        });
    try {
        const first = await post([...history(), { role: "system", content: [{ type: "text", text: NOTE, cache_control: CC }] }]);
        assert.equal(first.status, 200);
        await first.text();
        assert.ok(h.captured.length >= 2 && h.captured[1]!.includes("tokens saved"), "the fold over the mid-history note committed");

        // New Claude Code process: the old note is gone from the history and
        // the identical fresh note takes its id.
        const rebuilt = history().filter((m) => m.role !== "system");
        const second = await post([
            ...rebuilt,
            text("assistant", "Done after compress"),
            text("user", "NEXT_TURN: now add a regression test."),
            { role: "system", content: [{ type: "text", text: NOTE, cache_control: CC }] },
        ]);
        assert.equal(second.status, 200);
        await second.text();

        const sent = h.captured.at(-1)!;
        assert.ok(!sent.includes(NOTE), "the fresh note is pruned as covered");
        const marked = markedBlocks(sent);
        assert.equal(marked.length, 1, `exactly one message breakpoint, got ${marked.length}`);
        assert.ok(marked[0]!.text?.includes("NEXT_TURN"), "the breakpoint sits on the last surviving message");
        assert.deepEqual(marked[0]!.cache_control, CC);
    } finally {
        await h.close();
    }
});

test("carryPrunedCacheControls: pruned marks land on the nearest earlier markable survivor", () => {
    const A = { type: "ephemeral", ttl: "1h" };
    const B = { type: "ephemeral" };
    const original = [
        { id: "u1" },
        { id: "think", contentType: "reasoning" },
        { id: "raw", rawAnthropicMessage: {} },
        { id: "gone1" },
        { id: "u2" },
        { id: "gone2" },
        { id: "gone3" },
    ];
    const view = original.filter((m) => !m.id.startsWith("gone"));

    const untouched = new Map<string, unknown>([["u2", A]]);
    assert.strictEqual(carryPrunedCacheControls(original, view, untouched), untouched);

    assert.deepStrictEqual(carryPrunedCacheControls(original, view, new Map([["gone1", A]])), new Map([["u1", A]]));
    assert.deepStrictEqual(carryPrunedCacheControls(original, view, new Map([["gone2", A], ["gone3", B]])), new Map([["u2", A]]));
    assert.deepStrictEqual(carryPrunedCacheControls(original, view, new Map<string, unknown>([["u2", B], ["gone3", A]])), new Map([["u2", B]]));
    assert.deepStrictEqual(carryPrunedCacheControls([{ id: "gone1" }, ...view], view, new Map([["gone1", A]])), new Map());
});
