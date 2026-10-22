// #3429: pi-claude-bridge's Claude Code children join the pi conversation's
// plugin session through ANTHROPIC_BASE_URL + ANTHROPIC_CUSTOM_HEADERS in the
// environment the bridge builds for each child. The bridge emits a request on
// the serving pi session's event bus; bili's listener registers a transform
// that returns that one child's environment. The host below runs the
// bridge's side of the protocol (emit, run the registered transforms on a
// copy, spawn with the result) and spawns real children with it.
import { test } from "node:test";
import assert from "node:assert/strict";
import http from "node:http";
import { once } from "node:events";
import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { existsSync, mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { createBiliPlugin, INLINE_BRIDGE_WARNING } from "../src/agent/pi.ts";
import {
    CHILD_ENV_CHANNEL,
    childReachableOrigin,
    claudeBridgeChildEnvListener,
    firstPartyUpstream,
    headerSafeId,
    isClaudeBridgeModel,
    mergedCustomHeaders,
    routedBaseUrl,
    routedChildEnv,
} from "../src/agent/claude-bridge-routing.ts";
import { INLINE_ORIGIN } from "../src/inline-origin.ts";

const ROUTING_VARS = ["ANTHROPIC_BASE_URL", "ANTHROPIC_CUSTOM_HEADERS"] as const;

async function withEnv(vars: Record<string, string | undefined>, fn: () => Promise<void>): Promise<void> {
    const keys = new Set([...Object.keys(vars), ...ROUTING_VARS, "BILLION_CONTEXT_PROXY", "BILLION_CONTEXT_PLUGIN", "BILI_NON_HTTP_PROVIDERS"]);
    const saved = new Map<string, string | undefined>();
    for (const k of keys) saved.set(k, process.env[k]);
    for (const k of keys) {
        const v = vars[k];
        if (v === undefined) delete process.env[k];
        else process.env[k] = v;
    }
    try {
        await fn();
    } finally {
        for (const [k, v] of saved) {
            if (v === undefined) delete process.env[k];
            else process.env[k] = v;
        }
    }
}

test("isClaudeBridgeModel identifies the bridge transport only", () => {
    assert.equal(isClaudeBridgeModel({ api: "claude-bridge", baseUrl: "claude-bridge" }), true);
    assert.equal(isClaudeBridgeModel({ baseUrl: "claude-bridge" }), true);
    assert.equal(isClaudeBridgeModel({ api: "claude-bridge" }), true);
    assert.equal(isClaudeBridgeModel({ api: "other-transport", baseUrl: "other-transport" }), false, "another opaque extension transport is not the bridge");
    assert.equal(isClaudeBridgeModel({ api: "anthropic-messages", baseUrl: "https://api.anthropic.com" }), false);
    assert.equal(isClaudeBridgeModel(undefined), false);
});

test("routedBaseUrl wraps the default or the user's relay, keeps a same-proxy route, refuses a foreign one", () => {
    const origin = "http://127.0.0.1:18790";
    assert.equal(routedBaseUrl(origin, undefined), `${origin}/bili/https://api.anthropic.com`);
    assert.equal(routedBaseUrl(origin, "  "), `${origin}/bili/https://api.anthropic.com`);
    assert.equal(routedBaseUrl(origin, "https://relay.example.com/anthropic/"), `${origin}/bili/https://relay.example.com/anthropic`);
    assert.equal(routedBaseUrl(origin, `${origin}/bili/https://api.anthropic.com`), `${origin}/bili/https://api.anthropic.com`);
    assert.equal(routedBaseUrl(origin, `${origin}/bili/anthropic/https://api.anthropic.com`), `${origin}/bili/anthropic/https://api.anthropic.com`);
    assert.equal(routedBaseUrl(origin, "http://127.0.0.1:8787/bili/https://api.anthropic.com"), undefined);
    assert.equal(routedBaseUrl(origin, "https://relay.example.com/bili/notes"), `${origin}/bili/https://relay.example.com/bili/notes`, "a plain /bili/ path segment is not a bili route");
    assert.equal(routedBaseUrl(origin, "http://localhost:18790/bili/https://api.anthropic.com"), "http://localhost:18790/bili/https://api.anthropic.com", "a loopback alias of the same listener is the same proxy");
    assert.equal(routedBaseUrl(origin, "http://localhost:8787/bili/https://api.anthropic.com"), undefined, "a different port is a different proxy");
});

test("firstPartyUpstream recognizes only an Anthropic-wire route to api.anthropic.com", () => {
    assert.equal(firstPartyUpstream("http://127.0.0.1:18790/bili/https://api.anthropic.com"), true);
    assert.equal(firstPartyUpstream("http://127.0.0.1:18790/bili/anthropic/https://api.anthropic.com/v1"), true);
    assert.equal(firstPartyUpstream("http://127.0.0.1:18790/bili/https://API.Anthropic.com"), true);
    assert.equal(firstPartyUpstream("http://127.0.0.1:18790/bili/https://relay.example.com/anthropic"), false, "a user relay is not first-party");
    assert.equal(firstPartyUpstream("http://127.0.0.1:18790/bili/https://relay.example.com/api.anthropic.com"), false, "the host decides, not the path");
    assert.equal(firstPartyUpstream("http://127.0.0.1:18790/bili/https://api.anthropic.com.evil.example"), false);
    assert.equal(firstPartyUpstream("http://127.0.0.1:18790/bili/https://api.anthropic.com:8443"), false, "a non-default port is another endpoint");
    assert.equal(firstPartyUpstream("http://127.0.0.1:18790/bili/openai/https://api.anthropic.com"), false, "another wire is not Claude Code's");
    assert.equal(firstPartyUpstream("https://api.anthropic.com"), false, "an unrouted base is not this function's concern");
});

test("mergedCustomHeaders replaces only the owned plugin headers, case-insensitively", () => {
    assert.equal(mergedCustomHeaders(undefined, { agent: "pi", conversationId: "sid-1" }), "x-bili-plugin: pi\nx-bili-plugin-conversation: sid-1");
    assert.equal(
        mergedCustomHeaders("X-Team: core\nX-Bili-Plugin: stale\r\nx-bili-plugin-conversation: old\nX-Bili-Plugin-Model: m\nx-bili-plugin-context-window: 1\nx-bili-plugin-agent: keep\n\n", { agent: "pi", conversationId: "sid-2", contextWindow: 1000000.7, model: "claude-haiku-5-5" }),
        "X-Team: core\nx-bili-plugin-agent: keep\nx-bili-plugin: pi\nx-bili-plugin-conversation: sid-2\nx-bili-plugin-context-window: 1000000\nx-bili-plugin-model: claude-haiku-5-5",
    );
    assert.equal(mergedCustomHeaders(undefined, { agent: "pi", conversationId: "s", contextWindow: 0, model: "bad model\nx: y" }), "x-bili-plugin: pi\nx-bili-plugin-conversation: s", "unusable optional values are omitted");
});

test("childReachableOrigin rejects the in-process inline lane", () => {
    assert.equal(childReachableOrigin("http://127.0.0.1:18790"), true);
    assert.equal(childReachableOrigin("https://proxy.example.com"), true);
    assert.equal(childReachableOrigin(INLINE_ORIGIN), false);
    assert.equal(childReachableOrigin("ftp://x"), false);
    assert.equal(childReachableOrigin("not a url"), false);
    assert.equal(childReachableOrigin(undefined), false);
});

test("headerSafeId rejects ids that could split header lines", () => {
    assert.equal(headerSafeId("01a153f7-fdfc-739b-9995-f5412bd2f9b7"), true);
    assert.equal(headerSafeId("a b"), false);
    assert.equal(headerSafeId("a\nx-evil: 1"), false);
    assert.equal(headerSafeId(""), false);
    assert.equal(headerSafeId(undefined), false);
});

type Env = Record<string, string>;
type Transform = (env: Env) => Env;
type ChildModel = { id: string; contextWindow?: number };
type Handler = (event: unknown, ctx: unknown) => unknown;

/** A string-only snapshot of process.env, as the bridge's base environment. */
const baseEnv = (): Env => Object.fromEntries(Object.entries(process.env).filter((e): e is [string, string] => typeof e[1] === "string"));

const run = promisify(execFile);

/** What a child spawned with `env` sees. */
async function childSees(env: Env): Promise<{ base: string | null; headers: string | null }> {
    const { stdout } = await run(
        process.execPath,
        ["-e", "process.stdout.write(JSON.stringify({ base: process.env.ANTHROPIC_BASE_URL ?? null, headers: process.env.ANTHROPIC_CUSTOM_HEADERS ?? null }))"],
        { env },
    );
    return JSON.parse(stdout) as { base: string | null; headers: string | null };
}

/** A pi host with the synchronous inter-extension bus pi exposes as
 *  `pi.events`. `childEnv` plays the bridge: it emits one request on this
 *  host's bus, closes registration, and applies the transforms to copies of
 *  the base environment. */
function makeHost(opts: { events?: boolean } = {}) {
    const handlers = new Map<string, Handler[]>();
    const busHandlers = new Map<string, Array<(data: unknown) => void>>();
    const tools: string[] = [];
    const host = {
        tools,
        on(event: string, handler: Handler) {
            const list = handlers.get(event) ?? [];
            list.push(handler);
            handlers.set(event, list);
        },
        registerTool(tool: { name: string }) {
            if (!tools.includes(tool.name)) tools.push(tool.name);
        },
        registerCommand() {},
        getAllTools: () => tools.map((name) => ({ name })),
        async emit(event: string, ctx: unknown, payload: unknown = { type: event }): Promise<unknown> {
            let last: unknown;
            for (const h of handlers.get(event) ?? []) {
                const r = await h(payload, ctx);
                if (r !== undefined) last = r;
            }
            return last;
        },
        busListeners: (channel: string) => busHandlers.get(channel)?.length ?? 0,
        childEnv(req: { purpose?: string; sessionId: string; model?: ChildModel; version?: number }, base: Env = baseEnv()): Env {
            const transforms: Transform[] = [];
            let open = true;
            const request = Object.freeze({
                version: req.version ?? 1,
                purpose: req.purpose ?? "provider",
                sessionId: req.sessionId,
                model: Object.freeze({ ...(req.model ?? { id: "claude-haiku-5-5", contextWindow: 1000000 }) }),
                register(fn: Transform) {
                    if (!open) throw new Error("register() called after the request closed");
                    transforms.push(fn);
                },
            });
            try {
                for (const h of busHandlers.get(CHILD_ENV_CHANNEL) ?? []) h(request);
            } finally {
                open = false;
            }
            let env = { ...base };
            for (const t of transforms) env = { ...t({ ...env }) };
            return env;
        },
    } as Record<string, unknown> & { tools: string[]; emit: (event: string, ctx: unknown, payload?: unknown) => Promise<unknown>; busListeners: (c: string) => number; childEnv: (req: { purpose?: string; sessionId: string; model?: ChildModel; version?: number }, base?: Env) => Env };
    if (opts.events !== false) {
        host.events = {
            on(channel: string, handler: (data: unknown) => void) {
                const list = busHandlers.get(channel) ?? [];
                list.push(handler);
                busHandlers.set(channel, list);
                return () => {};
            },
        };
    }
    return host;
}

async function startManifestProxy(opts: { manifest?: boolean; status?: boolean } = {}): Promise<{ origin: string; close: () => Promise<void> }> {
    const server = http.createServer((req, res) => {
        if (req.url === "/__bili/plugin/manifest" && opts.manifest !== false) {
            res.writeHead(200, { "content-type": "application/json" });
            res.end(JSON.stringify({ tools: { anthropic: [{ name: "compress", description: "c", input_schema: { type: "object", properties: {} } }] } }));
            return;
        }
        if (req.url?.startsWith("/__bili/plugin/status") && opts.status === true) {
            res.writeHead(200, { "content-type": "application/json" });
            res.end(JSON.stringify({ ok: true }));
            return;
        }
        res.writeHead(404, { "content-type": "application/json" });
        res.end("{}");
    });
    server.listen(0, "127.0.0.1");
    await once(server, "listening");
    const origin = `http://127.0.0.1:${(server.address() as { port: number }).port}`;
    return { origin, close: () => new Promise((r) => server.close(() => r())) };
}

const bridgeModel = { api: "claude-bridge", baseUrl: "claude-bridge", provider: "claude-bridge", id: "claude-haiku-5-5", contextWindow: 1000000 };
const httpModel = { api: "anthropic-messages", baseUrl: "https://api.anthropic.com", provider: "anthropic", id: "claude-haiku-5-5", contextWindow: 1000000 };
const ctxFor = (sid: string, model: Record<string, unknown> = bridgeModel) => ({ sessionManager: { getSessionId: () => sid }, model, cwd: "/tmp" });
const routedHeaders = (sid: string, prefix = "", model = "claude-haiku-5-5", window = 1000000) => `${prefix}x-bili-plugin: pi\nx-bili-plugin-conversation: ${sid}\nx-bili-plugin-context-window: ${window}\nx-bili-plugin-model: ${model}`;
const clean = { base: null, headers: null };

test("routedChildEnv returns a routed copy and refuses a foreign proxy route", () => {
    const env = { PATH: "/bin", ANTHROPIC_CUSTOM_HEADERS: "X-Team: core" };
    const h = { agent: "pi", conversationId: "s" };
    assert.deepEqual(routedChildEnv(env, "http://127.0.0.1:1", h), {
        PATH: "/bin",
        ANTHROPIC_BASE_URL: "http://127.0.0.1:1/bili/https://api.anthropic.com",
        ANTHROPIC_CUSTOM_HEADERS: "X-Team: core\nx-bili-plugin: pi\nx-bili-plugin-conversation: s",
        _CLAUDE_CODE_ASSUME_FIRST_PARTY_BASE_URL: "1",
    });
    assert.deepEqual(env, { PATH: "/bin", ANTHROPIC_CUSTOM_HEADERS: "X-Team: core" }, "the input is never mutated");
    assert.equal(routedChildEnv({ ANTHROPIC_BASE_URL: "https://relay.example.com" }, "http://127.0.0.1:1", h)?._CLAUDE_CODE_ASSUME_FIRST_PARTY_BASE_URL, undefined, "a relay upstream is not marked first-party");
    assert.equal(routedChildEnv({ _CLAUDE_CODE_ASSUME_FIRST_PARTY_BASE_URL: "0" }, "http://127.0.0.1:1", h)?._CLAUDE_CODE_ASSUME_FIRST_PARTY_BASE_URL, "0", "a user value is kept");
    assert.equal(routedChildEnv({ ANTHROPIC_BASE_URL: "http://127.0.0.1:9/bili/https://api.anthropic.com" }, "http://127.0.0.1:1", h), undefined);
});

test("the listener registers only for well-formed provider requests with a routable session", () => {
    const seen: Array<[string, string]> = [];
    const listener = claudeBridgeChildEnvListener({ agent: "pi", originFor: (sid) => (sid === "ok" ? "http://127.0.0.1:1" : sid === "inline" ? INLINE_ORIGIN : undefined), onRouted: (sid, _o, outcome) => seen.push([sid, outcome]) });
    const attempt = (req: Record<string, unknown>): Transform[] => {
        const got: Transform[] = [];
        listener({ version: 1, purpose: "provider", sessionId: "ok", model: { id: "m" }, register: (fn: Transform) => got.push(fn), ...req });
        return got;
    };
    assert.equal(attempt({}).length, 1);
    for (const [name, req] of Object.entries({
        summary: { purpose: "summary" },
        askclaude: { purpose: "askclaude" },
        "future version": { version: 2 },
        "no register": { register: undefined },
        "no model id": { model: {} },
        "no session": { sessionId: "" },
        "unsafe session": { sessionId: "a\nb" },
        "unknown session": { sessionId: "other" },
        "inline origin": { sessionId: "inline" },
    })) assert.equal(attempt(req).length, 0, name);
    listener(undefined);
    listener("x");
    const [t] = attempt({ model: { id: "claude-opus-5-5", contextWindow: 200000 } });
    assert.deepEqual(t!({ ANTHROPIC_BASE_URL: "http://127.0.0.1:9/bili/https://api.anthropic.com" }), { ANTHROPIC_BASE_URL: "http://127.0.0.1:9/bili/https://api.anthropic.com" }, "a foreign route is kept as is");
    assert.equal(t!({}).ANTHROPIC_CUSTOM_HEADERS, "x-bili-plugin: pi\nx-bili-plugin-conversation: ok\nx-bili-plugin-context-window: 200000\nx-bili-plugin-model: claude-opus-5-5");
    assert.deepEqual(seen, [["ok", "foreign-route"], ["ok", "routed"]]);
});

test("a provider child of the session is routed; process.env is never written", async () => {
    const proxy = await startManifestProxy();
    try {
        await withEnv({ BILLION_CONTEXT_PROXY: proxy.origin }, async () => {
            const host = makeHost();
            createBiliPlugin("pi", { retryIntervalMs: 10 })(host as never);
            const before = JSON.stringify(process.env);
            assert.deepEqual(await childSees(host.childEnv({ sessionId: "sid-main" })), clean, "no route before the tools are registered");
            await host.emit("agent_start", ctxFor("sid-main"));
            assert.ok(host.tools.includes("compress"), "tools are registered at agent_start");
            assert.deepEqual(await childSees(host.childEnv({ sessionId: "sid-main" })), { base: `${proxy.origin}/bili/https://api.anthropic.com`, headers: routedHeaders("sid-main") });
            assert.deepEqual(
                await childSees(host.childEnv({ sessionId: "sid-main", model: { id: "claude-opus-5-5", contextWindow: 200000 } })),
                { base: `${proxy.origin}/bili/https://api.anthropic.com`, headers: routedHeaders("sid-main", "", "claude-opus-5-5", 200000) },
                "the headers name the request's model, not the session's",
            );
            for (const purpose of ["summary", "askclaude"]) assert.deepEqual(await childSees(host.childEnv({ purpose, sessionId: "sid-main" })), clean, `${purpose} children stay unrouted`);
            assert.deepEqual(await childSees(baseEnv()), clean, "a child spawned from process.env sees nothing");
            assert.equal(JSON.stringify(process.env), before, "process.env is byte-identical");
            await host.emit("session_shutdown", ctxFor("sid-main"));
        });
    } finally {
        await proxy.close();
    }
});

test("user routing values are wrapped in the child's copy only", async () => {
    const proxy = await startManifestProxy();
    try {
        await withEnv({ BILLION_CONTEXT_PROXY: proxy.origin, ANTHROPIC_BASE_URL: "https://relay.example.com", ANTHROPIC_CUSTOM_HEADERS: "X-Team: core" }, async () => {
            const host = makeHost();
            createBiliPlugin("pi", { retryIntervalMs: 10 })(host as never);
            await host.emit("agent_start", ctxFor("sid-user"));
            assert.deepEqual(await childSees(host.childEnv({ sessionId: "sid-user" })), { base: `${proxy.origin}/bili/https://relay.example.com`, headers: routedHeaders("sid-user", "X-Team: core\n") });
            assert.equal(process.env.ANTHROPIC_BASE_URL, "https://relay.example.com");
            assert.equal(process.env.ANTHROPIC_CUSTOM_HEADERS, "X-Team: core");
        });
    } finally {
        await proxy.close();
    }
});

test("overlapping sessions spawn concurrently, each child with its own routing", async () => {
    const proxy = await startManifestProxy();
    try {
        await withEnv({ BILLION_CONTEXT_PROXY: proxy.origin }, async () => {
            const a = makeHost();
            const b = makeHost();
            const plain = makeHost();
            createBiliPlugin("pi", { retryIntervalMs: 10 })(a as never);
            createBiliPlugin("pi", { retryIntervalMs: 10 })(b as never);
            createBiliPlugin("pi", { retryIntervalMs: 10 })(plain as never);
            await a.emit("agent_start", ctxFor("sid-a"));
            await b.emit("agent_start", ctxFor("sid-b"));
            await plain.emit("agent_start", ctxFor("sid-plain", httpModel));
            const envA = a.childEnv({ sessionId: "sid-a" });
            const envB = b.childEnv({ sessionId: "sid-b" });
            const [seenA, seenB, seenPlain, crossed] = await Promise.all([
                childSees(envA),
                childSees(envB),
                childSees(plain.childEnv({ sessionId: "sid-plain" })),
                childSees(a.childEnv({ sessionId: "sid-b" })),
            ]);
            assert.equal(seenA.headers, routedHeaders("sid-a"));
            assert.equal(seenB.headers, routedHeaders("sid-b"));
            assert.deepEqual(seenPlain, clean, "a session whose tools never registered for a bridge model stays unrouted");
            assert.deepEqual(crossed, clean, "a session's listener never routes another session's child");
        });
    } finally {
        await proxy.close();
    }
});

test("routing stays off when any precondition is missing", async () => {
    const proxy = await startManifestProxy();
    const noManifest = await startManifestProxy({ manifest: false });
    try {
        const cases: Array<{ name: string; agent: string; env: Record<string, string | undefined>; start?: unknown; sid?: string; events?: boolean }> = [
            { name: "omp lane", agent: "omp", env: { BILLION_CONTEXT_PROXY: proxy.origin } },
            { name: "no proxy", agent: "pi", env: {} },
            { name: "plugin kill switch", agent: "pi", env: { BILLION_CONTEXT_PROXY: proxy.origin, BILLION_CONTEXT_PLUGIN: "0" } },
            { name: "inline lane", agent: "pi", env: { BILLION_CONTEXT_PROXY: INLINE_ORIGIN } },
            { name: "tools not registered", agent: "pi", env: { BILLION_CONTEXT_PROXY: noManifest.origin } },
            { name: "session id changed after registration", agent: "pi", env: { BILLION_CONTEXT_PROXY: proxy.origin }, sid: "s2" },
            { name: "unsafe session id", agent: "pi", env: { BILLION_CONTEXT_PROXY: proxy.origin }, start: ctxFor("bad id"), sid: "bad id" },
            { name: "user already routes via another proxy", agent: "pi", env: { BILLION_CONTEXT_PROXY: proxy.origin, ANTHROPIC_BASE_URL: "http://127.0.0.1:9/bili/https://api.anthropic.com" } },
            { name: "host without an event bus", agent: "pi", env: { BILLION_CONTEXT_PROXY: proxy.origin }, events: false },
        ];
        for (const c of cases) {
            await withEnv(c.env, async () => {
                const host = makeHost({ events: c.events });
                createBiliPlugin(c.agent, { retryIntervalMs: 10 })(host as never);
                await host.emit("agent_start", c.start ?? ctxFor("s"));
                const env = host.childEnv({ sessionId: c.sid ?? "s" });
                assert.equal(env.ANTHROPIC_CUSTOM_HEADERS, undefined, `${c.name}: no plugin headers`);
                assert.equal(env.ANTHROPIC_BASE_URL, c.env.ANTHROPIC_BASE_URL, `${c.name}: base URL untouched`);
                if (c.agent !== "pi" || c.events === false) assert.equal(host.busListeners(CHILD_ENV_CHANNEL), 0, `${c.name}: no listener`);
                await host.emit("session_shutdown", ctxFor("s"));
            });
        }
    } finally {
        await proxy.close();
        await noManifest.close();
    }
});

test("a respawned proxy routes the next child to its new origin", async () => {
    const first = await startManifestProxy();
    const second = await startManifestProxy();
    try {
        await withEnv({ BILLION_CONTEXT_PROXY: first.origin }, async () => {
            const host = makeHost();
            createBiliPlugin("pi", { retryIntervalMs: 10 })(host as never);
            await host.emit("agent_start", ctxFor("sid-r"));
            assert.equal(host.childEnv({ sessionId: "sid-r" }).ANTHROPIC_BASE_URL, `${first.origin}/bili/https://api.anthropic.com`);
            process.env.BILLION_CONTEXT_PROXY = second.origin;
            assert.equal(host.childEnv({ sessionId: "sid-r" }).ANTHROPIC_BASE_URL, `${second.origin}/bili/https://api.anthropic.com`);
        });
    } finally {
        await first.close();
        await second.close();
    }
});

test("a session restart routes only the new session id", async () => {
    const proxy = await startManifestProxy();
    try {
        await withEnv({ BILLION_CONTEXT_PROXY: proxy.origin }, async () => {
            const host = makeHost();
            createBiliPlugin("pi", { retryIntervalMs: 10 })(host as never);
            await host.emit("agent_start", ctxFor("sid-old"));
            assert.equal(host.childEnv({ sessionId: "sid-old" }).ANTHROPIC_CUSTOM_HEADERS, routedHeaders("sid-old"));
            await host.emit("session_start", ctxFor("sid-new"), { type: "session_start", reason: "new" });
            await host.emit("agent_start", ctxFor("sid-new"));
            assert.equal(host.childEnv({ sessionId: "sid-old" }).ANTHROPIC_CUSTOM_HEADERS, undefined, "the previous session's id is no longer routed");
            assert.equal(host.childEnv({ sessionId: "sid-new" }).ANTHROPIC_CUSTOM_HEADERS, routedHeaders("sid-new"));
        });
    } finally {
        await proxy.close();
    }
});

test("cache warming is left to pi for claude-bridge sessions", async () => {
    await withEnv({ BILLION_CONTEXT_PROXY: "http://127.0.0.1:1" }, async () => {
        const host = makeHost();
        createBiliPlugin("pi", { retryIntervalMs: 10 })(host as never);
        assert.equal(await host.emit("cache_warming_decision", ctxFor("s"), { type: "cache_warming_decision", action: "warm" }), undefined);
    });
});

test("pi/bridge compaction keeps owning claude-bridge sessions, opt-in or not; other providers are unaffected", async () => {
    const proxy = await startManifestProxy({ status: true });
    try {
        const compact = (host: ReturnType<typeof makeHost>, model: Record<string, unknown>) =>
            host.emit("session_before_compact", ctxFor("s", model), { type: "session_before_compact", reason: "threshold" });
        const rows: Array<{ name: string; model: Record<string, unknown>; optIn?: string; cancel: boolean }> = [
            { name: "bridge, routed, carried, not opted in", model: bridgeModel, cancel: false },
            { name: "bridge, routed, carried, opted in", model: bridgeModel, optIn: "claude-bridge", cancel: false },
            { name: "http provider carried by the proxy", model: httpModel, cancel: true },
            { name: "another opaque provider, opted in", model: { api: "other-transport", baseUrl: "other-transport", provider: "other", id: "m" }, optIn: "other", cancel: true },
            { name: "another opaque provider, not opted in", model: { api: "other-transport", baseUrl: "other-transport", provider: "other", id: "m" }, cancel: false },
        ];
        for (const r of rows) {
            await withEnv({ BILLION_CONTEXT_PROXY: proxy.origin, BILI_NON_HTTP_PROVIDERS: r.optIn }, async () => {
                const host = makeHost();
                createBiliPlugin("pi", { retryIntervalMs: 10 })(host as never);
                await host.emit("agent_start", ctxFor("s", r.model));
                const result = (await compact(host, r.model)) as { cancel?: boolean } | undefined;
                assert.equal(result?.cancel === true, r.cancel, r.name);
                await host.emit("session_shutdown", ctxFor("s", r.model));
            });
        }
    } finally {
        await proxy.close();
    }
});

type Note = [string, string | undefined];
type WarnHost = { host: ReturnType<typeof makeHost>; notes: Note[]; logged: () => number; ctx: (o?: { model?: Record<string, unknown>; catalog?: unknown; ui?: unknown }) => Record<string, unknown> };

/** A plugin instance on a hermetic log and an offline fetch: the inline
 *  origin is never dialed and warnings land in a temp bili.log. */
async function withWarnHost(env: Record<string, string | undefined>, agent: string, fn: (h: WarnHost) => Promise<void>): Promise<void> {
    const dir = mkdtempSync(path.join(tmpdir(), "bili-3429-warn-"));
    const realFetch = globalThis.fetch;
    globalThis.fetch = (async () => {
        throw new Error("offline test");
    }) as typeof fetch;
    try {
        await withEnv({ ...env, XDG_STATE_HOME: dir }, async () => {
            const host = makeHost();
            createBiliPlugin(agent, { retryIntervalMs: 10 })(host as never);
            const notes: Note[] = [];
            const logFile = path.join(dir, "billion-context", "bili.log");
            const logged = () => (existsSync(logFile) ? readFileSync(logFile, "utf8").split("\n").filter((l) => l.includes(INLINE_BRIDGE_WARNING)).length : 0);
            const ctx = (o: { model?: Record<string, unknown>; catalog?: unknown; ui?: unknown } = {}) => ({
                sessionManager: { getSessionId: () => "sid-w" },
                model: o.model ?? httpModel,
                cwd: "/tmp",
                modelRegistry: o.catalog === undefined ? undefined : { getAll: o.catalog },
                ui: "ui" in o ? o.ui : { notify: (m: string, t?: string) => notes.push([m, t]) },
            });
            await fn({ host, notes, logged, ctx });
            await host.emit("session_shutdown", ctx());
            await new Promise((r) => setImmediate(r));
        });
    } finally {
        globalThis.fetch = realFetch;
        rmSync(dir, { recursive: true, force: true });
    }
}

const otherTransport = { api: "other-transport", baseUrl: "other-transport", provider: "other", id: "m" };
const catalogOf = (...models: Array<Record<string, unknown>>) => () => models;
const startup = { type: "session_start", reason: "startup" };

test("inline lane + pi-claude-bridge: startup warning truth table", async () => {
    const rows: Array<{ name: string; agent?: string; env: Record<string, string | undefined>; model?: Record<string, unknown>; catalog?: unknown; warn: boolean }> = [
        { name: "inline, bridge in the catalog, HTTP model selected", env: { BILLION_CONTEXT_PROXY: INLINE_ORIGIN }, catalog: catalogOf(httpModel, bridgeModel), warn: true },
        { name: "inline, bridge model selected, no catalog", env: { BILLION_CONTEXT_PROXY: INLINE_ORIGIN }, model: bridgeModel, warn: true },
        { name: "inline, bridge model selected, catalog throws", env: { BILLION_CONTEXT_PROXY: INLINE_ORIGIN }, model: bridgeModel, catalog: () => { throw new Error("boom"); }, warn: true },
        { name: "inline, stale raw BILI_PI_INLINE=0", env: { BILLION_CONTEXT_PROXY: INLINE_ORIGIN, BILI_PI_INLINE: "0" }, catalog: catalogOf(bridgeModel), warn: true },
        { name: "inline, no bridge loaded", env: { BILLION_CONTEXT_PROXY: INLINE_ORIGIN }, catalog: catalogOf(httpModel), warn: false },
        { name: "inline, another opaque transport", env: { BILLION_CONTEXT_PROXY: INLINE_ORIGIN }, model: otherTransport, catalog: catalogOf(otherTransport), warn: false },
        { name: "inline, catalog throws, HTTP model selected", env: { BILLION_CONTEXT_PROXY: INLINE_ORIGIN }, catalog: () => { throw new Error("boom"); }, warn: false },
        { name: "HTTP proxy lane with the bridge", env: { BILLION_CONTEXT_PROXY: "http://127.0.0.1:1" }, model: bridgeModel, catalog: catalogOf(bridgeModel), warn: false },
        { name: "raw BILI_PI_INLINE=1 but HTTP proxy lane in effect", env: { BILLION_CONTEXT_PROXY: "http://127.0.0.1:1", BILI_PI_INLINE: "1" }, model: bridgeModel, catalog: catalogOf(bridgeModel), warn: false },
        { name: "raw BILI_PI_INLINE=1 but no lane in effect", env: { BILI_PI_INLINE: "1" }, model: bridgeModel, catalog: catalogOf(bridgeModel), warn: false },
        { name: "plugin kill switch", env: { BILLION_CONTEXT_PROXY: INLINE_ORIGIN, BILLION_CONTEXT_PLUGIN: "0" }, model: bridgeModel, catalog: catalogOf(bridgeModel), warn: false },
        { name: "omp lane", agent: "omp", env: { BILLION_CONTEXT_PROXY: INLINE_ORIGIN }, model: bridgeModel, catalog: catalogOf(bridgeModel), warn: false },
    ];
    for (const r of rows) {
        await withWarnHost({ BILI_PI_INLINE: undefined, ...r.env }, r.agent ?? "pi", async ({ host, notes, logged, ctx }) => {
            await host.emit("session_start", ctx({ model: r.model, catalog: r.catalog }), startup);
            assert.deepEqual(notes, r.warn ? [[INLINE_BRIDGE_WARNING, "warning"]] : [], `${r.name}: toast`);
            assert.equal(logged(), r.warn ? 1 : 0, `${r.name}: log line`);
        });
    }
});

test("the inline bridge warning fires once per instance across startup and model events", async () => {
    await withWarnHost({ BILLION_CONTEXT_PROXY: INLINE_ORIGIN }, "pi", async ({ host, notes, logged, ctx }) => {
        const c = ctx({ model: bridgeModel, catalog: catalogOf(bridgeModel) });
        await host.emit("session_start", c, startup);
        await host.emit("session_start", c, { type: "session_start", reason: "new" });
        await host.emit("model_select", c, { type: "model_select", model: bridgeModel, previousModel: httpModel, source: "set" });
        await host.emit("agent_start", c);
        assert.equal(notes.length, 1);
        assert.equal(logged(), 1);
    });
});

test("a bridge registered after startup is caught by model_select or agent_start", async () => {
    await withWarnHost({ BILLION_CONTEXT_PROXY: INLINE_ORIGIN }, "pi", async ({ host, notes, logged, ctx }) => {
        await host.emit("session_start", ctx({ catalog: catalogOf(httpModel) }), startup);
        await host.emit("model_select", ctx({ catalog: catalogOf(httpModel) }), { type: "model_select", model: httpModel, previousModel: undefined, source: "set" });
        assert.equal(notes.length, 0, "no bridge yet");
        await host.emit("model_select", ctx({ catalog: catalogOf(httpModel) }), { type: "model_select", model: bridgeModel, previousModel: httpModel, source: "set" });
        assert.equal(notes.length, 1, "the selected model in the event counts");
        await host.emit("agent_start", ctx({ model: bridgeModel, catalog: catalogOf(bridgeModel) }));
        assert.equal(logged(), 1);
    });
    await withWarnHost({ BILLION_CONTEXT_PROXY: INLINE_ORIGIN }, "pi", async ({ host, notes, ctx }) => {
        await host.emit("session_start", ctx({ catalog: catalogOf(httpModel) }), startup);
        await host.emit("agent_start", ctx({ catalog: catalogOf(httpModel, bridgeModel) }));
        assert.equal(notes.length, 1, "a late catalog entry counts while an HTTP model stays selected");
    });
    await withWarnHost({}, "pi", async ({ host, notes, ctx }) => {
        await host.emit("session_start", ctx({ catalog: catalogOf(bridgeModel) }), startup);
        assert.equal(notes.length, 0, "no lane in effect yet");
        process.env.BILLION_CONTEXT_PROXY = INLINE_ORIGIN;
        await host.emit("agent_start", ctx({ catalog: catalogOf(bridgeModel) }));
        await host.emit("agent_start", ctx({ catalog: catalogOf(bridgeModel) }));
        assert.equal(notes.length, 1, "the lane taking effect later warns once");
    });
});

test("a missing or broken host UI never breaks startup and the log line still lands", async () => {
    const throwingUi = Object.defineProperty({}, "notify", { get: () => { throw new Error("stale"); } });
    for (const [name, ui] of [["no ui", undefined], ["no notify", {}], ["notify throws", { notify: () => { throw new Error("boom"); } }], ["notify getter throws", throwingUi]] as const) {
        await withWarnHost({ BILLION_CONTEXT_PROXY: INLINE_ORIGIN }, "pi", async ({ host, logged, ctx }) => {
            await host.emit("session_start", ctx({ model: bridgeModel, ui }), startup);
            await host.emit("agent_start", ctx({ model: bridgeModel, ui }));
            assert.equal(logged(), 1, name);
        });
    }
});
