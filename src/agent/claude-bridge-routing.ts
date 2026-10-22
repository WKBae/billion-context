// #3429: pi-claude-bridge serves its models from Claude Code child processes
// (provider api/baseUrl literal "claude-bridge"). Those children call
// Anthropic themselves, so the native fetch intercept and the
// before_provider_headers stamp never see them: bili gets no model requests
// for the pi conversation, the bili tools 404, and any routing the user adds
// by hand keys the fold state on the child's per-process session id, which
// the bridge re-mints on every reopen/resume.
//
// Claude Code honors ANTHROPIC_BASE_URL plus ANTHROPIC_CUSTOM_HEADERS, and the
// bridge builds a separate environment for every child it spawns. Before each
// query it emits CHILD_ENV_CHANNEL on the event bus of the pi session the
// child serves; a listener registers a transform that returns the complete
// environment for that one child (README "Claude Code child environment" in
// pi-claude-bridge). The listener below points the main conversation's
// children at the proxy's /bili/ lane and adds the x-bili-plugin* headers the
// header stamp sends, so their requests join the pi conversation's plugin
// session, keyed by the stable pi session id. Nothing is written to
// process.env, so no other child — another session's, a summary's, one the
// bridge spawns for a direct caller — can inherit the routing. A bridge
// without the event leaves its children unrouted.
import { INLINE_ORIGIN } from "../inline-origin.js";
import { asciiHeaderValue } from "./shared.js";

const CLAUDE_BRIDGE_TRANSPORT = "claude-bridge";
const ANTHROPIC_DEFAULT_UPSTREAM = "https://api.anthropic.com";
const BASE_URL_ENV = "ANTHROPIC_BASE_URL";
const CUSTOM_HEADERS_ENV = "ANTHROPIC_CUSTOM_HEADERS";
const OWNED_HEADERS = ["x-bili-plugin", "x-bili-plugin-conversation", "x-bili-plugin-context-window", "x-bili-plugin-model"];
const LOOPBACK_HOSTS = new Set(["localhost", "127.0.0.1", "[::1]"]);

/** Positive identification of pi-claude-bridge's transport. Other
 *  extension-provided transports with an opaque baseUrl do not spawn Claude
 *  Code and must stay untouched. */
export function isClaudeBridgeModel(model: { api?: unknown; baseUrl?: unknown } | undefined): boolean {
    if (model === undefined) return false;
    return model.api === CLAUDE_BRIDGE_TRANSPORT || model.baseUrl === CLAUDE_BRIDGE_TRANSPORT;
}

/** Whether a separate process can reach `origin`. The inline lane's
 *  pseudo-origin is served in-process only. */
export function childReachableOrigin(origin: string | undefined): origin is string {
    if (origin === undefined || origin === INLINE_ORIGIN) return false;
    try {
        const u = new URL(origin);
        return u.protocol === "http:" || u.protocol === "https:";
    } catch {
        return false;
    }
}

/** Origin identity for comparing proxies: loopback aliases name the same
 *  listener. */
function originKey(url: URL): string {
    const host = LOOPBACK_HOSTS.has(url.hostname) ? "loopback" : url.hostname;
    return `${url.protocol}//${host}:${url.port}`;
}

function biliRouteKey(url: string): string | undefined {
    try {
        const u = new URL(url);
        if (u.protocol !== "http:" && u.protocol !== "https:") return undefined;
        if (!/^\/bili\/(?:[a-z]+\/)?https?:\/\//.test(u.pathname)) return undefined;
        return originKey(u);
    } catch {
        return undefined;
    }
}

/** The base URL a routed child should use, or undefined when the user's own
 *  ANTHROPIC_BASE_URL already routes through a DIFFERENT bili proxy — the
 *  tools are registered against `origin`, so keying the child there would
 *  split the conversation across two proxies. A user relay is wrapped so the
 *  proxy forwards to it; a base already routed through `origin` is kept. */
export function routedBaseUrl(origin: string, userBase: string | undefined): string | undefined {
    const base = userBase?.trim() ? userBase.trim().replace(/\/+$/, "") : ANTHROPIC_DEFAULT_UPSTREAM;
    const routedVia = biliRouteKey(base);
    if (routedVia !== undefined) return routedVia === originKey(new URL(origin)) ? base : undefined;
    return `${origin}/bili/${base}`;
}

function ownedHeaderLine(line: string): boolean {
    return OWNED_HEADERS.includes(line.split(":", 1)[0]?.trim().toLowerCase() ?? "");
}

type BridgeRoutingHeaders = { agent: string; conversationId: string; contextWindow?: number; model?: string };

/** Header-safe ids only: a value carrying whitespace or control characters
 *  could split into extra header lines. */
export function headerSafeId(id: string | undefined): id is string {
    return typeof id === "string" && asciiHeaderValue(id) !== undefined;
}

/** The user's ANTHROPIC_CUSTOM_HEADERS lines with the owned x-bili-plugin*
 *  headers replaced by this conversation's values. Matching is
 *  case-insensitive and limited to the owned names; optional values that are
 *  not header-safe are omitted. */
export function mergedCustomHeaders(userHeaders: string | undefined, h: BridgeRoutingHeaders): string {
    const kept = (userHeaders ?? "").split(/\r?\n/).filter((line) => line.trim().length > 0 && !ownedHeaderLine(line));
    const lines = [...kept, `x-bili-plugin: ${h.agent}`, `x-bili-plugin-conversation: ${h.conversationId}`];
    if (h.contextWindow !== undefined && Number.isFinite(h.contextWindow) && h.contextWindow > 0) lines.push(`x-bili-plugin-context-window: ${Math.floor(h.contextWindow)}`);
    const model = h.model === undefined ? undefined : asciiHeaderValue(h.model);
    if (model !== undefined) lines.push(`x-bili-plugin-model: ${model}`);
    return lines.join("\n");
}

/** pi-claude-bridge's per-child environment request (protocol version 1).
 *  Only the fields this extension reads; the bridge freezes the request. */
export const CHILD_ENV_CHANNEL = "claude-bridge:child-env";
type ChildEnvRequest = {
    version: 1;
    purpose: string;
    sessionId: string;
    model: { id: string; contextWindow?: number };
    register(transform: (env: Record<string, string>) => Record<string, string>): void;
};

function childEnvRequest(data: unknown): ChildEnvRequest | undefined {
    if (data === null || typeof data !== "object") return undefined;
    const r = data as Partial<ChildEnvRequest>;
    if (r.version !== 1 || typeof r.purpose !== "string" || typeof r.sessionId !== "string" || typeof r.register !== "function") return undefined;
    if (r.model === null || typeof r.model !== "object" || typeof r.model.id !== "string") return undefined;
    return r as ChildEnvRequest;
}

type BridgeChildOutcome = "routed" | "foreign-route";

/** The child environment routed through `origin` as `h`'s conversation, or
 *  undefined when the user's ANTHROPIC_BASE_URL already routes through a
 *  different bili proxy. Never mutates `env`. */
export function routedChildEnv(env: Record<string, string>, origin: string, h: BridgeRoutingHeaders): Record<string, string> | undefined {
    const base = routedBaseUrl(origin, env[BASE_URL_ENV]);
    if (base === undefined) return undefined;
    return { ...env, [BASE_URL_ENV]: base, [CUSTOM_HEADERS_ENV]: mergedCustomHeaders(env[CUSTOM_HEADERS_ENV], h) };
}

/** Handler for CHILD_ENV_CHANNEL. `originFor(sessionId)` names the
 *  child-reachable proxy this session's bili tools are registered against, or
 *  undefined when the session's children must stay unrouted. Only the main
 *  conversation's provider children are routed: summary and AskClaude children
 *  are separate conversations the proxy must not key into this session. The
 *  handler registers synchronously, as the protocol requires. */
export function claudeBridgeChildEnvListener(opts: {
    agent: string;
    originFor: (sessionId: string) => string | undefined;
    onRouted?: (sessionId: string, origin: string, outcome: BridgeChildOutcome) => void;
}): (data: unknown) => void {
    return (data) => {
        const req = childEnvRequest(data);
        if (req === undefined || req.purpose !== "provider") return;
        const sid = req.sessionId;
        if (sid.length === 0 || !headerSafeId(sid)) return;
        const origin = opts.originFor(sid);
        if (!childReachableOrigin(origin)) return;
        const headers: BridgeRoutingHeaders = { agent: opts.agent, conversationId: sid, contextWindow: req.model.contextWindow, model: req.model.id };
        req.register((env) => {
            const routed = routedChildEnv(env, origin, headers);
            opts.onRouted?.(sid, origin, routed === undefined ? "foreign-route" : "routed");
            return routed ?? env;
        });
    };
}
