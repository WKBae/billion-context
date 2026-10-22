// #2970: hideConsumedCompressCalls recognized only the bare "compress" name, so
// MCP-lane hosts (tools named mcp__bili__<name>) bypassed inactive-call cleanup
// + live-range arg reduction. The host now routes the MCP compress form through
// the kernel's isCompressTool hook (wired in src/config.ts loadOptions), mirroring
// the #2578 isToolProtected wiring. Pins the predicate truth table + that wiring.
import { test } from "node:test";
import assert from "node:assert/strict";
import { createInitialState, hideConsumedCompressCalls, type CompressionBlock, type CoreMessage } from "acp-kernel";
import { loadOptions, isMcpBiliCompress } from "../src/config.ts";

test("isMcpBiliCompress recognizes exactly the compress tool forms", () => {
    // Bare name stays true so proxy / in-process lanes remain byte-identical.
    assert.equal(isMcpBiliCompress("compress"), true);
    assert.equal(isMcpBiliCompress("mcp__bili__compress"), true);
    // acp_rule is also MCP-exposed but is NOT a compress tool: matching it would
    // make its calls eligible for summary/arg stripping. Must stay false.
    assert.equal(isMcpBiliCompress("mcp__bili__acp_rule"), false);
    assert.equal(isMcpBiliCompress("acp_rule"), false);
    assert.equal(isMcpBiliCompress("mcp__bili__decompress"), false);
    assert.equal(isMcpBiliCompress("mcp__bili__search_context"), false);
    assert.equal(isMcpBiliCompress("mcp__bili__acp_status"), false);
    assert.equal(isMcpBiliCompress("mcp__other__compress"), false);
    assert.equal(isMcpBiliCompress("mcp__bili__xcompress"), false);
    assert.equal(isMcpBiliCompress("mcp__bili__compresses"), false);
    assert.equal(isMcpBiliCompress("mcp__bili__"), false);
});

test("loadOptions wires the MCP compress predicate into kernelConfig", () => {
    const pred = loadOptions({}).kernelConfig.isCompressTool;
    assert.equal(typeof pred, "function", "kernelConfig.isCompressTool is wired");
    assert.equal(pred!("compress"), true);
    assert.equal(pred!("mcp__bili__compress"), true);
    assert.equal(pred!("mcp__bili__acp_rule"), false);
    assert.equal(pred!("mcp__other__compress"), false);
    assert.equal(pred!("mcp__custom-tools__compress"), true);
});

// #3429: pi-claude-bridge exposes pi's tools to its Claude Code child through
// the MCP server "custom-tools", so bili's compress reaches the wire as
// mcp__custom-tools__compress.
test("isMcpBiliCompress recognizes pi-claude-bridge's custom-tools dialect exactly", () => {
    assert.equal(isMcpBiliCompress("mcp__custom-tools__compress"), true);
    assert.equal(isMcpBiliCompress("mcp__custom-tools__acp_rule"), false);
    assert.equal(isMcpBiliCompress("mcp__custom-tools__decompress"), false);
    for (const near of [
        "mcp__custom-tool__compress",
        "mcp__custom_tools__compress",
        "mcp__custom-tools_compress",
        "mcp__my-custom-tools__compress",
        "mcp__custom-tools__compresses",
        "mcp__custom-tools__xcompress",
        "mcp__custom-tools__mcp__bili__compress",
        "mcp__custom-tools__",
        "custom-tools__compress",
    ]) {
        assert.equal(isMcpBiliCompress(near), false, near);
    }
});

function block(fields: { blockId: string; compressCallId: string; active: boolean; startRef: string; endRef: string }): CompressionBlock {
    return {
        ...fields,
        runId: "r0",
        tier: 1,
        summary: "summary",
        directMessageIds: [],
        effectiveMessageIds: [],
        directBlockIds: [],
        createdAt: 1000,
        compressedTokens: 0,
        survivedCount: 0,
        generation: "young",
    };
}

function carrier(toolName: string, toolCallId: string, startId: string, endId: string, summary = "s"): CoreMessage[] {
    return [
        { id: `call-${toolCallId}`, role: "assistant", contentType: "tool-call", toolName, toolCallId, text: JSON.stringify({ content: [{ startId, endId, summary }] }) },
        { id: `result-${toolCallId}`, role: "tool", contentType: "tool-result", toolCallId, text: "ok" },
    ];
}

test("custom-tools compress carriers follow the consumed-call lifecycle through the wired predicate", () => {
    const isCompressTool = loadOptions({}).kernelConfig.isCompressTool!;
    const state = createInitialState();
    state.blocks.push(block({ blockId: "b-live", compressCallId: "plugin_live", active: true, startRef: "m00001", endRef: "m00050" }));
    state.blocks.push(block({ blockId: "b-dead", compressCallId: "plugin_dead", active: false, startRef: "m00060", endRef: "m00070" }));
    const msgs = [
        ...carrier("mcp__custom-tools__compress", "tool_live", "m00001", "m00050", "x".repeat(5000)),
        ...carrier("mcp__custom-tools__compress", "tool_dead", "m00060", "m00070"),
    ];
    const r = hideConsumedCompressCalls(state, msgs, isCompressTool);
    const kept = new Set(r.messages.map((m) => m.toolCallId));
    assert.ok(kept.has("tool_live"), "active carrier survives");
    assert.ok(!kept.has("tool_dead"), "inactive carrier is hidden");
    const live = r.messages.find((m) => m.contentType === "tool-call" && m.toolCallId === "tool_live")!;
    assert.equal(live.toolName, "mcp__custom-tools__compress", "wire name is preserved");
    assert.ok((live.text ?? "").length < 5000, "active carrier's long summary is reduced");
});

test("near-miss server names stay outside the consumed-call lifecycle", () => {
    const isCompressTool = loadOptions({}).kernelConfig.isCompressTool!;
    const state = createInitialState();
    state.blocks.push(block({ blockId: "b-dead", compressCallId: "plugin_dead", active: false, startRef: "m00060", endRef: "m00070" }));
    const msgs = carrier("mcp__custom-tool__compress", "tool_dead", "m00060", "m00070");
    const r = hideConsumedCompressCalls(state, msgs, isCompressTool);
    assert.equal(r.hidden, 0);
    assert.deepEqual(r.messages, msgs);
});
