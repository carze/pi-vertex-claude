import { mkdtempSync, mkdirSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { beforeAll, describe, expect, it, vi } from "vitest";

vi.mock(
	"@mariozechner/pi-ai",
	() => ({
		calculateCost: () => undefined,
		createAssistantMessageEventStream: () => ({
			push: () => undefined,
			end: () => undefined,
			[Symbol.asyncIterator]: () => ({
				next: async () => ({ done: true, value: undefined }),
			}),
		}),
	}),
	{ virtual: true },
);

let convertMessages: typeof import("../index.js").convertMessages;
let convertTools: typeof import("../index.js").convertTools;
let mapStopReason: typeof import("../index.js").mapStopReason;
let parseStreamingJson: typeof import("../index.js").parseStreamingJson;
let buildThinkingConfig: typeof import("../index.js").buildThinkingConfig;
let repairJson: typeof import("../index.js").repairJson;
let parseJsonWithRepair: typeof import("../index.js").parseJsonWithRepair;
let iterateSseMessages: typeof import("../index.js").iterateSseMessages;
let iterateAnthropicEvents: typeof import("../index.js").iterateAnthropicEvents;
let normalizeToolCallId: typeof import("../index.js").normalizeToolCallId;
let synthesizeMissingToolResults: typeof import("../index.js").synthesizeMissingToolResults;
let mapReasoningToEffort: typeof import("../index.js").mapReasoningToEffort;
let hasOpus47ApiRestrictions: typeof import("../index.js").hasOpus47ApiRestrictions;
let readSettingsEnv: typeof import("../index.js").readSettingsEnv;
let resolveSettingsEnv: typeof import("../index.js").resolveSettingsEnv;

beforeAll(async () => {
	const helpers = await import("../index.js");
	convertMessages = helpers.convertMessages;
	convertTools = helpers.convertTools;
	mapStopReason = helpers.mapStopReason;
	parseStreamingJson = helpers.parseStreamingJson;
	buildThinkingConfig = helpers.buildThinkingConfig;
	repairJson = helpers.repairJson;
	parseJsonWithRepair = helpers.parseJsonWithRepair;
	iterateSseMessages = helpers.iterateSseMessages;
	iterateAnthropicEvents = helpers.iterateAnthropicEvents;
	normalizeToolCallId = helpers.normalizeToolCallId;
	synthesizeMissingToolResults = helpers.synthesizeMissingToolResults;
	mapReasoningToEffort = helpers.mapReasoningToEffort;
	hasOpus47ApiRestrictions = helpers.hasOpus47ApiRestrictions;
	readSettingsEnv = helpers.readSettingsEnv;
	resolveSettingsEnv = helpers.resolveSettingsEnv;
});

describe("vertex-claude helpers", () => {
	it("parses partial JSON", () => {
		const result = parseStreamingJson("{\"a\": 1");
		expect(result).toMatchObject({ a: 1 });
	});

	it("returns empty object for empty input", () => {
		expect(parseStreamingJson("")).toEqual({});
	});

	it("maps known stop reasons and throws on unknown", () => {
		expect(mapStopReason("end_turn")).toBe("stop");
		expect(mapStopReason("tool_use")).toBe("toolUse");
		expect(() => mapStopReason("unknown")).toThrow(/Unhandled stop reason/);
	});

	it("adds cache_control to last tool_result block", () => {
		const model = {
			id: "test-model",
			name: "Test Model",
			api: "vertex-claude-api",
			provider: "google-vertex-claude",
			reasoning: false,
			input: ["text", "image"],
			cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
			contextWindow: 1000,
			maxTokens: 100,
		} as const;

		const messages = [
			{ role: "user", content: "hi" },
			{
				role: "toolResult",
				toolCallId: "tool-1",
				content: [{ type: "text", text: "ok" }],
				isError: false,
			},
		];

		const params = convertMessages(messages as any, model as any);
		const lastMessage = params[params.length - 1];
		const lastBlock = lastMessage.content[lastMessage.content.length - 1];

		expect(lastBlock.type).toBe("tool_result");
		expect(lastBlock.cache_control).toEqual({ type: "ephemeral" });
	});

	it("returns adaptive thinking with display=summarized and effort for Opus 4.7", () => {
		const result = buildThinkingConfig("claude-opus-4-7", "high", 16000);
		expect(result.thinking).toEqual({ type: "adaptive", display: "summarized" });
		expect(result.effort).toBe("high");
		expect(result.maxTokens).toBe(16000);
	});

	it("maps xhigh reasoning to effort=xhigh on Opus 4.7", () => {
		const result = buildThinkingConfig("claude-opus-4-7", "xhigh", 16000);
		expect(result.thinking).toEqual({ type: "adaptive", display: "summarized" });
		expect(result.effort).toBe("xhigh");
	});

	it("maps xhigh reasoning to effort=high on non-4.7 models (defensive)", () => {
		// Not currently reachable through this code path (only Opus 4.7 goes
		// adaptive here) but guards mapReasoningToEffort contract.
		expect(mapReasoningToEffort("xhigh", "claude-opus-4-6")).toBe("high");
		expect(mapReasoningToEffort("xhigh", "claude-sonnet-4-6")).toBe("high");
	});

	it("maps reasoning levels to their expected effort values", () => {
		expect(mapReasoningToEffort("minimal", "claude-opus-4-7")).toBe("low");
		expect(mapReasoningToEffort("low", "claude-opus-4-7")).toBe("low");
		expect(mapReasoningToEffort("medium", "claude-opus-4-7")).toBe("medium");
		expect(mapReasoningToEffort("high", "claude-opus-4-7")).toBe("high");
		expect(mapReasoningToEffort("xhigh", "claude-opus-4-7")).toBe("xhigh");
		expect(mapReasoningToEffort("bogus", "claude-opus-4-7")).toBe("high");
	});

	it("flags Opus 4.7 (and variants) as having sampling-param restrictions", () => {
		expect(hasOpus47ApiRestrictions("claude-opus-4-7")).toBe(true);
		expect(hasOpus47ApiRestrictions("claude-opus-4-7@20260115")).toBe(true);
	});

	it("does not flag Opus 4.6 / Sonnet 4.6 / older models as restricted", () => {
		expect(hasOpus47ApiRestrictions("claude-opus-4-6")).toBe(false);
		expect(hasOpus47ApiRestrictions("claude-opus-4-6@20251101")).toBe(false);
		expect(hasOpus47ApiRestrictions("claude-sonnet-4-6")).toBe(false);
		expect(hasOpus47ApiRestrictions("claude-sonnet-4@20250514")).toBe(false);
	});

	it("returns extended thinking with budget and display=summarized for non-Opus-4.7 models", () => {
		const result = buildThinkingConfig("claude-opus-4-6", "high", 32000);
		expect(result.thinking).toEqual({ type: "enabled", budget_tokens: 20480, display: "summarized" });
		expect(result.maxTokens).toBe(32000);
		expect(result.effort).toBeUndefined();
	});

	it("returns extended thinking for Sonnet 4.6", () => {
		const result = buildThinkingConfig("claude-sonnet-4-6", "medium", 64000);
		expect(result.thinking).toEqual({ type: "enabled", budget_tokens: 10240, display: "summarized" });
	});

	it("adjusts maxTokens when budget exceeds it", () => {
		const result = buildThinkingConfig("claude-opus-4-6", "high", 1000);
		expect(result.thinking).toEqual({ type: "enabled", budget_tokens: 20480, display: "summarized" });
		expect(result.maxTokens).toBe(20480 + 1024);
	});

	it("does not adjust maxTokens for adaptive thinking", () => {
		const result = buildThinkingConfig("claude-opus-4-7", "high", 1000);
		expect(result.maxTokens).toBe(1000);
	});

	it("uses custom thinking budgets when provided", () => {
		const result = buildThinkingConfig("claude-opus-4-6", "medium", 32000, { medium: 8000 });
		expect(result.thinking).toEqual({ type: "enabled", budget_tokens: 8000, display: "summarized" });
	});

	it("adds cache_control to last text block in user content arrays", () => {
		const model = {
			id: "test-model",
			name: "Test Model",
			api: "vertex-claude-api",
			provider: "google-vertex-claude",
			reasoning: false,
			input: ["text", "image"],
			cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
			contextWindow: 1000,
			maxTokens: 100,
		} as const;

		const messages = [
			{
				role: "user",
				content: [
					{ type: "text", text: "hello" },
					{ type: "image", data: "base64", mimeType: "image/png" },
				],
			},
		];

		const params = convertMessages(messages as any, model as any);
		const lastMessage = params[params.length - 1];
		const lastBlock = lastMessage.content[lastMessage.content.length - 1];

		expect(lastBlock.type).toBe("image");
		expect(lastBlock.cache_control).toEqual({ type: "ephemeral" });
	});
});

describe("convertTools", () => {
	const tool = {
		name: "read_file",
		description: "Read a file",
		parameters: {
			type: "object",
			properties: { path: { type: "string" } },
			required: ["path"],
		},
	};

	it("sets eager_input_streaming: true on every tool", () => {
		const result = convertTools([tool as any, { ...tool, name: "write_file" } as any]);
		expect(result).toHaveLength(2);
		expect(result[0].eager_input_streaming).toBe(true);
		expect(result[1].eager_input_streaming).toBe(true);
	});

	it("preserves name, description, and input_schema", () => {
		const [converted] = convertTools([tool as any]);
		expect(converted.name).toBe("read_file");
		expect(converted.description).toBe("Read a file");
		expect(converted.input_schema).toEqual({
			type: "object",
			properties: { path: { type: "string" } },
			required: ["path"],
		});
	});

	it("defaults missing properties and required to empty collections", () => {
		const bare = { name: "noop", description: "", parameters: { type: "object" } };
		const [converted] = convertTools([bare as any]);
		expect(converted.input_schema.properties).toEqual({});
		expect(converted.input_schema.required).toEqual([]);
	});
});

describe("normalizeToolCallId", () => {
	it("leaves already-conforming ids untouched", () => {
		expect(normalizeToolCallId("toolu_01ABCdef-_")).toBe("toolu_01ABCdef-_");
	});

	it("replaces disallowed characters with underscore", () => {
		expect(normalizeToolCallId("call:id.with|pipes")).toBe("call_id_with_pipes");
	});

	it("truncates ids longer than 64 characters", () => {
		const long = "a".repeat(100);
		const result = normalizeToolCallId(long);
		expect(result).toHaveLength(64);
		expect(result).toBe("a".repeat(64));
	});

	it("preserves deterministic mapping so tool_use and tool_result stay linked", () => {
		const raw = "call_abc:xyz.1";
		expect(normalizeToolCallId(raw)).toBe(normalizeToolCallId(raw));
	});
});

describe("convertMessages tool id normalization", () => {
	const model = {
		id: "test-model",
		name: "Test Model",
		api: "vertex-claude-api",
		provider: "google-vertex-claude",
		reasoning: false,
		input: ["text", "image"],
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
		contextWindow: 1000,
		maxTokens: 100,
	} as const;

	it("normalizes tool_use.id on assistant toolCall blocks", () => {
		const messages = [
			{
				role: "assistant",
				content: [{ type: "toolCall", id: "call:abc.xyz", name: "read", arguments: {} }],
			},
		];
		const [assistant] = convertMessages(messages as any, model as any);
		expect(assistant.content[0].type).toBe("tool_use");
		expect(assistant.content[0].id).toBe("call_abc_xyz");
	});

	it("normalizes tool_use_id on tool_result blocks, matching the paired tool_use", () => {
		const messages = [
			{
				role: "assistant",
				content: [{ type: "toolCall", id: "call:abc.xyz", name: "read", arguments: {} }],
			},
			{
				role: "toolResult",
				toolCallId: "call:abc.xyz",
				content: [{ type: "text", text: "ok" }],
				isError: false,
			},
			{
				role: "toolResult",
				toolCallId: "call:second!one",
				content: [{ type: "text", text: "ok2" }],
				isError: false,
			},
		];
		const params = convertMessages(messages as any, model as any);
		const assistant = params[0];
		const userWithResults = params[1];
		expect(assistant.content[0].id).toBe("call_abc_xyz");
		expect(userWithResults.content[0].tool_use_id).toBe("call_abc_xyz");
		expect(userWithResults.content[1].tool_use_id).toBe("call_second_one");
	});
});

describe("synthesizeMissingToolResults", () => {
	it("leaves fully resolved transcripts untouched", () => {
		const messages = [
			{ role: "user", content: "hi", timestamp: 0 },
			{
				role: "assistant",
				content: [{ type: "toolCall", id: "t1", name: "read", arguments: {} }],
			},
			{
				role: "toolResult",
				toolCallId: "t1",
				toolName: "read",
				content: [{ type: "text", text: "ok" }],
				isError: false,
				timestamp: 0,
			},
		];
		const result = synthesizeMissingToolResults(messages as any);
		expect(result).toHaveLength(3);
	});

	it("synthesizes a trailing tool_result for an unresolved tool_use at the end", () => {
		const messages = [
			{ role: "user", content: "read the file", timestamp: 0 },
			{
				role: "assistant",
				content: [{ type: "toolCall", id: "call_orphan", name: "read", arguments: {} }],
			},
		];
		const result = synthesizeMissingToolResults(messages as any);
		expect(result).toHaveLength(3);
		const last = result[result.length - 1] as any;
		expect(last.role).toBe("toolResult");
		expect(last.toolCallId).toBe("call_orphan");
		expect(last.toolName).toBe("read");
		expect(last.isError).toBe(true);
		expect(last.content).toEqual([{ type: "text", text: "No result provided" }]);
	});

	it("synthesizes only for tool calls still missing a result, not ones already resolved", () => {
		const messages = [
			{
				role: "assistant",
				content: [
					{ type: "toolCall", id: "t1", name: "read", arguments: {} },
					{ type: "toolCall", id: "t2", name: "bash", arguments: {} },
				],
			},
			{
				role: "toolResult",
				toolCallId: "t1",
				toolName: "read",
				content: [{ type: "text", text: "ok" }],
				isError: false,
				timestamp: 0,
			},
		];
		const result = synthesizeMissingToolResults(messages as any);
		const synthesized = result.filter((m: any) => m.role === "toolResult" && m.isError);
		expect(synthesized).toHaveLength(1);
		expect((synthesized[0] as any).toolCallId).toBe("t2");
		expect((synthesized[0] as any).toolName).toBe("bash");
	});

	it("flushes orphans before a user message interrupts the tool flow", () => {
		const messages = [
			{
				role: "assistant",
				content: [{ type: "toolCall", id: "orphan", name: "read", arguments: {} }],
			},
			{ role: "user", content: "never mind", timestamp: 0 },
		];
		const result = synthesizeMissingToolResults(messages as any);
		expect(result).toHaveLength(3);
		expect(result[0].role).toBe("assistant");
		expect(result[1].role).toBe("toolResult");
		expect((result[1] as any).toolCallId).toBe("orphan");
		expect(result[2].role).toBe("user");
	});

	it("flushes orphans before a subsequent assistant message", () => {
		const messages = [
			{
				role: "assistant",
				content: [{ type: "toolCall", id: "first_orphan", name: "read", arguments: {} }],
			},
			{ role: "assistant", content: [{ type: "text", text: "retrying" }] },
		];
		const result = synthesizeMissingToolResults(messages as any);
		expect(result).toHaveLength(3);
		expect(result[1].role).toBe("toolResult");
		expect((result[1] as any).toolCallId).toBe("first_orphan");
		expect(result[2].role).toBe("assistant");
	});
});

describe("convertMessages orphan tool results", () => {
	const model = {
		id: "test-model",
		name: "Test Model",
		api: "vertex-claude-api",
		provider: "google-vertex-claude",
		reasoning: false,
		input: ["text", "image"],
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
		contextWindow: 1000,
		maxTokens: 100,
	} as const;

	it("emits a synthetic tool_result so the Anthropic payload is well-formed", () => {
		const messages = [
			{ role: "user", content: "hi", timestamp: 0 },
			{
				role: "assistant",
				content: [{ type: "toolCall", id: "call:orphan", name: "read", arguments: {} }],
			},
		];
		const params = convertMessages(messages as any, model as any);
		const last = params[params.length - 1];
		expect(last.role).toBe("user");
		expect(last.content[0].type).toBe("tool_result");
		expect(last.content[0].tool_use_id).toBe("call_orphan");
		expect(last.content[0].is_error).toBe(true);
	});
});

describe("convertMessages thinking block conversion", () => {
	const vertexModel = {
		id: "claude-opus-4-7",
		name: "Claude Opus 4.7 (Vertex)",
		api: "vertex-claude-api",
		provider: "google-vertex-claude",
		reasoning: true,
		input: ["text", "image"],
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
		contextWindow: 1000,
		maxTokens: 100,
	} as const;

	it("preserves thinking signature when assistant came from same api+provider", () => {
		const messages = [
			{ role: "user", content: "hi", timestamp: 0 },
			{
				role: "assistant",
				api: "vertex-claude-api",
				provider: "google-vertex-claude",
				model: "claude-opus-4-7",
				content: [
					{ type: "thinking", thinking: "reasoning here", thinkingSignature: "abc123" },
					{ type: "text", text: "hello" },
				],
			},
		];
		const params = convertMessages(messages as any, vertexModel as any);
		const assistant = params[1];
		expect(assistant.role).toBe("assistant");
		expect(assistant.content[0]).toEqual({
			type: "thinking",
			thinking: "reasoning here",
			signature: "abc123",
		});
	});

	it("does not sanitize thinking content when signature is preserved", () => {
		// sanitizeSurrogates would mutate the signed payload and invalidate the signature
		const loneHigh = "\uD83D"; // lone high surrogate; would normally be replaced with U+FFFD
		const messages = [
			{ role: "user", content: "hi", timestamp: 0 },
			{
				role: "assistant",
				api: "vertex-claude-api",
				provider: "google-vertex-claude",
				model: "claude-opus-4-7",
				content: [
					{ type: "thinking", thinking: `text${loneHigh}more`, thinkingSignature: "abc123" },
				],
			},
		];
		const params = convertMessages(messages as any, vertexModel as any);
		const assistant = params[1];
		expect(assistant.content[0].thinking).toBe(`text${loneHigh}more`);
		expect(assistant.content[0].thinking).not.toContain("�");
	});

	it("wraps thinking from a foreign provider in <external-reasoning> tags and drops the foreign signature", () => {
		const messages = [
			{ role: "user", content: "hi", timestamp: 0 },
			{
				role: "assistant",
				api: "openai-completions",
				provider: "kimi-coding",
				model: "k2-turbo",
				content: [
					{ type: "thinking", thinking: "kimi's analysis", thinkingSignature: "kimi-sig" },
					{ type: "text", text: "answer" },
				],
			},
		];
		const params = convertMessages(messages as any, vertexModel as any);
		const assistant = params[1];
		expect(assistant.content[0]).toEqual({
			type: "text",
			text: "<external-reasoning>\nkimi's analysis\n</external-reasoning>",
		});
		// Foreign signature must not leak through under any field name
		expect(JSON.stringify(assistant.content)).not.toContain("kimi-sig");
	});

	it("wraps thinking when only the api differs (e.g. Anthropic Direct vs Vertex)", () => {
		const messages = [
			{ role: "user", content: "hi", timestamp: 0 },
			{
				role: "assistant",
				api: "anthropic-messages",
				provider: "anthropic",
				model: "claude-opus-4-7",
				content: [{ type: "thinking", thinking: "from direct API", thinkingSignature: "direct-sig" }],
			},
		];
		const params = convertMessages(messages as any, vertexModel as any);
		const assistant = params[1];
		expect(assistant.content[0].type).toBe("text");
		expect(assistant.content[0].text).toContain("<external-reasoning>");
		expect(assistant.content[0].text).toContain("from direct API");
		expect(JSON.stringify(assistant.content)).not.toContain("direct-sig");
	});

	it("converts same-provider thinking to plain text (unwrapped) when signature is missing", () => {
		const messages = [
			{ role: "user", content: "hi", timestamp: 0 },
			{
				role: "assistant",
				api: "vertex-claude-api",
				provider: "google-vertex-claude",
				model: "claude-opus-4-7",
				content: [{ type: "thinking", thinking: "no sig here" }],
			},
		];
		const params = convertMessages(messages as any, vertexModel as any);
		const assistant = params[1];
		expect(assistant.content[0]).toEqual({ type: "text", text: "no sig here" });
		expect(assistant.content[0].text).not.toContain("<external-reasoning>");
	});

	it("treats messages without api/provider metadata as foreign (defensive default)", () => {
		const messages = [
			{ role: "user", content: "hi", timestamp: 0 },
			{
				role: "assistant",
				content: [{ type: "thinking", thinking: "orphan", thinkingSignature: "stale-sig" }],
			},
		];
		const params = convertMessages(messages as any, vertexModel as any);
		const assistant = params[1];
		expect(assistant.content[0].type).toBe("text");
		expect(assistant.content[0].text).toContain("<external-reasoning>");
		expect(assistant.content[0].text).toContain("orphan");
		expect(JSON.stringify(assistant.content)).not.toContain("stale-sig");
	});
});

describe("repairJson", () => {
	it("leaves valid JSON untouched", () => {
		const input = '{"a":"b","c":1}';
		expect(repairJson(input)).toBe(input);
	});

	it("escapes raw tab characters inside strings", () => {
		const input = '{"text":"col1\tcol2"}';
		const repaired = repairJson(input);
		expect(JSON.parse(repaired)).toEqual({ text: "col1\tcol2" });
	});

	it("escapes raw newlines inside strings", () => {
		const input = '{"text":"line1\nline2"}';
		expect(JSON.parse(repairJson(input))).toEqual({ text: "line1\nline2" });
	});

	it("doubles backslash before invalid escape sequences like \\H", () => {
		const input = String.raw`{"path":"A\H"}`;
		const repaired = repairJson(input);
		expect(JSON.parse(repaired)).toEqual({ path: String.raw`A\H` });
	});

	it("preserves valid 4-digit unicode escapes", () => {
		const input = String.raw`{"text":"é"}`;
		expect(repairJson(input)).toBe(input);
		expect(JSON.parse(repairJson(input))).toEqual({ text: "é" });
	});

	it("does not corrupt valid 4-digit unicode escape sequences", () => {
		const input = String.raw`{"text":"é"}`;
		expect(repairJson(input)).toBe(input);
		expect(JSON.parse(repairJson(input))).toEqual({ text: "é" });
	});

	it("handles trailing lone backslash at end of string", () => {
		const input = '{"text":"hi\\';
		const repaired = repairJson(input);
		expect(repaired.endsWith("\\\\")).toBe(true);
	});

	it("leaves content outside strings untouched", () => {
		const input = '{"a":1,"b":2}';
		expect(repairJson(input)).toBe(input);
	});
});

describe("parseJsonWithRepair", () => {
	it("parses valid JSON without modification", () => {
		expect(parseJsonWithRepair<{ a: number }>('{"a":1}')).toEqual({ a: 1 });
	});

	it("repairs and parses JSON containing raw tab", () => {
		expect(parseJsonWithRepair<{ text: string }>('{"text":"a\tb"}')).toEqual({ text: "a\tb" });
	});

	it("repairs invalid \\H escape and parses", () => {
		const input = String.raw`{"path":"A\H"}`;
		expect(parseJsonWithRepair<{ path: string }>(input)).toEqual({ path: String.raw`A\H` });
	});

	it("throws when JSON is structurally unrecoverable", () => {
		expect(() => parseJsonWithRepair<unknown>("{not-json")).toThrow();
	});
});

describe("parseStreamingJson", () => {
	it("repairs the pi-mono malformed tool JSON repro", () => {
		const malformed = String.raw`{"path":"A\H","text":"col1\tcol2"}`;
		expect(parseStreamingJson(malformed)).toEqual({
			path: String.raw`A\H`,
			text: "col1\tcol2",
		});
	});

	it("tolerates partially streamed JSON and returns best-effort object", () => {
		const partial = '{"path":"A","text":"col1\tcol';
		const result = parseStreamingJson(partial);
		expect(result).toMatchObject({ path: "A" });
	});
});

function makeSseResponse(events: Array<{ event: string; data: string }>): Response {
	const body = events.map(({ event, data }) => `event: ${event}\ndata: ${data}\n`).join("\n");
	return new Response(body, {
		status: 200,
		headers: { "content-type": "text/event-stream" },
	});
}

describe("iterateSseMessages", () => {
	it("yields one event per SSE envelope", async () => {
		const response = makeSseResponse([
			{ event: "a", data: '{"x":1}' },
			{ event: "b", data: '{"y":2}' },
		]);
		const events: Array<{ event: string | null; data: string }> = [];
		for await (const ev of iterateSseMessages(response.body!)) {
			events.push({ event: ev.event, data: ev.data });
		}
		expect(events).toEqual([
			{ event: "a", data: '{"x":1}' },
			{ event: "b", data: '{"y":2}' },
		]);
	});

	it("aborts mid-stream when signal is aborted", async () => {
		const controller = new AbortController();
		const body = new ReadableStream<Uint8Array>({
			start(controller2) {
				controller2.enqueue(new TextEncoder().encode("event: a\ndata: 1\n\n"));
			},
		});
		controller.abort();
		await expect(async () => {
			for await (const _ of iterateSseMessages(body, controller.signal)) {
				// no-op
			}
		}).rejects.toThrow(/aborted/i);
	});
});

describe("iterateAnthropicEvents", () => {
	it("parses well-formed JSON envelopes", async () => {
		const response = makeSseResponse([
			{ event: "message_start", data: JSON.stringify({ type: "message_start" }) },
			{ event: "message_stop", data: JSON.stringify({ type: "message_stop" }) },
		]);
		const types: string[] = [];
		for await (const ev of iterateAnthropicEvents(response)) {
			types.push((ev as any).type);
		}
		expect(types).toEqual(["message_start", "message_stop"]);
	});

	it("repairs malformed tool JSON deltas that SDK's default parser would reject", async () => {
		// `\H` is an invalid JSON escape (SDK's JSON.parse hard-fails on this).
		// `\t` is a valid JSON escape and resolves to a real tab character when parsed.
		const malformedDelta = String.raw`{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"path\":\"A\H\",\"text\":\"col1\tcol2\"}"}}`;
		const response = makeSseResponse([{ event: "content_block_delta", data: malformedDelta }]);
		const events: any[] = [];
		for await (const ev of iterateAnthropicEvents(response)) {
			events.push(ev);
		}
		expect(events).toHaveLength(1);
		expect(events[0].delta.partial_json).toBe('{"path":"A\\H","text":"col1\tcol2"}');
	});

	it("skips ping events", async () => {
		const response = makeSseResponse([
			{ event: "ping", data: "{}" },
			{ event: "message_stop", data: JSON.stringify({ type: "message_stop" }) },
		]);
		const types: string[] = [];
		for await (const ev of iterateAnthropicEvents(response)) {
			types.push((ev as any).type);
		}
		expect(types).toEqual(["message_stop"]);
	});

	it("throws on SSE 'error' events", async () => {
		const response = makeSseResponse([{ event: "error", data: "overloaded" }]);
		await expect(async () => {
			for await (const _ of iterateAnthropicEvents(response)) {
				// no-op
			}
		}).rejects.toThrow(/overloaded/);
	});
});

describe("settings.json env cascade", () => {
	function writeSettings(dir: string, fileName: string, body: unknown): string {
		const claudeDir = join(dir, ".claude");
		mkdirSync(claudeDir, { recursive: true });
		const filePath = join(claudeDir, fileName);
		writeFileSync(filePath, JSON.stringify(body));
		return filePath;
	}

	it("returns env subset from a valid settings file", () => {
		const dir = mkdtempSync(join(tmpdir(), "pvc-settings-"));
		const filePath = writeSettings(dir, "settings.json", {
			env: {
				ANTHROPIC_VERTEX_PROJECT_ID: "my-proj",
				CLOUD_ML_REGION: "us-east5",
			},
			otherTopLevel: "ignored",
		});
		expect(readSettingsEnv(filePath)).toEqual({
			ANTHROPIC_VERTEX_PROJECT_ID: "my-proj",
			CLOUD_ML_REGION: "us-east5",
		});
	});

	it("returns {} when the file does not exist", () => {
		const missing = join(tmpdir(), "pvc-does-not-exist", "settings.json");
		expect(readSettingsEnv(missing)).toEqual({});
	});

	it("returns {} on malformed JSON", () => {
		const dir = mkdtempSync(join(tmpdir(), "pvc-settings-"));
		mkdirSync(join(dir, ".claude"), { recursive: true });
		const filePath = join(dir, ".claude", "settings.json");
		writeFileSync(filePath, "{ this is not json");
		expect(readSettingsEnv(filePath)).toEqual({});
	});

	it("filters out non-string env values", () => {
		const dir = mkdtempSync(join(tmpdir(), "pvc-settings-"));
		const filePath = writeSettings(dir, "settings.json", {
			env: {
				ANTHROPIC_VERTEX_PROJECT_ID: "my-proj",
				NESTED: { a: 1 },
				NUMERIC: 42,
				BOOLEAN: true,
			},
		});
		expect(readSettingsEnv(filePath)).toEqual({ ANTHROPIC_VERTEX_PROJECT_ID: "my-proj" });
	});

	it("returns {} when env field is missing", () => {
		const dir = mkdtempSync(join(tmpdir(), "pvc-settings-"));
		const filePath = writeSettings(dir, "settings.json", { description: "no env" });
		expect(readSettingsEnv(filePath)).toEqual({});
	});

	it("merges global home settings with working-dir local settings (local wins)", () => {
		const home = mkdtempSync(join(tmpdir(), "pvc-home-"));
		const cwd = mkdtempSync(join(tmpdir(), "pvc-cwd-"));
		writeSettings(home, "settings.json", {
			env: {
				ANTHROPIC_VERTEX_PROJECT_ID: "global-proj",
				CLOUD_ML_REGION: "global-region",
			},
		});
		writeSettings(cwd, "settings.local.json", {
			env: {
				ANTHROPIC_VERTEX_PROJECT_ID: "local-proj",
			},
		});
		expect(resolveSettingsEnv(cwd, home)).toEqual({
			ANTHROPIC_VERTEX_PROJECT_ID: "local-proj",
			CLOUD_ML_REGION: "global-region",
		});
	});

	it("returns global env only when no local settings exist", () => {
		const home = mkdtempSync(join(tmpdir(), "pvc-home-"));
		const cwd = mkdtempSync(join(tmpdir(), "pvc-cwd-"));
		writeSettings(home, "settings.json", {
			env: { ANTHROPIC_VERTEX_PROJECT_ID: "global-proj" },
		});
		expect(resolveSettingsEnv(cwd, home)).toEqual({ ANTHROPIC_VERTEX_PROJECT_ID: "global-proj" });
	});

	it("returns {} when neither file exists", () => {
		const home = mkdtempSync(join(tmpdir(), "pvc-home-"));
		const cwd = mkdtempSync(join(tmpdir(), "pvc-cwd-"));
		expect(resolveSettingsEnv(cwd, home)).toEqual({});
	});
});
