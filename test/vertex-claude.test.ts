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

	it("returns adaptive thinking for Opus 4.7", () => {
		const result = buildThinkingConfig("claude-opus-4-7", "high", 16000);
		expect(result.thinking).toEqual({ type: "adaptive" });
		expect(result.maxTokens).toBe(16000);
	});

	it("returns extended thinking with budget for non-Opus-4.7 models", () => {
		const result = buildThinkingConfig("claude-opus-4-6", "high", 32000);
		expect(result.thinking).toEqual({ type: "enabled", budget_tokens: 20480 });
		expect(result.maxTokens).toBe(32000);
	});

	it("returns extended thinking for Sonnet 4.6", () => {
		const result = buildThinkingConfig("claude-sonnet-4-6", "medium", 64000);
		expect(result.thinking).toEqual({ type: "enabled", budget_tokens: 10240 });
	});

	it("adjusts maxTokens when budget exceeds it", () => {
		const result = buildThinkingConfig("claude-opus-4-6", "high", 1000);
		expect(result.thinking).toEqual({ type: "enabled", budget_tokens: 20480 });
		expect(result.maxTokens).toBe(20480 + 1024);
	});

	it("does not adjust maxTokens for adaptive thinking", () => {
		const result = buildThinkingConfig("claude-opus-4-7", "high", 1000);
		expect(result.maxTokens).toBe(1000);
	});

	it("uses custom thinking budgets when provided", () => {
		const result = buildThinkingConfig("claude-opus-4-6", "medium", 32000, { medium: 8000 });
		expect(result.thinking).toEqual({ type: "enabled", budget_tokens: 8000 });
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
