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
let mapStopReason: typeof import("../index.js").mapStopReason;
let parseStreamingJson: typeof import("../index.js").parseStreamingJson;
let buildThinkingConfig: typeof import("../index.js").buildThinkingConfig;

beforeAll(async () => {
	const helpers = await import("../index.js");
	convertMessages = helpers.convertMessages;
	mapStopReason = helpers.mapStopReason;
	parseStreamingJson = helpers.parseStreamingJson;
	buildThinkingConfig = helpers.buildThinkingConfig;
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
