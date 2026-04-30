# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.7] - 2026-04-30

### Fixed
- "Invalid `signature` in `thinking` block" 400 from Vertex when swapping providers mid-conversation (e.g. Claude → ChatGPT → Kimi → Claude). Thinking-signature passthrough is now gated on `api`+`provider` match; foreign turns are rewrapped as text inside `<external-reasoning>` tags so Claude reads them as outside context. Same-provider signed thinking is now passed unsanitized so the signed bytes match what Anthropic verifies.
- Integration smoke test switched from retired `claude-3-5-haiku@20241022` to `claude-haiku-4-5@20251001`.

## [0.1.6] - 2026-04-27

### Added
- Cascade Vertex config from `.claude/settings.{local,}.json` so `claude-code-templates --setting=api/vertex-configuration` can drive the plugin without exporting env vars.

## [0.1.5] - 2026-04-24

### Fixed
- Strip non-default sampling params (`temperature`, `top_p`, `top_k`) for Opus 4.7, which now rejects them with a 400.

### Changed
- Bump `@anthropic-ai/sdk` to `^0.90.0`.

## [0.1.4] - 2026-04-22

### Added
- Claude Opus 4.7 (`claude-opus-4-7`) with adaptive thinking and `output_config.effort` mapping.
- Claude Sonnet 4.6 (`claude-sonnet-4-6`) and 1M context windows on Opus 4.7 / Sonnet 4.6 / Opus 4.6.
- Per-tool `eager_input_streaming` (replaces deprecated `fine-grained-tool-streaming-2025-05-14` beta header).

### Fixed
- Own the Anthropic SSE parsing loop to repair malformed tool JSON (raw control characters, invalid `\` escapes) that the SDK's `JSON.parse` would otherwise reject.
- Normalize `tool_use.id` and `tool_use_id` to Anthropic's `^[a-zA-Z0-9_-]+` pattern (max 64 chars).
- Synthesize a "No result provided" `tool_result` for orphaned `tool_use` blocks left behind by aborted turns.
- Pin Opus 4.7 thinking `display` to `summarized` to keep `tool_use` partial_json delivery intact.

### Changed
- Upgrade `@anthropic-ai/vertex-sdk` to `0.16.0`.
- Rebrand package to `@carze` scope.
- Extract `buildThinkingConfig` helper and expand its test coverage.

## [0.1.3] - 2026-02-07

### Added
- Support for Claude Opus 4.6 model (`claude-opus-4-6`)

## [0.1.2] - 2025-01-30

### Changed
- Further simplified README - removed Features and Common Issues sections
- Cleaner title and structure following Pi extension conventions
- Removed copyright from LICENSE and README

### Removed
- GitHub Actions workflow (tests need peer dependencies)

## [0.1.1] - 2025-01-30

### Changed
- Simplified README - removed unnecessary sections (pricing, fish shell, excessive troubleshooting)
- Cleaner, more focused documentation

## [0.1.0] - 2025-01-30

### Added
- Initial release
- Support for all Vertex AI Claude models (Opus, Sonnet, Haiku)
- Full streaming support
- Extended thinking for reasoning models
- Tool/function calling support
- Image input support
- Prompt caching
- Token usage tracking and cost calculation
- Comprehensive test suite
- NPM and GitHub distribution
