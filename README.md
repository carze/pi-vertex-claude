# Vertex Claude Provider for Pi

Access Claude models via Google Vertex AI.

## Installation

```bash
pi install npm:@isaacraja/pi-vertex-claude
```

## Setup

Authenticate with Google Cloud:

```bash
gcloud auth application-default login
```

Set your project:

```bash
export GOOGLE_CLOUD_PROJECT=your-project-id
```

Use the provider:

```bash
pi --provider google-vertex-claude --model claude-sonnet-4@20250514
```

## Shell Helper

Add to `~/.bashrc` or `~/.zshrc`:

```bash
piv() {
  GOOGLE_CLOUD_PROJECT=your-project-id \
  pi --provider google-vertex-claude --model claude-sonnet-4@20250514 "$@"
}
```

## Available Models

| Model | Context | Output |
|-------|---------|--------|
| `claude-opus-4-7` | 1M | 128K |
| `claude-opus-4-6` | 1M | 128K |
| `claude-sonnet-4-6` | 1M | 64K |
| `claude-opus-4-5@20251101` | 200K | 32K |
| `claude-opus-4-1@20250805` | 200K | 32K |
| `claude-opus-4@20250514` | 200K | 32K |
| `claude-sonnet-4-5@20250929` | 200K | 64K |
| `claude-sonnet-4@20250514` | 200K | 64K |
| `claude-3-7-sonnet@20250219` | 200K | 64K |
| `claude-haiku-4-5@20251001` | 200K | 64K |
| `claude-3-5-sonnet-v2@20241022` | 200K | 8K |
| `claude-3-5-haiku@20241022` | 200K | 8K |

## Prerequisites

- Google Cloud project with Vertex AI API enabled
- Claude models enabled in [Model Garden](https://console.cloud.google.com/vertex-ai/model-garden)
- `gcloud` CLI installed

## License

MIT
