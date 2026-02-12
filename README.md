# openai2claude-proxy

openai2claude-proxy is a minimal Claude-compatible proxy that forwards `/v1/messages` to an upstream provider (OpenAI-compatible `/responses` or Gemini `generateContent`).

## Highlights
- **Claude Code compatible**:
  - proxies OpenAI-style APIs to Claude `/v1/messages`, and **bypasses built-in tools** like `web_search` that other proxies may fail on.
- **OpenAI API + OpenRouter + Gemini support**:
  - access gpt-5.x-codex, Kimi K2.5, MiniMax M2.1, Grok 4.2, Qwen3 Coder Next, and similar models on OpenAI API and OpenRouter.
- Launch in **3 sec**:
  - TypeScript instead of Python. Say goodbye to slow Python.
  - Zero third-party dependencies for a tiny, fast footprint.
  - Runs on Bun (not Node) for near-instant startup.

## Requirements
- Bun
This guide uses Bun to run the proxy.


## Install Runtime

macOS:
```bash
brew install bun
```

Linux (Debian/Ubuntu):
```bash
curl -fsSL https://bun.sh/install | bash
```

Windows:
Use WSL (Windows Subsystem for Linux) and follow the Linux steps above.

Note: openai2claude-proxy is executed by bun to run TypeScript code.

## Configure
Set env vars (example for OpenAI):
```bash
export OPENAI_API_KEY=your_key
```

For OpenRouter:
```bash
export PROVIDER=openrouter
export OPENROUTER_API_KEY=your_key
```

For Gemini:
```bash
export PROVIDER=gemini
export GEMINI_API_KEY=your_key
```

Optional overrides:
- `OPENAI_BASE_URL` (default `https://api.openai.com/v1`)
- `OPENROUTER_BASE_URL` (default `https://openrouter.ai/api/v1`)
- `GEMINI_BASE_URL` (default `https://generativelanguage.googleapis.com/v1beta`)
- `PORT` (default `3000`)
- `BIND_ADDRESS` (default `127.0.0.1`)
- `VERBOSE_LOGGING` (`true` to enable request/response logging)

## Run
```bash
bunx github:okamototk/openai2claude-proxy --model gpt-5.2-codex
```

## Claude Code Configuration
- Install claude-code:
```bash
bun install @claude-ai/claude-code
```
- Configure claude-code env for openai2claude-proxy:
```bash
export ANTHROPIC_BASE_URL=http://localhost:3000
export ANTHROPIC_API_KEY=local
```
- Run claude-code with openai2claude-proxy model:
```bash
claude --model gpt-5.2-codex
```

## Endpoints
- `GET /health`
- `GET /`
- `POST /v1/messages`
- `GET /v1/models`

## Model selection
Use CLI flags to choose the upstream and downstream models at startup:
```bash
# same upstream/downstream
bun src/index.ts --model gpt-5.2-codex
# or
bun src/index.ts -m gpt-5.2-codex

# map upstream:downstream
bun src/index.ts --model gpt-5.3-codex:gpt-5.2-codex

# escape ':' in model names
bun src/index.ts --model openai/gpt-oss-120b\\:free
```

If no model is provided, the server defaults to `gpt-5.2-codex` for both upstream and downstream.

## License
MIT
