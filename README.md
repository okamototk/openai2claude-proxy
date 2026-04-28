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
- `PROXY_AUTO_WEB_SEARCH` (default `true`; set `false` to disable OpenAI server-side `web_search` injection)

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

### Web search hangs: quick fix
If Claude Code gets stuck after showing a `Web Search(...)` tool call, apply this checklist:

1. Use the latest Claude Code:
```bash
bun install -g @anthropic-ai/claude-code@latest
```
2. Enable tool search explicitly for custom proxy base URLs:
```bash
export ENABLE_TOOL_SEARCH=true
```
3. Prefer compatibility mapping when using GPT-5.3 codex upstream:
```bash
bun src/index.ts --model gpt-5.3-codex:gpt-5.2-codex
```
4. Verify stream markers in `server.log` for a healthy web-search roundtrip:
   - first response ends with `"stop_reason":"tool_use"`
   - web-search tool request is sent
   - second response ends with `"stop_reason":"end_turn"`
   - downstream emits `message_stop`

The proxy now logs a startup warning when `ANTHROPIC_BASE_URL` points to a non-Anthropic host and `ENABLE_TOOL_SEARCH` is not set.

Proxy-side workaround: `PROXY_AUTO_WEB_SEARCH` defaults to `true` when using `PROVIDER=openai`. This lets the upstream OpenAI Responses API use server-side `web_search` even if Claude Code fails to load its local WebSearch schema. Set `PROXY_AUTO_WEB_SEARCH=false` to disable it if web search changes behavior or incurs unwanted tool-use costs.

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
