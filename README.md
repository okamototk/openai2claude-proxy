# openai2claude-proxy

openai2claude-proxy is a minimal Anthropic-compatible proxy that forwards `/v1/messages` to an OpenAI-compatible `/responses` upstream (OpenAI or OpenRouter).

## Requirements
- Bun or Node
This guide use Bun for run proxy.


## Install Runtime
```bash
brew install node
brew install bun
```

Note: openai2calude-proxy is executed by bun to run TypeScript code. But need npx(node) to fetch code from GitHub repository.

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

Optional overrides:
- `OPENAI_BASE_URL` (default `https://api.openai.com/v1`)
- `OPENROUTER_BASE_URL` (default `https://openrouter.ai/api/v1`)
- `PORT` (default `3000`)
- `BIND_ADDRESS` (default `127.0.0.1`)
- `VERBOSE_LOGGING` (`true` to enable request/response logging)

## Run
```bash
npx https://github.com/okamototk/openai2claude-proxy --model gpt-5.2-codex
```

## Claude Code Configuration
- Install claude-code:
```bash
bun install @anthropic-ai/claude
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
bun src/index.ts --model arcee-ai/trinity-mini\:free:destmodel
```

If no model is provided, the server defaults to `gpt-5.2-codex` for both upstream and downstream.

## License
MIT
