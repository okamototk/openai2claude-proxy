# Agent Configuration

## Model Context Windows

This project uses an Claude-to-OpenAI API proxy. When the upstream model name matches one of the GPT-5 series keys below, the proxy maps `max_tokens` to `max_output_tokens` and uses the specified limits.

### GPT-5 Series Token Limits

| Model ID | Context Window (Total) | Max Input | Max Output | Notes |
|----------|------------------------|-----------|------------|-------|
| gpt-5.2 | 400,000 | 272,000 | 128,000 | Current stable flagship |
| gpt-5.2-thinking | 400,000 | 272,000 | 128,000 | Deep reasoning mode |
| gpt-5.2-codex | 400,000 | 272,000 | 128,000 | Optimized for coding (uses gpt-5.2 limits) |
| gpt-5.3-codex | 400,000 | 272,000 | 128,000 | Optimized for agents |
| gpt-5-mini | 400,000 | 272,000 | 128,000 | Faster, cheaper, same context size |
| gpt-5-pro | 1,000,000 | 728,000 | 272,000 | High-tier API only |

### Downstream Model Parameter Mapping

When the upstream model matches a GPT-5 series key, the proxy applies these downstream parameters:

```typescript
export const GPT5_MODEL_CONFIG: Record<string, { contextWindow: number; maxInput: number; maxOutput: number }> = {
  "gpt-5.2": { contextWindow: 400000, maxInput: 272000, maxOutput: 128000 },
  "gpt-5.2-thinking": { contextWindow: 400000, maxInput: 272000, maxOutput: 128000 },
  "gpt-5.3-codex": { contextWindow: 400000, maxInput: 272000, maxOutput: 128000 },
  "gpt-5-mini": { contextWindow: 400000, maxInput: 272000, maxOutput: 128000 },
  "gpt-5-pro": { contextWindow: 1000000, maxInput: 728000, maxOutput: 272000 },
};
```

### Usage

Provide model mappings via CLI. The default is `gpt-5.2-codex` for both upstream and downstream when no flags are given.

```bash
# same upstream/downstream
bun src/index.ts --model gpt-5.2-codex
# or
bun src/index.ts -m gpt-5.2-codex

# map upstream:downstream
bun src/index.ts --model gpt-5.3-codex:gpt-5.2-codex
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `PROVIDER` | Upstream API provider (`openai` or `openrouter`) | `openai` |
| `OPENAI_API_KEY` | OpenAI API key | - |
| `OPENROUTER_API_KEY` | OpenRouter API key | - |
| `OPENAI_BASE_URL` | OpenAI base URL | `https://api.openai.com/v1` |
| `OPENROUTER_BASE_URL` | OpenRouter base URL | `https://openrouter.ai/api/v1` |
| `PORT` | Server port | `8000` |
| `VERBOSE_LOGGING` | Enable verbose request/response logging (`true`) | - |

### Endpoints

- `GET /health` - Health check
- `GET /` - Config info
- `POST /v1/messages` - Claude-compatible messages endpoint
- `GET /v1/models` - List models

### References

- OpenAI API: https://developers.openai.com/api/reference/resources/responses/methods/create
- Claude API: https://platform.claude.com/docs/en/api/messages
- Gemini API: https://ai.google.dev/gemini-api/docs/text-generation?hl=ja#rest
