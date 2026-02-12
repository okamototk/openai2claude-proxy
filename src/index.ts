#!/usr/bin/env bun

import {
  mapOpenAIToClaude,
  mapClaudeToOpenAI,
  createClaudeStream,
  type ClaudeRequest,
  type OpenAIResponse,
} from "./openai_to_claude";
import {
  mapClaudeToGemini,
  mapGeminiToClaude,
  createClaudeStreamFromGemini,
  type GeminiResponse,
} from "./gemini_to_claude";
import { buildConfig, buildUpstream, getModelMapping, getPublicConfig } from "./config";
import { patchConsoleWithTimestamps } from "./logger";

patchConsoleWithTimestamps();
const CONFIG = buildConfig(process.env, process.argv);
const upstream = buildUpstream(CONFIG);

const UPSTREAM_TIMEOUT_MS = Number(process.env.UPSTREAM_TIMEOUT_MS || "30000");
const UPSTREAM_STREAM_TIMEOUT_MS = Number(
  process.env.UPSTREAM_STREAM_TIMEOUT_MS || process.env.UPSTREAM_TIMEOUT_MS || "120000"
);

const logVerbose = (...args: unknown[]) => {
  if (CONFIG.verboseLogging) {
    console.log(...args);
  }
};

const hasImageBlocks = (messages: ClaudeRequest["messages"]) =>
  messages.some(
    (msg) => Array.isArray(msg.content) && msg.content.some((block) => block.type === "image")
  );

const systemTextFromRequest = (system: ClaudeRequest["system"]) => {
  if (!system) return "";
  if (Array.isArray(system)) return system.map((block) => block.text).join("");
  return system;
};

const textFromContent = (content: ClaudeRequest["messages"][number]["content"]) => {
  if (typeof content === "string") return content;
  const parts: string[] = [];
  for (const block of content) {
    if (block.type === "text") {
      if (block.text) parts.push(block.text);
    } else if (block.type === "tool_use") {
      parts.push(block.name);
      parts.push(JSON.stringify(block.input ?? {}));
    } else if (block.type === "tool_result") {
      if (block.content) parts.push(block.content);
    } else if (block.type === "thinking") {
      if (block.thinking) parts.push(block.thinking);
    } else if (block.type === "redacted_thinking") {
      if (block.data) parts.push(block.data);
    }
  }
  return parts.join("\n");
};

const estimateTokens = (text: string) => (text ? Math.ceil(text.length / 4) : 0);

const estimateInputTokens = (payload: ClaudeRequest) => {
  const parts: string[] = [];
  const systemText = systemTextFromRequest(payload.system);
  if (systemText) parts.push(systemText);

  for (const message of payload.messages) {
    const text = textFromContent(message.content);
    if (text) parts.push(text);
  }

  if (payload.tools) {
    for (const tool of payload.tools) {
      if (tool.name) parts.push(tool.name);
      if (tool.type) parts.push(tool.type);
      if (tool.description) parts.push(tool.description);
      if (tool.input_schema) parts.push(JSON.stringify(tool.input_schema));
      if (tool.metadata) parts.push(JSON.stringify(tool.metadata));
    }
  }

  if (payload.tool_choice) {
    parts.push(JSON.stringify(payload.tool_choice));
  }

  return estimateTokens(parts.join("\n"));
};

const logUpstreamResponse = (details: { provider: string; status: number; stream: boolean; attempt?: number }) => {
  const attemptInfo = details.attempt !== undefined ? ` attempt=${details.attempt}` : "";
  console.log(
    `[messages] Upstream response received: provider=${details.provider} status=${details.status} stream=${details.stream}${attemptInfo}`
  );
};

const fetchWithTimeout = async (input: RequestInfo | URL, init: RequestInit, timeoutMs: number) => {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(input, { ...init, signal: controller.signal });
  } finally {
    clearTimeout(timeoutId);
  }
};

const jsonResponse = (body: unknown, status = 200, headers: Record<string, string> = {}) =>
  new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json", ...headers },
  });

const sleep = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));

const parseRetryAfterSeconds = (value: string | null): number | null => {
  if (!value) return null;
  const seconds = Number(value);
  if (!Number.isNaN(seconds)) return Math.max(0, Math.ceil(seconds));
  const dateMs = Date.parse(value);
  if (Number.isNaN(dateMs)) return null;
  const diffMs = dateMs - Date.now();
  return diffMs > 0 ? Math.ceil(diffMs / 1000) : 0;
};

const parseGeminiRetryDelaySeconds = (text: string): number | null => {
  try {
    const parsed = JSON.parse(text) as {
      error?: { details?: Array<{ [key: string]: unknown }> };
    };
    const details = parsed.error?.details || [];
    for (const detail of details) {
      const type = detail["@type"];
      const retryDelay = detail["retryDelay"];
      if (typeof type === "string" && type.includes("RetryInfo") && typeof retryDelay === "string") {
        const match = retryDelay.match(/^([0-9]+(?:\.[0-9]+)?)s$/);
        if (!match) return null;
        const seconds = Number(match[1]);
        if (Number.isNaN(seconds)) return null;
        return Math.max(0, Math.ceil(seconds));
      }
    }
    return null;
  } catch {
    return null;
  }
};

const handleModels = () =>
  jsonResponse({
    object: "list",
    data: CONFIG.modelMappings.map(m => ({ id: m.downstream, object: "model" })),
  });

const handleHealth = () => jsonResponse({ status: "ok" });

const handleConfig = () => jsonResponse(getPublicConfig(CONFIG, upstream));

const handleCountTokens = async (req: Request) => {
  const payload = (await req.json()) as ClaudeRequest;
  const inputTokens = estimateInputTokens(payload);
  return jsonResponse({ input_tokens: inputTokens });
};

const handleMessages = async (req: Request) => {
  const payload = (await req.json()) as ClaudeRequest;

  logVerbose("----------------------");
  logVerbose("[messages] Downstream request parameters:", JSON.stringify(payload, null, 2));
  logVerbose("[messages] Raw tools:", JSON.stringify(payload.tools, null, 2));
  logVerbose("----------------------");

  const downstreamModel = payload.model || CONFIG.defaultDownstreamModel;
  const mapping =
    getModelMapping(CONFIG, downstreamModel) ?? getModelMapping(CONFIG, CONFIG.defaultDownstreamModel);
  if (!mapping) {
    const allowedModels = CONFIG.modelMappings.map(m => m.downstream);
    console.log("[messages] Model not allowed:", downstreamModel);
    return jsonResponse({ error: `Model not allowed: '${downstreamModel}'`, allowedModels }, 400);
  }

  if (hasImageBlocks(payload.messages)) {
    console.log("[messages] Image content not supported by this proxy");
    return jsonResponse({ error: "Image content is not supported by this proxy." }, 400);
  }

  if (CONFIG.provider === "gemini") {
    const geminiPayload = mapClaudeToGemini(payload);

    logVerbose("----------------------");
    logVerbose("[messages] Upstream request parameters:", JSON.stringify(geminiPayload, null, 2));
    logVerbose("[messages] Upstream tools:", JSON.stringify(geminiPayload.tools, null, 2));
    logVerbose("----------------------");

    const endpoint = payload.stream ? "streamGenerateContent" : "generateContent";
    const url = payload.stream
      ? `${upstream.baseUrl}/models/${mapping.upstream}:${endpoint}?alt=sse`
      : `${upstream.baseUrl}/models/${mapping.upstream}:${endpoint}`;
    const headers: Record<string, string> = {
      "content-type": "application/json",
      "x-goog-api-key": upstream.apiKey,
    };

    if (!upstream.apiKey) {
      console.log("[messages] Missing upstream API key");
      return jsonResponse({ error: "Missing upstream API key" }, 401);
    }

    let upstreamResp: Response;
    let lastErrorText = "";
    let retryAfterSeconds: number | null = null;
    const maxRetries = Number(process.env.GEMINI_MAX_RETRIES || "1");
    for (let attempt = 0; attempt <= maxRetries; attempt += 1) {
      try {
        upstreamResp = await fetchWithTimeout(
          url,
          {
            method: "POST",
            headers,
            body: JSON.stringify(geminiPayload),
          },
          payload.stream ? UPSTREAM_STREAM_TIMEOUT_MS : UPSTREAM_TIMEOUT_MS
        );
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        console.log("[messages] Upstream request timed out:", message);
        return jsonResponse({ error: "Upstream request timed out", details: message }, 504);
      }

      logUpstreamResponse({ provider: "gemini", status: upstreamResp.status, stream: Boolean(payload.stream), attempt: attempt + 1 });

      if (upstreamResp.status !== 429) {
        retryAfterSeconds = null;
        lastErrorText = "";
        break;
      }

      lastErrorText = await upstreamResp.text();
      retryAfterSeconds =
        parseGeminiRetryDelaySeconds(lastErrorText) ??
        parseRetryAfterSeconds(upstreamResp.headers.get("retry-after"));

      if (retryAfterSeconds === null || attempt === maxRetries) break;
      console.log(`[messages] Upstream rate limited, retrying in ${retryAfterSeconds}s`);
      await sleep(retryAfterSeconds * 1000);
    }

    if (!upstreamResp.ok || !upstreamResp.body) {
      const text = lastErrorText || (await upstreamResp.text());
      console.log("[messages] Upstream response body:", text);
      const headers = retryAfterSeconds !== null ? { "retry-after": String(retryAfterSeconds) } : undefined;
      return jsonResponse({ error: "Upstream request failed", details: text }, upstreamResp.status, headers);
    }

    if (payload.stream) {
      const stream = await createClaudeStreamFromGemini(upstreamResp.body, mapping.downstream);
      return new Response(stream, {
        status: 200,
        headers: { "content-type": "text/event-stream", "cache-control": "no-cache" },
      });
    }

    const geminiData = (await upstreamResp.json()) as GeminiResponse;
    logVerbose("[messages] Upstream response data:", JSON.stringify(geminiData, null, 2));

    const claudeData = mapGeminiToClaude(geminiData, mapping.downstream);
    logVerbose("[messages] Downstream response data:", JSON.stringify(claudeData, null, 2));

    return jsonResponse(claudeData);
  }

  const openaiPayload = mapClaudeToOpenAI(payload, mapping.upstream);

  logVerbose("----------------------");
  logVerbose("[messages] Upstream request parameters:", JSON.stringify(openaiPayload, null, 2));
  logVerbose("[messages] Upstream tools:", JSON.stringify(openaiPayload.tools, null, 2));
  logVerbose("----------------------");

  const url = `${upstream.baseUrl}/responses`;
  const headers: Record<string, string> = {
    "content-type": "application/json",
    authorization: `Bearer ${upstream.apiKey}`,
  };

  if (!upstream.apiKey) {
    console.log("[messages] Missing upstream API key");
    return jsonResponse({ error: "Missing upstream API key" }, 401);
  }

  let upstreamResp: Response;
  try {
    upstreamResp = await fetchWithTimeout(
      url,
      {
        method: "POST",
        headers,
        body: JSON.stringify(openaiPayload),
      },
      payload.stream ? UPSTREAM_STREAM_TIMEOUT_MS : UPSTREAM_TIMEOUT_MS
    );
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    console.log("[messages] Upstream request timed out:", message);
    return jsonResponse({ error: "Upstream request timed out", details: message }, 504);
  }

  logUpstreamResponse({ provider: CONFIG.provider, status: upstreamResp.status, stream: Boolean(payload.stream) });

  if (!upstreamResp.ok || !upstreamResp.body) {
    const text = await upstreamResp.text();
    console.log("[messages] Upstream response body:", text);
    return jsonResponse({ error: "Upstream request failed", details: text }, upstreamResp.status);
  }

  if (payload.stream) {
    const stream = await createClaudeStream(upstreamResp.body, mapping.downstream);
    return new Response(stream, {
      status: 200,
      headers: { "content-type": "text/event-stream", "cache-control": "no-cache" },
    });
  }

  const openaiData = (await upstreamResp.json()) as OpenAIResponse;
  logVerbose("[messages] Upstream response data:", JSON.stringify(openaiData, null, 2));

  const claudeData = mapOpenAIToClaude(openaiData, mapping.downstream);
  logVerbose("[messages] Downstream response data:", JSON.stringify(claudeData, null, 2));

  return jsonResponse(claudeData);
};

const notFound = (req: Request) => {
  const url = new URL(req.url);
  console.log(`[request] Not found ${req.method} ${url.pathname}`);
  return jsonResponse({ error: "Not found" }, 404);
};

const checkUpstreamModels = async () => {
  if (CONFIG.provider === "gemini") {
    console.log("[startup] Model check skipped: provider=gemini");
    return;
  }
  if (!upstream.apiKey) {
    console.log("[startup] Model check skipped: missing upstream API key");
    return;
  }

  const url = `${upstream.baseUrl}/models`;
  const headers: Record<string, string> = {
    authorization: `Bearer ${upstream.apiKey}`,
  };

  try {
    const resp = await fetchWithTimeout(url, { headers }, UPSTREAM_TIMEOUT_MS);
    if (!resp.ok) {
      const text = await resp.text();
      console.log(`[startup] Model check failed: ${resp.status} ${text}`);
      return;
    }

    const data = (await resp.json()) as { data?: Array<{ id: string }> };
    const available = new Set((data.data || []).map(model => model.id));
    const results = CONFIG.modelMappings.map(mapping => ({
      upstream: mapping.upstream,
      exists: available.has(mapping.upstream),
    }));
    const missing = results.filter(result => !result.exists).map(result => result.upstream);

    const log = missing.length > 0 ? console.error : console.log;
    log(
      "[startup] Upstream model check:",
      JSON.stringify({ provider: CONFIG.provider, checked: results.map(result => result.upstream), missing }, null, 2)
    );
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    console.error("[startup] Model check error:", message);
  }
};

const checkToolChoiceSupport = async () => {
  if (CONFIG.provider === "gemini") {
    console.log("[startup] Tool choice check skipped: provider=gemini");
    return;
  }
  if (!upstream.apiKey) {
    console.log("[startup] Tool choice check skipped: missing upstream API key");
    return;
  }

  const url = `${upstream.baseUrl}/responses`;
  const headers: Record<string, string> = {
    "content-type": "application/json",
    authorization: `Bearer ${upstream.apiKey}`,
  };

  const results: Array<{
    upstream: string;
    ok: boolean;
    status?: number;
    statusText?: string;
    requestId?: string | null;
    error?: string;
    errorCode?: string;
    errorType?: string;
    errorMessage?: string;
  }> = [];

  for (const mapping of CONFIG.modelMappings) {
    const payload = {
      model: mapping.upstream,
      input: [
        {
          role: "user",
          content: "Use web_search to check https://status.openai.com/ availability.",
        },
      ],
      tools: [
        {
          type: "web_search",
        },
      ],
      tool_choice: "required",
      max_output_tokens: 256,
    };

    try {
      const resp = await fetchWithTimeout(
        url,
        {
          method: "POST",
          headers,
          body: JSON.stringify(payload),
        },
        UPSTREAM_TIMEOUT_MS
      );
      if (!resp.ok) {
        const text = await resp.text();
        let errorCode: string | undefined;
        let errorType: string | undefined;
        let errorMessage: string | undefined;
        try {
          const parsed = JSON.parse(text) as { error?: { code?: string; type?: string; message?: string } };
          errorCode = parsed?.error?.code;
          errorType = parsed?.error?.type;
          errorMessage = parsed?.error?.message;
        } catch {
          errorMessage = undefined;
        }
        const requestId =
          resp.headers.get("x-request-id") ||
          resp.headers.get("openai-request-id") ||
          resp.headers.get("request-id");
        console.error(
          "[startup] Tool choice failure detail:",
          JSON.stringify(
            {
              upstream: mapping.upstream,
              status: resp.status,
              statusText: resp.statusText,
              requestId,
              errorCode,
              errorType,
              errorMessage,
              rawError: text,
              payload: {
                model: mapping.upstream,
                tool_choice: payload.tool_choice,
                tools: payload.tools,
                max_output_tokens: payload.max_output_tokens,
              },
            },
            null,
            2
          )
        );
        results.push({
          upstream: mapping.upstream,
          ok: false,
          status: resp.status,
          statusText: resp.statusText,
          requestId,
          error: text,
          errorCode,
          errorType,
          errorMessage,
        });
        continue;
      }
      results.push({ upstream: mapping.upstream, ok: true, status: resp.status });
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      results.push({ upstream: mapping.upstream, ok: false, error: message });
    }
  }

  const failing = results.filter(result => !result.ok).map(result => result.upstream);
  const log = failing.length > 0 ? console.error : console.log;
  log(
    "[startup] Tool choice check:",
    JSON.stringify({ provider: CONFIG.provider, checked: results.map(result => result.upstream), failing }, null, 2)
  );
  if (failing.length > 0) {
    console.error("[startup] Tool choice failures:", JSON.stringify(results.filter(result => !result.ok), null, 2));
  }
};

const createServer = () => Bun.serve({
  hostname: CONFIG.bindAddress,
  port: CONFIG.port,
  fetch: (req) => {
    const url = new URL(req.url);
    if (req.method === "GET" && url.pathname === "/health") {
      return handleHealth();
    }
    if (req.method === "GET" && url.pathname === "/") {
      return handleConfig();
    }
    if (req.method === "POST" && url.pathname === "/v1/messages") {
      return handleMessages(req);
    }
    if (req.method === "POST" && url.pathname === "/v1/messages/count_tokens") {
      return handleCountTokens(req);
    }
    if (req.method === "GET" && url.pathname === "/v1/models") {
      return handleModels();
    }
    return notFound(req);
  },
});

const startServer = async () => {
  await checkUpstreamModels();
  await checkToolChoiceSupport();
  const server = createServer();
  console.log("Proxy config:", JSON.stringify(getPublicConfig(CONFIG, upstream), null, 2));
  console.log(`openai2claude-proxy listening on http://${CONFIG.bindAddress}:${server.port}`);
  return server;
};

if (import.meta.main) {
  await startServer();
}

export { createServer };
