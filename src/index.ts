#!/usr/bin/env bun

import {
  mapOpenAIToClaude,
  mapClaudeToOpenAI,
  createClaudeStream,
  type ClaudeRequest,
  type OpenAIResponse,
} from "./openai_to_claude";
import { buildConfig, buildUpstream, getModelMapping, getPublicConfig } from "./config";
const CONFIG = buildConfig(process.env, process.argv);
const upstream = buildUpstream(CONFIG);

const UPSTREAM_TIMEOUT_MS = Number(process.env.UPSTREAM_TIMEOUT_MS || "30000");

const logVerbose = (...args: unknown[]) => {
  if (CONFIG.verboseLogging) {
    console.log(...args);
  }
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

const jsonResponse = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json" } });

const handleModels = () =>
  jsonResponse({
    object: "list",
    data: CONFIG.modelMappings.map(m => ({ id: m.downstream, object: "model" })),
  });

const handleHealth = () => jsonResponse({ status: "ok" });

const handleConfig = () => jsonResponse(getPublicConfig(CONFIG, upstream));

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
    return jsonResponse({ error: `Model not allowed: '${downstreamModel}'`, allowedModels }, 400);
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
      UPSTREAM_TIMEOUT_MS
    );
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    return jsonResponse({ error: "Upstream request timed out", details: message }, 504);
  }

  const statusLog = upstreamResp.ok && upstreamResp.body ? logVerbose : console.log;
  statusLog("[messages] Upstream response status:", upstreamResp.status);

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

const notFound = () => jsonResponse({ error: "Not found" }, 404);

const checkUpstreamModels = async () => {
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
  if (!upstream.apiKey) {
    console.log("[startup] Tool choice check skipped: missing upstream API key");
    return;
  }

  const url = `${upstream.baseUrl}/responses`;
  const headers: Record<string, string> = {
    "content-type": "application/json",
    authorization: `Bearer ${upstream.apiKey}`,
  };

  const results: Array<{ upstream: string; ok: boolean; status?: number; error?: string }> = [];

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
      max_output_tokens: 16,
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
        results.push({ upstream: mapping.upstream, ok: false, status: resp.status, error: text });
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

await checkUpstreamModels();
await checkToolChoiceSupport();

const server = Bun.serve({
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
    if (req.method === "GET" && url.pathname === "/v1/models") {
      return handleModels();
    }
    return notFound();
  },
});

console.log("Proxy config:", JSON.stringify(getPublicConfig(CONFIG, upstream), null, 2));
console.log(`openai2claude-proxy listening on http://${CONFIG.bindAddress}:${server.port}`);
