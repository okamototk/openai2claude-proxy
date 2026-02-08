#!/usr/bin/env bun

import { 
  mapOpenAIToAnthropic, 
  mapAnthropicToOpenAI,
  createAnthropicStream,
  GPT5_MODEL_CONFIG, 
  type AnthropicMessage, 
  type AnthropicRequest, 
  type OpenAIToolCall, 
  type OpenAIRequest,
  type AnthropicMessageContent, 
  type AnthropicResponse, 
  type OpenAIResponse, 
  type OpenAIResponseItem 
} from "./openai_to_anhtorpic";

const DEFAULT_MODEL_NAME = "gpt-5.2-codex";

type ModelMapping = {
  upstream: string;
  downstream: string;
};

const parseModelArg = (modelArg: string): ModelMapping => {
  let upstream = "";
  let downstream: string | null = null;
  let escaped = false;

  for (let i = 0; i < modelArg.length; i += 1) {
    const char = modelArg[i]!;
    if (escaped) {
      if (downstream === null) {
        upstream += char;
      } else {
        downstream += char;
      }
      escaped = false;
      continue;
    }

    if (char === "\\") {
      escaped = true;
      continue;
    }

    if (char === ":" && downstream === null) {
      downstream = "";
      continue;
    }

    if (downstream === null) {
      upstream += char;
    } else {
      downstream += char;
    }
  }

  if (escaped) {
    if (downstream === null) {
      upstream += "\\";
    } else {
      downstream += "\\";
    }
  }

  if (downstream === null) {
    return { upstream, downstream: upstream };
  }

  return { upstream, downstream };
};

const resolveModelNames = (): ModelMapping[] => {
  const args = process.argv.slice(2);
  const modelArgs: string[] = [];

  for (let i = 0; i < args.length; i += 1) {
    const arg = args[i];
    if (arg === "--model" || arg === "-m") {
      const value = args[i + 1];
      if (value) {
        modelArgs.push(value);
        i += 1;
      }
    } else if (arg.startsWith("--model=")) {
      modelArgs.push(arg.split("=", 2)[1]!);
    } else if (arg.startsWith("-m=")) {
      modelArgs.push(arg.split("=", 2)[1]!);
    }
  }

  if (modelArgs.length === 0) {
    return [{ upstream: DEFAULT_MODEL_NAME, downstream: DEFAULT_MODEL_NAME }];
  }

  return modelArgs.map(parseModelArg);
};

const modelMappings = resolveModelNames();
const defaultModelMapping = modelMappings[0]!;

const CONFIG = {
  provider: (process.env.PROVIDER || "openai").toLowerCase(),
  openaiKey: process.env.OPENAI_API_KEY || "",
  openrouterKey: process.env.OPENROUTER_API_KEY || "",
  openaiBaseUrl: process.env.OPENAI_BASE_URL || "https://api.openai.com/v1",
  openrouterBaseUrl: process.env.OPENROUTER_BASE_URL || "https://openrouter.ai/api/v1",
  modelMappings,
  defaultUpstreamModel: defaultModelMapping.upstream,
  defaultDownstreamModel: defaultModelMapping.downstream,
  port: Number(process.env.PORT || "3000"),
  bindAddress: process.env.BIND_ADDRESS || "127.0.0.1",
  verboseLogging: (process.env.VERBOSE_LOGGING || "").toLowerCase() === "true",
};

const logVerbose = (...args: unknown[]) => {
  if (CONFIG.verboseLogging) {
    console.log(...args);
  }
};

const getModelMapping = (downstreamModel: string): ModelMapping | undefined => {
  return CONFIG.modelMappings.find(m => m.downstream === downstreamModel);
};

const upstream = (() => {
  if (CONFIG.provider === "openrouter") {
    return {
      baseUrl: CONFIG.openrouterBaseUrl,
      apiKey: CONFIG.openrouterKey,
    };
  }
  return {
    baseUrl: CONFIG.openaiBaseUrl,
    apiKey: CONFIG.openaiKey,
  };
})();

const jsonResponse = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json" } });

const handleModels = () =>
  jsonResponse({
    object: "list",
    data: CONFIG.modelMappings.map(m => ({ id: m.downstream, object: "model" })),
  });

const handleHealth = () => jsonResponse({ status: "ok" });

const getPublicConfig = () => ({
  provider: CONFIG.provider,
  modelMappings: CONFIG.modelMappings,
  defaultUpstreamModel: CONFIG.defaultUpstreamModel,
  defaultDownstreamModel: CONFIG.defaultDownstreamModel,
  openaiBaseUrl: CONFIG.openaiBaseUrl,
  openrouterBaseUrl: CONFIG.openrouterBaseUrl,
  port: CONFIG.port,
  bindAddress: CONFIG.bindAddress,
  hasUpstreamApiKey: Boolean(upstream.apiKey),
  verboseLogging: CONFIG.verboseLogging,
});

const handleConfig = () => jsonResponse(getPublicConfig());

const handleMessages = async (req: Request) => {
  const payload = (await req.json()) as AnthropicRequest;

  logVerbose("----------------------");
  logVerbose("[messages] Downstream request parameters:", JSON.stringify(payload, null, 2));
  logVerbose("[messages] Raw tools:", JSON.stringify(payload.tools, null, 2));
  logVerbose("----------------------");

  const downstreamModel = payload.model || CONFIG.defaultDownstreamModel;
  const mapping = getModelMapping(downstreamModel);
  if (!mapping) {
    const allowedModels = CONFIG.modelMappings.map(m => m.downstream);
    return jsonResponse({ error: `Model not allowed: '${downstreamModel}'`, allowedModels }, 400);
  }

  const openaiPayload = mapAnthropicToOpenAI(payload, mapping.upstream);

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

  const upstreamResp = await fetch(url, {
    method: "POST",
    headers,
    body: JSON.stringify(openaiPayload),
  });

  console.log("[messages] Upstream response status:", upstreamResp.status);

  if (!upstreamResp.ok || !upstreamResp.body) {
    const text = await upstreamResp.text();
    console.log("[messages] Upstream response body:", text);
    return jsonResponse({ error: "Upstream request failed", details: text }, upstreamResp.status);
  }

  if (payload.stream) {
    const stream = await createAnthropicStream(upstreamResp.body, mapping.downstream);
    return new Response(stream, {
      status: 200,
      headers: { "content-type": "text/event-stream", "cache-control": "no-cache" },
    });
  }

  const openaiData = (await upstreamResp.json()) as OpenAIResponse;
  console.log("[messages] Upstream response data:", JSON.stringify(openaiData, null, 2));

  const anthropicData = mapOpenAIToAnthropic(openaiData, mapping.downstream);
  console.log("[messages] Downstream response data:", JSON.stringify(anthropicData, null, 2));

  return jsonResponse(anthropicData);
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
    const resp = await fetch(url, { headers });
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
      JSON.stringify({ provider: CONFIG.provider, checked: results.length, missing }, null, 2)
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
      tool_choice: "auto",
      max_output_tokens: 16,
    };

    try {
      const resp = await fetch(url, {
        method: "POST",
        headers,
        body: JSON.stringify(payload),
      });
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
    JSON.stringify({ provider: CONFIG.provider, checked: results.length, failing }, null, 2)
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
    if (req.method === "GET" && url.pathname === "/health") {
      return handleHealth();
    }
    if (req.method === "GET" && url.pathname === "/") {
      return handleConfig();
    }
    return notFound();
  },
});

console.log("Proxy config:", JSON.stringify(getPublicConfig(), null, 2));
console.log(`openai2claude-proxy listening on http://${CONFIG.bindAddress}:${server.port}`);
