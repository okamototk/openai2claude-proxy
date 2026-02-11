type ClaudeMessageContent =
  | { type: "text"; text: string }
  | { type: "tool_use"; id: string; name: string; input: unknown }
  | { type: "tool_result"; tool_use_id: string; content: string }
  | { type: "thinking"; thinking: string; signature?: string }
  | { type: "redacted_thinking"; data: string; signature?: string }
  | { type: "image"; source: { type: string; media_type?: string; data?: string; url?: string } };

type ClaudeMessage = {
  role: "user" | "assistant";
  content: string | ClaudeMessageContent[];
};

type ClaudeRequest = {
  model: string;
  messages: ClaudeMessage[];
  max_tokens?: number;
  temperature?: number;
  top_p?: number;
  stop_sequences?: string[];
  stream?: boolean;
  system?: string | { type: "text"; text: string }[];
  tools?: Array<{
    name: string;
    description?: string;
    input_schema?: unknown;
    type?: "web_search";
    metadata?: unknown;
  }>;
  tool_choice?: unknown;
  thinking?: { type?: "enabled" | "disabled"; budget_tokens?: number } | { type?: "enabled" | "disabled"; budget?: number };
  output_config?: { effort?: string };
};

type OpenAIToolCall = {
  id: string;
  type: "function";
  function: { name: string; arguments: string };
};

type OpenAIInputContentBlock = { type: "input_text"; text: string };
type OpenAIOutputContentBlock =
  | { type: "output_text"; text: string }
  | { type: "refusal"; refusal: string }
  | { type: "function_call"; call_id?: string; name?: string; arguments?: string }
  | { type: "web_search_call"; call_id?: string; id?: string; arguments?: string; query?: string; max_results?: number; search_context_size?: string; user_location?: unknown }
  | { type: "web_search_result"; call_id?: string; id?: string; content?: unknown; results?: unknown; output?: unknown; text?: string }
  | { type: "reasoning"; text?: string; reasoning?: string; summary?: string; signature?: string }
  | { type: "output_reasoning"; text?: string; reasoning?: string; summary?: string; signature?: string }
  | { type: "redacted_reasoning"; data: string; signature?: string };

type OpenAIInputItem =
  | { role: "system"; content: string }
  | { role: "user"; content: string | OpenAIInputContentBlock[] }
  | { role: "assistant"; content: string };

type OpenAIRequest = {
  model: string;
  input: OpenAIInputItem[];
  max_tokens?: number;
  max_output_tokens?: number;
  temperature?: number;
  top_p?: number;
  stream?: boolean;
  tools?: Array<
    | { type: "function"; name: string; description?: string; parameters?: unknown }
    | { type: "web_search"; max_results?: number; search_context_size?: string; user_location?: unknown }
  >;
  tool_choice?: unknown;
  reasoning?: { effort?: string; summary?: string };
};

type ClaudeUsage = {
  cache_creation?: {
    ephemeral_1h_input_tokens?: number;
    ephemeral_5m_input_tokens?: number;
  };
  cache_creation_input_tokens?: number;
  cache_read_input_tokens?: number;
  input_tokens?: number;
  output_tokens?: number;
  reasoning_tokens?: number;
  inference_geo?: string;
  server_tool_use?: { web_search_requests?: number };
  service_tier?: "standard" | "priority" | "batch";
};

type ClaudeResponse = {
  id: string;
  type: "message";
  role: "assistant";
  content: Array<ClaudeMessageContent>;
  model: string;
  stop_reason: string | null;
  stop_sequence?: string | null;
  usage?: ClaudeUsage;
};

type OpenAIResponseItem = {
  type: "message";
  id: string;
  role: "assistant";
  content: Array<{
    type: "text" | "output_text" | "function_call" | "refusal" | "web_search_call" | "web_search_result" | "reasoning" | "output_reasoning" | "redacted_reasoning";
    text?: string;
    reasoning?: string;
    summary?: string;
    signature?: string;
    data?: string;
    call_id?: string;
    name?: string;
    arguments?: string;
    results?: unknown;
    output?: unknown;
    content?: unknown;
  }>;
  stop_reason?: string | null;
} | {
  type: "function_call";
  id: string;
  call_id: string;
  name: string;
  arguments: string;
} | {
  type: "web_search_call";
  id?: string;
  call_id?: string;
  arguments?: string;
  query?: string;
  max_results?: number;
  search_context_size?: string;
  user_location?: unknown;
} | {
  type: "web_search_result";
  id?: string;
  call_id?: string;
  content?: unknown;
  results?: unknown;
  output?: unknown;
  text?: string;
} | {
  type: "reasoning" | "output_reasoning";
  id?: string;
  text?: string;
  reasoning?: string;
  summary?: string;
  signature?: string;
} | {
  type: "redacted_reasoning";
  id?: string;
  data: string;
  signature?: string;
};

type OpenAIUsage = {
  input_tokens?: number;
  output_tokens?: number;
  total_tokens?: number;
  prompt_tokens?: number;
  completion_tokens?: number;
  reasoning_tokens?: number;
};

type OpenAIResponse = {
  id: string;
  object: "response";
  created_at: number;
  model: string;
  output: OpenAIResponseItem[];
  usage?: OpenAIUsage;
};

const mapOpenAIUsageToClaude = (usage?: OpenAIUsage): ClaudeUsage | undefined => {
  if (!usage) return undefined;
  return {
    input_tokens: usage.input_tokens ?? usage.prompt_tokens,
    output_tokens: usage.output_tokens ?? usage.completion_tokens,
    reasoning_tokens: usage.reasoning_tokens,
  };
};

const mapReasoningBlockToClaude = (block: {
  type?: string;
  text?: string;
  reasoning?: string;
  summary?: string;
  signature?: string;
  data?: string;
}): ClaudeMessageContent | null => {
  if (block.type === "reasoning" || block.type === "output_reasoning") {
    const thinking = block.text ?? block.reasoning ?? block.summary ?? "";
    return {
      type: "thinking",
      thinking,
      ...(block.signature ? { signature: block.signature } : {}),
    };
  }
  if (block.type === "redacted_reasoning" && block.data) {
    return {
      type: "redacted_thinking",
      data: block.data,
      ...(block.signature ? { signature: block.signature } : {}),
    };
  }
  return null;
};

const logReasoning = (label: string, payload: unknown) => {
  logVerbose(`[reasoning] ${label}`, payload);
};

const getReasoningText = (block: { text?: string; reasoning?: string; summary?: string }) =>
  block.text ?? block.reasoning ?? block.summary ?? "";

const countWebSearchRequests = (items: OpenAIResponseItem[] | undefined): number => {
  if (!items) return 0;
  const ids = new Set<string>();
  let fallbackCount = 0;
  for (const item of items) {
    if (item.type === "web_search_call" || item.type === "web_search_result") {
      const callId = getWebSearchCallId(item);
      if (callId) ids.add(callId);
      else fallbackCount += 1;
    }
    if (item.type === "message" && item.content) {
      for (const block of item.content) {
        if (block.type === "web_search_call" || block.type === "web_search_result") {
          const callId = getWebSearchCallId(block);
          if (callId) ids.add(callId);
          else fallbackCount += 1;
        }
      }
    }
  }
  return ids.size > 0 ? ids.size : fallbackCount;
};

const attachToolUseUsage = (usage: ClaudeUsage | undefined, webSearchRequests: number): ClaudeUsage | undefined => {
  if (!usage && webSearchRequests === 0) return usage;
  return {
    ...(usage ?? {}),
    ...(webSearchRequests > 0 ? { server_tool_use: { web_search_requests: webSearchRequests } } : {}),
  };
};

export const GPT5_MODEL_CONFIG: Record<string, { contextWindow: number; maxInput: number; maxOutput: number }> = {
  "gpt-5.2": { contextWindow: 400000, maxInput: 272000, maxOutput: 128000 },
  "gpt-5.2-thinking": { contextWindow: 400000, maxInput: 272000, maxOutput: 128000 },
  "gpt-5.3-codex": { contextWindow: 400000, maxInput: 272000, maxOutput: 128000 },
  "gpt-5-mini": { contextWindow: 400000, maxInput: 272000, maxOutput: 128000 },
  "gpt-5-pro": { contextWindow: 1000000, maxInput: 728000, maxOutput: 272000 },
};

const getDownstreamConfig = (model: string) => {
  const modelKey = Object.keys(GPT5_MODEL_CONFIG).find(k => model.includes(k));
  if (modelKey) {
    return GPT5_MODEL_CONFIG[modelKey];
  }
  return null;
};

export const mapFinishReason = (reason: string | null): string | null => {
  if (!reason) return null;
  if (reason === "stop") return "end_turn";
  if (reason === "length") return "max_tokens";
  if (reason === "tool_calls") return "tool_use";
  return reason;
};

const safeJsonParse = (value?: string): unknown | null => {
  if (!value) return null;
  try {
    return JSON.parse(value) as unknown;
  } catch {
    return null;
  }
};

const parseToolArguments = (value?: string): unknown => safeJsonParse(value) ?? {};

const logVerbose = (...args: unknown[]) => {
  if ((process.env.VERBOSE_LOGGING || "").toLowerCase() === "true") {
    console.log(...args);
  }
};

const stringifyToolResult = (value: unknown): string => {
  if (value === undefined || value === null) return "";
  return typeof value === "string" ? value : JSON.stringify(value);
};

const buildWebSearchInput = (block: {
  arguments?: string;
  query?: string;
  max_results?: number;
  search_context_size?: string;
  user_location?: unknown;
}): unknown => {
  const parsed = safeJsonParse(block.arguments);
  if (parsed) return parsed;
  const input: Record<string, unknown> = {};
  if (block.query !== undefined) input.query = block.query;
  if (block.max_results !== undefined) input.max_results = block.max_results;
  if (block.search_context_size !== undefined) input.search_context_size = block.search_context_size;
  if (block.user_location !== undefined) input.user_location = block.user_location;
  return input;
};

const extractWebSearchToolConfig = (tool: {
  input_schema?: unknown;
  metadata?: unknown;
  max_results?: number;
  search_context_size?: string;
  user_location?: unknown;
}): { max_results?: number; search_context_size?: string; user_location?: unknown } => {
  const schema = tool.input_schema as { properties?: Record<string, { default?: unknown; const?: unknown }> } | undefined;
  const props = schema?.properties ?? {};
  const metadata = tool.metadata as { max_results?: number; search_context_size?: string; user_location?: unknown } | undefined;
  return {
    max_results: tool.max_results ?? metadata?.max_results ?? (props.max_results?.default as number | undefined) ?? (props.max_results?.const as number | undefined),
    search_context_size: tool.search_context_size ?? metadata?.search_context_size ?? (props.search_context_size?.default as string | undefined) ?? (props.search_context_size?.const as string | undefined),
    user_location: tool.user_location ?? metadata?.user_location ?? props.user_location?.default ?? props.user_location?.const,
  };
};

const getWebSearchCallId = (block: { call_id?: string; id?: string }): string => {
  return block.call_id || block.id || "web_search";
};

const mapBudgetTokensToEffort = (budgetTokens: number): "low" | "medium" | "high" => {
  if (budgetTokens >= 10000) return "high";
  if (budgetTokens >= 5000) return "medium";
  return "low";
};

export const mapOpenAIToClaude = (openai: OpenAIResponse, model: string): ClaudeResponse => {
  const messageItem = openai.output.find(item => item.type === "message");
  const content: ClaudeMessageContent[] = [];
  const modelConfig = getDownstreamConfig(model);
  const webSearchRequests = countWebSearchRequests(openai.output);

  if (messageItem?.content) {
    for (const block of messageItem.content) {
      if ((block.type === "text" || block.type === "output_text") && block.text) {
        content.push({ type: "text", text: block.text });
      } else if (block.type === "function_call" && block.call_id && block.name) {
        content.push({ type: "tool_use", id: block.call_id, name: block.name, input: parseToolArguments(block.arguments) });
      } else if (block.type === "web_search_call") {
        const callId = getWebSearchCallId(block);
        const toolInput = buildWebSearchInput(block);
        content.push({ type: "tool_use", id: callId, name: "web_search", input: toolInput });
      } else if (block.type === "web_search_result") {
        const callId = getWebSearchCallId(block);
        const resultPayload = block.results ?? block.content ?? block.output ?? block.text;
        content.push({ type: "tool_result", tool_use_id: callId, content: stringifyToolResult(resultPayload) });
      } else if (block.type === "reasoning" || block.type === "output_reasoning" || block.type === "redacted_reasoning") {
        const mapped = mapReasoningBlockToClaude(block);
        if (mapped) {
          logReasoning("mapOpenAIToClaude.message", block);
          content.push(mapped);
        }
      }
    }
  }

  for (const item of openai.output) {
    if (item.type === "function_call") {
      const toolInput = parseToolArguments(item.arguments);
      content.push({ type: "tool_use", id: item.call_id, name: item.name, input: toolInput });
    } else if (item.type === "web_search_call") {
      const callId = getWebSearchCallId(item);
      const toolInput = buildWebSearchInput(item);
      content.push({ type: "tool_use", id: callId, name: "web_search", input: toolInput });
    } else if (item.type === "web_search_result") {
      const callId = getWebSearchCallId(item);
      const resultPayload = item.results ?? item.content ?? item.output ?? item.text;
      content.push({ type: "tool_result", tool_use_id: callId, content: stringifyToolResult(resultPayload) });
    } else if (item.type === "reasoning" || item.type === "output_reasoning" || item.type === "redacted_reasoning") {
      const mapped = mapReasoningBlockToClaude(item);
      if (mapped) {
        logReasoning("mapOpenAIToClaude.item", item);
        content.push(mapped);
      }
    }
  }

  let stopReason = mapFinishReason(messageItem?.stop_reason || null);
  if (modelConfig && openai.usage?.output_tokens && openai.usage.output_tokens >= modelConfig.maxOutput) {
    stopReason = "max_tokens";
  }

  const baseUsage = mapOpenAIUsageToClaude(openai.usage);
  const usage = attachToolUseUsage(baseUsage, webSearchRequests);
  if (!openai.usage) {
    console.log("[messages] Upstream response missing usage; downstream usage may be zero");
  }

  return {
    id: openai.id,
    type: "message",
    role: "assistant",
    content,
    model,
    stop_reason: stopReason,
    stop_sequence: messageItem?.stop_reason ? null : null,
    usage,
  };
};

// Claude to OpenAI conversion helpers
const textFromContent = (content: ClaudeMessage["content"]) => {
  if (typeof content === "string") return content;
  return content
    .filter((block) => block.type === "text")
    .map((block) => (block as { type: "text"; text: string }).text)
    .join("");
};

const extractToolCalls = (content: ClaudeMessage["content"]): OpenAIToolCall[] => {
  if (typeof content === "string") return [];
  const toolBlocks = content.filter((block) => block.type === "tool_use") as Array<{
    type: "tool_use";
    id: string;
    name: string;
    input: unknown;
  }>;
  return toolBlocks.map((block) => ({
    id: block.id,
    type: "function",
    function: { name: block.name, arguments: JSON.stringify(block.input ?? {}) },
  }));
};

const extractToolResults = (content: ClaudeMessage["content"]) => {
  if (typeof content === "string") return [] as Array<{ tool_call_id: string; content: string }>;
  const resultBlocks = content.filter((block) => block.type === "tool_result") as Array<{
    type: "tool_result";
    tool_use_id: string;
    content: string;
  }>;
  return resultBlocks.map((block) => ({ tool_call_id: block.tool_use_id, content: block.content }));
};

const mapThinkingBlockToOpenAI = (_block: ClaudeMessageContent): OpenAIOutputContentBlock | null => {
  return null;
};

const formatToolCalls = (calls: OpenAIToolCall[]) =>
  calls.map((call) => `[tool_call id=${call.id} name=${call.function.name} args=${call.function.arguments}]`).join("\n");

const formatToolCall = (call: OpenAIToolCall) => formatToolCalls([call]);

const formatToolResults = (results: Array<{ tool_call_id: string; content: string }>) =>
  results.map((result) => `[tool_result id=${result.tool_call_id}]\n${result.content}`).join("\n");

export const mapClaudeToOpenAI = (
  req: ClaudeRequest,
  upstreamModel: string,
): OpenAIRequest => {
  const messages: OpenAIInputItem[] = [];

  if (req.system) {
    const systemText = Array.isArray(req.system)
      ? req.system.map((block) => block.text).join("")
      : req.system;
    if (systemText) messages.push({ role: "system", content: systemText });
  }

  for (const msg of req.messages) {
    if (msg.role === "user") {
      const text = textFromContent(msg.content);
      const toolResults = extractToolResults(msg.content);
      const toolText = toolResults.length > 0 ? formatToolResults(toolResults) : "";
      const combined = [text, toolText].filter(Boolean).join("\n");

      if (combined) {
        messages.push({ role: "user", content: combined });
      }
    } else if (msg.role === "assistant") {
      if (typeof msg.content === "string") {
        if (msg.content) {
          messages.push({ role: "assistant", content: msg.content });
        }
        continue;
      }

      const textBlocks: string[] = [];
      for (const block of msg.content) {
        if (block.type === "text") {
          textBlocks.push(block.text);
        } else if (block.type === "tool_use") {
          textBlocks.push(formatToolCall({
            id: block.id,
            type: "function",
            function: { name: block.name, arguments: JSON.stringify(block.input ?? {}) },
          }));
        } else if (block.type === "thinking" || block.type === "redacted_thinking") {
          logReasoning("mapClaudeToOpenAI.message.skipped", block);
        }
      }

      const assistantText = textBlocks.join("\n");
      if (assistantText) {
        messages.push({ role: "assistant", content: assistantText });
      }
    }
  }

  // Responses endpoint expects max_output_tokens (OpenAI/OpenRouter).
  const tokenParam = {
    max_output_tokens: req.max_tokens !== undefined ? Math.max(16, req.max_tokens) : undefined,
  };

  const thinkingBudget = req.thinking && "budget_tokens" in req.thinking
    ? req.thinking.budget_tokens
    : req.thinking && "budget" in req.thinking
      ? req.thinking.budget
      : undefined;
  const inferredEffort = thinkingBudget !== undefined ? mapBudgetTokensToEffort(thinkingBudget) : undefined;

  const reasoning: OpenAIRequest["reasoning"] = {
    ...(req.output_config?.effort ? { effort: req.output_config.effort } : {}),
    ...(!req.output_config?.effort && inferredEffort ? { effort: inferredEffort } : {}),
  };

  if (Object.keys(reasoning).length > 0) {
    logReasoning("mapClaudeToOpenAI.request", reasoning);
  }

  return {
    model: upstreamModel,
    input: messages,
    ...tokenParam,
    temperature: req.temperature,
    top_p: req.top_p,
    stream: req.stream,
    tools: req.tools?.filter((tool) => {
      if (!tool.name) {
        console.log("[mapClaudeToOpenAI] Filtering out tool without name:", JSON.stringify(tool, null, 2));
        return false;
      }
      return true;
    }).map((tool) => {
      if (tool.name === "web_search" || tool.type === "web_search") {
        const { max_results, search_context_size, user_location } = extractWebSearchToolConfig(tool);
        logVerbose("[mapClaudeToOpenAI] Mapping web_search tool:", {
          max_results,
          search_context_size,
          user_location,
        });
        return {
          type: "web_search",
          ...(max_results !== undefined ? { max_results } : {}),
          ...(search_context_size !== undefined ? { search_context_size } : {}),
          ...(user_location !== undefined ? { user_location } : {}),
        };
      }
      return {
        type: "function",
        name: tool.name,
        description: tool.description,
        parameters: tool.input_schema,
      };
    }),
    tool_choice: req.tool_choice,
    ...(Object.keys(reasoning).length > 0 ? { reasoning } : {}),
  };
};

export const createClaudeStream = async (openaiStream: ReadableStream<Uint8Array>, model: string) => {
  const encoder = new TextEncoder();
  const decoder = new TextDecoder();
  const reader = openaiStream.getReader();

  const stream = new ReadableStream<Uint8Array>({
    async start(controller) {
      let sentMessageStart = false;
      let messageId = "";
      let contentBlockIndex = 0;
      let usage: OpenAIUsage | null = null;
      let stopReason: string | null = null;
      let pendingContentBlockStop: number | null = null;
      const webSearchCallIds = new Set<string>();
      let webSearchFallbackCount = 0;

      const recordWebSearch = (block: { call_id?: string; id?: string }) => {
        const callId = getWebSearchCallId(block);
        if (callId) {
          webSearchCallIds.add(callId);
        } else {
          webSearchFallbackCount += 1;
        }
      };

      const getWebSearchRequestCount = () => (webSearchCallIds.size > 0 ? webSearchCallIds.size : webSearchFallbackCount);

      const send = (event: string, data: unknown) => {
        controller.enqueue(encoder.encode(`event: ${event}\n`));
        controller.enqueue(encoder.encode(`data: ${JSON.stringify(data)}\n\n`));
      };

      const sendThinkingBlock = (thinkingText: string, signature?: string) => {
        if (pendingContentBlockStop !== null) {
          send("content_block_stop", { type: "content_block_stop", index: pendingContentBlockStop });
        }
        send("content_block_start", {
          type: "content_block_start",
          index: contentBlockIndex,
          content_block: { type: "thinking", thinking: "" },
        });
        if (thinkingText) {
          send("content_block_delta", {
            type: "content_block_delta",
            index: contentBlockIndex,
            delta: { type: "thinking_delta", thinking: thinkingText },
          });
        }
        if (signature) {
          send("content_block_delta", {
            type: "content_block_delta",
            index: contentBlockIndex,
            delta: { type: "signature_delta", signature },
          });
        }
        pendingContentBlockStop = contentBlockIndex;
        contentBlockIndex++;
      };

      const sendRedactedThinkingBlock = (data: string, signature?: string) => {
        if (pendingContentBlockStop !== null) {
          send("content_block_stop", { type: "content_block_stop", index: pendingContentBlockStop });
        }
        send("content_block_start", {
          type: "content_block_start",
          index: contentBlockIndex,
          content_block: { type: "redacted_thinking", data: "" },
        });
        if (data) {
          send("content_block_delta", {
            type: "content_block_delta",
            index: contentBlockIndex,
            delta: { type: "redacted_thinking_delta", data },
          });
        }
        if (signature) {
          send("content_block_delta", {
            type: "content_block_delta",
            index: contentBlockIndex,
            delta: { type: "signature_delta", signature },
          });
        }
        pendingContentBlockStop = contentBlockIndex;
        contentBlockIndex++;
      };

      const handleReasoningBlock = (block: { type?: string; text?: string; reasoning?: string; summary?: string; signature?: string; data?: string }) => {
        if (block.type === "redacted_reasoning" && block.data) {
          logReasoning("createClaudeStream.redacted", block);
          sendRedactedThinkingBlock(block.data, block.signature);
          return true;
        }
        if (block.type === "reasoning" || block.type === "output_reasoning") {
          const thinkingText = getReasoningText(block);
          logReasoning("createClaudeStream.thinking", block);
          sendThinkingBlock(thinkingText, block.signature);
          return true;
        }
        return false;
      };

      const handleThinkingDelta = (delta: { thinking?: string; signature?: string; data?: string }) => {
        if (pendingContentBlockStop === null) return false;
        if (delta.thinking) {
          send("content_block_delta", {
            type: "content_block_delta",
            index: pendingContentBlockStop,
            delta: { type: "thinking_delta", thinking: delta.thinking },
          });
        }
        if (delta.data) {
          send("content_block_delta", {
            type: "content_block_delta",
            index: pendingContentBlockStop,
            delta: { type: "redacted_thinking_delta", data: delta.data },
          });
        }
        if (delta.signature) {
          send("content_block_delta", {
            type: "content_block_delta",
            index: pendingContentBlockStop,
            delta: { type: "signature_delta", signature: delta.signature },
          });
        }
        return true;
      };

      const maybeHandleReasoningEvent = (eventType?: string, payload?: { delta?: { type?: string; thinking?: string; signature?: string; data?: string } }) => {
        if (!eventType) return false;
        if ((eventType.includes("reasoning") || eventType.includes("redacted")) && payload?.delta) {
          return handleThinkingDelta(payload.delta);
        }
        return false;
      };

      let buffer = "";
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() ?? "";
        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed.startsWith("data:")) continue;
          const payload = trimmed.replace(/^data:\s*/, "");
          if (payload === "[DONE]") {
            // Send pending content_block_stop if any
            if (pendingContentBlockStop !== null) {
              send("content_block_stop", { type: "content_block_stop", index: pendingContentBlockStop });
              pendingContentBlockStop = null;
            }
            // Send message_delta with usage and stop_reason before message_stop
            const messageDelta: { stop_reason: string | null; usage?: ClaudeUsage } = {
              stop_reason: stopReason,
            };
            const baseDeltaUsage = mapOpenAIUsageToClaude(usage ?? undefined);
            const deltaUsage = attachToolUseUsage(baseDeltaUsage, getWebSearchRequestCount());
            if (deltaUsage) {
              messageDelta.usage = deltaUsage;
            }
            if (!usage) {
              console.log("[messages] Upstream response missing usage; downstream usage may be zero");
            }
            send("message_delta", { type: "message_delta", delta: messageDelta });
            send("message_stop", { type: "message_stop" });
            controller.close();
            return;
          }
          let json: {
            id?: string;
            type?: string;
            stop_reason?: string | null;
            usage?: OpenAIUsage;
            output?: Array<{
              type?: string;
              id?: string;
              call_id?: string;
              name?: string;
              arguments?: string;
              stop_reason?: string | null;
              query?: string;
              max_results?: number;
              search_context_size?: string;
              user_location?: unknown;
              results?: unknown;
              output?: unknown;
              text?: string;
              reasoning?: string;
              summary?: string;
              signature?: string;
              data?: string;
              content?: Array<{
                type?: string;
                text?: string;
                reasoning?: string;
                summary?: string;
                signature?: string;
                data?: string;
                call_id?: string;
                name?: string;
                arguments?: string;
                query?: string;
                max_results?: number;
                search_context_size?: string;
                user_location?: unknown;
                results?: unknown;
                output?: unknown;
              }>;
            }>;
          };
          try {
            json = JSON.parse(payload) as typeof json;
          } catch (error) {
            console.log("[messages] Failed to parse upstream payload:", payload, error);
            continue;
          }

          if (maybeHandleReasoningEvent(json.type, json as { delta?: { type?: string; thinking?: string; signature?: string; data?: string } })) {
            continue;
          }

          // Extract stop_reason and usage from upstream events
          if (json.stop_reason !== undefined) {
            stopReason = mapFinishReason(json.stop_reason);
          }
          if (json.usage) {
            usage = json.usage;
          }
          // Extract from output items as well
          if (json.output) {
            for (const item of json.output) {
              if (item.stop_reason !== undefined) {
                stopReason = mapFinishReason(item.stop_reason);
              }
              if (item.type === "web_search_call" || item.type === "web_search_result") {
                recordWebSearch(item);
              }
              if (item.type === "message" && item.content) {
                for (const block of item.content) {
                  if (block.type === "web_search_call" || block.type === "web_search_result") {
                    recordWebSearch(block);
                  }
                }
              }
            }
          }

          if (json.type === "response.output_text.done") {
            // Send pending content_block_stop before moving to next block
            if (pendingContentBlockStop !== null) {
              send("content_block_stop", { type: "content_block_stop", index: pendingContentBlockStop });
              pendingContentBlockStop = null;
            }
            continue;
          }

          if (!sentMessageStart && json.id) {
            messageId = json.id;
            const baseMessageUsage = mapOpenAIUsageToClaude(usage ?? undefined);
            const messageUsage = attachToolUseUsage(baseMessageUsage, getWebSearchRequestCount());
            send("message_start", {
              type: "message_start",
              message: {
                id: messageId,
                type: "message",
                role: "assistant",
                model,
                content: [],
                stop_reason: null,
                stop_sequence: null,
                ...(messageUsage ? { usage: messageUsage } : {}),
              },
            });
            sentMessageStart = true;
          }

  if (json.output) {
    for (const item of json.output) {
      if (item.type === "message" && item.content) {
        for (const contentBlock of item.content) {
          if ((contentBlock.type === "text" || contentBlock.type === "output_text") && contentBlock.text) {
            // Send pending content_block_stop from previous block
            if (pendingContentBlockStop !== null) {
              send("content_block_stop", { type: "content_block_stop", index: pendingContentBlockStop });
            }
            send("content_block_start", {
              type: "content_block_start",
              index: contentBlockIndex,
              content_block: { type: "text", text: "" },
            });
            send("content_block_delta", {
              type: "content_block_delta",
              index: contentBlockIndex,
              delta: { type: "text_delta", text: contentBlock.text },
            });
            pendingContentBlockStop = contentBlockIndex;
            contentBlockIndex++;
          } else if (handleReasoningBlock(contentBlock)) {
            continue;
          } else if (contentBlock.type === "web_search_call") {
            recordWebSearch(contentBlock);
            if (pendingContentBlockStop !== null) {
              send("content_block_stop", { type: "content_block_stop", index: pendingContentBlockStop });
            }
            const callId = getWebSearchCallId(contentBlock);
            const toolInput = buildWebSearchInput(contentBlock);
            send("content_block_start", {
              type: "content_block_start",
              index: contentBlockIndex,
              content_block: { type: "tool_use", id: callId, name: "web_search", input: {} },
            });
            const partialJson = contentBlock.arguments || (toolInput && Object.keys(toolInput as Record<string, unknown>).length > 0 ? JSON.stringify(toolInput) : "");
            if (partialJson) {
              send("content_block_delta", {
                type: "content_block_delta",
                index: contentBlockIndex,
                delta: { type: "input_json_delta", partial_json: partialJson },
              });
            }
            pendingContentBlockStop = contentBlockIndex;
            contentBlockIndex++;
          } else if (contentBlock.type === "web_search_result") {
            recordWebSearch(contentBlock);
            if (pendingContentBlockStop !== null) {
              send("content_block_stop", { type: "content_block_stop", index: pendingContentBlockStop });
            }
            const callId = getWebSearchCallId(contentBlock);
            const resultPayload = contentBlock.results ?? contentBlock.content ?? contentBlock.output ?? contentBlock.text;
            const resultText = stringifyToolResult(resultPayload);
            send("content_block_start", {
              type: "content_block_start",
              index: contentBlockIndex,
              content_block: { type: "tool_result", tool_use_id: callId, content: "" },
            });
            if (resultText) {
              send("content_block_delta", {
                type: "content_block_delta",
                index: contentBlockIndex,
                delta: { type: "text_delta", text: resultText },
              });
            }
            pendingContentBlockStop = contentBlockIndex;
            contentBlockIndex++;
          }
        }
      } else if (item.type === "function_call" && item.call_id && item.name) {
        // Handle function calls in streaming
        if (pendingContentBlockStop !== null) {
          send("content_block_stop", { type: "content_block_stop", index: pendingContentBlockStop });
        }
        send("content_block_start", {
          type: "content_block_start",
          index: contentBlockIndex,
          content_block: { type: "tool_use", id: item.call_id, name: item.name, input: {} },
        });
        if (item.arguments) {
          send("content_block_delta", {
            type: "content_block_delta",
            index: contentBlockIndex,
            delta: { type: "input_json_delta", partial_json: item.arguments },
          });
        }
        pendingContentBlockStop = contentBlockIndex;
        contentBlockIndex++;
      } else if (handleReasoningBlock(item)) {
        continue;
      } else if (item.type === "web_search_call") {
        recordWebSearch(item);
        if (pendingContentBlockStop !== null) {
          send("content_block_stop", { type: "content_block_stop", index: pendingContentBlockStop });
        }
        const callId = getWebSearchCallId(item);
        const toolInput = buildWebSearchInput(item);
        send("content_block_start", {
          type: "content_block_start",
          index: contentBlockIndex,
          content_block: { type: "tool_use", id: callId, name: "web_search", input: {} },
        });
        const partialJson = item.arguments || (toolInput && Object.keys(toolInput as Record<string, unknown>).length > 0 ? JSON.stringify(toolInput) : "");
        if (partialJson) {
          send("content_block_delta", {
            type: "content_block_delta",
            index: contentBlockIndex,
            delta: { type: "input_json_delta", partial_json: partialJson },
          });
        }
        pendingContentBlockStop = contentBlockIndex;
        contentBlockIndex++;
      } else if (item.type === "web_search_result") {
        if (pendingContentBlockStop !== null) {
          send("content_block_stop", { type: "content_block_stop", index: pendingContentBlockStop });
        }
        const callId = getWebSearchCallId(item);
        const resultPayload = item.results ?? item.content ?? item.output ?? item.text;
        const resultText = stringifyToolResult(resultPayload);
        send("content_block_start", {
          type: "content_block_start",
          index: contentBlockIndex,
          content_block: { type: "tool_result", tool_use_id: callId, content: "" },
        });
        if (resultText) {
          send("content_block_delta", {
            type: "content_block_delta",
            index: contentBlockIndex,
            delta: { type: "text_delta", text: resultText },
          });
        }
        pendingContentBlockStop = contentBlockIndex;
        contentBlockIndex++;
      }
    }
  }
        }
      }

      const finalLine = buffer.trim();
      if (finalLine.startsWith("data:")) {
        const payload = finalLine.replace(/^data:\s*/, "");
        if (payload === "[DONE]") {
          if (pendingContentBlockStop !== null) {
            send("content_block_stop", { type: "content_block_stop", index: pendingContentBlockStop });
          }
          const messageDelta: { stop_reason: string | null; usage?: ClaudeUsage } = {
            stop_reason: stopReason,
          };
          const baseDeltaUsage = mapOpenAIUsageToClaude(usage ?? undefined);
          const deltaUsage = attachToolUseUsage(baseDeltaUsage, getWebSearchRequestCount());
          if (deltaUsage) {
            messageDelta.usage = deltaUsage;
          }
          if (!usage) {
            console.log("[messages] Upstream response missing usage; downstream usage may be zero");
          }
          send("message_delta", { type: "message_delta", delta: messageDelta });
          send("message_stop", { type: "message_stop" });
          controller.close();
          return;
        }
      }

      // Send pending content_block_stop if stream ends without [DONE]
      if (pendingContentBlockStop !== null) {
        send("content_block_stop", { type: "content_block_stop", index: pendingContentBlockStop });
      }
      // Send message_delta with usage and stop_reason before closing
      const messageDelta: { stop_reason: string | null; usage?: ClaudeUsage } = {
        stop_reason: stopReason,
      };
      const deltaUsage = mapOpenAIUsageToClaude(usage ?? undefined);
      if (deltaUsage) {
        messageDelta.usage = deltaUsage;
      }
      send("message_delta", { type: "message_delta", delta: messageDelta });
      send("message_stop", { type: "message_stop" });
      controller.close();
    },
  });

  return stream;
};

export type { ClaudeMessageContent, ClaudeResponse, OpenAIResponse, OpenAIResponseItem, ClaudeMessage, ClaudeRequest, OpenAIInputItem, OpenAIToolCall, OpenAIRequest };
