import { patchConsoleWithTimestamps } from "./logger";
import { dumpStreamMessage, logVerbose, safeJsonParse, stringifyToolResult, textFromClaudeContent } from "./common";

patchConsoleWithTimestamps();

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
  | { type: "reasoning"; text?: string; reasoning?: string; summary?: unknown; signature?: string }
  | { type: "output_reasoning"; text?: string; reasoning?: string; summary?: unknown; signature?: string }
  | { type: "redacted_reasoning"; data: string; signature?: string };

type OpenAIInputItem =
  | { role: "system"; content: string }
  | { role: "user"; content: string | OpenAIInputContentBlock[] }
  | { role: "assistant"; content: string }
  | { type: "function_call"; call_id: string; name: string; arguments: string }
  | { type: "function_call_output"; call_id: string; output: string };

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

const normalizeReasoningText = (value: unknown): string => {
  if (typeof value === "string") return value;
  if (Array.isArray(value)) {
    return value
      .map(normalizeReasoningText)
      .filter(Boolean)
      .join("\n");
  }
  if (value && typeof value === "object") {
    const record = value as Record<string, unknown>;
    return normalizeReasoningText(record.text ?? record.summary ?? record.reasoning ?? "");
  }
  return "";
};

const mapReasoningBlockToClaude = (block: {
  type?: string;
  text?: string;
  reasoning?: string;
  summary?: unknown;
  signature?: string;
  data?: string;
}): ClaudeMessageContent | null => {
  if (block.type === "reasoning" || block.type === "output_reasoning") {
    const thinking = normalizeReasoningText(block.text ?? block.reasoning ?? block.summary ?? "");
    if (!thinking && !block.signature) return null;
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

const getReasoningText = (block: { text?: string; reasoning?: string; summary?: unknown }) =>
  normalizeReasoningText(block.text ?? block.reasoning ?? block.summary ?? "");

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

const DEFAULT_MAX_TOKENS = 128000;

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

const finalizeClaudeStopReason = (reason: string | null): string => reason ?? "end_turn";

const sanitizeToolInput = (input: unknown): unknown => {
  if (!input || typeof input !== "object" || Array.isArray(input)) return input;
  const record = input as Record<string, unknown>;
  const sanitized: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(record)) {
    if (key === "pages" && value === "") continue;
    if (value && typeof value === "object" && !Array.isArray(value)) {
      sanitized[key] = sanitizeToolInput(value);
      continue;
    }
    sanitized[key] = value;
  }
  return sanitized;
};

const parseToolArguments = (value?: string): unknown => sanitizeToolInput(safeJsonParse(value) ?? {});

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
    } else if (item.type === "reasoning" || item.type === "output_reasoning" || item.type === "redacted_reasoning") {
      const mapped = mapReasoningBlockToClaude(item);
      if (mapped) {
        logReasoning("mapOpenAIToClaude.item", item);
        content.push(mapped);
      }
    }
  }

  let stopReason = mapFinishReason(messageItem?.stop_reason || null);
  if (!stopReason && content.some((block) => block.type === "tool_use")) {
    stopReason = "tool_use";
  }
  if (modelConfig && openai.usage?.output_tokens && openai.usage.output_tokens >= modelConfig.maxOutput) {
    stopReason = "max_tokens";
  }
  const finalStopReason = finalizeClaudeStopReason(stopReason);

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
    stop_reason: finalStopReason,
    stop_sequence: messageItem?.stop_reason ? null : null,
    usage,
  };
};

// Claude to OpenAI conversion helpers
const textFromContent = (content: ClaudeMessage["content"]) => textFromClaudeContent(content);

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

const mapToolChoiceToOpenAI = (
  toolChoice: ClaudeRequest["tool_choice"],
  tools?: ClaudeRequest["tools"],
): OpenAIRequest["tool_choice"] => {
  if (!toolChoice) return undefined;

  if (toolChoice === "auto" || toolChoice === "none" || toolChoice === "required") {
    return toolChoice;
  }

  if (typeof toolChoice !== "object") return toolChoice;

  const choice = toolChoice as { type?: string; name?: string };

  if (choice.type === "auto" || choice.type === "none") return choice.type;
  if (choice.type === "any" || choice.type === "required") return "required";

  const explicitName = choice.name;
  if (!explicitName) return undefined;

  const matchesWebSearch = explicitName === "web_search"
    || tools?.some((tool) => tool.name === explicitName && tool.type === "web_search");

  if (choice.type === "tool" && matchesWebSearch) {
    return { type: "web_search" };
  }

  if (choice.type === "tool" || choice.type === "function" || !choice.type) {
    return { type: "function", name: explicitName };
  }

  return undefined;
};

export const mapClaudeToOpenAI = (
  req: ClaudeRequest,
  upstreamModel: string,
): OpenAIRequest => {
  const messages: OpenAIInputItem[] = [];
  const knownToolCallIds = new Set<string>();

  if (req.system) {
    const systemText = Array.isArray(req.system)
      ? req.system.map((block) => block.text).join("")
      : req.system;
    if (systemText) messages.push({ role: "system", content: systemText });
  }

  for (const msg of req.messages) {
    if (msg.role === "user") {
      if (typeof msg.content === "string") {
        if (msg.content) messages.push({ role: "user", content: msg.content });
        continue;
      }

      let textBuffer: string[] = [];
      const flushUserText = () => {
        const text = textBuffer.join("\n");
        if (text) messages.push({ role: "user", content: text });
        textBuffer = [];
      };

      for (const block of msg.content) {
        if (block.type === "text") {
          if (block.text) textBuffer.push(block.text);
        } else if (block.type === "tool_result") {
          flushUserText();
          if (!knownToolCallIds.has(block.tool_use_id)) {
            logVerbose("[mapClaudeToOpenAI] Dropping tool_result with unknown tool_use_id", {
              tool_use_id: block.tool_use_id,
            });
            continue;
          }
          messages.push({
            type: "function_call_output",
            call_id: block.tool_use_id,
            output: block.content ?? "",
          });
        }
      }
      flushUserText();
    } else if (msg.role === "assistant") {
      if (typeof msg.content === "string") {
        if (msg.content) {
          messages.push({ role: "assistant", content: msg.content });
        }
        continue;
      }

      let textBlocks: string[] = [];
      const flushAssistantText = () => {
        const assistantText = textBlocks.join("\n");
        if (assistantText) {
          messages.push({ role: "assistant", content: assistantText });
        }
        textBlocks = [];
      };

      for (const block of msg.content) {
        if (block.type === "text") {
          if (block.text) textBlocks.push(block.text);
        } else if (block.type === "tool_use") {
          flushAssistantText();
          knownToolCallIds.add(block.id);
          messages.push({
            type: "function_call",
            call_id: block.id,
            name: block.name,
            arguments: JSON.stringify(block.input ?? {}),
          });
        } else if (block.type === "thinking" || block.type === "redacted_thinking") {
          logReasoning("mapClaudeToOpenAI.message.skipped", block);
        }
      }
      flushAssistantText();
    }
  }

  // Responses endpoint expects max_output_tokens (OpenAI/OpenRouter).
  const modelConfig = getDownstreamConfig(upstreamModel);
  const requestedMaxTokens = req.max_tokens ?? DEFAULT_MAX_TOKENS;
  const boundedMaxTokens = modelConfig
    ? Math.min(requestedMaxTokens, modelConfig.maxOutput)
    : requestedMaxTokens;
  const tokenParam = {
    max_output_tokens: Math.max(16, boundedMaxTokens),
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
    tool_choice: mapToolChoiceToOpenAI(req.tool_choice, req.tools),
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
      let activeTextBlockKey: string | null = null;
      let sawOutputTextDelta = false;
      let sawIncrementalOutput = false;
      const streamedFunctionCallItemIds = new Set<string>();
      const streamedFunctionCallCallIds = new Set<string>();
      const functionCallBlockIndexByItemId = new Map<string, number>();
      const functionCallHasArgumentDeltaByItemId = new Map<string, boolean>();
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
        dumpStreamMessage(`downstream ${event}`, data);
        controller.enqueue(encoder.encode(`event: ${event}\n`));
        controller.enqueue(encoder.encode(`data: ${JSON.stringify(data)}\n\n`));
      };

      const ensureMessageStart = (candidateId?: string) => {
        if (sentMessageStart) return;
        if (candidateId) {
          messageId = candidateId;
        }
        if (!messageId) {
          messageId = `msg_${Date.now()}`;
        }
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
      };

      const ensureTextContentBlock = (blockKey: string) => {
        ensureMessageStart();
        if (activeTextBlockKey === blockKey && pendingContentBlockStop !== null) {
          return pendingContentBlockStop;
        }
        if (pendingContentBlockStop !== null) {
          send("content_block_stop", { type: "content_block_stop", index: pendingContentBlockStop });
          pendingContentBlockStop = null;
        }
        send("content_block_start", {
          type: "content_block_start",
          index: contentBlockIndex,
          content_block: { type: "text", text: "" },
        });
        pendingContentBlockStop = contentBlockIndex;
        activeTextBlockKey = blockKey;
        contentBlockIndex++;
        return pendingContentBlockStop;
      };

      const startFunctionCallContentBlock = (item: { id?: string; call_id?: string; name?: string }) => {
        const callId = item.call_id;
        const name = item.name;
        if (!callId || !name) {
          return null;
        }
        ensureMessageStart();
        if (pendingContentBlockStop !== null) {
          send("content_block_stop", { type: "content_block_stop", index: pendingContentBlockStop });
          pendingContentBlockStop = null;
          activeTextBlockKey = null;
        }
        const blockIndex = contentBlockIndex;
        send("content_block_start", {
          type: "content_block_start",
          index: blockIndex,
          content_block: { type: "tool_use", id: callId, name, input: {} },
        });
        stopReason = "tool_use";
        pendingContentBlockStop = blockIndex;
        contentBlockIndex++;
        if (item.id) {
          functionCallBlockIndexByItemId.set(item.id, blockIndex);
          functionCallHasArgumentDeltaByItemId.set(item.id, false);
          streamedFunctionCallItemIds.add(item.id);
        }
        streamedFunctionCallCallIds.add(callId);
        return blockIndex;
      };

      const sendFunctionCallArgumentsDelta = (itemId: string | undefined, partialJson: string | undefined) => {
        if (!itemId || !partialJson) return;
        const blockIndex = functionCallBlockIndexByItemId.get(itemId);
        if (blockIndex === undefined) return;
        send("content_block_delta", {
          type: "content_block_delta",
          index: blockIndex,
          delta: { type: "input_json_delta", partial_json: partialJson },
        });
        functionCallHasArgumentDeltaByItemId.set(itemId, true);
      };

      const stopFunctionCallContentBlock = (itemId: string | undefined) => {
        if (!itemId) return;
        const blockIndex = functionCallBlockIndexByItemId.get(itemId);
        if (blockIndex === undefined) return;
        if (pendingContentBlockStop === blockIndex) {
          send("content_block_stop", { type: "content_block_stop", index: blockIndex });
          pendingContentBlockStop = null;
          activeTextBlockKey = null;
        }
        functionCallBlockIndexByItemId.delete(itemId);
        functionCallHasArgumentDeltaByItemId.delete(itemId);
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
          if (!thinkingText && !block.signature) return false;
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
          dumpStreamMessage("upstream openai", payload);
          if (payload === "[DONE]") {
            ensureMessageStart();
            // Send pending content_block_stop if any
            if (pendingContentBlockStop !== null) {
              send("content_block_stop", { type: "content_block_stop", index: pendingContentBlockStop });
              pendingContentBlockStop = null;
              activeTextBlockKey = null;
            }
            // Send message_delta with usage and stop_reason before message_stop
            const messageDelta: { stop_reason: string | null; usage?: ClaudeUsage } = {
              stop_reason: finalizeClaudeStopReason(stopReason),
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
            response?: {
              id?: string;
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
            item?: {
              id?: string;
              type?: string;
              call_id?: string;
              name?: string;
              arguments?: string;
              query?: string;
              max_results?: number;
              search_context_size?: string;
              user_location?: unknown;
            };
            delta?: { type?: string; thinking?: string; signature?: string; data?: string } | string;
            text?: string;
            arguments?: string;
            output_index?: number;
            content_index?: number;
            item_id?: string;
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

          if (json.type === "response.output_text.delta") {
            const textDelta = typeof json.delta === "string" ? json.delta : "";
            if (!textDelta) {
              continue;
            }
            sawOutputTextDelta = true;
            sawIncrementalOutput = true;
            ensureMessageStart(json.id || json.response?.id);
            const blockKey = `${json.item_id ?? json.output_index ?? "0"}:${json.content_index ?? 0}`;
            const blockIndex = ensureTextContentBlock(blockKey);
            send("content_block_delta", {
              type: "content_block_delta",
              index: blockIndex,
              delta: { type: "text_delta", text: textDelta },
            });
            continue;
          }

          if (json.type === "response.output_text.done") {
            sawIncrementalOutput = true;
            ensureMessageStart(json.id || json.response?.id);
            const blockKey = `${json.item_id ?? json.output_index ?? "0"}:${json.content_index ?? 0}`;
            const doneText = json.text ?? "";
            if (doneText && (activeTextBlockKey !== blockKey || pendingContentBlockStop === null)) {
              const blockIndex = ensureTextContentBlock(blockKey);
              send("content_block_delta", {
                type: "content_block_delta",
                index: blockIndex,
                delta: { type: "text_delta", text: doneText },
              });
            }
            if (activeTextBlockKey === blockKey && pendingContentBlockStop !== null) {
              send("content_block_stop", { type: "content_block_stop", index: pendingContentBlockStop });
              pendingContentBlockStop = null;
              activeTextBlockKey = null;
            }
            continue;
          }

          if (json.type === "response.output_item.added") {
            sawIncrementalOutput = true;
            const item = json.item;
            if (item?.type === "function_call") {
              startFunctionCallContentBlock(item);
            }
            continue;
          }

          if (json.type === "response.function_call_arguments.delta") {
            if (typeof json.delta === "string") {
              sendFunctionCallArgumentsDelta(json.item_id, json.delta);
            }
            continue;
          }

          if (json.type === "response.function_call_arguments.done") {
            const argumentsText = json.arguments;
            if (json.item_id && argumentsText) {
              const alreadySent = functionCallHasArgumentDeltaByItemId.get(json.item_id) === true;
              if (!alreadySent) {
                sendFunctionCallArgumentsDelta(json.item_id, argumentsText);
              }
            }
            continue;
          }

          if (json.type === "response.output_item.done") {
            sawIncrementalOutput = true;
            const item = json.item;
            if (item?.type === "function_call" && item.id) {
              stopReason = "tool_use";
              if (item.arguments) {
                const alreadySent = functionCallHasArgumentDeltaByItemId.get(item.id) === true;
                if (!alreadySent) {
                  sendFunctionCallArgumentsDelta(item.id, item.arguments);
                }
              }
              stopFunctionCallContentBlock(item.id);
            }
            continue;
          }

          // Extract stop_reason and usage from upstream events
          if (json.stop_reason !== undefined) {
            const mapped = mapFinishReason(json.stop_reason);
            if (mapped !== null) {
              stopReason = mapped;
            }
          }
          if (json.usage) {
            usage = json.usage;
          }
          if (json.response?.stop_reason !== undefined) {
            const mapped = mapFinishReason(json.response.stop_reason);
            if (mapped !== null) {
              stopReason = mapped;
            }
          }
          if (json.response?.usage) {
            usage = json.response.usage;
          }
          // Extract from output items as well
          const outputItems = json.output ?? json.response?.output;
          if (outputItems) {
            for (const item of outputItems) {
              if (item.stop_reason !== undefined) {
                const mapped = mapFinishReason(item.stop_reason);
                if (mapped !== null) {
                  stopReason = mapped;
                }
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

          if (!sentMessageStart) {
            const candidateId = json.id || json.response?.id;
            if (candidateId) {
              ensureMessageStart(candidateId);
            }
          }

  if (outputItems && !(json.type === "response.completed" && sawIncrementalOutput)) {
    ensureMessageStart(json.id || json.response?.id);
    for (const item of outputItems) {
      if (item.type === "message" && item.content) {
        for (const contentBlock of item.content) {
          if ((contentBlock.type === "text" || contentBlock.type === "output_text") && contentBlock.text) {
            if (sawOutputTextDelta && json.type?.startsWith("response.")) {
              continue;
            }
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
          } else if (contentBlock.type === "web_search_call" || contentBlock.type === "web_search_result") {
            recordWebSearch(contentBlock);
          }
        }
      } else if (item.type === "function_call" && item.call_id && item.name) {
        if (item.id && streamedFunctionCallItemIds.has(item.id)) {
          continue;
        }
        if (streamedFunctionCallCallIds.has(item.call_id)) {
          continue;
        }
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
      } else if (item.type === "web_search_call" || item.type === "web_search_result") {
        recordWebSearch(item);
      }
    }
  }
        }
      }

      const finalLine = buffer.trim();
      if (finalLine.startsWith("data:")) {
        const payload = finalLine.replace(/^data:\s*/, "");
        if (payload === "[DONE]") {
          ensureMessageStart();
          if (pendingContentBlockStop !== null) {
            send("content_block_stop", { type: "content_block_stop", index: pendingContentBlockStop });
            pendingContentBlockStop = null;
            activeTextBlockKey = null;
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
      ensureMessageStart();
      if (pendingContentBlockStop !== null) {
        send("content_block_stop", { type: "content_block_stop", index: pendingContentBlockStop });
        pendingContentBlockStop = null;
        activeTextBlockKey = null;
      }
      // Send message_delta with usage and stop_reason before closing
      const messageDelta: { stop_reason: string | null; usage?: ClaudeUsage } = {
        stop_reason: finalizeClaudeStopReason(stopReason),
      };
      const baseDeltaUsage = mapOpenAIUsageToClaude(usage ?? undefined);
      const deltaUsage = attachToolUseUsage(baseDeltaUsage, getWebSearchRequestCount());
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
