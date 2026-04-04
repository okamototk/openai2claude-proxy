import type { ClaudeMessage, ClaudeMessageContent, ClaudeRequest, ClaudeResponse, ClaudeUsage } from "./openai_to_claude";
import { patchConsoleWithTimestamps } from "./logger";

patchConsoleWithTimestamps();

type GeminiPart =
  | { text: string }
  | { functionCall: { name: string; args?: Record<string, unknown>; thought_signature?: string } }
  | { functionResponse: { name: string; response?: unknown } };

type GeminiContent = {
  role: "user" | "model";
  parts: GeminiPart[];
};

type GeminiTool = {
  functionDeclarations: Array<{
    name: string;
    description?: string;
    parameters?: unknown;
  }>;
};

type GeminiRequest = {
  contents: GeminiContent[];
  systemInstruction?: { role?: "system"; parts: Array<{ text: string }> };
  tools?: GeminiTool[];
  toolConfig?: {
    functionCallingConfig?: {
      mode?: "AUTO" | "ANY" | "NONE";
      allowedFunctionNames?: string[];
    };
  };
  generationConfig?: {
    temperature?: number;
    topP?: number;
    maxOutputTokens?: number;
    stopSequences?: string[];
  };
};

type GeminiResponse = {
  candidates?: Array<{
    content?: { role?: "model" | "user"; parts?: GeminiPart[] };
    finishReason?: string;
  }>;
  usageMetadata?: {
    promptTokenCount?: number;
    candidatesTokenCount?: number;
    totalTokenCount?: number;
  };
};

const logVerbose = (...args: unknown[]) => {
  if ((process.env.VERBOSE_LOGGING || "").toLowerCase() === "true") {
    console.log(...args);
  }
};

const safeJsonParse = (value?: string): unknown | null => {
  if (!value) return null;
  try {
    return JSON.parse(value) as unknown;
  } catch {
    return null;
  }
};

const stripGeminiUnsupportedSchemaKeys = (schema: unknown): unknown => {
  if (!schema || typeof schema !== "object") return schema;
  if (Array.isArray(schema)) return schema.map(stripGeminiUnsupportedSchemaKeys);

  const unsupportedKeys = new Set([
    "$schema",
    "additionalProperties",
    "propertyNames",
    "exclusiveMinimum",
    "const",
  ]);
  const entries = Object.entries(schema as Record<string, unknown>);
  const cleaned: Record<string, unknown> = {};
  for (const [key, value] of entries) {
    if (unsupportedKeys.has(key)) continue;
    cleaned[key] = stripGeminiUnsupportedSchemaKeys(value);
  }
  return cleaned;
};

const stringifyToolResult = (value: unknown): string => {
  if (value === undefined || value === null) return "";
  return typeof value === "string" ? value : JSON.stringify(value);
};

const mapGeminiFinishReason = (reason?: string): string | null => {
  if (!reason) return null;
  const normalized = reason.toLowerCase();
  if (normalized === "stop") return "end_turn";
  if (normalized === "max_tokens") return "max_tokens";
  return normalized;
};

const mapGeminiUsageToClaude = (usage?: GeminiResponse["usageMetadata"]): ClaudeUsage | undefined => {
  if (!usage) return undefined;
  return {
    input_tokens: usage.promptTokenCount,
    output_tokens: usage.candidatesTokenCount,
  };
};

const textFromContent = (content: ClaudeMessage["content"]) => {
  if (typeof content === "string") return content;
  return content
    .filter((block) => block.type === "text")
    .map((block) => (block as { type: "text"; text: string }).text)
    .join("");
};

const mapToolChoiceToGemini = (toolChoice: ClaudeRequest["tool_choice"]) => {
  if (!toolChoice) return undefined;
  if (toolChoice === "required") return { mode: "ANY" } as const;
  if (toolChoice === "auto") return { mode: "AUTO" } as const;
  if (toolChoice === "none") return { mode: "NONE" } as const;
  if (typeof toolChoice === "object") {
    const choice = toolChoice as { name?: string; type?: string };
    if (choice.name) {
      return { mode: "ANY", allowedFunctionNames: [choice.name] } as const;
    }
    if (choice.type === "tool") {
      const name = (toolChoice as { name?: string }).name;
      if (name) {
        return { mode: "ANY", allowedFunctionNames: [name] } as const;
      }
    }
  }
  return undefined;
};

const coerceToolResult = (content: string): unknown => {
  const parsed = safeJsonParse(content);
  if (parsed !== null) return parsed;
  return { result: content };
};

export const mapClaudeToGemini = (req: ClaudeRequest): GeminiRequest => {
  const contents: GeminiContent[] = [];
  const toolNameById = new Map<string, string>();

  if (req.system) {
    const systemText = Array.isArray(req.system)
      ? req.system.map((block) => block.text).join("")
      : req.system;
    if (systemText) {
      logVerbose("[mapClaudeToGemini] System instruction:", systemText);
    }
  }

  for (const msg of req.messages) {
    const role = msg.role === "assistant" ? "model" : "user";
    const parts: GeminiPart[] = [];

    if (typeof msg.content === "string") {
      if (msg.content) parts.push({ text: msg.content });
      contents.push({ role, parts });
      continue;
    }

    for (const block of msg.content) {
      if (block.type === "text") {
        if (block.text) parts.push({ text: block.text });
      } else if (block.type === "tool_use") {
        const args = block.input && typeof block.input === "object" ? (block.input as Record<string, unknown>) : {};
        toolNameById.set(block.id, block.name);
        parts.push({ functionCall: { name: block.name, args, thought_signature: block.id } });
      } else if (block.type === "tool_result") {
        const name = toolNameById.get(block.tool_use_id) || "tool_result";
        parts.push({ functionResponse: { name, response: coerceToolResult(block.content) } });
      }
    }

    const fallbackText = parts.length === 0 ? textFromContent(msg.content) : "";
    if (fallbackText) parts.push({ text: fallbackText });
    if (parts.length > 0) contents.push({ role, parts });
  }

  const toolDeclarations = req.tools?.filter(tool => tool.name).map(tool => ({
    name: tool.name,
    description: tool.description,
    parameters: tool.input_schema ? stripGeminiUnsupportedSchemaKeys(tool.input_schema) : undefined,
  }));
  const tools = toolDeclarations && toolDeclarations.length > 0
    ? [{ functionDeclarations: toolDeclarations }]
    : undefined;

  const toolChoice = mapToolChoiceToGemini(req.tool_choice);

  const generationConfig: GeminiRequest["generationConfig"] = {
    ...(req.temperature !== undefined ? { temperature: req.temperature } : {}),
    ...(req.top_p !== undefined ? { topP: req.top_p } : {}),
    ...(req.max_tokens !== undefined ? { maxOutputTokens: Math.max(16, req.max_tokens) } : {}),
    ...(req.stop_sequences ? { stopSequences: req.stop_sequences } : {}),
  };

  const systemText = req.system
    ? (Array.isArray(req.system) ? req.system.map((block) => block.text).join("") : req.system)
    : "";

  return {
    contents,
    ...(systemText ? { systemInstruction: { role: "system", parts: [{ text: systemText }] } } : {}),
    ...(tools ? { tools } : {}),
    ...(toolChoice ? { toolConfig: { functionCallingConfig: toolChoice } } : {}),
    ...(Object.keys(generationConfig).length > 0 ? { generationConfig } : {}),
  };
};

export const mapGeminiToClaude = (gemini: GeminiResponse, model: string): ClaudeResponse => {
  const candidate = gemini.candidates?.[0];
  const parts = candidate?.content?.parts || [];
  const content: ClaudeMessageContent[] = [];
  const toolCallIdsByName = new Map<string, string[]>();
  let toolIndex = 0;

  for (const part of parts) {
    if ("text" in part && part.text) {
      content.push({ type: "text", text: part.text });
    } else if ("functionCall" in part && part.functionCall) {
      const name = part.functionCall.name;
      const id = `gemini-call-${toolIndex}`;
      toolIndex += 1;
      const queue = toolCallIdsByName.get(name) ?? [];
      queue.push(id);
      toolCallIdsByName.set(name, queue);
      content.push({
        type: "tool_use",
        id,
        name,
        input: part.functionCall.args ?? {},
      });
    } else if ("functionResponse" in part && part.functionResponse) {
      const name = part.functionResponse.name;
      const queue = toolCallIdsByName.get(name) ?? [];
      const id = queue.shift() ?? `gemini-result-${toolIndex++}`;
      if (queue.length > 0) {
        toolCallIdsByName.set(name, queue);
      } else {
        toolCallIdsByName.delete(name);
      }
      content.push({
        type: "tool_result",
        tool_use_id: id,
        content: stringifyToolResult(part.functionResponse.response),
      });
    }
  }

  return {
    id: `gemini-${Date.now()}`,
    type: "message",
    role: "assistant",
    content,
    model,
    stop_reason: mapGeminiFinishReason(candidate?.finishReason),
    stop_sequence: null,
    usage: mapGeminiUsageToClaude(gemini.usageMetadata),
  };
};

export const createClaudeStreamFromGemini = async (geminiStream: ReadableStream<Uint8Array>, model: string) => {
  const encoder = new TextEncoder();
  const decoder = new TextDecoder();
  const reader = geminiStream.getReader();

  const stream = new ReadableStream<Uint8Array>({
    async start(controller) {
      let sentMessageStart = false;
      let contentBlockIndex = 0;
      let pendingContentBlockStop: number | null = null;
      let stopReason: string | null = null;
      let usage: GeminiResponse["usageMetadata"] | null = null;
      let toolIndex = 0;
      const toolCallIdsByName = new Map<string, string[]>();

      const send = (event: string, data: unknown) => {
        controller.enqueue(encoder.encode(`event: ${event}\n`));
        controller.enqueue(encoder.encode(`data: ${JSON.stringify(data)}\n\n`));
      };

      const ensureMessageStart = () => {
        if (sentMessageStart) return;
        send("message_start", {
          type: "message_start",
          message: {
            id: `gemini-${Date.now()}`,
            type: "message",
            role: "assistant",
            model,
            content: [],
            stop_reason: null,
            stop_sequence: null,
          },
        });
        sentMessageStart = true;
      };

      const closePendingBlock = () => {
        if (pendingContentBlockStop !== null) {
          send("content_block_stop", { type: "content_block_stop", index: pendingContentBlockStop });
          pendingContentBlockStop = null;
        }
      };

      const handleParts = (parts: GeminiPart[]) => {
        for (const part of parts) {
          if ("text" in part && part.text) {
            ensureMessageStart();
            closePendingBlock();
            send("content_block_start", {
              type: "content_block_start",
              index: contentBlockIndex,
              content_block: { type: "text", text: "" },
            });
            send("content_block_delta", {
              type: "content_block_delta",
              index: contentBlockIndex,
              delta: { type: "text_delta", text: part.text },
            });
            pendingContentBlockStop = contentBlockIndex;
            contentBlockIndex += 1;
          } else if ("functionCall" in part && part.functionCall) {
            ensureMessageStart();
            closePendingBlock();
            const name = part.functionCall.name;
            const id = `gemini-call-${toolIndex}`;
            toolIndex += 1;
            const queue = toolCallIdsByName.get(name) ?? [];
            queue.push(id);
            toolCallIdsByName.set(name, queue);
            send("content_block_start", {
              type: "content_block_start",
              index: contentBlockIndex,
              content_block: { type: "tool_use", id, name, input: {} },
            });
            const args = part.functionCall.args ? JSON.stringify(part.functionCall.args) : "";
            if (args) {
              send("content_block_delta", {
                type: "content_block_delta",
                index: contentBlockIndex,
                delta: { type: "input_json_delta", partial_json: args },
              });
            }
            pendingContentBlockStop = contentBlockIndex;
            contentBlockIndex += 1;
          } else if ("functionResponse" in part && part.functionResponse) {
            ensureMessageStart();
            closePendingBlock();
            const name = part.functionResponse.name;
            const queue = toolCallIdsByName.get(name) ?? [];
            const id = queue.shift() ?? `gemini-result-${toolIndex++}`;
            if (queue.length > 0) {
              toolCallIdsByName.set(name, queue);
            } else {
              toolCallIdsByName.delete(name);
            }
            const resultText = stringifyToolResult(part.functionResponse.response);
            send("content_block_start", {
              type: "content_block_start",
              index: contentBlockIndex,
              content_block: { type: "tool_result", tool_use_id: id, content: "" },
            });
            if (resultText) {
              send("content_block_delta", {
                type: "content_block_delta",
                index: contentBlockIndex,
                delta: { type: "text_delta", text: resultText },
              });
            }
            pendingContentBlockStop = contentBlockIndex;
            contentBlockIndex += 1;
          }
        }
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
          if (!trimmed) continue;
          const payload = trimmed.startsWith("data:") ? trimmed.replace(/^data:\s*/, "") : trimmed;
          if (payload === "[DONE]") {
            closePendingBlock();
            const messageDelta: { stop_reason: string | null; usage?: ClaudeUsage } = {
              stop_reason: stopReason,
            };
            const mappedUsage = mapGeminiUsageToClaude(usage ?? undefined);
            if (mappedUsage) messageDelta.usage = mappedUsage;
            send("message_delta", { type: "message_delta", delta: messageDelta });
            send("message_stop", { type: "message_stop" });
            controller.close();
            return;
          }

          let parsed: GeminiResponse | GeminiResponse[];
          try {
            parsed = JSON.parse(payload) as GeminiResponse | GeminiResponse[];
          } catch (error) {
            console.log("[messages] Failed to parse Gemini payload:", payload, error);
            continue;
          }

          const chunks = Array.isArray(parsed) ? parsed : [parsed];
          for (const chunk of chunks) {
            if (chunk.usageMetadata) usage = chunk.usageMetadata;
            const candidate = chunk.candidates?.[0];
            if (candidate?.finishReason) {
              stopReason = mapGeminiFinishReason(candidate.finishReason);
            }
            const parts = candidate?.content?.parts || [];
            if (parts.length > 0) handleParts(parts);
          }
        }
      }

      closePendingBlock();
      const messageDelta: { stop_reason: string | null; usage?: ClaudeUsage } = {
        stop_reason: stopReason,
      };
      const mappedUsage = mapGeminiUsageToClaude(usage ?? undefined);
      if (mappedUsage) messageDelta.usage = mappedUsage;
      send("message_delta", { type: "message_delta", delta: messageDelta });
      send("message_stop", { type: "message_stop" });
      controller.close();
    },
  });

  return stream;
};

export type { GeminiRequest, GeminiResponse };
