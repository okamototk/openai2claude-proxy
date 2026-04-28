import { describe, expect, it } from "vitest";
import { createClaudeStream, mapClaudeToOpenAI, mapFinishReason, mapOpenAIToClaude } from "../src/openai_to_claude";

const createOpenAIDataStream = (events: string[]) => {
  const encoder = new TextEncoder();
  return new ReadableStream<Uint8Array>({
    start(controller) {
      for (const event of events) {
        controller.enqueue(encoder.encode(`data: ${event}\n\n`));
      }
      controller.close();
    },
  });
};

const readStreamToString = async (stream: ReadableStream<Uint8Array>) => {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let output = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    output += decoder.decode(value, { stream: true });
  }
  output += decoder.decode();

  return output;
};

describe("mapFinishReason", () => {
  it("maps known reasons", () => {
    expect(mapFinishReason("stop")).toBe("end_turn");
    expect(mapFinishReason("length")).toBe("max_tokens");
    expect(mapFinishReason("tool_calls")).toBe("tool_use");
  });

  it("passes through unknown values", () => {
    expect(mapFinishReason("custom_reason")).toBe("custom_reason");
    expect(mapFinishReason(null)).toBeNull();
  });
});

describe("mapOpenAIToClaude", () => {
  it("converts a basic response message", () => {
    const openaiResponse = {
      id: "resp_1",
      object: "response",
      created_at: 0,
      model: "gpt-5.2-codex",
      output: [
        {
          type: "message",
          id: "msg_1",
          role: "assistant",
          content: [{ type: "output_text", text: "hello" }],
          stop_reason: "stop",
        },
      ],
      usage: { input_tokens: 5, output_tokens: 7 },
    } as const;

    const claude = mapOpenAIToClaude(openaiResponse, "gpt-5.2-codex");

    expect(claude.id).toBe("resp_1");
    expect(claude.role).toBe("assistant");
    expect(claude.content).toEqual([{ type: "text", text: "hello" }]);
    expect(claude.stop_reason).toBe("end_turn");
    expect(claude.usage?.input_tokens).toBe(5);
    expect(claude.usage?.output_tokens).toBe(7);
  });

  it("maps reasoning blocks to thinking content", () => {
    const openaiResponse = {
      id: "resp_reasoning",
      object: "response",
      created_at: 0,
      model: "gpt-5.2-codex",
      output: [
        {
          type: "message",
          id: "msg_reasoning",
          role: "assistant",
          content: [
            { type: "output_text", text: "answer" },
            { type: "reasoning", text: "trace" },
            { type: "output_reasoning", reasoning: "summary", signature: "sig_1" },
            { type: "redacted_reasoning", data: "REDACTED", signature: "sig_2" },
          ],
          stop_reason: "stop",
        },
      ],
    } as const;

    const claude = mapOpenAIToClaude(openaiResponse, "gpt-5.2-codex");

    expect(claude.content).toEqual([
      { type: "text", text: "answer" },
      { type: "thinking", thinking: "trace" },
      { type: "thinking", thinking: "summary", signature: "sig_1" },
      { type: "redacted_thinking", data: "REDACTED", signature: "sig_2" },
    ]);
  });

  it("ignores empty reasoning summary arrays", () => {
    const openaiResponse = {
      id: "resp_reasoning_empty",
      object: "response",
      created_at: 0,
      model: "gpt-5.2-codex",
      output: [
        {
          type: "message",
          id: "msg_reasoning_empty",
          role: "assistant",
          content: [
            { type: "output_text", text: "answer" },
            { type: "reasoning", summary: [] },
          ],
          stop_reason: "stop",
        },
      ],
    } as const;

    const claude = mapOpenAIToClaude(openaiResponse, "gpt-5.2-codex");

    expect(claude.content).toEqual([{ type: "text", text: "answer" }]);
  });

  it("strips empty pages from tool arguments", () => {
    const openaiResponse = {
      id: "resp_tool",
      object: "response",
      created_at: 0,
      model: "gpt-5.2-codex",
      output: [
        {
          type: "message",
          id: "msg_tool",
          role: "assistant",
          content: [
            {
              type: "function_call",
              call_id: "call_read",
              name: "Read",
              arguments: JSON.stringify({ filePath: "/tmp/file.txt", pages: "" }),
            },
          ],
          stop_reason: "tool_calls",
        },
      ],
    } as const;

    const claude = mapOpenAIToClaude(openaiResponse, "gpt-5.2-codex");

    expect(claude.content).toEqual([
      { type: "tool_use", id: "call_read", name: "Read", input: { filePath: "/tmp/file.txt" } },
    ]);
  });

  it("uses tool_use stop reason when response contains tool calls but no stop_reason", () => {
    const openaiResponse = {
      id: "resp_tool_stop",
      object: "response",
      created_at: 0,
      model: "gpt-5.3-codex",
      output: [
        {
          type: "message",
          id: "msg_tool_stop",
          role: "assistant",
          content: [
            {
              type: "function_call",
              call_id: "call_123",
              name: "WebSearch",
              arguments: "{\"query\":\"Mythos\"}",
            },
          ],
          stop_reason: null,
        },
      ],
    } as const;

    const claude = mapOpenAIToClaude(openaiResponse, "gpt-5.3-codex");

    expect(claude.stop_reason).toBe("tool_use");
  });

  it("does not expose web_search_call as tool_use content", () => {
    const openaiResponse = {
      id: "resp_web_search",
      object: "response",
      created_at: 0,
      model: "gpt-5.3-codex",
      output: [
        {
          type: "web_search_call",
          id: "ws_1",
          query: "Mythos",
        },
        {
          type: "message",
          id: "msg_web_search",
          role: "assistant",
          content: [
            { type: "output_text", text: "I found results." },
          ],
          stop_reason: "stop",
        },
      ],
      usage: { input_tokens: 10, output_tokens: 12 },
    } as const;

    const claude = mapOpenAIToClaude(openaiResponse, "gpt-5.3-codex");

    expect(claude.content).toEqual([{ type: "text", text: "I found results." }]);
    expect(claude.stop_reason).toBe("end_turn");
    expect(claude.usage?.server_tool_use?.web_search_requests).toBe(1);
  });

  it("defaults to end_turn when text output has null stop_reason", () => {
    const openaiResponse = {
      id: "resp_end_turn_default",
      object: "response",
      created_at: 0,
      model: "gpt-5.3-codex",
      output: [
        {
          type: "message",
          id: "msg_end_turn_default",
          role: "assistant",
          content: [{ type: "output_text", text: "Done." }],
          stop_reason: null,
        },
      ],
    } as const;

    const claude = mapOpenAIToClaude(openaiResponse, "gpt-5.3-codex");

    expect(claude.stop_reason).toBe("end_turn");
  });
});


describe("mapClaudeToOpenAI", () => {
  it("converts user/assistant messages into OpenAI input items", () => {
    const claudeRequest = {
      model: "gpt-5.2-codex",
      messages: [
        { role: "user", content: "Hello" },
        { role: "assistant", content: "Hi" },
      ],
      max_tokens: 10,
    } as const;

    const openai = mapClaudeToOpenAI(claudeRequest, "gpt-5.2-codex");

    expect(openai.model).toBe("gpt-5.2-codex");
    expect(openai.input).toEqual([
      { role: "user", content: "Hello" },
      { role: "assistant", content: "Hi" },
    ]);
  });

  it("drops thinking blocks from OpenAI input", () => {
    const claudeRequest = {
      model: "gpt-5.2-codex",
      messages: [
        {
          role: "assistant",
          content: [
            { type: "text", text: "answer" },
            { type: "thinking", thinking: "trace" },
            { type: "redacted_thinking", data: "REDACTED", signature: "sig_2" },
          ],
        },
      ],
      max_tokens: 10,
    } as const;

    const openai = mapClaudeToOpenAI(claudeRequest, "gpt-5.2-codex");

    expect(openai.input).toEqual([
      {
        role: "assistant",
        content: "answer",
      },
    ]);
  });

  it("maps tool use and tool result as structured responses items", () => {
    const claudeRequest = {
      model: "gpt-5.2-codex",
      messages: [
        {
          role: "assistant",
          content: [
            { type: "text", text: "Working on it" },
            { type: "tool_use", id: "call_1", name: "TaskCreate", input: { subject: "Review code" } },
          ],
        },
        {
          role: "user",
          content: [
            { type: "tool_result", tool_use_id: "call_1", content: "{\"id\":\"task_1\"}" },
            { type: "text", text: "proceed" },
          ],
        },
      ],
      max_tokens: 10,
    } as const;

    const openai = mapClaudeToOpenAI(claudeRequest, "gpt-5.2-codex");

    expect(openai.input).toEqual([
      { role: "assistant", content: "Working on it" },
      { type: "function_call", call_id: "call_1", name: "TaskCreate", arguments: "{\"subject\":\"Review code\"}" },
      { type: "function_call_output", call_id: "call_1", output: "{\"id\":\"task_1\"}" },
      { role: "user", content: "proceed" },
    ]);
  });

  it("stringifies structured tool result content for Responses API", () => {
    const claudeRequest = {
      model: "gpt-5.2-codex",
      messages: [
        {
          role: "assistant",
          content: [
            { type: "tool_use", id: "call_1", name: "ToolSearch", input: { query: "select:WebFetch" } },
          ],
        },
        {
          role: "user",
          content: [
            { type: "tool_result", tool_use_id: "call_1", content: [{ type: "tool_reference", tool_name: "WebFetch" }] },
          ],
        },
      ],
      max_tokens: 10,
    } as const;

    const openai = mapClaudeToOpenAI(claudeRequest, "gpt-5.2-codex");

    expect(openai.input).toEqual([
      { type: "function_call", call_id: "call_1", name: "ToolSearch", arguments: "{\"query\":\"select:WebFetch\"}" },
      { type: "function_call_output", call_id: "call_1", output: "[{\"type\":\"tool_reference\",\"tool_name\":\"WebFetch\"}]" },
    ]);
  });

  it("drops tool_result blocks when tool_use_id does not exist", () => {
    const claudeRequest = {
      model: "gpt-5.2-codex",
      messages: [
        {
          role: "user",
          content: [
            { type: "tool_result", tool_use_id: "missing_call", content: "{\"ok\":true}" },
            { type: "text", text: "continue" },
          ],
        },
      ],
      max_tokens: 10,
    } as const;

    const openai = mapClaudeToOpenAI(claudeRequest, "gpt-5.2-codex");

    expect(openai.input).toEqual([
      { role: "user", content: "continue" },
    ]);
  });

  it("maps Claude web_search tool choice without unsupported name field", () => {
    const claudeRequest = {
      model: "gpt-5.3-codex",
      messages: [{ role: "user", content: "search mythos" }],
      tools: [{ name: "web_search", type: "web_search" }],
      tool_choice: { type: "tool", name: "web_search" },
      max_tokens: 64,
    } as const;

    const openai = mapClaudeToOpenAI(claudeRequest, "gpt-5.3-codex");

    expect(openai.tool_choice).toEqual({ type: "web_search" });
  });

  it("maps Claude named tool choice to OpenAI function tool_choice", () => {
    const claudeRequest = {
      model: "gpt-5.3-codex",
      messages: [{ role: "user", content: "do math" }],
      tools: [{ name: "calculator", input_schema: { type: "object", properties: {} } }],
      tool_choice: { type: "tool", name: "calculator" },
      max_tokens: 64,
    } as const;

    const openai = mapClaudeToOpenAI(claudeRequest, "gpt-5.3-codex");

    expect(openai.tool_choice).toEqual({ type: "function", name: "calculator" });
  });

  it("can inject OpenAI server web_search when Claude Code omits the tool", () => {
    const claudeRequest = {
      model: "gpt-5.3-codex",
      messages: [{ role: "user", content: "web search Mythos" }],
      max_tokens: 64,
    } as const;

    const openai = mapClaudeToOpenAI(claudeRequest, "gpt-5.3-codex", { injectWebSearchTool: true });

    expect(openai.tools).toEqual([{ type: "web_search" }]);
  });

  it("does not inject web_search unless enabled", () => {
    const claudeRequest = {
      model: "gpt-5.3-codex",
      messages: [{ role: "user", content: "hello" }],
      max_tokens: 64,
    } as const;

    const openai = mapClaudeToOpenAI(claudeRequest, "gpt-5.3-codex");

    expect(openai.tools).toBeUndefined();
  });

  it("does not duplicate injected web_search tools", () => {
    const claudeRequest = {
      model: "gpt-5.3-codex",
      messages: [{ role: "user", content: "search mythos" }],
      tools: [{ name: "web_search", type: "web_search" }],
      max_tokens: 64,
    } as const;

    const openai = mapClaudeToOpenAI(claudeRequest, "gpt-5.3-codex", { injectWebSearchTool: true });

    expect(openai.tools).toEqual([{ type: "web_search" }]);
  });
});

describe("createClaudeStream", () => {
  it("maps response.output_text.delta events into Claude SSE text deltas", async () => {
    const openAIStream = createOpenAIDataStream([
      "{\"type\":\"response.created\",\"response\":{\"id\":\"resp_1\"}}",
      "{\"type\":\"response.output_text.delta\",\"item_id\":\"msg_1\",\"output_index\":0,\"content_index\":0,\"delta\":\"hello\"}",
      "{\"type\":\"response.output_text.done\",\"item_id\":\"msg_1\",\"output_index\":0,\"content_index\":0,\"text\":\"hello\"}",
      "{\"type\":\"response.completed\",\"response\":{\"id\":\"resp_1\",\"usage\":{\"input_tokens\":1,\"output_tokens\":1}}}",
      "[DONE]",
    ]);

    const claudeStream = await createClaudeStream(openAIStream, "gpt-5.3-codex");
    const sse = await readStreamToString(claudeStream);

    expect(sse).toContain("event: message_start");
    expect(sse).toContain("\"delta\":{\"type\":\"text_delta\",\"text\":\"hello\"}");
    expect(sse).toContain("event: content_block_stop");
    expect(sse).toContain("event: message_stop");
  });

  it("does not duplicate text when response.completed includes output snapshot", async () => {
    const openAIStream = createOpenAIDataStream([
      "{\"type\":\"response.created\",\"response\":{\"id\":\"resp_2\"}}",
      "{\"type\":\"response.output_text.delta\",\"item_id\":\"msg_2\",\"output_index\":0,\"content_index\":0,\"delta\":\"Hi!\"}",
      "{\"type\":\"response.output_text.done\",\"item_id\":\"msg_2\",\"output_index\":0,\"content_index\":0,\"text\":\"Hi!\"}",
      "{\"type\":\"response.completed\",\"response\":{\"id\":\"resp_2\",\"output\":[{\"type\":\"message\",\"content\":[{\"type\":\"output_text\",\"text\":\"Hi!\"}]}],\"usage\":{\"input_tokens\":1,\"output_tokens\":1}}}",
      "[DONE]",
    ]);

    const claudeStream = await createClaudeStream(openAIStream, "gpt-5.3-codex");
    const sse = await readStreamToString(claudeStream);

    const duplicateCount = sse.split("\"text\":\"Hi!\"").length - 1;
    expect(duplicateCount).toBe(1);
  });

  it("maps streamed function call argument events into tool_use blocks", async () => {
    const openAIStream = createOpenAIDataStream([
      "{\"type\":\"response.created\",\"response\":{\"id\":\"resp_fc_1\"}}",
      "{\"type\":\"response.output_item.added\",\"item\":{\"id\":\"fc_1\",\"type\":\"function_call\",\"status\":\"in_progress\",\"arguments\":\"\",\"call_id\":\"call_1\",\"name\":\"Bash\"},\"output_index\":0}",
      "{\"type\":\"response.function_call_arguments.delta\",\"delta\":\"{\\\"command\\\":\\\"git status\\\"}\",\"item_id\":\"fc_1\",\"output_index\":0}",
      "{\"type\":\"response.function_call_arguments.done\",\"arguments\":\"{\\\"command\\\":\\\"git status\\\"}\",\"item_id\":\"fc_1\",\"output_index\":0}",
      "{\"type\":\"response.output_item.done\",\"item\":{\"id\":\"fc_1\",\"type\":\"function_call\",\"status\":\"completed\",\"arguments\":\"{\\\"command\\\":\\\"git status\\\"}\",\"call_id\":\"call_1\",\"name\":\"Bash\"},\"output_index\":0}",
      "{\"type\":\"response.completed\",\"response\":{\"id\":\"resp_fc_1\",\"stop_reason\":\"tool_calls\",\"usage\":{\"input_tokens\":1,\"output_tokens\":1}}}",
      "[DONE]",
    ]);

    const claudeStream = await createClaudeStream(openAIStream, "gpt-5.3-codex");
    const sse = await readStreamToString(claudeStream);

    expect(sse).toContain("\"content_block\":{\"type\":\"tool_use\",\"id\":\"call_1\",\"name\":\"Bash\",\"input\":{}}}");
    expect(sse).toContain("\"partial_json\":\"{\\\"command\\\":\\\"git status\\\"}\"");
    expect(sse).toContain("\"stop_reason\":\"tool_use\"");
  });

  it("does not duplicate tool_use blocks when response.completed also includes output", async () => {
    const openAIStream = createOpenAIDataStream([
      "{\"type\":\"response.created\",\"response\":{\"id\":\"resp_fc_2\"}}",
      "{\"type\":\"response.output_item.added\",\"item\":{\"id\":\"fc_2\",\"type\":\"function_call\",\"status\":\"in_progress\",\"arguments\":\"\",\"call_id\":\"call_2\",\"name\":\"Bash\"},\"output_index\":0}",
      "{\"type\":\"response.function_call_arguments.done\",\"arguments\":\"{\\\"command\\\":\\\"git diff\\\"}\",\"item_id\":\"fc_2\",\"output_index\":0}",
      "{\"type\":\"response.output_item.done\",\"item\":{\"id\":\"fc_2\",\"type\":\"function_call\",\"status\":\"completed\",\"arguments\":\"{\\\"command\\\":\\\"git diff\\\"}\",\"call_id\":\"call_2\",\"name\":\"Bash\"},\"output_index\":0}",
      "{\"type\":\"response.completed\",\"response\":{\"id\":\"resp_fc_2\",\"output\":[{\"id\":\"fc_2\",\"type\":\"function_call\",\"call_id\":\"call_2\",\"name\":\"Bash\",\"arguments\":\"{\\\"command\\\":\\\"git diff\\\"}\"}],\"stop_reason\":\"tool_calls\",\"usage\":{\"input_tokens\":1,\"output_tokens\":1}}}",
      "[DONE]",
    ]);

    const claudeStream = await createClaudeStream(openAIStream, "gpt-5.3-codex");
    const sse = await readStreamToString(claudeStream);

    const toolUseCount = sse.split("\"type\":\"tool_use\"").length - 1;
    expect(toolUseCount).toBe(1);
  });

  it("keeps tool_use stop_reason even when response.completed stop_reason is null", async () => {
    const openAIStream = createOpenAIDataStream([
      "{\"type\":\"response.created\",\"response\":{\"id\":\"resp_fc_3\"}}",
      "{\"type\":\"response.output_item.added\",\"item\":{\"id\":\"fc_3\",\"type\":\"function_call\",\"status\":\"in_progress\",\"arguments\":\"\",\"call_id\":\"call_3\",\"name\":\"WebSearch\"},\"output_index\":0}",
      "{\"type\":\"response.function_call_arguments.done\",\"arguments\":\"{\\\"query\\\":\\\"Mythos\\\"}\",\"item_id\":\"fc_3\",\"output_index\":0}",
      "{\"type\":\"response.output_item.done\",\"item\":{\"id\":\"fc_3\",\"type\":\"function_call\",\"status\":\"completed\",\"arguments\":\"{\\\"query\\\":\\\"Mythos\\\"}\",\"call_id\":\"call_3\",\"name\":\"WebSearch\"},\"output_index\":0}",
      "{\"type\":\"response.completed\",\"response\":{\"id\":\"resp_fc_3\",\"stop_reason\":null,\"usage\":{\"input_tokens\":1,\"output_tokens\":1}}}",
      "[DONE]",
    ]);

    const claudeStream = await createClaudeStream(openAIStream, "gpt-5.3-codex");
    const sse = await readStreamToString(claudeStream);

    expect(sse).toContain("\"stop_reason\":\"tool_use\"");
  });

  it("does not emit tool_use stop_reason for server web_search calls", async () => {
    const openAIStream = createOpenAIDataStream([
      "{\"type\":\"response.created\",\"response\":{\"id\":\"resp_ws_1\"}}",
      "{\"type\":\"response.output_item.added\",\"item\":{\"id\":\"ws_1\",\"type\":\"web_search_call\",\"status\":\"in_progress\"},\"output_index\":0}",
      "{\"type\":\"response.output_item.done\",\"item\":{\"id\":\"ws_1\",\"type\":\"web_search_call\",\"status\":\"completed\",\"action\":{\"type\":\"search\",\"query\":\"Mythos\"}},\"output_index\":0}",
      "{\"type\":\"response.output_text.delta\",\"item_id\":\"msg_ws_1\",\"output_index\":1,\"content_index\":0,\"delta\":\"I found results.\"}",
      "{\"type\":\"response.output_text.done\",\"item_id\":\"msg_ws_1\",\"output_index\":1,\"content_index\":0,\"text\":\"I found results.\"}",
      "{\"type\":\"response.completed\",\"response\":{\"id\":\"resp_ws_1\",\"stop_reason\":\"stop\",\"output\":[{\"id\":\"ws_1\",\"type\":\"web_search_call\"}],\"usage\":{\"input_tokens\":1,\"output_tokens\":1}}}",
      "[DONE]",
    ]);

    const claudeStream = await createClaudeStream(openAIStream, "gpt-5.3-codex");
    const sse = await readStreamToString(claudeStream);

    expect(sse).not.toContain("\"stop_reason\":\"tool_use\"");
    expect(sse).not.toContain("\"type\":\"tool_use\"");
    expect(sse).toContain("\"stop_reason\":\"end_turn\"");
    expect(sse).toContain("\"server_tool_use\":{\"web_search_requests\":1}");
  });

  it("defaults streamed stop_reason to end_turn when upstream omits it", async () => {
    const openAIStream = createOpenAIDataStream([
      "{\"type\":\"response.created\",\"response\":{\"id\":\"resp_end_turn_stream\"}}",
      "{\"type\":\"response.output_text.delta\",\"item_id\":\"msg_end_turn_stream\",\"output_index\":0,\"content_index\":0,\"delta\":\"Done.\"}",
      "{\"type\":\"response.output_text.done\",\"item_id\":\"msg_end_turn_stream\",\"output_index\":0,\"content_index\":0,\"text\":\"Done.\"}",
      "{\"type\":\"response.completed\",\"response\":{\"id\":\"resp_end_turn_stream\",\"stop_reason\":null,\"usage\":{\"input_tokens\":1,\"output_tokens\":1}}}",
      "[DONE]",
    ]);

    const claudeStream = await createClaudeStream(openAIStream, "gpt-5.3-codex");
    const sse = await readStreamToString(claudeStream);

    expect(sse).toContain("\"stop_reason\":\"end_turn\"");
  });

  it("handles web-search roundtrip as tool_use then end_turn", async () => {
    const toolDispatchStream = createOpenAIDataStream([
      "{\"type\":\"response.created\",\"response\":{\"id\":\"resp_tool_dispatch\"}}",
      "{\"type\":\"response.output_text.delta\",\"item_id\":\"msg_tool_dispatch\",\"output_index\":0,\"content_index\":0,\"delta\":\"I will search now.\"}",
      "{\"type\":\"response.output_text.done\",\"item_id\":\"msg_tool_dispatch\",\"output_index\":0,\"content_index\":0,\"text\":\"I will search now.\"}",
      "{\"type\":\"response.output_item.added\",\"item\":{\"id\":\"fc_tool_dispatch\",\"type\":\"function_call\",\"status\":\"in_progress\",\"arguments\":\"\",\"call_id\":\"call_websearch_1\",\"name\":\"WebSearch\"},\"output_index\":1}",
      "{\"type\":\"response.function_call_arguments.done\",\"arguments\":\"{\\\"query\\\":\\\"Mythos\\\"}\",\"item_id\":\"fc_tool_dispatch\",\"output_index\":1}",
      "{\"type\":\"response.output_item.done\",\"item\":{\"id\":\"fc_tool_dispatch\",\"type\":\"function_call\",\"status\":\"completed\",\"arguments\":\"{\\\"query\\\":\\\"Mythos\\\"}\",\"call_id\":\"call_websearch_1\",\"name\":\"WebSearch\"},\"output_index\":1}",
      "{\"type\":\"response.completed\",\"response\":{\"id\":\"resp_tool_dispatch\",\"stop_reason\":\"tool_calls\",\"usage\":{\"input_tokens\":5,\"output_tokens\":3}}}",
      "[DONE]",
    ]);

    const toolDispatchClaudeStream = await createClaudeStream(toolDispatchStream, "gpt-5.3-codex");
    const toolDispatchSse = await readStreamToString(toolDispatchClaudeStream);

    expect(toolDispatchSse).toContain("\"stop_reason\":\"tool_use\"");
    expect(toolDispatchSse).toContain("\"type\":\"tool_use\"");

    const webSearchResultStream = createOpenAIDataStream([
      "{\"type\":\"response.created\",\"response\":{\"id\":\"resp_websearch_result\"}}",
      "{\"type\":\"response.output_item.added\",\"item\":{\"id\":\"ws_1\",\"type\":\"web_search_call\",\"status\":\"in_progress\"},\"output_index\":0}",
      "{\"type\":\"response.output_item.done\",\"item\":{\"id\":\"ws_1\",\"type\":\"web_search_call\",\"status\":\"completed\",\"action\":{\"type\":\"search\",\"query\":\"Mythos\"}},\"output_index\":0}",
      "{\"type\":\"response.output_text.delta\",\"item_id\":\"msg_websearch_result\",\"output_index\":1,\"content_index\":0,\"delta\":\"Top result is Merriam-Webster.\"}",
      "{\"type\":\"response.output_text.done\",\"item_id\":\"msg_websearch_result\",\"output_index\":1,\"content_index\":0,\"text\":\"Top result is Merriam-Webster.\"}",
      "{\"type\":\"response.completed\",\"response\":{\"id\":\"resp_websearch_result\",\"stop_reason\":\"stop\",\"usage\":{\"input_tokens\":7,\"output_tokens\":9},\"output\":[{\"id\":\"ws_1\",\"type\":\"web_search_call\"}]}}",
      "[DONE]",
    ]);

    const webSearchResultClaudeStream = await createClaudeStream(webSearchResultStream, "gpt-5.3-codex");
    const webSearchResultSse = await readStreamToString(webSearchResultClaudeStream);

    expect(webSearchResultSse).toContain("Top result is Merriam-Webster.");
    expect(webSearchResultSse).toContain("\"stop_reason\":\"end_turn\"");
    expect(webSearchResultSse).toContain("\"server_tool_use\":{\"web_search_requests\":1}");
  });
});
