import { describe, expect, it } from "vitest";
import { mapClaudeToOpenAI, mapFinishReason, mapOpenAIToClaude } from "../src/openai_to_claude";

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
});
