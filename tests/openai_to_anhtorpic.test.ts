import { describe, expect, it } from "vitest";
import { mapAnthropicToOpenAI, mapFinishReason, mapOpenAIToAnthropic } from "../src/openai_to_anhtorpic";

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

describe("mapOpenAIToAnthropic", () => {
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

    const anthropic = mapOpenAIToAnthropic(openaiResponse, "gpt-5.2-codex");

    expect(anthropic.id).toBe("resp_1");
    expect(anthropic.role).toBe("assistant");
    expect(anthropic.content).toEqual([{ type: "text", text: "hello" }]);
    expect(anthropic.stop_reason).toBe("end_turn");
    expect(anthropic.usage?.input_tokens).toBe(5);
    expect(anthropic.usage?.output_tokens).toBe(7);
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

    const anthropic = mapOpenAIToAnthropic(openaiResponse, "gpt-5.2-codex");

    expect(anthropic.content).toEqual([
      { type: "text", text: "answer" },
      { type: "thinking", thinking: "trace" },
      { type: "thinking", thinking: "summary", signature: "sig_1" },
      { type: "redacted_thinking", data: "REDACTED", signature: "sig_2" },
    ]);
  });
});


describe("mapAnthropicToOpenAI", () => {
  it("converts user/assistant messages into OpenAI input items", () => {
    const anthropicRequest = {
      model: "gpt-5.2-codex",
      messages: [
        { role: "user", content: "Hello" },
        { role: "assistant", content: "Hi" },
      ],
      max_tokens: 10,
    } as const;

    const openai = mapAnthropicToOpenAI(anthropicRequest, "gpt-5.2-codex");

    expect(openai.model).toBe("gpt-5.2-codex");
    expect(openai.input).toEqual([
      { role: "user", content: "Hello" },
      { role: "assistant", content: "Hi" },
    ]);
  });

  it("drops thinking blocks from OpenAI input", () => {
    const anthropicRequest = {
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

    const openai = mapAnthropicToOpenAI(anthropicRequest, "gpt-5.2-codex");

    expect(openai.input).toEqual([
      {
        role: "assistant",
        content: "answer",
      },
    ]);
  });
});
