type TextBlock = { type: string; text?: string };

export const isVerboseLoggingEnabled = () =>
  (process.env.VERBOSE_LOGGING || "").toLowerCase() === "true";

export const logVerbose = (...args: unknown[]) => {
  if (isVerboseLoggingEnabled()) {
    console.log(...args);
  }
};

const normalizeStreamDumpValue = (value: unknown): string => {
  if (typeof value === "string") return value;
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
};

export const dumpStreamMessage = (label: string, payload: unknown) => {
  if (!isVerboseLoggingEnabled()) return;
  const text = normalizeStreamDumpValue(payload).replace(/\r?\n/g, "\\n");
  const maxChars = 2000;
  const output = text.length > maxChars ? `${text.slice(0, maxChars)}...<truncated>` : text;
  console.log(`[stream] ${label}:`, output);
};

export const safeJsonParse = (value?: string): unknown | null => {
  if (!value) return null;
  try {
    return JSON.parse(value) as unknown;
  } catch {
    return null;
  }
};

export const stringifyToolResult = (value: unknown): string => {
  if (value === undefined || value === null) return "";
  return typeof value === "string" ? value : JSON.stringify(value);
};

export const textFromClaudeContent = (content: string | TextBlock[]) => {
  if (typeof content === "string") return content;
  return content
    .filter((block) => block.type === "text")
    .map((block) => block.text ?? "")
    .join("");
};
