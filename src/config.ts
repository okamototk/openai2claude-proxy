export type ModelMapping = {
  upstream: string;
  downstream: string;
};

export type Config = {
  provider: string;
  openaiKey: string;
  openrouterKey: string;
  openaiBaseUrl: string;
  openrouterBaseUrl: string;
  modelMappings: ModelMapping[];
  defaultUpstreamModel: string;
  defaultDownstreamModel: string;
  port: number;
  bindAddress: string;
  verboseLogging: boolean;
};

const DEFAULT_MODEL_NAME = "gpt-5.2-codex";

export const parseModelArg = (modelArg: string): ModelMapping => {
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

export const resolveModelNames = (argv: string[]): ModelMapping[] => {
  const args = argv.slice(2);
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

export const buildConfig = (env: Record<string, string | undefined>, argv: string[]): Config => {
  const modelMappings = resolveModelNames(argv);
  const defaultModelMapping = modelMappings[0]!;
  return {
    provider: (env.PROVIDER || "openai").toLowerCase(),
    openaiKey: env.OPENAI_API_KEY || "",
    openrouterKey: env.OPENROUTER_API_KEY || "",
    openaiBaseUrl: env.OPENAI_BASE_URL || "https://api.openai.com/v1",
    openrouterBaseUrl: env.OPENROUTER_BASE_URL || "https://openrouter.ai/api/v1",
    modelMappings,
    defaultUpstreamModel: defaultModelMapping.upstream,
    defaultDownstreamModel: defaultModelMapping.downstream,
    port: Number(env.PORT || "3000"),
    bindAddress: env.BIND_ADDRESS || "127.0.0.1",
    verboseLogging: (env.VERBOSE_LOGGING || "").toLowerCase() === "true",
  };
};

export const getModelMapping = (config: Config, downstreamModel: string): ModelMapping | undefined => {
  return config.modelMappings.find(m => m.downstream === downstreamModel);
};

export const buildUpstream = (config: Config): { baseUrl: string; apiKey: string } => {
  if (config.provider === "openrouter") {
    return {
      baseUrl: config.openrouterBaseUrl,
      apiKey: config.openrouterKey,
    };
  }
  return {
    baseUrl: config.openaiBaseUrl,
    apiKey: config.openaiKey,
  };
};

export const getPublicConfig = (config: Config, upstream: { apiKey: string }) => ({
  provider: config.provider,
  modelMappings: config.modelMappings,
  defaultUpstreamModel: config.defaultUpstreamModel,
  defaultDownstreamModel: config.defaultDownstreamModel,
  openaiBaseUrl: config.openaiBaseUrl,
  openrouterBaseUrl: config.openrouterBaseUrl,
  port: config.port,
  bindAddress: config.bindAddress,
  hasUpstreamApiKey: Boolean(upstream.apiKey),
  verboseLogging: config.verboseLogging,
});
