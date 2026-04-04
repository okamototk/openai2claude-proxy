const patchConsoleWithTimestamps = () => {
  const marker = "__timestamped" as const;
  const current = console.log as typeof console.log & { [marker]?: boolean };
  if (current[marker]) return;

  const withTimestamp = (original: (...args: unknown[]) => void) =>
    (...args: unknown[]) => original(new Date().toISOString(), ...args);

  const patchedLog = withTimestamp(console.log) as typeof console.log & { [marker]?: boolean };
  const patchedError = withTimestamp(console.error) as typeof console.error & { [marker]?: boolean };
  const patchedWarn = withTimestamp(console.warn) as typeof console.warn & { [marker]?: boolean };

  patchedLog[marker] = true;
  patchedError[marker] = true;
  patchedWarn[marker] = true;

  console.log = patchedLog;
  console.error = patchedError;
  console.warn = patchedWarn;
};

export { patchConsoleWithTimestamps };
