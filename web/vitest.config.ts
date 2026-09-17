import { defineConfig } from "vitest/config";

// Deliberately NOT vite.config.ts: that file reads TLS certs, logs at import
// and loads the React/Tailwind/Monaco plugins, none of which a unit test of
// a pure module needs. Tests live beside what they test as `*.test.ts(x)`;
// the automod gate's `frontend` rung runs `vitest run` when this file and
// the binary are both present.
export default defineConfig({
  test: {
    include: ["src/**/*.test.{ts,tsx}"],
    environment: "node",
    passWithNoTests: false,
  },
});
