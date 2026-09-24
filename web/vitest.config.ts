import { fileURLToPath } from "node:url";
import { defineConfig } from "vitest/config";

// Deliberately NOT vite.config.ts: that file reads TLS certs, logs at import
// and loads the React/Tailwind/Monaco plugins, none of which a unit test of
// a pure module needs. Tests live beside what they test as `*.test.ts(x)`;
// the automod gate's `frontend` rung runs `vitest run` when this file and
// the binary are both present.
export default defineConfig({
  // tsconfig's `@/*` path, so a component that imports `@/components/ui/…`
  // resolves under test the same way it does under tsc and vite.
  resolve: {
    alias: { "@": fileURLToPath(new URL("./src", import.meta.url)) },
  },
  test: {
    include: ["src/**/*.test.{ts,tsx}"],
    environment: "node",
    passWithNoTests: false,
  },
});
