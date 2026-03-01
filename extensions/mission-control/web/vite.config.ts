import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  base: "/mc/",
  build: {
    outDir: "../dist-web",
    emptyOutDir: true,
  },
  server: {
    port: 5173,
    proxy: {
      "/api/mc": {
        target: "http://127.0.0.1:18789",
        changeOrigin: true,
      },
    },
  },
});
